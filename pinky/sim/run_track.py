"""트랙 도로(road.json에 경로 종류가 들어 있는 형식)에서 경로마다 폐루프 주행을 한다. sim_closed.py와 같은 순서이고 처리 코드(lane_direct.py)는 그대로 쓴다.
다른 점: 경로 종류를 road.json의 routes에서 읽는다 / 이탈 기준은 그 경로의 차로 중심에서 가까운 쪽 선까지의 거리 / 경로 끝 END m 전에 끝낸다.
사용: python run_track.py ROAD_DIR OUT_DIR WEIGHTS(또는 gt = 정답 라벨로 주행) TURN(line 또는 area) [경로 번호 ...]"""
import csv, json, os, subprocess, sys, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 저장소 최상위
sys.path.insert(0, ROOT + "/pinky/sim"); sys.path.insert(0, ROOT)
import lane_direct                                          # 고친 처리 코드를 먼저 불러 둔다(sim_closed가 같은 것을 쓰게)
import sim_closed as S
from ultralytics import YOLO
road, out, wts, tmode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
END, FPS, SPEED, WMAX = 0.10, 15.0, 0.2, 1.5
class A: imgsz, conf = 640, 0.25
cam_yaml = ROOT + "/pinky/config/pinky_camera.yaml"; os.makedirs(out, exist_ok=True); calib = out + "/calib.npz"
subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
model, sc = (None if wts == "gt" else YOLO(wts)), S.Scene(road, cam_yaml)
rt = S.Routes(sc, S.LATERAL_MM, S.YAW_DEG)
routes = sc.info["routes"]; todo = [int(v) for v in sys.argv[5:]] or range(len(routes))
top0 = cv2.imread(road + "/floor.png"); ppm, Hpx = sc.info["ppm"], top0.shape[0]
res_rows = []
for pi in todo:
    r, t0 = routes[pi], time.time()
    rng = np.random.default_rng(0); np.random.seed(0)
    path, s_path = S.resample(np.asarray(sc.info["paths"][pi]) / 1000.0, 0.005); tan = np.gradient(path, axis=0)
    s_en, s_ex = r["en"] * 0.004 * (1000 / 1000) / 1.0 * (1 / 1.0) , r["ex"] * 0.004                 # 교차로 구간의 시작·끝(경로 위 거리 m, 경로 점 간격 4픽셀)
    s_en, s_ex = r["en"] * 4 / ppm / 1000, r["ex"] * 4 / ppm / 1000
    half = min(r["bounds_in"] + r["bounds_out"]) / 1000.0                                           # 가까운 쪽 선까지의 거리(m)
    x, y, yaw = path[0, 0], path[0, 1], float(np.arctan2(tan[0, 1], tan[0, 0]))
    tint = np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)])
    bg, live, dt = S.background(sc.H, sc.W, rng), lane_direct.Direct(calib, FPS, tmode), 1.0 / FPS
    if os.environ.get("KNOWN_W"): live.widths = [float(r["bounds_in"][0] + r["bounds_in"][1]) / 1000.0] * 5       # 시험용: 차로 폭을 이미 잰 상태에서 출발
    cmd, lost, still, move, started, rows, end = (0.0, 0.0), 0.0, 0, None, None, [], "시간 초과"
    for i in range(int((s_path[-1] / SPEED + 30) * FPS)):
        img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw, pi))); img = S.look(img, tint, 3.0, bg)
        dets = S.detect(model, img, inst, mine, sc, A())
        res = live.step(dets, (sc.W, sc.H), move, r["kind"], r.get("slot", 1))
        new = None if res is None else S.pursuit(res["center"], res["look"], SPEED, WMAX, S.CAM_FWD)
        if new is not None: cmd, lost, started = new, 0.0, (i if started is None else started)
        elif res is not None and lost < S.LOST_LANES * res["lane_w"]: lost += cmd[0] * dt
        elif started is None and i < S.MEASURE_SEC * FPS: cmd = (SPEED / 2, 0.0)
        else: cmd = (0.0, 0.0)
        k = int(np.argmin(np.linalg.norm(path - [x, y], axis=1))); t = tan[k] / (np.linalg.norm(tan[k]) + 1e-9)
        off = float(t[0] * (y - path[k, 1]) - t[1] * (x - path[k, 0]))
        zone = int(s_path[k] >= s_en) + int(s_path[k] > s_ex)                                        # 0 교차로 앞, 1 교차로 구간, 2 나온 뒤
        rows.append([i, x, y, yaw, cmd[0], cmd[1], off, s_path[k], zone, int(new is not None), sorted(set(c for c, _ in dets))])
        still = still + 1 if cmd[0] == 0 else 0
        if abs(off) > half or s_path[k] > s_path[-1] - END or still > S.STOP_SEC * FPS:
            end = "이탈" if abs(off) > half else "완주" if still <= S.STOP_SEC * FPS else "멈춤"; break
        o0, f0 = np.array([x, y]) + S.CAM_FWD * np.array([np.cos(yaw), np.sin(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
        yaw1 = yaw + cmd[1] * dt
        x, y = x + cmd[0] * np.cos(yaw + cmd[1] * dt / 2) * dt, y + cmd[0] * np.sin(yaw + cmd[1] * dt / 2) * dt
        o1 = np.array([x, y]) + S.CAM_FWD * np.array([np.cos(yaw1), np.sin(yaw1)])
        move = (yaw1 - yaw, float((o1 - o0) @ [f0[1], -f0[0]]), float((o1 - o0) @ f0)); yaw = yaw1
    tag = f"{pi:02d}_{r['name'].replace('>', '-').replace(' ', '_')}"
    with open(f"{out}/{tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "x", "y", "yaw", "v", "w", "offset", "progress", "zone", "has_center", "classes"]] + rows)
    top = top0.copy(); px = lambda p: np.c_[p[:, 0] * 1000 * ppm, Hpx - p[:, 1] * 1000 * ppm].astype(np.int32)
    cv2.polylines(top, [px(path)], False, (0, 160, 0), 4); cv2.polylines(top, [px(np.array([[a[1], a[2]] for a in rows]))], False, (0, 0, 255), 6)
    cv2.imwrite(f"{out}/{tag}_top.jpg", cv2.resize(top, None, fx=0.4, fy=0.4))
    o = np.abs([a[6] for a in rows[started or 0:]]); z = rows[-1][8]
    row = dict(i=pi, name=r["name"], kind=r["kind"], end=end, zone_end=z, done=round(float(rows[-1][7]), 2), total=round(float(s_path[-1] - END), 2), half_mm=round(half * 1000),
               med_mm=round(float(np.median(o)) * 1000, 1), max_mm=round(float(o.max()) * 1000, 1), no_center=sum(a[9] == 0 for a in rows[started or 0:]), frames=len(rows), sec=round(time.time() - t0))
    res_rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(res_rows, open(out + "/summary.json", "w"), ensure_ascii=False)
sc.close()
