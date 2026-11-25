"""주행하며 시험: 물리 주행(도로 그래프 경로 + RPP, 위치는 엔코더 오도메트리)을 하는 동안 차로 안 위치(lane_where)와 비전 확인을 같이 돌린다.

제어 주기(20 Hz)마다 물리를 한 걸음 가고, EVERY 주기마다 한 번: 실제 위치에서 카메라 화면을 만들고 → 모델(또는 정답 라벨)로 내 차로 선을 찾고
→ 추정 위치와 지도로 '어느 차로의 어디쯤인지'(상태, 벗어난 거리)를 내고 → 비전이 본 선으로 그 위치를 확인한다. 주행 제어에는 비전을 쓰지 않는다.
기록(NN.npy) 줄: [시간, 실제 x, y, yaw, 추정 x, y, yaw, 경로에서 벗어난 거리, 상태, 모듈이 낸 벗어난 거리, 왼쪽 선까지, 오른쪽 선까지, 확인 상태, 비전 옆 오차, 실제 상태]
  상태: 0 lane(차로 안), 1 out(밖), 2 cross(교차로 안), 3 other(길에 없는 다른 차로) / 확인 상태: 0 none(확인 못 함), 1 ok(확인됨), 2 mismatch(불일치)
사용: python drive_watch.py ROAD OUT [경로 번호 ...]
환경 변수: WEIGHTS(없으면 정답 라벨), PINKY_PRO, EVERY(기본 2 = 10 Hz), VIDEO(영상 만들 경로 번호들), LOOKFIX(고정 주시 거리 m)
"""
import json, math, os, subprocess, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
from phys_drive import PINKY_PRO, CTRL_HZ, GOAL, SLACK, dense_path, lateral, place, Physics, RPP, build_model
import lane_direct, lane_check
from ego_label import Routes, STEP
from route_graph import Graph
from lane_where import Where
from lane_geom import Camera
from draw_lane import visible
from where_test import truth
import sim_closed as S

STATES, CONF = ("lane", "out", "cross", "other"), ("none", "ok", "mismatch")
COL = dict(lane=(0, 220, 0), out=(0, 0, 255), cross=(0, 220, 255), other=(0, 140, 255))


def draw(img, cam, w, out, pose, row):
    """화면에 지도로 계산한 내 차로 선(추정 위치 기준, 교차로 안 포함)과 상태 글자를 겹친다."""
    ex = w.lane_ahead(out["k"], pose) if out["state"] != "out" else None       # 보라 = 칠해진 선이 있는 구간, 노랑 = 교차로 안(지도를 믿고 그림)
    for side in ("left", "right") if ex else ():
        c, z = ex[side], ex["zone"]; ok = visible(cam, c, img.shape[1], img.shape[0]); px = cam.to_image(c[ok]).astype(np.int32) if ok.any() else []
        idx = np.where(ok)[0]
        for a in range(len(px) - 1):
            if idx[a + 1] == idx[a] + 1:                                           # 화면 안에서 이어진 점끼리만 잇는다
                cv2.line(img, tuple(map(int, px[a])), tuple(map(int, px[a + 1])), (0, 255, 255) if z[idx[a]] else (255, 0, 255), 2, cv2.LINE_AA)
    lat = "" if out["lat"] is None else f" vision {out['lat'] * 1000:+.0f}mm"
    for i, txt in enumerate((f"{row[0]:5.1f}s {out['state']}  off {out['off'] * 1000:+.0f}mm  L {out['left'] * 1000:.0f}  R {out['right'] * 1000:.0f}",
                             f"check: {out['status']}{lat}")):
        cv2.putText(img, txt, (6, 20 + 20 * i), 0, 0.5, (0, 0, 0), 3); cv2.putText(img, txt, (6, 20 + 20 * i), 0, 0.5, COL[out["state"]] if i == 0 else (255, 255, 255), 1)
    return img


def run(phys_model, sc, rt, g, w, yolo, calib, cam, j, every, over, look, vid):
    """경로 j를 물리로 달리며 EVERY 주기마다 위치·비전 확인을 돌린다. 돌려주는 값: 기록 줄들, 도착 여부."""
    class A: imgsz, conf = 640, 0.25
    r, info = rt.r[j], sc.info["routes"][j]
    path, _ = dense_path(sc.info["paths"][j], info["en"], info["ex"])
    ph = Physics(phys_model, PINKY_PRO + "/pinky_mujoco/config/mujoco.yaml")
    t0 = path[1] - path[0]; yaw0 = math.atan2(t0[1], t0[0]); place(ph, path[0, 0], path[0, 1], yaw0)
    ctl = RPP(path, 1.0 / CTRL_HZ, **over); sub = round(1.0 / CTRL_HZ / ph.bridge_period)
    w.set_route(np.where(g.pid[:, 0] == j)[0])
    live, move, last, rows, done = lane_direct.Direct(calib, CTRL_HZ / every, "line"), None, None, [], False
    for n in range(int((ctl.s[-1] / ctl.p["desired_linear_vel"] + SLACK) * CTRL_HZ)):
        true = np.array(ph.pose()); o = ph.odom; c, s_ = math.cos(yaw0), math.sin(yaw0)       # 오도메트리(출발 자세 기준) → 지도 좌표
        est = np.array([path[0, 0] + c * o[0] - s_ * o[1], path[0, 1] + s_ * o[0] + c * o[1], yaw0 + o[2]])
        cmd, remain = ctl.step(est, ph.twist[0])                                              # 주행은 추정 위치와 경로만 씀
        if remain < GOAL:
            done = True; break
        if n % every == 0:
            x, y, yaw = true; f0 = np.array([math.cos(yaw), math.sin(yaw)]); o1 = true[:2] + S.CAM_FWD * f0
            if last is not None:                                                              # 직전 프레임 이후 카메라가 움직인 양(중심선 기억용)
                move = (float(yaw - last[2]), float((o1 - last[:2]) @ [last[4], -last[3]]), float((o1 - last[:2]) @ last[3:5]))
            last = np.r_[o1, yaw, f0]
            img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw))); img = S.look(img, *look)
            res = live.step(S.detect(yolo, img, inst, mine, sc, A()), (sc.W, sc.H), move, "lane")
            out = w.step(tuple(est), None if res is None else dict(left=res["left"], right=res["right"]))
            e, _ = lateral(path, true[:2])
            row = [n / CTRL_HZ, *true, *est, e, STATES.index(out["state"]), out["off"], out["left"], out["right"], CONF.index(out["status"]),
                   np.nan if out["lat"] is None else out["lat"], STATES.index(truth(r, true[:2] * 1000)[0])]
            rows.append(row)
            if vid is not None:
                vid.write(draw(img, cam, w, out, tuple(est), row))
        for _ in range(sub):
            ph.step(*cmd)
    return np.array(rows), done


def line(j, name, kind, D, done):
    """경로 하나의 요약 한 줄."""
    st, cf = D[:, 8].astype(int), D[:, 12].astype(int); lane = st == 0
    cnt = " ".join(f"{STATES[k]} {int((st == k).sum())}" for k in range(4) if (st == k).any())
    same = np.mean((st == D[:, 14]) | ((st == 3) & (D[:, 14] == 1)))
    return (f"{j:2d} {name:12s} {kind:8s} {'도착' if done else '미도착'} {D[-1, 0]:5.1f}s | 상태 {cnt} (실제와 같음 {same:.0%}) | "
            f"차로 가운데에서 최대 {np.abs(D[lane, 9]).max() * 1000 if lane.any() else 0:.0f} mm | "
            f"비전 확인: 확인됨 {int((cf[lane] == 1).sum())}, 불일치 {int((cf[lane] == 2).sum())}, 못 함 {int((cf[lane] == 0).sum())}")


def main():
    road, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    every = int(os.environ.get("EVERY", 2)); vids = [int(v) for v in os.environ.get("VIDEO", "").split()]
    over = dict(use_velocity_scaled_lookahead_dist=False, lookahead_dist=float(os.environ["LOOKFIX"])) if os.environ.get("LOOKFIX") else {}
    cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", out + "/calib.npz"
    subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    yolo = __import__("ultralytics").YOLO(os.environ["WEIGHTS"]) if os.environ.get("WEIGHTS") else None
    sc = S.Scene(road, cam_yaml); rt = Routes(sc, S.LATERAL_MM, S.YAW_DEG); g = Graph(rt, STEP, road)
    w, cam = Where(g, float(np.load(calib)["cam_h"]), S.CAM_FWD), Camera(calib)
    rng = np.random.default_rng(0); np.random.seed(0)
    look = (np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)]), 3.0, S.background(sc.H, sc.W, rng))
    phys_model = build_model(PINKY_PRO + "/pinky_mujoco", out + "/model", world=os.path.dirname(HERE) + "/ros/empty_world.yaml", simple_render=True)
    todo, lines, allD = [int(v) for v in sys.argv[3:]] or range(len(rt.r)), [], []
    for j in todo:
        vid = cv2.VideoWriter(f"{out}/{j:02d}.avi", cv2.VideoWriter_fourcc(*"MJPG"), CTRL_HZ / every, (sc.W, sc.H)) if j in vids else None
        D, done = run(phys_model, sc, rt, g, w, yolo, calib, cam, j, every, over, look, vid)
        np.save(f"{out}/{j:02d}.npy", D); allD.append(D)
        lines.append(line(j, sc.info["routes"][j]["name"], sc.info["routes"][j]["kind"], D, done)); print(lines[-1], flush=True)
        if vid is not None:
            vid.release()
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/{j:02d}.avi", "-c:v", "libx264", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", f"{out}/{j:02d}_watch.mp4"], check=True); os.remove(f"{out}/{j:02d}.avi")
    sc.close()
    D = np.vstack(allD); lane = D[:, 8] == 0; cf = D[lane, 12]
    lines.append(f"합계 {len(D)}프레임: 차로 안 {int(lane.sum())}, 그중 비전 확인됨 {np.mean(cf == 1):.0%}, 불일치 {int((cf == 2).sum())} ({np.mean(cf == 2):.1%}), 확인 못 함 {np.mean(cf == 0):.0%}")
    print(lines[-1]); open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
