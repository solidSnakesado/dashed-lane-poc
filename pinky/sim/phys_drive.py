"""물리 주행 시험 (Route Server 문서 '주행 검증 순서' 1단계: 비전 없이 도로 그래프 + RPP로 주행하며 위치 오차 측정).

로봇: pinky_pro의 pinky_mujoco가 핑키 URDF로 만드는 MuJoCo 물리 모델(바퀴 모터·마찰·가속 제한, 엔코더 오도메트리)을 그대로 쓴다.
      pinky_pro는 읽기만 한다. ROS2가 없는 환경에서는 ament_index 대역(AMENT_STUB 폴더)으로 패키지 경로만 찾는다.
경로: road.json의 경로 하나를 Route Server의 dense path처럼 5 cm 간격으로 다시 찍는다.
제어: rpp.py(핑키 nav2_params.yaml의 RPP 설정), 20 Hz. 물리는 pinky_mujoco 설정(0.002 s, 브리지 100 Hz).
위치: LOC=true → 시뮬의 실제 위치를 씀(추종 오차만 봄), LOC=odom → 엔코더 오도메트리만 씀(실물에서 위치 보정이 없을 때).
잰 것: 실제 위치가 경로에서 옆으로 벗어난 거리(교차로 밖/안), 오도메트리 위치와 실제 위치의 차이, 도착 여부.
사용: python phys_drive.py ROAD OUT [경로 번호 ...]   환경 변수: PINKY_PRO, LOC, MINLOOK(최소 주시 거리 m), LOOKFIX(고정 주시 거리 m), VIDEO(영상 만들 경로 번호들)
"""
import json, math, os, subprocess, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PINKY_PRO = os.environ.get("PINKY_PRO", os.path.join(os.path.dirname(HERE), "pinky_pro"))   # 기본: 저장소의 pinky_pro 서브모듈
sys.path[:0] = [os.environ.get("AMENT_STUB", HERE + "/ament_stub"), PINKY_PRO + "/pinky_mujoco", HERE]
from pinky_mujoco.model import build_model, Physics
from rpp import RPP

DENSE = 0.05          # dense path 점 간격(m), Route Server 기본값
CTRL_HZ = 20.0        # 핑키 controller_frequency
GOAL = 0.03           # 남은 경로가 이보다 짧으면 도착(m)
SLACK = 15.0          # 주행 시간 한도 = 경로 길이/속도 + 이 값(s)


def dense_path(p_mm, en, ex):
    """경로 점(mm) → 5 cm 간격 점(m)과 점마다 교차로 안 여부."""
    p = np.asarray(p_mm, float) / 1000.0
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    ss = np.r_[np.arange(0.0, s[-1], DENSE), s[-1]]
    q = np.c_[np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])]
    return q, (ss >= s[en]) & (ss <= s[min(ex, len(s) - 1)])


def lateral(path, xy):
    """점 xy가 경로(선분들)에서 떨어진 거리(m, 왼쪽 +)와 가장 가까운 선분 번호."""
    a, b = path[:-1], path[1:]; v = b - a
    t = np.clip(np.einsum("ij,ij->i", xy - a, v) / (np.einsum("ij,ij->i", v, v) + 1e-12), 0, 1)
    c = a + t[:, None] * v; d = np.linalg.norm(xy - c, axis=1); k = int(np.argmin(d))
    return float(np.sign(v[k, 0] * (xy[1] - a[k, 1]) - v[k, 1] * (xy[0] - a[k, 0])) * d[k]), k


def place(ph, x, y, yaw):
    """물리 로봇을 (x, y, yaw)에 세운다(바퀴 축 기준). 오도메트리는 0에서 시작."""
    ph.data.qpos[0:2] = [x, y]
    ph.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    ph.mj.mj_forward(ph.model, ph.data); ph.odom[:] = 0


def run(model, road_info, j, loc, over):
    r = road_info["routes"][j]
    path, junc = dense_path(road_info["paths"][j], r["en"], r["ex"])
    half = min(r["bounds_in"] + r["bounds_out"]) / 1000.0
    ph = Physics(model, PINKY_PRO + "/pinky_mujoco/config/mujoco.yaml")
    t0 = path[1] - path[0]; yaw0 = math.atan2(t0[1], t0[0]); place(ph, path[0, 0], path[0, 1], yaw0)
    ctl = RPP(path, 1.0 / CTRL_HZ, **over)
    sub = round(1.0 / CTRL_HZ / ph.bridge_period)                       # 제어 한 번에 브리지 주기 몇 번
    rows, cmd, done = [], (0.0, 0.0), False
    for n in range(int((ctl.s[-1] / ctl.p["desired_linear_vel"] + SLACK) * CTRL_HZ)):
        true = ph.pose()
        o = ph.odom; c, s_ = math.cos(yaw0), math.sin(yaw0)              # 오도메트리(출발 자세 기준) → 지도 좌표
        est = np.array([path[0, 0] + c * o[0] - s_ * o[1], path[0, 1] + s_ * o[0] + c * o[1], yaw0 + o[2]])
        use = true if loc == "true" else est
        cmd, remain = ctl.step(use, ph.twist[0])
        if remain < GOAL:
            done = True; break
        for _ in range(sub):
            ph.step(*cmd)
        e, k = lateral(path, true[:2])
        rows.append([n / CTRL_HZ, *true, *est, *cmd, e, int(junc[k] or junc[min(k + 1, len(junc) - 1)]),
                     float(np.linalg.norm(est[:2] - true[:2])), float((est[2] - true[2] + np.pi) % (2 * np.pi) - np.pi)])
    return np.array(rows), path, done, half


def summary(j, name, kind, R, done, half):
    e, z = np.abs(R[:, 9]) * 1000, R[:, 10] > 0
    part = lambda m: f"{np.median(e[m]):.1f}/{np.percentile(e[m], 95):.1f}/{e[m].max():.1f}" if m.any() else "-"
    out = sum(e > half * 1000)
    return (f"{j:2d} {name:12s} {kind:8s} {'도착' if done else '미도착'} {R[-1, 0]:5.1f}s | 경로에서 벗어난 거리 mm(중앙/95%/최대) "
            f"교차로 밖 {part(~z)} 안 {part(z)} | 차로 반폭 {half * 1000:.0f} mm 넘은 제어 주기 {out} | "
            f"오도메트리 오차 끝 {R[-1, 11] * 1000:.1f} mm, {math.degrees(R[-1, 12]):.2f}도")


def top_image(road, info, path, R, out_png):
    """위에서 본 그림: 경로(초록), 실제 궤적(빨강), 오도메트리 궤적(파랑)."""
    top = cv2.imread(road + "/floor.png"); H, ppm = top.shape[0], info["ppm"]
    px = lambda p: np.c_[p[:, 0] * 1000 * ppm, H - p[:, 1] * 1000 * ppm].astype(np.int32)
    cv2.polylines(top, [px(path)], False, (0, 200, 0), 5)
    cv2.polylines(top, [px(R[:, 4:6])], False, (255, 80, 0), 3)
    cv2.polylines(top, [px(R[:, 1:3])], False, (0, 0, 255), 3)
    cv2.imwrite(out_png, cv2.resize(top, None, fx=0.4, fy=0.4))


def video(road, info, j, R, out_mp4):
    """카메라 화면(실제 위치에서 렌더링)과 지도로 그린 주행 차선. 기존 시뮬 렌더러(scene.py)를 쓴다."""
    root = os.path.dirname(os.path.dirname(HERE))                         # 저장소 최상위
    sys.path[:0] = [root + "/pinky/sim", root]
    import sim_closed as S
    from lane_check import to_cam
    from lane_geom import Camera
    from draw_lane import visible
    cam_yaml, calib = root + "/pinky/config/pinky_camera.yaml", out_mp4 + ".npz"
    subprocess.run([sys.executable, root + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    cam, sc = Camera(calib), S.Scene(road, cam_yaml)
    r = info["routes"][j]; p = np.asarray(info["paths"][j], float) / 1000
    t = np.gradient(p, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True); n = np.c_[-t[:, 1], t[:, 0]]
    wl = np.asarray(r.get("wl", [r["bounds_in"][0]] * len(p))) / 1000; wr = np.asarray(r.get("wr", [r["bounds_in"][1]] * len(p))) / 1000
    lines = (p + n * wl[:, None], p - n * wr[:, None])
    vw = cv2.VideoWriter(out_mp4 + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), CTRL_HZ, (sc.W, sc.H))
    for row in R[::2]:
        x, y, yaw = row[1:4]
        img, _ = sc.render(x, y, yaw)
        for Q in lines:
            c = to_cam(Q, (x, y, yaw), S.CAM_FWD); ok = visible(cam, c, sc.W, sc.H) & (c[:, 1] < 0.7)
            if ok.sum() > 1:
                cv2.polylines(img, [cam.to_image(c[ok]).astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{row[0]:.1f}s off {row[9] * 1000:+.0f}mm", (5, 20), 0, 0.5, (255, 255, 255), 1)
        vw.write(img)
    vw.release(); sc.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4 + ".avi", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", out_mp4], check=True); os.remove(out_mp4 + ".avi")


def main():
    road, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    info = json.load(open(road + "/road.json"))
    todo = [int(v) for v in sys.argv[3:]] or range(len(info["routes"]))
    loc = os.environ.get("LOC", "true")
    over = {"min_lookahead_dist": float(os.environ["MINLOOK"])} if os.environ.get("MINLOOK") else {}
    if os.environ.get("LOOKFIX"):                                       # 시험용: 속도 비례 없이 고정 주시 거리
        over.update(use_velocity_scaled_lookahead_dist=False, lookahead_dist=float(os.environ["LOOKFIX"]))
    vids = [int(v) for v in os.environ.get("VIDEO", "").split()]
    model = build_model(PINKY_PRO + "/pinky_mujoco", out + "/model", world=os.path.dirname(HERE) + "/ros/empty_world.yaml", simple_render=True)
    lines = []
    for j in todo:
        r = info["routes"][j]
        R, path, done, half = run(model, info, j, loc, over)
        np.save(f"{out}/{j:02d}.npy", R)
        lines.append(summary(j, r["name"], r["kind"], R, done, half)); print(lines[-1], flush=True)
        top_image(road, info, path, R, f"{out}/{j:02d}_top.jpg")
        if j in vids:
            video(road, info, j, R, f"{out}/{j:02d}_cam.mp4")
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
