"""시뮬 폐루프 주행: 핑키가 인식한 중심선으로 스스로 조향하며 도로를 달린다.

사용 예 (저장소 루트에서):
  MUJOCO_GL=egl python pinky/sim/sim_closed.py ROAD_DIR OUT_DIR --weights ~/pinky_sim/models/pinky_v1/weights/best.pt
  (--weights를 빼면 모델 대신 시뮬 정답 라벨을 검출로 사용)

반복: 화면 생성 → 검출 → lane_live(차선·중심선) → lane_control(조향) → 명령대로 이동 → 다시 화면 생성.
출발 직후에는 천천히 직진하며 도로 규격을 잰다. 바퀴 미끄러짐 같은 물리는 없다(명령한 속도대로 움직임).
출력: closed.avi(주행 화면), closed.csv(프레임별 기록), closed_top.jpg(위에서 본 궤적), 화면 요약
"""
import argparse
import csv
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from ego_label import MINE, Routes, mine_yolo, route  # noqa: E402
from gen_dataset import LATERAL_MM, YAW_DEG, background, to_yolo  # noqa: E402
from lane_direct import Direct  # noqa: E402
from lane_geom import Camera  # noqa: E402
from lane_control import pursuit  # noqa: E402
from lane_live import Live  # noqa: E402
from scene import Scene  # noqa: E402
from sim_drive import END_MARGIN, look, resample  # noqa: E402

CAM_FWD = 0.033      # 바퀴 축에서 카메라까지 앞쪽 거리(m). pinky_camera.yaml의 forward_m과 같은 값
LOST_LANES = 1.0     # 중심선을 놓치면 직전 명령으로 차로 폭의 이 배수만큼만 더 가고 멈춤
MEASURE_SEC = 6.0    # 규격 측정을 위해 천천히 직진하는 최대 시간(초). 그래도 못 재면 멈춤
STOP_SEC = 2.0       # 출발한 뒤 이 시간(초) 동안 멈춰 있으면 끝냄


def detect(model, img, inst, mine, sc, a):
    """검출 결과 [(클래스 이름, 픽셀 폴리곤)]. 모델이 없으면 시뮬 정답 라벨을 쓴다(내 차로 선 라벨 포함)."""
    names = sc.info["classes"] + MINE
    if model is None:
        rows = [ln.split() for ln in to_yolo(inst, sc.info["instances"], names) + mine_yolo(mine, names)]
        return [(names[int(r[0])], np.array(r[1:], float).reshape(-1, 2) * (sc.W, sc.H)) for r in rows]
    r = model.predict(img, imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    return [(model.names[int(c)], np.asarray(xy, np.float64)) for xy, c in zip(r.masks.xy, r.boxes.cls.cpu().numpy()) if len(xy) >= 3]


def main():
    ap = argparse.ArgumentParser(description="시뮬 폐루프 주행")
    ap.add_argument("road"), ap.add_argument("out")
    ap.add_argument("--cam", default=os.path.join(ROOT, "pinky", "config", "pinky_camera.yaml"))
    ap.add_argument("--weights", default="", help="검출 모델. 비우면 시뮬 정답 라벨 사용")
    ap.add_argument("--imgsz", type=int, default=640), ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--path", type=int, default=0, help="달릴 경로 번호 (교차로 도로는 경로마다 직진·좌회전이 정해져 있음)")
    ap.add_argument("--speed", type=float, default=0.2, help="전진 속도(m/s)")
    ap.add_argument("--w-max", type=float, default=1.5, help="최대 회전 속도(rad/s)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direct", action="store_true", help="모델이 내놓은 내 차로 선(ego_left·ego_right)을 그대로 씀 (lane_direct.py)")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)
    calib = os.path.join(out, "calib.npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "pinky", "config", "pinky_camera.py"), a.cam, calib],
                   check=True, capture_output=True)
    model = None
    if a.weights:
        from ultralytics import YOLO
        model = YOLO(os.path.expanduser(a.weights))
    sc = Scene(os.path.expanduser(a.road), a.cam)
    lw = sc.info["lane_w"] / 1000.0
    path, s_path = resample(np.asarray(sc.info["paths"][a.path]) / 1000.0, 0.005)     # 실제 차로 중심(채점용)
    tan = np.gradient(path, axis=0)
    x, y, yaw = path[0, 0], path[0, 1], float(np.arctan2(tan[0, 1], tan[0, 0]))
    tint = np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)])
    bg, live, dt = background(sc.H, sc.W, rng), (Direct if a.direct else Live)(calib, a.fps), 1.0 / a.fps
    kind = route(sc.info, a.path)[0]                                  # 가는 방향(일반 도로·직진·좌회전). 실제 핑키에서는 지도의 경로에서 받는 값
    rt, cam_real = Routes(sc, LATERAL_MM, YAW_DEG), Camera(calib)      # 정답 선(모델 없이 돌릴 때), 그리기용 카메라
    vw = cv2.VideoWriter(os.path.join(out, "closed.avi"), cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (sc.W, sc.H))   # 기본 재생기로 열리는 형식
    cmd, lost, still, move, started, rows, end = (0.0, 0.0), 0.0, 0, None, None, [], "시간 초과"
    for i in range(int((s_path[-1] / a.speed + 30) * a.fps)):
        img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw, a.path)))
        img = look(img, tint, 3.0, bg)
        dets = detect(model, img, inst, mine, sc, a)
        res = live.step(dets, (sc.W, sc.H), move, kind) if a.direct else live.step(dets, (sc.W, sc.H), move)
        if a.direct:                                          # 모델이 내놓은 내 차로 선 마스크의 테두리(가는 노란 선)
            cv2.polylines(img, [np.round(p).astype(np.int32) for c, p in dets if c in MINE], True, (0, 220, 255), 1)
        new = None if res is None else pursuit(res["center"], res["look"], a.speed, a.w_max, CAM_FWD)
        if new is not None:
            cmd, lost, started = new, 0.0, (i if started is None else started)
        elif res is not None and lost < LOST_LANES * res["lane_w"]:
            lost += cmd[0] * dt                              # 중심선이 없으면 직전 명령을 잠깐만 유지
        elif started is None and i < MEASURE_SEC * a.fps:    # 아직 중심선을 못 만든 출발 직후: 양쪽 선이 함께 보일 때까지 천천히 직진
            cmd = (a.speed / 2, 0.0)
        else:
            cmd = (0.0, 0.0)
        k = int(np.argmin(np.linalg.norm(path - [x, y], axis=1)))        # 채점: 실제 차로 중심에서 벗어난 거리(왼쪽 +)
        t = tan[k] / (np.linalg.norm(tan[k]) + 1e-9)
        off = float(t[0] * (y - path[k, 1]) - t[1] * (x - path[k, 0]))
        rows.append([i, f"{x:.4f}", f"{y:.4f}", f"{yaw:.4f}", f"{cmd[0]:.3f}", f"{cmd[1]:.3f}", f"{off:.4f}",
                     f"{s_path[k]:.3f}", int(res is not None), int(new is not None)])
        if res is not None:                                   # 인식한 선을 화면에 그림 (중심선 초록)
            for key, col in (("left", (255, 80, 255)), ("right", (180, 130, 255)), ("center", (0, 255, 0))):
                if res[key] is not None:
                    px = cam_real.to_image(res[key][res[key][:, 1] > 0.08])
                    px = px[np.all(np.isfinite(px), axis=1) & (np.abs(px) < 5000).all(axis=1)].astype(np.int32)
                    cv2.polylines(img, [px], False, col, 3)
        cv2.putText(img, f"{i} off={off * 100:+.1f}cm v={cmd[0]:.2f} w={cmd[1]:+.2f}", (8, 20), 0, 0.5, (0, 255, 0), 1)
        vw.write(img)
        still = still + 1 if cmd[0] == 0 else 0
        if abs(off) > lw / 2 or s_path[k] > s_path[-1] - END_MARGIN or still > STOP_SEC * a.fps:
            end = "차로 이탈" if abs(off) > lw / 2 else "완주" if still <= STOP_SEC * a.fps else "중심선이 없어 멈춤"
            break
        o0, f0 = np.array([x, y]) + CAM_FWD * np.array([np.cos(yaw), np.sin(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
        yaw1 = yaw + cmd[1] * dt                              # 명령대로 이동 (두 바퀴 로봇)
        x, y = x + cmd[0] * np.cos(yaw + cmd[1] * dt / 2) * dt, y + cmd[0] * np.sin(yaw + cmd[1] * dt / 2) * dt
        o1 = np.array([x, y]) + CAM_FWD * np.array([np.cos(yaw1), np.sin(yaw1)])
        move = (yaw1 - yaw, float((o1 - o0) @ [f0[1], -f0[0]]), float((o1 - o0) @ f0))   # 바퀴 센서가 잰 것과 같은 값
        yaw = yaw1
    vw.release()
    sc.close()
    with open(os.path.join(out, "closed.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["frame", "x", "y", "yaw", "v", "w", "offset", "progress", "ready", "has_center"]] + rows)
    top = cv2.imread(os.path.join(os.path.expanduser(a.road), "floor.png"))
    ppm, h = sc.info["ppm"], top.shape[0]
    px = lambda p: np.c_[p[:, 0] * 1000 * ppm, h - p[:, 1] * 1000 * ppm].astype(np.int32)
    cv2.polylines(top, [px(path)], False, (0, 160, 0), max(1, int(2 * ppm)))                       # 실제 차로 중심(초록)
    cv2.polylines(top, [px(np.array([[float(r[1]), float(r[2])] for r in rows]))], False, (0, 0, 255), max(2, int(3 * ppm)))
    cv2.imwrite(os.path.join(out, "closed_top.jpg"), cv2.resize(top, None, fx=800 / max(top.shape[:2]), fy=800 / max(top.shape[:2])))
    off = np.abs([float(r[6]) for r in rows[started or 0:]]) / lw
    print(f"경로 {a.path}: {dict(lane='일반 도로', straight='교차로 직진', left='교차로 좌회전')[kind]}")
    print(f"결과: {end}  (주행 {float(rows[-1][7]):.2f} / {s_path[-1] - END_MARGIN:.2f} m, {len(rows)}프레임, 출발 전 측정 {started}프레임)")
    print(f"차로 중심에서 벗어난 거리(차로 폭 {lw * 1000:.0f} mm 대비): 중앙 {np.median(off):.1%}, 최대 {off.max():.1%}"
          f"  | 중심선 없는 프레임 {sum(r[9] == 0 for r in rows[started or 0:])}")
    print(f"저장: {out}/closed.avi, closed.csv, closed_top.jpg")


if __name__ == "__main__":
    main()
