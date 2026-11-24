"""체크 시험(환경 변수 WEIGHTS를 주면 모델 결과, 없으면 정답 라벨. PCAM = 다른 카메라 보정 스크립트): 로봇은 경로를 그대로 따라가고(도로 그래프 주행과 같음), 체크에는 일부러 틀린 위치를 준다. 체크가 넣은 오차를 맞게 내는지 본다.
사용: python check_test.py ROAD 경로번호 [옆오차mm 방향오차도 ...쌍]"""
import os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 저장소 최상위
sys.path.insert(0, ROOT + "/pinky/sim"); sys.path.insert(1, ROOT)
import lane_direct, lane_check
from ego_label import Routes
import sim_closed as S
road, pj = sys.argv[1], int(sys.argv[2]); errs = [(float(a), float(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])] or [(0, 0)]
cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", "/tmp/ck_calib.npz"
subprocess.run([sys.executable, __import__("os").environ.get("PCAM", ROOT + "/pinky/config/pinky_camera.py"), cam_yaml, calib], check=True, capture_output=True)
cam_h = float(np.load(calib)["cam_h"])
class A: imgsz, conf = 640, 0.25
W = __import__("os").environ.get("WEIGHTS")                                      # 주면 모델 결과로, 없으면 정답 라벨로
model = None if not W else __import__("ultralytics").YOLO(W)
sc = S.Scene(road, cam_yaml); rt = Routes(sc, S.LATERAL_MM, S.YAW_DEG); r = rt.r[pj]
kind = r["kind"]; lw = (r["wl"].mean() + r["wr"].mean()) / 1000
P = r["p"] / 1000; nrm = np.c_[-np.sin(r["ang"]), np.cos(r["ang"])]
LL, RR = P + nrm * (r["wl"] / 1000)[:, None], P - nrm * (r["wr"] / 1000)[:, None]       # 지도의 왼쪽 선, 오른쪽 선
traj, _ = S.resample(P, 0.2 / 15); traj = traj[: int(len(traj) * 0.92)]
tan = np.gradient(traj, axis=0); yaws = np.arctan2(tan[:, 1], tan[:, 0])
rng = np.random.default_rng(0); np.random.seed(0)
tint = np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)]); bg = S.background(sc.H, sc.W, rng)
seen_all, idx_all, move = [], [], None
live = lane_direct.Direct(calib, 15.0, "line")
for i in range(len(traj)):                                                       # 1) 진짜 위치로 달리며 비전 결과를 모은다
    x, y, yaw = traj[i, 0], traj[i, 1], float(yaws[i])
    img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw))); img = S.look(img, tint, 3.0, bg)
    res = live.step(S.detect(model, img, inst, mine, sc, A()), (sc.W, sc.H), move, "lane")
    k = int(np.argmin(np.linalg.norm(P - [x, y], axis=1)))
    seen_all.append(None if res is None else dict(left=res["left"], right=res["right"])); idx_all.append(k)
    if i + 1 < len(traj):
        o0, f0 = traj[i] + S.CAM_FWD * np.array([np.cos(yaw), np.sin(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
        o1 = traj[i + 1] + S.CAM_FWD * np.array([np.cos(yaws[i + 1]), np.sin(yaws[i + 1])])
        move = (float(yaws[i + 1] - yaw), float((o1 - o0) @ [f0[1], -f0[0]]), float((o1 - o0) @ f0))
sc.close()
print(f"경로 {pj} ({kind}), 차로 폭 {lw * 1000:.0f} mm, 프레임 {len(traj)}")
for e_mm, d_deg in errs:                                                         # 2) 틀린 위치로 체크
    out, skip = [], 0
    for i, (seen, k) in enumerate(zip(seen_all, idx_all)):
        if r["en"] <= k <= r["ex"]: skip += 1; continue                         # 교차로 안: 칠해진 선이 없어 체크하지 않음(lane_vision 끔)
        x, y, yaw = traj[i, 0], traj[i, 1], float(yaws[i]); hl = np.array([-np.sin(yaw), np.cos(yaw)])
        pose = (x + hl[0] * e_mm / 1000, y + hl[1] * e_mm / 1000, yaw + np.radians(d_deg))     # 추정 위치 = 진짜 + 오차
        lo, hi = (max(0, k - 20), min(len(P), k + 250, r["en"] + 1)) if k < r["en"] else (max(r["ex"], k - 20), min(len(P), k + 250))
        sl = slice(lo, hi)                                                        # 지도 쪽 선도 칠해진 구간만(교차로 안은 빼고)
        exp = dict(left=lane_check.to_cam(LL[sl], pose, S.CAM_FWD), right=lane_check.to_cam(RR[sl], pose, S.CAM_FWD) if r["right"] else None)
        c = None if seen is None else lane_check.check(seen, exp, cam_h, S.CAM_FWD, lw)
        out.append(c)
    got = [c for c in out if c]
    if not got: print(f"  넣은 오차 옆 {e_mm:+.0f} mm, 방향 {d_deg:+.1f}도: 추정 없음"); continue
    el = np.array([c["lat"] * 1000 for c in got]); ed = np.degrees([c["yaw"] for c in got if c["yaw"] is not None]); ny = len(ed); ed = ed if ny else np.array([np.nan]); wn = np.mean([c["warn"] for c in got])
    print(f"  넣은 오차 옆 {e_mm:+5.0f} mm, 방향 {d_deg:+4.1f}도 → 추정 옆 중앙 {np.median(el):+6.1f} (오차 95% {np.percentile(abs(el - e_mm), 95):4.1f}) mm, "
          f"방향 중앙 {np.median(ed):+5.2f} (오차 95% {np.percentile(abs(ed - d_deg), 95):4.2f})도 (방향 낸 프레임 {ny}) | 경고 {wn:.0%} | 체크한 프레임 {len(got)}/{len(out)} (교차로 안 제외 {skip})")
