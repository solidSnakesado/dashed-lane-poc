"""비전 감시 시험 (Route Server 문서 '주행 검증 순서' 2단계: 비전 감시 추가, 불일치 발생 위치 기록).

phys_drive.py가 남긴 물리 주행 기록(경로마다 NN.npy: 실제 위치와 로봇이 추정한 위치)을 다시 따라가며,
  실제 위치에서 카메라 화면을 만들고 → 모델(또는 정답 라벨)로 내 차로 선을 찾고 → 추정 위치와 지도로 계산한 선과 비교한다(lane_check).
교차로 안은 칠해진 선이 없으므로 체크하지 않는다(추정 위치가 교차로 안이면 건너뜀. 지도 선도 교차로 밖 부분만 씀).
기록: 프레임마다 [시간, 실제 x, y, 추정 x, y, 경로에서 벗어난 거리, 추정 위치의 실제 옆 오차, 체크가 낸 옆 오차, 경고, 상태]
  상태: 0 = 교차로 안(체크 안 함), 1 = 체크함, 2 = 선이 모자라 체크 못 함
사용: python vision_watch.py ROAD 주행기록폴더 OUT [경로 번호 ...]   환경 변수 WEIGHTS(없으면 정답 라벨), EVERY(몇 줄마다 한 프레임, 기본 2 = 10 Hz)
"""
import os, pickle, subprocess, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import lane_direct, lane_check
from ego_label import Routes
import sim_closed as S

CTRL_HZ = 20.0       # 주행 기록의 줄 간격(phys_drive.py의 제어 주기)


def watch(sc, rt, j, R, model, calib, cam_h, every, look):
    """경로 j의 주행 기록 R을 따라가며 프레임마다 체크한 결과 줄들을 돌려준다."""
    class A: imgsz, conf = 640, 0.25
    r = rt.r[j]; lw = (r["wl"].mean() + r["wr"].mean()) / 1000
    P = r["p"] / 1000; nrm = np.c_[-np.sin(r["ang"]), np.cos(r["ang"])]
    LL, RR = P + nrm * (r["wl"] / 1000)[:, None], P - nrm * (r["wr"] / 1000)[:, None]     # 지도의 왼쪽 선, 오른쪽 선
    live, move, rows, seen = lane_direct.Direct(calib, CTRL_HZ / every, "line"), None, [], []
    T = R[::every]
    for i, row in enumerate(T):
        x, y, yaw = row[1:4]; ex, ey, eyaw = row[4:7]
        img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw))); img = S.look(img, *look)
        det = S.detect(model, img, inst, mine, sc, A()); res = live.step(det, (sc.W, sc.H), move, "lane")
        seen.append(dict(cls=sorted(set(c for c, _ in det)), left=None if res is None else res["left"], right=None if res is None else res["right"]))
        if i + 1 < len(T):                                               # 다음 프레임까지 카메라가 움직인 양(기억용)
            x1, y1, yaw1 = T[i + 1][1:4]; f0 = np.array([np.cos(yaw), np.sin(yaw)])
            o0 = np.array([x, y]) + S.CAM_FWD * f0; o1 = np.array([x1, y1]) + S.CAM_FWD * np.array([np.cos(yaw1), np.sin(yaw1)])
            move = (float(yaw1 - yaw), float((o1 - o0) @ [f0[1], -f0[0]]), float((o1 - o0) @ f0))
        hl = np.array([-np.sin(yaw), np.cos(yaw)]); true_e = float((np.array([ex, ey]) - [x, y]) @ hl)   # 추정 위치의 실제 옆 오차(왼쪽 +)
        k = int(np.argmin(np.linalg.norm(P - [ex, ey], axis=1)))        # 감시는 추정 위치만 안다
        base = [row[0], x, y, ex, ey, row[9], true_e]
        if r["en"] <= k <= r["ex"]:
            rows.append(base + [np.nan, 0, 0]); continue
        lo, hi = (max(0, k - 20), min(len(P), k + 250, r["en"] + 1)) if k < r["en"] else (max(r["ex"], k - 20), min(len(P), k + 250))
        exp = dict(left=lane_check.to_cam(LL[lo:hi], (ex, ey, eyaw), S.CAM_FWD),
                   right=lane_check.to_cam(RR[lo:hi], (ex, ey, eyaw), S.CAM_FWD) if r["right"] else None)
        c = None if res is None else lane_check.check(dict(left=res["left"], right=res["right"]), exp, cam_h, S.CAM_FWD, lw)
        rows.append(base + ([np.nan, 0, 2] if c is None else [c["lat"], float(c["warn"]), 1]))
    return np.array(rows), lw, r, seen


def line(j, r, D, lw):
    """경로 하나의 요약 한 줄."""
    out, ok = D[D[:, 9] > 0], D[D[:, 9] == 1]
    diff = np.abs(ok[:, 7] - ok[:, 6]) * 1000 if len(ok) else np.array([np.nan])
    off = np.abs(out[:, 5]) * 1000 if len(out) else np.array([0.0])
    return (f"{j:2d} {r['name']:12s} {r['kind']:8s} 교차로 밖 프레임 {len(out):3d}, 체크함 {len(ok):3d} ({len(ok) / max(len(out), 1):4.0%}), 경고 {int(ok[:, 8].sum()):3d} | "
            f"체크가 낸 옆 오차와 실제의 차이 mm(중앙/95%/최대) {np.median(diff):.1f}/{np.percentile(diff, 95):.1f}/{diff.max():.1f} | "
            f"경로에서 벗어난 거리 최대 {off.max():.0f} mm (차로 폭 {lw * 1000:.0f})")


def main():
    road, runs, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    every = int(os.environ.get("EVERY", 2))
    cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", out + "/calib.npz"
    subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    cam_h = float(np.load(calib)["cam_h"])
    model = __import__("ultralytics").YOLO(os.environ["WEIGHTS"]) if os.environ.get("WEIGHTS") else None
    sc = S.Scene(road, cam_yaml); rt = Routes(sc, S.LATERAL_MM, S.YAW_DEG)
    for r, info in zip(rt.r, sc.info["routes"]):
        r.setdefault("name", info["name"])
    todo = [int(v) for v in sys.argv[4:]] or range(len(rt.r))
    rng = np.random.default_rng(0); np.random.seed(0)
    look = (np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)]), 3.0, S.background(sc.H, sc.W, rng))
    top = cv2.imread(road + "/floor.png"); H, ppm = top.shape[0], sc.info["ppm"]
    px = lambda p: np.c_[p[:, 0] * 1000 * ppm, H - p[:, 1] * 1000 * ppm].astype(np.int32)
    for j in todo:
        D, lw, r, seen = watch(sc, rt, j, np.load(f"{runs}/{j:02d}.npy"), model, calib, cam_h, every, look)
        np.save(f"{out}/{j:02d}.npy", D); pickle.dump(seen, open(f"{out}/{j:02d}_seen.pkl", "wb"))   # 프레임마다 비전이 낸 선(카메라 기준 m)도 남김
        txt = line(j, r, D, lw); print(txt, flush=True); open(f"{out}/{j:02d}.txt", "w").write(txt + "\n")
        for q, col, rad in ((D[D[:, 9] == 1], (0, 160, 0), 3), (D[D[:, 9] == 2], (0, 200, 255), 4), (D[D[:, 8] > 0], (0, 0, 255), 9)):
            for u, v in px(q[:, 1:3]) if len(q) else []:                 # 초록 = 체크함, 노랑 = 체크 못 함, 빨강 = 경고
                cv2.circle(top, (int(u), int(v)), rad, col, -1)
    sc.close()
    cv2.imwrite(f"{out}/watch_top_{'_'.join(map(str, todo))[:40]}.jpg", cv2.resize(top, None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
