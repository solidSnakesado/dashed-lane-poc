"""웨이포인트(점열) 학습 데이터 생성기 (YOLO-pose 형식). gen_dataset.py와 같은 사진을 만들되, 정답이 마스크가 아니라 '가는 순서의 점 K개'다.

  MUJOCO_GL=egl python gen_waypoints.py ROADS_DIR CAMERA_YAML OUT_DIR [--per-road 15] [--per-cross N] [--val 0.15] [--seed 0] [--check 20]
결과:
  OUT_DIR/images/{train,val}/*.jpg, OUT_DIR/labels/{train,val}/*.txt, OUT_DIR/data.yaml (kpt_shape 포함), OUT_DIR/check/
정답 한 줄 = 클래스, 상자(cx cy w h), 점 K개(x y 보임). 모두 0~1로 정규화. 화면 밖이거나 경로가 끝난 점은 "0 0 0".
클래스:
  ego_path  = 내 차로의 중심 경로(일반 도로, 교차로 직진). 로봇에서 가장 가까운 경로 위 지점부터 앞으로.
  left_path = 좌회전 경로(진입 차로 → 원호 → 나가는 차로를 한 줄로). 점열이 교차로에 닿을 만큼 가까울 때부터, 원호를 다 돌 때까지.
점의 위치: 경로를 따라 카메라 높이의 FIRST배 앞에서 시작해 카메라 높이의 GAP배 간격(도로 크기에 걸린 숫자는 쓰지 않음).
정답은 로봇의 위치·방향만으로 정해진다(ego_label.Routes.key와 같은 기준). 가는 방향은 사진에 없으므로 두 클래스를 함께 낸다.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

from ego_label import STEP, Routes
from gen_dataset import EMPTY_KEEP, LATERAL_MM, YAW_DEG, augment, pick_pose
from scene import Scene

NAMES = ["ego_path", "left_path"]
K = 10                 # 점의 수
FIRST, GAP = 2.0, 1.0  # 첫 점까지의 거리, 점 사이 간격 (카메라 높이의 배수)
MIN_SEEN = 3           # 화면에 보이는 점이 이보다 적으면 그 줄은 만들지 않음
PAD = 0.03             # 상자를 점들의 범위에서 이만큼(화면 대비) 넓힘


def project(sc, pts, cam):
    """바닥 위 점들 (N,2) [m] → 실제 화면 픽셀 (N,2), 화면 안에 보이는지 (N,)."""
    p, right, up, f = cam
    c = (np.c_[pts, np.zeros(len(pts))] - p) @ np.stack([right, -up, f]).T        # 카메라 좌표 (x 오른쪽, y 아래, z 앞)
    ok = c[:, 2] > 1e-3
    n = c[:, :2] / np.where(ok, c[:, 2], 1.0)[:, None]
    ok &= (np.abs(n[:, 0]) * sc.f < sc.iw / 2) & (np.abs(n[:, 1]) * sc.f < sc.ih / 2)   # 렌즈 왜곡 식이 맞는 범위 안
    px, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), sc.K, sc.dist)
    px = px.reshape(-1, 2)
    ok &= (px[:, 0] >= 0) & (px[:, 0] <= sc.W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= sc.H - 1)
    return px, ok


def waypoints(sc, rt, x, y, yaw, cam):
    """이 자세의 정답 줄들과 그림용 점들 [(클래스, 픽셀 (K,2), 보임 (K,))]."""
    gap = sc.h * 1000.0 / STEP                                   # 카메라 높이만큼의 경로 점 수
    lines, shown = [], []
    for j, zone in rt.key(x * 1000, y * 1000, yaw):
        r = rt.r[j]
        i0 = int(np.argmin(np.hypot(r["p"][:, 0] - x * 1000, r["p"][:, 1] - y * 1000)))
        idx = np.round(i0 + (FIRST + GAP * np.arange(K)) * gap).astype(int)
        left = r["kind"] == "left"
        if left and (zone == 2 or idx[-1] <= r["en"]):           # 다 돌았거나, 점열이 아직 교차로에 닿지 않음
            continue
        inside = idx < len(r["p"])
        px, ok = project(sc, r["p"][np.minimum(idx, len(r["p"]) - 1)] / 1000.0, cam)
        ok &= inside
        if ok.sum() < MIN_SEEN:
            continue
        u, v = px[ok, 0] / sc.W, px[ok, 1] / sc.H
        x0, x1, y0, y1 = max(0, u.min() - PAD), min(1, u.max() + PAD), max(0, v.min() - PAD), min(1, v.max() + PAD)
        kp = " ".join(f"{a / sc.W:.5f} {b / sc.H:.5f} 2" if o else "0 0 0" for (a, b), o in zip(px, ok))
        lines.append(f"{int(left)} {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f} {kp}")
        shown.append((int(left), px, ok))
    return lines, shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roads"), ap.add_argument("cam"), ap.add_argument("out")
    ap.add_argument("--per-road", type=int, default=15)
    ap.add_argument("--per-cross", type=int, default=0, help="교차로 도로의 도로당 장수 (0이면 --per-road와 같음)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=20)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    rnd = yaml.safe_load(open(a.cam))["randomize"]
    roads = sorted(glob.glob(os.path.join(os.path.expanduser(a.roads), "road_*")))
    for s in ("train", "val"):
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    os.makedirs(os.path.join(out, "check"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n_val = max(1, int(round(len(roads) * a.val)))
    val_roads = set(rng.choice(len(roads), n_val, replace=False).tolist())   # 검증은 학습에 안 쓴 도로로 (gen_dataset.py와 같은 도로)
    total, checked, counts = 0, 0, np.zeros(len(NAMES), int)
    for ri, road in enumerate(roads):
        split = "val" if ri in val_roads else "train"
        sc = Scene(road, a.cam)
        made, tries, rt = 0, 0, Routes(sc, LATERAL_MM, YAW_DEG)
        want = a.per_cross if a.per_cross and sc.info["type"] == "cross" else a.per_road
        while made < want and tries < want * 3:
            tries += 1
            x, y, yaw = pick_pose(rt, rng)
            jit = dict(dh=rng.uniform(-1, 1) * rnd["height_m"], dtilt=rng.uniform(-1, 1) * rnd["tilt_deg"], droll=rng.uniform(-1, 1) * rnd["roll_deg"])
            img, _ = sc.render(x, y, yaw, **jit)
            lines, shown = waypoints(sc, rt, x, y, yaw, sc.pose(x, y, yaw, **jit))
            if not lines and rng.random() > EMPTY_KEEP:
                continue
            img = augment(img, rng)
            name = f"{os.path.basename(road)}_{made:04d}"
            cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img)
            open(os.path.join(out, "labels", split, name + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
            for c, _, _ in shown:
                counts[c] += 1
            if checked < a.check:        # 확인용: ego_path 초록, left_path 주황. 첫 점은 크게, 선으로 가는 순서를 이음
                both = img.copy()
                for c, px, ok in shown:
                    col, q = ((0, 255, 0), (0, 140, 255))[c], px[ok].astype(int)
                    cv2.polylines(both, [q], False, col, 1)
                    for k, (u, v) in enumerate(q):
                        cv2.circle(both, (int(u), int(v)), 6 if k == 0 else 3, col, -1)
                cv2.imwrite(os.path.join(out, "check", name + ".jpg"), np.hstack([img, both]))
                checked += 1
            made += 1
        sc.close()
        total += made
        print(f"[{ri + 1}/{len(roads)}] {os.path.basename(road)} ({split}): {made}장")
    yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train", "val": "images/val", "kpt_shape": [K, 3],
                    "names": {i: n for i, n in enumerate(NAMES)}}, open(os.path.join(out, "data.yaml"), "w"),
                   allow_unicode=True, sort_keys=False)
    print(f"끝. 총 {total}장 (검증 도로 {n_val}개), 점열 수: " + ", ".join(f"{n} {c}" for n, c in zip(NAMES, counts)))
    print(f"저장: {out}  확인용 그림: {out}/check/")


if __name__ == "__main__":
    main()
