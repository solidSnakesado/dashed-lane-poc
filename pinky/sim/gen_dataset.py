"""핑키 차선 학습 데이터 생성기 (YOLO-seg). 생성한 도로마다 핑키 카메라 화면과 라벨을 무작위로 만든다.

  MUJOCO_GL=egl python gen_dataset.py ROADS_DIR CAMERA_YAML OUT_DIR [--per-road 100] [--per-cross N] [--per-cross2 N] [--per-track N] [--val 0.15] [--seed 0] [--check 20]
  (--per-cross = 교차로 도로에서만 도로당 장수를 따로 줌. 안 주면 --per-road와 같음.
   --per-cross2 = 편도 2차로 교차로에서만 따로 줌. 안 주면 --per-cross와 같음.
   --per-track = 트랙 도로(gen_track_save.py로 만든 것)에서만 따로 줌. 안 주면 --per-road와 같음)
결과:
  OUT_DIR/images/{train,val}/*.jpg, OUT_DIR/labels/{train,val}/*.txt, OUT_DIR/data.yaml (상대 경로)
  OUT_DIR/check/  라벨을 겹친 확인용 그림 (--check 장수)
무작위로 바꾸는 것: 차로 안 좌우 위치 ±60mm, 방향 ±15도(회전 원호 위에서는 차로 폭의 ±0.4배, ±30도), 카메라 기울기·좌우·높이(pinky_camera.yaml의 randomize),
  밝기·대비, 색조(실물의 초록빛 틀어짐 포함), 그림자, 흐림, 노이즈, JPEG 압축, 지평선 위·보드 밖 배경.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import yaml

from ego_label import MINE, STEP, TURN_LW, TURN_YAW, TURNS, Routes, mine_yolo
from scene import Scene, overlay

LATERAL_MM, YAW_DEG = 60, 15     # 차로 중심에서 벗어나는 범위
NEAR, NEAR_LW = 0.8, 4.0         # 교차로 도로에서 교차로 근처를 찍는 비율, 입구 앞으로 포함하는 거리(차로 폭의 배수)
NEAR_TRACK = 0.5                 # 트랙 도로에서의 그 비율. 교차로 밖에도 급한 모서리가 있어 낮춘다
TURN_IN = 0.5                    # 회전 경로에서 교차로 근처를 찍을 때 원호 위(교차로 안)에서 고르는 비율
EMPTY_KEEP = 0.1                 # 라벨이 하나도 없는 화면은 이 비율만 남김
MIN_AREA = 25                    # 이보다 작은 조각(픽셀)은 라벨에서 뺌
# 확인용 그림의 색(MINE 순서): 내 차로 왼쪽·오른쪽·중심, 좌회전 면·왼쪽·오른쪽, 우회전(가까운 쪽·가운데·먼 쪽)마다 면·왼쪽·오른쪽
CHECK_COL = [(255, 0, 255), (255, 150, 200), (0, 255, 0), (0, 0, 170), (0, 140, 255), (0, 255, 255), (170, 90, 0), (255, 255, 0), (255, 255, 255),
             (0, 110, 60), (120, 255, 120), (200, 255, 200), (110, 0, 110), (255, 120, 255), (255, 200, 255)]


def pick_pose(rt, rng):
    """경로 하나의 한 지점을 고르고 좌우 위치·방향을 흔든다. (x, y [m], yaw [rad])

    교차로 도로는 NEAR 비율만큼을 교차로 근처(입구에서 차로 폭의 NEAR_LW배 앞 ~ 나오는 곳)에서 고른다. 직진 경로의 교차로 안도 포함한다.
    회전(좌회전·우회전) 경로는 교차로를 나오기 전까지만 고른다(나온 뒤의 차로는 그 차로를 지나는 다른 경로에서 고른다).
    회전 원호 위에서는 더 크게 흔든다(TURN_LW, TURN_YAW). 벗어난 자세에서 회전 면이 치우쳐 보이는 것을 배우게 하려는 것이다."""
    r = rt.r[int(rng.integers(len(rt.r)))]
    n, cross, turn = len(r["p"]), r["kind"] != "lane", r["kind"] in TURNS
    lo, hi = 2, max(3, r["ex"] if turn else int(n * 0.85))
    if cross and rng.random() < (NEAR_TRACK if rt.info["type"] == "track" else NEAR):
        lo, hi = max(lo, r["en"] - int(NEAR_LW * rt.info["lane_w"] / STEP)), min(hi, r["ex"]) + 1
        lo = r["en"] if turn and rng.random() < TURN_IN else lo
    i = int(rng.integers(lo, max(lo + 1, hi)))
    yaw, arc = r["ang"][i], turn and r["en"] <= i <= r["ex"]
    lat, turn = (max(LATERAL_MM, TURN_LW * rt.info["lane_w"]), max(YAW_DEG, TURN_YAW)) if arc else (LATERAL_MM, YAW_DEG)
    x, y = r["p"][i] + rng.uniform(-lat, lat) * np.array([-np.sin(yaw), np.cos(yaw)])
    return x / 1000.0, y / 1000.0, yaw + np.radians(rng.uniform(-turn, turn))


def background(h, w, rng):
    """보드 너머의 방을 흉내 낸 무작위 배경(바닥색, 가구 같은 사각형, 줄, 노이즈)."""
    bg = np.full((h, w, 3), rng.integers(30, 200, 3), np.uint8)
    for _ in range(rng.integers(3, 12)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        cv2.rectangle(bg, (x0, y0), (x0 + rng.integers(10, w // 2), y0 + rng.integers(10, h // 2)),
                      rng.integers(0, 255, 3).tolist(), -1)
    for _ in range(rng.integers(0, 5)):
        cv2.line(bg, tuple(rng.integers(0, w, 2).tolist()), tuple(rng.integers(0, h, 2).tolist()),
                 rng.integers(150, 255, 3).tolist(), int(rng.integers(1, 4)))
    bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(1, 6))
    return np.clip(bg + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)


def augment(img, rng):
    """촬영 조건 무작위화. 바닥 밖(렌더링 결과가 완전히 검은 곳)은 배경으로 채운다."""
    h, w = img.shape[:2]
    out_floor = cv2.dilate((img.max(axis=2) == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img = img.astype(np.float32)
    img[out_floor] = background(h, w, rng)[out_floor]
    if rng.random() < 0.3:                                   # 부드러운 그림자
        m = np.zeros((h, w), np.float32)
        pts = rng.integers(0, [w, h], (rng.integers(3, 7), 2)).astype(np.int32)
        cv2.fillPoly(m, [cv2.convexHull(pts)], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(5, 30))
        img *= (1 - rng.uniform(0.2, 0.5) * m)[..., None]
    img = img * rng.uniform(0.6, 1.5) + rng.uniform(-25, 25)  # 대비, 밝기
    gain = np.array([rng.uniform(0.9, 1.15), rng.uniform(0.95, 1.2), rng.uniform(0.75, 1.05)])  # B, G, R
    img = np.clip(img * gain, 0, 255)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.2:                                   # 흔들림 번짐
        k = int(rng.integers(3, 8))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(ker, M, (k, k)))
    img = np.clip(img + rng.normal(0, rng.uniform(0, 8), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def to_yolo(inst, classes, names):
    """선 번호 그림 → YOLO-seg 줄들 (클래스 번호, 정규화된 다각형)."""
    h, w = inst.shape
    lines = []
    for k in np.unique(inst):
        if k == 0:
            continue
        cls = names.index(classes[str(int(k))])
        cs, _ = cv2.findContours((inst == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2)
            if len(c) >= 3:
                lines.append(f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roads")
    ap.add_argument("cam")
    ap.add_argument("out")
    ap.add_argument("--per-road", type=int, default=100)
    ap.add_argument("--per-cross", type=int, default=0, help="교차로 도로의 도로당 장수 (0이면 --per-road와 같음)")
    ap.add_argument("--per-cross2", type=int, default=0, help="편도 2차로 교차로의 도로당 장수 (0이면 --per-cross와 같음)")
    ap.add_argument("--per-track", type=int, default=0, help="트랙 도로의 도로당 장수 (0이면 --per-road와 같음)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=20)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    rnd = yaml.safe_load(open(a.cam))["randomize"]
    roads = sorted(glob.glob(os.path.join(os.path.expanduser(a.roads), "road_*")))
    names = json.load(open(os.path.join(roads[0], "road.json")))["classes"] + MINE
    for s in ("train", "val"):
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    os.makedirs(os.path.join(out, "check"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n_val = max(1, int(round(len(roads) * a.val)))
    val_roads = set(rng.choice(len(roads), n_val, replace=False).tolist())   # 검증은 학습에 안 쓴 도로로
    total, checked, counts = 0, 0, np.zeros(len(names), int)
    for ri, road in enumerate(roads):
        split = "val" if ri in val_roads else "train"
        sc = Scene(road, a.cam)
        made, tries, rt = 0, 0, Routes(sc, LATERAL_MM, YAW_DEG)
        want = a.per_cross if a.per_cross and sc.info["type"] == "cross" else a.per_road
        want = a.per_cross2 if a.per_cross2 and sc.info["type"] == "cross" and sc.info["lanes_per_dir"] == 2 else want
        want = a.per_track if a.per_track and sc.info["type"] == "track" else want
        while made < want and tries < want * 3:
            tries += 1
            x, y, yaw = pick_pose(rt, rng)
            img, inst, mine = sc.render(x, y, yaw, extra=rt.floor(rt.key(x * 1000, y * 1000, yaw)), dh=rng.uniform(-1, 1) * rnd["height_m"],
                                        dtilt=rng.uniform(-1, 1) * rnd["tilt_deg"], droll=rng.uniform(-1, 1) * rnd["roll_deg"])
            lines = to_yolo(inst, sc.info["instances"], names) + mine_yolo(mine, names)
            if not lines and rng.random() > EMPTY_KEEP:
                continue
            img = augment(img, rng)
            name = f"{os.path.basename(road)}_{made:04d}"
            cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img)
            open(os.path.join(out, "labels", split, name + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
            for ln in lines:
                counts[int(ln.split()[0])] += 1
            if checked < a.check:
                both = img.copy()        # 확인용: 가운데 = 기존 라벨, 오른쪽 = 내 차로 선과 회전 면·선(CHECK_COL의 색). 면을 먼저, 선을 위에
                for k in sorted(range(len(MINE)), key=lambda k: not MINE[k].endswith("_lane")):
                    both[(mine >> k) & 1 > 0] = CHECK_COL[k]
                cv2.imwrite(os.path.join(out, "check", name + ".jpg"), np.hstack([img, overlay(img, inst, sc.info["instances"]), both]))
                checked += 1
            made += 1
        sc.close()
        total += made
        print(f"[{ri + 1}/{len(roads)}] {os.path.basename(road)} ({split}): {made}장")
    yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train", "val": "images/val",
                    "names": {i: n for i, n in enumerate(names)}}, open(os.path.join(out, "data.yaml"), "w"),
                   allow_unicode=True, sort_keys=False)
    print(f"끝. 총 {total}장 (검증 도로 {n_val}개), 라벨 조각 수: " + ", ".join(f"{n} {c}" for n, c in zip(names, counts)))
    print(f"저장: {out}  확인용 그림: {out}/check/")


if __name__ == "__main__":
    main()
