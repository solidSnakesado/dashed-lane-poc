"""AI 허브 '승용 자율주행차 주간 도심도로' 2D 라벨 → YOLO 분할 데이터셋 변환.

라벨 JSON의 폴리곤 중 노면 표시 클래스만 골라 YOLO-seg 형식(txt)으로 바꾸고,
주행(scene) 단위로 학습용/검증용을 나눈 뒤 data.yaml까지 만든다.

사용법:
    python aihub_to_yoloseg.py ~/lane/aihub_sample ~/lane/aihub_seg_sample --step 1
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter

import cv2
import numpy as np

# AI 허브 클래스 → YOLO 클래스 번호. 여기에 없는 클래스는 배경으로 취급한다.
# 황색 차선을 추가하려면 "yellowLane": 4 한 줄을 넣고 NAMES에 이름을 추가하면 된다.
CLASS_MAP = {
    "whiteLane": 0,   # 흰색 차선 (점선 조각, 실선 모두 포함)
    "stopLane": 1,    # 정지선
    "crossWalk": 2,   # 횡단보도 줄무늬
    "roadMark": 3,    # 화살표 등 노면표시
}
NAMES = ["white_lane", "stop_line", "crosswalk", "road_mark"]
MIN_AREA = 40     # 이 넓이(픽셀²)보다 작은 조각은 학습에 도움이 안 되어 제외
VAL_EVERY = 5     # 주행 5건 중 1건을 검증용으로 사용


def scene_of(label_path):
    """라벨 경로에서 주행 이름을 뽑는다. (.../2D/<주행>/sensor_raw_data/camera/x.json)"""
    parts = label_path.split(os.sep)
    return parts[parts.index("2D") + 1]


def convert(label_path):
    """라벨 JSON 하나를 읽어 (이미지 파일명, YOLO 라벨 줄 목록, 클래스 번호 목록)을 돌려준다."""
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["information"]["resolution"]
    lines, classes = [], []
    for ann in data.get("annotations", []):
        cls = CLASS_MAP.get(ann.get("class", ""))
        if cls is None:
            continue
        pts = np.array(ann.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        # 화면 밖으로 나간 좌표를 화면 안으로 자름
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if cv2.contourArea(pts) < MIN_AREA:
            continue
        # YOLO 형식: 클래스 번호 뒤에 0~1로 정규화한 x y 좌표를 나열
        norm = (pts / np.array([w, h], dtype=np.float32)).reshape(-1)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
        classes.append(cls)
    return data["information"]["filename"], lines, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="AI 허브 데이터 최상위 폴더")
    ap.add_argument("out_dir", help="YOLO 데이터셋을 만들 폴더")
    ap.add_argument("--step", type=int, default=4,
                    help="주행마다 몇 장에 한 장씩 쓸지 (연속 프레임 중복 줄이기)")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    # 이미지 파일명 → 경로 사전
    images = {os.path.basename(p): p
              for p in glob.glob(os.path.join(args.root, "**", "*.jpg"), recursive=True)}
    # 2D 라벨만 모아서 주행별로 묶음
    scenes = {}
    for lp in sorted(glob.glob(os.path.join(args.root, "**", "*.json"), recursive=True)):
        if os.sep + "2D" + os.sep in lp:
            scenes.setdefault(scene_of(lp), []).append(lp)
    print(f"주행 수: {len(scenes)}, 2D 라벨 수: {sum(len(v) for v in scenes.values())}")

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    n_img = Counter()                         # 분할별 이미지 수
    n_obj = {"train": Counter(), "val": Counter()}  # 분할별 클래스별 조각 수
    missing = 0
    for i, scene in enumerate(sorted(scenes)):
        # 같은 주행의 프레임이 학습/검증에 섞이지 않도록 주행 단위로 나눔
        split = "val" if i % VAL_EVERY == 0 else "train"
        for lp in scenes[scene][::args.step]:
            name, lines, classes = convert(lp)
            if name not in images:
                missing += 1
                continue
            stem = os.path.splitext(name)[0]
            shutil.copy(images[name], os.path.join(out_dir, "images", split, name))
            with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_img[split] += 1
            n_obj[split].update(classes)

    # 학습 설정 파일
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")

    for split in ("train", "val"):
        print(f"\n[{split}] 이미지 {n_img[split]}장")
        for i, n in enumerate(NAMES):
            print(f"  {n:12s} {n_obj[split][i]}")
    if missing:
        print(f"\n이미지를 찾지 못한 라벨: {missing}개")
    print(f"\n완료: {out_dir}/data.yaml")


if __name__ == "__main__":
    main()
