"""labelme 차선 라벨(reviewed 체크된 프레임)을 Ultralytics YOLO 세그멘테이션 학습 형식으로 변환하는 스크립트.

사용법:
    python labelme_to_yoloseg.py ~/lane/dataset/images ~/lane/seg_dataset --max_frame 420

변환 방식:
    - 차선 꺾은선 하나를 두께 THICK_RATIO(이미지 너비 대비)로 그린 띠 모양 다각형으로 바꾼다.
    - 클래스는 슬롯 번호 그대로 사용한다 (0: lane0, 1: lane1, 2: lane2, 3: lane3).
    - 점선의 빈 구간과 가려진 구간도 라벨처럼 이어진 띠가 된다 (UFLDv2 학습과 같은 정답).
    - 이미지는 복사하지 않고 심볼릭 링크로 연결한다.
결과:
    <out>/images/train, <out>/labels/train, <out>/images/val, <out>/labels/val, <out>/data.yaml
    val은 학습 프레임 중 VAL_EVERY장마다 1장을 골라 만든다 (학습 확인용, 성능 평가는 별도 영상으로 함).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

SLOTS = ["lane0", "lane1", "lane2", "lane3"]
THICK_RATIO = 16 / 1640  # CULane 학습 마스크 두께(16px @ 1640)와 같은 비율
VAL_EVERY = 10


def polyline_to_polygon(points, w, h, thick):
    """꺾은선을 두께 thick의 띠로 그린 뒤 외곽선을 다각형으로 뽑는다."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.asarray(points)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, thickness=thick)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)  # 화면 밖으로 잘려 둘로 나뉜 경우 큰 쪽만 사용
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def convert(json_path, img_dir, label_path):
    data = json.load(open(json_path))
    w, h = data["imageWidth"], data["imageHeight"]
    thick = max(3, int(round(THICK_RATIO * w)))
    lines = []
    for s in data["shapes"]:
        if s["label"] not in SLOTS or len(s["points"]) < 2:
            continue
        poly = polyline_to_polygon(s["points"], w, h, thick)
        if poly is None:
            continue
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly)
        lines.append(f"{SLOTS.index(s['label'])} {coords}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir"), ap.add_argument("out_dir")
    ap.add_argument("--max_frame", type=int, default=420, help="이 번호 이하 프레임만 사용 (UFLDv2 학습과 같은 범위)")
    args = ap.parse_args()
    img_dir, out = os.path.expanduser(args.img_dir), os.path.expanduser(args.out_dir)

    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    used = []
    for jp in sorted(glob.glob(os.path.join(img_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        if not stem.isdigit() or int(stem) > args.max_frame:
            continue
        data = json.load(open(jp))
        if data.get("flags", {}).get("reviewed"):  # UFLDv2와 같은 프레임만 사용
            used.append((stem, jp, os.path.join(img_dir, data["imagePath"])))

    n_lanes = 0
    for i, (stem, jp, img) in enumerate(used):
        splits = ["train"] + (["val"] if i % VAL_EVERY == 0 else [])
        for split in splits:
            link = os.path.join(out, "images", split, stem + ".jpg")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(img), link)
            n = convert(jp, img_dir, os.path.join(out, "labels", split, stem + ".txt"))
        n_lanes += n

    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n")
        for k, name in enumerate(SLOTS):
            f.write(f"  {k}: {name}\n")
    n_val = len(glob.glob(os.path.join(out, "images", "val", "*.jpg")))
    print(f"완료: 학습 {len(used)}장 (검증 {n_val}장), 차선 {n_lanes}개 -> {out}/data.yaml")


if __name__ == "__main__":
    main()
