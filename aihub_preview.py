"""AI 허브 '승용 자율주행차 주간 도심도로' 2D 분할 라벨 확인용 스크립트.

차선 관련 클래스의 폴리곤을 원본 이미지 위에 그려서
점선이 조각별로 따로 라벨링되어 있는지 눈으로 확인한다.

사용법:
    python aihub_preview.py ~/lane/aihub_sample ~/lane/aihub_preview --step 10
"""
import argparse
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np

# 확인할 클래스와 표시 색상 (BGR)
COLORS = {
    "whiteLane": (0, 255, 0),      # 초록
    "yellowLane": (0, 255, 255),   # 노랑
    "blueLane": (255, 128, 0),     # 파랑
    "stopLane": (0, 0, 255),       # 빨강
    "crossWalk": (255, 0, 255),    # 자홍
    "roadMark": (255, 255, 0),     # 청록
}
ALPHA = 0.45  # 채움 투명도


def load_polygons(json_path):
    """라벨 JSON에서 (클래스, 꼭짓점 배열) 목록을 읽는다."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for ann in data.get("annotations", []):
        pts = np.array(ann.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:  # 꼭짓점 3개 미만은 폴리곤이 아니므로 제외
            items.append((ann.get("class", ""), pts))
    return data["information"]["filename"], items


def draw(img, items):
    """차선 관련 폴리곤을 반투명 채움 + 외곽선으로 그린다."""
    overlay = img.copy()
    for cls, pts in items:
        if cls in COLORS:
            cv2.fillPoly(overlay, [pts.astype(np.int32)], COLORS[cls])
    out = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)
    for cls, pts in items:
        if cls in COLORS:  # 조각 경계가 보이도록 외곽선을 따로 그림
            cv2.polylines(out, [pts.astype(np.int32)], True, COLORS[cls], 1)
    # 범례
    for i, (cls, color) in enumerate(COLORS.items()):
        cv2.putText(out, cls, (20, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="샘플 데이터 최상위 폴더")
    ap.add_argument("out_dir", help="미리보기 이미지 저장 폴더")
    ap.add_argument("--step", type=int, default=10, help="몇 장마다 한 장 저장할지")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 이미지 파일명 → 경로 사전 (전방 카메라 폴더 위치를 몰라도 찾기 위함)
    images = {os.path.basename(p): p
              for p in glob.glob(os.path.join(args.root, "**", "*.jpg"), recursive=True)}
    labels = sorted(p for p in glob.glob(os.path.join(args.root, "**", "*.json"), recursive=True)
                    if os.sep + "2D" + os.sep in p)
    print(f"2D 라벨 파일 수: {len(labels)}")

    total = Counter()   # 클래스별 폴리곤 수
    white_per_img = []  # 이미지별 흰색 차선 폴리곤 수
    saved = 0
    for i, lp in enumerate(labels):
        name, items = load_polygons(lp)
        total.update(cls for cls, _ in items)
        white_per_img.append(sum(1 for cls, _ in items if cls == "whiteLane"))
        if i % args.step != 0:
            continue
        if name not in images:
            print(f"이미지 없음: {name}")
            continue
        img = cv2.imread(images[name])
        out_path = os.path.join(args.out_dir, f"preview_{os.path.splitext(name)[0]}.jpg")
        cv2.imwrite(out_path, draw(img, items))
        saved += 1

    print("\n클래스별 폴리곤 수:")
    for cls, n in total.most_common():
        mark = " *" if cls in COLORS else ""
        print(f"  {cls:20s} {n}{mark}")
    if white_per_img:
        print(f"\n이미지당 whiteLane 폴리곤: 평균 {np.mean(white_per_img):.1f}, "
              f"최소 {min(white_per_img)}, 최대 {max(white_per_img)}")
    print(f"미리보기 {saved}장 저장: {args.out_dir}")


if __name__ == "__main__":
    main()
