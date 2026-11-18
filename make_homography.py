"""직선 도로 프레임의 lane1·lane2 라벨로 화면 → 바닥 평면(BEV) 변환 행렬(호모그래피)을 만드는 스크립트.

사용법:
    python make_homography.py ~/lane/dataset/images/0285.json ~/lane/H_drive.npy --lane_width 3.5 --far 30

원리:
    - 곧은 도로에서 lane1·lane2는 바닥에서 평행한 두 직선이다.
    - 화면 아래(가까운 쪽)와 FAR_Y(먼 쪽)에서 두 선의 위치 4점을 잡고,
      바닥 좌표의 직사각형 (가로 = 차선 폭, 세로 = --far 미터)에 대응시킨다.
주의:
    - 카메라 보정 없이 만든 근사 변환이다. 가로 방향(차선 폭)은 실제 값에 맞지만,
      세로 방향(거리)은 --far로 가정한 값이라 실제 거리와 다를 수 있다.
    - 핑키 실물에서는 바닥에 위치를 아는 기준점을 놓고 촬영해서 이 스크립트 대신 정확한 행렬을 만든다.
결과: 3x3 행렬을 .npy로 저장하고, 확인용 BEV 이미지(<저장이름>_check.jpg)를 함께 저장한다.
"""
import argparse
import json
import os

import cv2
import numpy as np

FAR_Y_RATIO = 0.72  # 먼 쪽 기준 높이 (이미지 높이 대비). 소실점보다 충분히 아래여야 함


def line_x_at(points, y):
    """꺾은선을 직선 x = a*y + b로 근사해서 높이 y에서의 x를 구한다."""
    p = np.asarray(points, dtype=np.float64)
    a, b = np.polyfit(p[:, 1], p[:, 0], 1)
    return a * y + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path"), ap.add_argument("out_npy")
    ap.add_argument("--lane_width", type=float, default=3.5, help="차선 폭 (m). 한국 도시 도로 약 3.0~3.5")
    ap.add_argument("--far", type=float, default=30.0, help="먼 쪽 기준 높이까지의 가정 거리 (m)")
    args = ap.parse_args()

    data = json.load(open(os.path.expanduser(args.json_path)))
    w, h = data["imageWidth"], data["imageHeight"]
    lanes = {s["label"]: s["points"] for s in data["shapes"]}
    if "lane1" not in lanes or "lane2" not in lanes:
        raise SystemExit("lane1과 lane2가 모두 있는 직선 도로 프레임을 지정하세요.")

    y_near, y_far = float(h), FAR_Y_RATIO * h
    src = np.float32([[line_x_at(lanes["lane1"], y_near), y_near], [line_x_at(lanes["lane2"], y_near), y_near],
                      [line_x_at(lanes["lane2"], y_far), y_far], [line_x_at(lanes["lane1"], y_far), y_far]])
    half = args.lane_width / 2
    # 바닥 좌표: x = 오른쪽(+), z = 앞쪽(+), 단위 m. 화면 맨 아래 = z 0
    dst = np.float32([[-half, 0], [half, 0], [half, args.far], [-half, args.far]])
    H = cv2.getPerspectiveTransform(src, dst)
    np.save(os.path.expanduser(args.out_npy), H)

    # 확인용: 바닥 좌표를 BEV 이미지(1m = 20px, 가로 ±10m, 세로 0~far)로 그려서 차선이 평행한지 확인
    ppm, half_w = 20, 10
    to_img = np.array([[ppm, 0, half_w * ppm], [0, -ppm, args.far * ppm], [0, 0, 1]], dtype=np.float64)
    img = cv2.imread(os.path.join(os.path.dirname(os.path.expanduser(args.json_path)), data["imagePath"]))
    bev = cv2.warpPerspective(img, to_img @ H, (2 * half_w * ppm, int(args.far * ppm)))
    check = os.path.splitext(os.path.expanduser(args.out_npy))[0] + "_check.jpg"
    cv2.imwrite(check, bev)
    print(f"저장: {args.out_npy}\n확인 이미지: {check} (차선이 세로로 평행하면 정상)")


if __name__ == "__main__":
    main()
