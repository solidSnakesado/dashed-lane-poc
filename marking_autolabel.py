"""영상 처리로 흰색 차선 표시(칠해진 부분만)를 자동 추출해 labelme 다각형 라벨로 저장하는 스크립트.
UFLDv2 등 딥러닝 모델은 쓰지 않는다 (양산 ADAS 방식 비교용 독립 라벨).

사용법:
    python marking_autolabel.py ~/lane/dataset/images ~/lane/marking --max_frame 420
    python marking_autolabel.py ~/lane/dataset/images ~/lane/marking --preview 0285   # 한 장만 확인 (미리보기 이미지 저장)

결과 폴더(<out>)에 이미지 심볼릭 링크와 같은 이름의 JSON을 만든다 (기존 UFLDv2 라벨과 섞이지 않게 분리).
labelme 실행: labelme ~/lane/marking --labels white_dashed,white_solid --validate-label exact --flags reviewed

처리 순서:
    1. 주변 도로보다 밝고(top-hat) 채도가 낮은(흰색) 픽셀을 뽑는다. 그늘·밝기 변화에 강하도록 주변 대비 기준을 쓴다.
    2. 지평선 위(건물·하늘)는 버린다.
    3. 픽셀 덩어리마다 모양을 검사해 차선 조각만 남긴다.
       - 가늘고 길 것, 사각형을 잘 채울 것 (차량·글자 제거)
       - 두께가 원근에 맞는 차선 두께일 것 (횡단보도처럼 두꺼운 표시 제거)
       - 방향이 소실점을 향할 것 (정지선·가로 표시 제거)
       - 바로 주변 도로보다 충분히 밝고 채도가 낮을 것 (가드레일·연석 반사, 파란선 제거)
    4. 원근을 고려한 길이로 점선(white_dashed)과 실선(white_solid)을 나눈다.
주의: 소실점 방향 검사 때문에 급한 곡선·회전 구간은 조각이 빠질 수 있다 → 검수 때 직접 추가.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

HORIZON = 0.67        # 지평선 높이 (이미지 높이 대비). drive.mp4 기준
SKIP_NEAR_H = 0.03    # 지평선 바로 아래(너무 먼 곳)는 버림 (이미지 높이 대비)
TOPHAT_K = 51         # 주변 대비 계산 창 크기 (px)
TOPHAT_T = 35         # 주변보다 이만큼 이상 밝아야 함
S_MAX, V_MIN = 60, 120  # 흰색 조건: 채도 상한, 밝기 하한
MIN_AREA = 50         # 최소 픽셀 수
MIN_ELONG = 2.3       # 길이 / 두께 최소값
MIN_FILL = 0.45       # 덩어리 면적 / 외접 사각형 면적 최소값
THICK_RATIO = (0.028, 0.16)  # 두께 / (지평선까지 높이) 허용 범위
ANGLE_TOL = 14.0      # 소실점 방향과의 각도 차이 허용값 (도)
SOLID_RATIO = 1.6     # 길이 / (지평선까지 높이) 가 이 이상이면 실선
PIECE_CONTRAST = 35   # 조각 밝기 중앙값 - 바로 주변 밝기 중앙값 최소값 (가드레일·연석 반사 제거)
PIECE_S_MAX = 35      # 조각 채도 중앙값 상한 (파란선·황색 반사 제거)


def extract(img):
    """흰색 차선 조각 목록을 반환한다: [(라벨, 원본 좌표 다각형), ...]"""
    h, w = img.shape[:2]
    s = img  # 먼 쪽의 작은 조각을 놓치지 않도록 원본 해상도에서 처리
    sh = s.shape[0]
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K)))
    m = ((th > TOPHAT_T) & (hsv[..., 1] < S_MAX) & (hsv[..., 2] > V_MIN)).astype(np.uint8)
    yh = HORIZON * sh
    m[:int(yh + SKIP_NEAR_H * sh)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    vp = np.array([s.shape[1] / 2, yh])  # 소실점 근사: 화면 가운데, 지평선 높이

    out = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if st[i, 4] < MIN_AREA:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (a, b), _ = cv2.minAreaRect(pts)
        length, thick = max(a, b), max(1.0, min(a, b))
        depth = max(cy - yh, 1.0)  # 지평선에서 떨어진 높이 = 가까울수록 큼
        if length / thick < MIN_ELONG or st[i, 4] / (a * b + 1e-6) < MIN_FILL:
            continue
        if not THICK_RATIO[0] <= thick / depth <= THICK_RATIO[1]:
            continue
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        to_vp = vp - np.array([cx, cy])
        cosang = abs(vx * to_vp[0] + vy * to_vp[1]) / (np.linalg.norm(to_vp) + 1e-6)
        if np.degrees(np.arccos(np.clip(cosang, 0, 1))) > ANGLE_TOL:
            continue
        # 칠해진 차선은 주변 아스팔트보다 확실히 밝음 (가드레일·연석·금속 반사는 대비가 약함)
        piece = (lab == i).astype(np.uint8)
        ring = cv2.dilate(piece, np.ones((15, 15), np.uint8)) - piece
        if np.median(hsv[..., 2][piece > 0]) - np.median(hsv[..., 2][ring > 0]) < PIECE_CONTRAST:
            continue
        if np.median(hsv[..., 1][piece > 0]) > PIECE_S_MAX:
            continue
        label = "white_solid" if length / depth >= SOLID_RATIO else "white_dashed"
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.0, True).reshape(-1, 2).astype(np.float64)
        if len(poly) >= 3:
            out.append((label, poly))
    return out


def save_json(path, img_name, w, h, pieces):
    shapes = [{"label": lab, "points": poly.round(1).tolist(), "group_id": None, "description": "",
               "shape_type": "polygon", "flags": {}} for lab, poly in pieces]
    json.dump({"version": "5.5.0", "flags": {"reviewed": False}, "shapes": shapes, "imagePath": img_name,
               "imageData": None, "imageHeight": h, "imageWidth": w}, open(path, "w"), indent=2)


def draw_preview(img, pieces, path):
    v = img.copy()
    for lab, poly in pieces:
        col = (0, 0, 255) if lab == "white_dashed" else (255, 0, 0)
        cv2.fillPoly(v, [poly.astype(np.int32)], col)
    cv2.imwrite(path, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir"), ap.add_argument("out_dir")
    ap.add_argument("--max_frame", type=int, default=420)
    ap.add_argument("--preview", default=None, help="이 프레임 하나만 처리하고 미리보기 저장 (예: 0285)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 JSON도 다시 만듦 (검수한 라벨이 지워지니 주의)")
    args = ap.parse_args()
    img_dir, out = os.path.expanduser(args.img_dir), os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if args.preview:
        paths = [os.path.join(img_dir, args.preview + ".jpg")]
    made = skipped = 0
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit() and int(stem) > args.max_frame and not args.preview:
            continue
        js = os.path.join(out, stem + ".json")
        if os.path.exists(js) and not args.overwrite and not args.preview:
            skipped += 1
            continue
        img = cv2.imread(p)
        pieces = extract(img)
        if args.preview:
            draw_preview(img, pieces, os.path.join(out, f"preview_{stem}.jpg"))
            print(f"{stem}: 조각 {len(pieces)}개 (빨강 점선, 파랑 실선) -> {out}/preview_{stem}.jpg")
            return
        link = os.path.join(out, os.path.basename(p))
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(p), link)
        save_json(js, os.path.basename(p), img.shape[1], img.shape[0], pieces)
        made += 1
    print(f"완료: {made}개 생성, {skipped}개 건너뜀 (이미 있음) -> {out}")


if __name__ == "__main__":
    main()
