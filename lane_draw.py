"""차선 후처리 결과를 원본 화면과 바닥으로 펼친 그림(BEV)에 그린다."""
import cv2
import numpy as np

SCALE, BEV_W, BEV_H = 20, 600, 1000     # BEV 그림: 1m = 20픽셀, 좌우 ±15m, 앞 50m
# 차선이 아닌 표시의 색 (BGR)
COLORS = {"stop_line": (0, 0, 255), "crosswalk": (0, 200, 255), "road_mark": (255, 255, 0),
          "rejected": (90, 90, 90)}
# 추적 번호별 색. 위의 표시 색(빨강, 노랑, 하늘)과 겹치지 않게 고름
PALETTE = [(255, 140, 60), (255, 80, 255), (180, 130, 255), (255, 200, 150),
           (200, 80, 160), (80, 160, 160), (210, 210, 210)]
CENTER = (0, 255, 0)                    # 중심선 색
SEG = (0, 255, 255)                     # 차선 분할 영역을 칠하는 색


def bev_px(g):
    """바닥 좌표(m) → BEV 그림 픽셀."""
    g = np.asarray(g).reshape(-1, 2)
    return np.c_[BEV_W / 2 + g[:, 0] * SCALE, BEV_H - g[:, 1] * SCALE].astype(np.int32)


def hull(g):
    """점 구름의 바깥 윤곽(BEV 픽셀)."""
    return cv2.convexHull(bev_px(g))


def img_px(cam, g):
    """바닥 좌표 → 원본 화면 픽셀. 차량 바로 앞(2m 이내)의 점은 화면 밖이라 뺀다."""
    g = np.asarray(g).reshape(-1, 2)
    return cam.to_image(g[g[:, 1] > 2.0]).astype(np.int32)


def polyline(img, pts, color, thick, dotted=False):
    """꺾은선을 그린다. dotted면 한 칸씩 건너뛰어 점선으로 그린다."""
    if len(pts) < 2:
        return
    if not dotted:
        cv2.polylines(img, [pts], False, color, thick)
        return
    for a, b in list(zip(pts[:-1], pts[1:]))[::2]:
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick)


def draw(img, polys, tracks, left, right, center, others, cam, title):
    """왼쪽에 원본, 오른쪽에 BEV를 붙인 그림을 만든다. 내 차로의 두 선(L1 = 왼쪽, L2 = 오른쪽)과 중심선만 그린다.

    노란 반투명 영역 = 두 선에 속한 차선 분할 영역. 이번 프레임에 안 보인 선과 선 끝에서 앞으로
    이어 그린 부분은 점선으로 그린다. 초록 = 중심선(추정이면 점선). polys, others는 --all-classes일 때만 쓴다.
    """
    seg = img.copy()
    for t in (left, right):                         # 두 선에 속한 조각만 반투명하게 칠함
        for p in (t.det.pieces if t is not None and t.det is not None else []):
            cv2.fillPoly(seg, [cam.to_image(p.poly).astype(np.int32)], SEG)
    for cls, px in polys:
        if cls != "white_lane":
            cv2.polylines(img, [px.astype(np.int32)], True, COLORS[cls], 2)
    cv2.addWeighted(seg, 0.5, img, 0.5, 0, dst=img)
    bev = np.zeros((BEV_H, BEV_W, 3), np.uint8)
    for z in range(10, 50, 10):                     # 10m 간격 눈금
        cv2.line(bev, (0, BEV_H - z * SCALE), (BEV_W, BEV_H - z * SCALE), (60, 60, 60), 1)
        cv2.putText(bev, f"{z}m", (5, BEV_H - z * SCALE - 5), 0, 0.5, (120, 120, 120), 1)
    cv2.line(bev, (BEV_W // 2, 0), (BEV_W // 2, BEV_H), (60, 60, 60), 1)
    for cls, g in others:
        if cls != "rejected":
            cv2.fillPoly(bev, [hull(g)], COLORS[cls])
    for t, role in ((left, "L1"), (right, "L2")):
        if t is None:
            continue
        col = PALETTE[t.id % len(PALETTE)]
        curve, more = t.curve.sample(0.5), t.curve.ahead()
        polyline(bev, bev_px(curve), col, 3, dotted=t.det is None)
        polyline(img, img_px(cam, curve), col, 5, dotted=t.det is None)
        polyline(bev, bev_px(more), col, 2, dotted=True)       # 앞으로 이어 그린 부분
        polyline(img, img_px(cam, more), col, 4, dotted=True)
        for p in (t.det.pieces if t.det is not None else []):
            cv2.fillPoly(bev, [hull(p.poly)], col)
        x, y = bev_px(curve[0])[0]
        cv2.putText(bev, f"{t.id}{'S' if t.kind == 'solid' else 'D'} {role}", (int(x) + 6, int(y) + 4),
                    0, 0.45, col, 1)
    if center is not None:
        curve, guess = center
        for pts, dot in ((curve.sample(0.5), guess), (curve.ahead(), True)):
            polyline(bev, bev_px(pts), CENTER, 2, dotted=dot)
            polyline(img, img_px(cam, pts), CENTER, 3, dotted=dot)
    img = cv2.resize(img, (int(img.shape[1] * BEV_H / img.shape[0]) // 2 * 2, BEV_H))
    cv2.putText(img, title, (10, 30), 0, 0.8, (0, 255, 0), 2)
    return np.hstack([img, bev])
