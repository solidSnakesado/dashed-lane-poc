"""양산 ADAS 구조(세그멘테이션 → 바닥 평면 변환 → 3차 다항식 → 칼만 필터)로 차선을 인식해 결과 영상을 저장하는 스크립트.

사용법:
    python seg_adas.py <YOLO-seg 가중치.pt> ~/lane/H_drive.npy 입력.mp4 출력.mp4

처리 순서 (프레임마다):
    1. YOLO 세그멘테이션으로 슬롯별(lane0~lane3) 차선 마스크를 얻는다.
    2. 마스크의 행마다 가운데 x를 구해 차선 중심점을 만든다.
    3. 호모그래피로 중심점을 바닥 평면 좌표(x: 옆, z: 앞, 단위 m)로 바꾼다.
    4. RANSAC으로 3차 다항식 x = C0 + C1·z + C2·z² + C3·z³을 피팅한다.
       C0: 차량 기준 옆 거리, C1: 진행 방향 차이, C2·C3: 곡률과 곡률 변화 (양산 ADAS 차선 모델과 같은 형식)
    5. 슬롯 순서(C0 기준 lane0 < lane1 < lane2 < lane3) 검사 후 칼만 필터로 C0~C3을 추적한다.
    6. 결과를 화면 좌표로 되돌려 그리고, lane1·lane2의 중간선(차로 중심 경로)을 흰색으로 그린다.
출력: 결과 영상 + 프레임별 lane1·lane2 계수(<출력이름>.csv)
"""
import sys

import cv2
import numpy as np
from ultralytics import YOLO

COLORS = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 220, 220), 3: (255, 0, 0)}  # BGR
SCALE = 0.25          # 마스크 처리용 축소 비율 (속도용)
Z_MAX = 40.0          # 사용할 최대 앞쪽 거리 (m)
MIN_PTS = 6           # 피팅에 필요한 최소 점 개수
TOL = 0.25            # RANSAC 허용 오차 (m)
HOLD = 5              # 검출이 끊겨도 추적을 유지할 최대 프레임 수
GATE = 1.0            # 직전 추적값과 C0가 이보다 크게 다르면 이상값으로 봄 (m)
Q = np.diag([0.02, 0.002, 1e-4, 1e-6])  # 프레임 사이 계수 변화량 (클수록 빠르게 따라감)


def centerline_points(mask):
    """마스크의 행마다 차선 픽셀의 가운데 x를 구한다. 반환: (N,2) 원본 화면 좌표."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.empty((0, 2))
    rows = np.unique(ys)
    cx = np.array([xs[ys == r].mean() for r in rows])
    return np.stack([cx, rows], 1) / SCALE


def fit_cubic(pts, rng=np.random.default_rng(0), iters=60):
    """바닥 좌표 점 (x, z)에 x = f(z) 3차 다항식을 RANSAC으로 피팅한다. 반환: (계수[C0..C3], 정상점 개수, z범위)"""
    if len(pts) < MIN_PTS:
        return None
    x, z = pts[:, 0], pts[:, 1]
    best = None
    for _ in range(iters):
        idx = rng.choice(len(pts), 4, replace=False)
        if np.ptp(z[idx]) < 1.0:
            continue
        c = np.polyfit(z[idx], x[idx], 3)
        inl = np.abs(np.polyval(c, z) - x) < TOL
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_PTS:
        return None
    c = np.polyfit(z[best], x[best], 3)[::-1]  # 낮은 차수부터: C0, C1, C2, C3
    return c, int(best.sum()), (z[best].min(), z[best].max())


class LaneKalman:
    """차선 하나의 계수 C0~C3을 추적하는 칼만 필터 (상태 = 계수, 계수는 천천히 변한다고 가정)."""

    def __init__(self, c, rng):
        self.x, self.P, self.miss, self.zr = c.copy(), np.diag([0.5, 0.05, 1e-3, 1e-5]), 0, rng

    def predict(self):
        self.P = self.P + Q

    def update(self, c, n, rng):
        R = np.diag([0.3, 0.03, 1e-3, 1e-5]) * (20.0 / max(n, 1))  # 점이 많을수록 측정을 더 믿음
        K = self.P @ np.linalg.inv(self.P + R)
        self.x = self.x + K @ (c - self.x)
        self.P = (np.eye(4) - K) @ self.P
        self.miss, self.zr = 0, rng


def order_check(fits):
    """C0(차량 옆 거리) 기준으로 슬롯 순서가 어긋난 차선을 버린다 (점이 적은 쪽을 버림)."""
    changed = True
    while changed:
        changed = False
        s = sorted(fits)
        for a, b in zip(s, s[1:]):
            if fits[a][0][0] >= fits[b][0][0]:
                del fits[a if fits[a][1] < fits[b][1] else b]
                changed = True
                break
    return fits


def to_image(c, zr, Hinv):
    """계수 c의 곡선을 z범위 zr에서 뽑아 화면 좌표로 바꾼다."""
    z = np.linspace(zr[0], zr[1], 30)
    x = c[0] + c[1] * z + c[2] * z ** 2 + c[3] * z ** 3
    pts = cv2.perspectiveTransform(np.stack([x, z], 1).reshape(-1, 1, 2).astype(np.float64), Hinv)
    return pts.reshape(-1, 2).astype(np.int32)


def main():
    weight, h_path, video_in, video_out = sys.argv[1:5]
    model, H = YOLO(weight), np.load(h_path)
    Hinv = np.linalg.inv(H)
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv = open(video_out.rsplit(".", 1)[0] + ".csv", "w")
    csv.write("frame,lane1_C0,lane1_C1,lane1_C2,lane1_C3,lane2_C0,lane2_C1,lane2_C2,lane2_C3\n")
    tracks, n = {}, 0
    sw, sh = int(w * SCALE), int(h * SCALE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, conf=0.25, verbose=False)[0]
        masks = {k: np.zeros((sh, sw), np.uint8) for k in range(4)}
        if r.masks is not None:
            for poly, cls in zip(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int)):
                if len(poly) >= 3 and cls in masks:
                    cv2.fillPoly(masks[cls], [(poly * SCALE).astype(np.int32)], 1)

        fits = {}
        for k, m in masks.items():
            pts = centerline_points(m)
            if len(pts) < MIN_PTS:
                continue
            g = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
            g = g[(g[:, 1] >= 0) & (g[:, 1] <= Z_MAX)]
            f = fit_cubic(g)
            if f is not None:
                fits[k] = f
        fits = order_check(fits)

        for k in range(4):
            t, f = tracks.get(k), fits.get(k)
            if t is not None:
                t.predict()
                if f is not None and abs(f[0][0] - t.x[0]) > GATE and t.miss < HOLD:
                    f = None  # 갑자기 튄 측정은 버림
            if f is not None:
                if t is None:
                    tracks[k] = LaneKalman(f[0], f[2])
                else:
                    t.update(f[0], f[1], f[2])
            elif t is not None:
                t.miss += 1
                if t.miss > HOLD:
                    del tracks[k]

        for k, t in tracks.items():
            cv2.polylines(frame, [to_image(t.x, t.zr, Hinv).reshape(-1, 1, 2)], False, COLORS[k], 5)
        row = [n]
        if 1 in tracks and 2 in tracks:
            zr = (max(tracks[1].zr[0], tracks[2].zr[0]), min(tracks[1].zr[1], tracks[2].zr[1]))
            if zr[1] > zr[0]:
                center = (tracks[1].x + tracks[2].x) / 2  # 차로 중심 경로
                cv2.polylines(frame, [to_image(center, zr, Hinv).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
        for k in (1, 2):
            row += list(tracks[k].x) if k in tracks else [""] * 4
        csv.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")
        for k, col in COLORS.items():
            cv2.putText(frame, f"lane{k}", (20, 50 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        out.write(frame)
        n += 1
        if n % 300 == 0:
            print(f"{n} frames")

    cap.release(), out.release(), csv.close()
    print(f"done: {n} frames -> {video_out}")


if __name__ == "__main__":
    main()
