"""UFLDv2 출력에 후처리(곡선 피팅 + 타당성 검사 + 프레임 간 평활화)를 적용해 결과 영상을 저장하는 스크립트.

사용법 (Ultra-Fast-Lane-Detection-v2 레포 최상위 폴더에서 실행, make_pseudo_labels.py가 같은 폴더에 있어야 함):
    python lane_postprocess.py configs/culane_res18.py <가중치.pth> 입력.mp4 출력.mp4

처리 순서 (슬롯마다):
    1. 행 기준 + 열 기준 출력점을 모두 모은다 (both 모드와 같음).
    2. RANSAC으로 2차 곡선을 피팅하고, 곡선에서 벗어난 점(횡단보도, 차량 등)을 버린다.
       점이 세로로 퍼져 있으면 x=f(y), 가로로 퍼져 있으면 y=f(x)로 피팅한다 (회전 구간 대응).
    3. 화면 아래쪽에서 슬롯 순서(lane0 < lane1 < lane2 < lane3)가 어긋난 선을 버린다.
    4. 직전 결과와 지수 평균으로 섞어 흔들림을 줄이고, 잠깐 검출이 끊기면 직전 결과를 유지한다.
"""
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from make_pseudo_labels import decode  # DALI 대체 처리도 이 파일 안에서 이루어짐
from utils.common import get_model
from utils.config import Config

COLORS = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 220, 220), 3: (255, 0, 0)}  # BGR
N_SAMPLE = 30      # 곡선 하나를 표현하는 점 개수
MIN_INLIER = 8     # 곡선으로 인정할 최소 정상점 개수
TOL_RATIO = 0.015  # RANSAC 허용 오차 (이미지 너비 대비 비율)
ALPHA = 0.4        # 새 결과 반영 비율 (작을수록 부드럽지만 반응이 느림)
HOLD = 5           # 검출이 끊겨도 직전 결과를 유지할 최대 프레임 수
JUMP_RATIO = 0.12  # 직전 결과와 이보다 멀리 떨어지면 이상값으로 봄 (이미지 너비 대비)


def ransac_curve(pts, tol, iters=60, rng=np.random.default_rng(0)):
    """점들에 2차 곡선을 RANSAC으로 피팅한다. 반환: 곡선 위 점 N_SAMPLE개 (가까운 쪽→먼 쪽) 또는 None."""
    if len(pts) < MIN_INLIER:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    # 점이 퍼진 방향에 따라 독립 변수를 정함: 세로로 길면 y, 가로로 길면 x
    vertical = np.ptp(pts[:, 1]) >= np.ptp(pts[:, 0]) * 0.5
    t, v = (pts[:, 1], pts[:, 0]) if vertical else (pts[:, 0], pts[:, 1])
    best = None
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        if np.ptp(t[idx]) < 1e-3:
            continue
        c = np.polyfit(t[idx], v[idx], 2)
        inl = np.abs(np.polyval(c, t) - v) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_INLIER:
        return None
    c = np.polyfit(t[best], v[best], 2)  # 정상점만으로 다시 피팅
    ts = np.linspace(t[best].min(), t[best].max(), N_SAMPLE)
    vs = np.polyval(c, ts)
    curve = np.stack([vs, ts], 1) if vertical else np.stack([ts, vs], 1)
    # 가까운 쪽(화면 아래, y가 큰 쪽)이 먼저 오도록 정렬
    return curve if curve[0, 1] >= curve[-1, 1] else curve[::-1]


def bottom_x(curve):
    """곡선의 가장 가까운 쪽(화면 아래쪽) 점의 x좌표."""
    return curve[0, 0]


def order_check(curves):
    """화면 아래쪽 x좌표 기준으로 슬롯 순서가 어긋난 선을 버린다 (점이 적은 쪽을 버림)."""
    changed = True
    while changed:
        changed = False
        slots = sorted(curves)
        for a, b in zip(slots, slots[1:]):
            if bottom_x(curves[a][0]) >= bottom_x(curves[b][0]):
                drop = a if curves[a][1] < curves[b][1] else b
                del curves[drop]
                changed = True
                break
    return {k: v[0] for k, v in curves.items()}


class Smoother:
    """슬롯별로 직전 결과와 섞고, 검출이 잠깐 끊기면 직전 결과를 유지한다."""

    def __init__(self, w):
        self.state, self.miss, self.jump = {}, {}, JUMP_RATIO * w

    def update(self, curves):
        out = {}
        for s in range(4):
            new, old = curves.get(s), self.state.get(s)
            if new is not None and old is not None and abs(bottom_x(new) - bottom_x(old)) > self.jump \
                    and self.miss.get(s, 0) < HOLD:
                new = None  # 직전 결과에서 갑자기 멀리 튄 선은 이상값으로 보고 이번 프레임은 버림
            if new is not None:
                self.state[s] = new if old is None else ALPHA * new + (1 - ALPHA) * old
                self.miss[s] = 0
            elif old is not None:
                self.miss[s] = self.miss.get(s, 0) + 1
                if self.miss[s] > HOLD:
                    del self.state[s]  # 너무 오래 끊기면 선을 지움
                    continue
            if s in self.state:
                out[s] = self.state[s]
        return out


def main():
    cfg_path, weight_path, video_in, video_out = sys.argv[1:5]

    cfg = Config.fromfile(cfg_path)
    cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)  # CULane 설정과 동일
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    net = get_model(cfg)
    sd = torch.load(weight_path, map_location="cpu")["model"]
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False)
    net.eval()
    tf = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    smoother, tol, n = Smoother(w), TOL_RATIO * w, 0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = tf(rgb)[:, -cfg.train_height:, :].unsqueeze(0).cuda()
            raw = decode(net(x), cfg.row_anchor, cfg.col_anchor, w, h, both=True)

            fitted = {}
            for s, pts in raw.items():
                c = ransac_curve(pts, tol)
                if c is not None:
                    fitted[s] = (c, len(pts))
            lanes = smoother.update(order_check(fitted))

            for s, c in lanes.items():
                cv2.polylines(frame, [c.astype(np.int32).reshape(-1, 1, 2)], False, COLORS[s], 5)
            for s, col in COLORS.items():
                cv2.putText(frame, f"lane{s}", (20, 50 + 40 * s), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            out.write(frame)
            n += 1
            if n % 300 == 0:
                print(f"{n} frames")

    cap.release()
    out.release()
    print(f"done: {n} frames -> {video_out}")


if __name__ == "__main__":
    main()
