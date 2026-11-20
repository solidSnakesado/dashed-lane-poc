"""차선 조각을 바닥 좌표(m)로 펼치고 조각별 특징을 계산한다 (3단계: 도로 펼치기).

바닥 좌표: x = 차량 오른쪽(+), z = 차량 앞쪽(+), 단위 m.
"""
from dataclasses import dataclass

import cv2
import numpy as np

Z_MAX = 60.0      # 이 거리(m)보다 먼 점은 변환 오차가 커서 버림
X_MAX = 20.0      # 좌우로 이 거리(m)보다 먼 점도 버림
STEP_PX = 4.0     # 폴리곤 테두리를 이 간격(픽셀)으로 촘촘히 찍음
BIN_M = 1.0       # 중심선 점 간격(m)
MAX_WIDTH = 1.0   # 이보다 굵으면 차선이 아닌 것으로 봄(m). 모델 마스크는 실제 선보다 굵게 나옴
MIN_RATIO = 1.3   # 길이가 폭의 이 배수보다 작으면(뭉툭한 덩어리) 차선이 아닌 것으로 봄
MIN_LEN = 0.4     # 이보다 짧은 조각은 버림(m)
MARK_OVERLAP = 0.5    # 꼭짓점의 이 비율 이상이 바닥 표시 안에 있으면 차선에서 제외
SHORT_LEN = 1.2   # 이보다 짧은 조각(m)은 방향을 믿지 않음
DIR_SPAN = 3.0    # 끝 방향을 구할 때 보는 구간 길이(m)


class Camera:
    """픽셀 ↔ 바닥 좌표 변환. npz(K, dist, H) 또는 npy(H만) 파일을 읽는다."""

    def __init__(self, path):
        if path.endswith(".npz"):
            d = np.load(path)
            self.K, self.dist, self.H = d["K"], d["dist"], d["H"]
        else:
            self.K, self.dist, self.H = None, None, np.load(path)
        # 화면 아래쪽(바닥)의 점에서 나눗셈 값이 양수가 되도록 행렬 부호를 맞춤
        if (self.H @ np.array([1000.0, 1e5, 1.0]))[2] < 0:
            self.H = -self.H
        self.Hinv = np.linalg.inv(self.H)

    def to_ground(self, px):
        """픽셀 좌표 (N,2) → 바닥 좌표 (N,2)와 지평선 아래 여부 (N,)."""
        px = np.asarray(px, np.float64).reshape(-1, 2)
        if self.K is not None:      # 렌즈 왜곡을 먼저 편다
            px = cv2.undistortPoints(px.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        g = np.c_[px, np.ones(len(px))] @ self.H.T
        ok = g[:, 2] > 1e-6
        return g[:, :2] / np.where(ok, g[:, 2], 1)[:, None], ok

    def to_image(self, g):
        """바닥 좌표 (N,2) → 픽셀 좌표 (N,2). 결과 그림에 곡선을 그릴 때 쓴다."""
        g = np.asarray(g, np.float64).reshape(-1, 2)
        u = np.c_[g, np.ones(len(g))] @ self.Hinv.T
        u = u[:, :2] / u[:, 2:3]
        if self.K is None:
            return u
        n = np.c_[u, np.ones(len(u))] @ np.linalg.inv(self.K).T   # 다시 렌즈 왜곡을 입힘
        px, _ = cv2.projectPoints(n, np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.reshape(-1, 2)


@dataclass
class Piece:
    """바닥에 펼친 차선 조각 하나."""
    poly: np.ndarray     # 바닥 좌표 윤곽 (N,2)
    line: np.ndarray     # 중심선 점열 (M,2), 차량에 가까운 쪽이 먼저
    length: float        # 길이(m)
    width: float         # 폭(m)
    trust: bool = True   # 방향을 믿을 수 있는지 (너무 짧은 조각은 False)
    parts: list = None   # 여러 조각을 이어 만든 경우, 원래 조각들(순서대로)

    @property
    def start(self):
        return self.line[0]

    @property
    def end(self):
        return self.line[-1]

    def direction(self, at_end=True):
        """끝(또는 시작) 부분의 진행 방향 단위 벡터. 끝에서 DIR_SPAN 안의 점들로 구한다."""
        line = self.line if at_end else self.line[::-1]
        near = line[np.linalg.norm(line - line[-1], axis=1) <= DIR_SPAN]
        d = (near[-1] - near[0]) if len(near) >= 2 else (line[-1] - line[-2])
        d = d / max(np.linalg.norm(d), 1e-9)
        return d if at_end else -d

    def reversed(self):
        """중심선 방향을 뒤집은 조각."""
        parts = [p.reversed() for p in self.parts[::-1]] if self.parts else None
        return Piece(self.poly, self.line[::-1].copy(), self.length, self.width, self.trust, parts)


def densify(pts, step=STEP_PX):
    """닫힌 폴리곤의 테두리를 일정 간격으로 촘촘히 찍는다."""
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def ground_poly(px, cam):
    """폴리곤을 바닥으로 펼친다. 믿을 수 있는 범위의 점이 너무 적으면 None."""
    g, ok = cam.to_ground(densify(np.asarray(px, np.float64).reshape(-1, 2)))
    keep = ok & (g[:, 1] > 0) & (g[:, 1] < Z_MAX) & (np.abs(g[:, 0]) < X_MAX)
    return g[keep] if keep.sum() >= 4 else None


def make_piece(px, cam):
    """픽셀 폴리곤 하나를 차선 조각으로 만든다. 만들 수 없으면 None."""
    g = ground_poly(px, cam)
    if g is None:
        return None
    c = g.mean(axis=0)
    _, _, vt = np.linalg.svd(g - c)
    v, n = vt[0], vt[1]                       # 길이 방향, 폭 방향
    t, s = (g - c) @ v, (g - c) @ n
    length = float(np.ptp(t))
    # 길이 방향으로 구간을 나눠, 구간마다 양쪽 테두리의 가운데를 중심선 점으로 삼음
    nb = max(int(round(length / BIN_M)), 1)
    edges = np.linspace(t.min(), t.max(), nb + 1)
    mids, widths = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t <= hi)
        if m.sum() >= 2:
            mids.append(((lo + hi) / 2, (s[m].max() + s[m].min()) / 2))
            widths.append(np.ptp(s[m]))
    if not mids:
        return None
    # 양 끝점을 더해 중심선이 조각 전체 길이를 덮게 함
    mids = [(t.min(), mids[0][1])] + mids + [(t.max(), mids[-1][1])]
    line = np.array([c + a * v + b * n for a, b in mids])
    if np.linalg.norm(line[0]) > np.linalg.norm(line[-1]):
        line = line[::-1].copy()              # 차량에 가까운 쪽이 먼저 오게
    return Piece(g, line, length, float(np.median(widths)), length >= SHORT_LEN)


def frame_to_pieces(polys, cam):
    """한 프레임의 (클래스, 픽셀 폴리곤) 목록을 차선 조각과 그 밖의 표시로 나눈다.

    돌려주는 값: (차선 조각 목록, [(클래스, 바닥 윤곽)] 목록)
    """
    pieces, others = [], []
    for cls, px in polys:
        if cls == "white_lane":
            p = make_piece(px, cam)
            if p is None:
                continue
            if p.length >= MIN_LEN and p.width <= MAX_WIDTH and p.length >= MIN_RATIO * p.width:
                pieces.append(p)
            else:
                others.append(("rejected", p.poly))   # 너무 짧거나 굵어 제외한 조각
        else:
            g = ground_poly(px, cam)
            if g is not None:
                others.append((cls, g))
    return pieces, others


def drop_marks(polys):
    """차선으로 검출된 폴리곤 가운데 바닥 표시(화살표·글자), 횡단보도, 정지선 검출과 겹치는 것을 뺀다.

    같은 자리를 모델이 차선과 바닥 표시로 함께 검출한 경우다. 차선 폴리곤만 돌려준다.
    """
    marks = [px.astype(np.float32) for cls, px in polys if cls != "white_lane"]
    out = []
    for cls, px in polys:
        if cls != "white_lane":
            continue
        inside = max((np.mean([cv2.pointPolygonTest(m, (float(x), float(y)), False) >= 0 for x, y in px])
                      for m in marks), default=0.0)
        if inside < MARK_OVERLAP:
            out.append((cls, px))
    return out
