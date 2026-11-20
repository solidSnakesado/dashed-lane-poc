"""화면에 보이는 차선 조각으로 도로 규격을 직접 잰다: 차로 폭, 선 굵기, 점선 길이, 점선 간격 (단위: 실제 m).

규격을 사람이 넣지 않기 위한 부분이다. 프레임마다 add()로 조각을 넣으면 측정값이 쌓이고, spec()이 가운데 값을 돌려준다.
도로 크기에 걸린 고정 숫자는 쓰지 않는다. 판단 기준은 모두 비율·각도·픽셀(카메라 해상도)이다.
"""
import cv2
import numpy as np

BORDER_PX = 4      # 화면 가장자리에서 이 픽셀 안에 닿은 끝은 '잘린 끝'으로 봄
SLIM = 3.0         # 길이가 폭의 이 배수 이상이어야 선 조각으로 봄
RES = 12.0         # 조각 길이가 먼 끝의 '세로 1픽셀이 차지하는 바닥 거리'의 이 배수 이상이어야 길이를 믿음
TURN = 30.0        # 같은 선·나란한 선으로 볼 방향 차 한계(도)
FAR = 0.85         # 지금까지 본 가장 먼 거리의 이 비율보다 먼 끝은 '멀어서 잘린 끝'일 수 있어 길이 측정에서 뺌
GRAZE = 10.0      # 카메라 높이의 이 배수보다 먼 바닥은 너무 비스듬히 보여서 재지 않음 (바닥과 이루는 각 약 6도 미만)
THICK_PX = 3.0     # 화면에서 굵기가 이 픽셀보다 가는 조각은 재지 않음
MIN_N = 15         # 측정값이 이 개수 이상 모여야 그 항목을 내놓음


def clip_far(g, zlim):
    """바닥 다각형 g에서 앞 zlim보다 먼 부분을 잘라 낸다. (잘라 낸 다각형, 잘렸는지)"""
    out = []
    for p, q in zip(g, np.roll(g, -1, 0)):
        if p[1] <= zlim:
            out.append(p)
        if (p[1] <= zlim) != (q[1] <= zlim):
            out.append(p + (q - p) * (zlim - p[1]) / (q[1] - p[1]))
    return np.array(out).reshape(-1, 2), bool((g[:, 1] > zlim).any())


class Shape:
    """바닥에 펼친 조각 하나: 중심, 방향(앞쪽을 향한 단위 벡터), 길이, 폭, 양 끝이 온전한지."""
    def __init__(self, g, px, cam, size, clipped=False):
        (cx, cy), (w, h), ang = cv2.minAreaRect(g.astype(np.float32))
        long_w = w >= h
        th = np.radians(ang if long_w else ang + 90.0)
        self.c, self.len = np.array([cx, cy]), max(w, h)
        self.wid = cv2.contourArea(g.astype(np.float32)) / max(self.len, 1e-9)   # 넓이 ÷ 길이: 휜 조각도 굵기가 맞게 나옴
        n = np.maximum(np.ceil(np.linalg.norm(np.roll(g, -1, 0) - g, axis=1) / max(self.wid, 1e-6)).astype(int), 1)
        self.edge = np.vstack([np.linspace(p, q, k, endpoint=False) for p, q, k in zip(g, np.roll(g, -1, 0), n)])   # 테두리 점
        self.d = np.array([np.cos(th), np.sin(th)])
        self.d = self.d if self.d[1] >= 0 else -self.d
        ends = self.ends = np.array([self.c - self.d * self.len / 2, self.c + self.d * self.len / 2])   # 가까운 끝, 먼 끝
        e = cam.to_image(ends)
        inside = lambda p: BORDER_PX <= p[0] < size[0] - BORDER_PX and BORDER_PX <= p[1] < size[1] - BORDER_PX
        cut = (px[:, 0].min() < BORDER_PX or px[:, 0].max() >= size[0] - BORDER_PX or px[:, 1].max() >= size[1] - BORDER_PX)
        self.ok_near, self.ok_far = inside(e[0]) and not cut, inside(e[1]) and not clipped
        self.thick = cv2.contourArea(px.astype(np.float32)) / max(np.linalg.norm(e[1] - e[0]), 1e-9)   # 화면에서의 굵기(픽셀)
        self.far_z = ends[1, 1]
        up, _ = cam.to_ground(np.array([e[1], e[1] - [0, 1]]))           # 먼 끝에서 세로 1픽셀이 차지하는 바닥 거리
        self.res = float(np.linalg.norm(up[1] - up[0]))

    def beside(self, o):
        """나란한 다른 선 o까지의 거리(선 중심 사이). o가 이 조각의 옆(양 끝 사이)에 있지 않으면 None."""
        d = np.linalg.norm(o.edge[:, None] - self.edge[None], axis=2)
        near = self.edge[d.argmin(axis=1)]                              # o의 테두리 점마다 가장 가까운 이쪽 테두리 점
        mid = np.minimum(*(np.linalg.norm(near - e, axis=1) for e in self.ends)) > 2 * max(self.wid, o.wid)
        return float(np.median(d.min(axis=1)[mid]) + self.wid / 2) if mid.sum() >= 3 else None

    def rel(self, o):
        """다른 조각 o의 중심이 이 조각의 축에서 (앞으로, 옆으로) 얼마나 떨어져 있는지."""
        v = o.c - self.c
        return float(v @ self.d), float(v @ np.array([self.d[1], -self.d[0]]))


class Measure:
    def __init__(self, cam, size, cam_h=None):
        self.cam, self.size, self.zmax = cam, size, 0.0
        self.zlim = GRAZE * cam_h if cam_h else np.inf     # cam_h = 카메라 높이(보정 파일). 없으면 거리 제한 없이 잼
        self.vals = {"line_w": [], "lane_w": [], "dash": [], "gap": []}

    def add(self, polys):
        """한 프레임의 차선 폴리곤(픽셀 좌표 배열들)을 넣는다."""
        near, full = [], []                              # near = 가까운 부분만(폭 측정용), full = 전체(길이 측정용)
        for px in polys:
            g, ok = self.cam.to_ground(px)
            if len(px) < 3:
                continue
            gc, clipped = clip_far(np.where(ok[:, None], g, 1e6), self.zlim)
            for lst, gg, cl in ((near, gc, clipped), (full, g, False)):
                if len(gg) < 3 or (lst is full and not ok.all()):
                    continue
                s = Shape(gg, self.cam.to_image(gg), self.cam, self.size, cl)
                if s.wid > 0 and s.len >= SLIM * s.wid and s.thick >= (THICK_PX if lst is near else 1.0):
                    lst.append(s)
        if not near:
            return
        self.zmax = max([self.zmax] + [s.far_z for s in full])
        wid = float(np.median([s.wid for s in near]))
        self.vals["line_w"] += [s.wid for s in near]
        cos = np.cos(np.radians(TURN))
        for a in near:                                   # 차로 폭: 옆에 나란히 있는 다른 선까지의 거리
            for b in near:
                if b is not a and abs(a.d @ b.d) >= cos and a.len >= b.len and abs(a.rel(b)[1]) > 3 * wid:
                    side = a.beside(b)
                    if side is not None and side > 3 * wid:
                        self.vals["lane_w"].append(side)
        chained, shapes = set(), full
        for a in full:                                   # 점선 간격: 같은 선 위에서 바로 앞에 있는 조각까지의 빈 거리
            nxt = None
            for b in full:
                if b is a or abs(a.d @ b.d) < cos:
                    continue
                along, side = a.rel(b)
                gap = along - a.len / 2 - b.len / 2
                if gap > 0 and abs(side) < 1.5 * wid + 0.2 * gap and (nxt is None or gap < nxt[0]):
                    nxt = (gap, b)
            if nxt is not None:
                chained |= {id(a), id(nxt[1])}
                if a.ok_far and nxt[1].ok_near and nxt[0] >= RES * a.res:
                    self.vals["gap"].append(nxt[0])
        for s in shapes:                                 # 점선 길이: 이어진 조각이고, 양 끝이 온전하고, 충분히 또렷한 것만
            if id(s) in chained and s.ok_near and s.ok_far and s.far_z < FAR * self.zmax and s.len >= RES * s.res:
                self.vals["dash"].append(s.len)

    def spec(self):
        """지금까지의 측정값(m). 아직 못 잰 항목은 빠진다. 차로 폭은 여러 차로 건너 잰 값이 섞이므로 가장 작은 무리를 씀."""
        out = {k: float(np.median(v)) for k, v in self.vals.items() if k != "lane_w" and len(v) >= MIN_N}
        v = np.sort(self.vals["lane_w"])
        if len(v) >= MIN_N:
            w = np.percentile(v, 10)
            for _ in range(3):                           # 가장 작은 무리의 가운데 값으로 좁혀 감
                w = float(np.median(v[(v > 0.8 * w) & (v < 1.25 * w)]))
            out["lane_w"] = w
        return out
