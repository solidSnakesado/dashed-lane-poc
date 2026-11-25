"""도로 그래프와 길 찾기: 지도(road.json의 경로들)를 점 그래프로 만들고, 출발점 → 목적지의 가장 짧은 길을 찾는다.

점: 경로마다 STEP 간격으로 다시 찍은 점(ego_label.Routes와 같은 점). 점마다 차로 왼쪽·오른쪽 선까지 거리와 교차로 안인지가 붙어 있다.
선(이동):
  같은 경로 안에서 다음 점으로 = 그 사이 길이.
  서로 다른 경로의 점이 거의 같은 자리·같은 방향이면 = 같은 차로를 같이 달리는 것이므로 서로 오갈 수 있게 잇는다.
    값 = 두 점 사이 거리 + 점 간격의 절반(벌칙). 벌칙이 없으면 경로 사이를 지그재그로 오가며 길이를 줄이는 길이 나온다.
  차선 변경: 교차로 밖에서 같은 방향으로 나란한 옆 차로(두 차로가 선 하나를 같이 씀)의 점으로 건너간다.
    두 차로 사이 선이 점선인 곳만(바닥 그림에서 그 선 앞뒤 LC_LEN 구간의 칠한 비율이 DASH 사이). 실선·선 없음은 넘지 않는다.
    값 = 옆으로 옮긴 거리 + 차로 폭의 LC_PEN배(벌칙). 벌칙 때문에 필요할 때만 바꾼다.
도로 크기에 걸린 숫자는 쓰지 않는다. 같은 자리 기준은 점 간격의 배수, 차선 변경 기준은 차로 폭의 배수다.
"""
import json
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

SAME = 0.9           # 같은 자리로 보는 거리 = 점 간격의 이 배수
SAME_YAW = 15.0      # 같은 방향으로 보는 방향 차(도)
TINY = 1e-6          # 0 대신 쓰는 아주 작은 값(0은 그래프에서 '선 없음'으로 읽힘)
JUMP = 0.5           # 다른 경로로 건너갈 때 벌칙 = 점 간격의 이 배수
LC_LEN = 2.5         # 차선 변경에 쓰는 길이 = 차로 폭의 이 배수(점선인지 보는 구간도 이 길이)
LC_PEN = 2.0         # 차선 변경 벌칙 = 차로 폭의 이 배수
LC_EVERY = 5         # 차선 변경 선은 점 이 개수마다 하나
DASH = (0.15, 0.85)  # 칠한 비율이 이 사이면 점선
GAP = 1.5            # 점선의 빈칸으로 보고 메우는 길이 = 그림에서 잰 빈칸 길이의 이 배수까지


class Graph:
    def __init__(self, rt, step, road=None):
        """rt = ego_label.Routes, step = 그 점 간격(mm), road = 도로 폴더(주면 차선 변경 선도 만듦)."""
        self.P, self.yaw, self.wl, self.wr, self.zone, self.pid = [], [], [], [], [], []
        rows, cols, w, base = [], [], [], 0
        for j, r in enumerate(rt.r):
            n = len(r["p"])
            self.P.append(r["p"]); self.yaw.append(r["ang"]); self.wl.append(r["wl"]); self.wr.append(r["wr"])
            i = np.arange(n)
            self.zone.append(((i >= r["en"]) & (i <= r["ex"])).astype(np.uint8))      # 1 = 교차로 안
            self.pid.append(np.c_[np.full(n, j), i])
            d = np.linalg.norm(np.diff(r["p"], axis=0), axis=1)
            rows += list(base + i[:-1]); cols += list(base + i[1:]); w += list(np.maximum(d, TINY))
            base += n
        self.P, self.yaw = np.vstack(self.P), np.concatenate(self.yaw)
        self.wl, self.wr, self.zone = np.concatenate(self.wl), np.concatenate(self.wr), np.concatenate(self.zone)
        self.pid = np.vstack(self.pid)
        self.tree = cKDTree(self.P)
        for a, b in self.tree.query_pairs(SAME * step):                                    # 같은 차로를 같이 달리는 점끼리
            if self.pid[a, 0] == self.pid[b, 0]:
                continue
            if abs((self.yaw[a] - self.yaw[b] + np.pi) % (2 * np.pi) - np.pi) < np.radians(SAME_YAW):
                d = float(np.linalg.norm(self.P[a] - self.P[b])) + JUMP * step
                t = np.array([np.cos(self.yaw[a]), np.sin(self.yaw[a])]); fw = (self.P[b] - self.P[a]) @ t
                if fw >= 0:                                                         # 뒤로 건너가는 선은 만들지 않음(지그재그로 거꾸로 가는 길을 막음)
                    rows.append(a); cols.append(b); w.append(d)
                if fw <= 0:
                    rows.append(b); cols.append(a); w.append(d)
        self.lc, self.paint = set(), None
        if road:
            self.paint = self.painted(road)                                        # 점마다 왼쪽·오른쪽 선 자리에 흰 선이 칠해져 있는지
            for a, b, d in self.lane_changes(road, step):
                rows.append(a); cols.append(b); w.append(d); self.lc.add((a, b))
        n = len(self.P)
        self.G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
        self.nxt = np.r_[np.where(self.pid[1:, 0] == self.pid[:-1, 0], np.arange(1, n), -1), -1]   # 같은 경로의 다음 점(없으면 -1)

    def painted(self, road):
        """점마다 (왼쪽 선, 오른쪽 선) 자리에 흰 선이 칠해져 있는지 (N,2). 바닥 그림에서 읽고, 점선의 빈칸은 칠한 것으로 메운다.
        빈칸 길이는 그림에서 잰다: 교차로 밖에서 양쪽이 칠로 막힌 빈 구간 길이의 중앙값. 그 GAP배까지만 메우므로 교차로의 빈 곳은 남는다."""
        info = json.load(open(road + "/road.json")); ppm = info["ppm"]
        inst = cv2.imread(road + "/inst.png", cv2.IMREAD_UNCHANGED); H, W = inst.shape[:2]
        white = np.isin(inst, [int(k) for k, v in info["instances"].items() if v == "white_lane"]).astype(np.uint8)
        white = cv2.dilate(white, np.ones((3, 3), np.uint8)).astype(bool)             # 1픽셀 흔들려도 칠한 것으로
        n = np.c_[-np.sin(self.yaw), np.cos(self.yaw)]; out, runs = [], []
        for pts in (self.P + n * self.wl[:, None], self.P - n * self.wr[:, None]):
            u, v = np.round(pts[:, 0] * ppm).astype(int), np.round(H - pts[:, 1] * ppm).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            raw = np.zeros(len(pts), bool); raw[ok] = white[v[ok], u[ok]]
            edge = np.r_[True, (raw[1:] != raw[:-1]) | (self.pid[1:, 0] != self.pid[:-1, 0]), True]   # 구간 경계(경로가 바뀌는 곳도 경계)
            st = np.where(edge)[0]
            for a, b in zip(st[:-1], st[1:]):                                       # 빈 구간 [a, b) 가운데 같은 경로 안에서 양쪽이 칠로 막힌 것
                if not raw[a] and a > 0 and b < len(raw) and raw[a - 1] and raw[b] and self.pid[a - 1, 0] == self.pid[b, 0]:
                    runs.append((len(out), a, b, bool(self.zone[a:b].any())))
            out.append(raw)
        free = [b - a for _, a, b, z in runs if not z]
        lim = GAP * float(np.median(free)) if free else 0.0
        for side, a, b, _ in runs:
            if b - a <= lim:
                out[side][a:b] = True
        return np.c_[out[0], out[1]]

    def lane_changes(self, road, step):
        """(옮기기 전 점, 옮긴 뒤 점, 값) 목록. 바닥 그림에서 두 차로 사이 선이 점선인지 본다."""
        info = json.load(open(road + "/road.json")); ppm = info["ppm"]
        inst = cv2.imread(road + "/inst.png", cv2.IMREAD_UNCHANGED); H, W = inst.shape[:2]
        paint = np.isin(inst, [int(k) for k, v in info["instances"].items() if v == "white_lane"])
        out = []
        for a in np.where(self.zone == 0)[0][::LC_EVERY]:
            t = np.array([np.cos(self.yaw[a]), np.sin(self.yaw[a])]); nl = np.array([-t[1], t[0]])
            lw = self.wl[a] + self.wr[a]
            for side, wa in ((1, self.wl[a]), (-1, self.wr[a])):
                for b in self.tree.query_ball_point(self.P[a] + side * nl * lw, 0.25 * lw):
                    if self.zone[b] or self.pid[b, 0] == self.pid[a, 0]:
                        continue
                    if abs((self.yaw[a] - self.yaw[b] + np.pi) % (2 * np.pi) - np.pi) > np.radians(SAME_YAW):
                        continue
                    wb = self.wr[b] if side > 0 else self.wl[b]
                    d = self.P[b] - self.P[a]
                    if abs(d @ nl * side - (wa + wb)) > 0.15 * lw or abs(d @ t) > step:
                        continue                                             # 선 하나를 같이 쓰는 바로 옆 차로가 아님
                    L = LC_LEN * lw / 2                                     # 점선인지: 사이 선 위 앞뒤 L 구간의 칠한 비율
                    h = int(L / step)                                       # 그 구간이 두 차로 모두 교차로 밖이어야 함(교차로 빈 곳이 섞이면 칠한 비율이 점선처럼 나옴)
                    if not all(0 <= c - h and c + h < len(self.P) and (self.pid[c - h:c + h + 1, 0] == self.pid[c, 0]).all()
                               and not self.zone[c - h:c + h + 1].any() for c in (a, b)):
                        continue
                    pts = self.P[a] + side * nl * wa + np.linspace(-L, L, int(2 * L))[:, None] * t
                    u, v = np.round(pts[:, 0] * ppm).astype(int), np.round(H - pts[:, 1] * ppm).astype(int)
                    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    near = np.zeros(len(pts), bool)
                    for du in (-1, 0, 1):                                   # 선 굵기 안에서 1픽셀 흔들려도 칠한 것으로
                        for dv in (-1, 0, 1):
                            near[ok] |= paint[np.clip(v[ok] + dv, 0, H - 1), np.clip(u[ok] + du, 0, W - 1)]
                    if DASH[0] < near.mean() < DASH[1]:
                        out.append((a, b, float(np.linalg.norm(d)) + LC_PEN * lw))
                    break
        return out

    def current(self, xy, yaw, prefer=()):
        """로봇 중심(mm)이 양쪽 선 사이에 들어 있는 차로의 점(교차로 밖). prefer(길의 점들)에 있으면 그것을 먼저. 없으면 -1."""
        ids = np.array(self.tree.query_ball_point(xy, float(np.max(self.wl + self.wr))), int)
        if len(ids) == 0:
            return -1
        c, sn = np.cos(self.yaw[ids]), np.sin(self.yaw[ids]); d = xy - self.P[ids]
        off, lon = c * d[:, 1] - sn * d[:, 0], c * d[:, 0] + sn * d[:, 1]
        turn = np.abs((self.yaw[ids] - yaw + np.pi) % (2 * np.pi) - np.pi)
        on = (-self.wr[ids] <= off) & (off < self.wl[ids]) & (np.abs(lon) <= 4.0 * SAME * 5) & (turn < np.radians(45))
        if not on.any():
            return -1
        ids, lon = ids[on], np.abs(lon[on])
        pref = np.isin(ids, list(prefer))
        if pref.any():
            ids, lon = ids[pref], lon[pref]
        return int(ids[np.argmin(lon)])

    def ahead(self, k, ids, length):
        """점 k에서 앞으로 length(mm)만큼 지금 차로를 따라간 점들. 길(ids) 위면 길을 따르되 차선 변경 선은 타지 않고 같은 경로로 간다."""
        pos = {int(v): i for i, v in enumerate(ids)}; out, s = [k], 0.0
        while s < length:
            i = pos.get(k)
            nk = int(ids[i + 1]) if i is not None and i + 1 < len(ids) and (k, int(ids[i + 1])) not in self.lc else int(self.nxt[k])
            if nk < 0:
                break
            s += float(np.linalg.norm(self.P[nk] - self.P[k])); out.append(nk); k = nk
        return np.array(out)

    def snap(self, xy, yaw):
        """위치(mm)·방향(rad)에 가장 가까운, 방향이 맞는 점 번호."""
        ids = self.tree.query_ball_point(xy, 300.0) or [int(self.tree.query(xy)[1])]
        ids = np.array(ids)
        dy = np.abs((self.yaw[ids] - yaw + np.pi) % (2 * np.pi) - np.pi)
        cost = np.linalg.norm(self.P[ids] - xy, axis=1) + 1000.0 * (dy > np.radians(45))
        return int(ids[np.argmin(cost)])

    def route(self, s, g):
        """점 s → 점 g의 가장 짧은 길. 점 번호 배열(같은 자리로 건너간 점은 하나만 남김), 길이(mm, 벌칙 포함). 길이 없으면 (None, inf)."""
        dist, pred = dijkstra(self.G, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return None, np.inf
        out, k = [g], g
        while k != s:
            k = pred[k]; out.append(k)
        out = np.array(out[::-1])
        keep = np.r_[True, np.linalg.norm(np.diff(self.P[out], axis=0), axis=1) > 0.5]  # 같은 자리 건너가기는 하나로
        return out[keep], float(dist[g])

    def lane(self, ids):
        """길의 점들 → 중심(mm), 방향, 왼쪽 선, 오른쪽 선(mm), 교차로 안 여부."""
        P, yaw = self.P[ids], self.yaw[ids]
        n = np.c_[-np.sin(yaw), np.cos(yaw)]
        return P, yaw, P + n * self.wl[ids, None], P - n * self.wr[ids, None], self.zone[ids]
