"""도로 그래프와 길 찾기: 지도(road.json의 경로들)를 점 그래프로 만들고, 출발점 → 목적지의 가장 짧은 길을 찾는다.

점: 경로마다 STEP 간격으로 다시 찍은 점(ego_label.Routes와 같은 점). 점마다 차로 왼쪽·오른쪽 선까지 거리와 교차로 안인지가 붙어 있다.
선(이동):
  같은 경로 안에서 다음 점으로 = 그 사이 길이.
  서로 다른 경로의 점이 거의 같은 자리·같은 방향이면 = 같은 차로를 같이 달리는 것이므로 서로 오갈 수 있게 잇는다.
    값 = 두 점 사이 거리 + 점 간격의 절반(벌칙). 벌칙이 없으면 경로 사이를 지그재그로 오가며 길이를 줄이는 길이 나온다.
  옆 차로로 건너가는 선은 이번에는 만들지 않는다(차선 변경 제외).
도로 크기에 걸린 숫자는 쓰지 않는다. 같은 자리 기준은 점 간격의 배수다.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

SAME = 0.9           # 같은 자리로 보는 거리 = 점 간격의 이 배수
SAME_YAW = 15.0      # 같은 방향으로 보는 방향 차(도)
TINY = 1e-6          # 0 대신 쓰는 아주 작은 값(0은 그래프에서 '선 없음'으로 읽힘)
JUMP = 0.5           # 다른 경로로 건너갈 때 벌칙 = 점 간격의 이 배수


class Graph:
    def __init__(self, rt, step):
        """rt = ego_label.Routes, step = 그 점 간격(mm)."""
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
                rows += [a, b]; cols += [b, a]; w += [d, d]
        n = len(self.P)
        self.G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()

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
