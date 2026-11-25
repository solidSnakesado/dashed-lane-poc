"""차로 안 위치: 지도(도로 그래프)와 추정 위치로 '지금 어느 차로의 어디쯤인지'를 내고, 비전이 본 내 차로 선으로 그 위치를 확인한다.

1) locate: 지도 + 추정 위치만 쓴다(학습·카메라 없음).
   상태 "lane" = 차로의 두 선 사이, "other" = 가려는 길에 없는 다른 차로의 두 선 사이(길을 준 경우만),
        "out" = 교차로 밖인데 어느 차로의 두 선 사이에도 없음, "cross" = 교차로 안(칠해진 선이 없어 안/밖을 따지지 않음)
   off = 차로 가운데에서 옆으로 벗어난 거리(m, 왼쪽 +), left / right = 왼쪽 선 / 오른쪽 선까지 거리(m, 선을 넘으면 음수)
2) confirm: 비전(lane_direct가 낸 내 차로 왼쪽·오른쪽 선)과 지도 선을 비교한다(lane_check).
   "ok" = 비전이 본 선과 맞음, "mismatch" = 옆으로 차로 폭의 lane_check.WARN배보다 어긋남(추정 위치가 틀렸을 수 있음),
   "none" = 확인 못 함(교차로 안, 차로 밖, 다른 차로, 선이 안 보이거나 모자람). lat = 비전이 본 추정 위치의 옆 오차(m, 왼쪽 +).
어느 지도에나 같은 방식으로 쓴다(고정 크기 없음: 거리는 지도의 차로 폭과 카메라 높이에서 나온다).
"""
import numpy as np
import lane_check

AHEAD = 12.0         # 비전과 비교할 지도 선의 길이 = 카메라 높이의 이 배수(lane_check.NEAR의 먼 쪽보다 넉넉하게)


class Where:
    def __init__(self, graph, cam_h, cam_fwd):
        """graph = route_graph.Graph, cam_h = 카메라 높이(m), cam_fwd = 바퀴 축에서 카메라까지 앞쪽 거리(m)."""
        self.g, self.cam_h, self.cam_fwd = graph, cam_h, cam_fwd
        self.route, self.on = None, set()

    def set_route(self, route_ids):
        """가려는 길(그래프 점 번호 배열). 겹친 차로에서는 이 길의 차로를 먼저 고르고, 비교할 지도 선도 이 길을 따라 만든다."""
        self.route = None if route_ids is None else np.asarray(route_ids, int)
        self.on = set() if self.route is None else set(int(v) for v in self.route)

    def offset(self, xy, k):
        """점 k의 차로 가운데에서 xy(mm)가 옆으로 벗어난 거리(mm, 왼쪽 +)."""
        g = self.g; d = xy - g.P[k]
        return float(np.cos(g.yaw[k]) * d[1] - np.sin(g.yaw[k]) * d[0])

    def nearest(self, xy, yaw):
        """두 선 사이에 든 차로가 없을 때 기준으로 삼을 점: 길이 있으면 길 위의 가장 가까운 점, 없으면 방향이 맞는 가장 가까운 점."""
        if self.route is not None:
            return int(self.route[np.argmin(np.linalg.norm(self.g.P[self.route] - xy, axis=1))])
        return self.g.snap(xy, yaw)

    def locate(self, pose):
        """pose = 추정 위치 (x m, y m, yaw rad), 바퀴 축 기준. 돌려주는 값: dict(state, k, path, off, left, right, lane_w). off·left·right는 점 k의 차로 기준."""
        g = self.g; xy, yaw = np.array(pose[:2], float) * 1000.0, float(pose[2])
        k = g.current(xy, yaw, self.on)
        near = self.nearest(xy, yaw)
        if g.zone[near] or (k >= 0 and g.zone[k]):                     # 교차로 안: 길(또는 가장 가까운 차로)이 교차로 구간이면 선이 없는 곳
            k, state = near if g.zone[near] else k, "cross"
        elif k >= 0:
            state = "lane" if self.route is None or k in self.on else "other"
        else:
            k, state = near, "out"
        off = self.offset(xy, k)
        return dict(state=state, k=int(k), path=int(g.pid[k, 0]), off=off / 1000.0, left=float(g.wl[k] - off) / 1000.0,
                    right=float(g.wr[k] + off) / 1000.0, lane_w=float(g.wl[k] + g.wr[k]) / 1000.0)

    def map_lines(self, k, pose):
        """점 k의 차로를 앞으로 따라간 지도 선(칠해진 선이 이어지는 데까지)을 추정 위치의 카메라 기준 점열로. 없으면 None."""
        g = self.g
        ids = g.ahead(k, self.route if self.route is not None else np.array([k]), AHEAD * self.cam_h * 1000.0)
        _, _, left, right, z = g.lane(ids)
        out = {}
        for side, line in enumerate((left, right)):                     # 쪽마다: 교차로 밖은 그대로, 교차로 구간은 그 선이 칠해진 데까지만
            have = (z == 0) if g.paint is None else (z == 0) | g.paint[ids, side]
            m = len(ids) if have.all() else int(np.argmin(have))        # 처음 끊기는 곳까지(그 뒤를 붙이면 빈 곳을 가로지르는 가짜 선분이 생긴다)
            out["left" if side == 0 else "right"] = lane_check.to_cam(line[:m] / 1000.0, pose, self.cam_fwd) if m >= 2 else None
        return None if out["left"] is None and out["right"] is None else out

    def lane_ahead(self, k, pose):
        """그리기용: 점 k에서 길을 따라 앞으로 간 내 차로의 왼쪽·오른쪽 선(카메라 기준 점열)과 점마다 교차로 안 여부.
        교차로 안도 뺀 곳 없이 준다(칠해진 선은 없지만 지도를 믿고 그리는 부분). 비전 확인에는 쓰지 않는다."""
        g = self.g
        ids = g.ahead(k, self.route if self.route is not None else np.array([k]), AHEAD * self.cam_h * 1000.0)
        if len(ids) < 2:
            return None
        _, _, left, right, zone = g.lane(ids)
        return dict(left=lane_check.to_cam(left / 1000.0, pose, self.cam_fwd), right=lane_check.to_cam(right / 1000.0, pose, self.cam_fwd), zone=zone)

    def confirm(self, where, seen, pose):
        """where = locate의 결과, seen = {"left": 점열, "right": 점열}(비전, 카메라 기준 m) 또는 None. 돌려주는 값: dict(status, lat, yaw, n)."""
        none = dict(status="none", lat=None, yaw=None, n=0)
        if where["state"] not in ("lane", "cross") or seen is None:     # 교차로 구간도 칠해진 선이 있으면 확인한다(없으면 지도 선이 없어 '확인 못 함')
            return none
        expect = self.map_lines(where["k"], pose)
        c = None if expect is None else lane_check.check(seen, expect, self.cam_h, self.cam_fwd, where["lane_w"])
        if c is None:
            return none
        return dict(status="mismatch" if c["warn"] else "ok", lat=c["lat"], yaw=c["yaw"], n=c["n"])

    def step(self, pose, seen=None):
        """한 프레임: 위치(locate)와 비전 확인(confirm)을 한 dict로."""
        w = self.locate(pose)
        w.update(self.confirm(w, seen, pose))
        return w
