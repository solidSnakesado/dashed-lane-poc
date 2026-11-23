"""모델이 직접 내놓은 '내 차로 왼쪽 선·오른쪽 선·중심선'(ego_left, ego_right, ego_center)과 좌회전 면(lturn_lane)을 바닥 위의 선과 중심선으로 바꾼다.

lane_live.py와 쓰는 법이 같지만 훨씬 단순하다: 조각 묶기·추적·좌우 고르기가 없다.
중심선만 기억한다: 앞서 본 중심선을 로봇이 움직인 만큼 옮겨 두고 새로 본 것과 합친다(move를 줄 때만).
선은 '앞 거리 순서'가 아니라 '선을 따라가는 순서'로 만든다(lane_trace.py). 교차로에서 돌 때는 갈 길이 옆으로 눕기 때문에 앞 거리로는 순서를 정할 수 없다.
도로 크기에 걸린 숫자는 쓰지 않는다. 거리 기준은 모두 카메라 높이의 배수다.
"""
import cv2
import numpy as np

from lane_geom import Camera
from lane_trace import trace

REACH = 12.0     # 카메라 높이의 이 배수까지만 선을 내보냄 (58 mm → 약 0.7 m). 그보다 먼 바닥은 위치 오차가 큼
SPAN = 2.0       # 선을 다듬을 때 주변으로 보는 거리 = 카메라 높이의 이 배수
LOOK = 1.25      # 조향 목표로 쓸 앞 거리 = 차로 폭의 이 배수 (차로 폭을 아직 모르면 카메라 높이의 5배)
EDGE = 2         # 화면 끝에서 이 픽셀 안에 닿은 경계는 잘린 것으로 봄
MIN_PX = 30      # 마스크 픽셀이 이보다 적으면 선으로 쓰지 않음
MAX_PTS = 600    # 선 하나에 쓰는 마스크 픽셀 수의 상한 (계산량 제한)
MEM = 2.0        # 중심선 기억이 흐려지는 거리 = 카메라 높이의 이 배수만큼 달릴 때마다 가중치가 1/e로 줄어듦
MEM_MIN = 0.05   # 가중치가 이보다 작아진 기억은 버림


class Direct:
    def __init__(self, calib_path, fps=None):
        """calib_path = 카메라 보정 파일(npz, 실제 m 단위)."""
        self.cam = Camera(calib_path)
        self.cam_h = float(np.load(calib_path)["cam_h"])
        self.widths = []                                   # 양쪽 선이 함께 보일 때 잰 차로 폭들
        self.mem = np.zeros((0, 3))                        # 기억한 중심선 점들 [오른쪽 m, 앞 m, 가중치] (지금 카메라 기준)
        self.prev = None                                   # 직전 프레임의 중심선 (지금 카메라 기준으로 옮겨 둠). 면을 따라갈 때 출발 방향으로 씀
        self.side = 0.0                                    # 도는 방향(왼쪽 -1). 옆으로 누운 선의 가는 방향을 정할 때 씀

    def carry(self, move):
        """기억한 점들을 로봇이 움직인 만큼 옮기고, 달린 거리만큼 가중치를 줄인다."""
        th, t = move[0], np.array(move[1:], float)
        turn = lambda q: np.c_[(q - t) @ [np.cos(th), np.sin(th)], (q - t) @ [-np.sin(th), np.cos(th)]]   # 왼쪽으로 th만큼 돈 새 방향 기준
        q, self.prev = turn(self.mem[:, :2]), None if self.prev is None else turn(self.prev)
        w = self.mem[:, 2] * np.exp(-np.linalg.norm(t) / (MEM * self.cam_h))
        keep = (w >= MEM_MIN) & (q[:, 1] > 0) & (np.linalg.norm(q, axis=1) <= REACH * self.cam_h)
        self.mem = np.c_[q, w][keep]

    def merge(self, seen):
        """새로 본 중심선(없으면 None)을 기억과 합쳐 다듬은 중심선을 돌려준다. 쓸 점이 없으면 None."""
        if seen is not None:
            self.mem = np.r_[self.mem, np.c_[seen, np.ones(len(seen))]]
        if len(self.mem) < 4:
            return None
        p, _ = trace(self.mem[:, :2], self.mem[:, 2], SPAN * self.cam_h, REACH * self.cam_h, self.side)
        return p if len(p) >= 2 and np.linalg.norm(np.diff(p, axis=0), axis=1).sum() >= self.cam_h else None

    def ground(self, px):
        """화면 픽셀들 → 쓸 수 있는 바닥 점들 (로봇 앞, REACH 안)."""
        g, ok = self.cam.to_ground(px.astype(float))
        return g[ok & (g[:, 1] > 0) & (np.linalg.norm(g, axis=1) <= REACH * self.cam_h)]

    def chains(self, g, side=0.0):
        """바닥 점 구름 → 선들 [(N,2), ...]. 로봇에서 가까운 것부터. 끊어져 있으면 여러 개가 나온다."""
        out, left = [], np.ones(len(g), bool)
        while left.sum() >= MIN_PX // 2 and len(out) < 4:
            p, seen = trace(g[left], np.ones(left.sum()), SPAN * self.cam_h, REACH * self.cam_h, side)
            left[np.nonzero(left)[0][seen]] = False
            if len(p) >= 2 and np.linalg.norm(np.diff(p, axis=0), axis=1).sum() >= self.cam_h:
                out.append(p)
        return out

    def lines(self, polys, size, side=0.0):
        """한 클래스의 마스크들 → 바닥 위 선들."""
        mask = np.zeros(size[::-1], np.uint8)
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
        v, u = np.nonzero(mask)
        if len(u) < MIN_PX:
            return []
        pick = np.linspace(0, len(u) - 1, min(len(u), MAX_PTS)).astype(int)
        return self.chains(self.ground(np.c_[u[pick], v[pick]]), side)

    def edge(self, polys, size, lane_w):
        """좌회전 면 → 차로 중심선 (N,2). 면을 따라 한 걸음씩 가면서, 가는 방향에 가로로 자른 단면의 바깥쪽(오른쪽) 끝에서 차로 폭의 절반만큼 안쪽을 잡는다.
        면의 가운데를 쓰지 않는 이유: 왼쪽으로 휘어 나가는 면은 화면 왼쪽 끝에서 잘려, 보이는 부분의 가운데가 오른쪽으로 치우친다.
        오른쪽 끝이 화면 밖으로 잘린 단면에서는 왼쪽 끝에서 절반만큼 바깥을 잡고, 양쪽이 다 잘렸으면 가던 방향을 유지한다."""
        w, h = size
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
        v, u = np.nonzero(mask)
        if len(u) < MIN_PX:
            return None
        pick = np.linspace(0, len(u) - 1, min(len(u), 5 * MAX_PTS)).astype(int)
        g, ok = self.cam.to_ground(np.c_[u[pick], v[pick]].astype(float))
        ok &= (g[:, 1] > 0) & (np.linalg.norm(g, axis=1) <= REACH * self.cam_h)
        g, cut_l, cut_r = g[ok], mask[v[pick], : EDGE + 1].any(axis=1)[ok], mask[v[pick], -EDGE - 1:].any(axis=1)[ok]
        if len(g) < MIN_PX:
            return None
        step, d, path = SPAN * self.cam_h / 2, np.array([0.0, 1.0]), []
        c = np.array([np.median(g[g[:, 1] <= g[:, 1].min() + step, 0]), g[:, 1].min() - step / 2])
        if self.prev is not None and len(self.prev) >= 2:                  # 출발점과 출발 방향: 직전 중심선이 면에 처음 닿는 곳
            hit = np.nonzero(np.linalg.norm(self.prev[:, None] - g[None, ::7], axis=2).min(axis=1) <= step / 2)[0]
            if len(hit):
                k = min(int(hit[0]), len(self.prev) - 2)
                d = (self.prev[k + 1] - self.prev[k]) / (np.linalg.norm(self.prev[k + 1] - self.prev[k]) + 1e-9)
                c = self.prev[k] - step / 2 * d
        for _ in range(int(2 * REACH * self.cam_h / step)):
            c2, n = c + step * d, np.array([d[1], -d[0]])
            rel = g - c2
            sel = (np.abs(rel @ d) <= step / 2) & (np.abs(rel @ n) <= 1.5 * lane_w)
            if sel.sum() < 5:
                break
            s = rel[sel] @ n
            i, j = int(np.argmax(s)), int(np.argmin(s))
            if not cut_r[sel][i]:
                c2 = c2 + (np.percentile(s, 95) - lane_w / 2) * n
            elif not cut_l[sel][j]:
                c2 = c2 + (np.percentile(s, 5) + lane_w / 2) * n
            d = (c2 - c) / (np.linalg.norm(c2 - c) + 1e-9)
            path.append(c2)
            c = c2
        return np.array(path) if len(path) >= 2 else None

    def join(self, egos, turn):
        """내 차로 중심선들 egos와 좌회전 선 turn을 가는 순서대로 잇는다: 좌회전 선이 시작되기 전의 ego + turn + turn이 끝나는 방향으로 이어지는 ego."""
        ego = egos[0] if egos else None
        if turn is None:
            return ego
        r, head, tail = SPAN * self.cam_h, turn[:0], turn[:0]
        if ego is not None and np.linalg.norm(ego[0]) < np.linalg.norm(turn[0]):
            head = ego[: int(np.argmin(np.linalg.norm(ego - turn[0], axis=1)))]
        t, best = turn[-1] - turn[max(0, len(turn) - 3)], 0.0
        for e in egos:
            k = int(np.argmin(np.linalg.norm(e - turn[-1], axis=1)))
            if np.linalg.norm(e[k] - turn[-1]) > r:
                continue
            for part in (e[k:], e[: k + 1][::-1]):                     # 좌회전 선이 끝나는 방향으로 가장 길게 이어지는 쪽
                if len(part) >= 2 and (part[-1] - part[0]) @ t > best:
                    tail, best = part, (part[-1] - part[0]) @ t
        return np.vstack([head, turn, tail])

    def step(self, polys, size, move=None, route="lane"):
        """polys = [(클래스 이름, 픽셀 폴리곤)], size = (화면 폭, 높이).
        route = 가는 방향. "left"(좌회전)이면 좌회전 면(lturn_lane)이 보이는 곳부터는 그 면에서 만든 선을 따라간다. 그 밖에는 내 차로 선(ego_*)만 쓴다.
        move = 직전 프레임 이후 카메라가 움직인 양 (왼쪽으로 돈 각 rad, 오른쪽 이동 m, 앞 이동 m). 없으면 기억을 쓰지 않는다.
        돌려주는 값은 lane_live.Live.step과 같은 형식. center는 로봇에서 가까운 곳부터 가는 순서."""
        self.side = -1.0 if route == "left" else 0.0
        of = lambda k: [p for c, p in polys if c == k]
        get = lambda k: self.lines(of(k), size, self.side) if of(k) else []
        first = lambda ch: ch[0] if ch else None
        left, right, mids = first(get("ego_left")), first(get("ego_right")), get("ego_center")
        mid = first(mids)
        if route == "left":                                # 좌회전 면 앞쪽(아직 갈라지기 전)과 뒤쪽(돌아 나갈 차로)은 내 차로 중심선, 가운데는 좌회전 면에서 만든 선
            lw0 = float(np.median(self.widths)) if self.widths else None
            turn = None if not of("lturn_lane") else self.edge(of("lturn_lane"), size, lw0) if lw0 else first(get("lturn_lane"))
            mid = self.join(mids, turn)                    # 차로 폭을 알면 면의 바깥쪽 경계 기준, 모르면 면의 가운데
        center, guess = mid, mid is None                   # 모델이 중심선을 직접 내놓으면 그것을 씀
        if left is not None and right is not None:         # 같은 앞 거리에서 두 선의 가운데 (차로 폭도 여기서 잼)
            ls, rs = left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]
            z = np.linspace(max(ls[0, 1], rs[0, 1]), min(ls[-1, 1], rs[-1, 1]), 12)
            if z[-1] > z[0]:
                xl, xr = np.interp(z, ls[:, 1], ls[:, 0]), np.interp(z, rs[:, 1], rs[:, 0])
                center, guess = (np.c_[(xl + xr) / 2, z], False) if mid is None else (center, guess)
                self.widths = (self.widths + [float(np.median(xr - xl))])[-60:]
        lane_w = float(np.median(self.widths)) if self.widths else None
        if move is not None:                               # 기억: 모델이 본 중심선(또는 두 선의 가운데)만 쌓고, 못 본 프레임은 기억으로 채움
            self.carry(move)
            kept = self.merge(center)
            center, guess = (kept, guess) if kept is not None else (center, guess)
        if center is None and (left is not None or right is not None):
            ln, side = (left, 1.0) if left is not None else (right, -1.0)
            t = np.gradient(ln, axis=0)
            t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
            nrm = np.c_[t[:, 1], -t[:, 0]]                                     # 선의 오른쪽 방향
            if lane_w:                                                         # 한쪽 선에서 차로 폭의 절반만큼 옮김
                center = ln + side * (lane_w / 2) * nrm
            else:                                # 차로 폭을 아직 모름(양쪽 선을 함께 본 적 없음): 지금의 선과의 간격을 유지
                k = int(np.argmin(np.linalg.norm(ln, axis=1)))
                center = ln + float(-ln[k] @ nrm[k]) * nrm
        self.prev = center
        return {"left": left, "right": right, "center": center, "guess": guess,
                "look": LOOK * lane_w if lane_w else 5 * self.cam_h, "lane_w": lane_w or 0.0}
