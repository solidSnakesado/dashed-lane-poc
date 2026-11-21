"""모델이 직접 내놓은 '내 차로 왼쪽 선·오른쪽 선·중심선'(ego_left, ego_right, ego_center)을 바닥 위의 선과 중심선으로 바꾼다.

lane_live.py와 쓰는 법이 같지만 훨씬 단순하다: 조각 묶기·추적·좌우 고르기가 없다.
중심선만 기억한다: 앞서 본 중심선을 로봇이 움직인 만큼 옮겨 두고 새로 본 것과 합친다(move를 줄 때만).
도로 크기에 걸린 숫자는 쓰지 않는다. 거리 기준은 모두 카메라 높이의 배수다.
"""
import cv2
import numpy as np

from lane_curve import local_fit
from lane_geom import Camera

REACH = 12.0     # 카메라 높이의 이 배수까지만 선을 내보냄 (58 mm → 약 0.7 m). 그보다 먼 바닥은 위치 오차가 큼
SPAN = 2.0       # 선을 다듬을 때 주변으로 보는 거리 = 카메라 높이의 이 배수
LOOK = 1.25      # 조향 목표로 쓸 앞 거리 = 차로 폭의 이 배수 (차로 폭을 아직 모르면 카메라 높이의 5배)
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

    def carry(self, move):
        """기억한 점들을 로봇이 움직인 만큼 옮기고, 달린 거리만큼 가중치를 줄인다."""
        th, t = move[0], np.array(move[1:], float)
        q = self.mem[:, :2] - t
        q = np.c_[q @ [np.cos(th), np.sin(th)], q @ [-np.sin(th), np.cos(th)]]     # 왼쪽으로 th만큼 돈 새 방향 기준
        w = self.mem[:, 2] * np.exp(-np.linalg.norm(t) / (MEM * self.cam_h))
        keep = (w >= MEM_MIN) & (q[:, 1] > 0) & (q[:, 1] <= REACH * self.cam_h)
        self.mem = np.c_[q, w][keep]

    def merge(self, seen):
        """새로 본 중심선(없으면 None)을 기억과 합쳐 다듬은 중심선을 돌려준다. 쓸 점이 없으면 None."""
        if seen is not None:
            self.mem = np.r_[self.mem, np.c_[seen, np.ones(len(seen))]]
        if len(self.mem) < 4 or np.ptp(self.mem[:, 1]) < self.cam_h:
            return None
        m = self.mem[np.argsort(self.mem[:, 1])]
        _, p, _, _ = local_fit(m[:, :2], m[:, 2], SPAN * self.cam_h, step=SPAN * self.cam_h / 4)
        return p if p[0, 1] <= p[-1, 1] else p[::-1]

    def line(self, polys, size):
        """한 클래스의 마스크들 → 바닥 위 선 (N,2) [오른쪽 m, 앞 m], 가까운 곳부터. 못 만들면 None."""
        mask = np.zeros(size[::-1], np.uint8)
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
        v, u = np.nonzero(mask)
        if len(u) < MIN_PX:
            return None
        pick = np.linspace(0, len(u) - 1, min(len(u), MAX_PTS)).astype(int)
        g, ok = self.cam.to_ground(np.c_[u[pick], v[pick]].astype(float))
        g = g[ok & (g[:, 1] > 0) & (g[:, 1] <= REACH * self.cam_h)]
        if len(g) < MIN_PX // 2 or np.ptp(g, axis=0).max() < self.cam_h:
            return None
        g = g[np.argsort(g[:, 1])]
        _, p, _, _ = local_fit(g, np.ones(len(g)), SPAN * self.cam_h, step=SPAN * self.cam_h / 4)
        return p if p[0, 1] <= p[-1, 1] else p[::-1]

    def step(self, polys, size, move=None):
        """polys = [(클래스 이름, 픽셀 폴리곤)], size = (화면 폭, 높이).
        move = 직전 프레임 이후 카메라가 움직인 양 (왼쪽으로 돈 각 rad, 오른쪽 이동 m, 앞 이동 m). 없으면 기억을 쓰지 않는다.
        돌려주는 값은 lane_live.Live.step과 같은 형식."""
        left, right, mid = (self.line([p for c, p in polys if c == k], size) if any(c == k for c, _ in polys) else None
                            for k in ("ego_left", "ego_right", "ego_center"))
        center, guess = mid, mid is None                   # 모델이 중심선을 직접 내놓으면 그것을 씀
        if left is not None and right is not None:         # 같은 앞 거리에서 두 선의 가운데 (차로 폭도 여기서 잼)
            z = np.linspace(max(left[0, 1], right[0, 1]), min(left[-1, 1], right[-1, 1]), 12)
            if z[-1] > z[0]:
                xl, xr = np.interp(z, left[:, 1], left[:, 0]), np.interp(z, right[:, 1], right[:, 0])
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
        return {"left": left, "right": right, "center": center, "guess": guess,
                "look": LOOK * lane_w if lane_w else 5 * self.cam_h, "lane_w": lane_w or 0.0}
