"""카메라 한 프레임의 검출 결과를 받아 내 차로의 양쪽 선과 중심선을 바로 내놓는다 (실시간용).

lane_run.py가 영상 파일을 통째로 처리하는 것과 달리, 프레임마다 step()을 한 번씩 부른다. 시뮬 주행과 실제 핑키에서 함께 쓴다.
도로 규격은 넣지 않는다: 처음 몇 프레임은 화면에서 규격을 재고(lane_params.Auto), 그동안 step()은 None을 돌려준다.
"""
import numpy as np

import lane_geom
import lane_roles
from lane_geom import Camera, drop_marks, frame_to_pieces
from lane_group import build_lanes
from lane_params import Auto
from lane_track import Tracker

REACH = 12.0     # 선을 내보내는 가장 먼 거리 = 카메라 높이의 이 배수 (핑키 카메라 높이 58 mm → 약 0.7 m)


class Live:
    def __init__(self, calib_path, fps):
        """calib_path = 카메라 보정 파일(npz, 실제 m 단위), fps = 초당 처리 프레임 수."""
        self.calib, self.cam, self.tracker, self.auto = calib_path, Camera(calib_path), Tracker(fps), None

    def step(self, polys, size, move=None):
        """polys = [(클래스 이름, 픽셀 폴리곤)], size = (화면 폭, 높이),
        move = 직전 프레임 이후 카메라가 움직인 양 (왼쪽으로 돈 각 rad, 오른쪽 이동 m, 앞 이동 m). 바퀴 센서에서 온 값.

        돌려주는 값: 규격 측정 중이면 None. 그 뒤로는 {"left", "right", "center": 점열 (N,2) [오른쪽 m, 앞 m] 또는 None,
        "guess": 중심선이 한쪽 선만으로 만든 것인지, "look": 조향 목표로 쓸 앞 거리(m), "lane_w": 측정한 차로 폭(m)}"""
        polys = drop_marks(polys)
        self.auto = self.auto or Auto(self.calib, size)
        if not self.auto.feed(polys, self.cam):
            return None
        u, odom = self.auto.scale, None                       # u = 후처리 좌표 ÷ 실제 m
        if move is not None:                                  # 로봇이 왼쪽으로 돌면 기억한 선은 오른쪽으로 돈다
            th = -move[0]
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            odom = (th, -rot @ (np.array(move[1:]) * u))
        pieces, _ = frame_to_pieces(polys, self.cam)
        self.tracker.update(build_lanes(pieces), odom)
        left, right = self.tracker.roles()
        center, guess = self.tracker.center(left, right)
        # 실제로 본 구간까지만 내보낸다: 그 너머로 이어 그린 부분(예측)은 지평선 근처에서 크게 휘어 보이므로 뺌
        # 또 카메라 높이의 REACH배보다 먼 곳은 위치 오차가 커서 내보내지 않음
        lim = REACH * self.auto.cam_h / 1000.0 * u if self.auto.cam_h else lane_geom.Z_MAX
        seen = [min(float(t.pts[:, 1].max()), lim) for t in (left, right) if t is not None]

        def pts(c, zmax):
            p = None if c is None else c.sample(0.5)
            p = None if p is None else p[p[:, 1] <= zmax]
            p = None if p is None else p[np.argsort(p[:, 1])]            # 가까운 곳부터 차례로 (순서가 뒤섞이면 선이 겹쳐 그려짐)
            return None if p is None or len(p) < 2 else p / u
        return {"left": pts(left and left.curve, seen[0] if left else 0), "right": pts(right and right.curve, seen[-1] if right else 0),
                "center": pts(center, min(seen, default=0)), "guess": guess, "look": lane_roles.Z_REF / u, "lane_w": 3.3 / u}
