"""로봇이 실제로 움직인 양(바퀴 센서 값)을 추적기에 넘겨준다. 차선으로 움직임을 짐작하지 않기 위한 입력.

지금은 시뮬용: sim_drive.py가 만든 gt.csv의 로봇 위치·방향(x, y, yaw)에서 프레임 사이 움직임을 계산한다.
실제 핑키에서는 같은 형식(회전, 옆·앞 이동)을 바퀴 엔코더(/odom)에서 만들어 넘기면 된다.
"""
import csv

import numpy as np

CAM_FWD = 0.033     # 바퀴 축 중심에서 카메라까지 앞쪽 거리(m). pinky_camera.yaml의 forward_m과 같은 값


class Odom:
    """프레임 번호를 주면 직전에 준 프레임 이후의 움직임을 돌려준다."""
    def __init__(self, gt_path, calib_path, noise=0.0):
        d = np.load(calib_path)
        self.scale = float(d["scale"]) if "scale" in d else 1.0     # 후처리 좌표는 실제 도로 크기로 키운 단위
        self.pose = {int(r["frame"]): (float(r["x"]), float(r["y"]), float(r["yaw"]))
                     for r in csv.DictReader(open(gt_path))}
        # 센서 오차 시험용: 이동 거리는 noise 비율만큼, 회전은 그 2.5배만큼 크게 잰 것으로 만듦
        self.k_dist, self.k_rot, self.last = 1.0 + noise, 1.0 + 2.5 * noise, None

    def cam(self, k):
        """프레임 k의 카메라 바닥 위치, 방향 각, 앞 방향, 오른쪽 방향."""
        x, y, yaw = self.pose[k]
        fwd, right = np.array([np.cos(yaw), np.sin(yaw)]), np.array([np.sin(yaw), -np.cos(yaw)])
        return np.array([x, y]) + CAM_FWD * fwd, yaw, fwd, right

    def step(self, name):
        """프레임 이름('f000012')을 받아 (돌릴 각 rad, 옮길 양 [옆, 앞])을 돌려준다. 첫 프레임은 움직임 0.

        돌려주는 값은 '기억해 둔 선을 얼마나 돌리고 옮길지'다: 로봇이 왼쪽으로 돌면 선은 오른쪽으로 돈다.
        """
        k = int(name.lstrip("f"))
        if self.last is None or k not in self.pose or self.last not in self.pose:
            self.last = k
            return 0.0, np.zeros(2)
        (o0, y0, f0, r0), (o1, y1, _, _) = self.cam(self.last), self.cam(k)
        self.last = k
        move = np.array([(o1 - o0) @ r0, (o1 - o0) @ f0]) * self.scale * self.k_dist   # 직전 방향 기준 옆·앞 이동
        th = -np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0)) * self.k_rot
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return th, -rot @ move
