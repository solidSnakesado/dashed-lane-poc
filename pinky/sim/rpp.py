"""Regulated Pure Pursuit(RPP) 컨트롤러를 파이썬으로 옮긴 것. Nav2 nav2_regulated_pure_pursuit_controller와 같은 순서로 계산한다.

쓰는 기능(핑키 nav2_params.yaml에서 켜져 있는 것만):
  주시 거리 = 속도 × lookahead_time (min~max 사이), 경로 위 주시점은 원과 선분의 교점으로 보간.
  경로 방향과 많이 어긋나면 제자리 회전(rotate_to_heading), 각가속도 제한.
  목적지 근처 감속(approach_velocity_scaling).
  곡률 감속(use_regulated_linear_velocity_scaling)은 핑키 설정에서 꺼져 있어 기본은 끔. 장애물 비용 감속은 코스트맵이 없어 뺌.
입력 위치는 바퀴 축 기준 (x, y, yaw) [m, rad]. 출력은 (선속도 m/s, 각속도 rad/s).
"""
import numpy as np

PINKY = dict(desired_linear_vel=0.2, lookahead_dist=0.6, use_velocity_scaled_lookahead_dist=True, lookahead_time=1.5,
             min_lookahead_dist=0.3, max_lookahead_dist=0.9, use_rotate_to_heading=True, rotate_to_heading_min_angle=0.35,
             rotate_to_heading_angular_vel=1.0, max_angular_accel=3.2, use_regulated_linear_velocity_scaling=False,
             regulated_linear_scaling_min_radius=0.5, regulated_linear_scaling_min_speed=0.05,
             approach_velocity_scaling_dist=0.6, min_approach_linear_velocity=0.05, max_robot_pose_search_dist=1.4875,
             max_linear=0.25, max_angular=1.5)        # 마지막 둘은 pinky_mujoco 브리지의 명령 제한(control.max_linear/angular)


class RPP:
    def __init__(self, path, dt, **over):
        """path = (N,2) m, 촘촘한 경로(Route Server dense path). dt = 제어 주기(s)."""
        self.p = dict(PINKY, **over); self.path = np.asarray(path, float); self.dt = dt
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.i0, self.w_prev = 0, 0.0

    def closest(self, xy):
        """경로에서 로봇과 가장 가까운 점(앞쪽으로만, 탐색 거리 안에서). Nav2처럼 지나간 경로는 잘라 낸다."""
        hi = int(np.searchsorted(self.s, self.s[self.i0] + self.p["max_robot_pose_search_dist"]))
        seg = self.path[self.i0:hi + 1]
        self.i0 += int(np.argmin(np.linalg.norm(seg - xy, axis=1)))
        return self.i0

    def carrot(self, xy, L):
        """로봇에서 거리 L인 경로 위 점(원과 선분의 교점). 경로 끝이 더 가까우면 끝점."""
        P = self.path[self.i0:]
        d = np.linalg.norm(P - xy, axis=1)
        k = int(np.argmax(d >= L)) if (d >= L).any() else -1
        if k <= 0:
            return P[k]
        a, b = P[k - 1] - xy, P[k] - xy; v = b - a                       # |a + t v| = L 인 t (0~1)
        A, B, C = v @ v, 2 * a @ v, a @ a - L * L
        t = (-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A + 1e-12)
        return xy + a + np.clip(t, 0, 1) * v

    def step(self, pose, v_now):
        """pose = (x, y, yaw), v_now = 지금 선속도(오도메트리). → (v, w), 남은 경로 길이(m)."""
        p, xy = self.p, np.array(pose[:2]); yaw = pose[2]
        i = self.closest(xy); remain = self.s[-1] - self.s[i]
        L = np.clip(abs(v_now) * p["lookahead_time"], p["min_lookahead_dist"], p["max_lookahead_dist"]) \
            if p["use_velocity_scaled_lookahead_dist"] else p["lookahead_dist"]
        c = self.carrot(xy, L) - xy
        cx, cy = np.cos(yaw) * c[0] + np.sin(yaw) * c[1], -np.sin(yaw) * c[0] + np.cos(yaw) * c[1]   # 로봇 기준(앞 x, 왼쪽 y)
        ang = np.arctan2(cy, cx)
        if p["use_rotate_to_heading"] and abs(ang) > p["rotate_to_heading_min_angle"]:
            w = np.sign(ang) * p["rotate_to_heading_angular_vel"]          # 제자리 회전(각가속도 제한)
            dw = p["max_angular_accel"] * self.dt
            w = float(np.clip(w, self.w_prev - dw, self.w_prev + dw)); self.w_prev = w
            return (0.0, w), remain
        d2 = cx * cx + cy * cy
        curv = 2.0 * cy / d2 if d2 > 1e-9 else 0.0
        v = p["desired_linear_vel"]
        if p["use_regulated_linear_velocity_scaling"] and abs(curv) > 1e-9 and 1 / abs(curv) < p["regulated_linear_scaling_min_radius"]:
            v = max(v * (1 / abs(curv)) / p["regulated_linear_scaling_min_radius"], p["regulated_linear_scaling_min_speed"])
        if remain < p["approach_velocity_scaling_dist"]:                   # 목적지 근처 감속
            v = max(v * remain / p["approach_velocity_scaling_dist"], p["min_approach_linear_velocity"])
        v = float(np.clip(v, -p["max_linear"], p["max_linear"]))
        w = float(np.clip(v * curv, -p["max_angular"], p["max_angular"])); self.w_prev = w
        return (v, w), remain
