"""실물 지도 작성용 자동 주행: 라이다로 양쪽 벽까지 거리를 보며 도로 가운데를 따라 달린다(오도메트리·회전 배수 오차와 관계없이 굽은 구간을 돈다).

광선 네 방향(로봇 기준 왼쪽·오른쪽 45도와 90도)의 벽까지 거리로 조향한다. 각 방향은 ±RAY_W 안의 점 거리의 중앙값, CAP에서 자른다.
  w = K90 × (왼쪽90 − 오른쪽90) + K45 × (왼쪽45 − 오른쪽45)   (+ = 왼쪽으로 돎)
  곧은 길에서 가운데 위치 오차 y와 방향 오차 θ에 대해 w ≈ −5 y − 1.0 θ 가 되게 고른 값(0.05 m/s에서 감쇠비 약 1, 계산값).
  굽은 구간에서는 바깥쪽 45도 광선이 짧아지고 안쪽은 길어져 저절로 안쪽으로 돈다.
도로 끝: 1 m 넘게 달린 뒤 양쪽 90도 광선이 모두 END_M보다 길거나(옆에 벽이 없음, 열린 끝) 앞 광선이 END_FRONT보다 짧으면(닫힌 끝) 멈추고,
  제자리에서 180도 돌아 되돌아온다. (실물 블럭 길은 둘째 직선 끝에 방 벽이 맞닿아 있어 닫힌 끝이다. 2026-11-26)
45도 광선은 CAP45에서 자르고 조향을 거른다: 실물에서 굽은 구간 안쪽 45도 광선이 블럭 틈·광택 면으로 빠져 0.6 m로 튀며 조향이 ±0.4 rad/s로 흔들렸다.
  제자리 돌기는 오도메트리로 재되, 오도메트리가 회전을 ODO_GAIN배 많이 세는 것(2026-11-26 제자리 돌기 시험 1.087)을 감안한다.
안전: 앞 ±10도 광선이 FRONT_M보다 짧으면 멈추고 끝낸다. 명령은 /cmd_vel_pc(핑키의 cmd_guard.py가 /cmd_vel로 넘김), 20 Hz.
라이다 장착 각도 오차 YAW_FIX(도, 1호기 −2.2)를 스캔 각도에 더해 쓴다(scan_fix.py와 같음).
사용: python wall_follow.py OUT [왕복 수=2]      환경 변수 V(기본 0.05), YAW_FIX(기본 −2.2)
결과: OUT/wall_follow.npy(시간, 오도메트리 x, y, yaw, v, w, 왼90, 오른90, 왼45, 오른45)
"""
import math, os, sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

V = float(os.environ.get("V", 0.05)); YAW_FIX = math.radians(float(os.environ.get("YAW_FIX", -2.2)))
K90, K45, W_MAX = 0.18, 1.64, 0.8
RAY_W, CAP, CAP45, END_M, FRONT_M, END_FRONT = math.radians(6), 0.6, 0.45, 0.45, 0.10, 0.25
SMOOTH = 0.6           # 조향 거르기: w = SMOOTH × 직전 w + (1 − SMOOTH) × 새 w
TURN_W, ODO_GAIN = 0.4, 1.087
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi


class Follow:
    def __init__(self):
        self.n = rclpy.create_node("wall_follow"); self.odom = None; self.rays = None; self.rows = []
        self.pub = self.n.create_publisher(Twist, "/cmd_vel_pc", 10)
        self.n.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.n.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

    def on_odom(self, m):
        p = m.pose.pose; self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_scan(self, s):
        r = np.array(s.ranges); a = wrap(s.angle_min + np.arange(len(r)) * s.angle_increment + math.pi + YAW_FIX)   # 로봇 기준 방향
        ok = np.isfinite(r) & (r > max(s.range_min, 0.05)); r, a = r[ok], a[ok]
        ray = lambda deg: float(min(np.median(r[np.abs(wrap(a - math.radians(deg))) < RAY_W]), CAP)) if (np.abs(wrap(a - math.radians(deg))) < RAY_W).any() else CAP
        self.rays = dict(L90=ray(90), R90=ray(-90), L45=min(ray(45), CAP45), R45=min(ray(-45), CAP45), F=ray(0))

    def send(self, v, w):
        t = Twist(); t.linear.x, t.angular.z = float(v), float(w); self.pub.publish(t)

    def step(self):
        rclpy.spin_once(self.n, timeout_sec=0.05)

    def stop(self):
        for _ in range(10):
            self.send(0, 0); self.step()

    def log(self, t0, v, w):
        R = self.rays; self.rows.append([time.monotonic() - t0, *self.odom, v, w, R["L90"], R["R90"], R["L45"], R["R45"]])

    def follow(self, t0):
        """도로 가운데를 따라 끝까지. 돌려주는 값: '끝'(도로 끝에 닿음) 또는 멈춘 까닭."""
        start, went, w = self.odom, 0.0, 0.0
        while rclpy.ok():
            self.step(); R = self.rays
            went = math.hypot(self.odom[0] - start[0], self.odom[1] - start[1])
            if R["F"] < FRONT_M:
                return f"앞 {R['F'] * 1000:.0f} mm에 벽"
            if went > 1.0 and ((R["L90"] > END_M and R["R90"] > END_M) or R["F"] < END_FRONT):
                return "끝"                                              # 옆 벽이 끝남(열린 끝) 또는 앞이 막힘(닫힌 끝)
            new = float(np.clip(K90 * (R["L90"] - R["R90"]) + K45 * (R["L45"] - R["R45"]), -W_MAX, W_MAX))
            w = SMOOTH * w + (1 - SMOOTH) * new
            self.send(V, w); self.log(t0, V, w)
        return "중단"

    def turn(self, t0):
        """제자리 180도(왼쪽). 오도메트리가 회전을 ODO_GAIN배 세므로 오도메트리로 180×ODO_GAIN도 돈다."""
        target = math.pi * ODO_GAIN; done = 0.0; last = self.odom[2]
        while rclpy.ok() and done < target:
            self.send(0, TURN_W); self.step(); self.log(t0, 0, TURN_W)
            done += abs(wrap(self.odom[2] - last)); last = self.odom[2]
        self.stop()


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); loops = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rclpy.init(); f = Follow(); t0 = time.monotonic()
    while (f.odom is None or f.rays is None) and time.monotonic() - t0 < 5:
        f.step()
    if f.odom is None or f.rays is None:
        print("/odom 또는 /scan이 안 들어옴(ROS_DOMAIN_ID, 핑키 bringup 확인)"); return
    R = f.rays; print(f"시작: 왼90 {R['L90']:.3f}, 오른90 {R['R90']:.3f}, 왼45 {R['L45']:.3f}, 오른45 {R['R45']:.3f}, 앞 {R['F']:.3f} m / {V} m/s, 왕복 {loops}번", flush=True)
    t0 = time.monotonic(); result = "끝"
    try:
        for i in range(loops * 2):
            r = f.follow(t0); print(f"  {i // 2 + 1}번째 왕복 {'가는' if i % 2 == 0 else '오는'} 길: {r} ({time.monotonic() - t0:.0f}s)", flush=True)
            if r != "끝":
                result = r; break
            f.stop(); f.turn(t0)
    except KeyboardInterrupt:
        result = "Ctrl+C로 멈춤"
    try:
        f.stop()
    except Exception:
        pass
    A = np.array(f.rows); np.save(out + "/wall_follow.npy", A)
    if len(A):
        mid = A[A[:, 4] > 0]; e = (mid[:, 6] - mid[:, 7]) / 2 * 1000
        print(f"{result}: {time.monotonic() - t0:.0f}s, 달리는 동안 가운데에서 벗어난 정도(90도 광선 기준) 중앙 {np.median(np.abs(e)):.0f} mm, 최대 {np.abs(e).max():.0f} mm → {out}")
    f.n.destroy_node(); rclpy.shutdown()
    sys.exit(0 if result == "끝" else 1)                                 # 실패면 뒤의 명령(&&)이 실행되지 않게


if __name__ == "__main__":
    main()
