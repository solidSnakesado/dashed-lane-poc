"""실물 핑키 점검: 라이다 장착 각도 오차 재기 + 명령 끊김 정지 확인(제어 PC에서 실행, 핑키에서 cmd_guard.py가 떠 있어야 함).

로봇을 블럭 길의 첫 직선 구간에 굽은 쪽을 보게 놓고 실행한다(벽과 정확히 나란할 필요 없음).
  1) 서 있는 채로 /scan(고치지 않은 원래 스캔)에서 좌우 벽을 직선으로 맞춘다 → 벽까지 거리 dL0, dR0, 스캔에서 본 벽 기울기 φ.
  2) /cmd_vel_pc로 V m/s 곧게 T초 보낸 뒤 명령을 그냥 끊는다(정지 명령을 보내지 않음).
     cmd_guard.py가 0.3초 안에 세워야 한다. 마지막 명령 뒤 오도메트리로 더 간 거리를 잰다.
  3) 다시 벽을 맞춘다 → dL1, dR1. 달린 거리 D(오도메트리)로 로봇이 벽에 대해 비스듬한 각 θ = asin((dL0 − dL1)/D) (오른쪽 벽도 같이 봄).
  4) 라이다가 URDF보다 δ만큼 돌아 붙어 있으면 스캔에서 본 벽 기울기는 φ = −θ − δ → δ = −φ − θ. scan_fix.py에 줄 값이 δ(도).
벽 맞추기: 로봇 기준 앞뒤 FIT_X 안, 옆 0.08~0.35 m 안의 점(왼쪽 y > 0, 오른쪽 y < 0)을 직선 y = a x + b로 맞춘다.
사용: python lidar_check.py [V(m/s)=0.05] [T(초)=6]
      python lidar_check.py --turn   제자리 돌기 시험: 오도메트리가 센 회전과 라이다 벽으로 본 실제 회전을 비교(회전 방향·회전량 확인)
"""
import math, sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

LIDAR_X, LIDAR_YAW = -0.017, math.pi     # URDF: 바퀴 축 기준 라이다 위치와 방향(180도 돌려 붙임)
FIT_X, NEAR, FAR = 0.25, 0.08, 0.35
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Check:
    def __init__(self):
        self.n = rclpy.create_node("lidar_check"); self.scan = self.odom = None
        self.pub = self.n.create_publisher(Twist, "/cmd_vel_pc", 10)
        self.n.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.n.create_subscription(Odometry, "/odom", self.on_odom, 50)

    def on_odom(self, m):
        p = m.pose.pose; self.odom = (time.monotonic(), p.position.x, p.position.y, yaw_of(p.orientation))

    def wait_odom(self, limit=5.0):
        """노드를 찾는 데 1초 넘게 걸릴 때가 있어(2026-11-26) /odom이 올 때까지 limit초 기다린다."""
        t0 = time.monotonic()
        while self.odom is None and time.monotonic() - t0 < limit:
            rclpy.spin_once(self.n, timeout_sec=0.05)
        self.spin(0.5)

    def spin(self, sec):
        t0 = time.monotonic()
        while time.monotonic() - t0 < sec:
            rclpy.spin_once(self.n, timeout_sec=0.01)

    def walls(self, k=20):
        """스캔 k장을 모아 좌우 벽 직선을 맞춘다. 돌려주는 값: (왼쪽 (기울기 rad, 거리 m, 점 수), 오른쪽 (…))."""
        P = []
        for _ in range(k):
            self.scan = None
            while self.scan is None:
                rclpy.spin_once(self.n, timeout_sec=0.05)
            s = self.scan; r = np.array(s.ranges); a = s.angle_min + np.arange(len(r)) * s.angle_increment
            ok = np.isfinite(r) & (r > s.range_min) & (r < s.range_max); r, a = r[ok], a[ok] + LIDAR_YAW
            P.append(np.c_[LIDAR_X + r * np.cos(a), r * np.sin(a)])
        P = np.vstack(P); out = []
        for side in (1, -1):
            m = (np.abs(P[:, 0]) < FIT_X) & (side * P[:, 1] > NEAR) & (side * P[:, 1] < FAR); Q = P[m]
            if len(Q) < 30:
                out.append(None); continue
            for _ in range(2):                                           # 크게 벗어난 점(이음매, 반사)을 빼고 다시 맞춤
                A = np.c_[Q[:, 0], np.ones(len(Q))]; (sl, b), *_ = np.linalg.lstsq(A, Q[:, 1], rcond=None)
                e = Q[:, 1] - (sl * Q[:, 0] + b); Q = Q[np.abs(e) < max(0.005, 3 * np.std(e))]
            out.append((math.atan(sl), abs(b) * math.cos(math.atan(sl)), len(Q), float(np.std(e))))
        return out

    def drive(self, v, sec):
        """v m/s로 sec초 명령을 보내고 그냥 끊는다. 돌려주는 값: (마지막 명령 때 오도메트리, 2초 뒤 오도메트리, 마지막 명령 뒤 처음 멈춘 시각(초))."""
        t0 = time.monotonic(); cmd = Twist(); cmd.linear.x = v
        while time.monotonic() - t0 < sec:
            self.pub.publish(cmd); self.spin(0.05)
        last = self.odom; t_last = time.monotonic(); prev = last; stop = None
        while time.monotonic() - t_last < 2.0:
            self.spin(0.02); o = self.odom
            if stop is None and o is not prev and math.hypot(o[1] - prev[1], o[2] - prev[2]) / max(o[0] - prev[0], 1e-3) < 0.005:
                stop = o[0] - t_last
            prev = o
        return last, self.odom, stop


def show(tag, w):
    for name, x in zip(("왼쪽", "오른쪽"), w):
        print(f"  {tag} {name} 벽: " + ("점이 모자람(벽이 안 보임)" if x is None else f"거리 {x[1] * 1000:.1f} mm, 기울기 {math.degrees(x[0]):+.2f}도, 점 {x[2]}개, 맞춤 흔들림 {x[3] * 1000:.1f} mm"))


def turn_test(c, w=0.3, sec=1.5):
    """제자리 돌기: +w로 sec초 → 멈춤 → −w로 sec초. 오도메트리가 센 회전과 라이다로 본 벽 기울기 변화(= 실제 회전, 부호 반대)를 비교한다."""
    stop = lambda: [c.pub.publish(Twist()) or c.spin(0.05) for _ in range(10)]
    res = []; walls = [c.walls()]; yaws = [c.odom[3]]
    for sign in (1, -1):
        cmd = Twist(); cmd.angular.z = sign * w; t0 = time.monotonic()
        while time.monotonic() - t0 < sec:
            c.pub.publish(cmd); c.spin(0.05)
        stop(); c.spin(0.5); walls.append(c.walls()); yaws.append(c.odom[3])
    for i, sign in enumerate((1, -1)):
        a, b = walls[i], walls[i + 1]
        if None in a or None in b:
            print(f"  돌기 {i + 1}: 벽이 안 보여 못 잼"); continue
        lid = -((b[0][0] - a[0][0]) + (b[1][0] - a[1][0])) / 2; odo = (yaws[i + 1] - yaws[i] + math.pi) % (2 * math.pi) - math.pi
        res.append((odo, lid))
        print(f"  돌기 {i + 1}: 명령 {'+' if sign > 0 else '-'}{w} rad/s({'왼쪽, 위에서 보아 반시계' if sign > 0 else '오른쪽, 위에서 보아 시계'}) {sec}초 "
              f"→ 오도메트리 {math.degrees(odo):+.1f}도, 라이다로 본 실제 {math.degrees(lid):+.1f}도, 명령대로라면 {math.degrees(sign * w * sec):+.1f}도")
    if res:
        r = np.mean([o / l for o, l in res if abs(l) > 0.05])
        print(f"오도메트리 / 실제 = {r:.3f} (1이면 맞음. 음수면 방향이 반대, 1보다 크면 오도메트리가 회전을 많이 셈 → 실제 바퀴 간격이 설정보다 넓음)")


def main():
    if "--turn" in sys.argv:
        rclpy.init(); c = Check(); c.wait_odom()
        if c.odom is None:
            print("/odom이 안 들어옴(ROS_DOMAIN_ID, 핑키 bringup 확인)"); return
        turn_test(c); c.n.destroy_node(); rclpy.shutdown(); return
    v = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05; sec = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    rclpy.init(); c = Check(); c.wait_odom()
    if c.odom is None:
        print("/odom이 안 들어옴(ROS_DOMAIN_ID, 핑키 bringup 확인)"); return
    w0 = c.walls(); show("처음", w0)
    if None in w0:
        print("벽이 양쪽 다 보여야 함. 로봇 자리를 확인"); return
    print(f"{v} m/s로 {sec}초 앞으로, 그다음 명령을 끊음", flush=True)
    o0 = c.odom; last, end, stop = c.drive(v, sec); c.spin(0.5); w1 = c.walls(); show("나중", w1)
    D = math.hypot(last[1] - o0[1], last[2] - o0[2]); coast = math.hypot(end[1] - last[1], end[2] - last[2])
    print(f"달린 거리 {D * 1000:.0f} mm(오도메트리), 명령을 끊은 뒤 더 간 거리 {coast * 1000:.1f} mm, 멈춘 때 {'못 잼' if stop is None else f'{stop:.2f}초 뒤'}, "
          f"방향 변화 {math.degrees(end[3] - o0[3]):+.2f}도")
    print("  명령 끊김 정지: " + ("됨" if coast < v * 0.6 else "안 된 것으로 보임(cmd_guard.py 확인)") + f" (기준: {v * 0.6 * 1000:.0f} mm 안)")
    if None in w1 or D < 0.1:
        print("다시 맞추기에 실패해 각도를 구하지 못함"); return
    th = [math.asin(max(-1, min(1, (w0[0][1] - w1[0][1]) / D))), math.asin(max(-1, min(1, (w1[1][1] - w0[1][1]) / D)))]
    phi = (w0[0][0] + w0[1][0] + w1[0][0] + w1[1][0]) / 4; theta = sum(th) / 2; delta = -phi - theta
    print(f"로봇이 벽에 대해 비스듬한 각: 왼쪽 벽 {math.degrees(th[0]):+.2f}도, 오른쪽 벽 {math.degrees(th[1]):+.2f}도 → {math.degrees(theta):+.2f}도")
    print(f"스캔에서 본 벽 기울기(평균) {math.degrees(phi):+.2f}도 → 라이다 장착 각도 오차 δ = {math.degrees(delta):+.2f}도 (문서의 1호기 값 −2.2도)")
    print(f"길 폭(벽 사이, 처음/나중): {(w0[0][1] + w0[1][1]) * 1000:.0f} / {(w1[0][1] + w1[1][1]) * 1000:.0f} mm (도면 430)")
    c.n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
