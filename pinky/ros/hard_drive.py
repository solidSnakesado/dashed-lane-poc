"""실물 지도 작성용 정해진 명령 주행(하드코딩): 도면의 길 가운데 선을 따라가도록 속도·시간을 미리 정해 보낸다. 센서로 방향을 고치지 않는다.

길 가운데 선: 직선 S → 왼쪽으로 90도(반지름 RC) → 직선 S (wall_map.py 값). 끝에서 제자리 180도 돌고 같은 길을 거꾸로(오른쪽으로 90도) 온다.
명령은 /cmd_vel_pc로 20 Hz(핑키의 cmd_guard.py가 /cmd_vel로 넘김). 로봇은 길 가운데 출발 자리에 굽은 쪽을 보게 놓는다.
안전: 라이다 앞쪽 ±40도 안에 STOP_M보다 가까운 점이 생기면 정지 명령을 보내고 끝낸다(벽을 뚫지 않게). 라이다 앞쪽 = 로봇 기준 앞.
돌기가 덜 되면 W_GAIN(회전 속도 배수, 기본 1)을 올려 본다. 결과: OUT/hard_drive.npy(시간, 오도메트리 x, y, yaw, 보낸 v, w).
사용: python hard_drive.py OUT [왕복 수=1]        환경 변수 W_GAIN, V(기본 0.05 m/s)
"""
import math, os, sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wall_map as M

V = float(os.environ.get("V", 0.05)); W_GAIN = float(os.environ.get("W_GAIN", 1.0))
TURN_W, STOP_M, FRONT = 0.5, 0.06, math.radians(40)     # 제자리 돌기 속도(rad/s), 앞 장애물 정지 거리(m, 로봇 앞머리 아님 라이다 기준 + 앞머리 0.059), 앞쪽 범위
LIDAR_YAW = math.pi                                     # URDF: 라이다를 180도 돌려 붙임
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def plan(loops):
    """(v, w, 초) 목록. 직선 S, 왼쪽 원호(반지름 RC, 90도), 직선 S, 제자리 180도, 거꾸로 같은 길(원호는 오른쪽), 제자리 180도."""
    arc_t = (M.RC * math.pi / 2) / V; w_arc = V / M.RC; turn_t = math.pi / TURN_W; seg = []
    for _ in range(loops):
        seg += [(V, 0.0, M.S / V), (V, w_arc, arc_t), (V, 0.0, M.S / V), (0.0, TURN_W, turn_t),
                (V, 0.0, M.S / V), (V, -w_arc, arc_t), (V, 0.0, M.S / V), (0.0, -TURN_W, turn_t)]
    return seg


class Drive:
    def __init__(self):
        self.n = rclpy.create_node("hard_drive"); self.odom = None; self.front = math.inf; self.rows = []
        self.pub = self.n.create_publisher(Twist, "/cmd_vel_pc", 10)
        self.n.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.n.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

    def on_odom(self, m):
        p = m.pose.pose; self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_scan(self, s):
        r = np.array(s.ranges); a = s.angle_min + np.arange(len(r)) * s.angle_increment + LIDAR_YAW
        a = (a + math.pi) % (2 * math.pi) - math.pi; ok = np.isfinite(r) & (r > s.range_min) & (np.abs(a) < FRONT)
        x = r[ok] * np.cos(a[ok]) - 0.017                               # 바퀴 축 기준 앞쪽 거리
        self.front = float(np.min(x - 0.042)) if ok.any() else math.inf  # 로봇 앞머리(바퀴 축 앞 0.042 m)에서 잰 거리

    def send(self, v, w):
        t = Twist(); t.linear.x, t.angular.z = float(v), float(w); self.pub.publish(t)


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); loops = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rclpy.init(); d = Drive(); t0 = time.monotonic()
    while d.odom is None and time.monotonic() - t0 < 5:
        rclpy.spin_once(d.n, timeout_sec=0.05)
    if d.odom is None:
        print("/odom이 안 들어옴(ROS_DOMAIN_ID, 핑키 bringup 확인)"); return
    seg = plan(loops); print(f"정해진 명령 {len(seg)}구간, 모두 {sum(s[2] for s in seg):.0f}초, {V} m/s, 회전 배수 {W_GAIN}", flush=True)
    done = "끝"; t0 = time.monotonic()
    try:
        for i, (v, w, sec) in enumerate(seg):
            print(f"  구간 {i + 1}: v {v:.3f}, w {w * W_GAIN:+.3f}, {sec:.1f}초", flush=True); ts = time.monotonic()
            while time.monotonic() - ts < sec:
                d.send(v, w * W_GAIN); rclpy.spin_once(d.n, timeout_sec=0.05)
                d.rows.append([time.monotonic() - t0, *d.odom, v, w * W_GAIN])
                if v > 0 and d.front < STOP_M:
                    done = f"앞 {d.front * 1000:.0f} mm에 벽 → 구간 {i + 1}에서 멈춤"; raise StopIteration
    except (StopIteration, KeyboardInterrupt) as e:
        done = done if isinstance(e, StopIteration) else "Ctrl+C로 멈춤"
    for _ in range(10):
        d.send(0, 0); time.sleep(0.05)
    np.save(out + "/hard_drive.npy", np.array(d.rows))
    print(f"{done}: {time.monotonic() - t0:.0f}s, 오도메트리 끝 값 ({d.odom[0]:.3f}, {d.odom[1]:.3f}, {math.degrees(d.odom[2]):.1f}도) → {out}")
    d.n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
