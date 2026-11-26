"""SLAM 지도 작성 주행(시뮬): Nav2 없이 길 가운데 선을 따라 천천히 왕복한다. 실물에서는 사람이 같은 길을 천천히 몰면 된다.

출발 쪽 길 가운데에 로봇을 세우고(오도메트리는 (0, 0, 0)에서 시작, 실물 핑키를 켠 것과 같음) → 길 가운데 선을 따라 끝까지 →
제자리에서 180도 돌기 → 되돌아와 출발 쪽까지 → 180도 돌기. 이것을 LOOPS번(기본 2) 한다.
돌기는 길 가운데에서 한다(차로 가운데에서 돌면 몸체 뒤쪽이 벽에 닿는다: 바퀴 축에서 가장 먼 모서리 약 0.14 m, 차로 가운데에서 벽까지 약 0.11 m).
따라가기는 /ground_truth(실제 위치)로 한다. 지도를 만드는 SLAM 쪽은 /odom과 /scan만 쓰므로 결과에는 영향이 없다.
속도 V(0.05 m/s), 회전 W(0.3 rad/s): SLAM 지도가 번지지 않게 느리게(2026-11-26 대화의 기준 범위 안).
결과: OUT/map_drive.npy(시간, 실제 x, y, yaw, 오도메트리 x, y, yaw). 나중에 지도 좌표와 비교할 때 쓴다.
사용: python map_drive.py OUT [--place | --go]        환경 변수 LOOPS(왕복 횟수)
  --place: 출발 자리에 세우기만 하고 끝냄 → SLAM을 켬 → --go: 세우지 않고 지금 자리에서 주행.
  SLAM은 로봇을 세운 뒤에 켜야 한다. 먼저 켜 두면 원점에서 본 스캔이 지도에 남고, 세우는 순간 위치가 0.58 m 뛰어
  지도가 약 10도 돌고 휘었다(2026-11-26 첫 지도: 도면 벽과의 어긋남 95% 17 mm, 최대 39 mm). 실물은 켠 자리에서 바로 SLAM을 켜면 된다.
"""
import math, os, sys
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wall_map as M

V, W, LOOK = 0.05, 0.3, 0.1        # 직진 속도(m/s), 제자리 회전 속도(rad/s), 주시 거리(m)
END_TOL, TURN_TOL = 0.01, math.radians(2)
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi


class Drive(Node):
    def __init__(self):
        super().__init__("map_drive", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10); self.reset_pub = self.create_publisher(PoseStamped, "/lane_sim/reset", 10)
        self.create_subscription(PoseStamped, "/ground_truth", self.on_truth, 50)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.truth = self.odom = None; self.rows = []

    def on_odom(self, m):
        p = m.pose.pose; self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_truth(self, m):
        p = m.pose; self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))
        if self.odom is not None:
            self.rows.append([self.get_clock().now().nanoseconds * 1e-9, *self.truth, *self.odom])

    def send(self, v, w):
        t = Twist(); t.linear.x, t.angular.z = float(v), float(w); self.cmd.publish(t)

    def step(self):
        rclpy.spin_once(self, timeout_sec=0.02)

    def place(self, x, y, yaw):
        m = PoseStamped(); m.header.frame_id = "odom0"; m.pose.position.x, m.pose.position.y = x, y
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        for _ in range(100):                                             # 실제 위치가 그 자리로 올 때까지(최대 약 2초)
            self.reset_pub.publish(m); [self.step() for _ in range(5)]
            if self.truth and math.hypot(self.truth[0] - x, self.truth[1] - y) < 0.01 and self.odom and math.hypot(*self.odom[:2]) < 0.01:
                return True
        return False

    def follow(self, P):
        """점열 P를 끝까지 따라간다(순수 추종). 끝점에 END_TOL 안으로 오면 선다."""
        k = 0
        while rclpy.ok():
            self.step()
            if self.truth is None:
                continue
            x, y, th = self.truth
            if math.hypot(P[-1][0] - x, P[-1][1] - y) < END_TOL:
                break
            d = np.hypot(P[k:, 0] - x, P[k:, 1] - y); k += int(np.argmin(d))
            ahead = np.where(np.hypot(P[k:, 0] - x, P[k:, 1] - y) > LOOK)[0]; g = P[k + ahead[0]] if len(ahead) else P[-1]
            a = wrap(math.atan2(g[1] - y, g[0] - x) - th); L = max(math.hypot(g[0] - x, g[1] - y), 1e-3)
            self.send(V, 2 * V * math.sin(a) / L)
        self.send(0, 0)

    def turn(self, target):
        while rclpy.ok():
            self.step()
            e = wrap(target - self.truth[2])
            if abs(e) < TURN_TOL:
                break
            self.send(0.0, math.copysign(min(W, max(0.1, 2 * abs(e))), e))
        self.send(0, 0); [self.step() for _ in range(25)]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); loops = int(os.environ.get("LOOPS", 2))
    U = np.arange(0.0, M.TOTAL + 1e-9, 0.01); fwd = np.array([M.point(u, 0.0) for u in U]); back = fwd[::-1]
    rclpy.init(); n = Drive()
    while n.truth is None or n.odom is None:
        n.step()
    if "--go" not in sys.argv and not n.place(float(fwd[0, 0]), float(fwd[0, 1]), 0.0):
        print("로봇을 출발 자리에 세우지 못함(브리지가 떠 있는지 확인)"); return
    if "--place" in sys.argv:
        print(f"출발 자리에 세움: 실제 ({n.truth[0]:.3f}, {n.truth[1]:.3f}), 오도메트리 ({n.odom[0]:.3f}, {n.odom[1]:.3f}). 이제 SLAM을 켜고 --go로 주행"); return
    if "--go" in sys.argv and math.hypot(n.truth[0] - fwd[0, 0], n.truth[1] - fwd[0, 1]) > 0.02:
        print("로봇이 출발 자리에 없음(먼저 --place)"); return
    t0 = n.get_clock().now().nanoseconds * 1e-9
    print(f"출발: 길 가운데 ({fwd[0, 0]:.3f}, {fwd[0, 1]:.3f}), 길이 {M.TOTAL:.2f} m, 왕복 {loops}번, {V} m/s", flush=True)
    for i in range(loops):
        n.follow(fwd); n.turn(-math.pi / 2); n.follow(back); n.turn(0.0)
        print(f"왕복 {i + 1} 끝: {n.get_clock().now().nanoseconds * 1e-9 - t0:.0f}s", flush=True)
    R = np.array(n.rows); R[:, 0] -= t0; np.save(out + "/map_drive.npy", R)
    e = np.hypot(R[-1, 1] - fwd[0, 0], R[-1, 2] - fwd[0, 1])
    print(f"끝: {R[-1, 0]:.0f}s, 출발 자리와의 차이 {e * 1000:.0f} mm, 오도메트리 끝 값 ({R[-1, 4]:.3f}, {R[-1, 5]:.3f}, {math.degrees(R[-1, 6]):.1f}도) → {out}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
