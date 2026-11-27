"""AMCL로 위치를 잡고 달리는 주행용 부품(track_runner.py --align ALIGN_JSON 에서 씀).

지도 좌표(map)는 SLAM 지도의 좌표, 도면 좌표는 wall_map.py의 좌표(= 시뮬 세계 좌표). 둘 사이 변환은 map_align.py의 align.json.
  - 그래프(route_nodes.json, lane_graph.geojson)는 map_align.py가 지도 좌표로 옮긴 것을 쓴다.
  - 로봇을 세울 때와 장애물을 놓을 때는 시뮬 세계(도면) 좌표로 바꿔 브리지에 보낸다.
  - 기록의 '실제 위치'는 /ground_truth를 지도 좌표로 바꾼 것, '추정 위치'는 그 순간의 TF map → base_footprint
    (= AMCL이 낸 map → odom + 지금 오도메트리, 컨트롤러가 쓰는 위치와 같음).
    처음에는 /amcl_pose를 썼으나, 그것은 AMCL이 갱신할 때(2 cm마다)만 나와서 주행 중에는 늘 늦은 값이었다(2026-11-26: 오차 중앙 30~40 mm로 크게 나옴).
  - map → odom 변환은 AMCL이 낸다(Runner가 내던 항등 변환은 내지 않음).
실물(real=True, track_runner.py --real): 시뮬 시간을 끄고, 로봇을 옮기지 않으며(사람이 출발 노드 자리에 놓음), 실제 위치가 없으므로 추정 위치만 기록한다.
로봇을 세운 뒤 /initialpose로 AMCL에 그 자리를 알리고, AMCL 추정이 그 자리로 올 때까지 기다린 뒤 로컬 코스트맵을 비운다.
  (시뮬에서 로봇을 순간 이동시키면 AMCL이 따라잡기 전의 스캔이 엉뚱한 자리에 벽으로 남아, 출발 직후 충돌 경고와 경로 다시 구하기가 났다(2026-11-26).
   실물은 순간 이동이 없으므로 이 단계는 시뮬에서만 필요하다.)
"""
import json, math, time
import numpy as np
import rclpy
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.srv import ClearEntireCostmap
from tf2_ros import Buffer, TransformListener
from route_runner import Runner, yaw_of

TOL, YAW_TOL, LIMIT = 0.02, 0.05, 10.0    # 세워진 것으로 볼 거리(m)·방향(rad), 기다리는 실제 시간(초)
COV = (0.02 ** 2, math.radians(3) ** 2)   # 초기 위치를 알릴 때의 분산(위치, 방향): 사람이 표시한 자리에 놓는 정도


class AmclRunner(Runner):
    amcl = True

    def __init__(self, align, real=False):
        a = json.load(open(align))["drawing_to_map"]; self.th, self.t = float(a["theta"]), np.array([a["tx"], a["ty"]], float)
        super().__init__(real); self.amcl_pose, self.real = None, real
        if real:                                                         # 실물: 실제 위치가 없으므로 추정 위치를 두 칸에 모두 적어 기록
            self.create_timer(0.05, self.record)
        self.buf = Buffer(); self.listener = TransformListener(self.buf, self)
        self.clear = self.create_client(ClearEntireCostmap, "local_costmap/clear_entirely_local_costmap")
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_amcl, 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

    def to_map(self, x, y, yaw):
        c, s = math.cos(self.th), math.sin(self.th)
        return np.array([c * x - s * y + self.t[0], s * x + c * y + self.t[1], yaw + self.th])

    def to_world(self, x, y, yaw):
        c, s = math.cos(self.th), math.sin(self.th); dx, dy = x - self.t[0], y - self.t[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, yaw - self.th])

    def on_odom(self, m):
        pass                                                             # 추정 자리(self.odom)에는 AMCL 위치를 둔다

    def on_amcl(self, m):
        p = m.pose.pose; self.amcl_pose = np.array([p.position.x, p.position.y, yaw_of(p.orientation)])

    def record(self):
        self.on_truth(None)
        if self.rows is not None and self.odom is not None:
            self.rows.append([self.get_clock().now().nanoseconds * 1e-9, *self.odom, *self.odom, *self.cmd])

    def on_truth(self, m):
        try:                                                             # 컨트롤러가 쓰는 것과 같은 지금 위치
            t = self.buf.lookup_transform("map", "base_footprint", Time()).transform
            self.odom = np.array([t.translation.x, t.translation.y, yaw_of(t.rotation)])
        except Exception:
            pass
        if m is not None:
            super().on_truth(m)

    def map_odom(self):
        pass

    def initial(self, x, y, yaw):
        m = PoseWithCovarianceStamped(); m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = COV[0]; m.pose.covariance[35] = COV[1]
        self.init_pub.publish(m)


def near(p, q, tol=TOL):
    return p is not None and math.hypot(p[0] - q[0], p[1] - q[1]) < tol and abs((p[2] - q[2] + math.pi) % (2 * math.pi) - math.pi) < YAW_TOL


def place_amcl(n, x, y, yaw):
    """지도 좌표 (x, y, yaw)에 로봇을 세우고 AMCL에 알린다. 실제 위치와 AMCL 추정이 모두 그 자리로 오면 True."""
    q = (x, y, yaw); w = n.to_world(x, y, yaw); t0 = time.monotonic(); sent = None
    if n.real:                                                           # 실물: 사람이 그 자리(바닥 표시)에 놓았다고 보고 AMCL에만 알린다
        n.truth = q
    while time.monotonic() - t0 < LIMIT and not near(n.truth, q):
        if sent is None or time.monotonic() - sent > 0.5:
            n.reset(*w); sent = time.monotonic()
        rclpy.spin_once(n, timeout_sec=0.01)
    if not near(n.truth, q):
        return False
    n.wait(0.5); n.amcl_pose = None; sent = None
    while time.monotonic() - t0 < 2 * LIMIT:
        if sent is None or time.monotonic() - sent > 1.0:
            n.initial(x, y, yaw); sent = time.monotonic()
        rclpy.spin_once(n, timeout_sec=0.01)
        if near(n.amcl_pose, q):
            n.wait(0.5)
            if n.clear.wait_for_service(timeout_sec=2.0):
                f = n.clear.call_async(ClearEntireCostmap.Request()); rclpy.spin_until_future_complete(n, f, timeout_sec=2.0)
            n.wait(0.5); return True
    return False


def est_error(R):
    """기록의 실제 위치와 AMCL 추정의 차이: 글자 한 줄(위치 mm 중앙/95%/최대, 방향 도 최대)."""
    e = np.hypot(R[:, 1] - R[:, 4], R[:, 2] - R[:, 5]) * 1000; a = np.degrees(np.abs((R[:, 3] - R[:, 6] + math.pi) % (2 * math.pi) - math.pi))
    return f"AMCL 위치 오차 mm(중앙/95%/최대) {np.median(e):.1f}/{np.percentile(e, 95):.1f}/{e.max():.1f}, 방향 최대 {a.max():.1f}도"
