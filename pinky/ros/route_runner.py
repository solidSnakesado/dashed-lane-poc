"""1-B 물리 주행 시험 실행·기록: 경로마다 로봇을 출발 자세에 리셋(map→odom은 항등) → Route Server에 경로 요청(노드 번호)
→ 받은 dense path를 Controller Server(RPP)에 FollowPath로 보냄 → 도착할 때까지 실제 위치(/ground_truth)와 /odom을 기록.
위치는 오도메트리만 씀(지도 위 출발 자세는 정확히 안다고 둠, A단계의 LOC=odom과 같은 조건).
결과: 경로마다 NN.npy(phys_drive.py와 같은 열), summary.txt.
사용(젯슨, ~/venvs/lane 파이썬): python route_runner.py NODES_JSON OUT [경로 번호 ...]
"""
import json, math, os, sys, time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputeRoute, FollowPath
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from scipy.spatial import cKDTree

SLACK = 20.0           # 주행 시간 한도 = 경로 길이/0.2 m/s + 이 값(시뮬 초)
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Runner(Node):
    def __init__(self, real=False):                                      # real: 실물(시뮬 시간 끔)
        super().__init__("lane_route_runner", parameter_overrides=[Parameter("use_sim_time", value=not real)])
        self.route_ac = ActionClient(self, ComputeRoute, "compute_route")
        self.follow_ac = ActionClient(self, FollowPath, "follow_path")
        self.reset_pub = self.create_publisher(PoseStamped, "/lane_sim/reset", 10)
        self.tf = StaticTransformBroadcaster(self)
        self.create_subscription(PoseStamped, "/ground_truth", self.on_truth, 50)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 50)
        self.truth = self.odom = None; self.cmd = (0.0, 0.0); self.rows = None

    def to_map(self, x, y, yaw):
        """world·odom·map이 모두 같은 좌표(리셋할 때 로봇과 오도메트리를 지도 위 출발 자세에 둠)."""
        return np.array([x, y, yaw])

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)

    def on_odom(self, m):
        p = m.pose.pose; self.odom = self.to_map(p.position.x, p.position.y, yaw_of(p.orientation))

    def on_truth(self, m):
        p = m.pose; self.truth = self.to_map(p.position.x, p.position.y, yaw_of(p.orientation))
        if self.rows is not None and self.odom is not None:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.rows.append([t, *self.truth, *self.odom, *self.cmd])

    def map_odom(self):
        """map→odom 항등 변환(한 번만)."""
        tf = TransformStamped(); tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = "map", "odom"; tf.transform.rotation.w = 1.0
        self.tf.sendTransform(tf)

    def reset(self, x, y, yaw):
        m = PoseStamped(); m.header.frame_id = "map"
        m.pose.position.x, m.pose.position.y = x, y
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.reset_pub.publish(m)

    def wait(self, sec):
        """시뮬 시간으로 sec초 기다린다."""
        t0 = self.get_clock().now()
        while (self.get_clock().now() - t0).nanoseconds * 1e-9 < sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def call(self, client, goal, timeout=None):
        f = client.send_goal_async(goal); rclpy.spin_until_future_complete(self, f)
        h = f.result()
        if not h.accepted:
            return None
        r = h.get_result_async(); t0 = time.monotonic()
        while not r.done():
            rclpy.spin_once(self, timeout_sec=0.01)
            if timeout and (self.get_clock().now().nanoseconds * 1e-9 - self.t_start) > timeout:
                h.cancel_goal_async(); self.wait(0.5); return "시간 초과"
        return r.result().result


def lateral(path, xy):
    a, b = path[:-1], path[1:]; v = b - a
    t = np.clip(np.einsum("ij,ij->i", xy - a, v) / (np.einsum("ij,ij->i", v, v) + 1e-12), 0, 1)
    c = a + t[:, None] * v; d = np.linalg.norm(xy - c, axis=1); k = int(np.argmin(d))
    return float(np.sign(v[k, 0] * (xy[1] - a[k, 1]) - v[k, 1] * (xy[0] - a[k, 0])) * d[k]), c[k]


def main():
    nodes, out = json.load(open(sys.argv[1])), sys.argv[2]; os.makedirs(out, exist_ok=True)
    todo = [int(v) for v in sys.argv[3:]] or range(len(nodes["routes"]))
    xy, yaw, zone = np.array(nodes["xy"]), np.array(nodes["yaw"]), np.array(nodes["zone"]); tree = cKDTree(xy)
    rclpy.init(); n = Runner()
    n.route_ac.wait_for_server(); n.follow_ac.wait_for_server(); n.map_odom(); n.wait(1.0)
    lines = []
    for j in todo:
        r = nodes["routes"][j]
        n.reset(float(xy[r["start"], 0]), float(xy[r["start"], 1]), float(yaw[r["start"]])); n.wait(1.0)
        g = ComputeRoute.Goal(); g.start_id, g.goal_id, g.use_start, g.use_poses = r["start"], r["goal"], False, False
        res = n.call(n.route_ac, g)
        if res is None or res.error_code or len(res.path.poses) < 2:
            lines.append(f"{j:2d} {r['name']}: 경로 요청 실패 ({None if res is None else res.error_code})"); print(lines[-1]); continue
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in res.path.poses])
        L = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        fg = FollowPath.Goal(); fg.path = res.path; fg.controller_id, fg.goal_checker_id = "FollowPath", "general_goal_checker"
        n.rows = []; n.t_start = n.get_clock().now().nanoseconds * 1e-9
        fr = n.call(n.follow_ac, fg, timeout=L / 0.2 + SLACK)
        rows, n.rows = n.rows, None
        R = []
        for t, x, y, a, ox, oy, oa, v, w in rows:                          # phys_drive.py와 같은 열
            e, c = lateral(path, np.array([x, y])); k = tree.query(c)[1]
            R.append([t - n.t_start, x, y, a, ox, oy, oa, v, w, e, int(zone[k]), math.hypot(ox - x, oy - y),
                      (oa - a + math.pi) % (2 * math.pi) - math.pi])
        R = np.array(R); np.save(f"{out}/{j:02d}.npy", R); np.save(f"{out}/{j:02d}_path.npy", path)
        ok = "도착" if fr is not None and fr != "시간 초과" and getattr(fr, "error_code", 1) == 0 else f"미도착({fr if isinstance(fr, str) else getattr(fr, 'error_code', '?')})"
        end = float(np.linalg.norm(R[-1, 1:3] - path[-1])) if len(R) else float("nan")
        e, z = np.abs(R[:, 9]) * 1000, R[:, 10] > 0
        part = lambda m: f"{np.median(e[m]):.1f}/{np.percentile(e[m], 95):.1f}/{e[m].max():.1f}" if m.any() else "-"
        lines.append(f"{j:2d} {r['name']:12s} {r['kind']:8s} {ok} {R[-1, 0]:5.1f}s 끝 거리 {end * 1000:.0f} mm | 경로에서 벗어난 거리 mm(중앙/95%/최대) "
                     f"교차로 밖 {part(~z)} 안 {part(z)} | 차로 반폭 넘음 {int((e > r['half'] * 1000).sum())} | 오도메트리 오차 끝 {R[-1, 11] * 1000:.1f} mm")
        print(lines[-1], flush=True)
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
