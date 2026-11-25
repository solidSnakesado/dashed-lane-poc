"""출발지 → 목적지 주행(ROS2): 찍은 출발지에 로봇을 세우고 → Route Server에 경로를 요청하고 → 경로가 지나는 정지선마다 잘라서
정지선 앞까지 Controller Server(RPP)로 주행 → 멈춰서 WAIT초 → 다음 구간 … → 목적지. 위치는 오도메트리만 쓴다(map→odom 항등).
로봇 쪽(lane_sim_bridge.py)과 제어 쪽(Route Server, Controller Server, 이 스크립트)은 ROS2 토픽·액션으로만 주고받는다.
결과: 쌍마다 NN.npy(시간, 실제 x, y, yaw, 추정 x, y, yaw, v, w), NN_path.npy, summary.txt(도착, 차로 밖 줄 수, 정지선마다 멈춘 위치).
사용(젯슨, ~/venvs/lane 파이썬): python goal_runner.py NODES_JSON OUT 출발 목적지 [출발 목적지 ...]
  출발·목적지 = 노드 번호 또는 x,y 또는 x,y,방향(도).  환경 변수 WAIT(정지 시간 초, 기본 1.0)
"""
import math, os, sys
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputeRoute, FollowPath
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from route_runner import Runner
import goal_plan as G

SLOW = 0.05            # 시간 한도 계산에 쓰는 느린 속도(m/s): 구간 길이/이 값 + 20초


def to_msg(node, seg):
    """점열을 nav_msgs/Path로. 점마다 방향은 가는 방향(마지막 점은 직전 방향)."""
    m = Path(); m.header.frame_id = "map"; m.header.stamp = node.get_clock().now().to_msg()
    d = np.diff(seg, axis=0); ang = np.arctan2(d[:, 1], d[:, 0]); ang = np.r_[ang, ang[-1]]
    for (x, y), a in zip(seg, ang):
        p = PoseStamped(); p.header = m.header; p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(a / 2), math.cos(a / 2); m.poses.append(p)
    return m


def main():
    nodes, out = G.Nodes(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
    wait = float(os.environ.get("WAIT", 1.0))
    rclpy.init(); n = Runner()
    n.route_ac.wait_for_server(); n.follow_ac.wait_for_server(); n.map_odom(); n.wait(1.0)
    lines = []
    for i, (ta, tb) in enumerate(zip(sys.argv[3::2], sys.argv[4::2])):
        a, b = nodes.pick(ta), nodes.pick(tb)
        n.reset(float(nodes.xy[a, 0]), float(nodes.xy[a, 1]), float(nodes.yaw[a])); n.wait(1.0)
        g = ComputeRoute.Goal(); g.start_id, g.goal_id, g.use_start, g.use_poses = a, b, False, False
        res = n.call(n.route_ac, g)
        if res is None or res.error_code or len(res.path.poses) < 2:
            lines.append(f"{i} 노드 {a} → {b}: 경로 요청 실패 ({None if res is None else res.error_code})"); print(lines[-1]); continue
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in res.path.poses])
        stops = G.stops_on(path, nodes); segs = G.cut(path, [s for s, _ in stops])
        print(f"{i} 노드 {a} → {b}: 경로 {G.arc(path)[-1]:.2f} m, 정지선 {len(stops)}곳, 구간 {len(segs)}개", flush=True)
        n.rows = []; t0 = n.get_clock().now().nanoseconds * 1e-9; done = "도착"
        for k, seg in enumerate(segs):
            fg = FollowPath.Goal(); fg.path = to_msg(n, seg); fg.controller_id, fg.goal_checker_id = "FollowPath", "general_goal_checker"
            n.t_start = n.get_clock().now().nanoseconds * 1e-9
            fr = n.call(n.follow_ac, fg, timeout=G.arc(seg)[-1] / SLOW + 20.0)
            if fr is None or isinstance(fr, str) or getattr(fr, "error_code", 1) != 0:
                done = f"미도착(구간 {k}: {fr if fr is None or isinstance(fr, str) else fr.error_code})"; break
            n.wait(wait if k < len(segs) - 1 else 0.5)                   # 정지선 앞에서 멈춤(마지막은 도착 후 잠깐)
        rows, n.rows = n.rows, None
        if len(rows) < 2:
            lines.append(f"{i} 노드 {a} → {b}: {done}, 기록 없음"); print(lines[-1]); continue
        R = np.array(rows); R[:, 0] -= t0
        np.save(f"{out}/{i:02d}.npy", R); np.save(f"{out}/{i:02d}_path.npy", path)
        lines.append(G.report(i, a, b, path, R, done, nodes, stops)); print(lines[-1], flush=True)
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
