"""출발지 → 목적지 주행(ROS2): 찍은 출발지에 로봇을 세우고 → Route Server에 경로를 요청하고 → 경로가 지나는 정지선마다 잘라서
정지선 앞까지 Controller Server(RPP)로 주행 → 멈춰서 WAIT초 → 다음 구간 … → 목적지. 위치는 오도메트리만 쓴다(map→odom 항등).
로봇 쪽(lane_sim_bridge.py)과 제어 쪽(Route Server, Controller Server, 이 스크립트)은 ROS2 토픽·액션으로만 주고받는다.
결과: 쌍마다 NN.npy(시간, 실제 x, y, yaw, 추정 x, y, yaw, v, w), NN_path.npy, summary.txt(도착, 차로 밖 줄 수, 정지선마다 멈춘 위치).
사용(젯슨, ~/venvs/lane 파이썬): python goal_runner.py NODES_JSON OUT 출발 목적지 [출발 목적지 ...]
  출발·목적지 = 노드 번호 또는 x,y 또는 x,y,방향(도).  환경 변수 WAIT(정지 시간 초, 기본 1.0)
"""
import math, os, sys, time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputeRoute, FollowPath
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from route_runner import Runner
import goal_plan as G

PLACE_TOL = 0.02       # 리셋 뒤 실제 위치·오도메트리가 출발지에서 이 거리(m) 안이면 세워진 것으로 본다
SLOW = 0.05            # 시간 한도 계산에 쓰는 느린 속도(m/s): 구간 길이/이 값 + 20초


def to_msg(node, seg):
    """점열을 nav_msgs/Path로. 점마다 방향은 가는 방향(마지막 점은 직전 방향)."""
    m = Path(); m.header.frame_id = "map"; m.header.stamp = node.get_clock().now().to_msg()
    d = np.diff(seg, axis=0); ang = np.arctan2(d[:, 1], d[:, 0]); ang = np.r_[ang, ang[-1]]
    for (x, y), a in zip(seg, ang):
        p = PoseStamped(); p.header = m.header; p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(a / 2), math.cos(a / 2); m.poses.append(p)
    return m


def place(n, x, y, yaw, limit=8.0):
    """로봇을 출발 자세에 세우고, 실제 위치와 오도메트리가 모두 그 자리로 온 것을 확인한 뒤 돌아온다(실제 시간 limit초까지).
    한동안 spin하지 않은 노드는 시뮬 시계와 받은 메시지가 밀려 있어서, 시뮬 시간으로 1초 기다리는 것만으로는 리셋이 반영되기 전에 넘어갈 수 있다."""
    t0 = time.monotonic(); sent = None
    near = lambda p: p is not None and math.hypot(p[0] - x, p[1] - y) < PLACE_TOL and abs((p[2] - yaw + math.pi) % (2 * math.pi) - math.pi) < 0.1
    while time.monotonic() - t0 < limit:
        if sent is None or (time.monotonic() - sent > 0.5 and not near(n.truth)):   # 한 번은 꼭 보내고, 반영될 때까지 0.5초마다 다시 보냄
            n.reset(x, y, yaw); sent = time.monotonic()
        rclpy.spin_once(n, timeout_sec=0.01)
        if near(n.truth) and near(n.odom) and time.monotonic() - sent > 0.3:
            n.wait(1.0); return near(n.truth) and near(n.odom)                 # 시계가 맞춰진 뒤 시뮬 시간으로 1초 더
    return False


def drive_pair(n, nodes, a, b, wait):
    """노드 a → b를 달린다. 돌려주는 값: (경로, 정지선 목록, 기록 R, 결과 글자). 경로를 못 받으면 경로가 None."""
    if not place(n, float(nodes.xy[a, 0]), float(nodes.xy[a, 1]), float(nodes.yaw[a])):
        return None, [], None, "로봇을 출발지에 세우지 못함(브리지가 떠 있는지 확인)"
    g = ComputeRoute.Goal(); g.start_id, g.goal_id, g.use_start, g.use_poses = a, b, False, False
    res = n.call(n.route_ac, g)
    if res is None or res.error_code or len(res.path.poses) < 2:
        return None, [], None, f"경로 요청 실패 ({None if res is None else res.error_code})"
    path = np.array([[p.pose.position.x, p.pose.position.y] for p in res.path.poses])
    stops = G.stops_on(path, nodes); segs = G.cut(path, [s for s, _ in stops])
    print(f"노드 {a} → {b}: 경로 {G.arc(path)[-1]:.2f} m, 정지선 {len(stops)}곳, 구간 {len(segs)}개", flush=True)
    n.path = path                                                        # 화면에 그리는 쪽(goal_click.py)에서 씀
    n.rows = []; t0 = n.get_clock().now().nanoseconds * 1e-9; done = "도착"
    for k, seg in enumerate(segs):
        fg = FollowPath.Goal(); fg.path = to_msg(n, seg); fg.controller_id, fg.goal_checker_id = "FollowPath", "general_goal_checker"
        n.t_start = n.get_clock().now().nanoseconds * 1e-9
        fr = n.call(n.follow_ac, fg, timeout=G.arc(seg)[-1] / SLOW + 20.0)
        if fr is None or isinstance(fr, str) or getattr(fr, "error_code", 1) != 0:
            done = f"미도착(구간 {k}: {fr if fr is None or isinstance(fr, str) else fr.error_code})"; break
        n.wait(wait if k < len(segs) - 1 else 0.5)                       # 정지선 앞에서 멈춤(마지막은 도착 후 잠깐)
    rows, n.rows = n.rows, None
    R = np.array(rows) if len(rows) > 1 else None
    if R is not None:
        R[:, 0] -= t0
    return path, stops, R, done


def main():
    nodes, out = G.Nodes(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
    wait = float(os.environ.get("WAIT", 1.0))
    rclpy.init(); n = Runner()
    n.route_ac.wait_for_server(); n.follow_ac.wait_for_server(); n.map_odom(); n.wait(1.0)
    lines = []
    for i, (ta, tb) in enumerate(zip(sys.argv[3::2], sys.argv[4::2])):
        a, b = nodes.pick(ta), nodes.pick(tb)
        path, stops, R, done = drive_pair(n, nodes, a, b, wait)
        if R is None:
            lines.append(f"{i} 노드 {a} → {b}: {done}" + ("" if path is None else ", 기록 없음")); print(lines[-1]); continue
        np.save(f"{out}/{i:02d}.npy", R); np.save(f"{out}/{i:02d}_path.npy", path)
        lines.append(G.report(i, a, b, path, R, done, nodes, stops)); print(lines[-1], flush=True)
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
