"""막힌 차로를 피해 가는 주행(ROS2): Route Server의 compute_and_track_route로 경로를 받고, 주행 중 경로가 바뀌면 그때마다 RPP에 다시 보낸다.

흐름: 출발지에 로봇을 세움 → (장애물을 놓음) → compute_and_track_route 요청 → 피드백의 rerouted가 참일 때마다 그 경로를 follow_path로 보냄
      → 경로가 다시 구해지는 경우: Route Server의 CollisionMonitor가 앞쪽 경로에서 막힌 엣지를 찾았을 때(라이다),
        또는 밖에서 도로 사정을 알렸을 때(road_event.py: 닫기, 열기, 비용 → 다시 구하기 요청).
      길이 없거나 컨트롤러가 주행을 끊으면 멈추고, RETRY초마다 지금 자리에서 경로를 다시 요청한다. 길이 나오면 다시 출발하고, HOLD초가 지나면 끝낸다.
      주행이 끝나면(어떻게 끝나든) 이 스크립트가 알렸던 닫기·비용은 모두 다시 열어 둔다.
      (새로 요청하면 Route Server가 라이다로 찾아 쌓아 둔 막힌 엣지 목록도 처음부터 다시 시작한다.)
위치는 오도메트리만 쓴다(map→odom 항등). --align ALIGN_JSON을 주면 AMCL 위치로 달린다(amcl_runner.py, 그래프는 map_align.py가 옮긴 것). 정지선 처리는 하지 않는다(wall_map.py의 길에는 정지선이 없음).
결과: NN.npy(시간, 실제 x, y, yaw, 추정 x, y, yaw, v, w), NN_paths.npz(받은 경로들), summary.txt.
사용: python track_runner.py NODES_JSON OUT [--align ALIGN_JSON] 출발 목적지 [b… | e…]... [출발 목적지 ...]      (lane_graph.geojson은 NODES_JSON과 같은 폴더)
  출발·목적지 = 노드 번호.
  b노드 = 장애물이 처음부터 그 노드 자리에 있음.  b노드@거리 = 로봇이 그 거리(m) 안으로 오면 나타남.
  e종류:A-B@시각 = 출발 뒤 그 시각(초)에 도로 사정을 알림. 종류 = close, open, cost(이때는 A-B=값). 여러 개 가능.
  예: python track_runner.py route_nodes.json out 0 232 b120   0 232 eclose:100-140@2 eopen:100-140@6
"""
import json, math, os, sys
import numpy as np
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputeAndTrackRoute, FollowPath
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from route_runner import Runner
from goal_runner import place
from road_event import Road, parse
from amcl_runner import AmclRunner, place_amcl, est_error
import goal_plan as G

BOX = (0.12, 0.12)     # 장애물 상자의 앞뒤, 좌우 길이(m). wall_map.py의 OBSTACLE과 같아야 함
BODY = 0.058           # 로봇 몸체 반폭(m). 장애물과의 여유를 이 값만큼 빼서 본다
SLOW = 0.05            # 시간 한도 = 차로 0의 길이/이 속도 + 30초(시뮬 초)
RETRY, HOLD = 1.0, 30.0   # 길이 없어 멈췄을 때: 이 간격(초)으로 다시 요청하고, 이 시간(초)이 지나도 안 열리면 끝냄
CONFIRM = 0.5          # 멈춘 뒤 다시 받은 경로가 이 시간(초) 동안 막히지 않아야 출발. 새 요청은 Route Server가 라이다로 찾은 막힘을
                       # 잊고 시작하므로, 바로 출발하면 장애물로 다시 들어갔다 멈추기를 되풀이했다(2026-11-26, 0.25 m 앞 장애물 70초)
OK = 4                 # action_msgs/GoalStatus.STATUS_SUCCEEDED


def box_dist(R, c, yaw):
    """기록의 실제 위치마다 장애물 상자 겉면까지 거리(m). 상자 안이면 0."""
    d = R[:, 1:3] - c; u = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw); v = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
    return np.hypot(np.maximum(np.abs(u) - BOX[0] / 2, 0), np.maximum(np.abs(v) - BOX[1] / 2, 0))


def obstacle(n, pub, pose=None):
    m = PoseStamped(); m.header.frame_id = "off" if pose is None else "map"; m.pose.orientation.w = 1.0
    if pose is not None and getattr(n, "amcl", False):
        pose = n.to_world(*pose)                                         # 브리지는 시뮬 세계(도면) 좌표로 받는다
    if pose is not None:
        m.pose.position.x, m.pose.position.y = float(pose[0]), float(pose[1])
        m.pose.orientation.z, m.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
    for _ in range(3):                                                   # 구독이 막 맺어진 때를 대비해 몇 번 보냄
        pub.publish(m); n.wait(0.1)


def nearest(nodes, pose):
    """그 자세에서 가장 가까운, 방향이 맞는(60도 안) 노드. 멈춘 자리에서 경로를 다시 요청할 때 출발 노드로 쓴다."""
    d, k = nodes.tree.query(pose[:2], k=min(50, len(nodes.xy)))
    ok = np.abs((nodes.yaw[k] - pose[2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(60)
    return int(k[ok][0]) if ok.any() else int(k[0])


def drive(n, track_ac, pub, road, nodes, a, b, block, events, limit):
    """노드 a → b. block = (노드, 나타나는 거리 또는 None) 또는 None. events = [(시각, kind, A, B, 값)]. limit = 시간 한도(시뮬 초).
    돌려주는 값: (기록 R, 받은 경로들, 결과 글자, 장애물 자세, 한 일 기록)."""
    obstacle(n, pub)
    if not (place_amcl if getattr(n, "amcl", False) else place)(n, float(nodes.xy[a, 0]), float(nodes.xy[a, 1]), float(nodes.yaw[a])):
        return None, [], "로봇을 출발지에 세우지 못함(브리지가 떠 있는지 확인)", None, []
    box = None if block is None else (float(nodes.xy[block[0], 0]), float(nodes.xy[block[0], 1]), float(nodes.yaw[block[0]]))
    shown = box is None
    if box is not None and block[1] is None:
        obstacle(n, pub, box); shown = True; n.wait(1.0)                 # 코스트맵에 들어갈 시간
    st = dict(new=None, th=None, tr=None, pend=None); log, touched = [], []; events = sorted(events)
    def on_fb(m):
        if m.feedback.rerouted and len(m.feedback.path.poses) >= 2:
            st["new"] = m.feedback.path
    def ask(start):                                                      # 경로 추적 요청(응답은 반복문에서 받음)
        g = ComputeAndTrackRoute.Goal(); g.start_id, g.goal_id, g.use_start, g.use_poses = start, b, False, False
        st["pend"], st["th"], st["tr"] = track_ac.send_goal_async(g, feedback_callback=on_fb), None, None
    now = lambda: n.get_clock().now().nanoseconds * 1e-9
    paths, ff, fh, fr = [], None, None, None; n.rows = []; t0 = now(); hold = retry = confirm = None; done = None; ask(a)
    while done is None:
        rclpy.spin_once(n, timeout_sec=0.01); t = now() - t0
        if st["pend"] is not None and st["pend"].done():
            st["th"], st["pend"] = st["pend"].result(), None
            st["tr"] = st["th"].get_result_async() if st["th"].accepted else None
            if st["tr"] is None:
                done = "경로 추적 요청이 거절됨"
        if not shown and n.truth is not None and math.hypot(n.truth[0] - box[0], n.truth[1] - box[1]) < block[1]:
            obstacle(n, pub, box); shown = True; log.append(f"{t:.1f}s 장애물 나타남")
        while events and t >= events[0][0]:                              # 정해 둔 시각에 도로 사정을 알림(닫기, 열기, 비용)
            _, kind, ea, eb, val = events.pop(0); cnt, _ = road.send(kind, ea, eb, val); touched.append((kind, ea, eb))
            log.append(f"{t:.1f}s {kind} {ea}-{eb}(엣지 {cnt}개)")
        if hold is not None and st["new"] is not None:                   # 멈춘 뒤 받은 경로는 CONFIRM초 동안 다시 막히지 않아야 출발한다
            if st["tr"] is not None and st["tr"].done():
                st["new"], confirm = None, None                          # 그 사이 Route Server가 막혔다고 끝냄(라이다에 장애물이 그대로 있음)
            elif confirm is None:
                confirm = t
        go = hold is None or (confirm is not None and t - confirm >= CONFIRM)
        if st["new"] is not None and ff is None and go:                  # 새 경로 → RPP에 다시 보냄(앞의 주행은 컨트롤러가 바꿔치기함)
            msg, st["new"] = st["new"], None
            paths.append((t, np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])))
            fg = FollowPath.Goal(); fg.path = msg; fg.controller_id, fg.goal_checker_id = "FollowPath", "general_goal_checker"
            ff = n.follow_ac.send_goal_async(fg)
            if hold is not None:
                log.append(f"{t:.1f}s 다시 출발(멈춰 있던 시간 {t - hold:.1f}s)"); hold = confirm = None
        if ff is not None and ff.done():
            fh = ff.result(); ff = None; fr = fh.get_result_async() if fh.accepted else None
            if fr is None:
                done = "주행 요청이 거절됨"
        tr = st["tr"]; fail = None
        if tr is not None and tr.done() and tr.result().result.error_code:
            fail = f"길이 없음(Route Server 오류 {tr.result().result.error_code})"
        elif hold is None and fr is not None and ff is None and st["new"] is None and fr.done():
            if fr.result().status != OK:
                fail = f"주행이 끊김(컨트롤러 오류 {fr.result().result.error_code})"
            else:
                done = "도착"                                            # 마지막 경로를 끝까지 달림. 추적이 안 끝났으면 아래에서 취소한다
        if fail and hold is None:                                        # 멈추고, RETRY초마다 지금 자리에서 경로를 다시 요청
            hold, retry, confirm = t, t, None; log.append(f"{t:.1f}s 멈춤: {fail}")
            if fh is not None:
                fh.cancel_goal_async()
            fh = fr = None
        if hold is not None and done is None:
            if t - hold > HOLD:
                done = f"길이 열리지 않음({HOLD:.0f}초 기다림), 멈춤"
            elif st["pend"] is None and (retry is None or (t - retry >= RETRY and (tr is None or tr.done()))):
                retry = t; ask(nearest(nodes, n.odom))
        if done is None and t > limit:
            done = "시간 초과"
    if done != "도착" and fh is not None:
        fh.cancel_goal_async()
    if st["th"] is not None and not (st["tr"] is not None and st["tr"].done()):
        st["th"].cancel_goal_async()         # 도착 판정 거리(0.02 m)가 노드 간격(0.01 m)보다 커서, 로봇이 마지막 앞 노드를 지나지 않고 서면 추적이 끝나지 않는다
    for kind, ea, eb in touched:                                         # 알렸던 도로 사정은 모두 되돌려 놓음(다음 주행에 남지 않게)
        if kind != "open":
            road.send("open", ea, eb, reroute=False)
    n.wait(0.5); rows, n.rows = n.rows, None; obstacle(n, pub)
    R = np.array(rows) if len(rows) > 1 else None
    if R is not None:
        R[:, 0] -= t0
    return R, paths, done, box, log


def report(i, a, b, block, R, paths, done, box, nodes, lane_n, log, amcl=False):
    """요약 한 줄. 차로 = 가장 가까운 차로 점의 차로 번호(차선 변경 길의 점은 빼고 봄)."""
    lane_xy = nodes.xy[nodes.zone == 0]; lane_of = lambda p: int(np.argmin(np.linalg.norm(lane_xy - p, axis=1))) // lane_n
    v = np.r_[0.0, np.linalg.norm(np.diff(R[:, 1:3], axis=0), axis=1) / np.maximum(np.diff(R[:, 0]), 1e-6)]
    still = float(np.diff(R[:, 0])[(v < 0.01)[1:] & (R[1:, 0] > 1.0)].sum())
    s = (f"{i} 노드 {a} → {b}: {done} {R[-1, 0]:.1f}s | 경로 받은 횟수 {len(paths)}(다시 구한 것 {max(len(paths) - 1, 0)}) | "
         f"차로 {lane_of(R[0, 1:3])} → {lane_of(R[-1, 1:3])} | 목적지까지 남은 거리 {np.linalg.norm(R[-1, 1:3] - nodes.xy[b]) * 1000:.0f} mm | 멈춰 있던 시간 {still:.1f}s")
    if box is not None:
        d = box_dist(R, np.array(box[:2]), box[2]); k = int(np.argmin(d)); gap = (d[k] - BODY) * 1000
        s += (f" | 장애물 노드 {block[0]}(차로 {lane_of(np.array(box[:2]))})" + ("" if block[1] is None else f", {block[1]} m에서 나타남")
              + f": 가장 가까울 때 몸체 옆 여유 약 {gap:.0f} mm, 그때 차로 {lane_of(R[k, 1:3])}" + (" ← 닿음" if gap <= 0 else ""))
    if amcl:
        s += " | " + est_error(R)
    return s + ("" if not log else "\n   " + " / ".join(log))


def main():
    align = sys.argv[sys.argv.index("--align") + 1] if "--align" in sys.argv else None
    if align:
        sys.argv[sys.argv.index("--align"):sys.argv.index("--align") + 2] = []
    nodes, out = G.Nodes(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
    r0 = json.load(open(sys.argv[1]))["routes"][0]; lane_n = r0["goal"] - r0["start"] + 1      # 차로 하나의 점 수(차로 k의 점은 노드 k*lane_n부터)
    jobs, args = [], sys.argv[3:]
    while args:
        a, b = int(args.pop(0)), int(args.pop(0)); block = None
        events = []
        while args and args[0][0] in "be":
            t = args.pop(0)
            if t[0] == "b":
                t = t[1:].split("@"); block = (int(t[0]), float(t[1]) if len(t) > 1 else None)
            else:                                                        # e종류:A-B[=값]@시각
                kind, rest = t[1:].split(":"); rng, at = rest.split("@"); events.append((float(at),) + parse(kind, rng))
        jobs.append((a, b, block, events))
    rclpy.init(); n = AmclRunner(align) if align else Runner()
    track_ac = ActionClient(n, ComputeAndTrackRoute, "compute_and_track_route"); pub = n.create_publisher(PoseStamped, "/lane_sim/obstacle", 10)
    road = Road(n, os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "lane_graph.geojson"))
    track_ac.wait_for_server(); n.follow_ac.wait_for_server(); n.map_odom(); n.wait(1.0)
    lines = []; limit = float(G.arc(nodes.xy[:lane_n])[-1]) / SLOW + 30.0
    for i, (a, b, block, events) in enumerate(jobs):
        R, paths, done, box, log = drive(n, track_ac, pub, road, nodes, a, b, block, events, limit)
        if R is None:
            lines.append(f"{i} 노드 {a} → {b}: {done}"); print(lines[-1], flush=True); continue
        np.save(f"{out}/{i:02d}.npy", R); np.savez(f"{out}/{i:02d}_paths.npz", t=[t for t, _ in paths], **{f"p{k}": p for k, (_, p) in enumerate(paths)})
        lines.append(report(i, a, b, block, R, paths, done, box, nodes, lane_n, log, bool(align))); print(lines[-1], flush=True)
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
