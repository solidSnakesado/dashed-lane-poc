"""Route Server의 그래프 다시 읽기(route_server/set_route_graph)를 시험한다. 로봇은 움직이지 않는다(FollowPath를 보내지 않음).

확인할 것(2026-11-27, 장애물 회피 노드를 덧붙여 다시 읽히는 방식을 쓸 수 있는지):
  1) 다시 읽기에 걸리는 시간과 성공 여부
  2) 덧붙인 노드(회피 점열)를 Route Server가 경로에 쓰는지: 원래 차로 구간을 닫으면 덧붙인 길로 가는지
  3) 닫아 둔 엣지(DynamicEdgesScorer)가 다시 읽은 뒤에도 닫힌 채로 남는지
  4) 경로 추적(compute_and_track_route) 중에 다시 읽으면: 추적이 계속되는지, 끝나는지, Route Server가 죽는지
덧붙이는 길: 차로 0의 A번에서 B번 노드까지, 길 가운데 쪽으로 OFF만큼 비켜 갔다 돌아오는 점열(시험용 가짜 회피 길).
  기존 노드·엣지는 번호와 좌표를 그대로 두고, 새 노드·엣지는 가장 큰 번호 뒤에 붙인다.
마지막에 연 엣지를 모두 다시 열고 원래 그래프 파일을 다시 읽혀 둔다(4에서 Route Server가 죽었으면 launch를 다시 켜야 함).
사용: python3 reload_test.py GRAPH_DIR [A B]      (GRAPH_DIR = lane_graph.geojson, route_nodes.json이 있는 폴더)
"""
import json, math, os, sys, time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import ComputeRoute, ComputeAndTrackRoute
from nav2_msgs.srv import SetRouteGraph, DynamicEdges

OFF, DS = -0.05, 0.02      # 비켜 가는 옆 거리(m, 차로 0의 오른쪽 = 길 가운데 쪽), 점 간격(m)


def detour(gdir, a, b, out):
    """원래 그래프 + 덧붙인 회피 점열을 out 파일로 쓰고 (새 노드 번호들, 닫을 원래 엣지 번호들)을 돌려준다."""
    g = json.load(open(gdir + "/lane_graph.geojson")); d = json.load(open(gdir + "/route_nodes.json"))
    P, Y = np.array(d["xy"]), np.array(d["yaw"]); feats = g["features"]
    nid = max(f["properties"]["id"] for f in feats) + 1
    L = float(np.sum(np.linalg.norm(np.diff(P[a:b + 1], axis=0), axis=1))); k = max(2, int(round(L / DS)))
    new, pts = [], []
    for j in range(1, k):
        t = j / k; s = a + t * (b - a); i = int(s); f = s - i                 # 차로 0 노드 사이를 이어 그 자리의 점과 방향
        p = P[i] * (1 - f) + P[i + 1] * f; h = Y[i]; tau = 2 * t if t < 0.5 else 2 * (1 - t)
        q = p + OFF * (3 * tau * tau - 2 * tau ** 3) * np.array([-math.sin(h), math.cos(h)])   # 부드럽게 비켰다 돌아옴
        pts.append(q); new.append(nid); nid += 1
    for i, q in zip(new, pts):
        feats.append({"type": "Feature", "properties": {"id": i, "frame": "map"}, "geometry": {"type": "Point", "coordinates": [round(float(q[0]), 4), round(float(q[1]), 4)]}})
    chain = [a] + new + [b]; xy = {a: P[a], b: P[b], **dict(zip(new, pts))}
    for u, v in zip(chain[:-1], chain[1:]):
        feats.append({"type": "Feature", "properties": {"id": nid, "startid": u, "endid": v, "metadata": {"detour": True}},
                      "geometry": {"type": "MultiLineString", "coordinates": [[[round(float(c), 4) for c in xy[u]], [round(float(c), 4) for c in xy[v]]]]}}); nid += 1
    json.dump(g, open(out, "w"))
    close = [f["properties"]["id"] for f in feats if f["properties"].get("startid") is not None and a <= f["properties"]["startid"] < b
             and f["properties"]["endid"] == f["properties"]["startid"] + 1]                  # 차로 0의 A~B 엣지
    return new, close


class T(Node):
    def __init__(self):
        super().__init__("reload_test")
        self.route = ActionClient(self, ComputeRoute, "compute_route"); self.track = ActionClient(self, ComputeAndTrackRoute, "compute_and_track_route")
        self.setg = self.create_client(SetRouteGraph, "route_server/set_route_graph"); self.adj = self.create_client(DynamicEdges, "route_server/DynamicEdgesScorer/adjust_edges")

    def wait(self, f, sec=10.0):
        t0 = time.monotonic()
        while not f.done() and time.monotonic() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.01)
        return f.result() if f.done() else None

    def set_graph(self, path):
        t0 = time.monotonic(); r = self.wait(self.setg.call_async(SetRouteGraph.Request(graph_filepath=path)))
        return (r is not None and r.success), (time.monotonic() - t0) * 1000

    def edges(self, close=(), open_=()):
        return self.wait(self.adj.call_async(DynamicEdges.Request(closed_edges=list(close), opened_edges=list(open_)))) is not None

    def compute(self, s, g):
        goal = ComputeRoute.Goal(); goal.start_id, goal.goal_id, goal.use_start, goal.use_poses = s, g, False, False
        h = self.wait(self.route.send_goal_async(goal))
        if h is None or not h.accepted:
            return None
        r = self.wait(h.get_result_async())
        return None if r is None else r.result


def ids(res):
    return [n.nodeid for n in res.route.nodes] if res is not None and not res.error_code else None


def main():
    gdir = os.path.abspath(sys.argv[1]); a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (60, 100)
    orig, alt = gdir + "/lane_graph.geojson", gdir + "/lane_graph_detour.geojson"
    d = json.load(open(gdir + "/route_nodes.json")); goal = d["routes"][0]["goal"]; start = d["routes"][0]["start"]
    new, close = detour(gdir, a, b, alt); S = set(new)
    print(f"덧붙인 회피 노드 {len(new)}개({new[0]}~{new[-1]}), 차로 0 노드 {a}-{b}의 엣지 {len(close)}개를 닫는 데 씀 → {alt}")
    rclpy.init(); n = T()
    for c in (n.setg, n.adj):
        if not c.wait_for_service(timeout_sec=5.0):
            print("서비스 없음:", c.srv_name); return
    n.route.wait_for_server(timeout_sec=5.0)
    r = ids(n.compute(start, goal)); print(f"[0] 원래 그래프 경로 {start}→{goal}: 노드 {None if r is None else len(r)}개")
    ok, ms = n.set_graph(alt); print(f"[1] 다시 읽기(회피 길 붙인 파일, 추적 없음): {'성공' if ok else '실패'}, {ms:.0f} ms")
    r = ids(n.compute(start, goal)); print(f"[2a] 닫기 전 경로: 회피 노드 {0 if r is None else len(S & set(r))}개 지남 (기대: 0, 원래 차로가 더 짧음)")
    n.edges(close=close); r = ids(n.compute(start, goal))
    print(f"[2b] 차로 0 {a}-{b} 닫은 뒤: " + ("경로 없음" if r is None else f"회피 노드 {len(S & set(r))}개 지남, 닫은 구간 노드 {len(set(range(a + 1, b)) & set(r))}개 지남"))
    ok, ms = n.set_graph(alt); r = ids(n.compute(start, goal))
    print(f"[3] 같은 파일 다시 읽은 뒤({'성공' if ok else '실패'}, {ms:.0f} ms): " + ("경로 없음" if r is None else f"닫은 구간 노드 {len(set(range(a + 1, b)) & set(r))}개 지남 → 닫힘이 {'남음' if not set(range(a + 1, b)) & set(r) else '풀림'}"))
    n.edges(open_=close); n.set_graph(orig)
    n.track.wait_for_server(timeout_sec=5.0); fb = []
    g = ComputeAndTrackRoute.Goal(); g.start_id, g.goal_id, g.use_start, g.use_poses = start, goal, False, False
    h = n.wait(n.track.send_goal_async(g, feedback_callback=lambda m: fb.append(time.monotonic())))
    if h is None or not h.accepted:
        print("[4] 추적 요청이 받아들여지지 않음"); return
    res = h.get_result_async(); t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0 and not res.done():
        rclpy.spin_once(n, timeout_sec=0.01)
    before = len(fb); ok, ms = n.set_graph(alt); t1 = time.monotonic()
    while time.monotonic() - t1 < 3.0 and not res.done():
        rclpy.spin_once(n, timeout_sec=0.01)
    after = sum(1 for t in fb if t > t1)
    st = "추적 끝남(오류 %s)" % res.result().result.error_code if res.done() else "추적 계속 중"
    print(f"[4] 추적 중 다시 읽기: {'성공' if ok else '실패'}, {ms:.0f} ms / 읽기 전 3초 피드백 {before}번, 읽은 뒤 3초 {after}번 / {st}")
    if not res.done():
        n.wait(h.cancel_goal_async(), 3.0)
    alive = n.setg.wait_for_service(timeout_sec=2.0); ok, ms = n.set_graph(orig) if alive else (False, 0)
    print(f"[5] 뒤처리: Route Server {'살아 있음' if alive else '응답 없음(죽었을 수 있음)'}, 원래 그래프 다시 읽기 {'성공' if ok else '실패'}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
