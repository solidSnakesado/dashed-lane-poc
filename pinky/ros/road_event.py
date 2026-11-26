"""도로 사정을 Route Server에 알린다: 구간을 닫기, 다시 열기, 비용 올리기. 알린 뒤 주행 중인 경로를 다시 구하게 한다.

Route Server의 두 서비스를 쓴다(nav2_route에 들어 있는 기능, wall_nav.yaml에서 켬).
  route_server/DynamicEdgesScorer/adjust_edges (nav2_msgs/srv/DynamicEdges): 닫을 엣지, 열 엣지, 비용을 더할 엣지
  route_server/ReroutingService/reroute (std_srvs/srv/Trigger): 추적 중인 경로를 그 자리에서 다시 구함
구간은 차로의 노드 번호 범위 A-B로 준다(차로의 점은 번호가 이어져 있음). 고르는 엣지 = 끝 노드가 아래 중 하나인 엣지 전부.
  1) 차로의 A 초과 B 이하 노드
  2) 그 노드들에서 차로 반폭 안에 있는 차선 변경 길의 점(route_nodes.json의 zone = 1). 구간 옆을 스쳐 지나거나 구간 안으로 들어오는 길을 막는다.
     (2가 없으면 구간보다 긴 차선 변경 길이 닫힌 구간을 건너뛰는 경로가 뽑힌다. 2026-11-26 시험에서 실제로 그랬음)
  비용이 고정된 엣지(차선 변경 길의 첫 엣지)에는 Route Server가 이 기능을 적용하지 않는다. 그 길은 둘째 엣지부터 닫히므로 결과는 같다.
이 서비스로 닫은 것은 이 서비스로만 열린다. Route Server가 라이다(CollisionMonitor)로 찾아 막은 엣지와는 따로 관리된다.
사용: python road_event.py GRAPH.geojson close|open A-B [A-B ...]      (open은 닫힘과 더한 비용을 모두 없앰)
      python road_event.py GRAPH.geojson cost A-B=값 [A-B=값 ...]       (값 = 엣지마다 더할 비용. 엣지 길이가 약 0.01 m이므로 0.01이면 그 구간이 두 배 길이로 계산됨)
      끝에 --no-reroute 를 붙이면 다시 구하기 요청은 보내지 않는다(다음 경로 계산부터 반영).
"""
import json, os, sys
import numpy as np
import rclpy
from scipy.spatial import cKDTree
from nav2_msgs.msg import EdgeCost
from nav2_msgs.srv import DynamicEdges
from std_srvs.srv import Trigger


class Road:
    def __init__(self, node, graph):
        P = [f["properties"] for f in json.load(open(graph))["features"]]
        self.edges = [(p["id"], p["startid"], p["endid"]) for p in P if "startid" in p]
        d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(graph)), "route_nodes.json")))   # 그래프와 같은 폴더
        self.xy, self.zone, self.half = np.array(d["xy"]), np.array(d["zone"]), np.array(d["wl"]); self.tree = cKDTree(self.xy)
        self.adj = node.create_client(DynamicEdges, "route_server/DynamicEdgesScorer/adjust_edges")
        self.re = node.create_client(Trigger, "route_server/ReroutingService/reroute")

    def ids(self, a, b):
        ends = set(range(a + 1, b + 1))
        for k in range(a + 1, b + 1):
            ends.update(j for j in self.tree.query_ball_point(self.xy[k], float(self.half[k]) * 1.01) if self.zone[j] == 1)
        return [e for e, _, t in self.edges if t in ends]

    def send(self, kind, a, b, value=0.0, reroute=True):
        """요청을 보내고 (엣지 수, 응답을 기다리는 future)를 돌려준다. reroute면 응답이 온 뒤 다시 구하기 요청도 보낸다."""
        ids = self.ids(a, b); req = DynamicEdges.Request()
        if kind == "close":
            req.closed_edges = ids
        elif kind == "open":
            req.opened_edges = ids; req.adjust_edges = [EdgeCost(edgeid=i, cost=0.0) for i in ids]
        elif kind == "cost":
            req.adjust_edges = [EdgeCost(edgeid=i, cost=float(value)) for i in ids]
        else:
            raise ValueError(kind)
        f = self.adj.call_async(req)
        if reroute:
            f.add_done_callback(lambda _: self.re.call_async(Trigger.Request()))
        return len(ids), f


def parse(kind, text):
    """'A-B' 또는 'A-B=값' → (kind, A, B, 값)."""
    rng, _, val = text.partition("="); a, b = rng.split("-")
    return kind, int(a), int(b), float(val) if val else 0.0


def main():
    args = [a for a in sys.argv[1:] if a != "--no-reroute"]; reroute = "--no-reroute" not in sys.argv
    graph, kind = args[0], args[1]
    rclpy.init(); n = rclpy.create_node("road_event"); road = Road(n, graph)
    if not road.adj.wait_for_service(timeout_sec=3.0):
        print("adjust_edges 서비스가 없음(Route Server가 떠 있는지, wall_nav.yaml에 DynamicEdgesScorer가 있는지 확인)"); return
    for t in args[2:]:
        _, a, b, v = parse(kind, t); cnt, f = road.send(kind, a, b, v, reroute=False)
        rclpy.spin_until_future_complete(n, f, timeout_sec=3.0)
        print(f"{kind} 노드 {a}-{b}: 엣지 {cnt}개" + (f", 더한 비용 {v}" if kind == "cost" else "") + f" → {'반영됨' if f.done() and f.result().success else '응답 없음'}")
    if reroute:
        f = road.re.call_async(Trigger.Request()); rclpy.spin_until_future_complete(n, f, timeout_sec=3.0)
        print("다시 구하기 요청:", "보냄" if f.done() and f.result().success else "응답 없음")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
