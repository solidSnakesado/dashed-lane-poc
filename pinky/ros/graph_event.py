"""주행 중 그래프 바꾸기(track_runner.py의 eswap:A-B@시각): 회피 길을 붙인 그래프를 Route Server에 다시 읽히고, 원래 차로 구간을 닫고, 지금 자리에서 다시 요청한다.

순서가 중요하다(2026-11-27 실물): 추적(compute_and_track_route) 중에 set_route_graph를 부르면 Route Server가 죽었다
  (RouteTracker::nodeAchieved에서 세그멘테이션 오류. 다시 읽기가 그래프를 비운 뒤에도 추적이 옛 그래프의 노드를 읽음).
  그래서 1) 추적을 취소하고 2) 추적이 끝난 것(결과 future 완료)을 확인한 뒤 3) 다시 읽고 구간을 닫고 4) 지금 자리에서 추적을 다시 요청한다.
  그동안 컨트롤러(RPP)는 앞서 받은 경로를 계속 따라간다(멈추지 않음). 새 경로가 오면 track_runner.py가 RPP에 바꿔 보낸다.
회피 길: begin(A, B)는 reload_test.py의 detour()(차로 0의 A~B 옆으로 5 cm, 시험용 가짜 길). avoid(obstacle_detour.Detour)를 주면
  주행 중 라이다로 앞쪽 장애물을 찾아 그 둘레의 회피 길을 만든다. 가까이 가며 더 보이면 다시 만들어 또 바꾼다(그때 앞서 닫은 엣지는 연다).
회피 중에 다시 만들기가 통로 없음을 내면(blocked) 지금 회피 길의 엣지를 닫고 Route Server에 경로를 다시 구하게 한다(ReroutingService).
  갈 길이 없으면 Route Server가 길 없음으로 끝내고 track_runner.py가 RPP를 멈춘 뒤 다시 묻는다. 뒤에 통로가 생기면 새 회피 길로 바꾼다
  (2026-11-27 실물 real_c7_17: 통로 없음을 내고도 앞서 받은 회피 길을 계속 달려 장애물에 닿음).
닫은 엣지는 다시 읽어도 남으므로(reload_test.py [3]) 주행이 끝나면 restore()로 열고 원래 그래프를 다시 읽힌다. 이때도 추적이 끝난 뒤에 한다.
"""
import json, time
import rclpy
from std_srvs.srv import Trigger
from nav2_msgs.srv import SetRouteGraph, DynamicEdges
from reload_test import detour
from obstacle_detour import Detour


class Swap:
    def __init__(self, node, gdir, map_yaml=None, scan_topic="/scan"):
        self.n, self.gdir, self.job, self.cancel, self.closed = node, gdir, None, False, None
        self.avoid = Detour(node, gdir, map_yaml, scan_topic) if map_yaml else None      # 장애물 회피(--avoid MAP_YAML)
        self.setg = node.create_client(SetRouteGraph, "route_server/set_route_graph")
        self.adj = node.create_client(DynamicEdges, "route_server/DynamicEdgesScorer/adjust_edges")
        self.rer, self.last = node.create_client(Trigger, "route_server/ReroutingService/reroute"), None   # 경로 다시 구하기 / 지금 회피 그래프 파일

    def call(self, client, req, sec=3.0):
        t0 = time.monotonic(); f = client.call_async(req)
        while not f.done() and time.monotonic() - t0 < sec:
            rclpy.spin_once(self.n, timeout_sec=0.01)
        return f.result() if f.done() else None

    def begin(self, a, b):
        new, close = detour(self.gdir, a, b, self.gdir + "/lane_graph_swap.geojson")
        self.job, self.cancel = (self.gdir + "/lane_graph_swap.geojson", close, f"회피 노드 {len(new)}개(차로 0 {a}-{b})", None), False

    def step(self, st, ask, here, log, t):
        """반복문에서 매번 부른다. 추적이 끝날 때까지 기다렸다가 그래프를 바꾸고 다시 요청한다."""
        if self.job is None and self.avoid is not None and (r := self.avoid.find(self.n.odom)) is not None:
            log.append(f"{t:.1f}s {r['note']}")
            if "file" in r:
                self.job, self.cancel, self.last = (r["file"], r["close"], "장애물 회피 노드", r["start"]), False, r["file"]
            elif r.get("blocked") and self.last:                         # 지금 달리는 회피 길이 막힘 → 닫고 다시 구하게 함
                ids = [f["properties"]["id"] for f in json.load(open(self.last))["features"] if (f["properties"].get("metadata") or {}).get("detour")]
                close = sorted(set(self.closed or []) | set(ids)); a = self.call(self.adj, DynamicEdges.Request(closed_edges=close)); self.closed = close
                q = self.call(self.rer, Trigger.Request())
                log.append(f"{t:.1f}s 지금 회피 길이 막힘: 회피 엣지 {len(ids)}개 닫기 {'성공' if a is not None and a.success else '실패'}, "
                           f"경로 다시 구하기 {'성공' if q is not None and q.success else '실패'} → 길이 없으면 멈춤")
        if self.job is None or st["pend"] is not None:
            return
        if st["tr"] is not None and not st["tr"].done():                 # 1) 추적 중이면 취소하고 끝나기를 기다림
            if not self.cancel:
                st["th"].cancel_goal_async(); self.cancel = True; log.append(f"{t:.1f}s 그래프 바꾸기: 추적 취소")
            return
        path, close, what, start = self.job; t0 = time.monotonic(); reopen = sorted(set(self.closed or []) - set(close))
        g = self.call(self.setg, SetRouteGraph.Request(graph_filepath=path))
        r = self.call(self.adj, DynamicEdges.Request(closed_edges=close, opened_edges=reopen))   # 앞서 닫은 것 가운데 이번에 안 닫는 것은 연다
        self.closed, self.job = close, None
        log.append(f"{t:.1f}s {what} 붙여 다시 읽기 {'성공' if g is not None and g.success else '실패'}, "
                   f"엣지 {len(close)}개 닫기 {'성공' if r is not None and r.success else '실패'} ({(time.monotonic() - t0) * 1000:.0f} ms) → 다시 요청")
        ask(here() if start is None else start)                           # 회피 길은 로봇 바로 앞의 첫 회피 노드에서 출발

    def restore(self, st):
        """주행이 끝난 뒤: 추적이 끝나기를 기다리고(최대 3초), 닫은 엣지를 열고 원래 그래프를 다시 읽힌다."""
        if self.closed is None:
            return
        t0 = time.monotonic()
        while st["tr"] is not None and not st["tr"].done() and time.monotonic() - t0 < 3.0:
            rclpy.spin_once(self.n, timeout_sec=0.01)
        if st["tr"] is not None and not st["tr"].done():
            print("그래프 되돌리지 않음: 추적이 3초 안에 끝나지 않음(이때 다시 읽으면 Route Server가 죽을 수 있음)", flush=True); return
        self.call(self.adj, DynamicEdges.Request(opened_edges=self.closed))
        g = self.call(self.setg, SetRouteGraph.Request(graph_filepath=self.gdir + "/lane_graph.geojson"))
        print("그래프 되돌림:", "성공" if g is not None and g.success else "실패", flush=True); self.closed = None
