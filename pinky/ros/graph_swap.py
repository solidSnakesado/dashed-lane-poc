"""주행 중 그래프 바꾸기 시험(2026-11-27): 회피 길을 붙인 그래프를 Route Server에 다시 읽히고, 원래 차로 구간을 닫고, 경로를 다시 구하게 한다.

track_runner.py로 달리는 동안 다른 터미널에서 실행한다. track_runner.py는 다시 구한 경로(피드백 rerouted)를 받아 RPP에 다시 보낸다.
회피 길은 reload_test.py의 detour()와 같다: 차로 0의 A번에서 B번 노드까지 길 가운데 쪽으로 5 cm 비켜 갔다 돌아오는 시험용 가짜 길(장애물 없음).
  로봇이 A번 노드에 닿기 전에 실행해야 한다(A번을 지난 뒤 닫으면 로봇 밑의 구간이 닫혀 경로가 없을 수 있음).
순서: 1) 회피 길 붙인 파일을 set_route_graph로 읽힘  2) 차로 0의 A~B 엣지 닫기(DynamicEdgesScorer)  3) 다시 구하기(ReroutingService)
--undo: 닫은 엣지를 다시 열고 원래 그래프 파일을 다시 읽힌다(주행이 끝난 뒤 실행). 닫힘은 다시 읽어도 남으므로 꼭 연다(reload_test.py [3]).
사용: python3 graph_swap.py GRAPH_DIR [A B] [--undo]      (기본 A B = 100 140: 굽은 구간 끝 ~ 둘째 직선, 출발 뒤 약 11초에 닿음)
"""
import os, sys, time
import rclpy
from std_srvs.srv import Trigger
from nav2_msgs.srv import SetRouteGraph, DynamicEdges
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reload_test import detour


def call(n, client, req, sec=5.0):
    t0 = time.monotonic(); f = client.call_async(req)
    while not f.done() and time.monotonic() - t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.01)
    return (f.result() if f.done() else None), (time.monotonic() - t0) * 1000


def main():
    undo = "--undo" in sys.argv; args = [a for a in sys.argv[1:] if a != "--undo"]
    gdir = os.path.abspath(args[0]); a, b = (int(args[1]), int(args[2])) if len(args) > 2 else (100, 140)
    orig, alt = gdir + "/lane_graph.geojson", gdir + "/lane_graph_swap.geojson"
    new, close = detour(gdir, a, b, alt)                                  # 같은 입력이면 같은 번호가 나오므로 --undo에서도 다시 만든다
    rclpy.init(); n = rclpy.create_node("graph_swap")
    setg = n.create_client(SetRouteGraph, "route_server/set_route_graph")
    adj = n.create_client(DynamicEdges, "route_server/DynamicEdgesScorer/adjust_edges")
    re = n.create_client(Trigger, "route_server/ReroutingService/reroute")
    for c in (setg, adj, re):
        if not c.wait_for_service(timeout_sec=5.0):
            print("서비스 없음:", c.srv_name); return
    t0 = time.monotonic()
    if undo:
        r, ms1 = call(n, adj, DynamicEdges.Request(opened_edges=close))
        g, ms2 = call(n, setg, SetRouteGraph.Request(graph_filepath=orig))
        print(f"되돌림: 엣지 {len(close)}개 열기 {'성공' if r is not None and r.success else '실패'}({ms1:.0f} ms), "
              f"원래 그래프 다시 읽기 {'성공' if g is not None and g.success else '실패'}({ms2:.0f} ms)")
    else:
        g, ms1 = call(n, setg, SetRouteGraph.Request(graph_filepath=alt))
        r, ms2 = call(n, adj, DynamicEdges.Request(closed_edges=close))
        t, ms3 = call(n, re, Trigger.Request())
        print(f"회피 노드 {len(new)}개({new[0]}~{new[-1]}) 붙인 그래프 다시 읽기 {'성공' if g is not None and g.success else '실패'}({ms1:.0f} ms) / "
              f"차로 0 {a}-{b} 엣지 {len(close)}개 닫기 {'성공' if r is not None and r.success else '실패'}({ms2:.0f} ms) / "
              f"다시 구하기 요청 {'보냄' if t is not None and t.success else '실패'}({ms3:.0f} ms)")
    print(f"전체 {(time.monotonic() - t0) * 1000:.0f} ms")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
