"""도로 그래프(route_graph.py)를 Nav2 Route Server가 읽는 GeoJSON 그래프로 내보낸다.

노드 = 경로 점(ego_label.Routes의 4 mm 간격 점, 번호는 route_graph.Graph의 점 번호). Route Server 노드 번호가 uint16이라 65535개까지.
엣지 = 같은 경로의 다음 점으로(값은 Route Server의 DistanceScorer가 길이로 계산),
       같은 차로를 같이 달리는 다른 경로의 점으로(앞으로만, 값 고정 = 거리 + 벌칙, route_graph와 같은 규칙, SAME_EVERY 점마다).
교차로 안 엣지에는 metadata lane_vision = false(비전 보정 끔, Route Server 문서의 제안 키). 차선 변경 엣지는 넣지 않는다.
함께 쓰는 파일(route_nodes.json): 경로마다 출발·도착 노드 번호, 노드 좌표·교차로 여부(시험 기록에서 씀).
사용: python route_export.py ROAD_DIR OUT_DIR
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT + "/pinky/sim", ROOT]
from ego_label import Routes, STEP
from route_graph import Graph, SAME, SAME_YAW, JUMP

SAME_EVERY = 5        # 같은 차로 건너가기 엣지는 점 이 개수마다(엣지 번호도 uint16 범위에 맞추려고)


class SC:
    pass


def main():
    road, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    sc = SC(); sc.info = json.load(open(road + "/road.json"))
    g = Graph(Routes(sc, 30, 15), STEP)
    P = g.P / 1000.0; n = len(P)
    assert n < 65535, "노드가 uint16 범위를 넘음"
    feats = [{"type": "Feature", "properties": {"id": int(i), "frame": "map"},
              "geometry": {"type": "Point", "coordinates": [round(float(P[i, 0]), 4), round(float(P[i, 1]), 4)]}} for i in range(n)]
    eid = n                                                              # 엣지 번호는 노드 번호 뒤부터
    def edge(a, b, cost=None):
        nonlocal eid
        props = {"id": eid, "startid": int(a), "endid": int(b), "metadata": {"lane_vision": bool(not (g.zone[a] or g.zone[b]))}}
        if cost is not None:
            props.update(cost=round(float(cost), 6), overridable=False)
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "MultiLineString",
                      "coordinates": [[[round(float(P[a, 0]), 4), round(float(P[a, 1]), 4)], [round(float(P[b, 0]), 4), round(float(P[b, 1]), 4)]]]}})
        eid += 1
    for a in range(n - 1):                                               # 같은 경로의 다음 점
        if g.pid[a, 0] == g.pid[a + 1, 0]:
            edge(a, a + 1)
    for a, b in g.tree.query_pairs(SAME * STEP):                         # 같은 차로를 같이 달리는 점(앞으로만)
        if g.pid[a, 0] == g.pid[b, 0] or (g.pid[a, 1] % SAME_EVERY and g.pid[b, 1] % SAME_EVERY):
            continue
        if abs((g.yaw[a] - g.yaw[b] + np.pi) % (2 * np.pi) - np.pi) >= np.radians(SAME_YAW):
            continue
        t = np.array([np.cos(g.yaw[a]), np.sin(g.yaw[a])]); fw = (g.P[b] - g.P[a]) @ t
        c = (float(np.linalg.norm(g.P[b] - g.P[a])) + JUMP * STEP) / 1000.0
        if fw >= 0:
            edge(a, b, c)
        if fw <= 0:
            edge(b, a, c)
    assert eid < 65535, "엣지 번호가 uint16 범위를 넘음"
    json.dump({"type": "FeatureCollection", "name": "graph", "features": feats}, open(out + "/lane_graph.geojson", "w"))
    starts = {j: int(np.where(g.pid[:, 0] == j)[0][0]) for j in range(len(sc.info["routes"]))}
    ends = {j: int(np.where(g.pid[:, 0] == j)[0][-1]) for j in range(len(sc.info["routes"]))}
    json.dump({"routes": [dict(name=r["name"], kind=r["kind"], start=starts[j], goal=ends[j],
                               half=min(r["bounds_in"] + r["bounds_out"]) / 1000.0) for j, r in enumerate(sc.info["routes"])],
               "xy": np.round(P, 4).tolist(), "yaw": np.round(g.yaw, 5).tolist(), "zone": g.zone.tolist()},
              open(out + "/route_nodes.json", "w"))
    print(f"노드 {n}, 엣지 {eid - n}, 경로 {len(starts)} → {out}/lane_graph.geojson")


if __name__ == "__main__":
    main()
