"""도로 그래프(route_graph.py)를 Nav2 Route Server가 읽는 GeoJSON 그래프로 내보낸다.

노드 = 경로 점(ego_label.Routes의 4 mm 간격 점, 번호는 route_graph.Graph의 점 번호). Route Server 노드 번호가 uint16이라 65535개까지.
엣지 = 같은 경로의 다음 점으로(값은 Route Server의 DistanceScorer가 길이로 계산),
       같은 차로를 같이 달리는 다른 경로의 점으로(앞으로만, 값 고정 = 거리 + 벌칙, route_graph와 같은 규칙, SAME_EVERY 점마다).
       차선 변경(route_graph가 점선에서만 만든 선, 값 = 옆으로 옮긴 거리 + 벌칙, metadata lane_change = true).
교차로 안 엣지에는 metadata lane_vision = false(비전 보정 끔, Route Server 문서의 제안 키).
차선 변경 엣지는 두 점을 곧게 잇는 선이다(옆으로 차로 폭만큼). 부드럽게 옮겨 가는 궤적(draw_lane.poses의 S자)은 경로를 받은 쪽에서 만들어야 한다.
정지선: 바닥에 칠해진 정지선(inst.png의 stop_line)을 경로가 가로질러 넘는 곳마다, 그 선의 가까운 쪽 가장자리에서 STOP_GAP만큼 앞의 노드에
       metadata stop_line = true, stop_gap = 그 노드에서 선까지 거리(m)를 넣는다. 로봇(바퀴 축)이 이 노드에 서면 앞머리가 선을 넘지 않는다.
함께 쓰는 파일(route_nodes.json): 경로마다 출발·도착 노드 번호, 노드 좌표·방향·교차로 여부·차로 가운데에서 양쪽 선까지 거리, 정지 노드 목록(stops: [노드 번호, 선까지 거리]).
사용: python route_export.py ROAD_DIR OUT_DIR
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT + "/pinky/sim", ROOT]
from ego_label import Routes, STEP
from route_graph import Graph, SAME, SAME_YAW, JUMP

STOP_GAP = 0.09        # 정지 노드에서 정지선까지(m) = 바퀴 축에서 로봇 앞머리까지 0.072(pinky_mujoco 물리 모델에서 잰 값) + 여유
STOP_CROSS = 60.0      # 경로가 정지선을 이 각도(도)보다 크게 가로지를 때만 그 선을 지키는 것으로 본다(스치듯 지나는 남의 선은 뺌)
SAME_EVERY = 5        # 같은 차로 건너가기 엣지는 점 이 개수마다(엣지 번호도 uint16 범위에 맞추려고)


class SC:
    pass


def stop_nodes(road, info, g):
    """정지 노드 {노드 번호: 선까지 거리 m}. 경로 점이 정지선 칠 위에 올라가는 첫 점에서 STOP_GAP만큼 앞 노드."""
    import cv2
    ins = cv2.imread(road + "/inst.png", cv2.IMREAD_UNCHANGED); H, ppm = ins.shape[0], info["ppm"]
    ids = [int(k) for k, v in info["instances"].items() if v == "stop_line"]
    axis = {}
    for k in ids:                                                        # 정지선마다 길게 뻗은 방향(지도 좌표)
        v, u = np.where(ins == k); c = np.c_[u, -v].astype(float); c -= c.mean(0)
        axis[k] = np.linalg.svd(c, full_matrices=False)[2][0]
    out = {}
    for j in np.unique(g.pid[:, 0]):
        pts = np.where(g.pid[:, 0] == j)[0]; p = g.P[pts]
        u = np.clip((p[:, 0] * ppm).astype(int), 0, ins.shape[1] - 1); v = np.clip((H - p[:, 1] * ppm).astype(int), 0, H - 1)
        hit = ins[v, u]; on = np.isin(hit, ids)
        for a in np.where(on & ~np.r_[False, on[:-1]])[0]:               # 선 위로 올라서는 첫 점마다
            t = np.array([np.cos(g.yaw[pts[a]]), np.sin(g.yaw[pts[a]])])
            if abs(t @ axis[int(hit[a])]) > np.cos(np.radians(STOP_CROSS)):
                continue
            b = max(a - int(round(STOP_GAP * 1000 / STEP)), 0)
            out[int(pts[b])] = round(float(np.linalg.norm(np.diff(p[b:a + 1], axis=0), axis=1).sum()) / 1000.0, 4)
    return out


def main():
    road, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    sc = SC(); sc.info = json.load(open(road + "/road.json"))
    g = Graph(Routes(sc, 30, 15), STEP, road)                             # road를 주면 차선 변경 선(g.lc)도 만든다
    P = g.P / 1000.0; n = len(P)
    stops = stop_nodes(road, sc.info, g)
    assert n < 65535, "노드가 uint16 범위를 넘음"
    feats = [{"type": "Feature", "properties": {"id": int(i), "frame": "map", **({"metadata": {"stop_line": True, "stop_gap": stops[i]}} if i in stops else {})},
              "geometry": {"type": "Point", "coordinates": [round(float(P[i, 0]), 4), round(float(P[i, 1]), 4)]}} for i in range(n)]
    eid = n                                                              # 엣지 번호는 노드 번호 뒤부터
    def edge(a, b, cost=None, lc=False):
        nonlocal eid
        props = {"id": eid, "startid": int(a), "endid": int(b), "metadata": {"lane_vision": bool(not (g.zone[a] or g.zone[b]))}}
        if lc:
            props["metadata"]["lane_change"] = True
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
    for a, b in sorted(g.lc):                                            # 차선 변경(점선 구간만)
        edge(a, b, float(g.G[a, b]) / 1000.0, lc=True)
    assert eid < 65535, "엣지 번호가 uint16 범위를 넘음"
    json.dump({"type": "FeatureCollection", "name": "graph", "features": feats}, open(out + "/lane_graph.geojson", "w"))
    starts = {j: int(np.where(g.pid[:, 0] == j)[0][0]) for j in range(len(sc.info["routes"]))}
    ends = {j: int(np.where(g.pid[:, 0] == j)[0][-1]) for j in range(len(sc.info["routes"]))}
    json.dump({"routes": [dict(name=r["name"], kind=r["kind"], start=starts[j], goal=ends[j],
                               half=min(r["bounds_in"] + r["bounds_out"]) / 1000.0) for j, r in enumerate(sc.info["routes"])],
               "xy": np.round(P, 4).tolist(), "yaw": np.round(g.yaw, 5).tolist(), "zone": g.zone.tolist(),
               "wl": np.round(g.wl / 1000.0, 4).tolist(), "wr": np.round(g.wr / 1000.0, 4).tolist(), "stops": sorted([int(k), v] for k, v in stops.items())},
              open(out + "/route_nodes.json", "w"))
    print(f"노드 {n}, 엣지 {eid - n} (차선 변경 {len(g.lc)}), 정지 노드 {len(stops)}, 경로 {len(starts)} → {out}/lane_graph.geojson")


if __name__ == "__main__":
    main()
