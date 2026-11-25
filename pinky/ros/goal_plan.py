"""출발지 → 목적지 주행에서 ROS와 무관한 계산(시험하기 쉽게 따로 둠): 찍은 곳을 노드로, 경로 위의 정지선 찾기, 정지선마다 경로 자르기, 주행 기록 채점.

route_export.py가 만든 route_nodes.json을 쓴다(노드 좌표·방향·교차로 여부·양쪽 선까지 거리, 정지 노드 목록).
거리 단위는 m. 경로(path)는 Route Server가 준 dense path의 (x, y) 점열.
"""
import json, math
import numpy as np
from scipy.spatial import cKDTree

FRONT = 0.072          # 바퀴 축에서 로봇 앞머리까지(m). pinky_mujoco 물리 모델의 충돌 형상에서 잰 값
NEAR = 0.02            # 정지 노드가 경로에서 이 거리 안에 있어야 그 경로의 정지선으로 본다
SAME_LINE = 0.15       # 이 거리 안의 정지 지점들은 같은 정지선(겹친 경로들이 각자 표시한 것)
STILL_V, STILL_T = 0.01, 0.5   # 이 속도(m/s)보다 느린 상태가 이 시간(초) 넘게 이어지면 '멈춤'


class Nodes:
    def __init__(self, path):
        d = json.load(open(path))
        self.xy, self.yaw, self.zone = np.array(d["xy"]), np.array(d["yaw"]), np.array(d["zone"])
        self.wl, self.wr = np.array(d["wl"]), np.array(d["wr"])
        self.stops = [(int(k), float(v)) for k, v in d["stops"]]
        self.map = d.get("map")                                          # 지도 그림 정보(없을 수 있음)
        self.tree = cKDTree(self.xy)

    def pick(self, text):
        """'123' = 노드 번호 그대로. 'x,y' 또는 'x,y,방향(도)' = 지도에 찍은 곳에서 가장 가까운 교차로 밖 노드(방향을 주면 그 방향의 차로)."""
        v = [float(t) for t in text.split(",")]
        if len(v) == 1:
            return int(v[0])
        d, k = self.tree.query(v[:2], k=min(200, len(self.xy)))
        ok = self.zone[k] == 0
        if len(v) > 2:
            ok &= np.abs((self.yaw[k] - math.radians(v[2]) + math.pi) % (2 * math.pi) - math.pi) < math.radians(60)
        if not ok.any():
            raise ValueError(f"{text}: 가까이에 맞는 차로가 없음")
        return int(k[ok][0])


class Graph:
    """그래프 파일에서 최소 비용 길을 찾는다(미리 보기·ROS 없는 시험용). 값은 Route Server와 같이: 엣지에 cost가 있으면 그 값, 없으면 길이."""
    def __init__(self, geojson, nodes):
        from scipy.sparse import csr_matrix
        E = [(p["startid"], p["endid"], p.get("cost")) for p in (x["properties"] for x in json.load(open(geojson))["features"]) if "startid" in p]
        r, c, w = zip(*[(i, j, k if k is not None else float(np.linalg.norm(nodes.xy[i] - nodes.xy[j]))) for i, j, k in E])
        self.m, self.nodes = csr_matrix((w, (r, c)), shape=(len(nodes.xy),) * 2), nodes

    def route(self, a, b, density=0.05):
        """a → b 길을 density(m) 간격의 점열로. 길이 없으면 None."""
        from scipy.sparse.csgraph import dijkstra
        d, pred = dijkstra(self.m, indices=a, return_predecessors=True)
        if not np.isfinite(d[b]):
            return None
        ids = [b]
        while ids[-1] != a:
            ids.append(int(pred[ids[-1]]))
        P = self.nodes.xy[ids[::-1]]; s = arc(P); q = np.r_[np.arange(0, s[-1], density), s[-1]]
        return np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])]


def arc(path):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def project(path, s, xy):
    """xy를 경로에 내린 점: (경로 따라 거리, 옆으로 벗어난 거리(왼쪽 +), 그 자리의 진행 방향)."""
    a, b = path[:-1], path[1:]; v = b - a; L = np.einsum("ij,ij->i", v, v) + 1e-12
    t = np.clip(np.einsum("ij,ij->i", xy - a, v) / L, 0, 1); c = a + t[:, None] * v
    k = int(np.argmin(np.linalg.norm(xy - c, axis=1))); d = xy - c[k]
    return float(s[k] + t[k] * math.sqrt(L[k])), float(np.sign(v[k, 0] * d[1] - v[k, 1] * d[0]) * np.linalg.norm(d)), math.atan2(v[k, 1], v[k, 0])


def stops_on(path, nodes, start_s=0.0):
    """경로가 지나는 정지선: [(멈출 곳까지 거리, 정지선까지 거리)]를 가는 순서대로. start_s보다 앞머리가 이미 넘어간 선은 뺀다."""
    s = arc(path); found = []
    for k, gap in nodes.stops:
        at, off, ang = project(path, s, nodes.xy[k])
        if abs(off) < NEAR and abs((ang - nodes.yaw[k] + math.pi) % (2 * math.pi) - math.pi) < math.radians(30) and 0 < at < s[-1]:
            found.append((at, at + gap))
    found.sort(); out = []
    for at, line in found:
        if out and line - out[-1][1] < SAME_LINE:
            continue                                                     # 같은 선을 겹친 경로가 또 표시한 것
        if line > start_s + FRONT and line < s[-1]:
            out.append((at, line))
    return out


def cut(path, at_list):
    """경로를 멈출 곳마다 잘라 구간(점열)들로. 구간의 끝점은 멈출 곳 그 자리."""
    s = arc(path); segs, s0 = [], 0.0
    point = lambda q: np.array([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])])
    for q in list(at_list) + [s[-1]]:
        if q - s0 < 1e-3:
            continue
        mid = path[(s > s0 + 1e-6) & (s < q - 1e-6)]
        segs.append(np.vstack([point(s0), mid, point(q)])); s0 = q
    return segs


def score(R, path, nodes, stops):
    """주행 기록 R(줄: 시간, 실제 x, y, yaw, ...)을 채점. 돌려주는 값: dict(벗어난 거리, 차로 밖 줄 수, 정지선마다 결과)."""
    s = arc(path); pr = np.array([project(path, s, p)[:2] for p in R[:, 1:3]]); at, e = pr[:, 0], pr[:, 1]
    k = nodes.tree.query(R[:, 1:3])[1]; zone = nodes.zone[k] > 0
    outside = (~zone) & ((e > nodes.wl[k]) | (e < -nodes.wr[k]))
    v = np.r_[0.0, np.linalg.norm(np.diff(R[:, 1:3], axis=0), axis=1) / np.maximum(np.diff(R[:, 0]), 1e-6)]
    still = v < STILL_V; res = []
    for stop_at, line in stops:
        m = still & (np.abs(at - stop_at) < 0.2)                         # 멈출 곳 근처에서 멈춰 있던 줄
        dwell = float(np.diff(R[:, 0])[m[1:]].sum()) if m.any() else 0.0
        if dwell < STILL_T:
            crossed = np.where(at + FRONT > line)[0]
            res.append(dict(stopped=False, dwell=dwell, front=None, v_at_line=float(v[crossed[0]]) if len(crossed) else None)); continue
        res.append(dict(stopped=True, dwell=dwell, front=float(line - (np.median(at[m]) + FRONT))))   # 앞머리에서 정지선까지 남은 거리(음수면 넘음)
    return dict(e=e, zone=zone, outside=int(outside.sum()), stops=res)


def report(i, a, b, path, R, done, nodes, stops):
    """쌍 하나의 요약 한 줄(물리 시험 goal_drive.py와 ROS2 주행 goal_runner.py가 같이 씀)."""
    sc = score(R, path, nodes, stops); e = np.abs(sc["e"]) * 1000; z = sc["zone"]
    part = lambda m: f"{np.median(e[m]):.1f}/{np.percentile(e[m], 95):.1f}/{e[m].max():.1f}" if m.any() else "-"
    st = ", ".join((f"멈춤 {s['dwell']:.1f}s 앞머리~선 {s['front'] * 1000:+.0f}mm" if s["stopped"] else "안 멈춤") for s in sc["stops"]) or "없음"
    return (f"{i} 노드 {a} → {b}: {done} {R[-1, 0]:.1f}s, 경로 {arc(path)[-1]:.2f} m | 벗어난 거리 mm(중앙/95%/최대) "
            f"교차로 밖 {part(~z)} 안 {part(z)} | 차로 밖 줄 {sc['outside']} | 정지선 {len(stops)}곳: {st}")
