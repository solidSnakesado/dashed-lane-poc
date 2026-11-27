"""SLAM 지도에서 바로 도로 그래프를 만든다(도면 없이). 사람이 점을 찍지 않는다.

배경(2026-11-26): 도면(wall_map.py)의 굽은 구간 블럭 좌표가 실제 배치와 달라(모서리 사이 거리가 실제보다 16~22 cm 김)
  map_align.py로 맞추면 길 양 끝이 10 cm 넘게 어긋났다. 실제로 잰 길이는 SLAM 지도와 맞았다(바깥 벽 대각선 지도 192.9 / 실제 194.5 cm).
방법: 씨앗 자세(지도 좌표, 로봇을 놓았던 자리)에서 길 가운데 선을 따라간다.
  1) 지금 점에서 진행 방향의 왼쪽·오른쪽 수직으로 광선을 쏴 벽 칸(점유)까지 거리 dl, dr을 잰다(작은 각도로 여러 개 쏴 중앙값).
  2) 점을 (dl − dr)/2만큼 옆으로 옮겨 가운데에 둔다. 앞으로 STEP 간 뒤 다시 1)~2), 방향은 옮긴 두 점을 잇는 방향을 부드럽게 따라간다.
  3) 양쪽 벽 중 하나라도 MAX_W 안에 없거나 폭이 W_RANGE 밖이면(열린 끝) 멈춘다. 앞쪽 END_GAP 안에 벽이 있으면(닫힌 끝) 멈춘다.
  씨앗에서 앞으로, 뒤로 두 번 따라가 잇는다. 씨앗 자리에 벽이 없으면 씨앗 방향으로 SEED_MAX까지 나아가며 찾는다.
  가운데 선과 폭을 SMOOTH 길이로 고르게 한 뒤 STEP 간격으로 다시 나눈다.
차로: 각 자리의 폭을 LANES줄로 똑같이 나눈다(wall_map.py와 같은 규칙: 차로 0이 왼쪽, 모두 같은 방향, 차선 변경 길과 벌칙도 같음).
결과(OUT 폴더): lane_graph.geojson, route_nodes.json(map_align.py의 결과와 같은 형식, 좌표는 지도 좌표),
  align.json(도면 → 지도 변환은 항등: track_runner.py --align 에 그대로 줌), amcl_init.yaml(길 가운데 출발 자리), overlay.png(지도 8배 위에 선)
씨앗: wall_follow.npy의 첫 줄(SLAM을 켠 직후 오도메트리 = 지도 좌표. slam_toolbox는 처음 map → odom을 항등으로 둔다) 또는 x y 방향(도).
사용: python map_graph.py MAP_YAML OUT (WALL_FOLLOW_NPY | x y yaw_deg)
"""
import json, math, os, sys
import cv2, numpy as np, yaml
from scipy.ndimage import uniform_filter1d
from wall_map import LANES, LC_EVERY, LC_LEN, LC_STEP, LC_PENALTY, STEP

MAX_W, W_RANGE, END_GAP, SEED_MAX = 0.40, (0.25, 0.60), 0.20, 0.40   # 옆 광선 최대 길이, 길 폭 범위, 닫힌 끝 앞 여유, 씨앗에서 벽 찾는 거리(m)
FAN, SMOOTH, TURN = np.radians([-8, -4, 0, 4, 8]), 0.20, 0.3          # 옆 광선 각도 묶음, 고르게 할 길이(m), 방향 따라가는 비율


class Grid:
    def __init__(self, path):
        info = yaml.safe_load(open(path)); img = cv2.imread(os.path.join(os.path.dirname(path), info["image"]), cv2.IMREAD_UNCHANGED)
        self.img, self.res, self.o = img, float(info["resolution"]), np.array(info["origin"][:2], float)
        self.occ = img < 255 * (1 - info["occupied_thresh"]) if not info.get("negate") else img > 255 * info["occupied_thresh"]
        self.H, self.W = img.shape

    def rc(self, P):
        P = np.atleast_2d(P); return (self.H - 1 - np.floor((P[:, 1] - self.o[1]) / self.res)).astype(int), np.floor((P[:, 0] - self.o[0]) / self.res).astype(int)

    def ray(self, p, a, lim):
        """p에서 각도 a로 lim까지 가며 처음 만나는 벽 칸까지의 거리(없으면 None)."""
        s = np.arange(0.0, lim, self.res / 4); P = p + np.c_[np.cos(a) * s, np.sin(a) * s]; r, c = self.rc(P)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W); hit = np.zeros(len(s), bool); hit[ok] = self.occ[r[ok], c[ok]]
        return float(s[np.argmax(hit)]) if hit.any() else None

    def side(self, p, h, sign):
        d = [self.ray(p, h + sign * math.pi / 2 + f, MAX_W) for f in FAN]; d = [v * math.cos(f) for v, f in zip(d, FAN) if v is not None]
        return float(np.median(d)) if len(d) >= 3 else None


def centre(g, p, h):
    """p를 가운데로 옮긴 점과 폭. 벽이 없거나 폭이 범위 밖이면 None."""
    dl, dr = g.side(p, h, +1), g.side(p, h, -1)
    if dl is None or dr is None or not W_RANGE[0] <= dl + dr <= W_RANGE[1]:
        return None
    n = np.array([-math.sin(h), math.cos(h)]); return p + n * (dl - dr) / 2, dl + dr


def trace(g, p, h):
    """가운데 선을 한쪽으로 따라간 점열(첫 점 포함)과 폭, 멈춘 까닭."""
    P, Wd = [p], [None]
    while True:
        if (f := g.ray(P[-1], h, END_GAP + STEP)) is not None and f <= END_GAP:
            return P, Wd, "닫힌 끝"
        c = centre(g, P[-1] + STEP * np.array([math.cos(h), math.sin(h)]), h)
        if c is None:
            return P, Wd, "열린 끝"
        d = c[0] - P[-1]; h += TURN * math.atan2(math.sin(math.atan2(d[1], d[0]) - h), math.cos(math.atan2(d[1], d[0]) - h))
        P.append(c[0]); Wd.append(c[1])
        if len(P) > 2000:
            return P, Wd, "너무 김"


def road(g, p, h):
    u = np.array([math.cos(h), math.sin(h)])
    for k in range(int(SEED_MAX / STEP) + 1):                           # 씨앗 자리에 벽이 없으면 앞으로 나아가며 찾는다
        if (c := centre(g, p + u * k * STEP, h)) is not None:
            break
    else:
        sys.exit("씨앗에서 %.2f m 안에 양쪽 벽이 없음" % SEED_MAX)
    F, Fw, why_f = trace(g, c[0], h); B, Bw, why_b = trace(g, c[0], h + math.pi)
    P = np.array(B[:0:-1] + F); Wd = np.array(Bw[:0:-1] + [c[1]] + Fw[1:], float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; k = max(1, int(SMOOTH / STEP))
    U = np.arange(0.0, s[-1] + 1e-9, STEP)
    C = np.c_[[np.interp(U, s, uniform_filter1d(P[:, i], k, mode="nearest")) for i in (0, 1)]].T
    Wu = np.interp(U, s, uniform_filter1d(Wd, k, mode="nearest"))
    return C, Wu, why_b, why_f


def graph(C, Wu):
    """wall_map.graph()와 같은 규칙. 옆 위치는 그 자리의 폭으로 정한다."""
    n = len(C); t = np.gradient(C, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True); N = np.c_[-t[:, 1], t[:, 0]]
    head = np.arctan2(t[:, 1], t[:, 0]); lane_w = Wu / LANES
    D = lambda k, i: Wu[i] / 2 - lane_w[i] * (k + 0.5)
    U = np.arange(n) * STEP
    pt = lambda u, d: (np.interp(u, U, C[:, 0]) + d * np.interp(u, U, N[:, 0]), np.interp(u, U, C[:, 1]) + d * np.interp(u, U, N[:, 1]))
    P = [np.array(pt(U[i], D(k, i))) for k in range(LANES) for i in range(n)]; yaw = [float(head[i]) for k in range(LANES) for i in range(n)]
    zone = [0] * len(P); half = [float(lane_w[i] / 2) for k in range(LANES) for i in range(n)]
    E = [(k * n + i, k * n + i + 1, None, False) for k in range(LANES) for i in range(n - 1)]
    m = int(round(LC_LEN / STEP)); T = np.arange(LC_STEP, LC_LEN - 1e-9, LC_STEP) / LC_LEN
    for i0 in range(0, n - m, int(round(LC_EVERY / STEP))):
        for a in range(LANES):
            for b in (a - 1, a + 1):
                if not 0 <= b < LANES:
                    continue
                Q = []
                for tt in T:                                             # 옆 위치를 부드럽게(양 끝 기울기 0)
                    i = i0 + tt * m; j = int(round(i)); da, db = D(a, j), D(b, j); Q.append(np.array(pt(U[i0] + tt * LC_LEN, da + (db - da) * (3 * tt * tt - 2 * tt ** 3))))
                ids = [a * n + i0] + list(range(len(P), len(P) + len(Q))) + [b * n + i0 + m]; full = [P[ids[0]]] + Q + [P[ids[-1]]]
                yaw += [math.atan2(q1[1] - q0[1], q1[0] - q0[0]) for q0, q1 in zip(full[:-2], full[2:])]
                P += Q; zone += [1] * len(Q); half += [float(lane_w[i0 + m // 2] / 2)] * len(Q)
                E += [(ids[0], ids[1], float(np.linalg.norm(full[1] - full[0])) + LC_PENALTY, True)] + [(p, q, None, True) for p, q in zip(ids[1:-1], ids[2:])]
    return np.array(P), np.array(yaw), zone, half, E, n


def overlay(g, C, P, n, path):
    k = 8; img = cv2.cvtColor(cv2.resize(g.img, (g.W * k, g.H * k), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    px = lambda Q: np.c_[(np.asarray(Q)[:, 0] - g.o[0]) / g.res * k, (g.H - (np.asarray(Q)[:, 1] - g.o[1]) / g.res) * k].astype(np.int32)
    cv2.polylines(img, [px(C)], False, (0, 160, 0), 2, cv2.LINE_AA)
    for j, col in enumerate([(0, 0, 255), (255, 0, 0)][:LANES]):
        cv2.polylines(img, [px(P[j * n:(j + 1) * n])], False, col, 2, cv2.LINE_AA)
    for q, col in ((C[0], (0, 200, 255)), (C[-1], (200, 0, 200))):
        cv2.circle(img, tuple(int(v) for v in px([q])[0]), 8, col, -1)
    cv2.imwrite(path, img)


def main():
    g, out = Grid(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
    if sys.argv[3].endswith(".npy"):
        r = np.load(sys.argv[3])[0]; x, y, h = float(r[1]), float(r[2]), float(r[3])
    else:
        x, y, h = float(sys.argv[3]), float(sys.argv[4]), math.radians(float(sys.argv[5]))
    C, Wu, why_b, why_f = road(g, np.array([x, y]), h)
    P, yaw, zone, half, E, n = graph(C, Wu); N = len(P); r4 = lambda v: [round(float(v[0]), 4), round(float(v[1]), 4)]
    assert N < 65535, "노드가 uint16 범위를 넘음"
    feats = [{"type": "Feature", "properties": {"id": i, "frame": "map"}, "geometry": {"type": "Point", "coordinates": r4(p)}} for i, p in enumerate(P)]
    for k, (a, b, cost, lc) in enumerate(E):
        props = {"id": N + k, "startid": a, "endid": b, "metadata": {"lane_vision": not lc, **({"lane_change": True} if lc else {})}}
        if cost is not None:
            props.update(cost=round(cost, 6), overridable=False)
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "MultiLineString", "coordinates": [[r4(P[a]), r4(P[b])]]}})
    json.dump({"type": "FeatureCollection", "name": "map_graph", "features": feats}, open(out + "/lane_graph.geojson", "w"))
    json.dump({"map": None, "routes": [{"name": f"lane{k}", "kind": "left", "start": k * n, "goal": (k + 1) * n - 1, "half": float(np.median(Wu)) / LANES / 2} for k in range(LANES)],
               "xy": P.round(4).tolist(), "yaw": np.round(yaw, 5).tolist(), "zone": zone, "wl": np.round(half, 4).tolist(), "wr": np.round(half, 4).tolist(), "stops": []},
              open(out + "/route_nodes.json", "w"))
    json.dump({"drawing_to_map": dict(theta=0.0, tx=0.0, ty=0.0), "source": "map_graph.py"}, open(out + "/align.json", "w"), indent=1)
    h0 = float(np.arctan2(*(C[1] - C[0])[::-1]))
    yaml.safe_dump({"amcl": {"ros__parameters": {"set_initial_pose": True, "initial_pose": {"x": float(C[0][0]), "y": float(C[0][1]), "z": 0.0, "yaw": h0}}}},
                   open(out + "/amcl_init.yaml", "w"))
    overlay(g, C, P, n, out + "/overlay.png")
    turn = math.degrees(np.unwrap(yaw[:n])[-1] - yaw[0]); lc = sum(1 for e in E if e[2] is not None)
    print(f"길 가운데 선 {(n - 1) * STEP:.2f} m (뒤쪽: {why_b}, 앞쪽: {why_f}), 꺾인 각 {turn:.1f}도")
    print(f"폭: 중앙 {np.median(Wu) * 1000:.0f} mm, 최소 {Wu.min() * 1000:.0f}, 최대 {Wu.max() * 1000:.0f} mm / 차로 {LANES}줄")
    print(f"노드 {N}(차로마다 {n}), 엣지 {len(E)}, 차선 변경 길 {lc}개 / 차로 시작·끝 노드:", [(k * n, (k + 1) * n - 1) for k in range(LANES)])
    print("출발 자세(지도): 길 가운데 x %.3f, y %.3f, 방향 %.1f도" % (C[0][0], C[0][1], math.degrees(h0)), "→", out)


if __name__ == "__main__":
    main()
