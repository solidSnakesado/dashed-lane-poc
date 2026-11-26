"""도면(wall_map.py의 블럭 길)을 SLAM 지도에 맞춘다: 도면의 벽 면 전체를 지도의 벽 칸에 겹치는 변환(회전 + 이동)을 찾고,
그 변환으로 도로 그래프를 지도 좌표로 옮겨 내보낸다. 사람이 점을 찍지 않는다.

방법: 지도의 벽 칸(점유)까지 거리 지도를 만들고, 도면 벽 면 위의 점들(5 mm 간격)을 옮겼을 때 그 거리의 제곱 평균이 가장 작은 변환을 찾는다.
  회전 2도·이동 5 cm 간격으로 전부 훑은 뒤 Nelder-Mead로 다듬는다. 거리는 CLIP(m)에서 자른다(지도에 없는 벽이 결과를 끌지 않게).
결과(OUT 폴더):
  align.json         변환(도면 → 지도: 회전 rad, 이동 m), 어긋남 통계, 기준점 4곳의 지도 좌표와 그 자리의 어긋남, 도면 출발 자세의 지도 좌표
  lane_graph.geojson, route_nodes.json  지도 좌표로 옮긴 그래프(노드 좌표와 방향만 바뀜, 번호·엣지·비용은 같음)
  overlay.png        지도(8배 확대)에 옮긴 도면 벽(빨강)을 겹친 그림
  amcl_init.yaml     AMCL을 켤 때의 초기 위치 = 길 가운데 출발 자리(지도 좌표). lane_nav.launch.py amcl:= 로 준다
어긋남 두 가지: 도면 벽 점 → 가장 가까운 지도 벽 칸(지도에서 벽이 빠졌거나 휘었는지), 지도 벽 칸 → 가장 가까운 도면 벽(지도에 없는 벽이 찍혔는지).
사용: python map_align.py MAP_YAML GRAPH_DIR OUT      (GRAPH_DIR = wall_map.py의 출력 폴더: blocks.json, refs.json, lane_graph.geojson, route_nodes.json)
"""
import json, math, os, sys
import cv2, numpy as np, yaml
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree

CLIP, STEP = 0.2, 0.005


class Map:
    def __init__(self, path):
        info = yaml.safe_load(open(path)); img = cv2.imread(os.path.join(os.path.dirname(path), info["image"]), cv2.IMREAD_UNCHANGED)
        occ = img < 255 * (1 - info["occupied_thresh"]) if not info.get("negate") else img > 255 * info["occupied_thresh"]
        self.img, self.occ, self.res, self.o = img, occ, float(info["resolution"]), np.array(info["origin"][:2], float)
        self.H, self.W = img.shape; self.dt = distance_transform_edt(~occ) * self.res
        r, c = np.nonzero(occ); self.walls = np.c_[self.o[0] + (c + 0.5) * self.res, self.o[1] + (self.H - r - 0.5) * self.res]

    def rc(self, P):
        return self.H - 0.5 - (P[:, 1] - self.o[1]) / self.res, (P[:, 0] - self.o[0]) / self.res - 0.5

    def dist(self, P):
        return map_coordinates(self.dt, self.rc(P), order=1, mode="constant", cval=CLIP)


def rot(th):
    c, s = math.cos(th), math.sin(th); return np.array([[c, -s], [s, c]])


def apply(p, P):
    return np.asarray(P) @ rot(p[0]).T + p[1:]


def wall_points(blocks, idx=None):
    P = []
    for i, b in enumerate(blocks):
        if idx is None or i in idx:
            a, c = np.array(b["face_from"]), np.array(b["face_to"]); n = max(1, int(np.linalg.norm(c - a) / STEP))
            P += [a + t * (c - a) for t in np.linspace(0, 1, n + 1)]
    return np.array(P)


def fit(m, D):
    cost = lambda p: float(np.mean(np.minimum(m.dist(apply(p, D)), CLIP) ** 2))
    lo, hi = m.walls.min(0), m.walls.max(0); c0 = D.mean(0); best = (math.inf, None)
    for th in np.radians(np.arange(-180, 180, 2)):
        Rc = rot(th) @ c0
        for tx in np.arange(lo[0], hi[0], 0.05) - Rc[0]:
            for ty in np.arange(lo[1], hi[1], 0.05) - Rc[1]:
                v = cost((th, tx, ty))
                if v < best[0]:
                    best = (v, (th, tx, ty))
    r = minimize(cost, best[1], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=5000))
    return np.array(r.x)


def stats(d):
    d = np.asarray(d) * 1000; return dict(median=round(float(np.median(d)), 1), p95=round(float(np.percentile(d, 95)), 1), max=round(float(d.max()), 1))


def export(gdir, out, p):
    g = json.load(open(gdir + "/lane_graph.geojson"))
    for f in g["features"]:
        geo = f["geometry"]
        if geo["type"] == "Point":
            geo["coordinates"] = [round(float(v), 4) for v in apply(p, [geo["coordinates"]])[0]]
        else:
            geo["coordinates"] = [[[round(float(v), 4) for v in q] for q in apply(p, line)] for line in geo["coordinates"]]
    json.dump(g, open(out + "/lane_graph.geojson", "w"))
    d = json.load(open(gdir + "/route_nodes.json"))
    d["xy"] = np.round(apply(p, d["xy"]), 4).tolist(); d["yaw"] = np.round(np.array(d["yaw"]) + p[0], 5).tolist()
    d["map"] = None                                                      # 도면 그림(map.jpg)은 지도 좌표와 맞지 않으므로 뺀다
    json.dump(d, open(out + "/route_nodes.json", "w"))


def overlay(m, blocks, p, path):
    k = 8; img = cv2.cvtColor(cv2.resize(m.img, (m.W * k, m.H * k), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for b in blocks:
        r, c = m.rc(apply(p, [b["face_from"], b["face_to"]])); q = [(int((c[i] + 0.5) * k), int((r[i] + 0.5) * k)) for i in (0, 1)]
        cv2.line(img, q[0], q[1], (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    m, gdir, out = Map(sys.argv[1]), sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    blocks = json.load(open(gdir + "/blocks.json")); refs = json.load(open(gdir + "/refs.json")); D = wall_points(blocks)
    p = fit(m, D); Dm = apply(p, D); inv = (m.walls - p[1:]) @ rot(p[0])
    extra, _ = cKDTree(D).query(inv)
    res = dict(rotation_deg=round(math.degrees(p[0]), 3), translation_m=[round(float(v), 4) for v in p[1:]],
               drawing_to_map=dict(theta=float(p[0]), tx=float(p[1]), ty=float(p[2])),
               drawing_wall_to_map_wall_mm=stats(m.dist(Dm)), map_wall_to_drawing_wall_mm=dict(stats(extra), over_40mm=int((extra > 0.04).sum()), cells=len(extra)),
               refs={k: dict(map=[round(float(v), 4) for v in apply(p, [q])[0]], miss_mm=round(float(m.dist(apply(p, [q]))[0]) * 1000, 1)) for k, q in refs.items()})
    walls = {b["wall"] for b in blocks}
    res["per_wall_mm"] = {w: stats(m.dist(apply(p, wall_points(blocks, {i for i, b in enumerate(blocks) if b["wall"] == w})))) for w in sorted(walls)}
    nodes = json.load(open(gdir + "/route_nodes.json"))
    res["lane_starts_in_map"] = [dict(node=r["start"], x=round(float(v[0]), 4), y=round(float(v[1]), 4), yaw_deg=round(math.degrees(nodes["yaw"][r["start"]] + p[0]), 2))
                                 for r, v in zip(nodes["routes"], apply(p, [nodes["xy"][r["start"]] for r in nodes["routes"]]))]
    json.dump(res, open(out + "/align.json", "w"), indent=1, ensure_ascii=False)
    export(gdir, out, p); overlay(m, blocks, p, out + "/overlay.png")
    st = np.mean([nodes["xy"][r["start"]] for r in nodes["routes"]], 0); s0 = apply(p, [st])[0]        # 길 가운데 출발 자리(map_drive.py가 서는 곳)
    yaw0 = float(np.mean([nodes["yaw"][r["start"]] for r in nodes["routes"]])) + p[0]
    yaml.safe_dump({"amcl": {"ros__parameters": {"set_initial_pose": True, "initial_pose": {"x": float(s0[0]), "y": float(s0[1]), "z": 0.0, "yaw": float(yaw0)}}}},
                   open(out + "/amcl_init.yaml", "w"))
    print(f"도면 → 지도: 회전 {res['rotation_deg']}도, 이동 {res['translation_m']} m")
    print("도면 벽 → 지도 벽 칸 mm:", res["drawing_wall_to_map_wall_mm"], "| 벽별:", res["per_wall_mm"])
    print("지도 벽 칸 → 도면 벽 mm:", res["map_wall_to_drawing_wall_mm"])
    for k, v in res["refs"].items():
        print(f"  기준점 {k}: 지도 {v['map']}, 그 자리의 어긋남 {v['miss_mm']} mm")
    print("차로 출발 자세(지도):", res["lane_starts_in_map"], "→", out)


if __name__ == "__main__":
    main()
