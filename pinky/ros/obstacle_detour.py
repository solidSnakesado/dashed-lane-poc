"""라이다(와 DETOUR_CAM=1이면 앞 카메라, cam_obstacle.py)로 앞쪽 차로의 새 장애물을 찾아 비켜 가는 회피 노드를 만든다(2026-11-27). graph_event.py의 Swap이 쓴다.
새 장애물 점 = 스캔 점 가운데 SLAM 지도의 벽 칸에서 WALL_GAP보다 멀고, 길 안(차로 노드에서 차로 폭 안)이며, 지도의 빈 칸인 점
  (real_c7_14: 바깥 벽 블록 이음매 너머 회색 칸의 점이 장애물로 묶여 통로를 막음). 지도 좌표 VOX 칸에 쌓고 칸마다 찍힌 스캔 수를 세어
  HITS번 넘게 찍힌 칸만 쓴다(real_c7_13: 위치가 0.3초 동안 방향 ±6° 튄 사이 점이 밀려 찍힘). 만들 때마다 GDIR/detour_log/에 칸·회피 길 저장.
찾기: 새 점을 로봇 차로 k의 가운데 선에 투영해(앞 s, 옆 l, 왼쪽 +) NEAR < s − s_로봇 < LOOK, |l| < 차로 반폭 + BODY + MARGIN인 점이
  MIN_PTS개 넘게 CONFIRM_N번 연달아 보이면 장애물. 크기는 걸린 점이 든 덩어리(한 칸 틈까지 이어진 칸)로 재고, 앞뒤 길이는 옆 폭 이상으로 둔다.
회피 길: 로봇 앞 LEAD에서 지금 옆 위치로 시작해 장애물 앞 PAD까지 옆 d로 부드럽게(3t² − 2t³) 옮기고, 장애물 뒤 PAD까지 유지한 뒤 RAMP에 걸쳐
  차로 노드 b로 돌아온다. 옆/앞 비율은 SLOPE까지. 가능 = 점마다 그 자리·방향에 놓은 실물 몸체 사각형(FP, wall_nav.yaml footprint와 같음)이
  지도 벽에서 WALL_M, 회피 구간의 쌓인 점에서 MARGIN + LOW_PAD 넘게 떨어짐(real_c7_12: 원으로 검사해 돌아간 몸체 모서리가 닿음).
  d 고르기: ±D_MAX를 D_STEP 간격으로 지금 옆 위치에서 가까운 순으로 보며 여유(몸체와 장애물·벽 사이 가장 짧은 거리)가 TARGET 이상인 첫 후보,
  없으면 여유가 가장 큰 후보(좁은 틈이면 통로 가운데. real_c7_15: 장애물에서 1 cm로 붙인 길에서 곡선 안쪽으로 질러가 닿음).
다시 만들기: 회피 중에도 점을 쌓고, 앞쪽 회피 길의 몸체가 쌓인 점에서 MARGIN 안이 되면 다시 만든다(REBUILD_DT마다, MAX_BUILD번까지). 닫는 엣지: 끝점 노드의
  몸체가 장애물에서 MARGIN 안인 엣지만. 결과: dict(file, close, start, note), 그래프는 GDIR/lane_graph_avoid.geojson.
"""
import json, math, os, time
from collections import deque
import cv2, numpy as np, yaml
from scipy.ndimage import binary_dilation, distance_transform_edt, label; from scipy.spatial import cKDTree
from rclpy.qos import qos_profile_sensor_data; from rclpy.time import Time
from sensor_msgs.msg import LaserScan; from cam_obstacle import CamPoints

WALL_GAP, LOOK, NEAR, VOX = 0.05, 1.0, 0.15, 0.01         # 지도 벽에서 이보다 먼 점만 / 앞쪽 보는 범위 / 점 쌓는 칸(m)
BODY, MARGIN, WALL_M, LOW_PAD = 0.058, 0.01, 0.01, 0.015   # 몸체 반폭, 장애물·벽과의 여유(사용자 요구: 양쪽 1 cm), 장애물 보정(아래 설명)
# LOW_PAD: 라이다 면(약 12.5 cm)보다 낮게 튀어나온 부분 몫, 장애물 점에만 더함. 핑키 바퀴 포함 110, 몸통 80 → 한쪽 15 mm(사용자 결정 2026-11-27, real_c7_16)
D_STEP, D_MAX = 0.005, 0.40                              # 옆 위치 후보 간격, 범위(차로 가운데 기준 ±m)
FP = (-0.076, 0.042, 0.055)                              # 몸체 사각형: 뒤, 앞, 옆 반폭(m, 로봇 기준). wall_nav.yaml footprint와 같아야 함
LEAD, RAMP, SLOPE, PAD, DS = 0.05, 0.25, 0.8, 0.04, 0.02   # 시작 앞 거리, 돌아오는 길이(m), 비킬 때 옆/앞 비율 한도, 장애물 앞뒤 여유, 점 간격(m)
HITS, TARGET = 3, 0.03                                   # 칸을 쓰는 데 필요한 스캔 수 / 이 여유(m)가 남으면 덜 비킴, 안 남으면 여유가 가장 큰 곳(통로 가운데)으로
MIN_PTS, CONFIRM_N, REBUILD_DT, MAX_BUILD, KEEP = 3, 2, 0.5, 6, 2.0   # KEEP: 스캔 시각의 TF를 기다리는 한도(초)


class Detour:
    def __init__(self, node, gdir, map_yaml, scan_topic):
        self.n, self.gdir, self.count = node, gdir, 0
        self.feats = json.load(open(gdir + "/lane_graph.geojson"))["features"]
        d = json.load(open(gdir + "/route_nodes.json")); r0 = d["routes"][0]; self.lane_n = r0["goal"] - r0["start"] + 1
        self.xy, self.yaw, self.half = np.array(d["xy"]), np.array(d["yaw"]), np.array(d["wl"])
        self.lanes = len(d["routes"]); self.lane_tree = cKDTree(self.xy[:self.lanes * self.lane_n])
        self.arc = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.xy[k * self.lane_n:(k + 1) * self.lane_n], axis=0), axis=1))] for k in range(self.lanes)]
        self.edges = [(f["properties"]["id"], f["properties"]["startid"], f["properties"]["endid"]) for f in self.feats if "startid" in f["properties"]]
        info = yaml.safe_load(open(map_yaml)); img = cv2.imread(map_yaml.rsplit("/", 1)[0] + "/" + info["image"], cv2.IMREAD_UNCHANGED)
        occ = img < 255 * (1 - info["occupied_thresh"]); self.res, self.o, self.H = float(info["resolution"]), np.array(info["origin"][:2]), img.shape[0]
        self.wall, self.free = distance_transform_edt(~occ) * self.res, (img >= 250).astype(float)   # 벽 칸까지 거리(m) / 지도 빈 칸(흰색)=1, 회색·벽=0
        self.cells, self.active, self.builds, self.t_build, self.told = {}, None, 0, 0.0, False
        self.q = deque(maxlen=30); node.create_subscription(LaserScan, scan_topic, lambda m: self.q.append(m), qos_profile_sensor_data)   # rclpy는 내장 함수를 콜백으로 못 받음
        self.cam = CamPoints(node) if os.environ.get("DETOUR_CAM") == "1" else None   # 앞 카메라의 바닥에 닿는 선도 쓰기(비전 5단계, cam_obstacle.py)

    def heading(self, Q):
        return np.arctan2(*np.gradient(Q, axis=0)[:, ::-1].T) if len(Q) > 1 else np.zeros(len(Q))

    def body(self, Q, H, P):
        """포즈 (Q, H)마다 몸체 사각형과 점 P 사이 가장 짧은 거리(m, 사각형 안이면 음수). 0.2 m 안에 점이 없으면 inf."""
        out = np.full(len(Q), np.inf); cx, hx = (FP[0] + FP[1]) / 2, (FP[1] - FP[0]) / 2
        if not len(P) or not len(Q):
            return out
        tree = cKDTree(P)
        for i, near in enumerate(tree.query_ball_point(Q, 0.2)):
            if near:
                v = P[near] - Q[i]; c, sn = math.cos(H[i]), math.sin(H[i])
                qx = np.abs(v[:, 0] * c + v[:, 1] * sn - cx) - hx; qy = np.abs(-v[:, 0] * sn + v[:, 1] * c) - FP[2]
                out[i] = np.min(np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0))
        return out

    def body_wall(self, Q, H):
        """몸체 사각형 둘레(1 cm 간격)와 지도 벽 사이 가장 짧은 거리(m). 가운데가 벽에서 옆 반폭 안이면 -1."""
        e = np.r_[np.arange(FP[0], FP[1], 0.01), FP[1]]; f = np.r_[np.arange(-FP[2], FP[2], 0.01), FP[2]]
        R = np.r_[np.c_[e, np.full(len(e), FP[2])], np.c_[e, np.full(len(e), -FP[2])], np.c_[np.full(len(f), FP[0]), f], np.c_[np.full(len(f), FP[1]), f]]
        c, sn = np.cos(H)[:, None], np.sin(H)[:, None]
        X = Q[:, :1] + R[:, 0] * c - R[:, 1] * sn; Y = Q[:, 1:] + R[:, 0] * sn + R[:, 1] * c
        return float(np.min(self.wall_dist(np.c_[X.ravel(), Y.ravel()]))) if np.all(self.wall_dist(Q) > FP[2]) else -1.0

    def wall_dist(self, P, G=None):
        G = self.wall if G is None else G; r = (self.H - 1 - np.floor((P[:, 1] - self.o[1]) / self.res)).astype(int); c = np.floor((P[:, 0] - self.o[0]) / self.res).astype(int)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.wall.shape[1]); d = np.full(len(P), -1.0); d[ok] = G[r[ok], c[ok]]
        return d

    def points(self):
        """아직 안 쓴 스캔들의 새 장애물 점(지도 좌표). 없으면 None.
        스캔마다 그 스캔이 찍힌 시각의 TF(map ← 라이다)로 옮긴다. 그 시각의 TF가 아직 안 왔으면 KEEP초까지 기다렸다 다시 해 보고, 넘으면 버린다.
        (2026-11-27 실물: 무선 때문에 핑키의 TF가 최대 3초 늦게 와서, 가장 최근 TF로 옮기면 장애물이 엉뚱한 자리에 찍혀 폭이 18 cm 넘게 번졌다)"""
        out, keep = [], deque(maxlen=30)
        newest = max((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in self.q), default=0.0)
        while self.q:
            m = self.q.popleft()
            try:
                t = self.n.buf.lookup_transform("map", m.header.frame_id, Time.from_msg(m.header.stamp)).transform
            except Exception:
                if newest - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9) < KEEP:
                    keep.append(m)
                continue
            out.append(self.to_map(m, t))
        self.q.extend(keep); out += [self.keep(P) for P in self.cam.points(self.n.buf)] if self.cam else []   # 영상 한 장 = 스캔 하나로 셈
        for P in out:                                                    # 스캔 하나에서는 칸마다 한 번만 셈
            for c in set(map(tuple, np.floor(P / VOX).astype(int).tolist())):
                self.cells[c] = self.cells.get(c, 0) + 1
        return np.concatenate(out) if out else None

    def to_map(self, m, t):
        r = np.asarray(m.ranges, float); a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max); x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        q = t.rotation; h = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        P = np.c_[t.translation.x + x * math.cos(h) - y * math.sin(h), t.translation.y + x * math.sin(h) + y * math.cos(h)]
        return self.keep(P)

    def keep(self, P):
        """지도에 없고(벽에서 WALL_GAP 넘게) 길 안이며 지도의 빈 칸인 점만(라이다·카메라 공통)."""
        dl, j = self.lane_tree.query(P) if len(P) else (np.zeros(0), np.zeros(0, int))
        return P[(self.wall_dist(P) > WALL_GAP) & (dl < 2 * self.half[j]) & (self.wall_dist(P, self.free) > 0)] if len(P) else P

    def lane_frame(self, k, P):
        """차로 k 가운데 선 기준 (앞 s, 옆 l, 차로 안 번호 j)."""
        base = k * self.lane_n; L = self.xy[base:base + self.lane_n]; j = cKDTree(L).query(P)[1]; h = self.yaw[base + j]
        dp = P - L[j]; return self.arc[k][j] + dp[:, 0] * np.cos(h) + dp[:, 1] * np.sin(h), -dp[:, 0] * np.sin(h) + dp[:, 1] * np.cos(h), j

    def at(self, k, s, off):
        base = k * self.lane_n; L = self.xy[base:base + self.lane_n]; h = np.interp(s, self.arc[k], np.unwrap(self.yaw[base:base + self.lane_n]))
        return np.c_[np.interp(s, self.arc[k], L[:, 0]) - off * np.sin(h), np.interp(s, self.arc[k], L[:, 1]) + off * np.cos(h)]

    def find(self, pose):
        """새 회피 길이 필요하면 dict(file, close, start, note), 못 피하면 dict(note)(한 번만), 아니면 None."""
        if pose is None or (P := self.points()) is None:
            return None
        C = [c for c, n in self.cells.items() if n >= HITS]; A = (np.array(C, float) + 0.5) * VOX if C else np.zeros((0, 2))
        k = int(self.lane_tree.query(pose[:2])[1]) // self.lane_n if self.active is None else self.active["k"]
        s_r, l_r, _ = self.lane_frame(k, np.asarray(pose[:2])[None]); s_r, l_r = float(s_r[0]), float(l_r[0])
        if self.active is not None:                                       # 회피 중: 지나갔으면 끝, 앞쪽 회피 점이 장애물에 가까워졌으면 다시 만듦
            if s_r > self.active["sb"]:
                self.active, self.builds, self.told = None, 0, False; return None
            ah = self.active["S"] > s_r; Q = self.active["Q"][ah]
            if not len(Q) or not len(A) or np.min(self.body(Q, self.active["H"][ah], A)) - LOW_PAD > MARGIN - 0.005:
                return None
            if time.monotonic() - self.t_build < REBUILD_DT or self.builds >= MAX_BUILD:
                return None
            return self.build(k, s_r, l_r, A, "다시 만듦", A[cKDTree(Q).query(A)[0] <= math.hypot(FP[0], FP[2]) + MARGIN + LOW_PAD])
        if not len(P):
            self.count = 0; return None
        s, l, j = self.lane_frame(k, P); hw = self.half[k * self.lane_n + j]
        hit = (s > s_r + NEAR) & (s < s_r + LOOK) & (np.abs(l) < hw + BODY + MARGIN + LOW_PAD)
        self.count = self.count + 1 if hit.sum() >= MIN_PTS else 0
        return self.build(k, s_r, l_r, A, "처음", P[hit]) if self.count >= CONFIRM_N else None

    def blob(self, A, seeds):
        """쌓인 점 A 가운데 seeds가 들어 있는 덩어리(VOX 칸이 한 칸 틈까지 이어진 것)의 점만."""
        C = np.floor(A / VOX).astype(int); c0 = C.min(0) - 2; G = np.zeros(tuple(C.max(0) - c0 + 3), bool); G[tuple((C - c0).T)] = True
        lab = label(binary_dilation(G, np.ones((3, 3), bool)))[0]; S = np.floor(seeds / VOX).astype(int) - c0
        S = S[np.all((S >= 0) & (S < G.shape), axis=1)]; keep = set(lab[tuple(S.T)].tolist()) - {0}
        return A[np.isin(lab[tuple((C - c0).T)], list(keep))]

    def build(self, k, s_r, l_r, A, why, seeds):
        if not len(seeds) or not len(O := self.blob(A, seeds)):
            return None
        s, l, _ = self.lane_frame(k, O); ahead = s > s_r + NEAR / 2; O, s, l = O[ahead], s[ahead], l[ahead]
        if not len(O):
            return None
        s0, s1, lo, hi = s.min(), s.max(), l.min(), l.max(); s1 = max(s1, s0 + (hi - lo))   # 앞면만 보이므로 앞뒤 ≥ 옆 폭
        sa, end = s_r + LEAD, self.arc[k][-1] - 0.02; sb = min(s1 + PAD + RAMP, end)   # 차로 끝을 넘으면 끝 노드로 돌아옴
        note = f"장애물({why}): 앞 {s0 - s_r:.2f}~{s1 - s_r:.2f} m, 옆 {lo:+.3f}~{hi:+.3f} m(차로 {k} 가운데 기준, 점 {len(O)}개)"
        if s1 + PAD + 2 * DS > end:
            return self.fail(note + " → 장애물 뒤로 차로가 모자라 회피 노드를 만들지 않음")
        b = int(np.searchsorted(self.arc[k], sb)); avail = s0 - PAD - sa; back = self.arc[k][b] - s1 - PAD
        sA, lA, _ = self.lane_frame(k, A); W = A[(sA > sa - 0.1) & (sA < self.arc[k][b] + 0.1) & (np.abs(lA) < D_MAX + 0.1)]   # 회피 구간 안의 점 전부
        # 여유 검사는 걸린 덩어리만이 아니라 구간 안의 점 전부와 함께(같은 날 실물: 두 덩어리를 번갈아 피함)
        best = None                                                      # 빈 통로 찾기(맨 위 설명의 d 고르기): (여유, d, Q, S, H)
        for d in sorted(np.arange(-D_MAX, D_MAX + 1e-9, D_STEP), key=lambda v: abs(v - l_r)):
            if abs(d - l_r) > SLOPE * max(avail, 0.0) + 0.005 or abs(d) > SLOPE * back + 0.005:   # 비키거나 돌아올 앞 거리가 모자람
                continue
            S = np.arange(sa, self.arc[k][b] - DS / 2, DS); up = np.clip((S - sa) / max(avail, DS), 0, 1); dn = np.clip((self.arc[k][b] - S) / max(back, DS), 0, 1)
            off = np.where(S < s0 - PAD, l_r + (d - l_r) * (3 * up ** 2 - 2 * up ** 3), d * (3 * dn ** 2 - 2 * dn ** 3)); Q = self.at(k, S, off); H = self.heading(Q)
            gw, go = self.body_wall(Q, H), float(np.min(self.body(Q, H, W))) - LOW_PAD   # 장애물 쪽 여유는 보정을 뺀 값
            if gw > WALL_M and go > MARGIN - 0.005 and (best is None or min(gw, go) > best[0] or min(gw, go) >= TARGET):
                best = (min(gw, go), d, Q, S, H)
                if best[0] >= TARGET:
                    break
        if best is not None:
            g, d, Q, S, H = best; self.active = dict(k=k, Q=Q, S=S, H=H, sb=float(self.arc[k][b])); self.builds += 1; self.t_build = time.monotonic(); self.told = False; self.save(Q, note)
            return dict(self.write(Q, k * self.lane_n + b, W), note=note + f" → 옆 {d:+.3f} m로 비켜 가는 회피 노드 {len(Q)}개, 여유 {min(g, 1) * 100:.1f} cm(차로 {k} 노드 {b}로 돌아옴)")
        return self.fail(note + f" → 몸체 사각형 + 여유 {MARGIN * 100:.0f} cm로 지나갈 통로가 없거나 비킬 거리가 모자라 회피 노드를 만들지 않음")

    def save(self, Q, note):
        os.makedirs(self.gdir + "/detour_log", exist_ok=True); K = np.array(list(self.cells), int).reshape(-1, 2)
        np.savez(f"{self.gdir}/detour_log/{time.time():.1f}.npz", cells=K, hits=np.array(list(self.cells.values())), vox=VOX, path=np.zeros((0, 2)) if Q is None else Q, note=note)

    def fail(self, note):
        if self.told:
            return None
        self.save(None, note); self.told = True; return dict(note=note, blocked=self.active is not None)   # 회피 중이면 지금 회피 길도 막힘

    def write(self, Q, gb, O):
        """원래 그래프 + 회피 점열(첫 점 → … → gb)을 파일로 쓰고, 닫을 엣지(끝점이 장애물 점 가까이)와 출발 노드 번호를 돌려준다."""
        feats = [dict(f) for f in self.feats]; nid = max(f["properties"]["id"] for f in feats) + 1; r4 = lambda p: [round(float(p[0]), 4), round(float(p[1]), 4)]
        ids = list(range(nid, nid + len(Q))); nid += len(Q); pos = {gb: self.xy[gb], **dict(zip(ids, Q))}
        feats += [{"type": "Feature", "properties": {"id": i, "frame": "map"}, "geometry": {"type": "Point", "coordinates": r4(q)}} for i, q in zip(ids, Q)]
        chain = ids + [gb]
        for u, v in zip(chain[:-1], chain[1:]):
            feats.append({"type": "Feature", "properties": {"id": nid, "startid": u, "endid": v, "metadata": {"detour": True}},
                          "geometry": {"type": "MultiLineString", "coordinates": [[r4(pos[u]), r4(pos[v])]]}}); nid += 1
        dn = self.body(self.xy, self.yaw, O); bad = set(np.where(dn < MARGIN + LOW_PAD)[0].tolist())   # 노드의 몸체가 장애물에 MARGIN + LOW_PAD 안
        close = [e for e, a, b in self.edges if a in bad or b in bad]
        path = self.gdir + "/lane_graph_avoid.geojson"; json.dump({"type": "FeatureCollection", "name": "avoid", "features": feats}, open(path, "w"))
        return dict(file=path, close=close, start=ids[0])
