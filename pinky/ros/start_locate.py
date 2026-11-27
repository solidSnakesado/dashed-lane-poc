"""로봇을 어디에 두든 스캔을 SLAM 지도에 맞춰 처음 위치를 찾고, 그 자리에서 출발할 차로 노드를 고른다(2026-11-27).

track_runner.py의 출발 노드 자리에 here를 주면 쓴다(예: here 201). 사람이 위치를 알려 주지 않는다.
방법(map_align.py와 같은 생각): 지도 벽 칸까지의 거리 지도를 만들고, 후보 자세에서 스캔 점이 벽에서 얼마나 떨어지는지(CAP에서 자름)의 평균을 점수로 한다.
  1) 후보 = 길 안(가장 가까운 차로 노드에서 차로 폭 안)의 지도 칸 전부 × 방향 STEP_DEG 간격. 스캔 점은 N_PTS개로 줄여 계산.
  2) 점수가 좋은 후보부터 서로 떨어진(DIST_SEP m 또는 ANG_SEP도) 것 TOP개를 Nelder-Mead로 다듬는다(점 전부, 거리 지도는 선형 보간).
  3) 가장 좋은 점수가 GOOD보다 나쁘거나, 떨어진 두 번째 후보의 점수가 첫째의 UNIQUE배 안이면 위치를 하나로 정할 수 없다고 보고 출발하지 않는다.
출발 노드: 찾은 자리에서 가장 가까운, 방향이 HEAD_OK도 안인 차로 노드. 없으면(길 방향과 맞지 않게 놓임) 출발하지 않는다(사용자 결정: 방향은 길에 맞춰 놓음).
차로 사이에 놓였을 때(2026-11-27 사용자 요구: 달리면서 차로를 맞춰 가야 함): 그 차로 가운데에서 옆으로 ENTRY_MIN보다 떨어져 있으면,
  로봇 앞 LEAD에서 지금 옆 위치로 시작해 옆/앞 SLOPE 이하(최소 RAMP)로 부드럽게 차로에 들어가는 진입 노드를 그래프에 덧붙여(Route Server에 다시 읽힘,
  아직 추적 전이라 안전) 그 첫 노드에서 출발한다. 주행이 끝나면 graph_event.Swap.restore가 원래 그래프로 되돌린다.
라이다 자세: TF base_footprint ← 스캔 프레임(고정)으로 스캔 점을 로봇 기준으로 옮겨 쓴다.
"""
import json, math, os, time
import cv2, numpy as np, rclpy, yaml
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sensor_msgs.msg import LaserScan
from nav2_msgs.srv import SetRouteGraph

CAP, STEP_DEG, N_PTS, TOP = 0.10, 3.0, 180, 12          # 점수에서 자르는 거리(m), 방향 간격(도), 줄인 점 수, 다듬을 후보 수
DIST_SEP, ANG_SEP, GOOD, UNIQUE, HEAD_OK = 0.15, 20.0, 0.03, 1.5, 60.0
ENTRY_MIN, LEAD, RAMP, SLOPE, DS = 0.015, 0.05, 0.25, 0.5, 0.02     # 진입 노드: 만드는 옆 거리, 시작 앞 거리, 가장 짧은 길이(m), 옆/앞 한도, 점 간격


class Locator:
    def __init__(self, node, map_yaml, xy, yaw, zone, half, scan_topic, swap=None):
        self.n, self.scan, self.swap, self.all_xy = node, None, swap, np.asarray(xy)
        info = yaml.safe_load(open(map_yaml)); img = cv2.imread(map_yaml.rsplit("/", 1)[0] + "/" + info["image"], cv2.IMREAD_UNCHANGED)
        occ = img < 255 * (1 - info["occupied_thresh"]); self.res, self.o, self.H = float(info["resolution"]), np.array(info["origin"][:2]), img.shape[0]
        self.dt = np.minimum(distance_transform_edt(~occ) * self.res, CAP)  # 벽 칸까지 거리(m, CAP에서 자름). 행 0이 지도 위쪽
        self.all_yaw = np.asarray(yaw); lane = np.where(np.asarray(zone) == 0)[0]; self.xy, self.yaw, self.half = np.asarray(xy)[lane], np.asarray(yaw)[lane], np.asarray(half)[lane]; self.ids = lane
        r, c = np.nonzero(img >= 250)                                      # 빈 칸 가운데 길 안인 칸이 후보 자리
        P = np.c_[self.o[0] + (c + 0.5) * self.res, self.o[1] + (self.H - r - 0.5) * self.res]; d, j = cKDTree(self.xy).query(P)
        self.cand = P[d < 2 * self.half[j]]
        node.create_subscription(LaserScan, scan_topic, lambda m: setattr(self, "scan", m), qos_profile_sensor_data)

    def score(self, pts, x, y, h):
        """스캔 점(로봇 기준)들을 자세 (x, y, h)에 놓았을 때 벽까지 거리의 평균(작을수록 잘 맞음). 지도 밖은 CAP."""
        X = x + pts[:, 0] * math.cos(h) - pts[:, 1] * math.sin(h); Y = y + pts[:, 0] * math.sin(h) + pts[:, 1] * math.cos(h)
        return float(np.mean(map_coordinates(self.dt, [self.H - 0.5 - (Y - self.o[1]) / self.res, (X - self.o[0]) / self.res - 0.5], order=1, cval=CAP)))

    def scan_points(self, wait=10.0):
        """(로봇 기준 스캔 점 또는 None, 못 받았을 때 까닭). 스캔과 TF를 따로 기다려 무엇이 없는지 알린다."""
        t0, t, err = time.monotonic(), None, "아직 시도 안 함"
        while time.monotonic() - t0 < wait and t is None:
            rclpy.spin_once(self.n, timeout_sec=0.05)
            if self.scan is not None:
                try:
                    t = self.n.buf.lookup_transform("base_footprint", self.scan.header.frame_id, Time()).transform
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
        if self.scan is None:
            return None, f"{wait:.0f}초 동안 스캔을 하나도 받지 못함"
        if t is None:
            return None, f"TF base_footprint ← {self.scan.header.frame_id}를 {wait:.0f}초 동안 못 찾음({err[:160]})"
        m = self.scan
        r = np.asarray(m.ranges, float); a = m.angle_min + np.arange(len(r)) * m.angle_increment; ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]); q = t.rotation; h = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.c_[t.translation.x + x * math.cos(h) - y * math.sin(h), t.translation.y + x * math.sin(h) + y * math.cos(h)], ""

    def locate(self):
        """(자세 [x, y, h] 또는 None, 설명)."""
        T = [time.monotonic()]; pts, why = self.scan_points(); T.append(time.monotonic())   # 단계별 걸린 시간 재기(2026-11-27)
        if pts is None or len(pts) < 30:
            return None, why or f"스캔 점이 {len(pts)}개뿐임"
        sub = pts[np.linspace(0, len(pts) - 1, min(N_PTS, len(pts))).astype(int)]; S = []
        for h in np.radians(np.arange(0, 360, STEP_DEG)):                  # 방향마다 후보 자리 전부의 점수(가장 가까운 칸 값)
            X = self.cand[:, :1] + sub[:, 0] * math.cos(h) - sub[:, 1] * math.sin(h); Y = self.cand[:, 1:] + sub[:, 0] * math.sin(h) + sub[:, 1] * math.cos(h)
            r = np.floor(self.H - (Y - self.o[1]) / self.res).astype(int); c = np.floor((X - self.o[0]) / self.res).astype(int)
            ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.dt.shape[1]); v = np.full(r.shape, CAP); v[ok] = self.dt[r[ok], c[ok]]
            S.append(np.c_[self.cand, np.full(len(self.cand), h), v.mean(1)])
        S = np.concatenate(S); S = S[np.argsort(S[:, 3])]; picked = []; T.append(time.monotonic())
        for x, y, h, _ in S:
            if all(math.hypot(x - p[0], y - p[1]) > DIST_SEP or abs((h - p[2] + math.pi) % (2 * math.pi) - math.pi) > math.radians(ANG_SEP) for p in picked):
                picked.append((x, y, h))
            if len(picked) >= TOP:
                break
        best = sorted(((r.fun, r.x) for r in (minimize(lambda v: self.score(pts, *v), p, method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-5, initial_simplex=[p, p + np.r_[0.02, 0, 0], p + np.r_[0, 0.02, 0], p + np.r_[0, 0, 0.05]])) for p in map(np.array, picked))),
                      key=lambda t: t[0])
        (f1, p1), rest = best[0], [b for b in best[1:] if math.hypot(*(b[1][:2] - best[0][1][:2])) > DIST_SEP
                                  or abs((b[1][2] - best[0][1][2] + math.pi) % (2 * math.pi) - math.pi) > math.radians(ANG_SEP)]
        T.append(time.monotonic()); self.took = f"스캔·TF 받기 {T[1] - T[0]:.1f}s, 후보 검사 {T[2] - T[1]:.1f}s, 다듬기 {T[3] - T[2]:.1f}s"
        f2 = rest[0][0] if rest else float("inf"); p1[2] = (p1[2] + math.pi) % (2 * math.pi) - math.pi
        note = f"처음 위치: x {p1[0]:.3f}, y {p1[1]:.3f}, 방향 {math.degrees(p1[2]):.1f}도 / 점수 {f1 * 1000:.1f} mm(다음 후보 {f2 * 1000:.1f} mm, 점 {len(pts)}개)"
        if f1 > GOOD:
            return None, note + " → 지도와 잘 맞지 않아 출발하지 않음"
        if f2 < f1 * UNIQUE:
            return None, note + " → 비슷한 자리가 둘 이상이라 출발하지 않음"
        return p1, note

    def start(self):
        """(출발 노드 번호, 자세, 설명). 못 정하면 (None, None, 설명)."""
        t0 = time.time() - os.stat("/proc/self").st_ctime                  # 프로그램 시작부터 여기까지(ROS 연결, 서버 기다림)
        self.took = ""; p, note = self.locate(); t1 = time.monotonic()
        print(f"[걸린 시간] 시작~위치 찾기 전 {t0:.1f}s / {self.took}", flush=True)
        if p is None:
            return None, None, note
        d, k = cKDTree(self.xy).query(p[:2], k=min(50, len(self.xy)))
        ok = (np.abs((self.yaw[k] - p[2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(HEAD_OK)) & (d < 2 * self.half[k])
        if not ok.any():
            return None, None, note + " → 가까운 차로 노드 중 방향이 맞는 것이 없음(반대를 보고 놓였거나 길 밖)"
        a = int(self.ids[k[ok][0]]); e = self.entry(a, p); print(f"[걸린 시간] 진입 노드 만들고 읽히기 {time.monotonic() - t1:.1f}s", flush=True)
        return (a, p, note + f" → 출발 노드 {a}") if e is None else (e[0], p, note + f" → 차로 가운데에서 옆 {e[1] * 1000:+.0f} mm라 진입 노드 {e[2]}개로 노드 {e[3]}에 들어감")

    def entry(self, a, p):
        """차로 밖(사이)이면 진입 노드를 붙인 그래프를 읽히고 (첫 노드 번호, 옆 거리, 노드 수, 들어가는 차로 노드)를, 아니면 None."""
        h = self.yaw[np.searchsorted(self.ids, a)]; dp = p[:2] - self.all_xy[a]; l = float(-dp[0] * math.sin(h) + dp[1] * math.cos(h))
        if abs(l) < ENTRY_MIN or self.swap is None:
            return None
        ids = np.asarray(self.ids); same = ids[ids > a]; run = [a]          # 같은 차로의 앞쪽 노드(번호가 이어짐)
        for i in same:
            if i != run[-1] + 1 or np.linalg.norm(self.all_xy[i] - self.all_xy[run[-1]]) > 0.03:   # 번호가 끊기거나 다음 차로로 넘어감
                break
            run.append(int(i))
        P = self.all_xy[run]; arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        s0 = max(0.0, float(dp[0] * math.cos(h) + dp[1] * math.sin(h)) + LEAD); L = max(RAMP, abs(l) / SLOPE)
        if s0 + L > arc[-1]:
            return None
        b = int(np.searchsorted(arc, s0 + L)); S = np.arange(s0, arc[b] - DS / 2, DS); t = np.clip((S - s0) / (arc[b] - s0), 0, 1)
        off = l * (1 - (3 * t ** 2 - 2 * t ** 3)); hh = np.interp(S, arc, np.unwrap(self.all_yaw[run]))
        Q = np.c_[np.interp(S, arc, P[:, 0]) - off * np.sin(hh), np.interp(S, arc, P[:, 1]) + off * np.cos(hh)]
        sw = self.swap; feats = json.load(open(sw.gdir + "/lane_graph.geojson"))["features"]; nid = max(f["properties"]["id"] for f in feats) + 1
        r4 = lambda q: [round(float(q[0]), 4), round(float(q[1]), 4)]; new = list(range(nid, nid + len(Q))); nid += len(Q); pos = {**dict(zip(new, Q)), run[b]: P[b]}
        feats += [{"type": "Feature", "properties": {"id": i, "frame": "map"}, "geometry": {"type": "Point", "coordinates": r4(q)}} for i, q in zip(new, Q)]
        for u, v in zip(new, new[1:] + [run[b]]):
            feats.append({"type": "Feature", "properties": {"id": nid, "startid": u, "endid": v, "metadata": {"entry": True}},
                          "geometry": {"type": "MultiLineString", "coordinates": [[r4(pos[u]), r4(pos[v])]]}}); nid += 1
        path = sw.gdir + "/lane_graph_entry.geojson"; json.dump({"type": "FeatureCollection", "name": "entry", "features": feats}, open(path, "w"))
        g = sw.call(sw.setg, SetRouteGraph.Request(graph_filepath=path))
        if g is None or not g.success:
            return None
        sw.closed = sw.closed or []                                      # 주행이 끝나면 원래 그래프로 되돌리게 함
        return new[0], l, len(Q), run[b]
