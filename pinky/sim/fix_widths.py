"""도로 폴더의 road.json에 경로 점마다 왼쪽·오른쪽 선까지 거리(wl, wr, mm)를 적는다. 바닥 그림의 칠해진 차선(white_lane)에서 잰다.
모서리에서 안쪽 선과 바깥 선이 같은 중심의 원이 아니어서 차로 폭이 바뀌는 도로(road_v5의 링 모서리)를 위한 것.
  교차로 밖: 경로의 법선 방향으로 지금 적힌 거리 ±WIN mm 안에서 칠해진 선의 가운데를 찾는다.
             점선의 빈 곳처럼 못 찾은 점은 앞뒤에서 찾은 값으로 잇고, 그 구간에서 하나도 못 찾으면 지금 값을 그대로 둔다.
  교차로 안: 지금처럼 들어가는 값 → 나오는 값으로 바뀐다.
  경로가 휘는 곳(곡률 > CURV, 앞뒤 MARGIN mm 포함)만 바꾼다. 곧은 곳에서는 교차로 입구처럼 선이 바깥으로 벌어지는 곳을 차로 폭으로 잘못 읽기 때문.
사용: python fix_widths.py ROAD_DIR"""
import json, sys
import cv2, numpy as np
WIN, SMOOTH = 45.0, 9          # 찾는 범위(mm), 다듬는 점 수(중앙값)
CURV, MARGIN = 0.5, 100.0      # 휘는 곳으로 보는 곡률(1/m), 그 앞뒤로 넓히는 거리(mm)

road = sys.argv[1]
info = json.load(open(road + "/road.json")); ppm = info["ppm"]
inst = cv2.imread(road + "/inst.png", cv2.IMREAD_UNCHANGED); H, W = inst.shape[:2]
lane = np.isin(inst, [int(k) for k, v in info["instances"].items() if v == "white_lane"])


def measure(P, n, d0):
    """점 P에서 법선 n 쪽으로, 거리 d0 근처의 칠해진 선 가운데까지 거리(mm). 없으면 nan."""
    offs = np.arange(d0 - WIN, d0 + WIN + 0.5, 1.0)
    pts = P + offs[:, None] * n
    u, v = np.round(pts[:, 0] * ppm).astype(int), np.round(H - pts[:, 1] * ppm).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); hit = np.zeros(len(offs), bool); hit[ok] = lane[v[ok], u[ok]]
    if not hit.any():
        return np.nan
    runs, a = [], None                                    # 이어진 칠한 구간들의 가운데
    for i, h in enumerate(np.r_[hit, False]):
        if h and a is None: a = i
        if not h and a is not None: runs.append(offs[(a + i - 1) // 2]); a = None
    return float(min(runs, key=lambda c: abs(c - d0)))


def fill(v, keep):
    """nan을 앞뒤 값으로 잇고 중앙값으로 다듬는다. 하나도 없으면 keep."""
    ok = ~np.isnan(v)
    if not ok.any():
        return keep
    v = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
    pad = np.r_[[v[0]] * (SMOOTH // 2), v, [v[-1]] * (SMOOTH // 2)]
    return np.array([np.median(pad[i:i + SMOOTH]) for i in range(len(v))])


for p, r in zip(info["paths"], info["routes"]):
    p = np.asarray(p, float); t = np.gradient(p, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]]
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1).mean()
    h = max(1, int(25.0 / ds)); q = np.r_[[p[0]] * h, p, [p[-1]] * h]                    # 곡률은 앞뒤 25 mm 떨어진 점으로(좌표 반올림 잡음을 피함)
    a = np.unwrap(np.arctan2(*(q[2 * h:] - q[:-2 * h]).T[::-1]))
    k = np.abs(np.gradient(a)) / ds * 1000                                                 # 곡률(1/m)
    m = int(MARGIN / ds); bend = np.convolve(k > CURV, np.ones(2 * m + 1), "same") > 0
    (l0, r0), (l1, r1) = r["bounds_in"], r["bounds_out"]
    f = np.clip((np.arange(len(p)) - r["en"]) / max(r["ex"] - r["en"], 1), 0, 1)
    wl, wr = l0 + (l1 - l0) * f, r0 + (r1 - r0) * f
    for a, b in ((0, r["en"]), (r["ex"] + 1, len(p))):     # 교차로 앞, 나온 뒤
        if b <= a:
            continue
        ml = np.array([measure(p[i], n[i], wl[i]) for i in range(a, b)])
        mr = np.array([measure(p[i], -n[i], wr[i]) for i in range(a, b)])
        fl, fr = fill(ml, wl[a:b]), fill(mr, wr[a:b])
        wl[a:b], wr[a:b] = np.where(bend[a:b], fl, wl[a:b]), np.where(bend[a:b], fr, wr[a:b])
    r["wl"], r["wr"] = np.round(wl, 1).tolist(), np.round(wr, 1).tolist()
    print(f"{r['name']}: 왼쪽 {wl.min():.0f}~{wl.max():.0f} mm, 오른쪽 {wr.min():.0f}~{wr.max():.0f} mm")
json.dump(info, open(road + "/road.json", "w"))
