"""후처리 결과(lane_drive.csv)를 시뮬 정답(gt.csv)과 비교해 숫자로 판정한다 (6-4단계).

사용 예: python pinky/sim/eval_drive.py $D/lane_drive.csv $D/gt.csv $D/calib_scaled.npz
- 단위: 결과 CSV는 실제 도로 크기(축척 배) 좌표라서 calib의 scale로 나눠 실제 m로 맞춤
- 짝: L1 ↔ 정답 왼쪽 선, L2 ↔ 정답 오른쪽 선, 중심선 ↔ 정답 중심선
- 카메라 화면 밖의 정답 점은 평가에서 뺌
- 직선·곡선 구간은 정답 중심선이 휘는 정도(반경)로 나눔
출력: 화면 요약표, 결과 CSV 옆에 eval_drive.csv (프레임별 오차와 판정)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lane_geom import Camera  # noqa: E402  (저장소 루트의 바닥 ↔ 화면 변환)

ROLES = (("lane1", "left"), ("lane2", "right"), ("center", "center"))
EXTRA = 0.10     # 결과 선 끝에서 이 거리(m)까지는 끝 방향으로 늘려서 비교
OK_ERR = 0.03    # 이 오차(m) 안이면 정답 선으로 판정
NEAR_ERR = 0.06  # 이 오차(m) 안이면 '3~6cm' (같은 선이지만 어긋남)
GOAL_ERR = 0.02  # 목표: 옆 오차 중앙값(m)
GOAL_BOTH = 0.95 # 목표: 양쪽 선 비율


def read_result(path, scale):
    """결과 CSV → {프레임 번호: {역할: (점열 (N,2) 실제 m, 추적 번호)}}"""
    pts = defaultdict(lambda: defaultdict(list))
    tid = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["role"] == "other":
                continue
            k = int(r["frame"].lstrip("f"))
            pts[k][r["role"]].append((float(r["x_m"]) / scale, float(r["z_m"]) / scale))
            tid[k][r["role"]] = int(r["track_id"])
    return {k: {role: (np.array(v), tid[k][role]) for role, v in d.items()} for k, d in pts.items()}


def x_at(p, z):
    """점열 p가 앞 z(m)를 지날 때의 옆 위치. 범위 밖이면 끝에서 EXTRA까지만 늘려 구함. 못 구하면 None."""
    if len(p) < 2:
        return None
    p = p[np.argsort(p[:, 1])]
    if p[0, 1] <= z <= p[-1, 1]:
        return float(np.interp(z, p[:, 1], p[:, 0]))
    (a, b), e = ((p[0], p[1]), p[0]) if z < p[0, 1] else ((p[-2], p[-1]), p[-1])
    if abs(e[1] - z) > EXTRA or abs(b[1] - a[1]) < 1e-6:
        return None
    return float(e[0] + (b[0] - a[0]) * (z - e[1]) / (b[1] - a[1]))


def visible(cam, scale, x, z, size):
    """정답 점(실제 m)이 카메라 화면 안에 보이는지. 왜곡 계산이 터지는 점은 되돌려 보아 걸러 냄."""
    if x is None:
        return False
    g = np.array([[x * scale, z * scale]])
    px = cam.to_image(g)
    if not np.all(np.isfinite(px)):
        return False
    u, v = px[0]
    if not (0 <= u < size[0] and 0 <= v < size[1]):
        return False
    back, ok = cam.to_ground(px)
    return bool(ok[0]) and np.linalg.norm(back[0] - g[0]) < 0.05 * scale


def gx(g, name, z):
    """정답 행에서 선 name의 앞 z(m) 옆 위치. 정답이 비어 있으면(선이 그 거리에 없음) None."""
    v = g[f"{name}_{z:.1f}"]
    return float(v) if v else None


def radius(zs, xs):
    """정답 중심선에 2차식을 맞춰 휘는 반경(m)을 구한다."""
    a = np.polyfit(zs, xs, 2)[0]
    return 1.0 / max(abs(2 * a), 1e-6)


def stat(v):
    """오차 목록의 (중앙값, 상위 10%, 개수)."""
    v = np.abs(np.array([e for e in v if e is not None]))
    return (np.median(v), np.percentile(v, 90), len(v)) if len(v) else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser(description="후처리 결과를 시뮬 정답과 비교")
    ap.add_argument("result", help="lane_run.py가 만든 lane_<이름>.csv")
    ap.add_argument("gt", help="sim_drive.py가 만든 gt.csv")
    ap.add_argument("calib", help="lane_run.py에 넣은 calib npz (축척 포함)")
    ap.add_argument("--curve-r", type=float, default=2.0, help="이 반경(m)보다 작으면 곡선 구간")
    ap.add_argument("--size", default="640x480", help="카메라 화면 크기")
    a = ap.parse_args()
    scale = float(np.load(a.calib)["scale"])
    cam, size = Camera(a.calib), tuple(int(s) for s in a.size.split("x"))
    res = read_result(a.result, scale)
    gt = list(csv.DictReader(open(a.gt)))
    zs = sorted({float(c.split("_")[1]) for c in gt[0] if c.startswith("left_")})

    seg_of, rows = {}, []
    errs = defaultdict(lambda: defaultdict(list))    # [구간][역할_거리] → 오차 목록
    cls = defaultdict(lambda: defaultdict(int))       # [구간][역할:판정] → 프레임 수
    both = defaultdict(lambda: [0, 0, 0])             # [구간] → [양쪽 있음, 양쪽 정답, 프레임 수]
    tip = defaultdict(list)                           # [구간] → 먼 쪽 오차의 프레임 간 변화
    prev, prev_far, switches = {}, {}, defaultdict(int)
    for g in gt:
        k = int(g["frame"])
        cz = [z for z in zs if visible(cam, scale, gx(g, "center", z), z, size)]
        cx = [gx(g, "center", z) for z in cz]
        seg = "곡선" if len(cz) >= 3 and radius(np.array(cz), np.array(cx)) < a.curve_r else "직선"
        seg_of[k] = seg
        fr = res.get(k, {})
        row, got, good, far = [k, seg], 0, 0, {}
        for role, name in ROLES:
            line = fr.get(role)
            vis = [z for z in zs if visible(cam, scale, gx(g, name, z), z, size)]
            e = {}
            for z in vis:
                x = None if line is None else x_at(line[0], z)
                e[z] = None if x is None else x - gx(g, name, z)
            for z in (0.2, 0.4):
                if z in e:
                    errs[seg][f"{role}_{z}"].append(e[z])
            # 0.4 m 지점으로 판정
            if 0.4 not in vis:
                c = "평가제외"
            elif line is None or e[0.4] is None:
                c = "없음" if line is None else "0.4m못닿음"
            elif abs(e[0.4]) < NEAR_ERR:
                c = "정답" if abs(e[0.4]) < OK_ERR else "3~6cm"
            else:
                other = {"left": "right", "right": "left"}.get(name)
                xo = None if other is None else gx(g, other, 0.4)
                x = e[0.4] + gx(g, name, 0.4)
                c = "반대선" if xo is not None and abs(x - xo) < OK_ERR else "다른선"
            cls[seg][f"{role}:{c}"] += 1
            if role != "center":
                got += line is not None
                good += c == "정답"
                if line is not None:     # 역할 바뀜: 직전 프레임과 추적 번호가 다르면
                    if role in prev and prev[role] != line[1]:
                        switches[seg] += 1
                    prev[role] = line[1]
            far[role] = {z: v for z, v in e.items() if v is not None}
            row += [c] + [("" if e.get(z) is None else f"{e[z]:.4f}") for z in (0.2, 0.4)]
        both[seg][0] += got == 2
        both[seg][1] += good == 2
        both[seg][2] += 1
        for role in ("lane1", "lane2"):   # 끝 떨림: 두 프레임 모두에 있는 가장 먼 거리에서 오차 변화
            last = prev_far.get(role, {})
            common = sorted(set(last) & set(far[role]))
            if common:
                tip[seg].append(far[role][common[-1]] - last[common[-1]])
        prev_far = far
        rows.append(row)

    out = os.path.join(os.path.dirname(os.path.abspath(a.result)), "eval_drive.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "segment"] + [f"{r}_{c}" for r, _ in ROLES for c in ("판정", "err_0.2", "err_0.4")])
        w.writerows(rows)
    report(errs, cls, both, tip, switches, a.curve_r)
    print(f"저장: {out}")



def report(errs, cls, both, tip, switches, curve_r):
    """구간별(직선·곡선·전체) 요약표를 출력하고 목표 통과 여부를 표시한다."""
    segs = sorted(both) + ["전체"]
    for d in (errs, cls):                     # 구간별 값을 더해 '전체'를 만듦 (목록은 이어 붙이고 개수는 더함)
        for s in list(d):
            for key, v in d[s].items():
                d["전체"][key] = d["전체"].get(key, type(v)()) + v
    both["전체"] = list(np.sum([both[s] for s in segs[:-1]], axis=0))
    tip["전체"] = sum((tip[s] for s in segs[:-1]), [])
    switches["전체"] = sum(switches[s] for s in segs[:-1])
    for s in segs:
        n = both[s][2]
        print(f"\n[{s}] {n}프레임" + (f" (곡선 = 반경 {curve_r} m 미만)" if s == "곡선" else ""))
        for z in (0.2, 0.4):
            lane = stat(errs[s][f"lane1_{z}"] + errs[s][f"lane2_{z}"])
            cen = stat(errs[s][f"center_{z}"])
            mark = "통과" if lane[0] <= GOAL_ERR else "미달"
            print(f"  옆 오차 {z} m  L1·L2 중앙 {lane[0]*100:.1f} cm / 상위10% {lane[1]*100:.1f} cm [{mark}]"
                  f"  | 중심선 중앙 {cen[0]*100:.1f} cm / 상위10% {cen[1]*100:.1f} cm")
        r = both[s][0] / max(n, 1)
        print(f"  양쪽 선 있음 {r:.0%} [{'통과' if r >= GOAL_BOTH else '미달'}]"
              f"  | 양쪽 정답(3 cm 이내) {both[s][1] / max(n, 1):.0%}")
        for role in ("lane1", "lane2", "center"):
            items = sorted((k.split(":")[1], v) for k, v in cls[s].items() if k.startswith(role + ":"))
            print(f"  {role:6s} 판정: " + ", ".join(f"{c} {v}" for c, v in items))
        t = stat(tip[s])
        print(f"  역할 바뀜 {switches[s]}회  | 끝 떨림(프레임 간 오차 변화) 중앙 {t[0]*100:.1f} cm / 상위10% {t[1]*100:.1f} cm")


if __name__ == "__main__":
    main()
