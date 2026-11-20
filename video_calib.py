"""보정값을 모르는 주행 영상의 바닥 평면(BEV) 변환 행렬을, 저장해 둔 차선 검출 결과만으로 만든다.

가정: 바닥은 평평하고 카메라는 좌우로 기울지 않았다. 그러면 화면 (u, v) → 바닥 (x 오른쪽, z 앞, m)는
    x = a (u - u0) / (v - v0),   z = b / (v - v0)      ((u0, v0) = 차선들이 모이는 소실점)
- 소실점: 차선 조각들을 직선으로 늘려 가장 많이 모이는 점 (프레임마다 구해 중앙값)
- a: 바닥에 펼쳐 묶은 차선들의 이웃 간격이 --lane-width(m)가 되도록 정함
- b: 같은 차선 위 점선 조각의 반복 주기가 --dash-period(m)가 되도록 정함 (도심 도로 규격: 3m 도색 + 5m 간격 = 8m)
차로 폭과 점선 주기는 가정값이므로, 실제와 다르면 거리 눈금이 그만큼 달라진다.

사용법:
    python video_calib.py test1_clip4_dets.json.gz ~/lane/H_test1.npy
"""
import argparse
import gzip
import json
import os

import numpy as np

from aihub_calib import fit_line, vanishing_point
from lane_geom import Camera, frame_to_pieces
from lane_group import build_lanes

CONF = 0.4        # 보정에는 확실한 검출만 사용
Z_CHECK = 12.0    # 차선 간격을 재는 거리(차량 앞, m)


def lane_polys(frame):
    """한 프레임의 흰색 차선 폴리곤(픽셀)들."""
    return [np.array(p[2:], np.float64).reshape(-1, 2) for p in frame["polys"] if p[0] == 0 and p[1] >= CONF]


def find_vp(frames):
    """프레임마다 소실점을 구해 중앙값을 돌려준다. (소실점, 쓴 프레임 수)"""
    vps = []
    for fr in frames:
        lines = [ln for ln in (fit_line(p) for p in lane_polys(fr)) if ln is not None]
        out = vanishing_point(lines) if len(lines) >= 3 else None
        if out is not None and out[1] >= 3:
            vps.append(out[0])
    return np.median(vps, axis=0), len(vps)


def camera(vp, a, b):
    """소실점과 두 눈금(a, b)으로 변환 행렬을 담은 카메라를 만든다."""
    cam = Camera.__new__(Camera)
    cam.K, cam.dist = None, None
    cam.H = np.array([[a, 0, -a * vp[0]], [0, 0, b], [0, 1, -vp[1]]])
    cam.Hinv = np.linalg.inv(cam.H)
    return cam


def measure(frames, cam):
    """지금의 눈금으로 바닥에 펼쳐 조각을 묶은 뒤, 차선 간격·점선 주기·조각 길이(m)를 잰다."""
    gaps, periods, lengths = [], [], []
    for fr in frames:
        polys = [("white_lane", p) for p in lane_polys(fr)]
        lanes = build_lanes(frame_to_pieces(polys, cam)[0])
        xs = []
        for ln in lanes:
            if ln.kind == "dashed" and len(ln.pieces) >= 3:
                start = [p.start[1] for p in ln.pieces]
                periods += list(np.diff(start))
                lengths += [p.length for p in ln.pieces[:3]]     # 가까운 조각일수록 길이가 정확함
            p = ln.sample(0.5)
            k = int(np.argmin(np.abs(p[:, 1] - Z_CHECK)))
            if ln.length > 8 and abs(p[k, 1] - Z_CHECK) < 1:
                xs.append(p[k, 0])
        xs = sorted(xs)
        gaps += [q - p for p, q in zip(xs, xs[1:])]
    return np.array(gaps), np.array(periods), np.array(lengths)


def mode_of(v, lo, hi):
    """가장 흔한 값(최빈 구간의 중앙값). 빠진 조각 때문에 생긴 두 배·세 배 값은 걸러진다."""
    v = v[(v > lo) & (v < hi)]
    hist, edges = np.histogram(v, bins=40)
    k = int(np.argmax(hist))
    near = v[(v >= edges[max(k - 1, 0)]) & (v <= edges[min(k + 2, 40)])]
    return float(np.median(near))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets", help="dump_dets.py가 저장한 검출 결과 (.json.gz)")
    ap.add_argument("out_npy", help="저장할 변환 행렬 (.npy)")
    ap.add_argument("--lane-width", type=float, default=3.3, help="이웃한 차선 사이 간격 가정(m)")
    ap.add_argument("--dash-period", type=float, default=8.0, help="점선의 반복 주기 가정(m)")
    ap.add_argument("--step", type=int, default=10, help="몇 프레임마다 한 장씩 볼지")
    args = ap.parse_args()

    with gzip.open(os.path.expanduser(args.dets), "rt", encoding="utf-8") as f:
        frames = json.load(f)["frames"][::args.step]
    vp, n = find_vp(frames)
    print(f"소실점: ({vp[0]:.0f}, {vp[1]:.0f})  (프레임 {n}개의 중앙값)")

    # 대강의 눈금에서 시작해, 잰 값이 가정값과 같아지도록 눈금을 고쳐 나간다 (묶기 결과가 눈금에 따라 달라지므로 반복)
    a, b = 1.4, 1.4 * 1400.0                     # 카메라 높이 1.4m, 초점거리 1400픽셀 정도에서 출발
    for it in range(4):
        gaps, periods, lengths = measure(frames, camera(vp, a, b))
        gap, period = mode_of(gaps, 1.5, 6.0), mode_of(periods, 3.0, 20.0)
        print(f"{it + 1}회: 차선 간격 {gap:.2f}m (표본 {len(gaps)}), 점선 주기 {period:.2f}m (표본 {len(periods)}), "
              f"조각 길이 중앙값 {np.median(lengths):.2f}m")
        a, b = a * args.lane_width / gap, b * args.dash_period / period
    print(f"결과: a = {a:.3f} (카메라 높이에 해당), b = {b:.1f} (초점거리로 치면 약 {b / a:.0f}픽셀)")
    print(f"점선 조각 길이 / 주기 = {np.median(lengths) / period:.2f} (3m 도색 + 5m 간격 규격이면 0.375)")

    np.save(os.path.expanduser(args.out_npy), camera(vp, a, b).H)
    for v in (1080, 900, 800, 750):
        if v - vp[1] > 5:
            print(f"  화면 높이 {v}픽셀 → 앞 {b / (v - vp[1]):.1f}m, 1픽셀당 옆 {a / (v - vp[1]) * 100:.1f}cm")
    print(f"저장: {args.out_npy}")


if __name__ == "__main__":
    main()
