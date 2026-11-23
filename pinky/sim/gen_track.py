"""트랙 도로 자동 생성기: 외곽 도로 + 안쪽 도로 + 구역(블록)으로 된 도로를 무작위로 만든다(배치와 경로는 track_layout.py).

  python gen_track.py --count 50 --out ~/pinky_sim/roads_t1 --seed 0
도로마다 폴더 하나(gen_roads.py와 같은 형식에 routes를 더함):
  floor.png, inst.png(선마다 번호), road.json(클래스, 경로, routes = 경로마다 종류·교차로 구간·들어갈 차로 번호·좌우 선까지의 거리), preview.jpg
번호를 붙이는 것: 흰 선(차로 사이 점선, 가장자리선, 외곽 바깥 선, 들머리 점선, 흰 중앙선), 정지선, 횡단보도.
번호를 붙이지 않는 것: 노면 화살표, 노란 중앙선, 블록. 사진에는 나오지만 라벨이 없어 모델은 차선이 아닌 것으로 배운다.
"""
import argparse
import json
import os

import cv2
import numpy as np

from gen_roads import pieces, strip
from track_layout import ARMS, DIN, Layout, fillet

CLASSES = ["white_lane", "stop_line", "crosswalk"]
PPM = 1.0                                                  # 1 mm당 픽셀 수


def dense(p0, p1, step=4.0):
    n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
    return np.linspace(p0, p1, n)


def build(rng):
    L, U = Layout(rng), rng.uniform
    lw, a, W, H = L.lw, L.a, L.W, L.H
    t, dash, gap = U(6, 14), U(30, 80), U(30, 80)          # 선 굵기, 점선 한 칸, 간격
    edge = rng.choice(["dash", "solid", "none"], p=[0.6, 0.2, 0.2])     # 블록 옆 가장자리선
    outer, e = rng.choice(["dash", "solid"], p=[0.6, 0.4]), U(8, 20)    # 외곽 바깥 선, 가장자리선을 도로 끝에서 띄우는 거리
    yellow_mid, mouth = rng.random() < 0.7, rng.random() < 0.7
    marks, plain = [], []                                  # marks: (클래스, 다각형) 번호를 붙임 / plain: (다각형, 색) 번호 없음
    def line(pts, style, cls="white_lane"):
        pts = np.asarray(pts, float)
        for p in (pieces(pts, dash, gap, U(0, dash + gap)) if style == "dash" else [pts]):
            if len(p) >= 2:
                marks.append((cls, strip(p, t)))
    px = lambda p: np.c_[np.asarray(p)[:, 0] * PPM, (H - np.asarray(p)[:, 1]) * PPM].round().astype(np.int32)
    size = (int(np.ceil(H * PPM)), int(np.ceil(W * PPM)))
    # 블록: 둥근 네모들을 한 그림에 칠한다(맞닿은 블록은 하나로 합쳐짐)
    bmask = np.zeros(size, np.uint8)
    for b in L.blocks.values():
        x0, x1, y0, y1 = b["box"]
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        r, c = b["r"], [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        poly, _ = fillet([((x0 + x1) / 2, y0)] + c + [((x0 + x1) / 2, y0)], [max(r[k], 0.5) for k in ("bl", "tl", "tr", "br")], 3.0)
        cv2.fillPoly(bmask, [px(poly)], 1)
    if edge != "none":                                     # 가장자리선: 블록에서 e만큼 떨어진 둘레
        grown = (cv2.distanceTransform(1 - bmask, cv2.DIST_L2, 5) < e * PPM).astype(np.uint8)
        for c in cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            q = c[::4, 0].astype(float)
            line(np.c_[q[:, 0] / PPM, H - q[:, 1] / PPM][np.r_[np.arange(len(q)), 0]], edge)
    # 외곽 바깥 선: 블록 바깥 모서리와 같은 중심으로 도는 둥근 네모
    o, (R0, R1, R2, R3) = L.m + e, [r + lw / 2 - e for _, _, r in L.rc]
    line(fillet([(W / 2, H - o), (W - o, H - o), (W - o, o), (o, o), (o, H - o), (W / 2, H - o)], [R1, R2, R3, R0], 3.0)[0], outer)
    eb = lw / 2 - e if edge != "none" else lw / 2          # 차로 중심에서 블록 쪽 선까지
    ax = lambda X, c, v: (c, v) if X in "NS" else (v, c)   # 갈래 X에서 줄 위치 c, 도로 방향 좌표 v → (x, y)
    ringv = {"N": H - a, "S": a, "E": W - a, "W": a}       # 갈래가 외곽과 만나는 쪽의 블록 끝
    for X in L.has:
        A, d = L.arm[X], np.array(DIN[X], float)
        v0, (ve, vs) = ringv[X], A["end"]                  # 외곽 쪽 끝, 교차로 쪽 끝(들어가는 차로, 나가는 차로)
        sg = float(np.sign(d @ np.array(ax(X, 0, 1))))     # 도로 방향 좌표가 교차로 쪽으로 커지면 +
        ins, out, lo, hi = A["c_in"], A["c_out"], *L.seg[X]
        junction = len(L.has) > 2                          # 안쪽 도로가 하나뿐이면 교차로가 없으므로 정지선·횡단보도를 그리지 않음
        cw = junction and rng.random() < 0.25 and abs(ve - v0) > 3.5 * lw
        back = U(0.9, 1.3) * lw * cw + U(0.05, 0.2) * lw   # 정지선을 교차로 끝에서 물리는 거리(횡단보도가 있으면 그 뒤로)
        for c0, c1 in zip(ins[:-1], ins[1:]):              # 들어가는 차로 사이 점선(정지선까지)
            line(dense(ax(X, (c0 + c1) / 2, v0 + sg * U(0, 0.6) * lw), ax(X, (c0 + c1) / 2, ve - sg * back)), "dash")
        mid = (ins[-1] + out) / 2                          # 중앙선: 들어가는 쪽과 나가는 쪽 사이
        far = ve if sg * (ve - vs) < 0 else vs
        if abs(far - v0) > lw:
            ml = strip(dense(ax(X, mid, v0), ax(X, mid, far)), t)
            plain.append((ml, (40, 215, 230))) if yellow_mid else marks.append(("white_lane", ml))
        rect = lambda c0, c1, va, vb: np.array([ax(X, c0, va), ax(X, c1, va), ax(X, c1, vb), ax(X, c0, vb)])
        st = U(18, 30)
        side = 1 if ins[0] < out else -1                   # 들어가는 차로들이 놓인 쪽
        if junction and rng.random() < 0.85:               # 정지선: 들어가는 차로 전체를 가로지름
            marks.append(("stop_line", rect(ins[0] - side * lw / 2, mid, ve - sg * back, ve - sg * (back + st))))
        if rng.random() < 0.7 and abs(vs - v0) > 1.5 * lw:  # 나가는 차로가 외곽에 닿기 전의 정지선
            marks.append(("stop_line", rect(mid, out + side * lw / 2, v0 + sg * 0.2 * lw, v0 + sg * (0.2 * lw + st))))
        if cw:                                             # 횡단보도: 갈래 전체 폭, 줄무늬는 도로 방향
            ct, v1, cl = U(25, 35), ve - sg * U(0.05, 0.15) * lw, U(0.8, 1.0) * lw
            for c in np.arange(lo + ct, hi - ct, 2 * ct):
                marks.append(("crosswalk", rect(c, c + ct, v1, v1 - sg * cl)))
        if rng.random() < 0.7:                             # 외곽 도로의 정지선(들머리 앞)
            s = A["s_in"][0] - lw / 2 - U(0.2, 0.45) * lw
            p, q = np.array(L.pt(s)), np.array(L.pt(s - st))
            n = np.array([-(p - q)[1], (p - q)[0]]) / st
            marks.append(("stop_line", np.array([p + n * lw / 2, p - n * lw / 2, q - n * lw / 2, q + n * lw / 2])))
        if mouth and edge != "none":                       # 들머리를 가로지르는 점선(외곽 도로의 안쪽 선을 이어 줌)
            s0, s1 = A["s_in"][0] - A["r_in"], A["s_out"] + A["r_out"]
            pts = np.array([L.pt(s) for s in np.arange(s0, s1, 4.0)])
            u = (pts[-1] - pts[0]) / np.linalg.norm(pts[-1] - pts[0])
            line(pts + eb * np.array([u[1], -u[0]]), "dash")
        if rng.random() < 0.7:                             # 화살표(번호 없음): 차로마다 허용 동작 모양
            for c, mv in list(zip(ins, A["lanes"])) + [(out, None)]:
                f = d if mv is not None else -d            # 그 차로의 진행 방향
                vm = (v0 + (ve if mv is not None else vs)) / 2
                p0, lf = np.array(ax(X, c, vm), float), np.array([-f[1], f[0]])
                for m_ in sorted(mv if mv is not None else {"R"}):
                    tip = p0 + f * 0.45 * lw + {"S": 0, "L": 1, "R": -1}[m_] * lf * 0.3 * lw
                    plain.append((np.array([p0 - f * 0.45 * lw, p0 + f * 0.1 * lw, tip]), "arrow"))
    return L, marks, plain, bmask, dict(t=t, dash=dash, gap=gap, eb=eb, ob=lw / 2 - e, edge=str(edge), outer=str(outer))
