"""gen_track.py의 뒷부분: 만든 도로를 그림과 road.json으로 저장한다(파일당 200줄 제한 때문에 나눔)."""
import json
import os

import cv2
import numpy as np

from gen_track import CLASSES, PPM, build
from track_layout import fillet

COLORS = {"white_lane": (0, 220, 255), "stop_line": (0, 0, 255), "crosswalk": (255, 120, 0)}
KIND = {"straight": (255, 200, 0), "left": (60, 60, 255), "right": (0, 165, 255)}
STICK = 0.35                                               # 가는 막대 화살표로 그리는 도로의 비율(나머지는 굵은 머리 화살표)
ARROW_T = 0.0                                              # 막대 화살표의 선 굵기(mm). 도로마다 save()에서 정함


def arrow(floor, q, wv, st):
    """화살표 하나를 바닥 그림에 그린다. q = (꼬리, 꺾이는 곳, 끝) 픽셀, st = 도로마다 정한 모양(없으면 가는 막대 화살표).
    굵은 모양: 몸통 선 + 속을 채운 세모 머리. 회전 화살표는 꺾인 뒤 옆으로 눕고, 길이·굵기·머리 크기는 도로마다 다르다."""
    if st is None:
        cv2.polylines(floor, [q[:2]], False, (wv, wv, wv), st0 := max(2, int(ARROW_T * PPM)))
        cv2.arrowedLine(floor, tuple(map(int, q[1])), tuple(map(int, q[2])), (wv, wv, wv), st0, tipLength=0.55)
        return
    q = q.astype(float); c = (q[0] + q[1]) / 2; f = (q[1] - q[0]) / (np.linalg.norm(q[1] - q[0]) + 1e-9)
    side = (q[2] - q[1]) - ((q[2] - q[1]) @ f) * f                                       # 끝이 옆으로 간 만큼(직진이면 0)
    tip = q[2] if np.linalg.norm(side) < 1 else q[1] + f * st["lean"] * np.linalg.norm(side) + side
    a, b, tip = c + (q[0] - c) * st["k"], c + (q[1] - c) * st["k"], c + (tip - c) * st["k"]
    u = (tip - b) / (np.linalg.norm(tip - b) + 1e-9); base = tip - u * st["hl"]; n = np.array([-u[1], u[0]]) * st["hw"] / 2
    cv2.polylines(floor, [np.round([a, b, base]).astype(np.int32)], False, (wv, wv, wv), st["t"], cv2.LINE_AA)
    cv2.fillPoly(floor, [np.round([tip, base + n, base - n]).astype(np.int32)], (wv, wv, wv), cv2.LINE_AA)


def save(rng, d):
    L, marks, plain, bmask, v = build(rng)
    H, W, lw = L.H, L.W, L.lw
    px = lambda p: np.c_[np.asarray(p)[:, 0] * PPM, (H - np.asarray(p)[:, 1]) * PPM].round().astype(np.int32)
    floor = np.full(bmask.shape + (3,), int(rng.uniform(5, 35)), np.uint8)
    hue = rng.uniform(22, 34) if rng.random() < 0.65 else rng.uniform(0, 180)          # 블록 색: 노란색 위주, 일부는 다른 색
    col = cv2.cvtColor(np.uint8([[[hue, rng.uniform(120, 255), rng.uniform(150, 240)]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    floor[bmask > 0] = col
    if rng.random() < 0.6:                                                             # 보드 끝의 테두리 선
        cv2.rectangle(floor, (3, 3), (floor.shape[1] - 4, floor.shape[0] - 4), col, int(rng.uniform(4, 10)))
    inst = np.zeros(bmask.shape, np.uint16)
    global ARROW_T
    ar, ARROW_T = v.pop("ar"), v["t"] * 1.3                                             # 화살표 모양: 도로마다 가는 막대(STICK 비율) 또는 굵은 머리
    st = None if ar.random() < STICK else dict(k=ar.uniform(0.9, 1.5), lean=ar.uniform(0.0, 0.5), t=max(2, int(v["t"] * ar.uniform(0.9, 1.8) * PPM)),
                                               hw=ar.uniform(0.16, 0.28) * lw * PPM, hl=ar.uniform(0.15, 0.4) * lw * PPM)
    for poly, c in plain:                                                              # 번호 없는 것: 노란 중앙선, 화살표
        if c == "arrow":
            arrow(floor, px(poly), int(rng.uniform(215, 255)), st)
        elif c == "arrow_r":                                                           # 외곽 화살표: 밝기도 화살표용 난수로(도로의 나머지 모양을 건드리지 않게)
            arrow(floor, px(poly), int(ar.uniform(215, 255)), st)
        else:
            cv2.fillPoly(floor, [px(poly)], c)
    for i, (cls, poly) in enumerate(marks, start=1):                                   # 번호 붙이는 것
        p, wv = np.asarray(poly) + rng.normal(0, 0.3, np.shape(poly)), int(rng.uniform(215, 255))
        cv2.fillPoly(floor, [px(p)], (wv, wv, wv)), cv2.fillPoly(inst, [px(p)], i)
    paths, routes, half = [], [], {"R": (v["ob"], v["eb"])}
    for X, A in L.arm.items():                                                         # 차로 중심에서 왼쪽·오른쪽 선까지
        for k in range(len(A["c_in"])):
            half[(X, "in", k)] = (lw / 2, v["eb"] if k == 0 else lw / 2)
        half[(X, "out", 0)] = (lw / 2, v["eb"])
    near = lambda q, p: int(np.argmin(np.linalg.norm(q - np.array(p, float), axis=1)))
    for name, kind, slot, pts, rad, zone, li, lo in L.routes:
        q, arcs = fillet(pts, rad)
        en, ex = arcs[zone] if isinstance(zone, int) else (near(q, zone[0]), near(q, zone[1]))
        paths.append(np.round(q, 1).tolist())
        routes.append(dict(name=name, kind=kind, slot=slot, en=en, ex=ex, bounds_in=[round(x, 1) for x in half["R" if li[0] == "R" else li]],
                           bounds_out=[round(x, 1) for x in half["R" if lo[0] == "R" else lo]]))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "floor.png"), floor), cv2.imwrite(os.path.join(d, "inst.png"), inst)
    info = dict(type="track", size_mm=[W, H], ppm=PPM, classes=CLASSES, lane_w=lw, lanes_per_dir=1, line_w=v["t"], dash=v["dash"], gap=v["gap"],
                edge=v["edge"], outer=v["outer"], arms={X: ["".join(sorted(m)) for m in A["lanes"]] for X, A in L.arm.items()},
                instances={str(k): c for k, (c, _) in enumerate(marks, start=1)}, paths=paths, routes=routes, path_right=[True] * len(paths))
    json.dump(info, open(os.path.join(d, "road.json"), "w"))
    pv = floor.copy()
    for k, (cls, _) in enumerate(marks, start=1):
        pv[inst == k] = COLORS[cls]
    for q, r in zip(paths, routes):
        cv2.polylines(pv, [px(q)], False, (60, 255, 60), 3)
    for q, r in zip(paths, routes):
        cv2.polylines(pv, [px(q[r["en"]:r["ex"] + 1])], False, KIND[r["kind"]], 6)
    s = 800 / max(pv.shape[:2])
    cv2.imwrite(os.path.join(d, "preview.jpg"), cv2.resize(pv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20), ap.add_argument("--out", default="~/pinky_sim/roads_t1"), ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for i in range(a.count):
        d = os.path.join(os.path.expanduser(a.out), f"road_{a.seed:02d}_{i:04d}")
        info = save(np.random.default_rng(a.seed * 100000 + i), d)
        kinds = [r["kind"] for r in info["routes"]]
        print(f"{os.path.basename(d)}: 갈래 {info['arms']}, 차로 폭 {info['lane_w']:.0f}mm, 크기 {info['size_mm'][0]:.0f}x{info['size_mm'][1]:.0f}mm, "
              f"선 {len(info['instances'])}개, 경로 직진 {kinds.count('straight')} 좌 {kinds.count('left')} 우 {kinds.count('right')}, 가장자리선 {info['edge']}")
