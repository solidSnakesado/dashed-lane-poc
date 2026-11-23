"""gen_track.py의 뒷부분: 만든 도로를 그림과 road.json으로 저장한다(파일당 200줄 제한 때문에 나눔)."""
import json
import os

import cv2
import numpy as np

from gen_track import CLASSES, PPM, build
from track_layout import fillet

COLORS = {"white_lane": (0, 220, 255), "stop_line": (0, 0, 255), "crosswalk": (255, 120, 0)}
KIND = {"straight": (255, 200, 0), "left": (60, 60, 255), "right": (0, 165, 255)}


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
    for poly, c in plain:                                                              # 번호 없는 것: 노란 중앙선, 화살표
        if c == "arrow":
            q, wv = px(poly), int(rng.uniform(215, 255))
            cv2.polylines(floor, [q[:2]], False, (wv, wv, wv), max(2, int(v["t"] * 1.3 * PPM)))
            cv2.arrowedLine(floor, tuple(map(int, q[1])), tuple(map(int, q[2])), (wv, wv, wv), max(2, int(v["t"] * 1.3 * PPM)), tipLength=0.55)
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
