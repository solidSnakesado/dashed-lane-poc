"""핑키용 도로 바닥 자동 생성기 (실제 도로 1/16 축소 규격 주변에서 무작위).

  python gen_roads.py --count 50 --out ~/pinky_sim/roads --seed 0 [--ppm 2]
도로마다 폴더 하나:
  floor.png  바닥 그림 (검은 바닥, 흰 테이프, 보드 이음매). ppm = 1mm당 픽셀 수
  inst.png   선마다 번호를 칠한 16비트 그림 (0 = 바닥). 3단계에서 카메라 화면으로 옮겨 라벨을 만든다
  road.json  번호별 클래스, 바닥 크기(mm), 주행 경로(차로 중심선, mm), 생성 설정
  preview.jpg 확인용 (차선 노랑, 정지선 빨강, 횡단보도 파랑, 경로 초록)
좌표: mm, x 오른쪽, y 위쪽(앞). 그림으로 그릴 때만 y를 뒤집는다.
"""
import argparse
import json
import os

import cv2
import numpy as np

CLASSES = ["white_lane", "stop_line", "crosswalk"]
COLORS = {"white_lane": (0, 220, 255), "stop_line": (0, 0, 255), "crosswalk": (255, 120, 0)}


def offset(line, d):
    """폴리라인을 진행 방향 왼쪽으로 d만큼 평행 이동."""
    t = np.gradient(line, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return line + np.c_[-t[:, 1], t[:, 0]] * d


def strip(line, w):
    """폴리라인을 폭 w의 테이프 다각형으로."""
    return np.vstack([offset(line, w / 2), offset(line, -w / 2)[::-1]])


def pieces(line, dash, space, phase):
    """폴리라인을 점선 조각들로 자른다."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    on = ((s + phase) % (dash + space)) < dash
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and on[j] == on[i]:
            j += 1
        if on[i] and s[j - 1] - s[i] > dash * 0.3:
            out.append(line[i:j])
        i = j
    return out

# 규격을 뽑는 범위(mm): 차로 폭, 선 굵기, 점선 길이, 점선 간격. 기본은 실제 도로의 1/16 주변, WIDE는 규격이 바뀌는 경우 시험용
RG = {"lane": (180, 230), "line": (9, 11), "dash": (170, 210), "gap": (280, 340)}
WIDE = {"lane": (150, 300), "line": (6, 20), "dash": (100, 350), "gap": (150, 450)}

class Road:
    """테이프 다각형(클래스 포함)과 주행 경로를 모은다."""

    def __init__(self, rng):
        self.rng, self.marks, self.paths = rng, [], []
        self.lane_t = rng.uniform(*RG["line"])           # 차선 테이프 폭
        self.stop_t = rng.uniform(17, 21)                # 정지선 (19mm)
        self.dash, self.space = rng.uniform(*RG["dash"]), rng.uniform(*RG["gap"])   # 점선 길이, 간격

    def solid(self, line):
        self.marks.append(("white_lane", strip(line, self.lane_t)))

    def dashed(self, line):
        for p in pieces(line, self.dash, self.space, self.rng.uniform(0, 500)):
            self.marks.append(("white_lane", strip(p, self.lane_t)))

    def carriageway(self, center, n, lw):
        """중심선 기준 왕복 도로: 가운데 실선, 같은 방향 차로 사이 점선, 양 끝 실선. 차로 중심 경로 저장."""
        self.solid(center)
        for side in (1, -1):
            for k in range(1, n):
                self.dashed(offset(center, side * k * lw))
            if self.rng.random() > 0.15:                 # 가끔 바깥 경계선이 없는 도로
                self.solid(offset(center, side * n * lw))
            for k in range(n):                           # 오른쪽 통행: 진행 방향 오른쪽(-) 차로
                lane = offset(center, -(k + 0.5) * lw)
                self.paths.append((lane if side == 1 else offset(center, (k + 0.5) * lw)[::-1]).tolist())


def winding(rng):
    """구불구불한 왕복 도로 (기본: 편도 2차로)."""
    r = Road(rng)
    n, lw = (2 if rng.random() < 0.75 else 1), rng.uniform(*RG["lane"])
    # 굽은 정도는 가장 급한 곳의 곡선 반경으로 정한다: 0.9~3.0m (실제 15~48m, 도시부 저속 도로의 급한 곡선)
    rad, lam, length = rng.uniform(900, 3000), rng.uniform(2000, 4500), rng.uniform(3000, 4800)
    amp = min(lam ** 2 / (4 * np.pi ** 2 * rad), 600)  # 사인 곡선의 꼭대기 곡률 = 진폭·(2π/주기)²
    w = 2 * (n * lw + amp * 1.3) + rng.uniform(100, 400)
    y = np.arange(0, length + 1, 4.0)
    ph, mix = rng.uniform(0, 2 * np.pi, 2), rng.uniform(0, 0.3)   # 두 번째 사인을 조금 섞어 불규칙하게
    wave = np.sin(2 * np.pi * y / lam + ph[0]) + mix * np.sin(2 * np.pi * y / (lam * 0.47) + ph[1])
    center = np.c_[w / 2 + amp * wave, y]
    r.carriageway(center, n, lw)
    return r, (w, length), {"type": "winding", "lanes_per_dir": n, "lane_w": lw, "amp": amp, "wavelength": lam, "min_radius": rad}


def cross(rng):
    """십자 또는 T자 교차로. 진입부마다 횡단보도와 정지선."""
    r = Road(rng)
    n, lw = (1 if rng.random() < 0.6 else 2), rng.uniform(*RG["lane"])
    half, arm = n * lw, rng.uniform(900, 1800)
    size = 2 * (arm + half)
    c = size / 2
    drop = int(rng.integers(4))                          # 30%는 한 팔을 뺀 T자 교차로
    arms = [0, 1, 2, 3] if rng.random() < 0.7 else [a for a in range(4) if a != drop]
    cw_len, cw_t, cw_gap = rng.uniform(240, 320), rng.uniform(27, 31), rng.uniform(27, 33)
    cw_start, stop_gap = rng.uniform(10, 60), rng.uniform(125, 310)
    for a in arms:
        rot = np.array([[np.cos(a * np.pi / 2), -np.sin(a * np.pi / 2)], [np.sin(a * np.pi / 2), np.cos(a * np.pi / 2)]])
        place = lambda p: (np.asarray(p) @ rot.T) + c   # 동쪽 팔 기준 좌표 → 바닥 좌표
        x = np.arange(half, half + arm + 1, 4.0)
        sx = half + cw_start + cw_len + stop_gap         # 정지선 위치. 중앙선과 점선은 여기서부터 시작
        xs = x[x >= sx]
        r.solid(place(np.c_[xs, np.zeros_like(xs)]))     # 중앙선(일단 흰 실선)
        for k in range(1, n):                            # 같은 방향 차로 사이 점선
            for sgn in (1, -1):
                r.dashed(place(np.c_[xs, np.full_like(xs, sgn * k * lw)]))
        for sgn in (1, -1):
            r.solid(place(np.c_[x, np.full_like(x, sgn * half)]))
        if rng.random() < 0.85:                          # 횡단보도
            k = int((2 * half - 20 + cw_gap) // (cw_t + cw_gap))
            y0 = -(k * cw_t + (k - 1) * cw_gap) / 2
            for i in range(k):
                yy = y0 + i * (cw_t + cw_gap)
                x0 = half + cw_start
                r.marks.append(("crosswalk", place([[x0, yy], [x0 + cw_len, yy], [x0 + cw_len, yy + cw_t], [x0, yy + cw_t]])))
        if rng.random() < 0.85:                          # 정지선: 교차로로 들어가는 쪽(서쪽으로 가는 차의 오른쪽 = 위쪽)
            r.marks.append(("stop_line", place([[sx, 0], [sx + r.stop_t, 0], [sx + r.stop_t, half], [sx, half]])))
        for k in range(n):                               # 경로: 들어오는 차로(북쪽, 서쪽으로 진행) → 직진, 좌회전
            yk = (k + 0.5) * lw
            inn = np.c_[x[::-1], np.full_like(x, yk)]
            if (a + 2) % 4 in arms:                      # 직진: 교차로를 지나 서쪽 팔의 같은 차로로
                r.paths.append(place(np.vstack([inn, np.c_[-x, np.full_like(x, yk)]])).tolist())
            if k == 0 and (a + 3) % 4 in arms:           # 좌회전: 남쪽 팔로. 원호 중심 (half, yk - R)
                R = half + yk
                t = np.linspace(np.pi / 2, np.pi, 40)
                arc = np.c_[half + R * np.cos(t), yk - R + R * np.sin(t)]
                down = np.c_[np.full_like(x, -yk), -x]
                r.paths.append(place(np.vstack([inn, arc, down])).tolist())
    if drop not in arms:                                 # T자: 빠진 팔 쪽은 도로 경계 실선으로 막음
        rot = np.array([[np.cos(drop * np.pi / 2), -np.sin(drop * np.pi / 2)], [np.sin(drop * np.pi / 2), np.cos(drop * np.pi / 2)]])
        yy = np.arange(-half, half + 1, 4.0)
        r.solid(np.c_[np.full_like(yy, half), yy] @ rot.T + c)
    return r, (size, size), {"type": "cross", "lanes_per_dir": n, "lane_w": lw, "arms": arms, "arm_len": arm}


def render(road, size, ppm, rng):
    """바닥 그림과 번호 그림을 같은 다각형으로 그린다(그림과 라벨이 어긋나지 않게)."""
    W, H = int(np.ceil(size[0] * ppm)), int(np.ceil(size[1] * ppm))
    floor = np.full((H, W, 3), int(rng.uniform(5, 30)), np.uint8)
    inst = np.zeros((H, W), np.uint16)
    to_px = lambda p: np.c_[p[:, 0] * ppm, H - p[:, 1] * ppm].round().astype(np.int32)
    for i, (cls, poly) in enumerate(road.marks, start=1):
        p = np.asarray(poly) + rng.normal(0, 0.3, np.shape(poly))   # 손으로 붙인 테이프의 미세한 비뚤어짐
        v = int(rng.uniform(215, 255))
        cv2.fillPoly(floor, [to_px(p)], (v, v, v))
        cv2.fillPoly(inst, [to_px(p)], i)
    bw, bh = (600, 900) if rng.random() < 0.5 else (900, 600)       # 폼보드 이음매(그림에만, 라벨에는 없음)
    ox, oy = rng.uniform(0, bw), rng.uniform(0, bh)
    seam = int(rng.uniform(0, 60))
    for x in np.arange(ox, size[0], bw):
        cv2.line(floor, (int(x * ppm), 0), (int(x * ppm), H), (seam,) * 3, max(1, int(ppm)))
    for y in np.arange(oy, size[1], bh):
        cv2.line(floor, (0, H - int(y * ppm)), (W, H - int(y * ppm)), (seam,) * 3, max(1, int(ppm)))
    return floor, inst, to_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20), ap.add_argument("--out", default="~/pinky_sim/roads")
    ap.add_argument("--seed", type=int, default=0), ap.add_argument("--ppm", type=float, default=2.0)
    ap.add_argument("--wide", action="store_true", help="규격(차로 폭·점선·선 굵기)을 넓은 범위에서 뽑기")
    ap.add_argument("--type", choices=["any", "winding", "cross"], default="any", help="만들 도로 종류")
    a = ap.parse_args()
    out, _ = os.path.expanduser(a.out), RG.update(WIDE if a.wide else {})
    for i in range(a.count):
        rng = np.random.default_rng(a.seed * 100000 + i)
        kind = a.type if a.type != "any" else ("winding" if rng.random() < 0.5 else "cross")
        road, size, info = (winding if kind == "winding" else cross)(rng)
        floor, inst, to_px = render(road, size, a.ppm, rng)
        d = os.path.join(out, f"road_{a.seed:02d}_{i:04d}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "floor.png"), floor)
        cv2.imwrite(os.path.join(d, "inst.png"), inst)
        info.update(size_mm=list(size), ppm=a.ppm, classes=CLASSES, line_w=road.lane_t, dash=road.dash, gap=road.space,
                    instances={str(k): c for k, (c, _) in enumerate(road.marks, start=1)}, paths=road.paths)
        json.dump(info, open(os.path.join(d, "road.json"), "w"))
        pv = floor.copy()
        for k, (cls, poly) in enumerate(road.marks, start=1):
            pv[inst == k] = COLORS[cls]
        for p in road.paths:
            cv2.polylines(pv, [to_px(np.asarray(p))], False, (0, 200, 0), max(2, int(3 * a.ppm)))
        s = 800 / max(pv.shape[:2])
        cv2.imwrite(os.path.join(d, "preview.jpg"), cv2.resize(pv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        print(f"{os.path.basename(d)}: {info['type']}, 편도 {info['lanes_per_dir']}차로, 차로 폭 {info['lane_w']:.0f}mm, "
              f"크기 {size[0]:.0f}x{size[1]:.0f}mm, 선 {len(road.marks)}개, 경로 {len(road.paths)}개")


if __name__ == "__main__":
    main()
