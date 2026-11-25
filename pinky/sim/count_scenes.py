"""학습 데이터에 어떤 장면이 얼마나 들어가는지 센다(사진은 만들지 않음): 화살표가 카메라 앞에 보이는 자세, 차로에서 벗어난 자세, 교차로 앞 자세.

데이터셋에는 사진을 찍은 자세가 남아 있지 않으므로, gen_dataset.py와 같은 규칙(pick_pose)으로 같은 수만큼 자세를 다시 뽑아서 센다.
  → 실제 v13 사진 그대로의 수가 아니라 같은 생성 규칙에서 나오는 비율이다(난수 순서가 달라 장마다 같지는 않다).
화살표: 바닥 그림에서 흰색인데 번호(inst)가 없는 칠. 트랙 도로(gen_track)에만 있다.
카메라 앞: 비전 확인이 쓰는 범위(카메라 높이의 lane_check.NEAR배 앞, 좌우로 차로 폭까지)에 화살표 칠이 하나라도 있으면 '보임'.
사용: MUJOCO_GL=egl python count_scenes.py ROADS_DIR CAMERA_YAML [--per-road 15] [--per-cross 75] [--per-cross2 150] [--per-track 80] [--seed 0]
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import lane_check as C
from ego_label import Routes
from gen_dataset import LATERAL_MM, YAW_DEG, pick_pose
from scene import Scene

OFF = 0.15           # 차로 가운데에서 차로 폭의 이 배수보다 벗어나면 '벗어난 자세'
AHEAD = 12.0         # 교차로 앞 = 교차로 입구까지 카메라 높이의 이 배수 안(lane_where.AHEAD와 같음)


def arrow_mask(road):
    """바닥 그림에서 화살표 칠(흰색인데 번호가 없는 곳)."""
    floor = cv2.imread(road + "/floor.png").astype(int); inst = cv2.imread(road + "/inst.png", cv2.IMREAD_UNCHANGED)
    white = (floor.min(axis=2) >= 200) & (floor.max(axis=2) - floor.min(axis=2) <= 25)
    return cv2.morphologyEx((white & (inst == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def sees(mask, ppm, x, y, yaw, cam_h, cam_fwd, lw):
    """카메라 앞 범위에 mask의 칠이 있는지. x, y [m], lw [m]."""
    hd = np.array([np.cos(yaw), np.sin(yaw)]); o = np.array([x, y]) + cam_fwd * hd
    f, l = np.meshgrid(np.linspace(C.NEAR[0] * cam_h, C.NEAR[1] * cam_h, 24), np.linspace(-lw, lw, 16))
    P = (o + f[..., None] * hd + l[..., None] * np.array([-hd[1], hd[0]])) * 1000.0
    u, v = (P[..., 0] * ppm).astype(int), (mask.shape[0] - P[..., 1] * ppm).astype(int)
    ok = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    return bool(mask[v[ok], u[ok]].any())


def where(rt, x, y, yaw):
    """자세가 올라탄 차로: (벗어난 거리/차로 폭, 구간('앞'·'안'·'뒤'), 교차로 입구까지 mm, 외곽 도로인지). 올라탄 차로가 없으면 None."""
    best = None
    for r, info in zip(rt.r, rt.info["routes"] if "routes" in rt.info else [{}] * len(rt.r)):
        d = np.hypot(r["p"][:, 0] - x * 1000, r["p"][:, 1] - y * 1000); i = int(np.argmin(d))
        if abs((r["ang"][i] - yaw + np.pi) % (2 * np.pi) - np.pi) > np.radians(45):
            continue
        e = np.array([x * 1000, y * 1000]) - r["p"][i]
        off = float(np.cos(r["ang"][i]) * e[1] - np.sin(r["ang"][i]) * e[0]); lw = float(r["wl"][i] + r["wr"][i])
        if abs(off) > lw or (best is not None and abs(off) >= abs(best[4])):
            continue
        zone = "앞" if i < r["en"] else ("안" if i <= r["ex"] else "뒤")
        to_en = float(np.linalg.norm(np.diff(r["p"][i:r["en"] + 1], axis=0), axis=1).sum()) if i < r["en"] else np.inf
        ring = str(info.get("name", "")).split(">")[0] == "R" if zone == "앞" else str(info.get("name", "")).split(">")[-1].startswith("R")
        best = (abs(off) / lw, zone, to_en, ring, off)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roads"), ap.add_argument("cam")
    ap.add_argument("--per-road", type=int, default=15), ap.add_argument("--per-cross", type=int, default=75)
    ap.add_argument("--per-cross2", type=int, default=150), ap.add_argument("--per-track", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    e = yaml.safe_load(open(a.cam))["extrinsic"]; cam_h, cam_fwd = e["height_m"], e["forward_m"]
    roads = sorted(glob.glob(os.path.join(os.path.expanduser(a.roads), "road_*")))
    rng = np.random.default_rng(a.seed); rows = []
    for ri, road in enumerate(roads):
        sc = Scene(road, a.cam); rt = Routes(sc, LATERAL_MM, YAW_DEG); info = sc.info; sc.close()
        kind = "트랙" if info["type"] == "track" else ("교차로2" if info["type"] == "cross" and info["lanes_per_dir"] == 2 else ("교차로1" if info["type"] == "cross" else "굽은"))
        want = {"트랙": a.per_track, "교차로2": a.per_cross2, "교차로1": a.per_cross, "굽은": a.per_road}[kind]
        mask = arrow_mask(road) if info["type"] == "track" else None
        has = bool(mask is not None and mask.any())
        for _ in range(want):
            x, y, yaw = pick_pose(rt, rng); w = where(rt, x, y, yaw); lw = info["lane_w"] / 1000.0
            arrow = mask is not None and sees(mask, info["ppm"], x, y, yaw, cam_h, cam_fwd, lw)
            rows.append((kind, has, arrow, w is not None, w[0] > OFF if w else False, w[1] if w else "-",
                         bool(w and w[2] < AHEAD * cam_h * 1000), bool(w and w[3])))
        if (ri + 1) % 50 == 0:
            print(f"[{ri + 1}/{len(roads)}]", flush=True)
    n = len(rows); pc = lambda k: f"{k:6d} ({k / max(n, 1):5.1%})"
    print(f"\n자세 {n}개 (도로 {len(roads)}개). 비율은 전체 자세에 대한 것")
    for kind in ("굽은", "교차로1", "교차로2", "트랙"):
        R = [r for r in rows if r[0] == kind]
        print(f"  {kind}: 자세 {len(R)}, 화살표가 그려진 도로의 자세 {sum(r[1] for r in R)}, 화살표가 앞에 보임 {sum(r[2] for r in R)}")
    A = [r for r in rows if r[2]]
    print(f"화살표가 앞에 보이는 자세        {pc(len(A))}")
    print(f"  그중 차로에서 벗어난 자세      {pc(sum(r[4] for r in A))}")
    print(f"  그중 차로 가운데 근처          {pc(sum(r[3] and not r[4] for r in A))}")
    print(f"  그중 교차로 앞(입구까지 {AHEAD * cam_h:.2f} m 안) {pc(sum(r[6] for r in A))}")
    print(f"  그중 교차로 안                 {pc(sum(r[5] == '안' for r in A))}")
    print(f"  그중 외곽 도로 위              {pc(sum(r[7] for r in A))}")
    print(f"차로에서 벗어난 자세(화살표 무관) {pc(sum(r[4] for r in rows))}")
    print(f"교차로 앞 자세(화살표 무관)       {pc(sum(r[6] for r in rows))}")
    print(f"  그중 차로에서 벗어난 자세      {pc(sum(r[6] and r[4] for r in rows))}")
    print(f"교차로 안 자세                    {pc(sum(r[5] == '안' for r in rows))}")
    print(f"외곽 도로 위 자세                 {pc(sum(r[7] for r in rows))}")
    print(f"  그중 화살표가 앞에 보임        {pc(sum(r[7] and r[2] for r in rows))}")


if __name__ == "__main__":
    main()
