"""AI 허브 전방 카메라 보정값으로 바닥 평면(BEV) 변환 행렬을 만든다.
meta_data.json의 내부 보정값(초점거리, 주점, 왜곡)과 카메라 높이를 쓰고,
카메라가 숙여진 각도와 좌우로 틀어진 각도는 차선 라벨의 소실점에서 구한다.
(외부 보정값 Rx·Ry·Rz는 기준 좌표계와 부호 규칙을 알 수 없어 비교용으로만 출력)
사용법:
    python aihub_calib.py ~/lane/aihub_val ~/lane/aihub_calib
"""
import argparse, glob, json, os
from collections import defaultdict

import cv2, numpy as np

MIN_LEN = 80      # 직선으로 쓸 조각의 최소 길이(픽셀)
MIN_RATIO = 5     # 길이/폭이 이 값 이상인 길쭉한 조각만 사용
VP_TOL = 8        # 소실점과 직선 사이 허용 거리(픽셀)
COLORS = {"whiteLane": (255, 255, 255), "stopLane": (0, 0, 255),
          "crossWalk": (0, 255, 255), "roadMark": (255, 255, 0)}

def load_calib(meta_path):
    """메타 파일에서 전방 카메라의 K, 왜곡 계수, 외부 보정값을 읽는다."""
    with open(meta_path, encoding="utf-8") as f:
        front = json.load(f)["calibration"]["camera"]["front"]
    i, e = front["Intrinsic"], front["Extrinsic"]
    K = np.array([[i["Fx"], 0, i["Cx"]], [0, i["Fy"], i["Cy"]], [0, 0, 1]], float)
    # Param0~4를 OpenCV 순서(k1, k2, p1, p2, k3)로 가정
    dist = np.array([i["Distortion"][f"Param{k}"] for k in range(5)], float)
    ext = tuple(round(e[k], 3) for k in ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))
    return K, dist, ext

def undistort(pts, K, dist):
    """왜곡을 편 픽셀 좌표로 바꾼다. dist가 None이면 그대로 돌려준다."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if dist is None:
        return pts
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

def read_polys(label_path):
    """라벨 JSON에서 (클래스, 꼭짓점 배열) 목록을 읽는다."""
    with open(label_path, encoding="utf-8") as f:
        anns = json.load(f).get("annotations", [])
    out = [(a.get("class", ""), np.array(a.get("polygon", []), np.float64).reshape(-1, 2))
           for a in anns]
    return [(c, p) for c, p in out if len(p) >= 3]

def fit_line(pts):
    """길쭉한 조각이면 직선 (a, b, c) (a²+b²=1)을, 아니면 None을 돌려준다."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    v, n = vt[0], vt[1]
    length, width = np.ptp((pts - c) @ v), np.ptp((pts - c) @ n)
    # 너무 짧거나 뭉툭한 조각, 가로로 누운 조각(소실점으로 향하지 않음)은 제외
    if length < MIN_LEN or length < MIN_RATIO * max(width, 1) or abs(v[1]) < 0.2:
        return None
    # 조각을 앞뒤 절반으로 나눠 각 절반의 중심을 잇는다 (원근 때문에 생기는 방향 치우침 방지)
    t = (pts - c) @ v
    a, b = pts[t < 0].mean(axis=0), pts[t >= 0].mean(axis=0)
    n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    return np.array([n[0], n[1], -n @ a])

def vanishing_point(lines):
    """직선들이 가장 많이 모이는 점을 찾는다. (점, 모인 직선 수, 평균 오차)"""
    L = np.array(lines)
    best = None
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            p = np.cross(L[i], L[j])
            # 거의 평행한 두 직선은 교점이 불안정하므로 건너뜀
            if abs(L[i, :2] @ L[j, :2]) > 0.985 or abs(p[2]) < 1e-9:
                continue
            d = np.abs(L @ (p / p[2]))
            key = (int((d < VP_TOL).sum()), -d[d < VP_TOL].sum())
            if best is None or key > best[0]:
                best = (key, d < VP_TOL)
    if best is None:
        return None
    A = L[best[1]]
    vp = np.linalg.lstsq(A[:, :2], -A[:, 2], rcond=None)[0]   # 모인 직선들로 다시 계산
    return vp, len(A), float(np.sqrt(np.mean((A[:, :2] @ vp + A[:, 2]) ** 2)))

def make_h(K, vp, cam_h):
    """소실점과 카메라 높이로 '왜곡 편 픽셀 → 바닥 (x 오른쪽, z 앞, m)' 행렬을 만든다."""
    f = np.linalg.inv(K) @ np.array([vp[0], vp[1], 1.0])
    f /= np.linalg.norm(f)                          # 도로 진행 방향
    r = np.cross([0, 1, 0], f) / np.linalg.norm(np.cross([0, 1, 0], f))   # 오른쪽 방향 (좌우 기울기 0 가정)
    down = np.cross(f, r)                           # 바닥을 향하는 방향
    H = np.vstack([r, f, down / cam_h]) @ np.linalg.inv(K)
    pitch = np.degrees(np.arcsin(-f[1]))            # +면 카메라가 아래로 숙여짐
    yaw = np.degrees(np.arctan2(f[0], f[2]))        # +면 도로가 화면 오른쪽으로 치우침
    return H, pitch, yaw

def to_ground(pts, K, dist, H):
    """픽셀 좌표를 바닥 좌표로 바꾼다. (좌표 배열, 지평선 아래 여부)"""
    u = undistort(pts, K, dist)
    g = np.c_[u, np.ones(len(u))] @ H.T
    ok = g[:, 2] > 1e-6
    return g[:, :2] / np.where(ok, g[:, 2], 1)[:, None], ok

def lane_widths(polys, K, dist, H):
    """한 프레임에서 이웃한 차선 사이 간격(m)들을 구한다."""
    xs = []
    for cls, pts in polys:
        if cls == "whiteLane" and fit_line(undistort(pts, K, dist)) is not None:
            g, ok = to_ground(pts.mean(axis=0), K, dist, H)
            if ok[0] and 6 < g[0, 1] < 30:
                xs.append(g[0, 0])
    groups = []                                     # 옆 위치가 1m 이내면 같은 선
    for x in sorted(xs):
        groups[-1].append(x) if groups and x - groups[-1][-1] < 1.0 else groups.append([x])
    c = [np.mean(g) for g in groups]
    return [b - a for a, b in zip(c, c[1:]) if 2.3 < b - a < 4.7]

def preview(img_path, polys, K, dist, H, out_path):
    """원본(라벨 윤곽) 옆에 바닥으로 펼친 그림을 붙여 저장한다. 1m = 20픽셀."""
    img = cv2.imread(img_path)
    bev = np.zeros((1000, 600, 3), np.uint8)
    for z in range(10, 50, 10):                     # 10m 간격 눈금
        cv2.line(bev, (0, 1000 - z * 20), (600, 1000 - z * 20), (60, 60, 60), 1)
        cv2.putText(bev, f"{z}m", (5, 995 - z * 20), 0, 0.5, (120, 120, 120), 1)
    for cls, pts in polys:
        if cls not in COLORS:
            continue
        cv2.polylines(img, [pts.astype(np.int32)], True, COLORS[cls], 2)
        g, ok = to_ground(pts, K, dist, H)
        if ok.all() and g[:, 1].max() < 60:
            px = np.c_[300 + g[:, 0] * 20, 1000 - g[:, 1] * 20].astype(np.int32)
            cv2.fillPoly(bev, [px], COLORS[cls])
    img = cv2.resize(img, (int(img.shape[1] * 1000 / img.shape[0]), 1000))
    cv2.imwrite(out_path, np.hstack([img, bev]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="추출한 AI 허브 폴더 (images, labels가 들어 있는 곳)")
    ap.add_argument("out_dir", help="보정 결과를 저장할 폴더")
    ap.add_argument("--step", type=int, default=4, help="주행마다 몇 장에 한 장씩 볼지")
    ap.add_argument("--preview", default="/mnt/c/Users/Snake/Downloads/aihub_bev_check.jpg")
    ap.add_argument("--dist", choices=["on", "off", "auto"], default="on",
                    help="왜곡 보정: on=항상 함, off=안 함, auto=직선이 더 잘 모이는 쪽")
    args = ap.parse_args()
    root, out_dir = os.path.expanduser(args.root), os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 보정값이 같은 주행끼리 묶음
    groups, calibs = defaultdict(list), {}
    for mp in sorted(glob.glob(os.path.join(root, "images", "*", "*_meta_data.json"))):
        K, dist, ext = load_calib(mp)
        key = (tuple(K.ravel().round(3)), tuple(dist.round(6)), ext)
        groups[key].append(os.path.basename(os.path.dirname(mp)))
        calibs[key] = (K, dist, ext)
    print(f"주행 {sum(len(v) for v in groups.values())}건, 보정값 종류 {len(groups)}개")

    scene_map = {}
    for gi, (key, scenes) in enumerate(groups.items()):
        K, dist, ext = calibs[key]
        print(f"\n[보정 {gi}] 주행 {len(scenes)}건, 외부 보정값(Tx Ty Tz Rx Ry Rz) = {ext}")
        frames = []
        for s in scenes:
            pat = os.path.join(root, "labels", "2D", s, "sensor_raw_data", "camera", "*.json")
            frames += [(p, read_polys(p)) for p in sorted(glob.glob(pat))[::args.step]]
        # 왜곡을 편 경우와 안 편 경우 각각 소실점을 구해, 직선이 더 잘 모이는 쪽을 고름
        res = {}
        for name, d in (("왜곡 보정 안 함", None), ("왜곡 보정 함", dist)):
            vps, errs, best = [], [], (0, None)
            for fi, (_, polys) in enumerate(frames):
                lines = [fit_line(undistort(p, K, d)) for c, p in polys if c == "whiteLane"]
                lines = [ln for ln in lines if ln is not None]
                out = vanishing_point(lines) if len(lines) >= 2 else None
                if out is None:
                    continue
                vps.append(out[0])
                if out[1] >= 3:
                    errs.append(out[2])
                    best = max(best, (out[1], fi))
            vp = np.median(vps or [[np.nan, np.nan]], axis=0)   # 소실점을 못 구하면 nan
            err = float(np.median(errs)) if errs else float("nan")
            res[name] = (d, vp, err, best[1])
            print(f"  {name}: 소실점 ({vp[0]:.0f}, {vp[1]:.0f}), 프레임 {len(vps)}개, 직선 모임 오차 {err:.2f}픽셀")
        name = {"on": "왜곡 보정 함", "off": "왜곡 보정 안 함"}.get(args.dist) or \
            min(res, key=lambda k: res[k][2] if res[k][2] == res[k][2] else 1e9)
        d, vp, _, show = res[name]
        H, pitch, yaw = make_h(K, vp, ext[2])
        w = [x for _, polys in frames for x in lane_widths(polys, K, d, H)]
        print(f"  선택: {name} / 숙임 {pitch:.2f}도, 좌우 틀어짐 {yaw:.2f}도, 높이 {ext[2]}m")
        q = np.percentile(w or [np.nan], [25, 50, 75])
        print(f"  차선 폭 확인: 중앙값 {q[1]:.2f}m (표본 {len(w)}개, 25~75% 범위 {q[0]:.2f}~{q[2]:.2f}m)")
        np.savez(os.path.join(out_dir, f"calib{gi}.npz"), K=K, dist=d if d is not None else np.zeros(5), H=H, cam_h=ext[2], vp=vp)
        scene_map.update({s: f"calib{gi}.npz" for s in scenes})
        if gi == 0 and show is not None:
            lp, polys = frames[show]
            ip = lp.replace(os.path.join("labels", "2D"), "images").replace(".json", ".jpg")
            preview(ip, polys, K, d, H, args.preview)
            print(f"  미리보기: {args.preview}  ({os.path.basename(ip)})")
    with open(os.path.join(out_dir, "scene_map.json"), "w") as f:
        json.dump(scene_map, f, indent=1)
    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()
