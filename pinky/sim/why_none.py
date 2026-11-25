"""비전 확인이 '확인 못 함'으로 끝난 프레임을 이유별로 센다(lane_check.py는 고치지 않고, 같은 계산을 따라가며 버려진 점만 기록).

프레임의 이유(차로 안 프레임만):
  ok      = 확인함
  nosee   = 비전이 내 차로 선을 하나도 못 냄
  nomap   = 지도에 비교할 선이 없음(앞이 교차로·길 끝)
  few     = 체크 범위(카메라 높이의 NEAR배) 안의 비전 점이 MIN_N개 미만
  dropped = 범위 안 점은 MIN_N개 이상인데 맞추는 중에 버려져 MIN_N개 아래로 떨어짐  ← 원인을 찾는 대상
dropped 프레임에서는 실패한 바로 그 반복에서 점마다 버려진 이유를 센다:
  end_far / end_near = 지도 선의 먼 쪽 / 가까운 쪽 끝점과 짝이 됨(비전 점이 지도 선 범위 밖)
  far  = 지도 선에서 차로 폭의 0.5배(첫 회)·0.25배(이후)보다 떨어짐
  trim = 다 맞춘 뒤 TRIM으로 버려짐
  nopair = 그 쪽(왼쪽/오른쪽) 선이 지도나 비전 한쪽에만 있어 비교에 못 들어감
사용: python why_none.py ROAD 주행기록폴더 감시기록폴더 OUT.pkl   환경 변수 EVERY(기본 2)
"""
import os, pickle, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import lane_check as C
from ego_label import Routes, STEP
from route_graph import Graph
from lane_where import Where
import sim_closed as S


def in_range(q, cam_h):
    """체크 범위 안의 점만(lane_check.check와 같은 조건)."""
    r = np.linalg.norm(q, axis=1)
    return q[(q[:, 1] >= C.NEAR[0] * cam_h) & (r <= C.NEAR[1] * cam_h)]


def trace(seen, expect, cam_h, cam_fwd, lane_w, iters=4):
    """lane_check.check와 같은 계산. 돌려주는 값: (결과가 None인지, 실패한 반복 번호, 점마다 이유 dict, 쪽마다 자세한 값)."""
    pairs, nopair, sides = [], 0, []
    for side in ("left", "right"):
        q, m = seen.get(side), expect.get(side)
        if q is None or m is None or len(q) < 2 or len(m) < 2:
            nopair += 0 if q is None or len(q) < 2 else len(in_range(q, cam_h))
            continue
        pairs.append((in_range(q, cam_h), m)); sides.append(side)
    total = sum(len(q) for q, _ in pairs)
    if total < C.MIN_N:
        return True, -1, dict(nopair=nopair, total=total), None
    both = sum(len(q) >= C.MIN_N // 2 for q, _ in pairs) == 2
    th, t = 0.0, np.zeros(2)
    dead = [np.zeros(len(q), bool) for q, _ in pairs]
    prev = None
    for it in range(2 * iters + total):
        A, y, KEEP, worst, why, det = [], [], [], (0.0, None), dict(end_far=0, end_near=0, far=0, trim=0), []
        for pi, (q, m) in enumerate(pairs):
            c, s_ = np.cos(th), np.sin(th); piv = np.array([0.0, -cam_fwd])
            mm = (m - piv) @ np.array([[c, s_], [-s_, c]]) + piv + t
            k = np.linalg.norm(q[:, None] - mm[None], axis=2).argmin(axis=1)
            tg = np.gradient(mm, axis=0)[k]; tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            n = np.c_[tg[:, 1], -tg[:, 0]]
            res = np.sum(n * (q - mm[k]), axis=1)
            far = np.abs(res) > (0.5 if it == 0 else 0.25) * lane_w
            e0, e1 = k == 0, k == len(mm) - 1
            keep = ~far & ~e0 & ~e1 & ~dead[pi]
            # 지도 선의 양 끝 중 카메라에서 먼 쪽이 어느 끝인지(끝점 짝의 방향을 가르기 위해)
            far_end_is_last = np.linalg.norm(mm[-1]) >= np.linalg.norm(mm[0])
            ef = e1 if far_end_is_last else e0
            why["trim"] += int(dead[pi].sum())
            why["end_far"] += int((ef & ~dead[pi]).sum()); why["end_near"] += int(((e0 | e1) & ~ef & ~dead[pi]).sum())
            why["far"] += int((far & ~e0 & ~e1 & ~dead[pi]).sum())
            det.append(dict(side=sides[pi], q=q, mm=mm, keep=keep.copy(), far=far, end=e0 | e1, dead=dead[pi].copy(), res=res))
            perp = np.c_[-(mm[k, 1] - piv[1]), mm[k, 0] - piv[0]]
            if it >= iters and keep.any() and np.abs(res[keep]).max() > worst[0]:
                worst = (float(np.abs(res[keep]).max()), (pi, int(np.where(keep)[0][np.abs(res[keep]).argmax()])))
            A.append(np.c_[np.sum(n * perp, axis=1), n][keep]); y.append(res[keep]); KEEP.append(keep)
        if it >= iters:
            sig = b"".join(k_.tobytes() for k_ in KEEP)
            if worst[0] <= C.TRIM * lane_w and sig == prev:
                break
            prev = sig
            if worst[0] > C.TRIM * lane_w:
                dead[worst[1][0]][worst[1][1]] = True
                cut = lambda L: [v[np.where(k_)[0] != worst[1][1]] if pi == worst[1][0] else v for pi, (v, k_) in enumerate(zip(L, KEEP))]
                A, y = cut(A), cut(y)
        A, y = np.vstack(A), np.concatenate(y)
        if len(y) < C.MIN_N:
            why.update(nopair=nopair, total=total, left_n=len(y))
            return True, it, why, det
        reg = np.sqrt(len(y)) * np.array([[0, 0, 1.0]]) * 0.3
        M = np.r_[A, reg] if both else np.r_[A, reg][:, 1:]
        d = np.linalg.lstsq(M, np.r_[y, 0.0], rcond=None)[0]
        d = d if both else np.r_[0.0, d]
        th, t = th + d[0], t + d[1:]
    return False, it, dict(nopair=nopair, total=total), None


def main():
    road, runs, watch, out = sys.argv[1:5]; every = int(os.environ.get("EVERY", 2))
    cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", "/tmp/why_calib.npz"
    subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    sc = S.Scene(road, cam_yaml); rt = Routes(sc, S.LATERAL_MM, S.YAW_DEG); names = [r["name"] for r in sc.info["routes"]]; sc.close()
    cam_h = float(np.load(calib)["cam_h"]); g = Graph(rt, STEP, road); w = Where(g, cam_h, S.CAM_FWD)
    rec, differ = [], 0
    for j in range(len(rt.r)):
        if not os.path.exists(f"{watch}/{j:02d}_seen.pkl"):
            continue
        R = np.load(f"{runs}/{j:02d}.npy")[::every]; seen = pickle.load(open(f"{watch}/{j:02d}_seen.pkl", "rb"))
        w.set_route(np.where(g.pid[:, 0] == j)[0])
        for i, (row, s) in enumerate(zip(R, seen)):
            pose = tuple(row[4:7]); wh = w.locate(pose)
            if wh["state"] != "lane":
                continue
            base = dict(route=j, name=names[j], i=i, t=float(row[0]), x=float(row[1]), y=float(row[2]), yaw=float(row[3]), pose=pose)
            sv = dict(left=s["left"], right=s["right"])
            if s["left"] is None and s["right"] is None:
                rec.append(dict(base, why="nosee")); continue
            exp = w.map_lines(wh["k"], pose)
            if exp is None:
                rec.append(dict(base, why="nomap", seen=sv)); continue
            none, it, why, det = trace(sv, exp, cam_h, S.CAM_FWD, wh["lane_w"])
            differ += none != (C.check(sv, exp, cam_h, S.CAM_FWD, wh["lane_w"]) is None)   # 따라간 계산이 원래 것과 같은지
            kind = "ok" if not none else ("few" if it < 0 else "dropped")
            rec.append(dict(base, why=kind, it=it, cnt=why, det=det, seen=sv, exp=exp, lane_w=wh["lane_w"]))
    pickle.dump(rec, open(out, "wb"))
    n = len(rec); print(f"차로 안 {n}프레임, 원래 계산과 결과가 다른 프레임 {differ}")
    for kind in ("ok", "dropped", "nomap", "few", "nosee"):
        m = sum(r["why"] == kind for r in rec); print(f"  {kind:8s} {m:5d} ({m / n:.0%})")
    D = [r for r in rec if r["why"] == "dropped"]
    tot = {k: sum(r["cnt"][k] for r in D) for k in ("end_far", "end_near", "far", "trim", "nopair", "left_n", "total")}
    print(f"dropped {len(D)}프레임의 범위 안 점 {tot['total']}개(비교에 못 들어간 점 {tot['nopair']}개는 따로):")
    for k in ("end_far", "end_near", "far", "trim", "left_n"):
        print(f"  {k:9s} {tot[k]:6d} ({tot[k] / max(tot['total'], 1):.0%})")
    first = sum(r["it"] == 0 for r in D); print(f"  첫 반복에서 바로 실패 {first}프레임, 그 뒤 {len(D) - first}프레임")
    # 프레임마다 가장 많이 버린 이유
    top = [max(("end_far", "end_near", "far", "trim"), key=lambda k: r["cnt"][k]) for r in D]
    print("  프레임별 가장 큰 이유: " + ", ".join(f"{k} {top.count(k)}" for k in ("end_far", "end_near", "far", "trim")))


if __name__ == "__main__":
    main()
