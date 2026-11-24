"""주행 차선 그리기: 도로 그래프로 찾은 길을 따라가며, 지도와 위치로 계산한 주행 차선(왼쪽·오른쪽 선)을 카메라 화면에 투영해 그린다.

학습은 쓰지 않는다. 그리는 데 쓰는 것은 지도(차로 중심, 선까지 거리), 로봇 위치·방향, 카메라 보정값뿐이다.
  보라 = 칠해진 선이 있는 구간(나중에 2차 체크로 비전과 비교하는 부분)
  노랑 = 교차로 안(칠해진 선이 없음, 지도를 믿고 그대로 그림)
차선 변경: 길에 차선 변경이 들어 있으면 로봇은 차로 폭의 LC_LEN배 길이에 걸쳐 S자로 옆 차로로 옮겨 간다.
  주행 차선 = 프레임마다 로봇 중심이 양쪽 선 사이에 들어 있는 차로(선을 넘는 순간 바뀜, 학습 라벨 규칙과 같음). 그 차로를 앞으로 따라가며 그린다.
확인: 칠해진 구간에서 그린 선의 점마다 같은 화면 줄에서 가장 가까운 실제 선 픽셀(시뮬 정답 선 그림)을 찾아,
  그 픽셀을 바닥으로 펼친 점과 그린 점의 거리(mm)를 잰다. 점선의 빈 곳처럼 근처(SEARCH 픽셀)에 선이 없으면 세지 않는다.
  앞 거리는 2차 체크와 같은 범위(lane_check.NEAR)만 쓴다.
사용: python draw_lane.py ROAD OUT 쌍수 시드 [앞 거리 m] [lc = 차선 변경이 든 길만]
"""
import os, subprocess, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                  # 저장소 최상위
sys.path[:0] = [HERE + "/pinky/sim", HERE]
from ego_label import Routes, STEP
from route_graph import Graph, LC_LEN
from lane_check import to_cam, NEAR
from lane_geom import Camera
import sim_closed as S

AHEAD = 0.7          # 앞쪽으로 그리는 길이(m, 길 위 거리)
FPS, SPEED = 15.0, 0.2
MIN_LEN = 2.0        # 이보다 짧은 길은 고르지 않음(m)
NEAR_FWD = 0.02      # 카메라 앞 이 거리(m)보다 가까운 점은 그리지 않음(카메라 뒤쪽 점이 화면에 뒤집혀 찍히는 것을 막음)
SEARCH = 30          # 같은 줄에서 실제 선을 찾는 범위(픽셀)
MARGIN = 0.1         # 렌즈 왜곡을 입히기 전 화면 밖으로 이 비율까지만 그림(더 밖은 왜곡 식이 맞지 않아 선이 튐)


def visible(cam, c, w, h):
    """카메라 바닥 좌표 점들 중 화면 안(왜곡 전, 여유 MARGIN)에 들어오는 것."""
    u = np.c_[c, np.ones(len(c))] @ cam.Hinv.T
    ok = (c[:, 1] > NEAR_FWD) & (u[:, 2] > 1e-9)
    u = u[:, :2] / np.where(ok, u[:, 2], 1)[:, None]
    return ok & (u[:, 0] > -MARGIN * w) & (u[:, 0] < (1 + MARGIN) * w) & (u[:, 1] > -MARGIN * h) & (u[:, 1] < (1 + MARGIN) * h)
COL = {0: (255, 0, 255), 1: (0, 220, 255)}


def pieces(px, z, idx=None):
    """화면 점들을 교차로 안/밖이 바뀌는 곳(이어서 그림)과 화면 밖으로 빠진 곳(끊음)에서 나눠 (점들, 구간) 목록으로."""
    idx = np.arange(len(z)) if idx is None else idx
    out, a = [], 0
    for b in range(1, len(z) + 1):
        gap = b < len(z) and idx[b] - idx[b - 1] > 1
        if b == len(z) or z[b] != z[a] or gap:
            out.append((px[a:b if gap else min(b + 1, len(z))], int(z[a]))); a = b
    return out


def poses(g, ids):
    """길의 점들 → 로봇이 지나는 점(mm)과 길 위 거리(m), 차선 변경마다 (가운데 거리, 옆으로 옮긴 벡터, 넘는 선까지 거리)."""
    P = g.P[ids].astype(float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    jump = np.array([(int(a), int(b)) in g.lc for a, b in zip(ids[:-1], ids[1:])])
    s = np.r_[0.0, np.cumsum(np.where(jump, 0.0, seg))]
    Q, lcs = P.copy(), []
    for j in np.where(jump)[0]:
        D = P[j + 1] - P[j]; L = LC_LEN * (g.wl[ids[j]] + g.wr[ids[j]])
        w = np.clip((s - (s[j] - L / 2)) / L, 0, 1); w = w * w * (3 - 2 * w)
        Q += np.where((np.arange(len(P)) <= j)[:, None], D * w[:, None], -D * (1 - w[:, None]))
        side = np.sign(D @ [-np.sin(g.yaw[ids[j]]), np.cos(g.yaw[ids[j]])])
        lcs.append((s[j] / 1000, D, g.wl[ids[j]] if side > 0 else g.wr[ids[j]], int(ids[j])))
    keep = np.r_[True, np.diff(s) > 1e-9]
    return Q[keep], s[keep] / 1000.0, lcs


def main():
    road, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ahead = float(sys.argv[5]) if len(sys.argv) > 5 else AHEAD
    only_lc = len(sys.argv) > 6 and sys.argv[6] == "lc"
    os.makedirs(out, exist_ok=True)
    cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", out + "/calib.npz"
    subprocess.run([sys.executable, HERE + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    cam = Camera(calib); cam_h = float(np.load(calib)["cam_h"])
    sc = S.Scene(road, cam_yaml); g = Graph(Routes(sc, S.LATERAL_MM, S.YAW_DEG), STEP, road)
    rng = np.random.default_rng(seed); np.random.seed(seed)
    tint = np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)]); bg = S.background(sc.H, sc.W, rng)
    top0 = cv2.imread(road + "/floor.png"); ppm, Hpx = sc.info["ppm"], top0.shape[0]
    tpx = lambda p: np.c_[p[:, 0] * ppm, Hpx - p[:, 1] * ppm].astype(np.int32)
    outside, k, rows, tries = np.where(g.zone == 0)[0], 0, [], 0
    while k < n and tries < 5000:
        tries += 1
        ids, _ = g.route(*map(int, rng.choice(outside, 2)))
        if ids is None:
            continue
        Q, s, lcs = poses(g, ids)
        if s[-1] < MIN_LEN or (only_lc and not lcs):
            continue
        L = LC_LEN * float(np.median(g.wl[ids] + g.wr[ids])) / 1000.0               # 시험 길 고르기: 차선 변경이 출발 직후·영상 끝 뒤이거나 서로 겹치면 뺌
        if only_lc and (lcs[0][0] < L / 2 or lcs[-1][0] > s[-1] - ahead - L / 2 or any(b[0] - a[0] < L for a, b in zip(lcs, lcs[1:]))):
            continue
        tq = np.gradient(Q, axis=0); qyaw = np.arctan2(tq[:, 1], tq[:, 0])
        top = top0.copy(); cv2.polylines(top, [tpx(Q)], False, (0, 160, 0), 4)
        th = sc.H; top = cv2.resize(top, (int(top.shape[1] * th / top.shape[0]), th)); f = th / Hpx
        vw = cv2.VideoWriter(f"{out}/raw{k}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (sc.W + top.shape[1], sc.H))
        dist, lanes, cross, prefer = [], [], [], set(int(v) for v in ids)
        for fi, t in enumerate(np.arange(0.0, s[-1] - ahead, SPEED / FPS)):
            i = int(np.searchsorted(s, t)); x, y, yw = Q[i, 0] / 1000, Q[i, 1] / 1000, float(qyaw[i])
            cur = g.current(Q[i], yw, prefer)
            if cur < 0:                                                          # 교차로 안: 길 위의 가장 가까운 점
                cur = int(ids[np.argmin(np.linalg.norm(g.P[ids] - Q[i], axis=1))])
            lane_ids = g.ahead(cur, ids, ahead * 1000 + 100.0)
            lanes.append(int(g.pid[cur, 0]))
            for s0, D, wa, a in lcs:                                             # 로봇 중심이 넘는 선 너머에 있는지(차선 변경마다)
                nd = D / np.linalg.norm(D)                                       # 로봇 중심 / 그리는 차로의 점이 넘는 선 너머인지
                near_lc = abs(t - s0) <= LC_LEN * (g.wl[a] + g.wr[a]) / 1000.0               # 그 차선 변경 근처 프레임만
                cross.append((fi, s0, near_lc and bool((Q[i] - g.P[a]) @ nd > wa), near_lc and bool((g.P[cur] - g.P[a]) @ nd > wa)))
            img, inst = sc.render(x, y, yw); img = S.look(img, tint, 3.0, bg)
            _, _, Lf, Rt, z = g.lane(lane_ids)
            vis = img.copy()
            for Q2 in (Lf, Rt):
                c = to_cam(Q2 / 1000, (x, y, yw), S.CAM_FWD); ok = visible(cam, c, sc.W, sc.H)
                if ok.sum() < 2:
                    continue
                px = cam.to_image(c[ok]); zz = z[ok]
                cc = c[ok]; near = (cc[:, 1] > NEAR[0] * cam_h) & (cc[:, 1] < NEAR[1] * cam_h)
                for (u, v), g0 in zip(px[(zz == 0) & near].astype(int), cc[(zz == 0) & near]):
                    if not (0 <= v < sc.H and 0 <= u < sc.W):
                        continue
                    cols = np.where(inst[v, max(u - SEARCH, 0):u + SEARCH + 1] > 0)[0] + max(u - SEARCH, 0)
                    if len(cols):
                        gl, _ = cam.to_ground([[cols[np.argmin(np.abs(cols - u))], v]]); dist.append(np.linalg.norm(gl[0] - g0) * 1000)
                for q, zc in pieces(px.astype(np.int32), zz, np.where(ok)[0]):
                    cv2.polylines(vis, [q], False, COL[zc], 2, cv2.LINE_AA)
            cv2.putText(vis, sc.info["routes"][lanes[-1]]["name"], (5, 20), 0, 0.5, (255, 255, 255), 1)
            tp = top.copy(); cv2.circle(tp, tuple((tpx(Q[[i]]) * f).astype(int)[0]), 8, (0, 255, 0), -1)
            vw.write(np.hstack([vis, tp]))
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/raw{k}.mp4", "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", f"{out}/lane{k}.mp4"], check=True); os.remove(f"{out}/raw{k}.mp4")
        d = np.array(dist)
        names = [sc.info["routes"][j]["name"] for j in g.pid[ids, 0]]
        seq = [names[0]] + [b for a, b in zip(names, names[1:]) if a != b]
        msg = ""
        for s0, D, wa, a in lcs:                                                 # 차선 변경: 선을 넘은 프레임과 주행 차선이 바뀐 프레임
            fc = next((fi for fi, ss, v, _ in cross if ss == s0 and v), None)
            sw = next((fi for fi, ss, _, v in cross if ss == s0 and v), None)
            msg += f" | 차선 변경({sc.info['routes'][int(g.pid[a, 0])]['name']}에서): 로봇 중심이 선 넘은 프레임 {fc}, 주행 차선 바뀐 프레임 {sw}"
        rows.append(f"쌍 {k}: 길이 {s[-1]:.2f} m, 경로 {seq}, 잰 점 {len(d)}개, 실제 선까지 거리 중앙값 {np.median(d):.1f} mm, "
                    f"95% {np.percentile(d, 95):.1f} mm, 최대 {d.max():.1f} mm" + msg)
        print(rows[-1], flush=True); k += 1
    sc.close()
    open(f"{out}/result.txt", "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
