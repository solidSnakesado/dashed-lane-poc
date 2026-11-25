"""차로 안 위치(lane_where.py) 시험: 물리 주행 기록을 따라가며 지도 + 추정 위치로 낸 값이 실제와 맞는지, 비전 확인이 틀린 위치를 잡는지 본다.

입력: phys_drive.py의 주행 기록(NN.npy: 실제 위치, 추정 위치)과 vision_watch.py의 기록(NN_seen.pkl: 프레임마다 비전이 본 내 차로 선).
넣을 옆 오차(mm)마다 추정 위치를 그만큼 왼쪽으로 옮겨 넣는다(0이면 주행 기록의 추정 위치 그대로).
  1) 위치: 모듈이 낸 상태(차로 안/밖/교차로)와 벗어난 거리를 '추정 위치 기준 정답'과 비교한다. 정답 = 주행한 경로의 가운데·선까지 거리로 직접 계산.
     넣은 오차가 0일 때는 실제 위치 기준 정답과도 비교한다(실제로 차로를 벗어난 프레임을 '밖'으로 내는지).
  2) 비전 확인: 확인됨 / 불일치 / 확인 못 함의 수와, 비전이 낸 옆 오차가 넣은 오차와 맞는지.
     길 끝 구간(앞쪽 지도 선이 교차로가 아니라 시험하는 길이 끝나서 짧아지는 프레임)은 따로 센다. 길 하나만 넘기는 이 시험에서만 생기는 구간이다.
사용: python where_test.py ROAD 주행기록폴더 감시기록폴더 [넣을 옆 오차 mm ...]   환경 변수 EVERY(vision_watch와 같은 값, 기본 2), YAW(넣을 방향 오차 도, 반시계 +)
"""
import os, pickle, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
from ego_label import Routes, STEP
from route_graph import Graph
from lane_where import Where, AHEAD
import sim_closed as S


def truth(r, xy_mm):
    """경로 r 기준 정답: (상태, 벗어난 거리 mm). 경로의 가장 가까운 점에서 직접 계산."""
    k = int(np.argmin(np.linalg.norm(r["p"] - xy_mm, axis=1))); d = xy_mm - r["p"][k]
    off = float(np.cos(r["ang"][k]) * d[1] - np.sin(r["ang"][k]) * d[0])
    if r["en"] <= k <= r["ex"]:
        return "cross", off
    return ("lane" if -r["wr"][k] <= off < r["wl"][k] else "out"), off


def main():
    road, runs, watch = sys.argv[1:4]; errs = [float(v) for v in sys.argv[4:]] or [0.0]
    every, yaw_e = int(os.environ.get("EVERY", 2)), np.radians(float(os.environ.get("YAW", 0)))
    cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", "/tmp/where_calib.npz"
    subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
    sc = S.Scene(road, cam_yaml); rt = Routes(sc, S.LATERAL_MM, S.YAW_DEG); sc.close()
    g = Graph(rt, STEP, road); w = Where(g, float(np.load(calib)["cam_h"]), S.CAM_FWD)
    names = ("lane", "out", "cross", "other")
    for e in errs:
        conf = np.zeros((4, 4), int); conf_true = np.zeros((4, 4), int); doff, stat, lat, missed, yaws = [], dict(ok=0, mismatch=0, none=0), [], [], []
        zone = dict(ok=0, mismatch=0, none=0)                                            # 교차로 구간(칠해진 선이 있는 곳만 확인됨)
        tail = dict(ok=0, mismatch=0, none=0); need = AHEAD * w.cam_h * 1000.0          # 길 끝 구간의 수, 지도 선이 다 나올 때의 길이(mm)
        for j, r in enumerate(rt.r):
            if not os.path.exists(f"{watch}/{j:02d}_seen.pkl"):                  # 기록이 있는 경로만
                continue
            R = np.load(f"{runs}/{j:02d}.npy")[::every]; seen = pickle.load(open(f"{watch}/{j:02d}_seen.pkl", "rb"))
            w.set_route(np.where(g.pid[:, 0] == j)[0])
            for row, s in zip(R, seen):
                x, y, yaw = row[4:7]; hl = np.array([-np.sin(yaw), np.cos(yaw)])
                pose = (x + hl[0] * e / 1000, y + hl[1] * e / 1000, yaw + yaw_e)   # 모듈에 넣는 추정 위치
                out = w.step(pose, None if s["left"] is None and s["right"] is None else dict(left=s["left"], right=s["right"]))
                st, off = truth(r, np.array(pose[:2]) * 1000)
                conf[names.index(st), names.index(out["state"])] += 1
                if st == out["state"] != "cross":
                    doff.append(out["off"] * 1000 - off)
                if e == 0:                                                       # 실제 위치 기준: 실제로 차로를 벗어난 프레임
                    st_t, off_t = truth(r, row[1:3] * 1000)
                    conf_true[names.index(st_t), names.index(out["state"])] += 1
                    if st_t == "out" and out["state"] not in ("out", "other"):
                        missed.append((j, row[0], off_t, out["off"] * 1000))
                if st == "cross":
                    zone[out["status"]] += 1
                if st == "lane":
                    ids = g.ahead(out["k"], w.route, need)                        # 앞쪽 지도 선이 길 끝 때문에 짧은지
                    short = not g.zone[ids].any() and np.linalg.norm(np.diff(g.P[ids], axis=0), axis=1).sum() < need
                    (tail if short else stat)[out["status"]] += 1
                    if out["lat"] is not None and not short:
                        lat.append(out["lat"] * 1000)
                    if out["yaw"] is not None and not short:
                        yaws.append(np.degrees(out["yaw"]))
        d, lat = np.abs(doff), np.array(lat); n = sum(stat.values()); nt = sum(tail.values())
        print(f"넣은 옆 오차 {e:+.0f} mm" + (f", 방향 오차 {np.degrees(yaw_e):+.1f}도 → 비전이 낸 방향 오차 중앙 {np.median(yaws):+.2f}도 (방향을 낸 프레임 {len(yaws)})" if yaw_e and yaws else ""))
        print(f"  위치(추정 위치 기준 정답 → 모듈): " + ", ".join(f"{a}→{b} {conf[i, k]}" for i, a in enumerate(names) for k, b in enumerate(names) if conf[i, k]))
        print(f"  벗어난 거리 차이 mm: 중앙 {np.median(d):.2f}, 95% {np.percentile(d, 95):.2f}, 최대 {d.max():.2f} ({len(d)}프레임)")
        if e == 0:
            print(f"  실제 위치 기준 정답 → 모듈: " + ", ".join(f"{a}→{b} {conf_true[i, k]}" for i, a in enumerate(names) for k, b in enumerate(names) if conf_true[i, k]))
            for m in missed[:10]:
                print(f"    놓침: 경로 {m[0]} {m[1]:.1f}s 실제 벗어남 {m[2]:+.0f} mm, 모듈 {m[3]:+.0f} mm")
        print(f"  교차로 구간 {sum(zone.values())}프레임: 확인됨 {zone['ok']}, 불일치 {zone['mismatch']}, 확인 못 함 {zone['none']}")
        print(f"  길 끝 구간 {nt}프레임(따로 셈): 확인됨 {tail['ok']} ({tail['ok'] / max(nt, 1):.0%}), 불일치 {tail['mismatch']}, 확인 못 함 {tail['none']}")
        print(f"  비전 확인(길 끝 구간을 뺀 차로 안 {n}프레임): 확인됨 {stat['ok']} ({stat['ok'] / n:.0%}), 불일치 {stat['mismatch']} ({stat['mismatch'] / n:.0%}), 확인 못 함 {stat['none']} ({stat['none'] / n:.0%})"
              + (f" | 확인한 것 중 불일치 {stat['mismatch'] / len(lat):.0%}, 비전이 낸 옆 오차 중앙 {np.median(lat):+.1f} mm, 넣은 값과 차이 95% {np.percentile(np.abs(lat - e), 95):.1f} mm" if len(lat) else ""))


if __name__ == "__main__":
    main()
