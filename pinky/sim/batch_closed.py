"""여러 도로에서 폐루프 주행(sim_closed.py)을 차례로 돌리고 결과를 표로 모은다.

사용 예 (저장소 루트에서):
  python pinky/sim/batch_closed.py --weights ~/pinky_sim/models/pinky_v2b/weights/best.pt --direct --wide
  (--weights를 빼면 시뮬 정답 라벨로 주행. --wide = 넓은 규격 도로. --roads로 이미 만든 도로 폴더를 줄 수도 있음)

굽은 도로는 도로마다 달리는 차로를 바꿔 가며(안쪽 차로, 바깥쪽 차로) 주행한다.
교차로(--type cross)는 도로마다 직진 경로 하나와 좌회전 경로 하나를 주행한다. 영상은 --out/<도로 이름>_<경로>/closed.avi.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
from ego_label import route  # noqa: E402

KIND = {"lane": "차로", "straight": "직진", "left": "좌회전"}


def main():
    ap = argparse.ArgumentParser(description="여러 도로 폐루프 주행")
    ap.add_argument("--count", type=int, default=5), ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--wide", action="store_true", help="넓은 규격 도로")
    ap.add_argument("--type", choices=["winding", "cross"], default="winding", help="도로 종류 (굽은 도로, 교차로)")
    ap.add_argument("--roads", default="", help="이미 만든 도로 폴더 (주면 새로 만들지 않음)")
    ap.add_argument("--out", default="~/pinky_sim/closed_batch")
    ap.add_argument("--weights", default=""), ap.add_argument("--direct", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    roads = os.path.expanduser(a.roads) or os.path.join(out, "roads")
    os.makedirs(out, exist_ok=True)
    if not a.roads:
        subprocess.run([sys.executable, os.path.join(SIM, "gen_roads.py"), "--count", str(a.count), "--seed", str(a.seed),
                        "--type", a.type, "--out", roads] + ["--wide"] * a.wide, check=True, capture_output=True)
    names = sorted(d for d in os.listdir(roads) if os.path.isfile(os.path.join(roads, d, "road.json")))[:a.count]
    rows, env = [], dict(os.environ, MUJOCO_GL="egl")
    runs = []                                                 # (도로 이름, 도로 정보, 경로 번호, 경로 종류)
    for i, name in enumerate(names):
        info = json.load(open(os.path.join(roads, name, "road.json")))
        kinds = [route(info, j)[0] for j in range(len(info["paths"]))]
        if info["type"] == "cross":                           # 교차로: 직진 하나, 좌회전 하나
            runs += [(name, info, kinds.index(k), k) for k in ("straight", "left") if k in kinds]
        else:                                                 # 굽은 도로: 0 = 중앙선 옆 차로, 1 = 바깥쪽 차로
            runs.append((name, info, i % info["lanes_per_dir"], "lane"))
    for i, (name, info, lane, kind) in enumerate(runs):
        tag = name if kind == "lane" else f"{name}_{kind}"
        t = time.time()
        cmd = [sys.executable, os.path.join(SIM, "sim_closed.py"), os.path.join(roads, name), os.path.join(out, tag),
               "--path", str(lane)] + (["--weights", a.weights] if a.weights else []) + ["--direct"] * a.direct
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        txt = r.stdout + r.stderr
        m1 = re.search(r"결과: (.+?)\s+\(주행 ([\d.]+) / ([\d.]+) m", txt)
        m2 = re.search(r"중앙 ([\d.]+)%, 최대 ([\d.]+)%\s+\| 중심선 없는 프레임 (\d+)", txt)
        if not (m1 and m2):
            print(f"[{i + 1}/{len(runs)}] {tag}: 실패\n" + "\n".join(txt.strip().splitlines()[-4:]))
            continue
        row = {"도로": name, "경로": f"{KIND[kind]} {lane}", "차로폭mm": round(info["lane_w"]), "선mm": round(info.get("line_w", 0)),
               "점선mm": f"{info.get('dash', 0):.0f}/{info.get('gap', 0):.0f}", "결과": m1.group(1),
               "주행m": f"{m1.group(2)}/{m1.group(3)}", "중앙%": m2.group(1), "최대%": m2.group(2), "중심선없음": m2.group(3)}
        rows.append(row)
        print(f"[{i + 1}/{len(runs)}] {name} {row['경로']}: {row['결과']}, 중앙 {row['중앙%']}%, 최대 {row['최대%']}%  ({time.time() - t:.0f}초)", flush=True)
    if not rows:
        return
    keys = list(rows[0])
    print("\n" + "  ".join(f"{k:>12s}" for k in keys))
    for r in rows:
        print("  ".join(f"{str(r[k]):>12s}" for k in keys))
    done = sum(r["결과"] == "완주" for r in rows)
    print(f"\n완주 {done}/{len(rows)}, 벗어난 거리(차로 폭 대비) 중앙값의 평균 {sum(float(r['중앙%']) for r in rows) / len(rows):.1f}%,"
          f" 최대값 중 가장 큰 것 {max(float(r['최대%']) for r in rows):.1f}%")
    with open(os.path.join(out, "closed_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {out}/closed_summary.csv, 영상: {out}/<도로 이름>[_straight|_left]/closed.avi")


if __name__ == "__main__":
    main()
