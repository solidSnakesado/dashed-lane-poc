"""여러 도로에서 폐루프 주행(sim_closed.py)을 차례로 돌리고 결과를 표로 모은다.

사용 예 (저장소 루트에서):
  python pinky/sim/batch_closed.py --weights ~/pinky_sim/models/pinky_v2b/weights/best.pt --direct --wide
  (--weights를 빼면 시뮬 정답 라벨로 주행. --wide = 넓은 규격 도로. --roads로 이미 만든 도로 폴더를 줄 수도 있음)

도로마다 달리는 차로를 바꿔 가며(안쪽 차로, 바깥쪽 차로) 주행한다. 영상은 --out/<도로 이름>/closed.avi.
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


def main():
    ap = argparse.ArgumentParser(description="여러 도로 폐루프 주행")
    ap.add_argument("--count", type=int, default=5), ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--wide", action="store_true", help="넓은 규격 도로")
    ap.add_argument("--roads", default="", help="이미 만든 도로 폴더 (주면 새로 만들지 않음)")
    ap.add_argument("--out", default="~/pinky_sim/closed_batch")
    ap.add_argument("--weights", default=""), ap.add_argument("--direct", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    roads = os.path.expanduser(a.roads) or os.path.join(out, "roads")
    os.makedirs(out, exist_ok=True)
    if not a.roads:
        subprocess.run([sys.executable, os.path.join(SIM, "gen_roads.py"), "--count", str(a.count), "--seed", str(a.seed),
                        "--type", "winding", "--out", roads] + ["--wide"] * a.wide, check=True, capture_output=True)
    names = sorted(d for d in os.listdir(roads) if os.path.isfile(os.path.join(roads, d, "road.json")))[:a.count]
    rows, env = [], dict(os.environ, MUJOCO_GL="egl")
    for i, name in enumerate(names):
        info = json.load(open(os.path.join(roads, name, "road.json")))
        lane = i % info["lanes_per_dir"]                      # 0 = 중앙선 옆 차로, 1 = 바깥쪽 차로
        t = time.time()
        cmd = [sys.executable, os.path.join(SIM, "sim_closed.py"), os.path.join(roads, name), os.path.join(out, name),
               "--path", str(lane)] + (["--weights", a.weights] if a.weights else []) + ["--direct"] * a.direct
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        txt = r.stdout + r.stderr
        m1 = re.search(r"결과: (.+?)\s+\(주행 ([\d.]+) / ([\d.]+) m", txt)
        m2 = re.search(r"중앙 ([\d.]+)%, 최대 ([\d.]+)%\s+\| 중심선 없는 프레임 (\d+)", txt)
        if not (m1 and m2):
            print(f"[{i + 1}/{len(names)}] {name}: 실패\n" + "\n".join(txt.strip().splitlines()[-4:]))
            continue
        row = {"도로": name, "차로": lane, "차로폭mm": round(info["lane_w"]), "선mm": round(info.get("line_w", 0)),
               "점선mm": f"{info.get('dash', 0):.0f}/{info.get('gap', 0):.0f}", "결과": m1.group(1),
               "주행m": f"{m1.group(2)}/{m1.group(3)}", "중앙%": m2.group(1), "최대%": m2.group(2), "중심선없음": m2.group(3)}
        rows.append(row)
        print(f"[{i + 1}/{len(names)}] {name} 차로 {lane}: {row['결과']}, 중앙 {row['중앙%']}%, 최대 {row['최대%']}%  ({time.time() - t:.0f}초)", flush=True)
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
    print(f"저장: {out}/closed_summary.csv, 영상: {out}/<도로 이름>/closed.avi")


if __name__ == "__main__":
    main()
