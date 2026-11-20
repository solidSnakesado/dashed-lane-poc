"""여러 시뮬 도로에서 주행 → 검출 → 후처리 → 평가를 한 번에 돌리고 도로별로 요약한다 (6-5단계).

사용 예 (저장소 루트에서):
  처음:        python pinky/sim/batch_eval.py
  후처리만 다시: python pinky/sim/batch_eval.py --post-only

도로마다 하는 일 (--post-only면 4·5만):
  1. gen_roads.py   구불구불한 도로 생성 (학습에 안 쓴 시드)
  2. sim_drive.py   주행 영상과 정답(gt.csv)
  3. dump_dets.py   모델 검출 결과 저장 / pinky_camera.py  카메라 변환 파일 (도로 규격은 넣지 않음)
  4. lane_run.py    후처리 (도로 규격은 화면에서 직접 잼, 로봇 움직임은 gt.csv 값을 넘김)
  5. eval_drive.py  정답과 비교
결과: --out 아래 roads/(도로), drives/<도로 이름>/(영상·정답·검출·결과), batch_summary.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SIM, CONF = os.path.join(ROOT, "pinky", "sim"), os.path.join(ROOT, "pinky", "config")


def run(args, log, env=None):
    """스크립트 하나를 돌린다. 실패하면 마지막 출력을 보여 주고 False."""
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, env=env, cwd=ROOT)
    with open(log, "a") as f:
        f.write(f"$ {' '.join(args)}\n{r.stdout}{r.stderr}\n")
    if r.returncode != 0:
        print(f"    실패: {os.path.basename(args[0])}\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-5:]))
    return r.returncode == 0


def make_inputs(road, drv, a, seed, log):
    """주행 영상·정답·검출 결과·변환 파일을 만든다 (처음 한 번)."""
    env = dict(os.environ, MUJOCO_GL="egl")
    steps = [
        ([os.path.join(SIM, "sim_drive.py"), road, a.cam, drv, "--seed", str(seed)] + ["--gt-dets"] * a.gt_dets, env),
        ([os.path.join(ROOT, "dump_dets.py"), os.path.join(drv, "drive.mp4"), a.weights,
          "--stride", "1", "--imgsz", str(a.imgsz), "--out", drv], None),
        ([os.path.join(CONF, "pinky_camera.py"), a.cam, os.path.join(drv, "calib.npz")], None),   # 실제 m 단위
    ]
    return all(run(s, log, e) for s, e in (steps[::2] if a.gt_dets else steps))   # 정답 검출을 쓰면 모델 단계는 건너뜀


def post(road, drv, log, a):
    """후처리와 평가를 돌린다 (후처리를 고칠 때마다 다시)."""
    mp4, calib = os.path.join(drv, "drive.mp4"), os.path.join(drv, "calib.npz")
    dets = os.path.join(drv, "drive_gtdets.json.gz" if a.gt_dets else "drive_dets.json.gz")
    return (run([os.path.join(ROOT, "lane_run.py"), "dets", mp4, dets, calib, "--out", drv, "--odom",
                 os.path.join(drv, "gt.csv"), "--odom-noise", str(a.odom_noise),
                 "--auto"], log)
            and run([os.path.join(SIM, "eval_drive.py"), os.path.join(drv, "lane_drive.csv"),
                     os.path.join(drv, "gt.csv"), calib], log))


def summarize(name, road, drv):
    """도로 하나의 평가 결과를 한 줄로 요약한다. 오차 목록도 함께 돌려줘 전체 합계에 씀."""
    roles = {}
    with open(os.path.join(drv, "lane_drive.csv")) as f:
        for r in csv.DictReader(f):
            roles.setdefault(int(r["frame"].lstrip("f")), set()).add(r["role"])
    rows = list(csv.DictReader(open(os.path.join(drv, "eval_drive.csv"))))
    n, first = len(rows), int(rows[0]["frame"])          # 규격 측정 구간(처음 first프레임)은 평가에 없음
    both = sum({"lane1", "lane2"} <= v for k, v in roles.items() if k >= first)
    e2 = [abs(float(r[f"{k}_err_m"])) * 100 for r in rows for k in ("lane1", "lane2") if r[f"{k}_err_m"]]      # cm
    e4 = [abs(float(r[f"{k}_err_pct"])) for r in rows for k in ("lane1", "lane2") if r[f"{k}_err_pct"]]      # 차로 폭 대비 %
    good = sum(r["lane1_판정"] == "정답" and r["lane2_판정"] == "정답" for r in rows)
    judged = sum("평가제외" not in (r["lane1_판정"], r["lane2_판정"]) for r in rows)   # 양쪽 선 모두 화면 안인 프레임
    c1, c2 = Counter(r["lane1_판정"] for r in rows), Counter(r["lane2_판정"] for r in rows)
    curve = sum(r["segment"] == "곡선" for r in rows)
    info = json.load(open(os.path.join(road, "road.json")))
    s = {"도로": name, "차로폭mm": round(info["lane_w"]), "점선mm": f"{info.get('dash', 0):.0f}/{info.get('gap', 0):.0f}",
         "선mm": round(info.get("line_w", 0)), "곡선%": round(100 * curve / max(n, 1)),
         "양쪽%": round(100 * both / max(n, 1)), "양쪽정답%": round(100 * good / max(judged, 1)),
         "오차cm": round(float(np.median(e2)), 1) if e2 else "", "오차%": round(float(np.median(e4)), 1) if e4 else "",
         "L1틀림": c1["다른선"] + c1["반대선"], "L1없음": c1["없음"],
         "L2틀림": c2["다른선"] + c2["반대선"], "L2없음": c2["없음"], "L2어긋남": c2["어긋남"], "프레임": n, "측정": first}
    return s, (both, good, n, e2, e4, c1, c2, judged)


def main():
    ap = argparse.ArgumentParser(description="여러 시뮬 도로에서 일괄 평가")
    ap.add_argument("--count", type=int, default=10, help="도로 수")
    ap.add_argument("--seed", type=int, default=30, help="도로 시드 (학습용 1과 다르게)")
    ap.add_argument("--out", default="~/pinky_sim/batch", help="결과 폴더")
    ap.add_argument("--weights", default="~/pinky_sim/models/pinky_v1/weights/best.pt")
    ap.add_argument("--cam", default=os.path.join(CONF, "pinky_camera.yaml"))
    ap.add_argument("--odom-noise", type=float, default=0.0, help="로봇 움직임 값에 섞을 오차 비율 (예: 0.02)")
    ap.add_argument("--wide", action="store_true", help="도로 규격을 넓은 범위에서 뽑기 (규격이 바뀌는 경우 시험)")
    ap.add_argument("--gt-dets", action="store_true", help="모델 대신 시뮬 정답 라벨을 검출로 사용 (후처리만 시험)")
    ap.add_argument("--imgsz", type=int, default=640, help="모델 입력 크기 (학습 때와 같게)")
    ap.add_argument("--post-only", action="store_true", help="저장된 영상·검출로 후처리·평가만 다시")
    a = ap.parse_args()
    out, a.weights = os.path.expanduser(a.out), os.path.expanduser(a.weights)
    roads_dir, log = os.path.join(out, "roads"), os.path.join(out, "batch_log.txt")
    os.makedirs(out, exist_ok=True)
    open(log, "w").close()
    t0 = time.time()
    if not a.post_only:
        print(f"도로 {a.count}개 생성 (시드 {a.seed}) ...")
        if not run([os.path.join(SIM, "gen_roads.py"), "--count", str(a.count), "--seed", str(a.seed),
                    "--type", "winding", "--out", roads_dir] + ["--wide"] * a.wide, log):
            return
    names = sorted(d for d in os.listdir(roads_dir) if os.path.isfile(os.path.join(roads_dir, d, "road.json")))
    summary, tot = [], []
    for i, name in enumerate(names):
        road, drv = os.path.join(roads_dir, name), os.path.join(out, "drives", name)
        t = time.time()
        print(f"[{i + 1}/{len(names)}] {name}")
        if not a.post_only and not make_inputs(road, drv, a, i, log):
            continue
        if not post(road, drv, log, a):
            continue
        s, raw = summarize(name, road, drv)
        summary.append(s)
        tot.append(raw)
        print(f"    양쪽 {s['양쪽%']}%, 양쪽 정답 {s['양쪽정답%']}%, 오차 {s['오차%']}% (차로 폭 대비)  ({time.time() - t:.0f}초)")
    if not summary:
        print("평가된 도로가 없습니다. 기록:", log)
        return
    report(summary, tot, out)
    print(f"전체 {time.time() - t0:.0f}초, 기록: {log}")


def report(summary, tot, out):
    """도로별 요약표와 전체 합계를 출력하고 batch_summary.csv로 저장한다."""
    keys = list(summary[0])
    print("\n" + "  ".join(f"{k:>9s}" for k in keys))
    for s in summary:
        print("  ".join(f"{str(s[k]):>9s}" for k in keys))
    both, good, n = (sum(t[j] for t in tot) for j in range(3))
    judged = sum(t[7] for t in tot)
    e2, e4 = sum((t[3] for t in tot), []), sum((t[4] for t in tot), [])
    c1, c2 = sum((t[5] for t in tot), Counter()), sum((t[6] for t in tot), Counter())
    print(f"\n[전체 {len(summary)}개 도로, {n}프레임]")
    print(f"  양쪽 선 있음 {both / n:.0%} [{'통과' if both / n >= 0.95 else '미달'}]  | 양쪽 정답 {good / max(judged, 1):.0%} (판정 가능한 {judged}프레임 중)")
    print(f"  옆 오차(차로 폭 대비)  L1·L2 중앙 {np.median(e4):.1f}% / 상위10% {np.percentile(e4, 90):.1f}%"
          f" [{'통과' if np.median(e4) <= 10 else '미달'}]  (길이로는 중앙 {np.median(e2):.1f} cm)")
    for k, c in (("lane1", c1), ("lane2", c2)):
        print(f"  {k} 판정: " + ", ".join(f"{x} {v}" for x, v in sorted(c.items())))
    path = os.path.join(out, "batch_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
