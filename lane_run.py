"""차선 후처리(3~7단계)를 주행 1건에 실행하고 결과 영상·그림·CSV를 만든다.
3 도로 펼치기 → 4 조각 묶기 → 5 곡선 맞춤 → 6 차선 번호 → 7 추적.
사용법:
    정답 라벨 입력:  python lane_run.py aihub ~/lane/aihub_val ~/lane/aihub_calib --scene 08_100616_220614
    (같은 주행에 모델 적용: 위 명령에 --weights ~/lane/runs/seg_aihub/weights/best.pt 추가)
    학습 모델 입력:  python lane_run.py video ~/lane/videos/turn.mp4 ~/lane/runs/seg_aihub/weights/best.pt ~/lane/H_drive.npy
    저장한 검출 입력: python lane_run.py dets ~/lane/videos/turn.mp4 turn_dets.json.gz ~/lane/H_drive.npy
"""
import argparse
import csv
import glob
import gzip
import json
import os

import cv2
import numpy as np

from lane_draw import draw
from lane_geom import Camera, drop_marks, frame_to_pieces
from lane_group import build_lanes
from lane_track import Tracker

# AI 허브 클래스 이름 → 이 프로젝트의 클래스 이름 (YOLO 학습 클래스와 같음)
AIHUB_CLASS = {"whiteLane": "white_lane", "stopLane": "stop_line",
               "crossWalk": "crosswalk", "roadMark": "road_mark"}
AIHUB_FPS = 2       # AI 허브 프레임 간격은 약 0.5초


def read_aihub(label_path):
    """AI 허브 라벨 JSON → [(클래스, 픽셀 폴리곤 (N,2))]."""
    with open(label_path, encoding="utf-8") as f:
        anns = json.load(f).get("annotations", [])
    out = []
    for a in anns:
        cls = AIHUB_CLASS.get(a.get("class", ""))
        pts = np.array(a.get("polygon", []), np.float64).reshape(-1, 2)
        if cls and len(pts) >= 3:
            out.append((cls, pts))
    return out


def predict(model, img, conf):
    """이미지 한 장에 모델을 적용해 [(클래스, 픽셀 폴리곤)]을 돌려준다."""
    # retina_masks: 마스크를 원본 해상도로 만들어 가는 선의 윤곽을 살림
    r = model.predict(img, imgsz=960, conf=conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [(model.names[int(c)], np.asarray(xy, np.float64)) for xy, c in zip(r.masks.xy, cls) if len(xy) >= 3]


def aihub_frames(root, calib_dir, scene, weights, conf):
    """AI 허브 주행 1건의 (이름, 이미지, 폴리곤 목록)을 차례로 내놓는다. weights를 주면 라벨 대신 모델을 쓴다."""
    with open(os.path.join(calib_dir, "scene_map.json")) as f:
        cam = Camera(os.path.join(calib_dir, json.load(f)[scene]))     # 주행에 맞는 보정 파일
    labels = sorted(glob.glob(os.path.join(root, "labels", "2D", scene, "sensor_raw_data", "camera", "*.json")))
    if weights:
        from ultralytics import YOLO
        model = YOLO(weights)

    def gen():
        for lp in labels:
            name = os.path.splitext(os.path.basename(lp))[0]
            img = cv2.imread(os.path.join(root, "images", scene, "sensor_raw_data", "camera", name + ".jpg"))
            yield name, img, (predict(model, img, conf) if weights else read_aihub(lp))
    return gen(), cam, AIHUB_FPS, len(labels)


def video_frames(path, weights, calib, stride, start, end, conf):
    """영상 프레임을 학습 모델에 넣어 (이름, 이미지, 폴리곤 목록)을 차례로 내놓는다."""
    from ultralytics import YOLO        # 정답 라벨만 쓸 때는 필요 없으므로 여기서 불러옴
    model, cap = YOLO(weights), cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first = int(start * fps)
    last = int(end * fps) if end > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    def gen():
        for k in range(first, last):
            ok, img = cap.read()
            if not ok:
                break
            if (k - first) % stride == 0:
                yield f"f{k:06d}", img, predict(model, img, conf)
    return gen(), Camera(calib), fps / stride, (last - first) // stride


def dets_frames(path, dets_path, calib, conf):
    """dump_dets.py가 저장한 검출 결과와 영상을 함께 읽는다. 모델 없이 후처리만 돌릴 때 쓴다."""
    with gzip.open(dets_path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cap, names = cv2.VideoCapture(path), d["names"]

    def gen():
        pos = -1
        for fr in d["frames"]:
            while pos < fr["k"]:                    # 저장된 프레임 번호까지 영상을 넘김
                ok, img = cap.read()
                pos += 1
            polys = [(names[str(p[0])], np.array(p[2:], np.float64).reshape(-1, 2))
                     for p in fr["polys"] if p[1] >= conf]
            yield f"f{fr['k']:06d}", img, polys
    return gen(), Camera(calib), d["fps"] / d["stride"], len(d["frames"])


def csv_rows(name, tracks, left, right, center):
    """한 프레임의 선과 중심선을 CSV 행으로 만든다. 점은 2m 간격."""
    rows = []
    for t in tracks:
        role = "lane1" if t is left else "lane2" if t is right else "other"
        for k, (x, z) in enumerate(t.curve.sample(2.0)):
            rows.append([name, role, t.id, t.kind, int(t.det is not None), k, f"{x:.3f}", f"{z:.3f}"])
    if center is not None:
        for k, (x, z) in enumerate(center[0].sample(2.0)):
            rows.append([name, "center", -1, "", int(not center[1]), k, f"{x:.3f}", f"{z:.3f}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("aihub", help="AI 허브 정답 라벨로 실행")
    a.add_argument("root", help="추출한 AI 허브 폴더 (images, labels가 들어 있는 곳)")
    a.add_argument("calib_dir", help="aihub_calib.py가 만든 보정 폴더")
    a.add_argument("--scene", required=True, help="주행 이름 (예: 08_100616_220614)")
    a.add_argument("--weights", default="", help="주면 정답 라벨 대신 이 모델의 검출 결과를 씀")
    a.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 기준")
    v = sub.add_parser("video", help="영상에 학습 모델을 적용해 실행")
    v.add_argument("video"), v.add_argument("weights"), v.add_argument("calib", help="H 행렬(.npy) 또는 보정(.npz)")
    v.add_argument("--stride", type=int, default=1, help="몇 프레임마다 한 장씩 처리할지")
    v.add_argument("--start", type=float, default=0, help="시작 시각(초)")
    v.add_argument("--end", type=float, default=0, help="끝 시각(초), 0이면 끝까지")
    v.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 기준")
    d = sub.add_parser("dets", help="dump_dets.py로 저장한 검출 결과로 실행 (모델 불필요)")
    d.add_argument("video"), d.add_argument("dets"), d.add_argument("calib")
    d.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 기준")
    for p in (v, d):
        p.add_argument("--all-classes", action="store_true", help="정지선·횡단보도·노면표시도 함께 쓰기 (기본: 차선만)")
    for p in (a, v, d):
        p.add_argument("--out", default="/mnt/c/Users/Snake/Downloads", help="결과 저장 폴더")
    args = ap.parse_args()
    ex = os.path.expanduser
    if args.mode == "aihub":
        frames, cam, fps, total = aihub_frames(ex(args.root), ex(args.calib_dir), args.scene,
                                               ex(args.weights), args.conf)
        tag = args.scene + ("_model" if args.weights else "")
    else:
        if args.mode == "video":
            frames, cam, fps, total = video_frames(ex(args.video), ex(args.weights), ex(args.calib),
                                                   args.stride, args.start, args.end, args.conf)
        else:
            frames, cam, fps, total = dets_frames(ex(args.video), ex(args.dets), ex(args.calib), args.conf)
        tag = os.path.splitext(os.path.basename(args.video))[0]
    print(f"{tag}: 약 {total}프레임, 초당 {fps:.1f}장")

    tracker, writer, rows = Tracker(fps), None, []
    stat, n_piece, n_lane = {"both": 0, "one": 0, "none": 0}, 0, 0
    seen_cls = {}                                           # 클래스별 입력 폴리곤 수
    base = os.path.join(ex(args.out), f"lane_{tag}")
    lane_only = args.mode != "aihub" and not args.all_classes
    for i, (name, img, polys) in enumerate(frames):
        polys = drop_marks(polys) if lane_only else polys   # 차선만 남기고, 바닥 표시와 겹친 것은 뺌
        for cls, _ in polys:
            seen_cls[cls] = seen_cls.get(cls, 0) + 1
        pieces, others = frame_to_pieces(polys, cam)        # 3단계
        lanes = build_lanes(pieces)                         # 4~5단계
        tracks = tracker.update(lanes)                      # 7단계
        left, right = tracker.roles()                       # 6단계
        pts, guess = tracker.center(left, right)
        center = None if pts is None else (pts, guess)
        stat["both" if left and right else "one" if left or right else "none"] += 1
        n_piece, n_lane = n_piece + len(pieces), n_lane + len(lanes)
        ids = f"L1={left.id if left else '-'} L2={right.id if right else '-'}"
        frame = draw(img, polys, tracks, left, right, center, others, cam,
                     f"{name}  pieces={len(pieces)} lanes={len(lanes)} tracks={len(tracks)} {ids}")
        if writer is None:
            writer = cv2.VideoWriter(base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if i == total // 2:                                 # 가운데 프레임은 그림으로도 저장
            cv2.imwrite(base + ".jpg", frame)
        rows += csv_rows(name, tracks, left, right, center)
        if args.mode == "aihub" or i % 100 == 0:
            print(f"  {name}: 조각 {len(pieces)} → 선 {len(lanes)} → 추적 {len(tracks)}, {ids}", flush=True)
    writer.release()
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "role", "track_id", "kind", "seen", "idx", "x_m", "z_m"])
        w.writerows(rows)
    n = max(sum(stat.values()), 1)
    print("입력 폴리곤(프레임당 평균): " + ", ".join(f"{k} {c / n:.1f}" for k, c in sorted(seen_cls.items())))
    print(f"처리 {n}프레임, 프레임당 평균 조각 {n_piece / n:.1f}개 → 선 {n_lane / n:.1f}개")
    print(f"양쪽 선 모두 있음 {stat['both']}프레임, 한쪽만 {stat['one']}, 없음 {stat['none']}")
    print(f"추적 번호를 새로 만든 횟수: {tracker.next_id}")
    print(f"저장: {base}.mp4 / .jpg / .csv")


if __name__ == "__main__":
    main()
