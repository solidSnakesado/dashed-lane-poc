"""영상에 학습 모델을 적용해 프레임별 검출 폴리곤을 파일 하나로 저장한다.

모델 실행(GPU)과 후처리 개발을 나누기 위한 것이다. 저장한 파일만 있으면 모델 없이 후처리를 돌릴 수 있다.

사용법:
    python dump_dets.py ~/lane/videos/turn.mp4 ~/lane/runs/seg_aihub/weights/best.pt
결과: --out 폴더(기본: 윈도우 다운로드 폴더)에 <영상이름>_dets.json.gz
핑키 모델: python dump_dets.py drive.mp4 best.pt --stride 1 --imgsz 640 --out 결과폴더
"""
import argparse
import gzip
import json
import os

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"), ap.add_argument("weights")
    ap.add_argument("--stride", type=int, default=2, help="몇 프레임마다 한 장씩 처리할지")
    ap.add_argument("--start", type=float, default=0, help="시작 시각(초)")
    ap.add_argument("--end", type=float, default=0, help="끝 시각(초), 0이면 끝까지")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="저장할 최소 신뢰도. 낮게 저장해 두고 후처리에서 기준을 다시 고른다")
    ap.add_argument("--imgsz", type=int, default=960, help="모델 입력 크기. 학습 때와 같게(AI 허브 960, 핑키 640)")
    ap.add_argument("--out", default="/mnt/c/Users/Snake/Downloads", help="결과 저장 폴더")
    args = ap.parse_args()
    video = os.path.expanduser(args.video)

    model, cap = YOLO(os.path.expanduser(args.weights)), cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first = int(args.start * fps)
    last = int(args.end * fps) if args.end > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    frames, total = [], 0
    for k in range(first, last):
        ok, img = cap.read()
        if not ok:
            break
        if (k - first) % args.stride:
            continue
        # retina_masks: 마스크를 원본 해상도로 만들어 가는 선의 윤곽을 살림
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)[0]
        polys = []
        if r.masks is not None:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for xy, c, p in zip(r.masks.xy, cls, conf):
                if len(xy) >= 3:
                    # [클래스 번호, 신뢰도, x1, y1, x2, y2, ...] (좌표는 0.1픽셀 단위로 반올림)
                    polys.append([int(c), round(float(p), 3)] + [round(float(v), 1) for v in xy.reshape(-1)])
        frames.append({"k": k, "polys": polys})
        total += len(polys)
        if len(frames) % 100 == 0:
            print(f"  {len(frames)}프레임 처리", flush=True)

    name = os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(os.path.expanduser(args.out), f"{name}_dets.json.gz")
    data = {"video": os.path.basename(video), "fps": fps, "stride": args.stride, "size": size,
            "names": {int(k): v for k, v in model.names.items()}, "frames": frames}
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"프레임 {len(frames)}장, 폴리곤 {total}개 (프레임당 평균 {total / max(len(frames), 1):.1f}개)")
    print(f"저장: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
