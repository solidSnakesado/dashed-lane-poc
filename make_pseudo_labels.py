"""사전학습된 UFLDv2 모델로 labelme 초기 라벨(lane0~lane3 꺾은선)을 생성하는 스크립트.

사용법 (Ultra-Fast-Lane-Detection-v2 레포 최상위 폴더에서 실행):
    python make_pseudo_labels.py configs/culane_res18.py weights/culane_res18.pth ~/lane/dataset/images

각 이미지 옆에 <이미지이름>.json(labelme 형식)을 저장한다. 기본적으로 이미 있는 .json은 덮어쓰지 않는다.

파인튜닝한 모델로 아직 확인하지 않은 프레임의 라벨을 다시 만들 때:
    python make_pseudo_labels.py configs/culane_res18.py ~/lane/runs/exp1/last.pth ~/lane/dataset/images \
        --redo_unreviewed --start 213 --end 420
    - 범위 안에서 reviewed가 체크되지 않은 프레임만 다시 만든다 (원래 파일은 .json.bak으로 백업).
    - reviewed가 체크된 프레임은 절대 건드리지 않는다.
슬롯 의미: lane0 / lane3 = 열 기준 슬롯 (옆 차선 / 회전 유도선), lane1 / lane2 = 행 기준 슬롯 (내 차선).
"""
import argparse
import shutil
import sys
import types

# 학습 전용 DALI 모듈을 빈 모듈로 대체 (추론에는 DALI / my_interp가 필요 없음)
_stub = types.ModuleType("data.dali_data")
_stub.TrainCollect = None
sys.modules["data.dali_data"] = _stub

import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.common import get_model
from utils.config import Config

ROW_SLOTS = [1, 2]
COL_SLOTS = [0, 3]
KEEP_EVERY = 2  # N개 점마다 1개만 남겨서 라벨 수정을 쉽게 함


def decode(pred, row_anchor, col_anchor, w, h, local_width=1, both=False):
    """demo.pred2coords와 같은 방식으로 디코딩하되 슬롯 번호를 유지한다. 반환값: {슬롯: [(x, y), ...]}
    both=True: 모든 슬롯을 행 기준과 열 기준으로 모두 읽어 합친다 (회전 구간처럼 선이 가로로 누운 경우 확인용).
    이때 점 순서는 보장되지 않으므로 선이 아니라 점으로 그려야 한다."""
    row_slots, col_slots = ([0, 1, 2, 3], [0, 1, 2, 3]) if both else (ROW_SLOTS, COL_SLOTS)
    row_th, col_th = (0.2, 0.2) if both else (0.5, 0.25)  # 차선으로 인정할 최소 검출 비율
    _, num_grid_row, num_cls_row, _ = pred["loc_row"].shape
    _, num_grid_col, num_cls_col, _ = pred["loc_col"].shape
    max_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    max_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()
    loc_row = pred["loc_row"].cpu()
    loc_col = pred["loc_col"].cpu()

    lanes = {}
    for i in row_slots:
        if valid_row[0, :, i].sum() > num_cls_row * row_th:
            pts = []
            for k in range(num_cls_row):
                if valid_row[0, k, i]:
                    idx = torch.arange(max(0, max_row[0, k, i] - local_width),
                                       min(num_grid_row - 1, max_row[0, k, i] + local_width) + 1)
                    x = (loc_row[0, idx, k, i].softmax(0) * idx.float()).sum() + 0.5
                    pts.append((float(x / (num_grid_row - 1) * w), float(row_anchor[k] * h)))
            lanes[i] = lanes.get(i, []) + pts
    for i in col_slots:
        if valid_col[0, :, i].sum() > num_cls_col * col_th:
            pts = []
            for k in range(num_cls_col):
                if valid_col[0, k, i]:
                    idx = torch.arange(max(0, max_col[0, k, i] - local_width),
                                       min(num_grid_col - 1, max_col[0, k, i] + local_width) + 1)
                    y = (loc_col[0, idx, k, i].softmax(0) * idx.float()).sum() + 0.5
                    pts.append((float(col_anchor[k] * w), float(y / (num_grid_col - 1) * h)))
            lanes[i] = lanes.get(i, []) + pts
    return lanes


def thin(pts):
    # 점 개수를 줄이되 마지막 점은 항상 유지
    if len(pts) <= 2:
        return pts
    kept = pts[::KEEP_EVERY]
    if kept[-1] != pts[-1]:
        kept.append(pts[-1])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"), ap.add_argument("weights"), ap.add_argument("img_dir")
    ap.add_argument("--redo_unreviewed", action="store_true", help="reviewed가 아닌 프레임의 라벨을 다시 생성")
    ap.add_argument("--start", type=int, default=0, help="처리할 첫 프레임 번호")
    ap.add_argument("--end", type=int, default=10 ** 9, help="처리할 마지막 프레임 번호")
    args = ap.parse_args()
    cfg_path, weight_path = args.config, args.weights
    img_dir = os.path.expanduser(args.img_dir)

    cfg = Config.fromfile(cfg_path)
    # CULane 설정의 행/열 기준 위치 (이미지 높이·너비 대비 비율)
    cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    net = get_model(cfg)
    sd = torch.load(weight_path, map_location="cpu")["model"]
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False)
    net.eval()

    tf = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    made = skipped = 0
    with torch.no_grad():
        for p in paths:
            out_json = os.path.splitext(p)[0] + ".json"
            if not (args.start <= int(os.path.splitext(os.path.basename(p))[0]) <= args.end):
                continue  # 범위 밖 프레임
            if os.path.exists(out_json):
                with open(out_json) as f:
                    reviewed = bool(json.load(f).get("flags", {}).get("reviewed", False))
                # 확인한 프레임은 항상 보호, 그 외 기존 파일은 --redo_unreviewed일 때만 다시 생성
                if reviewed or not args.redo_unreviewed:
                    skipped += 1
                    continue
                shutil.copy(out_json, out_json + ".bak")  # 원래 라벨 백업
            img = Image.open(p).convert("RGB")
            w, h = img.size
            # demo.py와 같은 전처리: 리사이즈 후 아래쪽 train_height만큼 잘라냄
            x = tf(img)[:, -cfg.train_height:, :].unsqueeze(0).cuda()
            lanes = decode(net(x), cfg.row_anchor, cfg.col_anchor, w, h)

            shapes = []
            for slot in sorted(lanes):
                pts = thin(lanes[slot])
                if len(pts) < 2:
                    continue
                shapes.append({"label": f"lane{slot}", "points": [[round(a, 1), round(b, 1)] for a, b in pts],
                               "group_id": None, "description": "", "shape_type": "linestrip", "flags": {}})
            with open(out_json, "w") as f:
                json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes,
                           "imagePath": os.path.basename(p), "imageData": None,
                           "imageHeight": h, "imageWidth": w}, f, indent=2)
            made += 1
    print(f"done: {made} json created, {skipped} skipped (보호/기존 유지), total images {len(paths)}")


if __name__ == "__main__":
    main()
