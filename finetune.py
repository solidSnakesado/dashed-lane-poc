"""labelme 라벨(reviewed 체크된 프레임)로 UFLDv2를 파인튜닝하는 스크립트.

사용법 (Ultra-Fast-Lane-Detection-v2 레포 최상위 폴더에서 실행):
    python finetune.py configs/culane_res18.py weights/culane_res18.pth ~/lane/dataset/images ~/lane/runs/exp1

주요 특징:
    - DALI / my_interp / CULane 형식 변환이 필요 없다. labelme JSON을 직접 읽어 학습 라벨을 만든다.
    - 라벨 생성 방식은 원본 data/dali_data.py의 TrainCollect와 같다 (행/열 기준 위치에서 선을 보간).
    - 사전학습 가중치는 백본뿐 아니라 헤드까지 모두 불러온다 (원본 train.py는 백본만 불러옴).
    - 매 에폭 가중치를 저장한다 (ep_XXX.pth, last.pth). ufld_video_poc.py에서 바로 사용할 수 있다.
"""
import sys
import types

# 학습 전용 DALI 모듈을 빈 모듈로 대체 (이 스크립트는 DALI 없이 자체 데이터 로더를 사용)
_stub = types.ModuleType("data.dali_data")
_stub.TrainCollect = None
sys.modules["data.dali_data"] = _stub

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils.common import calc_loss, get_model, inference
from utils.config import Config
from utils.factory import get_loss_dict

SLOTS = ["lane0", "lane1", "lane2", "lane3"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def interp_lane(pts, locs, direction):
    """선(pts, 먼 쪽→가까운 쪽 순서)이 각 기준 위치(locs)를 지나는 좌표를 구한다.
    direction 0: 행 기준(y=loc에서 x를 구함), 1: 열 기준(x=loc에서 y를 구함). 없으면 -1.
    원본 my_interp와 같이 가까운 쪽(배열 끝)부터 찾아 처음 만나는 구간을 사용한다."""
    out = np.full(len(locs), -1.0, dtype=np.float32)
    a, b = 1 - direction, direction  # a: 기준 축, b: 구할 축
    for k, v in enumerate(locs):
        for i in range(len(pts) - 1, 0, -1):
            p, q = pts[i], pts[i - 1]
            if (p[a] - v) * (q[a] - v) <= 0 and p[a] != q[a]:
                t = (v - q[a]) / (p[a] - q[a])
                out[k] = q[b] + t * (p[b] - q[b])
                break
    return out


def extend_to_bottom(xs):
    """원본 TrainCollect._extend와 같은 처리: 화면 아래까지 닿지 않은 선을 직선으로 연장."""
    if xs[-1] > 0:
        return xs
    idx = np.where(xs > 0)[0]
    if len(idx) < 6:
        return xs
    half = idx[len(idx) // 2:]
    p = np.polyfit(half, xs[half], 1)
    out = xs.copy()
    out[half[0]:] = np.polyval(p, np.arange(half[0], len(xs)))
    return out


class LabelmeLaneDataset(Dataset):
    def __init__(self, img_dir, cfg, max_frame, augment=True):
        self.cfg, self.augment = cfg, augment
        self.items = []
        for jp in sorted(glob.glob(os.path.join(img_dir, "*.json"))):
            if int(os.path.splitext(os.path.basename(jp))[0]) > max_frame:
                continue  # 검증 구간은 제외
            data = json.load(open(jp))
            if data.get("flags", {}).get("reviewed"):  # 확인한 프레임만 사용
                self.items.append((os.path.join(img_dir, data["imagePath"]), data))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cfg = self.cfg
        img_path, data = self.items[i]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 증강: 원본과 같은 범위의 확대/축소(0.8~1.2), 회전(±6도), 이동(가로±200, 세로±100 / CULane 기준 비례)
        if self.augment:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-6, 6), np.random.uniform(0.8, 1.2))
            m[0, 2] += np.random.uniform(-200, 200) * w / 1640
            m[1, 2] += np.random.uniform(-100, 100) * h / 590
        else:
            m = np.float32([[1, 0, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, m, (w, h), borderValue=0)

        row_locs = cfg.row_anchor * h
        col_locs = cfg.col_anchor * w
        rows = np.full((len(row_locs), 4), -1.0, dtype=np.float32)
        cols = np.full((len(col_locs), 4), -1.0, dtype=np.float32)
        for s in data["shapes"]:
            if s["label"] not in SLOTS or len(s["points"]) < 2:
                continue
            j = SLOTS.index(s["label"])
            pts = np.asarray(s["points"], dtype=np.float32)
            pts = pts @ m[:, :2].T + m[:, 2]          # 이미지와 같은 변환을 점에도 적용
            if pts[0, 1] > pts[-1, 1]:
                pts = pts[::-1]                        # 먼 쪽→가까운 쪽 순서로 통일
            rows[:, j] = extend_to_bottom(interp_lane(pts, row_locs, 0))
            cols[:, j] = interp_lane(pts, col_locs, 1)

        # 원본과 같은 방식으로 칸 번호(정수 라벨)와 비율(실수 라벨) 생성, 화면 밖은 -1
        rows_f = np.where((rows >= 0) & (rows <= w), rows / w, -1).astype(np.float32)
        cols_f = np.where((cols >= 0) & (cols <= h), cols / h, -1).astype(np.float32)
        rows_i = np.where(rows_f >= 0, (rows_f * (cfg.num_cell_row - 1)).astype(np.int64), -1)
        cols_i = np.where(cols_f >= 0, (cols_f * (cfg.num_cell_col - 1)).astype(np.int64), -1)

        # 이미지 전처리: 리사이즈 → 아래쪽 train_height만큼 자르기 → 정규화 (추론과 동일)
        img = cv2.resize(img, (cfg.train_width, int(cfg.train_height / cfg.crop_ratio)))
        img = img[-cfg.train_height:].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return {"images": torch.from_numpy(img.transpose(2, 0, 1).copy()),
                "labels_row": torch.from_numpy(rows_i), "labels_col": torch.from_numpy(cols_i),
                "labels_row_float": torch.from_numpy(rows_f), "labels_col_float": torch.from_numpy(cols_f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"), ap.add_argument("weights"), ap.add_argument("img_dir"), ap.add_argument("out_dir")
    ap.add_argument("--max_frame", type=int, default=420, help="이 번호 이하 프레임만 학습에 사용")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)  # CULane 설정과 동일
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    out_dir = os.path.expanduser(args.out_dir)

    ds = LabelmeLaneDataset(os.path.expanduser(args.img_dir), cfg, args.max_frame)
    print(f"학습 프레임 수 (reviewed, {args.max_frame}번 이하): {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    net = get_model(cfg).to(args.device)
    sd = torch.load(args.weights, map_location="cpu")["model"]
    print(net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False))

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    warmup = min(100, total // 10)
    # 학습률: 처음 warmup 동안 선형 증가 후 코사인 감소
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    loss_dict = get_loss_dict(cfg)
    writer = SummaryWriter(out_dir)

    step = 0
    for ep in range(args.epochs):
        net.train()
        ep_loss = 0.0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = calc_loss(loss_dict, inference(net, batch, "CULane"), writer, step, ep)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item()
            step += 1
        ep_loss /= max(1, len(loader))
        writer.add_scalar("loss/epoch", ep_loss, ep)
        print(f"epoch {ep + 1}/{args.epochs}  loss {ep_loss:.4f}  lr {sched.get_last_lr()[0]:.5f}")
        torch.save({"model": net.state_dict()}, os.path.join(out_dir, "last.pth"))
        if (ep + 1) % 10 == 0:
            torch.save({"model": net.state_dict()}, os.path.join(out_dir, f"ep_{ep + 1:03d}.pth"))
    writer.close()
    print(f"완료: {out_dir}/last.pth")


if __name__ == "__main__":
    main()
