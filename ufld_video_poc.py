"""UFLDv2 모델로 주행 영상을 추론해 차선을 슬롯별 색으로 그린 결과 영상을 저장하는 스크립트.

사용법 (Ultra-Fast-Lane-Detection-v2 레포 최상위 폴더에서 실행, make_pseudo_labels.py가 같은 폴더에 있어야 함):
    python ufld_video_poc.py configs/culane_res18.py <가중치.pth> 입력.mp4 출력.mp4 [both]

마지막에 both를 붙이면 모든 슬롯을 행 기준·열 기준으로 모두 읽어 점으로 표시한다 (회전 구간 확인용).

슬롯 색: lane0 빨강, lane1 초록, lane2 노랑, lane3 파랑 (labelme 라벨 색과 같음)
"""
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# make_pseudo_labels.py의 디코딩 함수를 재사용 (DALI 대체 처리도 그 파일 안에서 이루어짐)
from make_pseudo_labels import decode
from utils.common import get_model
from utils.config import Config

COLORS = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 220, 220), 3: (255, 0, 0)}  # BGR


def main():
    cfg_path, weight_path, video_in, video_out = sys.argv[1:5]
    both = len(sys.argv) > 5 and sys.argv[5] == "both"

    cfg = Config.fromfile(cfg_path)
    cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)  # CULane 설정과 동일
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

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # demo.py와 같은 전처리: 리사이즈 후 아래쪽 train_height만큼 잘라냄
            x = tf(rgb)[:, -cfg.train_height:, :].unsqueeze(0).cuda()
            lanes = decode(net(x), cfg.row_anchor, cfg.col_anchor, w, h, both=both)
            for slot, pts in lanes.items():
                pts = [(int(a), int(b)) for a, b in pts]
                if both:  # 점 순서가 섞일 수 있으므로 점으로 표시
                    for p in pts:
                        cv2.circle(frame, p, 6, COLORS[slot], -1)
                else:
                    for p, q in zip(pts, pts[1:]):
                        cv2.line(frame, p, q, COLORS[slot], 4)
            # 왼쪽 위에 슬롯 색 범례 표시
            for slot, c in COLORS.items():
                cv2.putText(frame, f"lane{slot}", (20, 50 + 40 * slot), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
            out.write(frame)
            n += 1
            if n % 300 == 0:
                print(f"{n} frames")

    cap.release()
    out.release()
    print(f"done: {n} frames -> {video_out}")


if __name__ == "__main__":
    main()
