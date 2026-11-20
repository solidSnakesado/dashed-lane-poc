"""시뮬레이션 주행 영상 생성 (6-1). 핑키가 차로 중심선을 따라 달리는 카메라 영상과 장마다의 정답을 만든다.

  MUJOCO_GL=egl python sim_drive.py ROAD_DIR CAMERA_YAML OUT_DIR [--path 0] [--speed 0.2] [--fps 15]
결과 (OUT_DIR):
  drive.mp4  640×480 카메라 영상
  gt.csv     장마다 핑키 위치·방향, 차로 중심에서 벗어난 거리, 정답 선 3개의 옆 위치
             (카메라 바로 아래 바닥 기준, x 오른쪽, z 앞, 실제 m. 앞 0.1~1.0m 지점마다. 화면 밖이면 비움)
  meta.json  도로, 경로, 속도, 차로 폭
  check.jpg  첫 장면 위에 정답 선(왼쪽 빨강, 오른쪽 파랑, 중심 초록)을 겹친 확인용 그림
핑키 움직임: 중심선을 따라가며 좌우로 천천히 흔들림(±wobble), 카메라 기울기는 프레임마다 아주 조금 흔들림.
"""
import argparse
import csv
import gzip
import json
import os

import cv2
import numpy as np

from gen_dataset import background, to_yolo
from scene import Scene

ZS = np.round(np.arange(0.1, 1.01, 0.1), 2)     # 정답을 기록할 앞 거리(m)
END_MARGIN = 1.0                                # 경로 끝 이 거리(m) 전에 멈춤(앞에 도로가 없어지므로)


def resample(p, step):
    """폴리라인을 일정 간격으로 다시 나눈다. (점들, 누적 거리)"""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.arange(0, s[-1], step)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], t


def lateral_at(line, origin, yaw):
    """세계 좌표의 선을 카메라 바닥 기준 좌표로 옮겨, 앞 거리 ZS마다 옆 위치 x를 구한다(없으면 nan)."""
    d = line - origin
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    z, x = d @ fwd, d @ right
    ok = (z > 0) & (z < 1.6) & (np.abs(x) < 1.0)
    if ok.sum() < 2:
        return np.full(len(ZS), np.nan)
    z, x = z[ok], x[ok]
    o = np.argsort(z)
    z, x = z[o], x[o]
    out = np.interp(ZS, z, x, left=np.nan, right=np.nan)
    return out


def offset_line(p, d):
    """진행 방향 왼쪽으로 d만큼 평행 이동한 선."""
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return p + np.c_[-t[:, 1], t[:, 0]] * np.reshape(d, (-1, 1))


def look(img, rng_tint, noise, bg):
    """바닥 밖은 고정 배경으로 채우고, 영상 전체에 고정된 색조와 약한 노이즈(실물 카메라 흉내)."""
    out_floor = cv2.dilate((img.max(axis=2) == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img = img.copy()
    img[out_floor] = bg[out_floor]
    out = np.clip(img.astype(np.float32) * rng_tint + np.random.normal(0, noise, img.shape), 0, 255)
    return out.astype(np.uint8)


def draw_gt(img, sc, pose, lines):
    """정답 선을 실제 카메라 화면에 투영해 그린다."""
    p, right, up, f = pose
    R = np.stack([right, -up, f])
    rv, _ = cv2.Rodrigues(R)
    tv = -R @ p
    for line, col in zip(lines, [(0, 0, 255), (255, 0, 0), (0, 200, 0)]):
        P = np.c_[line, np.zeros(len(line))]
        cam = (R @ P.T).T + tv
        keep = (cam[:, 2] > 0.05) & (cam[:, 2] < 1.6) & (np.abs(cam[:, 0] / np.maximum(cam[:, 2], 1e-6)) < 0.65) \
            & (np.abs(cam[:, 1] / np.maximum(cam[:, 2], 1e-6)) < 0.5)            # 화각 밖 점은 왜곡식이 발산하므로 제외
        if keep.sum() < 2:
            continue
        uv, _ = cv2.projectPoints(P[keep], rv, tv, sc.K, sc.dist)
        cv2.polylines(img, [uv.reshape(-1, 2).astype(np.int32)], False, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("road")
    ap.add_argument("cam")
    ap.add_argument("out")
    ap.add_argument("--path", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.2, help="m/s")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--wobble", type=float, default=0.03, help="좌우 흔들림 진폭(m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-dets", action="store_true", help="정답 라벨을 검출 결과 형식(drive_gtdets.json.gz)으로도 저장")
    a = ap.parse_args()
    rng, dets = np.random.default_rng(a.seed), []
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    sc = Scene(a.road, a.cam)
    lw, names = sc.info["lane_w"] / 1000.0, sc.info["classes"]
    path = np.asarray(sc.info["paths"][a.path]) / 1000.0
    c, s = resample(path, a.speed / a.fps)
    keep = s < s[-1] - END_MARGIN
    c, s = c[keep], s[keep]
    t = s / a.speed
    ph = rng.uniform(0, 2 * np.pi, 2)                                  # 좌우 흔들림: 두 주기의 합
    off = a.wobble * (0.7 * np.sin(2 * np.pi * t / 3.2 + ph[0]) + 0.3 * np.sin(2 * np.pi * t / 1.9 + ph[1]))
    pos = offset_line(c, -off)                                         # +off = 오른쪽으로 벗어남
    tan = np.gradient(c, axis=0)
    yaw = np.arctan2(tan[:, 1], tan[:, 0]) - np.arctan2(np.gradient(off), np.maximum(np.gradient(s), 1e-6))
    left, rightl = offset_line(path, lw / 2), offset_line(path, -lw / 2)   # 정답: 내 차로 양쪽 선과 중심선
    tint = np.array([rng.uniform(0.95, 1.1), rng.uniform(1.0, 1.15), rng.uniform(0.8, 0.95)])
    bg = background(sc.H, sc.W, rng)                                   # 보드 너머 배경(영상 내내 같음)
    vw = cv2.VideoWriter(os.path.join(a.out, "drive.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (sc.W, sc.H))
    f = open(os.path.join(a.out, "gt.csv"), "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["frame", "t", "x", "y", "yaw", "offset"] + [f"{n}_{z:.1f}" for n in ("left", "right", "center") for z in ZS])
    for i in range(len(pos)):
        dtilt = float(np.clip(rng.normal(0, 0.1), -0.2, 0.2))
        img, inst = sc.render(pos[i, 0], pos[i, 1], yaw[i], dtilt=dtilt)
        if a.gt_dets:                                                  # 모델 없이 후처리만 시험할 때 쓰는 '완벽한 검출'
            polys = [ln.split() for ln in to_yolo(inst, sc.info["instances"], names)]
            dets.append({"k": i, "polys": [[int(p[0]), 1.0] + [round(float(v) * (sc.W, sc.H)[j % 2], 1)
                                                                 for j, v in enumerate(p[1:])] for p in polys]})
        img = look(img, tint, 3.0, bg)
        cam_xy = sc.pose(pos[i, 0], pos[i, 1], yaw[i])[0][:2]          # 카메라 바로 아래 바닥 = 정답 좌표 원점
        gl, gr, gc = (lateral_at(L, cam_xy, yaw[i]) for L in (left, rightl, path))
        wr.writerow([i, f"{t[i]:.3f}", f"{pos[i, 0]:.4f}", f"{pos[i, 1]:.4f}", f"{yaw[i]:.5f}", f"{off[i]:.4f}"]
                    + ["" if np.isnan(v) else f"{v:.4f}" for v in np.r_[gl, gr, gc]])
        vw.write(img)
        if i == 0:
            cv2.imwrite(os.path.join(a.out, "check.jpg"),
                        draw_gt(img.copy(), sc, sc.pose(pos[i, 0], pos[i, 1], yaw[i], dtilt=dtilt), (left, rightl, path)))
    vw.release()
    if a.gt_dets:
        with gzip.open(os.path.join(a.out, "drive_gtdets.json.gz"), "wt", encoding="utf-8") as g:
            json.dump({"video": "drive.mp4", "fps": a.fps, "stride": 1, "size": [sc.W, sc.H],
                       "names": {str(k): n for k, n in enumerate(names)}, "frames": dets}, g)
    f.close()
    sc.close()
    json.dump({"road": os.path.abspath(a.road), "path": a.path, "speed": a.speed, "fps": a.fps, "frames": len(pos),
               "lane_w_m": lw, "wobble_m": a.wobble, "road_info": {k: sc.info[k] for k in ("type", "lanes_per_dir")}},
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"저장: {a.out}  {len(pos)}장 ({len(pos) / a.fps:.1f}초), 차로 폭 {lw * 1000:.0f}mm")


if __name__ == "__main__":
    main()
