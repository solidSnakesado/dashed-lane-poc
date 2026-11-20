"""pinky_camera.yaml을 읽어 카메라 행렬, 왜곡, 바닥 변환 행렬을 만든다.

바닥 좌표: x 오른쪽, z 앞쪽(m), 카메라 바로 아래 바닥이 원점 (lane_geom.py의 BEV와 같은 방향).
  python pinky_camera.py pinky_camera.yaml pinky_camera.npz   # 후처리용 npz(K, dist, H, cam_h, vp) 저장
"""
import sys

import numpy as np
import yaml


def load(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    i, e = c["intrinsic"], c["extrinsic"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
    return c, K, np.array(i["dist"], float), e["height_m"], e["tilt_down_deg"], e["roll_deg"]


def rotation(tilt_deg, roll_deg):
    """세계(x 오른쪽, y 아래, z 앞) → 카메라 회전. 아래로 숙임 후 광축 둘레로 좌우 기울어짐."""
    t, r = np.radians(tilt_deg), np.radians(roll_deg)
    pitch = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    roll = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return roll @ pitch


def homography(K, height, tilt_deg, roll_deg):
    """왜곡을 편 화면 픽셀 → 바닥 (x, z)[m] 변환 행렬 H와 소실점."""
    R = rotation(tilt_deg, roll_deg)
    G = K @ np.column_stack([R[:, 0], R[:, 2], height * R[:, 1]])   # 바닥 (x, z, 1) → 화면
    vp = K @ R[:, 2]
    return np.linalg.inv(G), vp[:2] / vp[2]


if __name__ == "__main__":
    c, K, dist, h, tilt, roll = load(sys.argv[1])
    H, vp = homography(K, h, tilt, roll)
    G = np.linalg.inv(H)
    for x, z in ((0.0, 0.15), (-0.1, 0.3), (0.1, 0.5)):             # 확인: 바닥 점 → 화면 → 바닥
        p = G @ [x, z, 1]; q = H @ [p[0] / p[2], p[1] / p[2], 1]
        print(f"바닥 ({x:+.2f}, {z:.2f}) m → 화면 ({p[0] / p[2]:.0f}, {p[1] / p[2]:.0f}) → ({q[0] / q[2]:+.3f}, {q[1] / q[2]:.3f})")
    print(f"소실점(수평선) 화면 위치: ({vp[0]:.0f}, {vp[1]:.0f})")
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], K=K, dist=dist, H=H, cam_h=h, vp=vp)
        print(f"저장: {sys.argv[2]}")
