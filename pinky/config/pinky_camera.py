"""pinky_camera.yaml을 읽어 카메라 행렬, 왜곡, 바닥 변환 행렬을 만든다.

바닥 좌표: x 오른쪽, z 앞쪽(m), 카메라 바로 아래 바닥이 원점 (lane_geom.py의 BEV와 같은 방향).
  python pinky_camera.py pinky_camera.yaml pinky_camera.npz                    # 실제 m 그대로
  python pinky_camera.py pinky_camera.yaml pinky_scaled.npz --lane-width 0.206  # 후처리용: 실제 도로 축척으로 확대
--lane-width를 주면 바닥 좌표에 (3.3 m ÷ 차로 폭)을 곱한다. 후처리 수치(차로 폭 3.3 m, 점선 주기 8 m 등)가
실제 도로 기준이므로, 핑키 도로를 같은 축척으로 키워 넘기면 후처리 코드를 고치지 않고 쓸 수 있다.
"""
import argparse
import numpy as np
import yaml


def load(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    i, e = c["intrinsic"], c["extrinsic"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
    return c, K, np.array(i["dist"], float), e["height_m"], e["tilt_down_deg"], e["roll_deg"]


def rotation(tilt_deg, roll_deg):
    """세계(x 오른쪽, y 아래, z 앞) → 카메라 회전. 아래로 숙임 후 광축 둘레로 좌우 기울어짐.
    roll +: 화면 오른쪽이 위로 들림(카메라의 오른쪽 축이 위를 향함). calib_pose_live.py가 재는 값, scene.py가 그리는 값과 같은 정의.
    (예전에는 부호가 반대여서, 바닥으로 옮긴 위치가 가까운 곳에서 약 8 mm 옆으로 치우쳤다)"""
    t, r = np.radians(tilt_deg), -np.radians(roll_deg)
    pitch = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    roll = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return roll @ pitch


def homography(K, height, tilt_deg, roll_deg):
    """왜곡을 편 화면 픽셀 → 바닥 (x, z)[m] 변환 행렬 H와 소실점."""
    R = rotation(tilt_deg, roll_deg)
    G = K @ np.column_stack([R[:, 0], R[:, 2], height * R[:, 1]])   # 바닥 (x, z, 1) → 화면
    vp = K @ R[:, 2]
    return np.linalg.inv(G), vp[:2] / vp[2]


REF_LANE_W = 3.3     # 후처리가 기준으로 삼는 실제 도로 차로 폭(m)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--lane-width", type=float, default=0.0, help="도로의 차로 폭(m). 주면 실제 도로 축척으로 확대")
    a = ap.parse_args()
    c, K, dist, h, tilt, roll = load(a.yaml)
    H, vp = homography(K, h, tilt, roll)
    scale = REF_LANE_W / a.lane_width if a.lane_width > 0 else 1.0
    H = np.diag([scale, scale, 1.0]) @ H                              # 바닥 좌표만 확대(화면 쪽은 그대로)
    G = np.linalg.inv(H)
    for x, z in ((0.0, 0.15 * scale), (-0.1 * scale, 0.3 * scale), (0.1 * scale, 0.5 * scale)):   # 확인: 바닥 → 화면 → 바닥
        p = G @ [x, z, 1]; q = H @ [p[0] / p[2], p[1] / p[2], 1]
        print(f"바닥 ({x:+.2f}, {z:.2f}) m → 화면 ({p[0] / p[2]:.0f}, {p[1] / p[2]:.0f}) → ({q[0] / q[2]:+.3f}, {q[1] / q[2]:.3f})")
    print(f"소실점(수평선) 화면 위치: ({vp[0]:.0f}, {vp[1]:.0f})")
    if a.out:
        np.savez(a.out, K=K, dist=dist, H=H, cam_h=h * scale, vp=vp, scale=scale)
        print(f"저장: {a.out}  (축척 {scale:.2f}배)")
