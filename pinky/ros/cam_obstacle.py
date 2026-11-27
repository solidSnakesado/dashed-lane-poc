"""앞 카메라 영상에서 장애물이 바닥에 닿는 선을 찾아 지도 좌표 점으로 바꾼다(2026-11-27, 비전 5단계). obstacle_detour.py의 Detour가 쓴다.

까닭: 라이다는 바닥 위 약 12.5 cm 한 면만 보므로 그보다 낮은 부분(핑키 바퀴 등)이 장애물 폭에 들어가지 않았다(real_c7_16: 바퀴 1개쯤 닿음).
방법(학습·수동 라벨 없음, 고정 크기 없음):
  1) 바닥 = 채도 FLOOR_S 미만이고 밝기 FLOOR_V 넘는 화소(흰 도로 바닥, 그림자 포함). 작은 잡티는 OPEN으로 지움.
  2) 열마다 영상 아래에서 위로 올라가며 처음 만나는 바닥이 아닌 화소 = 그 열에서 바닥에 닿는 선(장애물, 벽, 먼 배경).
  3) 내부 보정(ost.yaml)과 장착 자세(mount.yaml: 높이, 숙임, 옆 기울기, 앞 위치)로 그 화소를 바닥(로봇 좌표)으로 옮기고,
     영상 시각의 TF(map ← base_footprint)로 지도 좌표로 옮긴다. TF가 아직 안 왔으면 KEEP초까지 기다린다(라이다와 같음).
  4) 카메라에서 RANGE 안인 점만 쓴다(멀수록 바닥 위치 오차가 커짐). 벽·배경 점은 Detour.keep(지도 벽 WALL_GAP, 길 안, 빈 칸)이 거른다.
정확도(같은 날 실측): 바닥 표시 150~400 mm에서 거리 오차 0.3~3.4 mm. 장애물 핑키 폭 110 mm를 105 mm로 잼.
"""
import math, os
from collections import deque
import cv2, numpy as np, yaml
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage

FLOOR_S, FLOOR_V, RANGE, COL_STEP, KEEP = 70, 110, (0.09, 0.60), 2, 2.0   # 바닥 색 기준 / 쓰는 거리(m) / 열 간격(px) / TF 기다림(초)
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "cam1")


class CamPoints:
    def __init__(self, node, topic="/camera/image_raw/compressed", calib_dir=CALIB_DIR):
        self.n, self.q = node, deque(maxlen=10)
        c = yaml.safe_load(open(calib_dir + "/ost.yaml")); m = yaml.safe_load(open(calib_dir + "/mount.yaml"))
        self.K, self.D = np.array(c["camera_matrix"]["data"], float).reshape(3, 3), np.array(c["distortion_coefficients"]["data"], float)
        self.h, self.cx, self.cy = float(m["height_m"]), float(m["x_m"]), float(m["y_m"])
        p, r, y = (math.radians(float(m[k])) for k in ("pitch_down_deg", "roll_deg", "yaw_deg"))
        R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)          # 카메라(x 오른쪽, y 아래, z 앞) → 로봇(x 앞, y 왼쪽, z 위)
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])   # 숙임(앞이 아래로 +)
        Rx = np.array([[1, 0, 0], [0, math.cos(-r), -math.sin(-r)], [0, math.sin(-r), math.cos(-r)]])   # 옆 기울기(영상 오른쪽이 위로 +)
        self.R = Rz @ Ry @ Rx @ R0; self.rays = None
        node.create_subscription(CompressedImage, topic, lambda msg: self.q.append(msg), qos_profile_sensor_data)

    def ground(self, img):
        """영상 한 장 → 로봇 좌표 바닥 점 (N, 2). 열마다 바닥에 닿는 선 하나."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); H, W = hsv.shape[:2]
        floor = ((hsv[:, :, 1] < FLOOR_S) & (hsv[:, :, 2] > FLOOR_V)).astype(np.uint8)
        floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))[:, ::COL_STEP]
        up = floor[::-1] == 0; has = up.any(0); v = H - 1 - np.argmax(up, 0)   # 아래에서 처음 바닥이 아닌 행
        u = np.arange(0, W, COL_STEP)[has] + 0.5; v = v[has] + 0.5
        if not len(u):
            return np.zeros((0, 2))
        p = cv2.undistortPoints(np.c_[u, v][:, None, :].astype(np.float64), self.K, self.D).reshape(-1, 2)
        d = (self.R @ np.c_[p, np.ones(len(p))].T).T; ok = d[:, 2] < -1e-6
        s = self.h / -d[ok, 2]; P = np.c_[self.cx + s * d[ok, 0], self.cy + s * d[ok, 1]]
        rng = np.hypot(P[:, 0] - self.cx, P[:, 1] - self.cy)
        return P[(rng > RANGE[0]) & (rng < RANGE[1])]

    def points(self, buf):
        """아직 안 쓴 영상들 → 영상마다 지도 좌표 점 배열의 목록. TF는 영상 시각의 map ← base_footprint."""
        out, keep = [], deque(maxlen=10)
        stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        newest = max((stamp(m) for m in self.q), default=0.0)
        while self.q:
            m = self.q.popleft()
            try:
                t = buf.lookup_transform("map", "base_footprint", Time.from_msg(m.header.stamp)).transform
            except Exception:
                if newest - stamp(m) < KEEP:
                    keep.append(m)
                continue
            img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            P = self.ground(img); q = t.rotation; h = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            out.append(np.c_[t.translation.x + P[:, 0] * math.cos(h) - P[:, 1] * math.sin(h), t.translation.y + P[:, 0] * math.sin(h) + P[:, 1] * math.cos(h)])
        self.q.extend(keep)
        return out
