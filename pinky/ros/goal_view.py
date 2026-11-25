"""출발지 → 목적지 주행의 화면: 핑키 카메라 화면과 주행 라인, 탑뷰의 로봇·주행 라인, 영상 기록. goal_click.py가 쓴다(ROS와 무관).

카메라 화면 = 로봇의 실제 위치에서 핑키 카메라(보정값 pinky_camera.yaml)가 보는 road 바닥을 그린 것(시뮬 브리지가 보낸 영상이 아님).
주행 라인(하늘색) = 지도와 추정 위치로 계산한 지금 차로의 왼쪽·오른쪽 선(lane_where). 교차로 안도 길을 따라 이어서 그린다.
탑뷰에는 같은 주행 라인과 로봇(실제 크기의 네모, 앞쪽에 짧은 선)을 그린다.
"""
import os, subprocess, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT + "/pinky/sim", ROOT]

SKY = (235, 206, 135)                       # 주행 라인 색: 하늘색(BGR)
BODY = (-0.125, 0.072, 0.058)               # 로봇 몸: 바퀴 축에서 뒤·앞 끝, 반폭(m). pinky_mujoco 물리 모델의 충돌 형상에서 잰 값
NEAR_MM = 8.0                               # 경로에서 이 거리(mm) 안의 그래프 점을 그 길의 점으로 본다
FPS = 10.0                                  # 영상과 화면 갱신(시뮬 시간 기준)


class DriveView:
    def __init__(self, road, calib_dir):
        from ego_label import Routes, STEP
        from route_graph import Graph
        from lane_where import Where
        from lane_geom import Camera
        import sim_closed as S
        self.S = S; cam_yaml, calib = ROOT + "/pinky/config/pinky_camera.yaml", calib_dir + "/calib.npz"
        subprocess.run([sys.executable, ROOT + "/pinky/config/pinky_camera.py", cam_yaml, calib], check=True, capture_output=True)
        self.sc = S.Scene(road, cam_yaml); rt = Routes(self.sc, S.LATERAL_MM, S.YAW_DEG); self.g = Graph(rt, STEP, road)
        self.cam = Camera(calib); self.w = Where(self.g, float(np.load(calib)["cam_h"]), S.CAM_FWD)
        rng = np.random.default_rng(0); np.random.seed(0)
        self.look = (np.array([1.0, 1.05, 0.9]), 1.0, S.background(self.sc.H, self.sc.W, rng))   # 색조, 흐림, 바닥 밖 배경
        self.vid = self.avi = None

    def set_route(self, path):
        """Route Server가 준 경로(m 점열)를 그래프 점 번호의 줄로 바꿔 둔다(주행 라인이 길을 따라가도록).
        여러 경로가 겹친 차로에서는 같은 자리에 점이 여럿이므로, 앞 점과 같은 경로의 점을 먼저 골라 줄이 끊기지 않게 한다."""
        g = self.g; s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        q = np.arange(0.0, s[-1], NEAR_MM / 1000.0); P = np.c_[np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])] * 1000.0
        ang = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])); ids, on = [], set()
        for p, a, cand in zip(P, ang, g.tree.query_ball_point(P, NEAR_MM)):
            cand = [k for k in cand if abs((g.yaw[k] - a + np.pi) % (2 * np.pi) - np.pi) < np.radians(30)]   # 같은 방향의 점만
            if not cand:
                continue
            on.update(cand)
            same = [k for k in cand if ids and g.pid[k, 0] == g.pid[ids[-1], 0] and k >= ids[-1]]
            k = min(same or cand, key=lambda k: np.linalg.norm(g.P[k] - p))
            if ids and k == ids[-1]:
                continue
            ids += list(range(ids[-1] + 1, k + 1)) if same else [k]              # 같은 경로면 사이 점을 채움
        self.w.set_route(np.array(ids)); self.w.on |= on                          # 겹친 다른 경로의 점도 '길 위'로 본다

    def lane(self, est):
        """추정 위치의 상태와, 앞쪽 주행 라인(세계 좌표 m): (locate 결과, 왼쪽 선, 오른쪽 선) 또는 선이 없으면 None."""
        out = self.w.locate(est)
        if out["state"] == "out":
            return out, None, None
        from lane_where import AHEAD
        k = self.w.nearest(np.array(est[:2]) * 1000.0, est[2]) if self.w.route is not None else out["k"]   # 길의 줄 위의 점에서 시작해야 길을 따라 이어진다
        ids = self.g.ahead(k, self.w.route if self.w.route is not None else np.array([k]), AHEAD * self.w.cam_h * 1000.0)
        if len(ids) < 2:
            return out, None, None
        _, _, left, right, _ = self.g.lane(ids)
        return out, left / 1000.0, right / 1000.0

    def camera(self, true, est, left, right):
        """핑키 카메라 화면에 주행 라인(하늘색)을 겹친 그림."""
        from draw_lane import visible
        import lane_check
        img, _ = self.sc.render(*true); img = self.S.look(img, *self.look)
        for line in (left, right):
            if line is None:
                continue
            c = lane_check.to_cam(line, est, self.S.CAM_FWD); ok = visible(self.cam, c, img.shape[1], img.shape[0]); idx = np.where(ok)[0]
            px = self.cam.to_image(c[ok]).astype(np.int32) if ok.any() else []
            for a in range(len(px) - 1):
                if idx[a + 1] == idx[a] + 1:                                   # 화면 안에서 이어진 점끼리만 잇는다
                    cv2.line(img, tuple(map(int, px[a])), tuple(map(int, px[a + 1])), SKY, 3, cv2.LINE_AA)
        return img

    def record(self, path, size):
        """영상 기록 시작(MJPG avi로 쓰고, 끝날 때 ffmpeg가 있으면 mp4로 바꾼다)."""
        self.avi = path + ".avi"; self.vid = cv2.VideoWriter(self.avi, cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)

    def write(self, frame):
        if self.vid is not None:
            self.vid.write(frame)

    def stop(self):
        """영상 기록을 끝내고 저장된 파일 이름을 돌려준다."""
        if self.vid is None:
            return None
        self.vid.release(); self.vid = None; mp4 = self.avi[:-4] + ".mp4"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.avi, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", mp4], check=True)
            os.remove(self.avi); return mp4
        except Exception:
            return self.avi                                                    # ffmpeg가 없거나 실패하면 avi 그대로


def robot(img, px, true):
    """탑뷰에 로봇을 실제 크기의 네모로 그린다. px = 세계 좌표(m) → 픽셀 함수."""
    x, y, yaw = true; f = np.array([np.cos(yaw), np.sin(yaw)]); l = np.array([-f[1], f[0]]); o = np.array([x, y])
    box = np.array([o + BODY[0] * f + BODY[2] * l, o + BODY[1] * f + BODY[2] * l, o + BODY[1] * f - BODY[2] * l, o + BODY[0] * f - BODY[2] * l])
    cv2.fillPoly(img, [px(box)], (255, 255, 255)); cv2.polylines(img, [px(box)], True, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, tuple(map(int, px(o)[0])), tuple(map(int, px(o + BODY[1] * f)[0])), (255, 0, 255), 2, cv2.LINE_AA)


def compose(top, cam, lines, height=720):
    """탑뷰(왼쪽)와 카메라 화면(오른쪽 위), 글자(오른쪽 아래)를 한 그림으로. 돌려주는 값: 그림, 탑뷰 축소 비율."""
    k = height / top.shape[0]; left = cv2.resize(top, (int(round(top.shape[1] * k)), height), interpolation=cv2.INTER_AREA)
    right = np.full((height, 640, 3), 30, np.uint8)
    if cam is not None:
        right[:480] = cam
    for i, t in enumerate(lines):
        cv2.putText(right, t, (12, 512 + 30 * i), 0, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([left, right]), k
