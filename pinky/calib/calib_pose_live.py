"""핑키 카메라 장착 자세 측정 (높이·기울기·좌우 기울어짐). 노트북에서 실시간 미리보기로 진행한다.

핑키에서 영상 송출 (calib_live.py와 같음):
  rpicam-vid -n --rotation 180 --mode 640:480:10:P --width 640 --height 480 --framerate 15 \
             --codec h264 --inline --intra 5 --bitrate 6000000 -t 0 --listen -o tcp://0.0.0.0:5000
노트북:
  python calib_pose_live.py tcp://192.168.14.91:5000 ~/calib_live/camera_intrinsics.yaml ~/calib_live
진행: 체커보드를 바닥에 눕히고(긴 쪽이 가로), 핑키를 바닥에 내려놓고 손을 뗀다.
  1.5초 동안 계속 보이면 자동 측정. 화면 안내에 따라 핑키를 들었다 내려놓기를 5번 반복한다.
결과: ~/calib_live/camera_extrinsics.yaml (5번의 평균과 흔들림 폭)
높이는 체커보드 윗면 기준이다. 폼보드 도로 위를 달리므로 체커보드를 폼보드 위에 놓고 재면 그대로 쓸 수 있다.
"""
import argparse
import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0")
import cv2
import numpy as np

COLS, ROWS, SQUARE = 8, 6, 30.0   # 체커보드 안쪽 모서리 수, 칸 크기(mm)
TRIALS = 5                        # 들었다 내려놓기 반복 횟수
HOLD_SEC = 1.5                    # 이 시간 동안 계속 보이면 측정
MOVE_MM = 8.0                     # 측정 후 카메라 위치가 이만큼 바뀌거나 체커보드가 사라지면 다음 측정 준비


class Latest:
    """영상을 별도 스레드에서 읽어 가장 최근 프레임만 들고 있는다."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame, self.n, self.ok = None, 0, self.cap.isOpened()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            self.frame, self.n = f, self.n + 1


def pose(gray, K, dist):
    """체커보드를 찾아 카메라 자세를 구한다. (높이 mm, 아래로 기울기 도, 좌우 기울어짐 도, 카메라 위치) 또는 None."""
    for size in ((COLS, ROWS), (ROWS, COLS)):
        ok, pts = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            break
    else:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
    obj = np.zeros((size[0] * size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2) * SQUARE
    ok, rv, tv = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rv)
    cam = (-R.T @ tv).ravel()                       # 체커보드 좌표계에서 카메라 위치(mm), 판 위가 z = 0
    up = np.array([0, 0, np.sign(cam[2]) or 1.0])   # 판에서 카메라 쪽을 향하는 방향 = 위
    axis = R.T @ np.array([0, 0, 1.0])               # 카메라가 바라보는 방향
    right = R.T @ np.array([1.0, 0, 0])              # 카메라 화면의 오른쪽 방향
    tilt = np.degrees(np.arcsin(np.clip(-axis @ up, -1, 1)))     # 수평에서 아래로 숙인 각
    roll = np.degrees(np.arcsin(np.clip(right @ up, -1, 1)))     # +: 화면 오른쪽이 위로 들림
    return abs(cam[2]), tilt, roll, cam, pts, size


class Measurer:
    """미리보기를 막지 않도록 측정 계산을 별도 스레드에서 한다."""

    def __init__(self, cam, K, dist):
        self.cam, self.K, self.dist, self.result = cam, K, dist, None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        last = -1
        while self.cam.ok:
            if self.cam.frame is None or self.cam.n == last:
                time.sleep(0.01)
                continue
            last = self.cam.n
            self.result = pose(cv2.cvtColor(self.cam.frame, cv2.COLOR_BGR2GRAY), self.K, self.dist)


FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
_font = None


def text(img, lines, y0, col):
    """안내 문구를 한글로 그린다(OpenCV는 한글을 못 그려서 Pillow 사용). 글꼴이 없으면 그대로 출력만."""
    global _font
    from PIL import Image, ImageDraw, ImageFont
    if _font is None:
        path = next((p for p in FONTS if os.path.exists(p)), None)
        _font = ImageFont.truetype(path, 18) if path else ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for i, s in enumerate(lines):
        d.text((8, y0 + i * 24), s, font=_font, fill=(col[2], col[1], col[0]), stroke_width=3, stroke_fill=(0, 0, 0))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("intrinsics")
    ap.add_argument("out")
    a = ap.parse_args()
    fs = cv2.FileStorage(os.path.expanduser(a.intrinsics), cv2.FILE_STORAGE_READ)
    K, dist = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
    fs.release()
    out = os.path.expanduser(a.out)
    cam = Latest(a.src)
    meas = Measurer(cam, K, dist)
    trials, window, seen_since, armed, ref = [], [], None, True, None
    print("연결 중... (핑키에서 rpicam-vid가 실행 중이어야 함)")
    while cam.ok or cam.frame is None:
        if cam.frame is None:
            if not cam.ok:
                print("영상을 받지 못했습니다. 핑키의 rpicam-vid와 주소를 확인해 주세요.")
                return
            time.sleep(0.05)
            continue
        vis, r, now = cam.frame.copy(), meas.result, time.time()
        h, w = vis.shape[:2]
        if r is not None:
            cv2.drawChessboardCorners(vis, r[5], r[4], True)
        if not armed:                                            # 측정 후: 들었다 내려놓기를 기다림
            if r is None or np.linalg.norm(r[3] - ref) > MOVE_MM:
                armed, seen_since, window = True, None, []
            guide = [f"측정 완료 {len(trials)}/{TRIALS}", "핑키를 들었다가 같은 자리에 내려놓으세요.", "내려놓은 뒤 손을 떼세요."]
            col = (0, 220, 0)
        elif r is None:
            seen_since, window = None, []
            guide = [f"{len(trials) + 1}/{TRIALS}번째: 체커보드를 찾지 못함",
                     "체커보드를 바닥에 눕히고 긴 쪽을 가로로 두세요.", "앞쪽 끝이 화면 아래쪽 끝 가까이 오게 핑키를 가까이 두세요.",
                     "체커보드 전체가 화면 안에 들어와야 합니다.", "핑키에서 손을 떼세요."]
            col = (0, 0, 255)
        else:
            seen_since = seen_since or now
            window.append(r[:3])
            left = HOLD_SEC - (now - seen_since)
            guide = [f"{len(trials) + 1}/{TRIALS}번째: 그대로 기다리세요... {max(left, 0):.1f}초",
                     f"높이 {r[0]:.1f} mm   아래로 {r[1]:.1f}도   좌우 {r[2]:.1f}도"]
            col = (0, 255, 255)
            if left <= 0:                                        # 1.5초 동안의 중앙값을 측정값으로
                m = np.median(np.array(window), axis=0)
                trials.append(m)
                ref, armed = r[3], False
                print(f"측정 {len(trials)}: 높이 {m[0]:.1f} mm, 아래로 {m[1]:.2f}도, 좌우 {m[2]:.2f}도")
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), col, 4)
        text(vis, guide, 10, col)
        text(vis, ["q: 종료"], h - 30, (255, 255, 255))
        cv2.imshow("pinky camera pose", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q") or len(trials) >= TRIALS:
            break
    cv2.destroyAllWindows()
    if not trials:
        print("측정값이 없습니다.")
        return
    t = np.array(trials)
    mean, spread = t.mean(axis=0), t.max(axis=0) - t.min(axis=0)
    print(f"평균: 높이 {mean[0]:.1f} mm, 아래로 {mean[1]:.2f}도, 좌우 {mean[2]:.2f}도 ({len(t)}번)")
    print(f"흔들림 폭(최대-최소): 높이 {spread[0]:.1f} mm, 기울기 {spread[1]:.2f}도, 좌우 {spread[2]:.2f}도")
    path = os.path.join(out, "camera_extrinsics.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for k, v in (("height_mm", mean[0]), ("tilt_down_deg", mean[1]), ("roll_deg", mean[2]),
                 ("height_spread_mm", spread[0]), ("tilt_spread_deg", spread[1]), ("roll_spread_deg", spread[2])):
        fs.write(k, float(v))
    fs.write("trials", t)
    fs.release()
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
