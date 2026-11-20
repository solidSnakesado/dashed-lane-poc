"""핑키 카메라 보정용 실시간 미리보기 + 수동 촬영 (노트북에서 실행).

핑키에서 먼저 영상을 내보낸다:
  rpicam-vid -n --rotation 180 --mode 640:480:10:P --width 640 --height 480 --framerate 15 \
             --codec h264 --inline --intra 5 --bitrate 6000000 -t 0 --listen -o tcp://0.0.0.0:5000
노트북:
  python calib_live.py tcp://192.168.14.91:5000 ~/calib_live
사용법: 체커보드가 1.5초 동안 계속 보이면 자동 저장한다(스페이스바로 바로 저장도 가능).
  이미 찍은 자세와 비슷하면 'same pose - move'를 띄우고 저장하지 않는다.
  저장 직후 체커보드 검출을 해서, 쓸 수 있으면 초록 'OK', 못 쓰면 빨강 'NG'를 크게 띄우고 NG 사진은 지운다.
  테두리 색(초록 = 체커보드 보임, 빨강 = 안 보임)은 참고용이며, 미리보기가 밀리지 않게 따로 계산한다.
키: 스페이스 저장, d 마지막 사진 삭제, q 종료
"""
import argparse
import glob
import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0")
import cv2
import numpy as np

COLS, ROWS = 8, 6        # 체커보드 안쪽 모서리 수
GOAL = 30                # 목표 장수
IDEAL = {"x": 0.7, "y": 0.7, "size": 0.4, "skew": 0.5}   # 지표별로 이 폭만큼 퍼지면 충분
SHOW_SEC = 1.5           # 저장 결과(OK/NG)를 화면에 띄우는 시간
AUTO_SEC = 1.5           # 체커보드가 이 시간 동안 계속 보이면 자동 저장
GAP_SEC = 2.5            # 저장 후 다음 자동 저장까지 최소 간격
NOVEL = 0.15             # 저장된 자세들과의 차이가 이보다 작으면 같은 자세로 보고 저장 안 함


class Latest:
    """영상을 별도 스레드에서 계속 읽어 가장 최근 프레임만 들고 있는다(밀림 방지)."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if "://" in src else cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.n, self.ok, self.file = None, 0, self.cap.isOpened(), "://" not in src
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            self.frame, self.n = f, self.n + 1
            if self.file:
                time.sleep(1 / 15)                     # 시험용 영상 파일은 실제 속도로 재생


class Watcher:
    """미리보기용 검출: 최근 프레임을 절반 크기로 줄여 빠르게 검사한다. 화면 표시를 막지 않는다."""

    def __init__(self, cam):
        self.cam, self.seen = cam, False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.cam.ok:
            f = self.cam.frame
            if f is None:
                time.sleep(0.05)
                continue
            small = cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            self.seen = any(cv2.findChessboardCorners(small, s, flags)[0] for s in ((COLS, ROWS), (ROWS, COLS)))


def detect_full(img):
    """저장한 사진에서 정밀 검출. (모서리, 패턴 크기) 또는 (None, None)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for size in ((COLS, ROWS), (ROWS, COLS)):
        ok, pts = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            return pts, size
    return None, None


def params(pts, size, w, h):
    """자세 지표: 화면 속 위치 x·y, 크기, 비스듬함(0~1). ROS camera_calibration과 같은 방식."""
    p = pts.reshape(-1, 2)
    c = size[0]
    q = np.array([p[0], p[c - 1], p[-1], p[-c]])                   # 바깥 네 모서리
    d1, d2 = q[2] - q[0], q[1] - q[3]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])               # 대각선 외적으로 넓이
    a, b = q[0] - q[1], q[2] - q[1]
    ang = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return {"x": float(np.mean(p[:, 0]) / w), "y": float(np.mean(p[:, 1]) / h),
            "size": float(np.sqrt(area / (w * h))), "skew": float(min(1.0, 2 * abs(np.pi / 2 - ang)))}


def progress(saved):
    """지표마다 얼마나 고르게 찍혔는지(0~1)."""
    if len(saved) < 2:
        return {k: 0.0 for k in IDEAL}
    return {k: min(1.0, (max(s[k] for s in saved) - min(s[k] for s in saved)) / v) for k, v in IDEAL.items()}


def overlay(vis, seen, prog, n, banner):
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 200, 0) if seen else (0, 0, 255), 4)
    cv2.rectangle(vis, (0, 0), (175, 100), (0, 0, 0), -1)
    for i, (k, v) in enumerate(prog.items()):
        y = 8 + i * 18
        cv2.putText(vis, k, (6, y + 11), 0, 0.45, (255, 255, 255), 1)
        cv2.rectangle(vis, (55, y), (165, y + 12), (80, 80, 80), -1)
        cv2.rectangle(vis, (55, y), (55 + int(110 * v), y + 12), (0, 200, 0) if v >= 1 else (0, 200, 255), -1)
    cv2.putText(vis, f"saved {n}/{GOAL}", (6, 94), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, "SPACE: save   d: delete last   q: quit", (8, h - 10), 0, 0.5, (255, 255, 255), 1)
    if banner is not None:                                        # 저장 결과를 크게 표시
        text, col = banner
        cv2.rectangle(vis, (0, h // 2 - 45), (w, h // 2 + 25), (0, 0, 0), -1)
        cv2.putText(vis, text, (20, h // 2), 0, 1.3, col, 3)


def save(frame, out, saved, files):
    """사진을 저장하고 바로 검출한다. 쓸 수 없으면 지운다. 화면에 띄울 글자와 색을 돌려준다."""
    pts, size = detect_full(frame)
    if pts is None:
        print("NG: 체커보드 모서리를 다 찾지 못해 저장하지 않음 (잘림, 흐림, 너무 비스듬함)")
        return ("NG - not saved", (0, 0, 255))
    name = os.path.join(out, f"img_{len(glob.glob(os.path.join(out, 'img_*.png'))):03d}.png")
    cv2.imwrite(name, frame)
    h, w = frame.shape[:2]
    pr = params(pts, size, w, h)
    if min((sum(abs(pr[k] - q[k]) for k in IDEAL) for q in saved), default=9.0) < NOVEL:
        os.remove(name)
        return ("same pose - move", (255, 200, 0))
    saved.append(pr); files.append(name)
    print(f"OK {len(saved)}장: {os.path.basename(name)}  x {pr['x']:.2f} y {pr['y']:.2f} "
          f"크기 {pr['size']:.2f} 비스듬 {pr['skew']:.2f}")
    return (f"OK  #{len(saved)}", (0, 220, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--nogui", action="store_true", help="시험용: 화면 없이 30프레임마다 저장")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    cam = Latest(a.src)
    watch = Watcher(cam)
    saved, files, banner, banner_t, last_n = [], [], None, 0.0, -1
    seen_since, last_save = None, 0.0
    print("연결 중... (핑키에서 rpicam-vid가 실행 중이어야 함)")
    while True:
        if cam.frame is None or cam.n == last_n:                   # 새 프레임이 올 때까지 대기
            if not cam.ok and cam.frame is not None:
                break
            if not cam.ok and cam.frame is None:
                print("영상을 받지 못했습니다. 핑키의 rpicam-vid와 주소를 확인해 주세요.")
                return
            time.sleep(0.01)
            continue
        f, last_n = cam.frame, cam.n
        if a.nogui:
            if last_n % 30 == 15:
                save(f, out, saved, files)
            continue
        now = time.time()
        seen_since = (seen_since or now) if watch.seen else None
        countdown = None
        if seen_since is not None and now - last_save > GAP_SEC:     # 손 없이 자동 저장
            left = AUTO_SEC - (now - seen_since)
            if left <= 0:
                banner, banner_t, last_save, seen_since = save(f, out, saved, files), now, now, None
            else:
                countdown = (f"capture in {left:.1f}s", (0, 255, 255))
        vis = f.copy()
        if countdown is not None and now - banner_t >= SHOW_SEC:
            banner, banner_t = countdown, now - SHOW_SEC + 0.05
        overlay(vis, watch.seen, progress(saved), len(saved), banner if time.time() - banner_t < SHOW_SEC else None)
        cv2.imshow("pinky calib", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            banner, banner_t = save(f, out, saved, files), time.time()
        if key == ord("d") and files:
            os.remove(files.pop()); saved.pop()
            banner, banner_t = ("deleted last", (0, 200, 255)), time.time()
            print("마지막 사진 삭제")
    pr = progress(saved)
    print(f"끝. 저장 {len(saved)}장, 고르게 찍힌 정도: " + ", ".join(f"{k} {v:.0%}" for k, v in pr.items()))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
