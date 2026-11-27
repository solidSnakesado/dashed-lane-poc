"""바닥에 수직으로 세운 체커보드로 카메라 장착 자세(바닥 위 높이, 숙인 각도, 옆 기울기, 판 기준 방향)를 실시간으로 보고 저장한다(2026-11-27, 비전 3단계).

전제: 판이 바닥에 수직이고 체커보드 줄이 바닥과 나란함. 로봇은 그 바닥 위, 판 앞에 둔다. 로봇은 움직이지 않는다(사람이 옮겨 가며 맞춤).
  (처음에는 바닥에 눕혀 썼으나 카메라 높이 약 3.7 cm에서 먼 줄이 6~8 px로 눌리고 종이가 울어 모서리를 찾지 못함)
사용: python3 cam_pose.py CALIB.yaml OUT_DIR H0_M [TOPIC] [COLS ROWS SQUARE_M]
  H0_M: 바닥에서 체커보드 맨 아래 안쪽 모서리 줄까지 높이(m, 자로 잼). 기본: TOPIC /camera/image_raw/compressed, 안쪽 모서리 8 x 6, 한 칸 0.030 m
창: 실시간 영상 위에 찾은 모서리(색 선), 다시 투영한 점(분홍), 값(높이, 숙임, 옆 기울기, 돌아간 각도, 판까지 거리, 다시 투영 오차)을 그린다.
  s = 지금 장면 저장(OUT_DIR/NN_raw.jpg, NN_pose.jpg, NN_pose.yaml, 터미널에도 값 출력), q = 끝.
방법: findChessboardCorners → cornerSubPix → solvePnP(내부 보정 K, D). 카메라 중심 C = −Rᵀt(체커보드 좌표).
  위 방향 = 체커보드 축 가운데 영상 위쪽과 가장 가까운 축. 높이 = 맨 아래 모서리 줄보다 위인 정도 + H0, 숙인 각도 = 광축이 아래로 기운 각도,
  옆 기울기 = 영상 가로축이 수평과 이루는 각도(영상 오른쪽이 위로 +), 돌아간 각도 = 광축을 수평으로 내린 방향과 판 정면 방향의 각도.
"""
import math, os, sys, time
import cv2, numpy as np, rclpy, yaml
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def solve(img, K, D, pat, sq, h0):
    """(값 dict, 모서리 px, 다시 투영 점) 또는 (None, None, None)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, px = cv2.findChessboardCorners(g, pat, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None, None, None
    px = cv2.cornerSubPix(g, px, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    cols, rows = pat; obj = np.zeros((rows * cols, 3)); obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq   # 체커보드 좌표(판 면 z = 0)
    _, rv, tv = cv2.solvePnP(obj, px, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, D); err = np.linalg.norm(proj.reshape(-1, 2) - px.reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(rv); C = (-R.T @ tv).ravel()                    # 카메라 중심(체커보드 좌표)
    zc, xc, yc = R.T @ np.r_[0, 0, 1.0], R.T @ np.r_[1.0, 0, 0], R.T @ np.r_[0, 1.0, 0]   # 광축, 영상 가로축, 영상 아래축(체커보드 좌표)
    up = max([np.r_[1.0, 0, 0], np.r_[-1.0, 0, 0], np.r_[0, 1.0, 0], np.r_[0, -1.0, 0]], key=lambda a: float(a @ -yc))   # 영상 위쪽과 가장 가까운 축 = 위
    height = float(C @ up) - float((obj @ up).min()) + h0                # 바닥 위 카메라 높이 = 맨 아래 모서리 줄 위 높이 + 잰 값
    pitch = math.degrees(math.asin(np.clip(-zc @ up, -1, 1))); roll = math.degrees(math.asin(np.clip(xc @ up, -1, 1)))
    nrm = np.r_[0, 0, np.sign(C[2]) or 1.0]                              # 판에서 카메라 쪽으로 향하는 판 수직 방향(수평)
    hz = zc - (zc @ up) * up; hz /= np.linalg.norm(hz); yaw = math.degrees(math.atan2(float(np.cross(-nrm, hz) @ up), float(hz @ -nrm)))
    res = dict(height_m=round(height, 4), pitch_down_deg=round(pitch, 2), roll_deg=round(roll, 2), yaw_to_board_deg=round(yaw, 2),
               dist_to_board_m=round(abs(float(C[2])), 4), bottom_row_height_m=h0,
               reproj_err_px_mean=round(float(err.mean()), 3), reproj_err_px_max=round(float(err.max()), 3))
    return res, px, proj


def lines(res):
    return [f"height {res['height_m'] * 1000:.1f} mm  pitch(down+) {res['pitch_down_deg']:+.2f}  roll {res['roll_deg']:+.2f} deg",
            f"yaw {res['yaw_to_board_deg']:+.2f} deg  dist {res['dist_to_board_m'] * 1000:.1f} mm  reproj {res['reproj_err_px_mean']:.2f}/{res['reproj_err_px_max']:.2f} px"]


def main():
    calib, out, h0 = sys.argv[1], sys.argv[2], float(sys.argv[3]); topic = sys.argv[4] if len(sys.argv) > 4 else "/camera/image_raw/compressed"
    cols, rows, sq = (int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else (8, 6, 0.030)
    c = yaml.safe_load(open(calib)); K = np.array(c["camera_matrix"]["data"], float).reshape(3, 3); D = np.array(c["distortion_coefficients"]["data"], float)
    os.makedirs(out, exist_ok=True); k = len([f for f in os.listdir(out) if f.endswith("_pose.yaml")])
    rclpy.init(); n = rclpy.create_node("cam_pose"); box = {"m": None, "t": []}
    n.create_subscription(CompressedImage, topic, lambda m: (box.__setitem__("m", m), box["t"].append(time.monotonic())), qos_profile_sensor_data)
    cv2.namedWindow("cam_pose", cv2.WINDOW_NORMAL); last = None
    print("창에서 s = 저장, q = 끝", flush=True)
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.02)
        if box["m"] is not None:
            img = cv2.imdecode(np.frombuffer(bytes(box["m"].data), np.uint8), cv2.IMREAD_COLOR); box["m"] = None
            if (img.shape[1], img.shape[0]) != (c["image_width"], c["image_height"]):
                print(f"영상 크기 {img.shape[1]}x{img.shape[0]}가 보정 크기와 다름"); break
            res, px, proj = solve(img, K, D, (cols, rows), sq, h0); vis = img.copy()
            box["t"] = [t for t in box["t"] if time.monotonic() - t < 2.0]
            if res is not None:
                cv2.drawChessboardCorners(vis, (cols, rows), px, True)
                for p in proj.reshape(-1, 2):
                    cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (255, 0, 255), -1)
            txt = [f"{len(box['t']) / 2.0:.1f} Hz  chessboard {cols}x{rows}: {'FOUND' if res else 'not found'}"] + (lines(res) if res else [])
            for i, s in enumerate(txt):
                cv2.putText(vis, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if res else (0, 0, 255), 2)
            cv2.imshow("cam_pose", vis); last = (img, vis, res)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and last is not None:
            img, vis, res = last
            if res is None:
                print("체커보드를 못 찾은 장면이라 저장하지 않음", flush=True); continue
            cv2.imwrite(f"{out}/{k:02d}_raw.jpg", img); cv2.imwrite(f"{out}/{k:02d}_pose.jpg", vis)
            yaml.safe_dump(dict(res, calib=calib, topic=topic), open(f"{out}/{k:02d}_pose.yaml", "w"), allow_unicode=True, sort_keys=False)
            print(f"저장 {out}/{k:02d}_*: " + " / ".join(lines(res)), flush=True); k += 1
    cv2.destroyAllWindows(); n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
