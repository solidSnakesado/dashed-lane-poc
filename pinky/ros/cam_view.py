"""카메라 압축 영상을 젯슨 화면 창에 실시간으로 띄우고, s를 누르면 그 장면을 저장한다(2026-11-27, 비전 3·4단계). q를 누르면 끝.

사용: python3 cam_view.py [OUT_DIR] [TOPIC] [COLS ROWS]   (기본 저장 폴더 없음(저장 안 함), /camera/image_raw/compressed, 안쪽 모서리 8 x 6)
화면 위쪽 글자: 받은 주기(Hz)와 체커보드 모서리를 찾았는지(찾았으면 모서리를 그림. 저장되는 사진에는 그리지 않음).
저장: OUT_DIR/NN.jpg(받은 그대로의 JPEG 바이트), NN.txt(영상 시각 sec.nanosec).
"""
import os, sys, time
import cv2, numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None; topic = sys.argv[2] if len(sys.argv) > 2 else "/camera/image_raw/compressed"
    pat = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (8, 6)
    if out:
        os.makedirs(out, exist_ok=True)
    k, last = len([f for f in os.listdir(out) if f.endswith(".jpg")]) if out else 0, None
    rclpy.init(); n = rclpy.create_node("cam_view"); box = {"m": None, "t": []}
    n.create_subscription(CompressedImage, topic, lambda m: (box.__setitem__("m", m), box["t"].append(time.monotonic())), qos_profile_sensor_data)
    cv2.namedWindow("cam_view", cv2.WINDOW_NORMAL)
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.02)
        if box["m"] is not None:
            last = box["m"]; img = cv2.imdecode(np.frombuffer(bytes(last.data), np.uint8), cv2.IMREAD_COLOR); box["m"] = None
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, px = cv2.findChessboardCorners(g, pat, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
            if ok:
                cv2.drawChessboardCorners(img, pat, px, True)
            box["t"] = [t for t in box["t"] if time.monotonic() - t < 2.0]
            txt = f"{len(box['t']) / 2.0:.1f} Hz / chessboard {pat[0]}x{pat[1]}: {'FOUND' if ok else 'not found'}"
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.imshow("cam_view", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and out and last is not None:
            open(f"{out}/{k:02d}.jpg", "wb").write(bytes(last.data)); st = last.header.stamp
            open(f"{out}/{k:02d}.txt", "w").write(f"{st.sec}.{st.nanosec:09d}\n"); print(f"저장 {out}/{k:02d}.jpg", flush=True); k += 1
    cv2.destroyAllWindows(); n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
