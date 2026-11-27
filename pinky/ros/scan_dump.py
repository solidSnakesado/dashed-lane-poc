"""스캔 한 장과 라이다 고정 변환을 파일로 저장한다(2026-11-27, start_locate.py가 실물에서 지도와 맞지 않은 원인 확인용). 로봇은 움직이지 않는다.
저장: OUT.npz — ranges, angle_min, angle_increment, range_min, range_max, frame, tf(base_footprint ← 스캔 프레임: x, y, z, qx, qy, qz, qw)
사용: python3 scan_dump.py OUT [TOPIC]      (TOPIC 기본 /scan_fix)
"""
import sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def main():
    out, topic = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/scan_fix"
    rclpy.init(); n = rclpy.create_node("scan_dump"); buf = Buffer(); TransformListener(buf, n); box = {}
    n.create_subscription(LaserScan, topic, lambda m: box.setdefault("m", m), qos_profile_sensor_data)
    t0, tf = time.monotonic(), None
    while time.monotonic() - t0 < 10.0 and (tf is None or "m" not in box):
        rclpy.spin_once(n, timeout_sec=0.05)
        if "m" in box and tf is None:
            try:
                tf = buf.lookup_transform("base_footprint", box["m"].header.frame_id, Time()).transform
            except Exception:
                pass
    if "m" not in box or tf is None:
        print("스캔" if "m" not in box else "TF", "를 10초 안에 받지 못함"); return
    m = box["m"]; t, q = tf.translation, tf.rotation
    np.savez(out, ranges=np.asarray(m.ranges, float), angle_min=m.angle_min, angle_increment=m.angle_increment, range_min=m.range_min,
             range_max=m.range_max, frame=m.header.frame_id, tf=[t.x, t.y, t.z, q.x, q.y, q.z, q.w])
    r = np.asarray(m.ranges, float); ok = np.isfinite(r) & (r > m.range_min)
    print(f"저장: {out}.npz / 점 {len(r)}개 중 유효 {ok.sum()}개, 거리 {r[ok].min():.3f}~{r[ok].max():.3f} m, 프레임 {m.header.frame_id}, TF {np.round([t.x, t.y, t.z, q.z, q.w], 3).tolist()}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
