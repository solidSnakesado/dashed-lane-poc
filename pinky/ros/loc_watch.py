"""주행 중 추정 위치가 멈추는 원인 확인용 기록(2026-11-27). 로봇을 움직이지 않는다. track_runner.py와 함께 다른 터미널에서 돌린다.

실물 5차 주행에서 속도 명령은 0.05 m/s인데 추정 위치(TF map → base_footprint)가 3초 가까이 바뀌지 않다가 건너뛰었다.
어느 단계에서 끊기는지 보려고, 20 Hz로 다음을 함께 적는다(시각은 이 PC의 벽시계).
  TF map → odom(AMCL이 냄)의 stamp, TF odom → base_footprint(핑키 bringup이 냄)의 stamp, 둘을 이은 map → base_footprint의 x, y,
  /amcl_pose, /odom, /scan, /scan_fix를 마지막으로 받은 시각.
Ctrl+C로 끄면 OUT.npy에 저장하고, 항목마다 "끊김"(받은 간격 또는 stamp가 GAP초 넘게 멈춘 구간)을 출력한다.
사용: python3 loc_watch.py OUT [TF_TOPIC]        (예: ~/pinky_sim/runs/real_c7_6/loc_watch, TF_TOPIC 기본 /tf_local)
"""
import sys, time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

GAP = 0.3
COLS = ["벽시계", "map→odom stamp", "odom→base stamp", "map→base x", "map→base y", "/amcl_pose 받음", "/odom 받음", "/scan 받음", "/scan_fix 받음"]


def main():
    out, tf = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tf_local"   # 볼 TF 토픽(기본 /tf_local: tf_relay.py, 핑키 /tf를 보려면 /tf)
    rclpy.init(args=["--ros-args", "-r", f"/tf:={tf}"]); n = rclpy.create_node("loc_watch"); buf = Buffer(); TransformListener(buf, n)
    last = dict(amcl=np.nan, odom=np.nan, scan=np.nan, fix=np.nan); rows = []
    seen = lambda k: (lambda m: last.__setitem__(k, time.time()))
    n.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", seen("amcl"), 10)
    n.create_subscription(Odometry, "/odom", seen("odom"), 10)
    n.create_subscription(LaserScan, "/scan", seen("scan"), qos_profile_sensor_data)
    n.create_subscription(LaserScan, "/scan_fix", seen("fix"), qos_profile_sensor_data)
    st = lambda t: t.header.stamp.sec + t.header.stamp.nanosec * 1e-9

    def tick():
        row = [time.time()]
        for a, b in (("map", "odom"), ("odom", "base_footprint")):
            try:
                row.append(st(buf.lookup_transform(a, b, Time())))
            except Exception:
                row.append(np.nan)
        try:
            t = buf.lookup_transform("map", "base_footprint", Time()).transform.translation; row += [t.x, t.y]
        except Exception:
            row += [np.nan, np.nan]
        rows.append(row + [last["amcl"], last["odom"], last["scan"], last["fix"]])
    n.create_timer(0.05, tick); print("기록 중. Ctrl+C로 끝냄", flush=True)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    R = np.array(rows); np.save(out + ".npy", R); t = R[:, 0] - R[0, 0]
    print(f"{len(R)}줄, {t[-1]:.1f}초 → {out}.npy")
    for c in (1, 2, 5, 6, 7, 8):                                         # 값이 GAP초 넘게 그대로인 구간
        v = R[:, c]; ch = np.r_[True, np.diff(v) != 0] & np.isfinite(v); idx = np.where(ch)[0]
        gaps = [(t[i], t[j] - t[i]) for i, j in zip(idx[:-1], idx[1:]) if t[j] - t[i] > GAP]
        print(f"  {COLS[c]}: 받은 값 {int(np.isfinite(v).sum())}줄, 끊김 {len(gaps)}번" + ("" if not gaps else ", 가장 긴 것 %.2f초(%.1f초부터) / " % max(gaps, key=lambda g: g[1])[::-1]
                                                  + ", ".join(f"{a:.1f}s+{d:.2f}" for a, d in gaps[:8])))
    xy = R[:, 3:5]; mv = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]; idx = np.where(mv)[0]
    gaps = [(t[i], t[j] - t[i]) for i, j in zip(idx[:-1], idx[1:]) if t[j] - t[i] > GAP]
    print(f"  map→base 위치가 그대로: {len(gaps)}번 / " + ", ".join(f"{a:.1f}s+{d:.2f}" for a, d in gaps[:8]))
    n.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
