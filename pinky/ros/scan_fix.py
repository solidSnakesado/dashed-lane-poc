"""실물 라이다의 장착 각도 오차를 고친 스캔을 낸다(제어 PC에서 실행, 핑키 코드는 고치지 않음).

/scan을 받아 각도 시작·끝(angle_min, angle_max)에 YAW_FIX를 더해 /scan_fix로 다시 낸다. 좌표 이름(rplidar_link)과 거리 값은 그대로.
라이다가 URDF보다 YAW_FIX만큼 돌아가 붙어 있으면, 센서가 θ 방향이라고 잰 점은 실제로 θ + YAW_FIX 방향에 있다.
1호기 −2.2°, 2호기 −4.0°(pinky_real_robot_values.md, 1호기는 재확인 필요).
사용: python scan_fix.py [YAW_FIX_도]       (기본 −2.2, 1호기)
"""
import math, sys
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def main():
    fix = math.radians(float(sys.argv[1]) if len(sys.argv) > 1 else -2.2)
    rclpy.init(); n = rclpy.create_node("scan_fix")
    pub = n.create_publisher(LaserScan, "/scan_fix", qos_profile_sensor_data)

    def on_scan(m):
        m.angle_min += fix; m.angle_max += fix; pub.publish(m)
    n.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    n.get_logger().info(f"/scan → /scan_fix, 각도 {math.degrees(fix):+.2f}도 고침")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
