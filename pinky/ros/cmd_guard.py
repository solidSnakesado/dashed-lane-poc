"""핑키(로봇)에서 실행하는 명령 끊김 정지: 제어 PC가 보낸 /cmd_vel_pc를 /cmd_vel로 넘기고, TIMEOUT초 넘게 새 명령이 없으면 정지 명령을 보낸다.

실물 핑키 bringup은 /cmd_vel을 받을 때만 모터 속도를 바꾸고, 명령이 끊기면 마지막 속도로 계속 달린다(2026-11-19 실측, pinky_real_robot_values.md).
제어 PC의 정지 명령도 무선이 끊기면 오지 않으므로, 정지는 로봇 안에서 판단해야 한다. 핑키 코드는 고치지 않고 이 노드를 따로 띄운다.
시간은 로봇 자신의 단조 시계로 잰다(PC와 시계가 어긋나도 상관없음).
TIMEOUT 0.3초: 제어 주기 20 Hz(0.05초) + 잰 PC 쪽 무선 지연 최대 약 0.29초. 0.1 m/s면 그 사이 3 cm를 더 간다.
사용(핑키): python3 cmd_guard.py [TIMEOUT초]
"""
import sys, time
import rclpy
from geometry_msgs.msg import Twist


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    rclpy.init(); n = rclpy.create_node("cmd_guard")
    pub = n.create_publisher(Twist, "/cmd_vel", 10); st = dict(last=None, stopped=True, stops=0)

    def on_cmd(m):
        st["last"] = time.monotonic(); pub.publish(m)
        if st["stopped"] and (m.linear.x or m.angular.z):
            n.get_logger().info("명령 받음, 넘김 시작")
        st["stopped"] = False

    def check():
        if st["last"] is not None and time.monotonic() - st["last"] > timeout:
            pub.publish(Twist())                                         # 끊긴 동안 계속 정지 명령(한 번만 보내면 그것도 잃을 수 있음)
            if not st["stopped"]:
                st["stops"] += 1; n.get_logger().warn(f"명령 끊김 {timeout}초 넘음 → 정지 ({st['stops']}번째)")
            st["stopped"] = True
    n.create_subscription(Twist, "/cmd_vel_pc", on_cmd, 10)
    n.create_timer(0.05, check)
    n.get_logger().info(f"/cmd_vel_pc → /cmd_vel, {timeout}초 넘게 끊기면 정지")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
