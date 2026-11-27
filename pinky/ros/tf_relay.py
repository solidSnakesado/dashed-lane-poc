"""젯슨에서 쓸 TF를 /tf_local로 낸다: odom → base_footprint는 /odom 토픽으로 만들고, 나머지는 핑키의 /tf를 그대로 넘긴다(2026-11-27).

까닭: 같은 핑키 bringup이 같은 내용을 /odom과 /tf로 내는데, 공용 Wi-Fi를 건너 젯슨에 닿을 때 /tf만 크게 늦었다
  (0.3초 넘게 늦은 시간 /tf 54%, /odom 9%. 핑키 안에서 잰 /tf는 평균 2.4 ms로 제때 나옴 → 전달 구간 문제).
  TF가 늦으면 AMCL·컨트롤러·장애물 회피가 옛 위치를 써서 장애물이 번지고 "진행 없음"으로 섰다.
쓰는 쪽: lane_nav.launch.py(sim:=false)가 AMCL·컨트롤러·Route Server의 /tf를 /tf_local로 바꿔 띄우고 이것도 함께 띄운다.
  track_runner.py --real, loc_watch.py도 /tf_local을 본다. /tf_static(라이다 고정 위치 등)은 그대로 쓴다. 핑키 쪽은 바꾸지 않는다.
  AMCL이 내는 map → odom은 /tf_local로 바로 나간다(AMCL의 /tf를 바꿨으므로).
사용: python3 tf_relay.py [ODOM_TOPIC]      (기본 /odom)
"""
import sys
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

TF_QOS = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)


def main():
    odom_topic = sys.argv[1] if len(sys.argv) > 1 else "/odom"
    rclpy.init(); n = rclpy.create_node("tf_relay"); pub = n.create_publisher(TFMessage, "/tf_local", TF_QOS); seen = {"odom": 0, "tf": 0}

    def on_odom(m):                                                       # odom → base_footprint를 /odom 값으로
        t = TransformStamped(); t.header = m.header; t.child_frame_id = m.child_frame_id or "base_footprint"
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z; t.transform.rotation = q
        pub.publish(TFMessage(transforms=[t])); seen["odom"] += 1

    def on_tf(m):                                                         # 핑키의 /tf에서 odom → base_footprint만 빼고 넘김
        keep = [t for t in m.transforms if not (t.header.frame_id.lstrip("/") == "odom" and t.child_frame_id.lstrip("/") == "base_footprint")]
        if keep:
            pub.publish(TFMessage(transforms=keep)); seen["tf"] += 1
    n.create_subscription(Odometry, odom_topic, on_odom, 50)
    n.create_subscription(TFMessage, "/tf", on_tf, TF_QOS)
    def report():                                                         # 10초마다 넘긴 개수를 알리고 0으로 되돌림
        n.get_logger().info(f"10초 동안 {odom_topic}로 만든 TF {seen['odom']}개, /tf에서 넘긴 것 {seen['tf']}개"); seen.update(odom=0, tf=0)
    n.create_timer(10.0, report)
    n.get_logger().info(f"{odom_topic} → /tf_local(odom → base_footprint), /tf(그 밖) → /tf_local")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
