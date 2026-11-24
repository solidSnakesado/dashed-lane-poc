"""pinky_pro의 MuJoCo 브리지(pinky_mujoco.bridge.Bridge)를 그대로 쓰되, 시험용으로 두 가지만 더해 실행한다.
  1) 젯슨(aarch64)에서는 pinky_mujoco가 colcon 빌드를 건너뛰므로, 서브모듈 소스를 바로 불러 쓴다(pinky_pro는 고치지 않음).
  2) /lane_sim/reset (geometry_msgs/PoseStamped)을 받으면 로봇을 그 자세(지도 좌표, 바퀴 축 기준)에 다시 세우고
     오도메트리도 그 자세에서 시작한다. 지도(map) = 시뮬 세계(world) = odom 출발점이라 map→odom은 항등 변환으로 둔다.
     (처음에는 로봇을 원점에 두고 map→odom을 경로마다 바꿨으나, 두 번째 경로부터 주행이 어긋나 이 방식으로 바꿈)
  3) --delay: 실물 핑키에서 잰 무선 지연(2026-11-24, 젯슨↔핑키 같은 Wi-Fi)을 흉내 낸다.
     /cmd_vel은 받은 뒤 지연 분포에서 뽑은 시간만큼 늦게 바퀴에 반영하고 0.6%는 버린다.
     /odom과 odom→base_footprint TF도 같은 분포만큼 늦게 내보낸다(내용은 보낼 때가 아니라 잰 순간의 값).
     /ground_truth는 평가용이라 늦추지 않는다. 메시지 순서는 바뀌지 않게 한다(ROS2 reliable과 같음).
바닥은 평평하고 벽이 없는 빈 세계(empty_world.yaml)라 로봇이 놓인 자리는 물리에 영향이 없다. 앞 카메라는 끈다(1단계는 비전 없음).
사용(젯슨, ~/venvs/lane 파이썬, ROS2와 lane_ros_ws를 source한 상태): python lane_sim_bridge.py [--viewer] [--delay]
"""
import math, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
SRC = ROOT / "pinky" / "pinky_pro" / "pinky_mujoco"
sys.path.insert(0, str(SRC))
import collections
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from pinky_mujoco.model import build_model, Physics
from pinky_mujoco.bridge import Bridge, set_planar_pose, stamp

# 잰 한쪽 지연(ROS2 왕복 ÷ 2)의 분위수: (누적 비율, ms). 왕복 최소 1.3, 중앙 7.6, 50 ms 넘음 11%, 100 ms 넘음 7%, 95% 181.7, 99% 357.3, 최대 475.5
DELAY_Q = [(0.0, 0.65), (0.5, 3.8), (0.89, 25.0), (0.93, 50.0), (0.95, 90.9), (0.99, 178.7), (1.0, 237.8)]
LOSS = 0.006            # 잰 손실 비율(620개 중 4개)


class LaneBridge(Bridge):
    def __init__(self, physics, delay=False):
        super().__init__(physics)
        self.create_subscription(PoseStamped, "/lane_sim/reset", self.reset, 10)
        self.delay, self.rng = delay, np.random.default_rng(0)
        self.cmd_q, self.state_q, self.last_cmd, self.last_state = collections.deque(), collections.deque(), 0.0, 0.0
        if delay:
            self.get_logger().info(f"무선 지연 흉내: 한쪽 지연 분위수 {DELAY_Q} ms, 손실 {LOSS:.1%}")

    def draw(self):
        """지연 분포에서 한 번 뽑기(초)."""
        p, q = zip(*DELAY_Q)
        return float(np.interp(self.rng.random(), p, q)) / 1000.0

    def receive_command(self, msg):
        if not self.delay:
            return super().receive_command(msg)
        if self.rng.random() < LOSS:
            return
        due = max(self.physics.data.time + self.draw(), self.last_cmd)       # 순서는 그대로
        self.last_cmd = due; self.cmd_q.append((due, msg))

    def advance(self):
        while self.cmd_q and self.cmd_q[0][0] <= self.physics.data.time:
            super().receive_command(self.cmd_q.popleft()[1])
        while self.state_q and self.state_q[0][0] <= self.physics.data.time:
            for m in self.state_q.popleft()[1]:
                (self.tf.sendTransform if isinstance(m, TransformStamped) else self.odom_pub.publish)(m)
        super().advance()

    def publish_state(self, now):
        if not self.delay:
            return super().publish_state(now)
        ph = self.physics                                                     # 잰 순간의 값으로 메시지를 만들고 늦게 보냄
        odom = Odometry(); odom.header.stamp, odom.header.frame_id = now, "odom"; odom.child_frame_id = "base_footprint"
        set_planar_pose(odom.pose.pose, ph.odom)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = map(float, ph.twist)
        for i, v in enumerate([0.0025, 0.0025, 1e6, 1e6, 1e6, 0.01]):
            odom.pose.covariance[7 * i] = v; odom.twist.covariance[7 * i] = v
        tf = TransformStamped(); tf.header = odom.header; tf.child_frame_id = odom.child_frame_id
        tf.transform.translation.x, tf.transform.translation.y = odom.pose.pose.position.x, odom.pose.pose.position.y
        tf.transform.rotation = odom.pose.pose.orientation
        if self.rng.random() >= LOSS:
            due = max(ph.data.time + self.draw(), self.last_state); self.last_state = due
            self.state_q.append((due, (odom, tf)))
        truth = PoseStamped(); truth.header.stamp, truth.header.frame_id = now, "world"
        set_planar_pose(truth.pose, ph.pose()); self.truth_pub.publish(truth)  # 평가용은 바로

    def reset(self, msg):
        ph = self.physics; p = msg.pose
        yaw = math.atan2(2 * (p.orientation.w * p.orientation.z), 1 - 2 * p.orientation.z ** 2)
        t = ph.data.time                                         # 시뮬 시간은 되돌리지 않음(/clock이 뒤로 가면 Nav2가 TF를 비우고 기다리는 시간이 꼬임)
        ph.mj.mj_resetData(ph.model, ph.data)
        ph.data.time = t
        ph.data.qpos[0:3] = [p.position.x, p.position.y, ph.settings["spawn"]["position"][2]]
        ph.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        ph.data.ctrl[:] = 0.0
        ph.mj.mj_forward(ph.model, ph.data)
        ph.odom[:] = [p.position.x, p.position.y, yaw]; ph.twist[:] = 0.0
        self.command, self.command_time = (0.0, 0.0), -math.inf
        self.cmd_q.clear(); self.state_q.clear(); self.last_cmd = self.last_state = ph.data.time
        self.get_logger().info(f"리셋: ({p.position.x:.3f}, {p.position.y:.3f}, {math.degrees(yaw):.1f}도)에 세움, 오도메트리도 여기서 시작")


def main():
    config = str(SRC / "config" / "mujoco.yaml")
    out = os.environ.get("PINKY_MODEL_DIR", str(Path.home() / ".cache" / "lane_sim_model"))
    model = build_model(SRC, out, None, config, str(HERE / "empty_world.yaml"), simple_render=True)
    physics = Physics(model, config)
    physics.settings["camera"]["enabled"] = False
    rclpy.init()
    node = LaneBridge(physics, delay="--delay" in sys.argv)
    viewer = None
    if "--viewer" in sys.argv:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(physics.model, physics.data)
    deadline = time.monotonic()
    try:
        while rclpy.ok() and (viewer is None or viewer.is_running()):   # pinky_mujoco.bridge.main과 같은 실시간 반복
            rclpy.spin_once(node, timeout_sec=0.0)
            node.advance()
            if viewer and node.tick % 3 == 0:
                viewer.sync()
            deadline += physics.bridge_period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer:
            viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
