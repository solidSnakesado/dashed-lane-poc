"""wall_map.py의 블럭 길에서 slam_toolbox로 지도를 만든다(시뮬 예행). Nav2(lane_nav.launch.py)와 같이 띄우지 않는다.
띄우는 것: base_footprint → rplidar_link 고정 변환(핑키 URDF 값, lane_nav.launch.py의 scan_tf와 같음),
          slam_toolbox(async, 수명 주기 노드를 이 파일이 configure → activate 함. slam_toolbox의 online_async_launch.py와 같은 방식).
slam_toolbox가 map → odom 변환을 낸다. 지도 작성 주행은 map_drive.py, 저장은 nav2_map_server의 map_saver_cli.
사용: ros2 launch wall_slam.launch.py [params:=/절대경로/wall_slam.yaml]
"""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    slam = LifecycleNode(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox", namespace="",
                         output="screen", parameters=[LaunchConfiguration("params"), {"use_sim_time": True}])
    tf = Node(package="tf2_ros", executable="static_transform_publisher", output="screen",
              arguments=["--x", "-0.017", "--z", "0.125", "--yaw", "3.141592653589793",
                         "--frame-id", "base_footprint", "--child-frame-id", "rplidar_link"],
              parameters=[{"use_sim_time": True}])
    configure = EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam), transition_id=Transition.TRANSITION_CONFIGURE))
    activate = RegisterEventHandler(OnStateTransition(target_lifecycle_node=slam, start_state="configuring", goal_state="inactive",
                                    entities=[EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam),
                                                                          transition_id=Transition.TRANSITION_ACTIVATE))]))
    return LaunchDescription([DeclareLaunchArgument("params", default_value=str(HERE / "wall_slam.yaml")), tf, slam, configure, activate])
