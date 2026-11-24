"""1-B 물리 주행 시험: Nav2 Route Server + Controller Server(RPP)만 띄운다(BT Navigator, 지도 서버, AMCL 없음).
설정: 핑키 nav2_params.yaml(서브모듈 pinky_pro/pinky_navigation) 위에 lane_nav.yaml을 덮어 읽는다.
사용(젯슨): ros2 launch lane_nav.launch.py graph:=/절대경로/lane_graph.geojson [rpp:=/절대경로/덮어쓸.yaml]
"""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
PINKY_PARAMS = HERE.parent / "pinky_pro" / "pinky_navigation" / "params" / "nav2_params.yaml"


def setup(context):
    files = [str(PINKY_PARAMS), str(HERE / "lane_nav.yaml")]
    extra = LaunchConfiguration("rpp").perform(context)
    if extra:
        files.append(extra)                                      # 시험용 RPP 값 덮어쓰기(예: 주시 거리)
    graph = {"graph_filepath": LaunchConfiguration("graph").perform(context)}
    return [
        Node(package="nav2_controller", executable="controller_server", output="screen", parameters=files,
             remappings=[("cmd_vel", "/cmd_vel")]),
        Node(package="nav2_route", executable="route_server", output="screen", parameters=files + [graph]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_lane",
             output="screen", parameters=files),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("graph"), DeclareLaunchArgument("rpp", default_value=""),
                              OpaqueFunction(function=setup)])
