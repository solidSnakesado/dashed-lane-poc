"""1-B 물리 주행 시험: Nav2 Route Server + Controller Server(RPP)만 띄운다(BT Navigator, 지도 서버, AMCL 없음).
설정: 핑키 nav2_params.yaml(서브모듈 pinky_pro/pinky_navigation) 위에 lane_nav.yaml을 덮어 읽는다.
사용: ros2 launch lane_nav.launch.py graph:=/절대경로/lane_graph.geojson [rpp:=/절대경로/덮어쓸.yaml] [scan_tf:=true]
scan_tf:=true 이면 base_footprint → rplidar_link 고정 변환을 내보낸다(핑키 URDF 값: 앞뒤 -0.017, 높이 0.125 m, 180도 돌려 장착).
  시뮬 브리지는 odom → base_footprint만 내보내므로, 벽이 있는 세계에서 /scan을 코스트맵에 넣으려면 필요하다.
  실물이나 robot_state_publisher가 같은 변환을 내보내는 곳에서는 켜지 않는다.
sim:=false 이면 실물용: real.yaml을 마지막에 덮어 읽고(시뮬 시간 끔, /scan_fix, 느린 속도), scan_fix.py를 띄우고(lidar_yaw:=도, 기본 −2.2 = 1호기),
  컨트롤러 출력을 /cmd_vel_pc로 보낸다(핑키에서 cmd_guard.py가 /cmd_vel로 넘김). 이때 scan_tf는 켜지 않는다(핑키 bringup이 라이다 변환을 냄).
map:=지도.yaml 이면 지도 서버와 AMCL도 띄운다(설정 wall_amcl.yaml, 덮어쓸 초기 위치 amcl:=amcl_init.yaml은 map_align.py가 만듦).
  map → odom은 AMCL이 낸다. 이때 주행은 track_runner.py --align 으로 한다.
"""
import sys
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
PINKY_PARAMS = HERE.parent / "pinky_pro" / "pinky_navigation" / "params" / "nav2_params.yaml"


def setup(context):
    arg = lambda k: LaunchConfiguration(k).perform(context)
    real = arg("sim") == "false"; map_yaml = arg("map")
    files = [str(PINKY_PARAMS), str(HERE / "lane_nav.yaml")] + [f for f in [arg("rpp")] if f]   # rpp: 시험용 RPP 값 덮어쓰기(예: 주시 거리)
    if map_yaml:
        files += [str(HERE / "wall_amcl.yaml")] + [f for f in [arg("amcl")] if f]
    if real:
        files.append(str(HERE / "real.yaml"))                   # 맨 마지막에 읽어야 덮어씀
    graph = {"graph_filepath": arg("graph")}; sim_time = {"use_sim_time": not real}
    loc = [] if not map_yaml else [                              # 지도 서버 + AMCL(맨 앞에 띄움), 로컬 코스트맵은 지도 좌표로
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen", parameters=files + [{"yaml_filename": map_yaml}]),
        Node(package="nav2_amcl", executable="amcl", name="amcl", output="screen", parameters=files)]
    tf = [Node(package="tf2_ros", executable="static_transform_publisher", output="screen",
               arguments=["--x", "-0.017", "--z", "0.125", "--yaw", "3.141592653589793",
                          "--frame-id", "base_footprint", "--child-frame-id", "rplidar_link"],
               parameters=[sim_time])] if arg("scan_tf") == "true" and not real else []
    fix = [ExecuteProcess(cmd=[sys.executable, str(HERE / "scan_fix.py"), arg("lidar_yaw")], output="screen")] if real else []
    return tf + fix + loc + [
        Node(package="nav2_controller", executable="controller_server", output="screen", parameters=files,
             remappings=[("cmd_vel", "/cmd_vel_pc" if real else "/cmd_vel")]),
        Node(package="nav2_route", executable="route_server", output="screen", parameters=files + [graph]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_lane",
             output="screen", parameters=files),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("graph"), DeclareLaunchArgument("rpp", default_value=""),
                              DeclareLaunchArgument("scan_tf", default_value="false"),
                              DeclareLaunchArgument("map", default_value=""), DeclareLaunchArgument("amcl", default_value=""),
                              DeclareLaunchArgument("sim", default_value="true"), DeclareLaunchArgument("lidar_yaw", default_value="-2.2"),
                              OpaqueFunction(function=setup)])
