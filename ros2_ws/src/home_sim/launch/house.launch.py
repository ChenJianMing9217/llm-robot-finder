"""房子模擬：房子＋家具＋杯子＋有彩色相機的 TurtleBot3 Waffle Pi。

參數：gui（要不要開 Gazebo 畫面，預設 true）、x_pose、y_pose、yaw（機器人出生的位置和方向，預設在前院 -2.0、-0.5，面向右）
相機影像：/camera/image_raw（跟 yolo_vision 的假相機同一個主題，YOLO 節點不用改）

生成機器人的部分沒有用 turtlebot3_gazebo 的 spawn 程式，因為它的橋接設定把 cmd_vel 定成 TwistStamped，
Nav2 送的是 Twist → 格式對不上，機器人一動也不動。這裡改用自己的 config/waffle_pi_bridge.yaml。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3 = get_package_share_directory("turtlebot3_gazebo")
    gz = get_package_share_directory("ros_gz_sim")
    home = get_package_share_directory("home_sim")
    world = os.path.join(home, "worlds", "house_cup.world")
    robot = os.path.join(tb3, "models", "turtlebot3_waffle_pi", "model.sdf")   # 有彩色相機的機型
    bridge = os.path.join(home, "config", "waffle_pi_bridge.yaml")

    def include(path, args, **kw):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=args.items(), **kw)

    os.environ["TURTLEBOT3_MODEL"] = "waffle_pi"          # turtlebot3 的 robot_state_publisher 會讀這個
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="要不要開 Gazebo 畫面"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("yaw", default_value="0.0", description="出生時面向的方向（弧度）"),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(tb3, "models")),   # 房子的模型放在這裡
        include(os.path.join(gz, "launch", "gz_sim.launch.py"),                           # 模擬核心（物理、感測器）
                {"gz_args": ["-r -s -v2 ", world], "on_exit_shutdown": "true"}),
        include(os.path.join(gz, "launch", "gz_sim.launch.py"),                           # 畫面（可以不開）
                {"gz_args": "-g -v2 ", "on_exit_shutdown": "true"}, condition=IfCondition(LaunchConfiguration("gui"))),
        include(os.path.join(tb3, "launch", "robot_state_publisher.launch.py"), {"use_sim_time": "true"}),
        Node(package="ros_gz_sim", executable="create", output="screen",                  # 把機器人放進世界
             arguments=["-name", "waffle_pi", "-file", robot, "-x", LaunchConfiguration("x_pose"),
                        "-y", LaunchConfiguration("y_pose"), "-z", "0.01", "-Y", LaunchConfiguration("yaw")]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",     # Gazebo ↔ ROS 的主題橋接
             arguments=["--ros-args", "-p", f"config_file:={bridge}"]),
        Node(package="ros_gz_image", executable="image_bridge", output="screen",          # 相機影像
             arguments=["/camera/image_raw"]),
    ])
