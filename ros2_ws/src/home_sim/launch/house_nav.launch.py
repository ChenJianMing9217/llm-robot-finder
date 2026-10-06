"""房子模擬＋Nav2＋RViz，一行開好。

畫地圖：ros2 launch home_sim house_nav.launch.py slam:=True
導航：  ros2 launch home_sim house_nav.launch.py map:=<地圖 .yaml 的完整路徑>（房子用 maps/house_map_v2_fixed.yaml）
        （導航模式開起來後，1 分鐘內要設定起點：ros2 run nav_tools set_start --places house）
參數：slam、map、gui（Gazebo 畫面）、rviz（RViz 畫面）

Nav2 的設定用 nav2_bringup 預設的，只改一個：機器人在代價地圖裡的半徑 0.22 → 0.20。
  Waffle Pi 寬 30.6、長 28.1 公分，最遠的角離中心約 20.8 公分 → 0.20 還蓋得住整台車；
  大門只有 0.9 公尺、門口還有信箱，0.22 時「不算貼牆」的通道只剩約 40 公分，偶爾卡在門框（實測）
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    nav2 = get_package_share_directory("nav2_bringup")
    home = get_package_share_directory("home_sim")

    params = RewrittenYaml(source_file=os.path.join(nav2, "params", "nav2_params.yaml"),
                           param_rewrites={"robot_radius": "0.20"}, convert_types=True)

    def include(path, args, **kw):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=args.items(), **kw)

    return LaunchDescription([
        DeclareLaunchArgument("slam", default_value="False", description="True = 一邊開一邊畫地圖"),
        DeclareLaunchArgument("map", default_value="", description="導航用的地圖（.yaml）"),
        DeclareLaunchArgument("gui", default_value="true", description="要不要開 Gazebo 畫面"),
        DeclareLaunchArgument("rviz", default_value="true", description="要不要開 RViz"),
        include(os.path.join(home, "launch", "house.launch.py"), {"gui": LaunchConfiguration("gui")}),
        include(os.path.join(nav2, "launch", "bringup_launch.py"),
                {"slam": LaunchConfiguration("slam"), "map": LaunchConfiguration("map"),
                 "use_sim_time": "true", "autostart": "true", "params_file": params}),
        include(os.path.join(nav2, "launch", "rviz_launch.py"), {}, condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
