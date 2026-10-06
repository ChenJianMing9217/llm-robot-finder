"""看一下：機器人現在看到什麼、在哪裡（需要模擬開著）。

用法：ros2 run robot_tools look                 （看 1.5 秒）
      ros2 run robot_tools look --seconds 3     （看久一點）
      ros2 run robot_tools look --conf 0.4      （門檻，預設 0.25：寧可多懷疑，之後再靠近確認）
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from robot_tools.eyes import Eyes, describe


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="機器人現在看到什麼")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    node = Node("look")
    result = Eyes(node, args.conf).look(args.seconds)
    print(describe(result))
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if result else 1)
