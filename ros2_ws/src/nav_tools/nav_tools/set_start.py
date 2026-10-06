"""告訴機器人「你現在在起點」—— 跟 RViz 的 2D Pose Estimate 按鈕做的事一樣。

只在剛開模擬、機器人還在起點時用一次。機器人已經開走了還用它，定位會被弄錯。
用法：ros2 run nav_tools set_start --places house    （房子；不給 --places 也是房子）

地點表有 start_step_out 時（房子），設好起點後會先「挪出牆邊」：出生點離房子南牆只有 25 公分，
在代價地圖裡等於貼著牆（Nav2 預設的機器人半徑 0.22 公尺＋地圖每格 5 公分的誤差），Nav2 第一趟會規劃不出路。
（後來 house_nav.launch.py 也把半徑改成 0.20；兩個一起做，第一趟才穩）
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args

from nav_tools.places import _load, make_pose


def wait_amcl_pose(node, timeout):
    """等 AMCL 說出「我現在在哪」；等不到回傳 None。"""
    got = []
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)   # 晚來的也能拿到最後一則
    sub = node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", got.append, latched)
    t0 = time.monotonic()
    while not got and time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[-1] if got else None


def step_out(node, turn_deg, distance):
    """轉 turn_deg 度、再往前開 distance 公尺。閉迴路：一邊看里程計一邊修正。"""
    pub = node.create_publisher(Twist, "cmd_vel", 10)
    odom = []
    sub = node.create_subscription(Odometry, "odom", odom.append, 10)

    def pose():
        p = odom[-1].pose.pose
        return p.position.x, p.position.y, 2 * math.atan2(p.orientation.z, p.orientation.w)

    t0 = time.monotonic()
    while not odom and time.monotonic() - t0 < 5:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not odom:
        print("（收不到里程計，沒有挪）")
        return
    target = pose()[2] + math.radians(turn_deg)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:              # 轉向：角度差越大轉越快（P 控制）
        rclpy.spin_once(node, timeout_sec=0.05)
        err = math.atan2(math.sin(target - pose()[2]), math.cos(target - pose()[2]))
        if abs(err) < math.radians(2):
            break
        cmd = Twist()
        cmd.angular.z = max(-0.8, min(0.8, 2.0 * err))
        pub.publish(cmd)
    pub.publish(Twist())
    x0, y0, _ = pose()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:              # 往前開：剩越少開越慢
        rclpy.spin_once(node, timeout_sec=0.05)
        x, y, _ = pose()
        left = distance - math.hypot(x - x0, y - y0)
        if left < 0.01:
            break
        cmd = Twist()
        cmd.linear.x = max(0.05, min(0.15, 1.5 * left))
        pub.publish(cmd)
    pub.publish(Twist())
    node.destroy_subscription(sub)


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="告訴機器人你在起點")
    parser.add_argument("--places", default=None, help="用哪張地點表，例如 house（預設：house）")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    node = Node("set_start")
    table = _load(args.places)
    start = table["places"]["起點"]
    pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    t0 = time.monotonic()
    while pub.get_subscription_count() == 0 and time.monotonic() - t0 < 10:   # 等 AMCL 在聽
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("找不到 AMCL：模擬有開嗎？")
    else:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose = make_pose(start).pose
        # 「大概在這附近」的範圍。RViz 按鈕給 0.25（前後左右約 50 公分、方向約 15 度）；
        # 機器人每次都從同一個出生點出發、位置很確定 → 給小一點（約 14 公分、6 度）
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.02
        msg.pose.covariance[35] = 0.01
        pub.publish(msg)
        print(f"已告訴機器人：你在起點 ({start['x']}, {start['y']})")
        pose = wait_amcl_pose(node, 5.0)
        if pose:
            p = pose.pose.pose.position
            print(f"AMCL 收到了，現在認為自己在 ({p.x:.2f}, {p.y:.2f}) ✅")
            out = table.get("start_step_out")
            if out:
                print(f"出生點離牆太近 → 先轉 {out['turn']} 度、往外開 {out['distance']} 公尺")
                step_out(node, out["turn"], out["distance"])
                pose = wait_amcl_pose(node, 3.0)
                if pose:
                    p = pose.pose.pose.position
                    print(f"挪好了，現在在 ({p.x:.2f}, {p.y:.2f}) ✅")
        else:
            print("AMCL 還沒回應，等幾秒再用 go_to 試試")
    node.destroy_node()
    rclpy.try_shutdown()
