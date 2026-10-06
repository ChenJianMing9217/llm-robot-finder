"""巡邏畫地圖：SLAM 模式下，照地點表裡排好的路線（explore）一個點一個點開，把整間房子看過一遍。

用法（先開 SLAM：ros2 launch home_sim house_nav.launch.py slam:=True）：
  ros2 run nav_tools explore --places house              （從第 1 個點開始）
  ros2 run nav_tools explore --places house --start 12   （從第 12 個點接著開，例如中途停掉之後）
走完再存地圖：ros2 run nav2_map_server map_saver_cli -f <檔名>

真正的「自動探索」（frontier exploration）會自己找「已知和未知的交界」開過去；
我們有房子的設計檔，所以事先把路線排好（Nav2 也有現成的 waypoint follower，做的是同一件事）。

⚠️ 實測踩到的兩個坑：
   1. SLAM 為了省力，機器人要「走超過 0.5 公尺或轉超過約 30 度」才把新的雷射畫進地圖。
      目的地還不在地圖裡（錯誤代碼 204）時，機器人站著等是沒用的 → 原地轉 90 度（逼 SLAM 畫一次）、等地圖更新再試；
      還是不行 → 先往目標的方向走一半（雷射只看得到 3.5 公尺，走近一點遠處的牆才會進地圖），再試一次。
   2. SLAM 偶爾開機時卡在「已設定、未啟動」（Nav2 啟動流程的時間差：「啟動」指令比「設定完成」先到）
      → 地圖一直不出來、機器人不知道自己在哪 → 出發前檢查，卡住就幫它按啟動。
"""
import argparse
import math
import sys
import time

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from nav_tools.go_to import navigate, where_am_i
from nav_tools.places import _load


def turn_and_wait(node, degrees=90, wait=6.0):
    """原地轉一下（Nav2 的 spin），再等 SLAM 更新地圖（每 5 秒一次）。"""
    if not hasattr(node, "_spin_client"):
        node._spin_client = ActionClient(node, Spin, "spin")
    if node._spin_client.wait_for_server(timeout_sec=5.0):
        goal = Spin.Goal()
        goal.target_yaw = math.radians(degrees)
        goal.time_allowance.sec = 15
        sent = node._spin_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, sent, timeout_sec=5.0)
        handle = sent.result()
        if handle is not None and handle.accepted:
            rclpy.spin_until_future_complete(node, handle.get_result_async(), timeout_sec=20.0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        rclpy.spin_once(node, timeout_sec=0.1)


def _call(node, srv_type, name, request):
    client = node.create_client(srv_type, name)
    if not client.wait_for_service(timeout_sec=3.0):
        return None
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    return future.result()


def ensure_slam(node):
    """出發前確認機器人知道自己在哪；SLAM 卡在「未啟動」就幫它按啟動。"""
    if where_am_i(node, 15.0) is not None:
        return True
    state = _call(node, GetState, "/slam_toolbox/get_state", GetState.Request())
    if state is not None and state.current_state.id == State.PRIMARY_STATE_INACTIVE:
        print("⚠️ SLAM 卡在「已設定、未啟動」（啟動流程的時間差）→ 幫它按啟動")
        request = ChangeState.Request()
        request.transition.id = Transition.TRANSITION_ACTIVATE
        _call(node, ChangeState, "/slam_toolbox/change_state", request)
    return where_am_i(node, 20.0) is not None


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="照排好的路線巡邏一圈（畫地圖用）")
    parser.add_argument("--places", default=None, help="用哪張地點表，例如 house")
    parser.add_argument("--start", type=int, default=1, help="從第幾個點開始（預設 1）")
    parser.add_argument("--timeout", type=float, default=120.0, help="每一段最多等幾秒（預設 120）")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    route = _load(args.places).get("explore") or []
    if not route:
        print("❌ 這張地點表沒有巡邏路線（explore）")
        rclpy.try_shutdown()
        sys.exit(1)
    node = Node("explore")
    if not ensure_slam(node):
        print("❌ 機器人不知道自己在哪：SLAM 有開嗎？（ros2 launch home_sim house_nav.launch.py slam:=True）")
        rclpy.try_shutdown()
        sys.exit(1)
    t0, failed = time.monotonic(), []
    for i in range(args.start - 1, len(route)):
        p = dict(route[i])
        nxt = route[i + 1] if i + 1 < len(route) else None
        p["yaw"] = math.degrees(math.atan2(nxt["y"] - p["y"], nxt["x"] - p["x"])) if nxt else 0.0  # 面向下一個點
        label = f"第 {i + 1}/{len(route)} 個：{p['name']}"
        print(f"\n▶ {label}")
        ok = navigate(node, p, label, args.timeout, feedback=False)
        if not ok:
            print("  → 原地轉 90 度、等 6 秒（讓 SLAM 把看到的畫進地圖），再試一次")
            turn_and_wait(node)
            ok = navigate(node, p, label, args.timeout, feedback=False)
        here = None if ok else where_am_i(node, 5.0)
        if not ok and here is not None:
            mid = {"x": (here.x + p["x"]) / 2, "y": (here.y + p["y"]) / 2, "yaw": p["yaw"]}
            print(f"  → 先往目標方向走一半，到 ({mid['x']:.1f}, {mid['y']:.1f})，讓雷射看得更遠")
            navigate(node, mid, "半路", args.timeout, feedback=False)
            turn_and_wait(node, 45)
            ok = navigate(node, p, label, args.timeout, feedback=False)
        if not ok:
            failed.append(label)
            print("  → 跳過，繼續下一個")
    minutes = (time.monotonic() - t0) / 60
    print(f"\n巡邏結束：{len(route) - args.start + 1 - len(failed)} 個點到了、{len(failed)} 個沒到，花了 {minutes:.1f} 分鐘")
    for label in failed:
        print(f"  ❌ {label}")
    print("下一步（模擬先不要關）：存地圖 → ros2 run nav2_map_server map_saver_cli -f <存檔路徑，不含副檔名>")
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if not failed else 1)
