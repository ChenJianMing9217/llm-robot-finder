"""用程式送目的地：查地點名稱表 → 交給 Nav2 → 回報還剩幾公尺 → 回報結果。

用法：ros2 run nav_tools go_to 廚房                  （名字要在地點表裡）
      ros2 run nav_tools go_to 廚房 --places house     （房子用 places_house.yaml）
      ros2 run nav_tools go_to 廚房 --timeout 30       （超過 30 秒就取消，預設 120 秒）
      ros2 run nav_tools go_to --list                  （列出認得的地點）

機器人工具箱的 go_to 也是用這裡的 navigate()：輸入一個地點名字，輸出成功或失敗。
送給 Nav2 的是一個「動作」（action）：目標（去哪）→ 回報（還剩幾公尺）→ 結果（成功、失敗、取消）。
"""
import argparse
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformListener

from nav_tools.places import load_places, make_pose

STATUS = {GoalStatus.STATUS_ABORTED: "放棄", GoalStatus.STATUS_CANCELED: "被取消"}


def where_am_i(node, timeout):
    """問 TF：「地圖 → 機器人」這段關係存在嗎？存在就代表知道自己在哪（AMCL、SLAM 都適用）。"""
    if not hasattr(node, "_nav_tf"):                   # 同一個節點只建一次（巡邏會連續問幾十次，每次新建會越積越多）
        node._nav_tf = Buffer()
        node._nav_tf_listener = TransformListener(node._nav_tf, node)
    buf = node._nav_tf
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            return buf.lookup_transform("map", "base_link", rclpy.time.Time()).transform.translation
        except Exception:
            pass
    return None


def go_to(node, name, timeout, places_name=None):
    places = load_places(places_name)
    # 防護 1：名字不在表上 → 直接說不認得，不要亂猜座標
    if name not in places:
        print(f"❌ 不認得「{name}」這個地點。認得的有：{'、'.join(places)}")
        return False
    return navigate(node, places[name], name, timeout)


def navigate(node, p, label, timeout, feedback=True):
    """叫 Nav2 開到 p（x、y 公尺，yaw 度），label 是印出來用的名字。go_to、approach、explore 共用。
    feedback=False：不印「還剩幾公尺」（巡邏要開幾十段，畫面會太長）。"""
    # 防護 2：機器人要先知道自己在哪，不然 Nav2 沒辦法規劃
    here = where_am_i(node, 10.0)                  # 電腦很忙時，TF 要等久一點才收得到（3 秒曾經不夠）
    if here is None:
        print("❌ 機器人還不知道自己在哪（地圖上找不到它）。導航模式剛開的話，先執行：ros2 run nav_tools set_start")
        return False
    if not hasattr(node, "_nav_client"):               # 同上：Nav2 的連線也只建一次
        node._nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    client = node._nav_client
    if not client.wait_for_server(timeout_sec=10.0):
        print("❌ 找不到 Nav2（navigate_to_pose）：模擬有開嗎？")
        return False

    name = label
    print(f"現在在 ({here.x:.2f}, {here.y:.2f}) → 出發：{name} ({p['x']:.2f}, {p['y']:.2f})，到了面向 {p.get('yaw', 0):.0f} 度")
    goal = NavigateToPose.Goal()
    goal.pose = make_pose(p)
    last = [0.0]

    def on_feedback(msg):                              # 動作的「回報」：每秒印一次
        if feedback and time.monotonic() - last[0] >= 1.0:
            last[0] = time.monotonic()
            print(f"  還剩 {msg.feedback.distance_remaining:.2f} 公尺")

    sent = client.send_goal_async(goal, feedback_callback=on_feedback)
    rclpy.spin_until_future_complete(node, sent, timeout_sec=10.0)
    handle = sent.result()
    if handle is None or not handle.accepted:
        print("❌ Nav2 拒絕了這個目的地")
        return False

    t0 = time.monotonic()                              # 量「過了多久」用碼錶：系統時鐘可能被校時程式調動，碼錶不會
    result = handle.get_result_async()
    while not result.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        # 防護 3：開太久都沒到 → 取消，不要一直等下去
        if time.monotonic() - t0 > timeout:
            print(f"⏱ 超過 {timeout:.0f} 秒還沒到 → 取消")
            rclpy.spin_until_future_complete(node, handle.cancel_goal_async(), timeout_sec=5.0)
            return False
    secs = time.monotonic() - t0
    status = result.result().status
    if status == GoalStatus.STATUS_SUCCEEDED:
        print(f"✅ 到了：{name}（花了 {secs:.0f} 秒）")
        return True
    detail = result.result().result
    reason = getattr(detail, "error_msg", "") or f"錯誤代碼 {getattr(detail, 'error_code', '?')}"
    print(f"❌ 沒有到 {name}：Nav2 {STATUS.get(status, f'狀態 {status}')}（{reason}）")
    return False


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="用地點名字叫機器人開過去")
    parser.add_argument("place", nargs="?", help="地點名字，例如：廚房")
    parser.add_argument("--timeout", type=float, default=120.0, help="最多等幾秒（預設 120）")
    parser.add_argument("--list", action="store_true", help="列出認得的地點")
    parser.add_argument("--places", default=None, help="用哪張地點表，例如 house（預設：house）")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if args.list or not args.place:
        for name, p in load_places(args.places).items():
            print(f"  {name}：({p['x']}, {p['y']})，到了面向 {p.get('yaw', 0)} 度")
        rclpy.try_shutdown()
        return
    node = Node("go_to")
    ok = go_to(node, args.place, args.timeout, args.places)
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)
