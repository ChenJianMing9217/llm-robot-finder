"""靠近再看（先懷疑、再確認）：用 look 算出東西在地圖上的位置 → 叫 Nav2 開到它前面 → 再看一次。

用法：ros2 run robot_tools approach cup              （名稱用 YOLO 的英文名稱）
      ros2 run robot_tools approach cup --stop 0.6   （停在東西前方幾公尺；不給就看東西大小）

停多遠看東西大小：相機離地只有 10 公分、上下只看得到約 ±24 度 → 大東西靠太近只拍到底部一截，YOLO 反而認不出來
（實測：沙發 1.9 公尺外信心 0.79，開到 0.6 公尺就認不出來；冰箱也一樣）→ 小東西 0.6、中等 1.0、大東西 1.4 公尺。

「懷疑」用低門檻（預設 0.15：寧可多懷疑，反正會靠近再確認；1.85 公尺外的杯子信心可能不到 0.25）。
「確認」的標準：靠近後出現在 8 成以上的畫面、信心 0.5 以上。
"""
import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from nav_tools.go_to import navigate, where_am_i
from robot_tools.eyes import Eyes, describe, describe_one, zh

CONFIRM_RATIO, CONFIRM_CONF = 0.8, 0.5
STOP = {"couch": 1.4, "bed": 1.4, "refrigerator": 1.4, "dining table": 1.2, "tv": 1.2, "bench": 1.2,
        "chair": 1.0, "toilet": 1.0, "potted plant": 1.0, "suitcase": 1.0, "oven": 1.0, "sink": 1.0}


def stop_distance(target):
    """停在東西前方幾公尺：大東西要遠一點，相機才拍得到整體（小東西 0.6）。"""
    return STOP.get(target, 0.6)


def find(result, target):
    return next((o for o in result["objects"] if o["name"] == target), None) if result else None


def approach(node, eyes, target, stop=None, timeout=90.0, remembered=None):
    """回傳 (靠近前, 靠近後)：各是 look 結果裡的那一筆，沒看到是 None。stop 不給就看東西大小（stop_distance）。
    remembered：之前看到它的那一筆（例如 look_around 轉一圈時）。現在畫面裡沒有它，就改用記憶裡的位置。"""
    stop = stop_distance(target) if stop is None else stop
    result = eyes.look(1.5)
    if result is None:
        print("❌ 沒有收到相機畫面（模擬有開嗎？）")
        return None, None
    before = find(result, target)
    if before is None and remembered and remembered.get("map"):
        before = remembered
        print(f"現在畫面裡沒有{zh(target)}，改用之前看到的位置：地圖 ({before['map'][0]:.1f}, {before['map'][1]:.1f})"
              f"（當時信心 {before['conf']:.2f}）")
    elif before is None:
        print(f"❌ 現在沒看到{zh(target)}（{target}）→ 先轉個方向，或換個地方再看。這次看到的是：")
        print(describe(result))
        return None, None
    else:
        print(f"靠近前：{describe_one(before)}")
    if before["map"] is None:
        print(f"❌ 看得到{zh(target)}，但算不出它在哪（{before['note']}）")
        return before, None
    here = where_am_i(node, 10.0)
    if here is None:
        print("❌ 機器人還不知道自己在哪")
        return before, None
    px, py = before["map"]
    dx, dy = px - here.x, py - here.y
    dist = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx))
    goal = None
    if abs(dist - stop) > 0.15:                    # 停在東西前方 stop 公尺，面向它（太遠就靠近、太近就退後）
        goal = {"x": px - stop * dx / dist, "y": py - stop * dy / dist, "yaw": yaw}
    elif before is remembered:                     # 已經夠近，但沒面向它 → 原地轉過去
        goal = {"x": here.x, "y": here.y, "yaw": yaw}
    if goal and not navigate(node, goal, f"{zh(target)}前方 {stop} 公尺", timeout):
        return before, None
    after = find(eyes.look(1.5), target)
    if after is None:
        print(f"⚠️ 靠近之後反而沒看到{zh(target)} → 剛剛可能是看錯")
    else:
        print(f"靠近後：{describe_one(after)}")
    return before, after


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="靠近某樣東西再看一次")
    parser.add_argument("target", help="YOLO 的英文名稱，例如 cup")
    parser.add_argument("--stop", type=float, default=None, help="停在東西前方幾公尺（不給就看東西大小）")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--conf", type=float, default=0.15, help="懷疑的門檻（預設 0.15）")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    node = Node("approach")
    before, after = approach(node, Eyes(node, args.conf), args.target, args.stop, args.timeout)
    ok = bool(after) and after["ratio"] >= CONFIRM_RATIO and after["conf"] >= CONFIRM_CONF
    if before and after:
        verdict = "確認了 ✅" if ok else "還是不確定 ⚠️"
        print(f"結論：{zh(args.target)}出現的比例 {100 * before['ratio']:.0f}% → {100 * after['ratio']:.0f}%、"
              f"信心 {before['conf']:.2f} → {after['conf']:.2f} → {verdict}")
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)
