"""地點名稱表（語意地圖）：把「名字」翻成 Nav2 看得懂的「目的地」。

預設用房子的表 places_house.yaml（maps/house_map_v2）。換別的場地：新增 places_<名字>.yaml，
用 --places <名字>，或環境變數 NAV_PLACES=<名字> 選擇。
"""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped


def _load(name):
    name = os.environ.get("NAV_PLACES", "") if name is None else name
    path = os.path.join(get_package_share_directory("nav_tools"), f"places_{name or 'house'}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_places(name=None):
    return _load(name)["places"]


def room_of(x, y, name=None):
    """地圖座標 (x, y) 在哪個房間（地點表裡的 rooms）；都不在就回傳「前院」。"""
    for room, r in (_load(name).get("rooms") or {}).items():
        if r["x"][0] <= x <= r["x"][1] and r["y"][0] <= y <= r["y"][1]:
            return room
    return "前院"


def make_pose(place):
    """地點 → 目的地（PoseStamped）= 座標系 + 位置 + 方向。"""
    pose = PoseStamped()
    pose.header.frame_id = "map"                     # 這個座標是用哪張地圖算的（TF 的觀念）
    pose.pose.position.x = float(place["x"])
    pose.pose.position.y = float(place["y"])
    yaw = math.radians(place.get("yaw", 0))          # 到了要面向哪（度 → 弧度）
    pose.pose.orientation.z = math.sin(yaw / 2)      # 方向用「四元數」表示（ROS 的慣例）
    pose.pose.orientation.w = math.cos(yaw / 2)
    return pose
