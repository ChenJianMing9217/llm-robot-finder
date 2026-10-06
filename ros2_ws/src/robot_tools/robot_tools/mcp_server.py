"""機器人工具箱：把導航（Nav2）和眼睛（YOLO）包成 MCP 工具，任何支援 MCP 的 Agent 都能接。

  where_am_i   我在哪：在哪個房間、地圖座標、面向
  go_to        去房子裡的某個房間（Nav2 自己規劃路線、穿過門）
  look         看一下：看到什麼、信心多高、在哪裡（需要時才跑 YOLO）
  look_around  原地轉一圈，每 45 度看一次（走進房間、沒看到要找的東西時用）
  approach     靠近某樣東西再看一次（先懷疑、再確認；「確認」由程式判斷，不是模型說了算）

記憶：look、look_around 看過的東西會記住地圖位置（SEEN）。approach 時畫面裡沒有它（例如轉完一圈已經背對它），
      就開回記憶裡的位置再看 —— 機器人「記得東西放在哪」，這就是語意地圖（semantic map）的雛形。

⚠️ MCP 用 stdio 傳訊息：stdout 是「通訊專用的線」。工具裡 print 的過程不能直接印到 stdout（會把通訊弄亂）
   → 每個工具執行時把 print 收集起來，當成結果的一部分回傳給 Agent（Agent 也看得到過程）。
啟動：ros2 run robot_tools robot_mcp（通常由 Agent 那一端自動啟動；模擬、導航、set_start 要先做好）
"""
import contextlib
import io
import math
import time
from typing import Annotated, Literal

import rclpy
import rclpy.time
from action_msgs.msg import GoalStatus
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from nav2_msgs.action import Spin
from pydantic import Field
from rclpy.action import ActionClient
from rclpy.node import Node

from nav_tools.go_to import go_to as nav_go_to
from nav_tools.places import load_places, room_of
from robot_tools.approach import CONFIRM_CONF, CONFIRM_RATIO, approach as do_approach
from robot_tools.eyes import Eyes, describe, zh

PLACES = load_places("house")
PlaceName = Literal[tuple(PLACES)]


def quiet(fn, *args):
    """執行 fn，把它 print 的過程收集起來（不讓它印到 stdout 弄亂通訊）。回傳 (結果, 過程文字)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip() and not ln.strip().startswith("還剩")]
    return out, "\n".join(lines)


rclpy.init()
node = Node("robot_mcp")
with contextlib.redirect_stdout(io.StringIO()):
    eyes = Eyes(node, conf=0.15)                   # 懷疑用低門檻，確認交給 approach
spin_client = ActionClient(node, Spin, "spin")
SEEN = {}                                          # 記憶：每樣東西最後一次看到的那一筆（有地圖位置的才記）


def _remember(objects):
    for o in objects:
        if o.get("map"):
            SEEN[o["name"]] = o

server = MCPServer(name="機器人工具箱", log_level="WARNING", instructions=(
    "你可以控制一台在房子裡的小型輪型機器人（TurtleBot3）。房間有：" + "、".join(PLACES) + "。"
    "相機離地只有約 10 公分、只看得到前方左右各約 30 度，看得清楚的是地上的東西。"
    "物體辨識用 YOLO，只認得 80 種常見物品，名稱是英文（例如 cup、refrigerator、couch、chair、bed）；"
    "它會把沒學過的東西認成最像的類別（例如把牆認成 bed），信心低於 0.5 的要用 approach 靠近確認。"))


def _pose(timeout=5.0):
    """(x, y, 面向度數)；不知道自己在哪回傳 None。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            t = eyes.tf.lookup_transform("map", "base_link", rclpy.time.Time()).transform
            return t.translation.x, t.translation.y, math.degrees(2 * math.atan2(t.rotation.z, t.rotation.w))
        except Exception:
            pass
    return None


@server.tool()
def where_am_i() -> str:
    """機器人現在在哪：在哪個房間、地圖座標、面向哪個方向。出發前、到達後都可以用來確認。"""
    p = _pose()
    if p is None:
        raise ToolError("機器人還不知道自己在哪（定位還沒設定）。請使用者先執行 set_start。")
    x, y, yaw = p
    return f"在「{room_of(x, y, 'house')}」，地圖座標 ({x:.1f}, {y:.1f})，面向 {yaw:.0f} 度（0 = 東、90 = 北、-90 = 南、180 = 西）"


@server.tool()
def go_to(place: Annotated[PlaceName, Field(description="要去的房間")]) -> str:
    """叫機器人自己開到房子裡的某個房間：會自動規劃路線、繞過障礙物、穿過門。要花 10 秒到 3 分鐘。
    到了之後機器人會面向房間裡的主要東西，可以接著用 look 看看有什麼。"""
    ok, log = quiet(nav_go_to, node, place, 300.0, "house")
    if not ok:
        raise ToolError(f"沒有到「{place}」。過程：\n{log}\n可以再試一次，或先去鄰近的房間再過去。")
    return log


@server.tool()
def look(seconds: Annotated[float, Field(description="看幾秒（1～5）", ge=1.0, le=5.0)] = 1.5) -> str:
    """用相機看一下面前有什麼：每樣東西出現在幾成的畫面、信心多高、偏左或偏右幾度、大約幾公尺、在地圖的哪裡。
    很快（約 2 秒）。信心低（< 0.5）或只在少數畫面出現的，可能是看錯，要用 approach 靠近確認。"""
    result = eyes.look(seconds)
    if result is None:
        raise ToolError("沒有收到相機畫面（模擬可能沒開）。")
    _remember(result["objects"])
    return describe(result)


def _spin(angle):
    goal = Spin.Goal()
    goal.target_yaw = float(angle)
    goal.time_allowance.sec = 15
    sent = spin_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, sent, timeout_sec=5.0)
    handle = sent.result()
    if handle is None or not handle.accepted:
        return False
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=20.0)
    return result.done() and result.result().status == GoalStatus.STATUS_SUCCEEDED


@server.tool()
def look_around() -> str:
    """原地轉一圈，每轉 45 度看一次（共 8 個方向），整理出四周看到的東西和它們在地圖上的位置。約 30 秒。
    走進房間、用 look 沒看到要找的東西時用。"""
    if not spin_client.wait_for_server(timeout_sec=5.0):
        raise ToolError("找不到原地旋轉的功能（Nav2 的 spin）。")
    best, turned = {}, 0
    for k in range(8):
        if k and not _spin(math.radians(45)):
            break
        turned = k + 1
        result = eyes.look(1.0)
        for o in (result or {}).get("objects", []):
            score = o["ratio"] * o["conf"]
            if score > best.get(o["name"], (None, -1.0))[1]:
                best[o["name"]] = (o, score)
    _remember(o for o, _ in best.values())
    if not best:
        return f"轉了 {turned} 個方向，什麼都沒看到。"
    lines = [f"轉了 {turned} 個方向（每 45 度一次）看到："]
    for name, (o, _) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        where = f"在地圖 ({o['map'][0]:.1f}, {o['map'][1]:.1f})" if o["map"] else o["note"]
        lines.append(f"  {o['zh']}（{name}）：看得最清楚的那次 {100 * o['ratio']:.0f}% 的畫面、信心 {o['conf']:.2f}，{where}")
    return "\n".join(lines)


@server.tool()
def approach(target: Annotated[str, Field(description="YOLO 的英文名稱，例如 cup")]):
    """靠近某樣東西再看一次（先懷疑、再確認）：算出它在地圖上的位置 → 開到它前方（小東西約 0.6 公尺、大東西約 1.4 公尺）→ 再看一次。
    結論由程式判斷：靠近後 8 成以上的畫面看到、信心 0.5 以上才算「確認」。約 10～60 秒。
    要先看過它：現在畫面裡看得到，或之前用 look／look_around 看過（會開回上次看到的位置）。"""
    names = set(eyes.model.names.values())
    if target not in names:
        raise ToolError(f"YOLO 不認得「{target}」。它只認得 80 種英文名稱，例如：{', '.join(sorted(names)[:25])}……")
    (before, after), log = quiet(do_approach, node, eyes, target, None, 90.0, SEEN.get(target))
    if after:
        _remember([after])
    ok = bool(after) and after["ratio"] >= CONFIRM_RATIO and after["conf"] >= CONFIRM_CONF
    verdict = "確認了 ✅" if ok else ("現在沒看到它，無法確認" if not before else "還是不確定 ⚠️（可能是看錯）")
    text = f"{log}\n結論：{zh(target)}（{target}）→ {verdict}"
    # 靠近後的畫面也一起回傳：Agent 那邊的程式可以請 VLM 再看一次（第二意見）；大腦只讀文字，看不到圖
    jpeg = eyes.snap() if before else None
    return [text, Image(data=jpeg, format="jpeg")] if jpeg else text


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
