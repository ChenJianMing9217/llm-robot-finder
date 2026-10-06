"""影像格式轉換：ROS 的 Image 訊息 ↔ numpy 陣列（OpenCV、YOLO 用的格式）。

ROS 有現成的 cv_bridge，但它是用系統的 OpenCV 編譯的，跟 YOLO 環境裡的 OpenCV 版本不同，
混在一起容易出問題 → 自己寫，只有幾行。
"""
import array

import numpy as np
from sensor_msgs.msg import Image


def to_msg(frame, header):
    """numpy 陣列（高 × 寬 × 3，藍綠紅）→ Image 訊息。"""
    msg = Image()
    msg.header = header
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"               # 每個像素 3 個數字：藍、綠、紅（OpenCV 的順序）
    msg.step = msg.width * 3            # 一列有幾個位元組
    msg.data = array.array("B", np.ascontiguousarray(frame).tobytes())
    return msg


def from_msg(msg):
    """Image 訊息 → numpy 陣列（高 × 寬 × 3，藍綠紅）。"""
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"不支援的影像格式：{msg.encoding}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return frame[:, :, ::-1] if msg.encoding == "rgb8" else frame   # 紅綠藍 → 藍綠紅


def default_model():
    """YOLO 權重放哪：環境變數 YOLO_MODEL，沒設定就用 yolo11n.pt（Ultralytics 第一次會自動下載）。"""
    import os
    if os.environ.get("YOLO_MODEL"):
        return os.environ["YOLO_MODEL"]
    return "yolo11n.pt"
