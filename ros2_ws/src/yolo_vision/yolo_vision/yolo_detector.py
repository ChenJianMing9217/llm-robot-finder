"""YOLO 偵測節點：機器人的「眼睛」。

訂閱：/camera/image_raw（sensor_msgs/Image）—— 不管是假相機、Gazebo 還是真相機都可以
發布：/yolo/detections（vision_msgs/Detection2DArray）—— 給程式用：每個物品的種類、信心、框框
      /yolo/image（sensor_msgs/Image）—— 給人看：畫上框框的影像
參數：model（模型檔）、conf（信心門檻，低於它的不算）
"""
import time
from collections import Counter

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from yolo_vision.images import default_model, from_msg, to_msg

DEFAULT_MODEL = default_model()                    # 環境變數 YOLO_MODEL，沒設定就用 yolo11n.pt


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.model = YOLO(self.declare_parameter("model", DEFAULT_MODEL).value)
        self.conf = self.declare_parameter("conf", 0.4).value
        self.det_pub = self.create_publisher(Detection2DArray, "/yolo/detections", 10)
        self.img_pub = self.create_publisher(Image, "/yolo/image", 10)
        # 只留「最新的一張」：來不及處理就跳過舊的，寧可少看幾張，也不要越積越多、看到的都是過去
        latest_only = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, latest_only)
        self.count, self.busy, self.seen = 0, 0.0, Counter()
        self.create_timer(2.0, self.report, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info("YOLO 準備好了，等相機的影像……")

    def on_image(self, msg):
        t0 = time.monotonic()
        result = self.model(from_msg(msg), conf=self.conf, verbose=False)[0]
        out = Detection2DArray()
        out.header = msg.header                       # 沿用相機的時間章和座標系：「那一刻、從相機看」
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()     # 框框的左上角、右下角（像素）
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x, det.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = result.names[int(box.cls)]   # 例如 "cup"
            hyp.hypothesis.score = float(box.conf)                 # 信心 0～1
            det.results.append(hyp)
            out.detections.append(det)
            self.seen[hyp.hypothesis.class_id] += 1
        self.det_pub.publish(out)
        self.img_pub.publish(to_msg(result.plot(), msg.header))   # result.plot() = 畫好框框的影像
        self.count += 1
        self.busy += time.monotonic() - t0

    def report(self):
        """每 2 秒報告一次：處理了幾張、每張花多久、平均每張看到什麼。"""
        if self.count == 0:
            return
        seen = "、".join(f"{name} ×{n / self.count:.1f}" for name, n in self.seen.most_common(5)) or "（沒有）"
        self.get_logger().info(f"每秒處理 {self.count / 2:.1f} 張，每張 {1000 * self.busy / self.count:.0f} 毫秒；"
                               f"平均每張看到：{seen}")
        self.count, self.busy, self.seen = 0, 0.0, Counter()


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
