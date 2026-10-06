"""假相機：把影片一張一張播出去，假裝是相機。

WSL 裡沒有相機，所以先用影片代替。對其他節點來說，這跟真的相機沒有差別：
都是往 /camera/image_raw 這個主題送 Image 訊息（換成 Gazebo 裡的模擬相機也一樣）。

參數：video（影片路徑）、fps（每秒幾張，0 = 照影片原本的速度）、width（縮小到多寬，0 = 不縮）
"""
import cv2
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from yolo_vision.images import to_msg


class VideoCamera(Node):
    def __init__(self):
        super().__init__("video_camera")
        self.path = self.declare_parameter("video", "").value
        self.width = self.declare_parameter("width", 640).value
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不開影片：{self.path!r}（用 -p video:=影片路徑 指定）")
        fps = self.declare_parameter("fps", 0.0).value or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        # 控制節奏用穩定時鐘（碼錶）；訊息上的時間章才用系統時間
        self.create_timer(1.0 / fps, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f"開始播放 {self.path}（每秒 {fps:.0f} 張，播完從頭再來）")

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:                                            # 播完了 → 從頭再來
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return
        if self.width and frame.shape[1] > self.width:        # 太大就縮小（傳輸、偵測都比較快）
            h = round(frame.shape[0] * self.width / frame.shape[1])
            frame = cv2.resize(frame, (self.width, h))
        header = Header()
        header.stamp = self.get_clock().now().to_msg()        # 時間章：這張是什麼時候「拍」的
        header.frame_id = "camera"                            # 座標系：從相機的角度看的（給 TF 用）
        self.pub.publish(to_msg(frame, header))


def main():
    rclpy.init()
    node = VideoCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
