"""機器人的眼睛（給 Agent 用的工具）：需要時才看，看一小段時間，整理成「看到什麼、在哪裡」。

跟 yolo_vision 的 yolo_detector 節點不同：YOLO 只在呼叫 look() 時才跑，平常不佔 CPU、顯示卡。

位置怎麼算（相機離地約 9 公分、水平看前方；東西放在地上）：
  左右：框框中心離畫面中線多遠 → 偏左／偏右幾度
  遠近：框框底部（東西碰到地板的地方）比畫面中線低多少 → 越低越近
        距離 = 相機高度 × 焦距 ÷（底部 − 中線），焦距從 /camera/camera_info 讀
  最後用 TF 把「相機前方幾公尺、左邊幾公尺」換算成地圖座標
限制：同一種東西有好幾個時，每張畫面只取信心最高的那一個。
"""
import math
import statistics
import time
from collections import defaultdict

import cv2
import rclpy
import rclpy.time
import tf2_geometry_msgs  # noqa: F401  讓 TF 會換算 PointStamped
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO

from yolo_vision.images import default_model, from_msg

MODEL = default_model()                            # 環境變數 YOLO_MODEL，沒設定就用 yolo11n.pt
ZH = {"cup": "杯子", "refrigerator": "冰箱", "couch": "沙發", "chair": "椅子", "bed": "床", "tv": "電視",
      "dining table": "餐桌", "bottle": "瓶子", "person": "人", "potted plant": "盆栽", "bowl": "碗", "vase": "花瓶",
      "book": "書", "clock": "時鐘", "oven": "烤箱", "sink": "水槽", "microwave": "微波爐", "toilet": "馬桶",
      "laptop": "筆電", "train": "火車", "bench": "長椅", "suitcase": "行李箱"}


def zh(name):
    return ZH.get(name, name)


class Eyes:
    def __init__(self, node, conf=0.25):
        self.node, self.conf = node, conf
        self.model = YOLO(MODEL)
        self.collecting, self.frames, self.info = False, [], None
        node.create_subscription(Image, "/camera/image_raw", self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, qos_profile_sensor_data)
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, node)

    def _on_image(self, msg):
        if self.collecting:
            self.frames.append(msg)

    def _on_info(self, msg):
        self.info = msg

    def look(self, seconds=1.5, wait=5.0):
        """看 seconds 秒（從收到第一張畫面開始算）。回傳 {frames, seconds, objects}；objects 每樣東西一筆。
        等了 wait 秒都沒收到畫面回傳 None。"""
        self.frames, self.collecting = [], True
        t0 = time.monotonic()
        while not self.frames and time.monotonic() - t0 < wait:   # 剛啟動時，跟相機「接上線」要一點時間
            rclpy.spin_once(self.node, timeout_sec=0.05)
        t0 = time.monotonic()
        while self.frames and time.monotonic() - t0 < seconds:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.collecting = False
        frames = self.frames
        if not frames:
            return None
        hits = defaultdict(list)                   # 名稱 → 每張畫面裡信心最高的那一個：(信心, 中心 u, 底部 v)
        for msg in frames:
            result = self.model(from_msg(msg), conf=self.conf, verbose=False)[0]
            best = {}
            for box in result.boxes:
                name, c = result.names[int(box.cls)], float(box.conf)
                x1, _, x2, y2 = box.xyxy[0].tolist()
                if c > best.get(name, (0.0,))[0]:
                    best[name] = (c, (x1 + x2) / 2, y2)
            for name, h in best.items():
                hits[name].append(h)
        last = frames[-1]
        objects = []
        for name, hs in hits.items():
            u = statistics.median(h[1] for h in hs)
            v = statistics.median(h[2] for h in hs)
            objects.append(dict(name=name, zh=zh(name), ratio=len(hs) / len(frames), conf=max(h[0] for h in hs),
                                **self._where(u, v, last.width, last.height, last.header.frame_id)))
        objects.sort(key=lambda o: (-o["ratio"], -o["conf"]))
        return dict(frames=len(frames), seconds=seconds, objects=objects)

    def snap(self, wait=3.0, max_side=640, quality=85):
        """拍一張現在的畫面，回傳 JPEG（給 VLM 看第二意見用）；收不到畫面回傳 None。"""
        self.frames, self.collecting = [], True
        t0 = time.monotonic()
        while not self.frames and time.monotonic() - t0 < wait:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.collecting = False
        if not self.frames:
            return None
        frame = from_msg(self.frames[-1])
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def _where(self, u, v, w, h, frame_id):
        """畫面上的 (u 框框中心, v 框框底部) → 方向（度，正 = 偏左）、距離（公尺）、地圖座標。"""
        if self.info is not None and self.info.k[0] > 0:
            fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        else:                                      # 沒收到 camera_info：用 Waffle Pi 相機的水平視角 62.2 度推算
            fx = fy = (w / 2) / math.tan(1.0856 / 2)
            cx, cy = w / 2, h / 2
        out = dict(bearing=math.degrees(math.atan((cx - u) / fx)), distance=None, map=None, note="")
        if v >= h - 2:
            out["note"] = "底部超出畫面（很近）"
            return out
        if v <= cy + 1:
            out["note"] = "底部在水平線上方（不在地上，或很遠）"
            return out
        cam_h = self._camera_height(frame_id)
        d = cam_h * fy / (v - cy)                   # 相機前方幾公尺
        out["distance"] = d
        p = PointStamped()                          # 相機座標系：x 前、y 左、z 上；地上那一點在相機下方 cam_h
        p.header.frame_id = frame_id
        p.point.x, p.point.y, p.point.z = d, d * (cx - u) / fx, -cam_h
        try:
            q = self.tf.transform(p, "map", timeout=Duration(seconds=0.5))
            out["map"] = (q.point.x, q.point.y)
        except Exception:
            out["note"] = "換算不到地圖座標（TF 還沒準備好）"
        return out

    def _camera_height(self, frame_id):
        try:
            return self.tf.lookup_transform("base_footprint", frame_id, rclpy.time.Time()).transform.translation.z
        except Exception:
            return 0.094                            # Waffle Pi：相機離地約 9.4 公分


def describe_one(o):
    """一樣東西 → 一行中文。"""
    side = "正前方" if abs(o["bearing"]) < 5 else f"偏{'左' if o['bearing'] > 0 else '右'} {abs(o['bearing']):.0f} 度"
    far = f"大約 {o['distance']:.1f} 公尺" if o["distance"] else o["note"]
    where = f"，在地圖 ({o['map'][0]:.1f}, {o['map'][1]:.1f})" if o["map"] else ""
    return f"{o['zh']}（{o['name']}）：{100 * o['ratio']:.0f}% 的畫面、信心最高 {o['conf']:.2f}、{side}、{far}{where}"


def describe(result):
    """look() 的結果 → 中文（人看、之後 Agent 也看這個）。"""
    if result is None:
        return "❌ 沒有收到相機畫面（模擬有開嗎？）"
    lines = [f"看了 {result['seconds']:.1f} 秒（{result['frames']} 張畫面）："]
    lines += [f"  {describe_one(o)}" for o in result["objects"]] or ["  什麼都沒看到"]
    return "\n".join(lines)
