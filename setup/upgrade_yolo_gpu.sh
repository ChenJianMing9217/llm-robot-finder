#!/usr/bin/env bash
# upgrade_yolo_gpu.sh —— 讓 YOLO 改用顯示卡（RTX 5070）（使用者自己執行，不需要 sudo）
#
# 為什麼：房子模擬、導航、YOLO 全部擠在 CPU 上 → YOLO 吃掉 4～5 個核心，
#   相機從每秒 23 張掉到 8 張，畫面一卡一卡（實測）。
#   改用顯示卡後：YOLO 只吃 0.3 個核心、每張 13 毫秒，相機恢復每秒 25 張。
# 做法：把 ~/yolo_venv 裡 CPU 版的 PyTorch 換成 CUDA 12.8 版（5070 是 Blackwell 架構，要 12.8 以上）。
#   numpy 一樣鎖在 1.x（ROS 的訊息程式需要）。YOLO 程式不用改：ultralytics 看到顯示卡就會自動用。
#
# 用法（在 Ubuntu 終端機）：
#   bash setup/upgrade_yolo_gpu.sh
set -e
source "$HOME/.local/bin/env"
export UV_HTTP_TIMEOUT=300          # NVIDIA 的下載站偶爾很慢，等久一點

echo "[1/2] 把 PyTorch 換成支援顯示卡的版本（約 3 GB，第一次下載要等一陣子）"
uv pip install --python "$HOME/yolo_venv/bin/python" --reinstall-package torch --reinstall-package torchvision \
    torch torchvision "numpy<2" --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[2/2] 檢查"
source /opt/ros/jazzy/setup.bash
"$HOME/yolo_venv/bin/python" - <<'PY'
import numpy, torch
from vision_msgs.msg import Detection2DArray
print(f"numpy {numpy.__version__}（要 1.x）、PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"顯示卡：{torch.cuda.get_device_name(0)} ✅")
else:
    print("❌ 用不到顯示卡")
PY
