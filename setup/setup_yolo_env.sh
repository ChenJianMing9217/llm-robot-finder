#!/usr/bin/env bash
# setup_yolo_env.sh —— 建立 YOLO 節點要用的 Python 環境（使用者自己執行，不需要 sudo）
#
# 為什麼要另外建環境：ROS 用的是系統的 Python，裡面沒有 YOLO（ultralytics、PyTorch）。
#   直接裝進系統的 Python 容易把 ROS 弄壞 → 另外建一個獨立的環境 ~/yolo_venv。
# 三個重點：
#   1. --system-site-packages：新環境「看得到」系統和 ROS 的套件（rclpy 等）
#   2. numpy 必須小於 2：ROS 的訊息程式是用 numpy 1.x 編譯的，換成 2.x 會壞
#   3. PyTorch 用 CPU 版：GPU 版要多下載 3 GB 以上；YOLO 最小的模型用 CPU 每秒也能處理十幾張
#      （房子模擬＋導航一起跑時 CPU 會不夠，有 NVIDIA 顯示卡就再跑 upgrade_yolo_gpu.sh）
#
# 用法（在 Ubuntu 終端機）：
#   bash setup/setup_yolo_env.sh
set -e
VENV="$HOME/yolo_venv"

echo "[1/4] 安裝 uv（裝 Python 套件的工具，Windows 那邊也是用它）"
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env"
uv --version

echo "[2/4] 建立獨立的 Python 環境：$VENV"
if [ ! -d "$VENV" ]; then
  uv venv --system-site-packages --python /usr/bin/python3 "$VENV"
fi

echo "[3/4] 安裝 PyTorch（CPU 版）"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] 安裝 YOLO（ultralytics），numpy 鎖在 1.x"
uv pip install --python "$VENV/bin/python" ultralytics "numpy<2" "opencv-python<4.12" "mcp==2.2.0"

echo ""
echo "=== 檢查 ==="
source /opt/ros/jazzy/setup.bash
"$VENV/bin/python" - <<'PY'
import numpy, torch, ultralytics, rclpy
from vision_msgs.msg import Detection2DArray
print(f"numpy {numpy.__version__}（要 1.x）、PyTorch {torch.__version__}、ultralytics {ultralytics.__version__}")
print("ROS 的套件（rclpy、vision_msgs）也找得到 ✅")
PY
echo ""
echo "完成 ✅ 環境在 $VENV（大約 1.3 GB）"
