"""全局配置：IP/串口/速度/限幅/路径 + rm75_curobo_demo 的 import 引导."""

import sys
from pathlib import Path

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJ_DIR = PKG_DIR.parent
DATA_DIR = PROJ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# rm75_curobo_demo 位置：Orin 上在 ~/rm75_curobo_demo，笔记本开发时用本地副本
_RM75_DEMO_CANDIDATES = [
    Path("/home/robot/rm75_curobo_demo"),        # Orin
    Path.home() / "rm75_curobo_demo",            # 笔记本（开发/语法检查）
]
RM75_DEMO_DIR = next((p for p in _RM75_DEMO_CANDIDATES if p.is_dir()), None)
if RM75_DEMO_DIR is not None:
    sys.path.insert(0, str(RM75_DEMO_DIR))
# 注意：只 import curobo_bridge.rm75_robot，绝不 import curobo_bridge.bridge
# （bridge 会拖起 torch/curobo + CUDA 初始化，采集线程用不到）

# ------------------------------------------------------------------
# 硬件
# ------------------------------------------------------------------
ARM_IP = "192.168.101.119"   # 换到力传感器正常的臂（.120 的 Fz 通道故障待修）
ARM_PORT = 8080

SKIN_PORT = "/dev/ttyUSB0"
SKIN_BAUD = 921600

# ------------------------------------------------------------------
# 运动参数（speed = 关节速度 °/s；demo 常规 30~80，精确定位 15~20）
# ------------------------------------------------------------------
SPEED_TRANSIT = 45       # 自由空间移动（悬停高度平移，别太快：离面仅 5mm）
SPEED_DESCEND = 5        # 接近/下探（安全关键，保持慢）
SPEED_PRESS = 2          # 按压段（安全关键，保持慢）
HOVER_MM = 5.0           # 点位上方悬停高度
STEP_MM = 0.25           # 下探步进
MAX_PROBE_BELOW_MM = 20.0  # touch-off 下探低于示教平面的最大深度
                            # （示教平面允许偏高；z0 每压自动重测，力限 15N 兜底）
HOLD_S = 0.8             # 保压采样窗口
SETTLE_S = 0.3           # 撤离后稳定

# ------------------------------------------------------------------
# 采集参数
# ------------------------------------------------------------------
GRID_PITCH_MM = 5.0      # 默认网格间距（9×9 覆盖 40×40 皮肤）
GRID_HALF_MM = 20.0
DEPTHS_MM = (0.5, 1.0, 1.5)
TOUCH_THR_N = 0.2        # 触发阈值下限（preflight 会按 4×FT 噪声 std 调大）
BASELINE_RECAL_EVERY = 50   # 每 N 点批量重估皮肤基线

# ------------------------------------------------------------------
# 安全
# ------------------------------------------------------------------
ABORT_FZ_N = 15.0        # |Fz| 中止
ABORT_F_ANY_N = 20.0     # 任意力轴中止
WS_MARGIN_MM = 10.0      # 工作空间盒外扩
WS_Z_BELOW_MM = 5.0      # 平面以下允许（压深+余量）
WS_Z_ABOVE_MM = 60.0     # 平面以上允许
SKIN_STALE_S = 0.5       # 皮肤流断流告警

# ------------------------------------------------------------------
# 流速率（SDK 共享一条 TCP，轮询过满会拖慢运动指令下发）
# ------------------------------------------------------------------
FT_RATE_HZ = 100.0
POSE_RATE_HZ = 50.0
SAFETY_RATE_HZ = 50.0

# HDF5 schema 版本
SCHEMA_VERSION = 1
