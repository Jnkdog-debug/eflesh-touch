#!/usr/bin/env python3
"""Orin 环境自检 —— 依赖 / SDK import / HDF5 写读往返 / 事件字段一致性."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("python:", sys.executable)

# 1) 依赖
import h5py  # noqa: E402
import serial  # noqa: E402
import numpy as np  # noqa: E402
print(f"[1] deps OK: h5py {h5py.__version__}, pyserial {serial.VERSION}, numpy {np.__version__}")

# 2) SDK import（不连接）—— 先跑 config 的 sys.path 引导
import eflesh_calib.config as cfg  # noqa: E402
assert cfg.RM75_DEMO_DIR is not None, "找不到 rm75_curobo_demo 目录"
from curobo_bridge.rm75_robot import RM75Robot  # noqa: E402
print(f"[2] RM75Robot import OK (demo dir: {cfg.RM75_DEMO_DIR})")

# 3) recorder HDF5 写读往返（合成数据）
from eflesh_calib.clock import SessionClock  # noqa: E402
from eflesh_calib.recorder import Hdf5Recorder  # noqa: E402
clk = SessionClock()
out = Path("/tmp/eflesh_selftest.h5")
rec = Hdf5Recorder(out, clk, {"session": "selftest"})
rng = np.random.default_rng(0)
for i in range(50):
    rec.q("skin").append((clk.now_ns(), i, rng.normal(0, 100, (5, 3)).astype("f4")))
    rec.q("ft").append((clk.now_ns(), rng.normal(0, 1, 6).astype("f4"), 0))
    rec.q("pose").append((clk.now_ns(), rng.normal(0, 0.3, 3), np.zeros(3), np.zeros(7)))
rec.log_phase(clk.now_ns(), 0, 1)
rec.log_press({"press_id": "t1", "batch": "b", "ix": 0, "iy": 0, "status": 1})
rec.close()
with h5py.File(out, "r") as f:
    assert f["skin/t"].shape[0] == 50, f["skin/t"].shape
    assert f["ft/wrench"].shape == (50, 6)
    assert f["pose/joints"].shape == (50, 7)
    assert len(f["events/press_id"]) == 1
    assert f["phases/phase_code"][0] == 1
print("[3] HDF5 写读往返 OK")

# 4) 事件字段一致性（真实 import 路径）
from eflesh_calib import press as P  # noqa: E402
import ast  # noqa: E402
src = open(P.__file__).read()
keys = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "log_press":
        keys = [k.value for k in node.args[0].keys]
cols = [n for n, _ in __import__("eflesh_calib.recorder", fromlist=["x"]).EVENT_COLS]
missing = set(keys) - set(cols)
assert not missing, f"recorder 缺列: {missing}"
print("[4] 事件字段一致性 OK")

# 5) 训练侧（torch）
try:
    import torch  # noqa: E402
    from train.models import MLP  # noqa: E402
    m = MLP(15, 4)
    y = m(torch.zeros(2, 15))
    assert y.shape == (2, 4)
    print(f"[5] torch {torch.__version__} + MLP OK (cuda={torch.cuda.is_available()})")
except ImportError as e:
    print(f"[5] torch 跳过: {e}")

print("\n=== Orin 自检全部通过 ===")
time.sleep(0.1)
