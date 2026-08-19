#!/usr/bin/env python3
"""
合成数据端到端测试 —— 无硬件验证整条软件链:
  模拟皮肤/FT/位姿三流 + 按压事件 → HDF5 → build_dataset → sync_check → ridge 训练。

模拟规律（供标签可学）:
  ΔB[0..2] ∝ (x, y, Fz)，ΔB[3..14] 为噪声；FT Fz = k × depth × (1 + 位置梯度)。
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from eflesh_calib.clock import SessionClock
from eflesh_calib.recorder import (PH_APPROACH, PH_DESCEND, PH_HOLD, PH_PRESS,
                                   PH_RETRACT, PH_TARE, PH_TOUCHOFF, ST_OK)

PROJ = Path(__file__).resolve().parent.parent
OUT = Path("/tmp/synth_e2e.h5")
rng = np.random.default_rng(42)

clk = SessionClock()
from eflesh_calib.recorder import Hdf5Recorder  # noqa: E402
rec = Hdf5Recorder(OUT, clk, {"session": "synth", "thr_touch_N": 0.5,
                              "calib_json": "synth"})


def rel(t_ns):
    return clk.to_rel(t_ns)


N_PRESS = 30
idx_xy = [(rng.integers(-2, 3), rng.integers(-2, 3)) for _ in range(N_PRESS)]
T_BASE = clk.now_ns()          # 虚拟时间轴：每压间隔 5s，不重叠（贴近真实节律）
for i, (ix, iy) in enumerate(idx_xy):
    gx, gy = ix * 5.0, iy * 5.0
    depth = float(rng.choice([0.5, 1.0, 1.5]))
    t0 = T_BASE + int(i * 6e9)
    rec.log_phase(t0, i, PH_APPROACH)
    # 真实节律: approach 0.5s → tare → 基线段 → descend 1s → touchoff → press → hold 0.8s → retract
    t_a = t0 + 0.5e9
    rec.log_phase(t0 + 0.52e9, i, PH_TARE)
    t_d = t0 + 2.0e9
    rec.log_phase(t_d, i, PH_DESCEND)
    t_c = t0 + 3.0e9
    rec.log_phase(t_c, i, PH_TOUCHOFF)
    rec.log_phase(t0 + 3.3e9, i, PH_PRESS)
    t_hs = t0 + 3.8e9
    t_he = t0 + 4.6e9
    rec.log_phase(t_hs, i, PH_HOLD)
    rec.log_phase(t_he, i, PH_RETRACT)
    t_r = t0 + 5.0e9

    fz_true = 8.0 * depth * (1 + 0.02 * gx)

    # 皮肤流（34Hz 模拟，窗口逻辑与 170Hz 一致）
    n_skin = int(5.0 * 34)
    for k in range(n_skin):
        t = t0 + (k / n_skin) * 5.0e9
        contact = t > t_c
        holding = t_hs < t < t_he
        B = np.zeros((5, 3), dtype="f4")
        if contact:
            amp = fz_true if holding else fz_true * 0.6
            B[0] = [gx * 5, gy * 5, amp]
        B += rng.normal(0, 0.3, (5, 3)).astype("f4")   # MLX90393 实测噪声 ~0.1-0.5µT
        rec.q("skin").append((t, k, B))
    # FT 流
    n_ft = int(5.0 * 50)
    for k in range(n_ft):
        t = t0 + (k / n_ft) * 5.0e9
        fz = fz_true if t > t_c + 0.01e9 else 0.0
        w = np.array([0, 0, fz + rng.normal(0, 0.02), 0, 0, 0], dtype="f4")
        rec.q("ft").append((t, w, 0))
    # 位姿流
    n_p = int(5.0 * 40)
    for k in range(n_p):
        t = t0 + (k / n_p) * 5.0e9
        z = 0.215 - (0.005 if t < t_c else 0.005 + depth * 1e-3 * min(1, (t - t_c) / 0.3e9))
        rec.q("pose").append((t, np.array([0.55 + gx * 1e-3, -0.1 + gy * 1e-3, z]),
                             np.zeros(3), np.zeros(7)))

    rec.log_press({
        "press_id": f"{ix:+03d}{iy:+03d}_d{depth:.1f}", "batch": "synth",
        "ix": int(ix), "iy": int(iy), "gx_mm": gx, "gy_mm": gy,
        "x_meas_mm": gx + rng.normal(0, 0.05), "y_meas_mm": gy + rng.normal(0, 0.05),
        "depth_cmd_mm": depth, "z_zero_skin_mm": 0.0, "z_hold_meas_mm": depth,
        "t_tare": rel(t0 + 0.52e9), "t_above": rel(t_a), "t_descend": rel(t_d),
        "t_touchoff": rel(t_c), "t_press": rel(t0 + 3.3e9),
        "t_hold_start": rel(t_hs), "t_hold_end": rel(t_he), "t_retract": rel(t_r),
        "fz_touchoff_N": 0.5, "fz_hold_mean_N": fz_true, "fz_peak_N": fz_true,
        "status": ST_OK,
    })

rec.close(extra_attrs={"skin_lost": 0, "skin_badck": 0, "skin_mean_hz": 34.0})
print(f"synthetic HDF5: {OUT}")

# ---- build_dataset ----
r = subprocess.run([sys.executable, str(PROJ / "eflesh_calib/build_dataset.py"),
                    "--h5", str(OUT), "--out", "/tmp/synth_ds.npz"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, "build_dataset 失败"

# ---- 验证标签可学: ΔB 与 (x,y,Fz) 线性关系应被 ridge 学到 ----
ds = dict(np.load("/tmp/synth_ds.npz", allow_pickle=True))
from train.models import RidgeMultiOutput  # noqa: E402
X, y = ds["X"].astype(float), ds["y"].astype(float)
m = RidgeMultiOutput().fit(X, y)
pred = m.predict(X)
for c, name in enumerate(("x_mm", "y_mm", "z_mm", "Fz")):
    rmse = float(np.sqrt(np.mean((pred[:, c] - y[:, c]) ** 2)))
    scale = max(float(np.std(y[:, c])), 1e-6)
    print(f"  {name}: RMSE={rmse:.4f} (std={scale:.3f})  R2={1-(rmse**2)/(scale**2):.3f}")
    assert rmse < 0.3 * scale, f"{name} 没学到"

# ---- sync_check（合成数据接触同步，应接近 0 残差）----
r = subprocess.run([sys.executable, str(PROJ / "eflesh_calib/sync_check.py"), str(OUT)],
                   capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, "sync_check 失败"

print("\n=== 合成端到端测试全部通过 ===")
