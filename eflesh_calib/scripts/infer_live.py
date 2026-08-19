#!/usr/bin/env python3
"""
实时推理 —— 皮肤帧 → 模型 → (x, y, z, Fz)，纯 CPU。

原理:
  ΔB = B − B₀（B₀ 在线维护：开机采 3s 无接触中位数；
  之后连续 2s 无接触自动刷新 —— 温漂/蠕变在线零点）
  x = (ΔB − mean)/std → MLP → ŷ·std + mean

Usage (Orin, 皮肤 USB 在本机；artifact 从训练机拷来):
  python scripts/infer_live.py --artifact data/ds_v1f_mlp.pt
  python scripts/infer_live.py --artifact ... --port /dev/ttyUSB1
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from eflesh_calib import config
from eflesh_calib.skin_stream import SkinStats, parse_frame  # noqa: F402
from eflesh_calib.skin_stream import skin_reader
from eflesh_calib.util import Latest


def load_model(artifact_path: str):
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)
    from train.models import MLP
    x_mean = np.asarray(art["x_mean"], dtype=np.float64)
    model = MLP(len(x_mean), art["out_dim"])
    model.load_state_dict(art["state_dict"])
    model.eval()
    return (model,
            x_mean, np.asarray(art["x_std"], dtype=np.float64),
            np.asarray(art["y_mean"], dtype=np.float64),
            np.asarray(art["y_std"], dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--port", default=config.SKIN_PORT)
    ap.add_argument("--baud", type=int, default=config.SKIN_BAUD)
    ap.add_argument("--init-s", type=float, default=3.0, help="开机基线采集时长")
    args = ap.parse_args()

    model, x_mean, x_std, y_mean, y_std = load_model(args.artifact)
    names = ["x", "y", "z", "Fz"] + ["Fx", "Fy", "Mx", "My", "Mz"]
    print(f"模型加载 OK: in={len(x_mean)} out={len(y_mean)}")

    q: deque = deque(maxlen=2000)
    latest = Latest()
    stats = SkinStats()
    stop = threading.Event()
    th = threading.Thread(target=skin_reader,
                          args=(args.port, args.baud, q, stats, latest, stop),
                          daemon=True)
    th.start()

    # ---- 开机基线 ----
    print(f"采 {args.init_s}s 无接触基线（别碰皮肤）…")
    t0 = time.time()
    while time.time() - t0 < args.init_s and len(q) < 300:
        time.sleep(0.05)
    frames = list(q.copy())
    B0 = np.median(np.stack([f[2] for f in frames]).reshape(len(frames), -1), axis=0)
    print(f"基线就绪（{len(frames)} 帧）。开始推理 —— 按压皮肤试试。Ctrl+C 退出\n")

    calm_since = None   # 无接触起始时刻（用于在线刷新基线）
    t_last_print = 0.0
    try:
        while True:
            v = latest.get()
            if v is None:
                time.sleep(0.005)
                continue
            _, B = v
            b = B.ravel().astype(np.float64)
            db = b - B0
            mag = float(np.linalg.norm(db))

            # 在线零点：持续无接触 → 慢慢刷新基线
            now = time.time()
            if mag < 20.0:                      # µT，视为无接触
                if calm_since is None:
                    calm_since = now
                elif now - calm_since > 2.0:
                    B0 = 0.98 * B0 + 0.02 * b   # 慢 EMA，不跳变
            else:
                calm_since = None

            x = (db - x_mean) / x_std
            with torch.no_grad():
                yn = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            y = yn.numpy()[0] * y_std + y_mean

            if now - t_last_print > 0.1:        # 10Hz 打印（推理本身 184Hz）
                lab = "  <无接触>" if mag < 20.0 else ""
                vals = "  ".join(f"{n}={y[i]:+7.2f}" for i, n in enumerate(names[:len(y)]))
                print(f"\r[{stats.hz():5.1f}Hz] {vals}  ‖ΔB‖={mag:7.0f}µT{lab}   ",
                      end="", flush=True)
                t_last_print = now
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stop.set()
        th.join(timeout=2.0)


if __name__ == "__main__":
    main()
