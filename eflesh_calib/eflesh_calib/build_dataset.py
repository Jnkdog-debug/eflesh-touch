#!/usr/bin/env python3
"""
数据管线 —— HDF5 会话 → 训练 npz。

每压一条样本（hold 稳定窗均值）:
  X  = ΔB (15,)   = B − B0（接触前窗口逐通道中位数）
  y  = [x_mm, y_mm, z_indent_mm, Fz, (Fx,Fy,Mx,My,Mz 可选)]
       位置用 touch-off 实测投影；Fz 为 FT 插值到 hold 窗中点
  附: press_id / batch / ix / iy / checkerboard split

切分: 棋盘格 (ix+iy)%2 空间 80/20 + 按批留出，绝不逐帧随机切。

Usage:
  python pipeline/build_dataset.py --h5 "data/batch_*_batch00*.h5" --out data/ds_v1.npz
  python pipeline/build_dataset.py --h5 data/x.h5 --frames   # 逐帧模式（hold 内每帧一条）
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from eflesh_calib import config
from eflesh_calib.util import interp_to

_R = np.array(config.FT_R_TO_SKIN, dtype=float)
_T = np.array(config.FT_LEVER_SENSOR_M, dtype=float)


def wrench_to_skin(w):
    """传感器系(tared) 6 维力 → 皮肤系接触点 6 维力。
    力: F_sk = R·F_s（R 经 4 方向拖动 + 法向下压验证）；
    力矩: 先扣杆臂 M_contact = M_s − t×F_s，再 R 映射。
    竖直杆臂不污染 Mz；t 的横向分量对 Mz 的影响已含在 t×F 内。"""
    w = np.asarray(w, dtype=float)
    F = _R @ w[:3]
    M = _R @ (w[3:] - np.cross(_T, w[:3]))
    return np.concatenate([F, M])


def build(paths: list[str], frame_mode: bool = False, full_force: bool = False):
    X, Y, PIDS, BATCH, IX, IY, TS = [], [], [], [], [], [], []
    for path in paths:
        with h5py.File(path, "r") as h5:
            if "events/press_id" not in h5:
                print(f"{Path(path).name}: 无 events（中断/未正常收尾的会话）→ 跳过")
                continue
            ts = h5["skin/t"][:]
            B = h5["skin/B"][:].reshape(len(ts), -1)
            tf = h5["ft/t"][:]
            W = h5["ft/wrench"][:]
            ev = {k: h5[f"events/{k}"][:] for k in (
                "press_id", "batch", "ix", "iy", "status",
                "x_meas_mm", "y_meas_mm", "z_hold_meas_mm",
                "t_above", "t_descend", "t_hold_start", "t_hold_end")}
            # 剪切/扭转批新增列（老文件没有 → NaN 填充，位置标签退回 touch-off 投影）
            for k in ("x_hold_meas_mm", "y_hold_meas_mm",
                      "drag_dx_cmd_mm", "drag_dy_cmd_mm", "twist_cmd_deg"):
                ev[k] = (h5[f"events/{k}"][:] if f"events/{k}" in h5
                         else np.full(len(ev["status"]), np.nan, dtype="f4"))
            n = len(ev["status"])
            n_used = 0
            for i in range(n):
                if int(ev["status"][i]) != 1:
                    continue
                t_a, t_d = ev["t_above"][i], ev["t_descend"][i]
                hs, he = ev["t_hold_start"][i], ev["t_hold_end"][i]
                if not all(np.isfinite(v) for v in (t_a, t_d, hs, he)):
                    continue
                mb = (ts >= t_a + 0.2) & (ts <= t_d - 0.05)
                mh = (ts >= hs + 0.2) & (ts <= he)
                if mb.sum() < 5 or mh.sum() < 5:
                    continue
                B0 = np.median(B[mb], axis=0)
                dB = B[mh] - B0
                # FT 插值到皮肤帧时刻（或窗中点）
                if frame_mode:
                    t_q = ts[mh]
                    Wq = interp_to(t_q, tf, W.astype(np.float64))
                    for dB_k, t_k, W_k in zip(dB, t_q, Wq):
                        X.append(dB_k.astype(np.float32))
                        Y.append(_labels(ev, i, W_k, full_force))
                        TS.append(t_k)
                        n_used += 1
                else:
                    t_mid = float(ts[mh].mean())
                    Wq = interp_to(np.array([t_mid]), tf, W.astype(np.float64))[0]
                    X.append(np.mean(dB, axis=0).astype(np.float32))
                    Y.append(_labels(ev, i, Wq, full_force))
                    TS.append(t_mid)
                    n_used += 1
                pid = ev["press_id"][i]
                pid = pid.decode() if isinstance(pid, bytes) else str(pid)
                bat = ev["batch"][i]
                bat = bat.decode() if isinstance(bat, bytes) else str(bat)
                reps = int(mh.sum()) if frame_mode else 1
                PIDS.extend([pid] * reps)
                BATCH.extend([bat] * reps)
                IX.extend([int(ev["ix"][i])] * reps)
                IY.extend([int(ev["iy"][i])] * reps)
            print(f"{Path(path).name}: {n} presses → {n_used} samples")

    X = np.stack(X)
    Y = np.stack(Y)
    ix = np.array(IX, dtype=np.int16)
    iy = np.array(IY, dtype=np.int16)
    checker = (ix + iy) % 2 == 0   # True = 训练格
    return dict(X=X, y=Y,
                press_id=np.array(PIDS), batch=np.array(BATCH),
                ix=ix, iy=iy, t=np.array(TS),
                checkerboard_train=checker)


def _labels(ev, i, wrench, full_force: bool):
    """位置标签: 剪切/扭转批优先用 hold 末端实测投影（拖动终点/扭转时接触点），
    法向批/老文件无此字段 → touch-off 实测投影（历史行为不变）。"""
    xh, yh = ev["x_hold_meas_mm"][i], ev["y_hold_meas_mm"][i]
    x = float(xh) if np.isfinite(xh) else float(ev["x_meas_mm"][i])
    y = float(yh) if np.isfinite(yh) else float(ev["y_meas_mm"][i])
    lab = [x, y, float(ev["z_hold_meas_mm"][i]), float(wrench[2])]
    if full_force:
        w_sk = wrench_to_skin(wrench)
        lab += [float(v) for v in w_sk[[0, 1, 3, 4, 5]]]   # Fx,Fy,Mx,My,Mz（Fz 已在第 4 列，去重）
    return np.array(lab, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="glob，如 'data/batch_*.h5'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", action="store_true", help="hold 内逐帧（默认窗均值）")
    ap.add_argument("--full-force", action="store_true", help="标签含完整 6 维力")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.h5))
    if not paths:
        sys.exit(f"无匹配文件: {args.h5}")
    ds = build(paths, frame_mode=args.frames, full_force=args.full_force)
    if args.full_force:
        print("6 维力已变换到皮肤系接触点 (R + 杆臂 t, 见 config.FT_R_TO_SKIN)")

    np.savez_compressed(args.out, **ds)
    n = len(ds["X"])
    print(f"\n输出: {args.out}")
    print(f"样本 {n}  X{ds['X'].shape}  y{ds['y'].shape}")
    print(f"棋盘格切分: train={int(ds['checkerboard_train'].sum())} "
          f"val={int((~ds['checkerboard_train']).sum())}")
    ok = np.isfinite(ds["y"]).all(axis=1)
    print(f"标签有效行: {int(ok.sum())}/{n}")


if __name__ == "__main__":
    main()
