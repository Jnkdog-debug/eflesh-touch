#!/usr/bin/env python3
"""
评估 —— 各输出 RMSE / 误差-位置热图 / 误差-力曲线。

Usage:
  python train/evaluate.py --data data/ds_v1.npz --artifact data/ds_v1_mlp.pt
  python train/evaluate.py --data data/ds_v1.npz --model ridge
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from train.models import MLP, RidgeMultiOutput
from train.train_model import TARGET_NAMES, split_data

UNITS = ["mm", "mm", "mm", "N", "N", "N", "N·m", "N·m", "N·m"]   # 兼容旧引用; 表内单位现随 names 推导
# 验收线（采集计划笔记）: x,y ≤ 0.5mm, z ≤ 0.16mm, 力 <5% 量程（按数据实际范围算）
TARGETS = {"x_mm": 0.5, "y_mm": 0.5, "z_mm": 0.16}


def predict_mlp(artifact_path, X):
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)
    model = MLP(X.shape[1], art["out_dim"])
    model.load_state_dict(art["state_dict"])
    model.eval()
    with torch.no_grad():
        xn = torch.tensor((X - np.asarray(art["x_mean"])) / np.asarray(art["x_std"]),
                          dtype=torch.float32)
        pred = model(xn).numpy()
    return pred * np.asarray(art["y_std"]) + np.asarray(art["y_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--artifact", default=None, help=".pt（mlp）")
    ap.add_argument("--model", choices=["ridge", "mlp"], default="mlp")
    ap.add_argument("--out", default="eval_report.png")
    args = ap.parse_args()

    ds = dict(np.load(args.data, allow_pickle=True))
    Xtr, ytr, Xva, yva = split_data(ds)
    print(f"val 样本 {len(Xva)}")

    if args.model == "ridge":
        pred = RidgeMultiOutput().fit(Xtr, ytr).predict(Xva)
    else:
        assert args.artifact, "mlp 需要 --artifact"
        pred = predict_mlp(args.artifact, Xva)

    extra = [n for n in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")][:max(0, yva.shape[1] - 4)]
    names = TARGET_NAMES + extra
    units = ["mm", "mm", "mm", "N"] + ["N·m" if n.startswith("M") else "N" for n in extra]
    err = pred - yva

    print("\n===== RMSE（验证集） =====")
    for c in range(yva.shape[1]):
        rmse = float(np.sqrt(np.mean(err[:, c] ** 2)))
        r2 = 1 - np.sum(err[:, c] ** 2) / max(np.sum((yva[:, c] - yva[:, c].mean()) ** 2), 1e-9)
        line = f"  {names[c]:5s}: RMSE={rmse:8.4f} {units[c]:3s}  R²={r2:.3f}"
        if names[c] in TARGETS:
            line += f"   目标≤{TARGETS[names[c]]}  {'PASS' if rmse <= TARGETS[names[c]] else 'MISS'}"
        print(line)

    # ---- 图: 误差热图 + 误差-力曲线 + 散点 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # 位置误差热图（网格化）
    pe = np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2) if yva.shape[1] >= 2 else np.abs(err[:, 0])
    mask_va = (~ds["checkerboard_train"] if "checkerboard_train" in ds
               else np.ones(len(ds["ix"]), dtype=bool))
    hx, hy = ds["ix"][mask_va], ds["iy"][mask_va]
    sc = axes[0].scatter(hx, hy, c=pe, cmap="hot", s=90, marker="s")
    axes[0].set_title("position error (mm) over grid")
    axes[0].set_xlabel("ix"); axes[0].set_ylabel("iy")
    plt.colorbar(sc, ax=axes[0])
    # 误差 vs 力幅值
    if yva.shape[1] >= 4:
        f_abs = np.abs(yva[:, 3])
        axes[1].scatter(f_abs, np.abs(err[:, 3]), s=6, alpha=0.5)
        axes[1].set_xlabel("|Fz| true (N)"); axes[1].set_ylabel("|Fz err| (N)")
        axes[1].set_title("force error vs magnitude")
    # x 预测 vs 真值散点
    axes[2].scatter(yva[:, 0], pred[:, 0], s=6, alpha=0.5)
    lim = [min(yva[:, 0].min(), pred[:, 0].min()), max(yva[:, 0].max(), pred[:, 0].max())]
    axes[2].plot(lim, lim, "r--", lw=1)
    axes[2].set_xlabel("x true (mm)"); axes[2].set_ylabel("x pred (mm)")
    axes[2].set_title("x: pred vs true")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\n图: {args.out}")


if __name__ == "__main__":
    main()
