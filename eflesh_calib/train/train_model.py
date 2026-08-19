#!/usr/bin/env python3
"""
训练 CLI —— Adam 1e-3 / 1000 epoch / MSE，z-score 统计只用训练折。
artifact 兼容官方 characterization/train.py 的 .pt 约定:
  {state_dict, mode, out_dim, x_mean, x_std, y_mean, y_std}

Usage:
  python train/train_model.py --data data/ds_v1.npz --model ridge
  python train/train_model.py --data data/ds_v1.npz --model mlp --epochs 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from train.models import MLP, RidgeMultiOutput

TARGET_NAMES = ["x_mm", "y_mm", "z_mm", "Fz"]


def split_data(ds: dict, split: str = "checkerboard"):
    X, y = ds["X"].astype(np.float32), ds["y"].astype(np.float32)
    if split == "checkerboard" and "checkerboard_train" in ds:
        m = ds["checkerboard_train"]
    else:
        m = np.ones(len(X), dtype=bool)
        m[::5] = False   # 无棋盘格信息时退化为按 press 隔 5 取 1（不是逐帧随机）
    ok = np.isfinite(y).all(axis=1) & np.isfinite(X).all(axis=1)
    return X[m & ok], y[m & ok], X[~m & ok], y[~m & ok]


def train_mlp(Xtr, ytr, Xva, yva, epochs=1000, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mean, y_std = ytr.mean(0), ytr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - x_mean) / x_std, device=device)
    yn = torch.tensor((ytr - y_mean) / y_std, device=device)
    Xv = torch.tensor((Xva - x_mean) / x_std, device=device)
    yvn = torch.tensor((yva - y_mean) / y_std, device=device)

    model = MLP(Xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    ds = torch.utils.data.TensorDataset(Xn, yn)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    import time as _t
    t0 = _t.time()
    best_val, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = lossf(model(Xv), yvn).item()
        if val < best_val:
            best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 20 == 0:
            el = _t.time() - t0
            eta = el / (ep + 1) * (epochs - ep - 1)
            print(f"  epoch {ep+1}/{epochs}: val={val:.4f} "
                  f"({el:.0f}s, ETA {eta/60:.1f}min)", flush=True)

    model.load_state_dict(best_state)
    artifact = {"state_dict": model.state_dict(), "mode": "mlp",
                "out_dim": ytr.shape[1],
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std}
    return model, artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["ridge", "mlp"], default="mlp")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="默认自动；Orin 首次 CUDA 有 JIT 编译延迟时可强制 cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = dict(np.load(args.data, allow_pickle=True))
    Xtr, ytr, Xva, yva = split_data(ds)
    print(f"train {len(Xtr)}  val {len(Xva)}  in {Xtr.shape[1]}  out {ytr.shape[1]}")

    stem = Path(args.data).stem
    if args.model == "ridge":
        m = RidgeMultiOutput().fit(Xtr, ytr)
        pred = m.predict(Xva)
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}  （首个 epoch 前如有 JIT 编译可能等待数分钟）")
        model, artifact = train_mlp(Xtr, ytr, Xva, yva, args.epochs, args.lr,
                                    args.seed, device)
        out = args.out or str(Path(args.data).parent / f"{stem}_mlp.pt")
        torch.save(artifact, out)                    # 先保存——指标崩了也不丢模型
        print(f"artifact: {out}")
        with torch.no_grad():
            xn = torch.tensor((Xva - np.asarray(artifact["x_mean"])) /
                              np.asarray(artifact["x_std"]), device=device)
            pred = model(xn).cpu().numpy()
            pred = pred * np.asarray(artifact["y_std"]) + np.asarray(artifact["y_mean"])

    names = TARGET_NAMES + [f"F{i}" for i in ("x", "y")] + ["Mx", "My", "Mz"]
    print("\n===== 验证集 RMSE =====")
    for c in range(yva.shape[1]):
        rmse = float(np.sqrt(np.mean((pred[:, c] - yva[:, c]) ** 2)))
        print(f"  {names[c] if c < len(names) else f'y{c}'}: {rmse:.4f}")


if __name__ == "__main__":
    main()
