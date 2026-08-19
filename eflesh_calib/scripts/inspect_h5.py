#!/usr/bin/env python3
"""
HDF5 会话文件检查 —— 流统计 + 快图（掉帧/速率/Fz vs ‖ΔB‖ 回放）。

Usage:
  python scripts/inspect_h5.py data/batch_xxx.h5            # 文本统计
  python scripts/inspect_h5.py data/batch_xxx.h5 --plot out.png
  python scripts/inspect_h5.py data/batch_xxx.h5 --press 12  # 单压回放图
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from eflesh_calib.recorder import PHASE_NAMES, STATUS_NAMES


def load(h5_path):
    h5 = h5py.File(h5_path, "r")
    return h5


def text_report(h5, path):
    print(f"===== {path} =====")
    for k in ("session", "created_utc", "grid_pitch_mm", "depths_mm",
              "thr_touch_N", "skin_lost", "skin_badck", "skin_resync",
              "skin_mean_hz", "air_gap_mm"):
        if k in h5.attrs:
            print(f"{k:18s}: {h5.attrs[k]}")

    for stream in ("skin", "ft", "pose"):
        if stream in h5:
            t = h5[f"{stream}/t"][:]
            dur = t[-1] - t[0] if len(t) > 1 else 0
            hz = (len(t) - 1) / dur if dur > 0 else 0
            gaps = np.diff(t)
            print(f"{stream:5s}: {len(t):7d} 帧  {hz:6.1f}Hz  "
                  f"maxgap={gaps.max()*1e3 if len(gaps) else 0:6.1f}ms")

    if "events/press_id" in h5:
        ids = [s.decode() for s in h5["events/press_id"][:]]
        st = h5["events/status"][:]
        fz = h5["events/fz_hold_mean_N"][:]
        print(f"\npresses: {len(ids)}")
        for name, code in STATUS_NAMES.items():
            n = int((st == name).sum())
            if n:
                print(f"  {code:16s}: {n}")
        ok = np.isfinite(fz)
        if ok.any():
            print(f"Fz(hold) 范围: {fz[ok].min():+.2f} ~ {fz[ok].max():+.2f} N  "
                  f"mean={fz[ok].mean():+.2f}N")
        # 各深度 Fz 分布（深度-力关系 sanity）
        d = h5["events/depth_cmd_mm"][:]
        for dv in np.unique(d):
            m = (d == dv) & ok
            if m.any():
                print(f"  depth {dv:.1f}mm: Fz mean={fz[m].mean():+.2f}N "
                      f"std={fz[m].std():.2f} n={int(m.sum())}")
    if "phases/t" in h5:
        pc = h5["phases/phase_code"][:]
        print(f"\nphases: {len(pc)} 条", {PHASE_NAMES[c]: int((pc == c).sum())
                                          for c in np.unique(pc)})


def plot_overview(h5, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ts = h5["skin/t"][:]
    B = h5["skin/B"][:].reshape(len(ts), -1)
    axes[0].plot(ts, B, lw=0.4)
    axes[0].set_ylabel("skin B (µT)")

    tf = h5["ft/t"][:]
    W = h5["ft/wrench"][:]
    axes[1].plot(tf, W[:, :3], lw=0.5)
    axes[1].set_ylabel("FT force (N)")

    tp = h5["pose/t"][:]
    z = h5["pose/xyz"][:][:, 2]
    axes[2].plot(tp, z * 1e3, lw=0.5)
    axes[2].set_ylabel("TCP z (mm)")
    axes[2].set_xlabel("t (s, session)")
    for ax in axes:
        for t in h5["phases/t"][:][h5["phases/phase_code"][:] == 4]:
            ax.axvline(t, color="r", alpha=0.15, lw=0.5)
    fig.suptitle("session overview (red = touch-off)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"图: {out_png}")


def plot_press(h5, idx: int, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev_ids = [s.decode() for s in h5["events/press_id"][:]]
    hs, he = h5["events/t_hold_start"][idx], h5["events/t_hold_end"][idx]
    lo, hi = hs - 3.0, he + 1.0
    ts = h5["skin/t"][:]
    m = (ts >= lo) & (ts <= hi)
    B = h5["skin/B"][:][m].reshape(-1, 15)
    tf = h5["ft/t"][:]
    mf = (tf >= lo) & (tf <= hi)
    fz = h5["ft/wrench"][:][mf, 2]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a1.plot(ts[m], B, lw=0.6)
    a1.axvline(hs, color="g", ls="--", label="hold start")
    a1.axvline(he, color="r", ls="--", label="hold end")
    a1.legend(); a1.set_ylabel("B (µT)")
    a2.plot(tf[mf], fz, lw=0.8, color="k")
    a2.axhline(0, color="gray", lw=0.4)
    a2.set_ylabel("Fz (N)"); a2.set_xlabel("t (s)")
    fig.suptitle(f"press {ev_ids[idx]}  depth={h5['events/depth_cmd_mm'][idx]}mm")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"图: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--press", type=int, default=None, help="第 N 压（0-based）回放图")
    args = ap.parse_args()

    h5 = load(args.h5)
    text_report(h5, args.h5)
    if args.press is not None:
        out = args.plot or f"press_{args.press}.png"
        plot_press(h5, args.press, out)
    elif args.plot:
        plot_overview(h5, args.plot)
    h5.close()


if __name__ == "__main__":
    main()
