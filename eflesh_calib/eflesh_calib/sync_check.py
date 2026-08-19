#!/usr/bin/env python3
"""
跨流对齐验证 —— 无硬件触发时的接触起点互相关。

方法:
  每次按压分别在皮肤流（‖ΔB‖ 首次 > 5× 基线噪声）和 FT 流
  （|Fz| 首次 > touch 阈值）里独立检测接触起点，Δt = t_skin − t_ft。
  串口缓冲延迟近似常值 → 取 median(Δt) 为皮肤延迟补偿，
  验收看补偿后 |Δt − median| 的 p95 < 5ms。

Usage:
  python pipeline/sync_check.py data/batch_xxx.h5 [--calib skin_frame.json]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from eflesh_calib import config


def onset_delta_ts(h5, thr_n: float) -> np.ndarray:
    """
    返回每压的 Δt = t_skin_onset − t_ft_onset（秒），无效压为 nan。

    起点定义: 各自轨迹首次超过「该压最大值的 30%」—— 金属探针接近磁铁时
    会有磁接近效应（接触前就有 ΔB），用绝对阈值会把起点提前 1~2s；
    30% 分位对应压入中段，两路是同一物理时刻。
    """
    ts = h5["skin/t"][:]
    B = h5["skin/B"][:].reshape(len(ts), -1)
    tf = h5["ft/t"][:]
    fz = h5["ft/wrench"][:, 2]
    ev = {k: h5[f"events/{k}"][:] for k in
          ("t_above", "t_descend", "t_touchoff", "status")}
    n = len(ev["t_above"])
    out = np.full(n, np.nan)

    for i in range(n):
        if int(ev["status"][i]) != 1:
            continue
        t_a, t_d, t_c = ev["t_above"][i], ev["t_descend"][i], ev["t_touchoff"][i]
        if not np.isfinite(t_c):
            continue
        mb = (ts >= t_a + 0.2) & (ts <= t_d - 0.05)
        if mb.sum() < 5:
            continue
        B0 = np.median(B[mb], axis=0)
        # 窗口: 下探 → 接触后 0.8s（覆盖压入段）
        mw = (ts >= t_d) & (ts <= t_c + 0.8)
        idx = np.where(mw)[0]
        if len(idx) < 10:
            continue
        db = np.linalg.norm(B[idx] - B0, axis=1)
        if db.max() < 20.0:   # 信号太弱（µT），不可靠
            continue
        thr_skin = 0.3 * db.max()
        hit = np.where(db > thr_skin)[0]
        if len(hit) == 0:
            continue
        t_skin = ts[idx[hit[0]]]
        # FT 同窗口同分位
        mf = (tf >= t_d) & (tf <= t_c + 0.8)
        jf = np.where(mf)[0]
        if len(jf) < 10:
            continue
        fa = np.abs(fz[jf])
        if fa.max() < thr_n:
            continue
        thr_ft = 0.3 * fa.max()
        hitf = np.where(fa > thr_ft)[0]
        if len(hitf) == 0:
            continue
        t_ft = tf[jf[hitf[0]]]
        out[i] = t_skin - t_ft
    return out


def report(h5_path: str, thr_n: float | None = None, verbose: bool = True) -> dict:
    with h5py.File(h5_path, "r") as h5:
        if thr_n is None:
            thr_n = float(h5.attrs.get("thr_touch_N", config.TOUCH_THR_N))
        dt = onset_delta_ts(h5, thr_n)
    valid = dt[np.isfinite(dt)]
    if len(valid) < 5:
        if verbose:
            print(f"有效按压不足（{len(valid)}），无法统计对齐")
        return {"n": len(valid), "pass": False}

    med = float(np.median(valid))
    resid = np.abs(valid - med)
    p95 = float(np.percentile(resid, 95))
    res = {"n": int(len(valid)), "median_offset_ms": med * 1e3,
           "p95_after_comp_ms": p95 * 1e3, "pass": p95 < 5e-3}
    if verbose:
        print(f"有效按压 {res['n']}")
        print(f"皮肤常值延迟（median Δt）: {med*1e3:+.2f} ms → 建议 skin_latency_comp_s = {med:.6f}")
        print(f"补偿后残差 p95: {p95*1e3:.2f} ms  "
              f"{'PASS (<5ms)' if res['pass'] else '*** FAIL — 考虑 GPIO 硬件触发 ***'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--calib", default=None)
    args = ap.parse_args()
    thr = None
    if args.calib:
        import json
        thr = float(json.loads(Path(args.calib).read_text()).get("meta", {}).get("thr_touch_n", 0))
        if thr <= 0:
            thr = None
    report(args.h5, thr_n=thr)


if __name__ == "__main__":
    main()
