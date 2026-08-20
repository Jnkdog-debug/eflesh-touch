#!/usr/bin/env python3
"""
深度扫描 —— 找这颗皮肤的磁-弹耦合"门槛深度"。

在几个代表点位各压一遍递增深度，事后从 H5 算每颗芯片的 Σ|ΔB| 与 Fz，
回答:压多深才能让全部区域出 mT 级信号(batch001 的 0.5-1.5mm 只在个别点位有耦合)。

Usage:
  python scripts/depth_sweep.py --calib data/skin_frame_xxx.json
  python scripts/depth_sweep.py --calib ... --points 0,0 -14,10 14,-14 --depths 1,2,3,4,5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eflesh_calib import config
from eflesh_calib.calib_frame import SkinFrame
from eflesh_calib.press import PressMachine
from eflesh_calib.recorder import STATUS_NAMES
from eflesh_calib.runtime import Session
from eflesh_calib.traj import PressTarget


def analyze(h5_path: Path, plan: list[PressTarget]):
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        t = np.asarray(f["skin/t"])
        B = np.asarray(f["skin/B"]).reshape(-1, 5, 3)
        ev = f["events"]
        ft_t = np.asarray(f["ft/t"])
        ft_w = np.asarray(f["ft/wrench"])
        rows = [(ev[k][:]) for k in ("t_above", "t_descend", "t_hold_start", "t_hold_end")]

    print("\n========== 深度扫描结果 ==========")
    print(f"{'点':>10s} {'深度mm':>6s} {'Fz(N)':>6s} " + "".join(f"{'S'+str(s+1):>8s}" for s in range(5)) + f" {'总计µT':>9s}")
    for i, tgt in enumerate(plan):
        ta, td, hs, he = (r[i] for r in rows)
        ia, ib = np.searchsorted(t, [ta + 0.2, td - 0.05])
        ih, ij = np.searchsorted(t, [hs + 0.2, he - 0.1])
        if ib - ia < 5 or ij - ih < 5:
            print(f"({tgt.gx_mm:+5.1f},{tgt.gy_mm:+5.1f}) {tgt.depth_mm:6.1f}   帧不足，跳过")
            continue
        base = B[ia:ib].mean(0)
        dB = np.abs(B[ih:ij].mean(0) - base).sum(1)
        t_mid = 0.5 * (hs + he)
        fz = float(np.interp(t_mid, ft_t, ft_w[:, 2]))
        print(f"({tgt.gx_mm:+5.1f},{tgt.gy_mm:+5.1f}) {tgt.depth_mm:6.1f} {fz:6.2f} "
              + "".join(f"{v:8.0f}" for v in dB) + f" {dB.sum():9.0f}")

    print("\n判读: 找『所有点位总计都 ≥2000µT』的最小深度 = batch002 的起步深度档")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--points", default="0,0 -14,10 14,-14",
                    help="皮肤系坐标 x,y 空格分隔（默认 中心/左上热区/右下死角）")
    ap.add_argument("--depths", default="1,2,3,4,5")
    ap.add_argument("--arm", default=config.ARM_IP)
    args = ap.parse_args()

    sf = SkinFrame.load(args.calib)
    thr = float(sf.meta.get("thr_touch_n", config.TOUCH_THR_N))
    pts = [tuple(float(v) for v in p.split(",")) for p in args.points.split()]
    depths = [float(d) for d in args.depths.split(",")]

    plan = [PressTarget(press_id=f"sweep_x{int(x)}_y{int(y)}_d{d:g}", ix=0, iy=0,
                        gx_mm=x, gy_mm=y, depth_mm=d, batch="sweep")
            for (x, y) in pts for d in depths]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    h5_path = config.DATA_DIR / f"batch_{stamp}_sweep.h5"
    attrs = {"session": "sweep", "calib_json": str(args.calib),
             "depths_mm": depths, "points": pts, "thr_touch_N": thr}

    print(f"点位 {pts}  深度 {depths}  → {len(plan)} 压, 约 {len(plan)*0.35:.0f} 分钟")
    print("!!! 手放急停 !!! 深按压前所未试过,第一下 4mm/5mm 盯着力读数\n")

    counts = {}
    with Session(arm_ip=args.arm, skin=True, record=h5_path, attrs=attrs) as s:
        pm = PressMachine(s, sf, thr_touch_n=thr)
        for i, tgt in enumerate(plan):
            if s.abort_event.is_set():
                print("\n[ABORT] 中止")
                break
            st = pm.run_press(tgt, i)
            counts[STATUS_NAMES.get(st, st)] = counts.get(STATUS_NAMES.get(st, st), 0) + 1
            print(f"[{i+1}/{len(plan)}] {tgt.press_id} → {STATUS_NAMES.get(st, st)}", flush=True)
        if s.skin_stats:
            print("皮肤链路:", s.skin_stats.line())

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"\n数据: {h5_path}")
    analyze(h5_path, plan[:i + 1])


if __name__ == "__main__":
    main()
