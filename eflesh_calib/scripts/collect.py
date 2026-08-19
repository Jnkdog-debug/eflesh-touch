#!/usr/bin/env python3
"""
主采集脚本 —— 网格 × 深度按压 + 三流 HDF5 录制。

Usage:
  python scripts/collect.py --session m1_trial --calib data/skin_frame_xxx.json \
      --grid 3 --depths 1.0                 # M1 试采 3×3
  python scripts/collect.py --session batch001 --calib ... \
      --pitch 5 --depths 0.5,1.0,1.5 --resume   # 正式 9×9×3
  python scripts/collect.py --session dry1 --calib ... --air-gap 5   # 空中演练
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
from eflesh_calib.traj import make_grid, press_plan


def collect_done_ids(data_dir: Path, session: str) -> set[str]:
    """扫同 session 前次文件里 status=ok 的 press_id（--resume 用）."""
    done = set()
    for f in sorted(data_dir.glob(f"batch_*_{session}.h5")):
        import h5py
        try:
            with h5py.File(f, "r") as h5:
                if "events/press_id" not in h5:
                    continue
                ids = [s.decode() for s in h5["events/press_id"][:]]
                sts = h5["events/status"][:]
                done |= {i for i, s in zip(ids, sts) if int(s) == 1}
        except OSError:
            continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--calib", required=True, help="skin_frame JSON")
    ap.add_argument("--pitch", type=float, default=config.GRID_PITCH_MM)
    ap.add_argument("--margin-mm", type=float, default=0.5,
                    help="网格相对示教边缘的内缩量 mm（默认 0.5）")
    ap.add_argument("--grid", type=int, default=None, help="N×N 子集（如 3）")
    ap.add_argument("--depths", default=",".join(str(d) for d in config.DEPTHS_MM))
    ap.add_argument("--order", default="random", choices=["random", "serpentine"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", default=None, help="批名（默认 session 名）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--air-gap", type=float, default=None, help=">0 = 空中演练模式")
    ap.add_argument("--arm", default=config.ARM_IP)
    ap.add_argument("--limit", type=int, default=None, help="最多压 N 次（调试）")
    args = ap.parse_args()

    sf = SkinFrame.load(args.calib)
    thr = float(sf.meta.get("thr_touch_n", config.TOUCH_THR_N))
    depths = [float(x) for x in args.depths.split(",")]

    # 网格范围取示教实际半幅宽（旧 JSON 无此字段则退回配置默认 ±20mm）
    half_x = float(sf.meta.get("half_x_mm", config.GRID_HALF_MM))
    half_y = float(sf.meta.get("half_y_mm", config.GRID_HALF_MM))
    half = min(half_x, half_y) - args.margin_mm

    grid = make_grid(pitch_mm=args.pitch, half_mm=half, subset_n=args.grid)
    plan = press_plan(grid=grid, depths_mm=depths,
                      batch=args.batch or args.session,
                      order=args.order, seed=args.seed)

    if args.resume:
        done = collect_done_ids(config.DATA_DIR, args.session)
        before = len(plan)
        plan = [t for t in plan if t.press_id not in done]
        print(f"[resume] 跳过已完成 {before - len(plan)}/{before}")
    if args.limit:
        plan = plan[:args.limit]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    h5_path = config.DATA_DIR / f"batch_{stamp}_{args.session}.h5"
    attrs = {
        "arm_ip": args.arm, "skin_port": config.SKIN_PORT, "skin_baud": config.SKIN_BAUD,
        "calib_json": str(args.calib),
        "session": args.session, "grid_pitch_mm": args.pitch,
        "grid_n": args.grid or "full", "depths_mm": depths,
        "order": args.order, "thr_touch_N": thr,
        "hold_s": config.HOLD_S, "air_gap_mm": args.air_gap if args.air_gap else 0.0,
        "abort_fz_N": config.ABORT_FZ_N, "abort_f_any_N": config.ABORT_F_ANY_N,
    }

    print(f"网格范围 ±{half:.1f}mm (示教半幅 x±{half_x:.1f} y±{half_y:.1f}, 内缩 {args.margin_mm}mm)")
    print(f"计划按压 {len(plan)} 次 → {h5_path}")
    print(f"touch 阈值 {thr:.2f}N  深度 {depths}  顺序 {args.order}")

    counts = {}
    with Session(arm_ip=args.arm, skin=True, record=h5_path, attrs=attrs) as s:
        pm = PressMachine(s, sf, thr_touch_n=thr)
        t0 = time.time()
        for i, target in enumerate(plan):
            if s.abort_event.is_set():
                print("\n[ABORT] 手动/安全中止 — 停止采集")
                break
            st = pm.run_press(target, i, air_gap_mm=args.air_gap)
            counts[STATUS_NAMES.get(st, st)] = counts.get(STATUS_NAMES.get(st, st), 0) + 1
            el = time.time() - t0
            eta = el / (i + 1) * (len(plan) - i - 1)
            print(f"[{i+1}/{len(plan)}] {target.press_id} → {STATUS_NAMES.get(st, st)}  "
                  f"({el/60:.1f}min, ETA {eta/60:.1f}min)", flush=True)
        print("\n===== 会话统计 =====")
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        if s.skin_stats:
            print("皮肤链路:", s.skin_stats.line())

    print(f"\n数据文件: {h5_path}")


if __name__ == "__main__":
    main()
