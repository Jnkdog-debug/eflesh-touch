#!/usr/bin/env python3
"""
FT 双字段对比诊断 —— 臂自动下探压皮肤，同时打印:
  raw  = force_data        (原始)
  zero = zero_force_data   (控制器"系统外受力")

判定:
  raw 变化大 + zero 不动  → 控制器零力路径坏 → 改用 raw + 软件去皮（我来改）
  raw 也不动              → 传感器/硬件问题，另查
  zero 正常响应           → 阈值问题

Usage:
  python scripts/ft_probe_test.py --calib data/skin_frame_*.json [--start -8] [--end -18]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eflesh_calib import config
from eflesh_calib.calib_frame import SkinFrame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--start", type=float, default=-6.0, help="起始 z_skin (mm)")
    ap.add_argument("--end", type=float, default=-18.0, help="结束 z_skin (mm)")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--arm", default=config.ARM_IP)
    args = ap.parse_args()

    from curobo_bridge.rm75_robot import RM75Robot
    arm = RM75Robot(args.arm, config.ARM_PORT)
    if not arm.connect():
        sys.exit(1)
    sf = SkinFrame.load(args.calib)

    def pose_at(z_mm):
        p = sf.skin_to_base(0.0, 0.0, z_mm)
        return [float(p[0]), float(p[1]), float(p[2])] + list(sf.rpy_probe)

    def read_both():
        ret, d = arm.robot.rm_get_force_data()
        if ret != 0 or not isinstance(d, dict):
            return None, None
        return d["force_data"][2], d["zero_force_data"][2]

    print("移到中心上方 5mm …")
    arm.move_pose(pose_at(config.HOVER_MM), speed=config.SPEED_TRANSIT, block=1)
    time.sleep(0.5)
    arm.clear_force_data()
    time.sleep(0.3)
    raw0, zero0 = read_both()
    print(f"去皮参考: raw={raw0:+.2f}  zero={zero0:+.3f}")
    print(f"\n{'z_skin(mm)':>10} {'raw Fz':>10} {'Δraw':>8} {'zero Fz':>9} {'Δzero':>8}")

    z = args.start
    try:
        while z >= args.end - 1e-6:
            arm.move_pose(pose_at(z), speed=config.SPEED_DESCEND, block=1)
            time.sleep(0.15)
            raw, zero = read_both()
            if raw is None:
                print(f"{z:+10.1f}  读取失败")
            else:
                print(f"{z:+10.1f} {raw:+10.2f} {raw-raw0:+8.2f} "
                      f"{zero:+9.3f} {zero-zero0:+8.3f}", flush=True)
            z -= args.step
    except KeyboardInterrupt:
        print("\n中断，撤退")
    finally:
        print("\n撤回到上方 …")
        arm.move_pose(pose_at(config.HOVER_MM), speed=config.SPEED_TRANSIT, block=1)
        arm.disconnect()


if __name__ == "__main__":
    main()
