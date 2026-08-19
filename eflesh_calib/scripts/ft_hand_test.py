#!/usr/bin/env python3
"""
FT 手推测试 —— 实时打印 raw/zero 双字段，用手推探针看有无响应。

Usage:
  python scripts/ft_hand_test.py --arm 192.168.101.120   # 你的臂
  python scripts/ft_hand_test.py --arm 192.168.101.119   # 对照臂（demo 里力是好用的）
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eflesh_calib import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=config.ARM_IP)
    args = ap.parse_args()

    from curobo_bridge.rm75_robot import RM75Robot
    arm = RM75Robot(args.arm, config.ARM_PORT)
    if not arm.connect():
        sys.exit(1)
    print(f"\n用手沿任意方向用力推探针/末端，观察 Δraw 和 Δzero（正常应推动几 N 就有几 N 变化）")
    print("Ctrl+C 结束\n")
    time.sleep(0.5)
    arm.clear_force_data()
    time.sleep(0.3)

    ret, d = arm.robot.rm_get_force_data()
    raw0 = list(d["force_data"]) if isinstance(d, dict) else [0.0] * 6
    zero0 = list(d["zero_force_data"]) if isinstance(d, dict) else [0.0] * 6
    print("分别做: ①水平推 X 向 ②水平推 Y 向 ③向下压 ④拧末端 → 看哪个 Δ 有反应")
    try:
        while True:
            ret, d = arm.robot.rm_get_force_data()
            if ret == 0 and isinstance(d, dict):
                raw = list(d["force_data"])
                zero = list(d["zero_force_data"])
                dr = [raw[i] - raw0[i] for i in range(6)]
                dz = [zero[i] - zero0[i] for i in range(6)]
                print(f"\rΔraw[Fx{dr[0]:+7.1f} Fy{dr[1]:+7.1f} Fz{dr[2]:+8.1f} "
                      f"Mx{dr[3]:+6.2f} My{dr[4]:+6.2f} Mz{dr[5]:+6.2f}] "
                      f"Δzero.Fz={dz[2]:+6.2f}   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n完成")
    arm.disconnect()


if __name__ == "__main__":
    main()
