#!/usr/bin/env python3
"""
示教皮肤平面 —— 轻触 4 角 + 中心，程序读位姿拟合存 JSON。

推荐用示教器移动机械臂（本程序只负责读位姿标记，不关心你怎么移的），
键盘点动是备用手段。

按键:
  （示教器移动后）m  标记当前点（顺序 TL → TR → BR → BL → C）
  x/X y/Y z/Z        基系 ±x ±y ±z 移动一步（大写为负向，备用）
  1                  步长循环 5mm → 0.5mm → 0.1mm
  p                  打印当前位姿 + Fz（辅助轻触判断）
  u                  撤销上一个标记
  q                  拟合 + 校验 + 保存退出
  s                  急停（arm.stop）

Usage:
  python scripts/teach_plane.py [--arm 192.168.101.120] [--side 40]
"""

import argparse
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eflesh_calib import config
from eflesh_calib.calib_frame import CORNER_NAMES, build_skin_frame

STEPS_MM = [5.0, 0.5, 0.1]
KEYS_BASE = {"x": 0, "y": 1, "z": 2}


def getch():
    ch = sys.stdin.read(1)
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=config.ARM_IP)
    ap.add_argument("--side", type=float, default=40.0, help="皮肤边长 mm（几何校验用）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/skin_frame_<ts>.json）")
    args = ap.parse_args()

    from curobo_bridge.rm75_robot import RM75Robot
    arm = RM75Robot(args.arm, config.ARM_PORT)
    if not arm.connect():
        sys.exit(1)
    # 不调用 force_sensor_on —— 那是重心标定会动臂（demo 封装误名）；
    # 传感器数据直接可读，清零即可
    time.sleep(0.3)
    arm.clear_force_data()

    marks = []          # [(名字, pose[6])]
    rpy_probe = None
    step_i = 0

    print(__doc__)
    print(f"当前位姿: {[f'{v:.4f}' for v in arm.get_pose()]}")
    print(f"步长: {STEPS_MM[step_i]}mm\n")

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            ch = getch()
            if ch == "q":
                break
            if ch == "s":
                arm.stop()
                continue
            if ch == "1":
                step_i = (step_i + 1) % len(STEPS_MM)
                print(f"步长 → {STEPS_MM[step_i]}mm")
                continue
            if ch == "p":
                pose = arm.get_pose()
                ret, d = arm.robot.rm_get_force_data()
                fz = d["zero_force_data"][2] if ret == 0 and isinstance(d, dict) else float("nan")
                print(f"pose={[f'{v:.4f}' for v in pose]}  Fz={fz:+.2f}N")
                continue
            if ch == "m":
                pose = arm.get_pose()
                name = CORNER_NAMES[len(marks)] if len(marks) < len(CORNER_NAMES) else None
                if name is None:
                    print("5 个点已标满，按 u 撤销或 q 完成")
                    continue
                marks.append((name, list(pose)))
                if rpy_probe is None:
                    rpy_probe = list(pose[3:6])
                print(f"标记 {name}: {[f'{v:.4f}' for v in pose]}")
                continue
            if ch == "u":
                if marks:
                    name, _ = marks.pop()
                    print(f"撤销 {name}")
                continue

            if ch.lower() in KEYS_BASE:
                axis = KEYS_BASE[ch.lower()]
                sign = 1.0 if ch.islower() else -1.0
                pose = list(arm.get_pose())
                pose[axis] += sign * STEPS_MM[step_i] * 1e-3
                arm.move_pose(pose, speed=config.SPEED_DESCEND, block=1)
                continue
    except KeyboardInterrupt:
        print("\n(Ctrl+C 退出，不保存)")
        arm.disconnect()
        return
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    if len(marks) < 5:
        print(f"只标记了 {len(marks)}/5 个点，不拟合。断开退出。")
        arm.disconnect()
        return

    corners = np.array([p for _, p in marks])[:, :3]
    sf = build_skin_frame(corners, rpy_probe,
                          meta={"arm_ip": args.arm, "side_mm": args.side,
                                "operator_marks": [n for n, _ in marks]})
    geo = sf.check_geometry(expect_side_mm=args.side)

    print("\n========== 拟合结果 ==========")
    print(f"平面 RMS: {sf.residual_mm:.3f} mm  {'PASS' if sf.residual_mm < 0.5 else 'FAIL (>0.5mm, 重标)'}")
    print(f"边长 mm: {[f'{s:.1f}' for s in geo['sides_mm']]}  (期望 {args.side})")
    print(f"边长极差: {geo['side_spread_mm']:.2f} mm  "
          f"{'PASS' if geo['side_spread_mm'] < 1.0 else 'FAIL (>1mm, 有角标错)'}")
    print(f"对角线差: {geo['diag_diff_mm']:.2f} mm")
    print(f"中心偏差: {geo['center_dev_mm']:.2f} mm  "
          f"{'PASS' if geo['center_dev_mm'] < 1.0 else 'WARN (>1mm)'}")
    print(f"rpy_probe: {[f'{v:.4f}' for v in sf.rpy_probe]}")

    out = Path(args.out) if args.out else (
        config.DATA_DIR / f"skin_frame_{__import__('time').strftime('%Y%m%d_%H%M%S')}.json")
    sf.save(out)
    print(f"\n已保存: {out}")
    arm.disconnect()


if __name__ == "__main__":
    main()
