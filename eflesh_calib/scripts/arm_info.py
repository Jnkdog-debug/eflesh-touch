#!/usr/bin/env python3
"""查机械臂产品/固件信息 —— 对比两台臂的型号与版本."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eflesh_calib import config


def dump(arm_ip: str):
    from curobo_bridge.rm75_robot import RM75Robot
    arm = RM75Robot(arm_ip, config.ARM_PORT)
    if not arm.connect():
        return
    print(f"\n===== {arm_ip} =====")
    info = arm.robot.rm_get_arm_software_info()
    if info[0] == 0:
        d = info[1]
        print("software_info:", json.dumps(d, ensure_ascii=False, default=str, indent=2))
    st = arm.robot.rm_get_current_arm_state()
    if st[0] == 0:
        print("arm_state keys:", list(st[1].keys()))
        keep = {k: v for k, v in st[1].items() if k not in ("joint", "pose")}
        print("state(extra):", json.dumps(keep, ensure_ascii=False, default=str, indent=2))
    # 全量状态里常有力传感器在位信息
    allst = arm.robot.rm_get_arm_all_state()
    if allst[0] == 0:
        d = allst[1]
        print("all_state keys:", list(d.keys()) if isinstance(d, dict) else type(d))
        if isinstance(d, dict):
            for k in d:
                if "force" in k.lower() or "fz" in k.lower() or "6d" in str(d[k]).lower()[:40]:
                    print(f"  {k} = {d[k]}")
    ret, fd = arm.robot.rm_get_force_data()
    print(f"force_data ret={ret}: {fd}")
    arm.disconnect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=config.ARM_IP)
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()
    if args.both:
        for ip in ("192.168.101.119", "192.168.101.120"):
            dump(ip)
            time.sleep(0.5)
    else:
        dump(args.arm)
