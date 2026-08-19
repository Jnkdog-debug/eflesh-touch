#!/usr/bin/env python3
"""
预检查（正式采集前的门禁，对应采集计划笔记 §3）:

  --ft-only                六维力自检（读数→清零→手压测试）
  --calib skin_frame.json  全项预检:
    1) 皮肤空载噪声（--idle 分钟，正式建议 10）
    2) FT 空载噪声 → touch 阈值 = max(0.2, 4×std) 回写 JSON
    3) 磁干扰飞越（臂扫 3×3、离面 5mm 不接触）—— 门禁: 波动 < 3× 空载噪声
    4) 单点重复性（中心 20 压，CV < 5%）

v2: 所有 SDK 调用主线程顺序执行（block=1 运动，同 go_home），无并发轮询。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eflesh_calib import config
from eflesh_calib.calib_frame import SkinFrame
from eflesh_calib.runtime import Session
from eflesh_calib.traj import make_grid, waypoints


def ft_only(arm_ip: str):
    from curobo_bridge.rm75_robot import RM75Robot
    arm = RM75Robot(arm_ip, config.ARM_PORT)
    if not arm.connect():
        sys.exit(1)
    # 不调用 rm_set_force_sensor —— 那是"自动重心标定"会让机械臂运动

    print("[1] 原始 dict（看全部字段）:")
    ret, d = arm.robot.rm_get_force_data()
    print(f"    ret={ret}")
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"    {k} = {[f'{x:+.3f}' for x in v]}")

    print("\n[2] 清零 (对 zero_force_data 生效):", arm.clear_force_data())
    time.sleep(0.3)
    samples = []
    for _ in range(20):
        ret, d = arm.robot.rm_get_force_data()
        if ret == 0 and isinstance(d, dict):
            samples.append(d["zero_force_data"])
        time.sleep(0.05)
    if samples:
        a = np.array(samples)
        print(f"[3] 清零后 zero_force_data 20 读:")
        print(f"    mean={[f'{v:+.3f}' for v in a.mean(axis=0)]}")
        print(f"    std ={[f'{v:.3f}' for v in a.std(axis=0)]}")

    print("\n[4] 手压测试: 用手轻推探针，观察 F 变化 (Ctrl+C 结束)")
    try:
        while True:
            ret, d = arm.robot.rm_get_force_data()
            if ret == 0 and isinstance(d, dict):
                w = d["zero_force_data"]
                print(f"\rzero F=[{', '.join(f'{v:+6.2f}' for v in w[:3])}] "
                      f"M=[{', '.join(f'{v:+6.3f}' for v in w[3:])}]   ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n完成")
    arm.disconnect()


def sample_skin(session, seconds: float, label: str) -> np.ndarray:
    """采一段皮肤帧 → (N,15) 数组（带倒计时进度）."""
    q = session.q_skin
    n0 = len(q)
    t0 = time.time()
    t_end = t0 + seconds
    next_msg = t0 + 5.0
    while time.time() < t_end and not session.abort_event.is_set():
        time.sleep(0.1)
        now = time.time()
        if now >= next_msg:
            print(f"  [{label}] 剩余 {int(t_end - now):3d}s  已采 {len(q)-n0} 帧",
                  flush=True)
            next_msg += 5.0
    frames = list(q.copy())[n0:]   # copy() 单次 C 调用，原子；直接迭代会与写线程冲突
    if not frames:
        print(f"[{label}] 无皮肤帧!")
        return np.zeros((0, 15))
    return np.stack([f[2].ravel() for f in frames])


def full_preflight(args):
    sf = SkinFrame.load(args.calib)

    with Session(arm_ip=args.arm, skin=True,
                 record=config.DATA_DIR / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}.h5",
                 attrs={"kind": "preflight", "calib_json": args.calib}) as s:
        # ---------- 1) 皮肤空载噪声 ----------
        print(f"\n===== 1) 皮肤空载噪声（{args.idle*60:.0f}s，期间不要碰皮肤/桌子） =====")
        idle = sample_skin(s, args.idle * 60, "idle")
        idle_std = idle.std(axis=0) if len(idle) else np.full(15, np.nan)
        print(f"帧数 {len(idle)}  ({len(idle)/(args.idle*60+1e-9):.1f}Hz)")
        print(f"每通道 std(µT): {[f'{v:.1f}' for v in idle_std]}")
        drift = (idle[-50:].mean(axis=0) - idle[:50].mean(axis=0))
        print(f"首尾 50 帧均值漂移(µT): {[f'{v:+.1f}' for v in drift]}")

        # ---------- 2) FT 空载噪声 → touch 阈值 ----------
        print("\n===== 2) FT 空载噪声（探针悬空 10s） =====")
        time.sleep(0.5)
        s.arm.clear_force_data()
        time.sleep(0.5)
        fz = []
        t_end = time.time() + 10
        while time.time() < t_end:
            v = s.read_fz()
            if v is not None:
                fz.append(v)
            time.sleep(0.02)
        fz_std = float(np.std(fz)) if fz else float("nan")
        thr = max(config.TOUCH_THR_N, 4.0 * fz_std)
        print(f"样本 {len(fz)}  Fz std={fz_std:.4f}N → touch 阈值 = max(0.2, 4σ) = {thr:.3f}N")

        # ---------- 皮肤无数据直接 FAIL（此前 nan 会被误判 PASS）----------
        if len(idle) == 0:
            print("\n*** 皮肤无数据（USB 掉了/端口变了/没插）→ 预检 FAIL ***")
            print("*** 检查: ls /dev/ttyUSB* ; 自检: python -m eflesh_calib.skin_stream -t 10 ***")
            sys.exit(3)

        # ---------- 3) 磁干扰飞越（门禁）----------
        print("\n===== 3) 磁干扰飞越（臂扫 3×3 @+5mm，不接触皮肤） =====")
        grid = make_grid(pitch_mm=args.pitch, subset_n=3)
        t_fly0 = time.time()
        for k, (ix, iy, gx, gy) in enumerate(grid):
            if s.abort_event.is_set():
                print("中止")
                return
            wp = waypoints(type("T", (), {"gx_mm": gx, "gy_mm": gy})(), sf,
                           hover_mm=config.HOVER_MM)
            ret = s.move_pose(wp["above"], speed=config.SPEED_TRANSIT)
            time.sleep(0.3)
            print(f"  ({k+1}/{len(grid)}) gx={gx:+.0f} gy={gy:+.0f}  ret={ret}  "
                  f"({time.time()-t_fly0:.0f}s)", flush=True)
        print("飞越完成，采 3s 皮肤 …")
        time.sleep(1.0)
        fly = sample_skin(s, 3.0, "flyover")

        # ---------- 4) 单点重复性 ----------
        print("\n===== 4) 中心单点重复性（20 压 @ depth 1.0mm） =====")
        from eflesh_calib.press import PressMachine
        from eflesh_calib.traj import PressTarget
        pm = PressMachine(s, sf, thr_touch_n=thr)
        press_mags = []
        for i in range(20):
            if s.abort_event.is_set():
                print("中止")
                break
            t = PressTarget(f"rep{i:02d}", 0, 0, 0.0, 0.0, 1.0, "preflight")
            st = pm.run_press(t, 10000 + i)
            if st != 1:
                print(f"  第 {i+1} 压异常 status={st}")
                if s.abort_event.is_set() or st == 3:   # 中止/工作空间违例 → 停止后续压
                    print("  连续异常，停止重复性测试（先解决上面的问题）")
                    break
            if not s.rec._events:
                continue
            ev = s.rec._events[-1]
            hs, he = ev["t_hold_start"], ev["t_hold_end"]
            if not (hs == hs and he == he):   # nan 检查
                continue
            t0_ns = s.clock.t0_ns
            ta, td = ev["t_above"], ev["t_descend"]   # 基线窗口 = 下探前（未接触！）
            lo, hi = hs - 1e-9, he + 1e-9
            snap = list(s.q_skin.copy())

            def _rel(f):
                return (f[0] - t0_ns) * 1e-9

            base = [f for f in snap if ta + 0.2 <= _rel(f) <= td - 0.05]
            hold = [f for f in snap if lo <= _rel(f) <= hi]
            if len(base) >= 3 and len(hold) >= 5:
                b0 = np.mean([f[2] for f in base], axis=0)
                m = np.mean([np.linalg.norm((f[2] - b0).ravel()) for f in hold])
                press_mags.append(m)
            print(f"  压 {i+1}/20 完成", flush=True)

        # ---------- 飞越判定 + 重复性统计 + 回写 ----------
        fly_std = fly.std(axis=0) if len(fly) else np.full(15, np.nan)
        ratio = fly_std / np.maximum(idle_std, 1e-6)
        gate_pass = bool(np.all(np.nan_to_num(ratio) < 3.0))
        print(f"\n飞行段 std/空载 std: {[f'{v:.1f}' for v in ratio]}")
        print(f"门禁 (<3×): {'PASS' if gate_pass else '*** FAIL — 挪皮肤/改姿态后重测 ***'}")

        meta = dict(sf.meta)
        if len(press_mags) >= 10:
            a = np.array(press_mags)
            a_ss = a[3:] if len(a) > 6 else a   # 跳过前 3 压（TPU 初压沉降）
            cv = float(a_ss.std() / a_ss.mean()) if a_ss.mean() > 0 else float("nan")
            print(f"重复性: 全部 {len(a)} 压 mean={a.mean():.1f}µT；"
                  f"去掉前 3 压沉降后 n={len(a_ss)} mean={a_ss.mean():.1f}µT "
                  f"CV={cv*100:.2f}%  {'PASS' if cv < 0.05 else '*** FAIL (>5%) ***'}")
            meta["repeat_cv"] = cv
        else:
            print("重复性有效样本不足（事后可用 inspect_h5.py 核对）")

        meta.update({"thr_touch_n": thr, "fz_idle_std": fz_std,
                     "flyover_gate_pass": gate_pass,
                     "preflight_utc": time.strftime("%Y-%m-%dT%H:%M:%S")})
        sf.meta = meta
        out = sf.save(args.calib)
        print(f"\ntouch 阈值已回写: {out}  thr={thr:.3f}N")

        print("\n========== 预检汇总 ==========")
        print(f"皮肤空载 std max: {np.nanmax(idle_std):.1f} µT")
        print(f"FT Fz std:        {fz_std:.4f} N")
        print(f"磁干扰门禁:       {'PASS' if gate_pass else 'FAIL'}")
        if not gate_pass:
            sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-only", action="store_true")
    ap.add_argument("--calib", default=None, help="skin_frame JSON（全项预检）")
    ap.add_argument("--idle", type=float, default=1.0, help="空载噪声时长（分钟，正式用 10）")
    ap.add_argument("--pitch", type=float, default=config.GRID_PITCH_MM)
    ap.add_argument("--arm", default=config.ARM_IP)
    args = ap.parse_args()

    if args.ft_only:
        ft_only(args.arm)
    elif args.calib:
        full_preflight(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
