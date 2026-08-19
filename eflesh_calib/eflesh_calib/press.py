"""
单次按压状态机（v2 顺序版 —— 所有 SDK 调用都在主线程，同 go_home 用法）:

  APPROACH → TARE → DESCEND(粗1mm+精0.25mm, touch-off) → PRESS → HOLD(0.8s) → RETRACT

要点:
  - 阻塞 move_pose(block=1)，无并发轮询（并发会卡死 SDK 共享 TCP，实测教训）
  - HOLD 窗口主线程直读 FT(~80Hz)+位姿(~10Hz)，自动入 HDF5
  - 每压各自 touch-off 定 z0；位置标签用实测投影
  - 粗下探用上一次 z_zero 估计，只留 1mm 精探段（提速）
  - 力限幅/工作空间检查内联在每步之间
"""

import time

import numpy as np

from . import config
from .calib_frame import SkinFrame
from .recorder import (PH_APPROACH, PH_DESCEND, PH_HOLD, PH_PRESS, PH_RETRACT,
                       PH_TARE, PH_TOUCHOFF, ST_ABORT_FORCE, ST_ABORT_WS,
                       ST_OK, ST_SKIN_STALE)
from .traj import PressTarget, waypoints


class PressMachine:
    def __init__(self, session, sf: SkinFrame, thr_touch_n: float = config.TOUCH_THR_N):
        self.s = session
        self.sf = sf
        self.thr = thr_touch_n
        self.prev_z_zero = None   # 上一压的 z_zero（皮肤系 mm），用于粗下探估计

    # ------------------------------------------------------------------
    def _pose_at(self, gx_mm, gy_mm, z_skin_mm):
        p = self.sf.skin_to_base(gx_mm, gy_mm, z_skin_mm)
        return [float(p[0]), float(p[1]), float(p[2])] + list(self.sf.rpy_probe)

    def _fz(self):
        return self.s.read_fz()

    def _force_abort(self, fz) -> bool:
        """内联力限幅检查（读到的 wrench 已是 zero_force_data）."""
        if fz is None:
            return False
        if abs(fz) > config.ABORT_FZ_N:
            print(f"[SAFETY] |Fz|={fz:.1f}N > {config.ABORT_FZ_N}N → 中止")
            return True
        return False

    def _phase(self, idx, code):
        self.s.rec.log_phase(self.s.clock.now_ns(), idx, code)

    # ------------------------------------------------------------------
    def run_press(self, target: PressTarget, press_idx: int,
                  air_gap_mm: float | None = None) -> int:
        s = self.s
        arm = s.arm

        # ---------- APPROACH ----------
        self._phase(press_idx, PH_APPROACH)
        wp = waypoints(target, self.sf)
        t_above = s.clock.now_ns()
        ret = s.move_pose(wp["above"], speed=config.SPEED_TRANSIT)
        if ret != 0:
            print(f"[press {target.press_id}] approach ret={ret}")
        time.sleep(0.1)
        s.read_pose()

        # ---------- TARE ----------
        self._phase(press_idx, PH_TARE)
        arm.clear_force_data()
        t_tare = s.clock.now_ns()
        time.sleep(0.15)

        # ---------- DESCEND / TOUCH-OFF ----------
        self._phase(press_idx, PH_DESCEND)
        t_descend = s.clock.now_ns()

        # 粗下探终点：估计表面上方 1mm（首压用名义面 z=0）
        est_surface = self.prev_z_zero if self.prev_z_zero is not None else 0.0
        coarse_stop = est_surface + 1.0

        z_skin = config.HOVER_MM
        # 粗段: 1mm 步进
        while air_gap_mm is None and z_skin > coarse_stop:
            if s.abort_event.is_set():
                self._retract(wp)
                return ST_ABORT_FORCE
            fz = self._fz()
            if self._force_abort(fz):
                s.abort_event.set()
                self._retract(wp)
                return ST_ABORT_FORCE
            s.move_pose(self._pose_at(target.gx_mm, target.gy_mm,
                                      max(z_skin - 1.0, coarse_stop)),
                        speed=config.SPEED_DESCEND)
            z_skin -= 1.0
        # 精段: 0.25mm 步进直到触发（带诊断打印：看臂有没有真的往下走、Fz 读数）
        z_floor = -config.MAX_PROBE_BELOW_MM
        z_zero = None
        t_touchoff = None
        fz_touchoff = float("nan")
        n_step = 0
        while True:
            if s.abort_event.is_set():
                self._retract(wp)
                return ST_ABORT_FORCE
            fz = self._fz()
            if self._force_abort(fz):
                s.abort_event.set()
                self._retract(wp)
                return ST_ABORT_FORCE
            if air_gap_mm is None and fz is not None and abs(fz) > self.thr:
                pr = s.read_pose()
                z_meas = float(self.sf.base_to_skin(pr[1])[2]) if pr else z_skin
                z_zero = z_meas
                t_touchoff = s.clock.now_ns()
                fz_touchoff = fz
                self._phase(press_idx, PH_TOUCHOFF)
                print(f"    [descend] 接触! z0={z_zero:+.2f}mm fz={fz:+.2f}N")
                break
            if z_skin <= z_floor:
                if air_gap_mm is None:
                    pr = s.read_pose()
                    z_meas = float(self.sf.base_to_skin(pr[1])[2]) if pr else float("nan")
                    print(f"[press {target.press_id}] 到下探下限 {z_floor}mm 仍未接触 "
                          f"(实测z={z_meas:+.2f}mm, fz={fz}) → abort")
                    if z_meas == z_meas and z_meas < z_floor + 1.0:
                        print("    ↑ 实测z也到位 → 示教平面高于真实表面：重新示教"
                              "（标记时探针须轻触皮肤）")
                    self._retract(wp)
                    return ST_ABORT_WS
                break
            ret = s.move_pose(self._pose_at(target.gx_mm, target.gy_mm,
                                            max(z_skin - config.STEP_MM, z_floor)),
                              speed=config.SPEED_DESCEND)
            if ret != 0:
                print(f"    [descend] move ret={ret} @z={z_skin:+.2f}（被拒?）")
            z_skin -= config.STEP_MM
            n_step += 1
            if air_gap_mm is None and n_step % 8 == 0:
                pr = s.read_pose()
                z_meas = float(self.sf.base_to_skin(pr[1])[2]) if pr else float("nan")
                print(f"    [descend] 指令z={z_skin:+.2f} 实测z={z_meas:+.2f} fz={fz}",
                      flush=True)

        # 实测接触点投影（位置标签）
        pr = s.read_pose()
        if pr is not None:
            x_m, y_m, _ = self.sf.base_to_skin(pr[1])
        else:
            x_m, y_m = target.gx_mm, target.gy_mm
        if z_zero is not None:
            self.prev_z_zero = z_zero

        # ---------- PRESS ----------
        self._phase(press_idx, PH_PRESS)
        t_press = s.clock.now_ns()
        z_target = (air_gap_mm - target.depth_mm) if air_gap_mm is not None \
            else (z_zero - target.depth_mm)
        z_cur = float(self.sf.base_to_skin(pr[1])[2]) if pr is not None else z_skin
        while z_cur > z_target + config.STEP_MM * 0.5:
            if s.abort_event.is_set():
                self._retract(wp)
                return ST_ABORT_FORCE
            fz = self._fz()
            if self._force_abort(fz):
                s.abort_event.set()
                self._retract(wp)
                return ST_ABORT_FORCE
            s.move_pose(self._pose_at(target.gx_mm, target.gy_mm,
                                      max(z_cur - config.STEP_MM, z_target)),
                        speed=config.SPEED_PRESS)
            pr = s.read_pose()
            z_cur = float(self.sf.base_to_skin(pr[1])[2]) if pr else z_cur - config.STEP_MM

        # ---------- HOLD: 主线程直读 FT(~80Hz) + 位姿(~10Hz)，自动入 HDF5 ----------
        self._phase(press_idx, PH_HOLD)
        t_hold_start = s.clock.now_ns()
        fz_samples = []
        stale = False
        next_pose = t_hold_start
        while s.clock.now_ns() - t_hold_start < config.HOLD_S * 1e9:
            if s.abort_event.is_set():
                self._retract(wp)
                return ST_ABORT_FORCE
            fz = self._fz()
            if fz is not None:
                fz_samples.append(fz)
                if abs(fz) > config.ABORT_FZ_N:
                    print(f"[SAFETY] hold |Fz|={fz:.1f}N → 中止")
                    s.abort_event.set()
                    self._retract(wp)
                    return ST_ABORT_FORCE
            if s.clock.now_ns() >= next_pose:
                s.read_pose()
                next_pose += int(0.1e9)
            if s.skin_stats is not None and s.skin_stats.age_s() > config.SKIN_STALE_S:
                stale = True
            time.sleep(0.012)
        t_hold_end = s.clock.now_ns()
        pr = s.read_pose()
        z_hold = float(self.sf.base_to_skin(pr[1])[2]) if pr else None

        # ---------- RETRACT ----------
        t_retract = s.clock.now_ns()
        self._phase(press_idx, PH_RETRACT)
        s.move_pose(wp["above"], speed=config.SPEED_TRANSIT)
        time.sleep(config.SETTLE_S)

        fz_mean = float(np.nanmean(fz_samples)) if fz_samples else float("nan")
        fz_peak = float(np.nanmax(np.abs(fz_samples))) if fz_samples else float("nan")
        c = s.clock

        s.rec.log_press({
            "press_id": target.press_id, "batch": target.batch,
            "ix": target.ix, "iy": target.iy,
            "gx_mm": target.gx_mm, "gy_mm": target.gy_mm,
            "x_meas_mm": x_m, "y_meas_mm": y_m,
            "depth_cmd_mm": target.depth_mm,
            "z_zero_skin_mm": z_zero if z_zero is not None else float("nan"),
            "z_hold_meas_mm": (z_hold - z_zero) if (z_hold is not None and z_zero is not None) else float("nan"),
            "t_tare": c.to_rel(t_tare), "t_above": c.to_rel(t_above),
            "t_descend": c.to_rel(t_descend),
            "t_touchoff": c.to_rel(t_touchoff) if t_touchoff else float("nan"),
            "t_press": c.to_rel(t_press),
            "t_hold_start": c.to_rel(t_hold_start), "t_hold_end": c.to_rel(t_hold_end),
            "t_retract": c.to_rel(t_retract),
            "fz_touchoff_N": fz_touchoff, "fz_hold_mean_N": fz_mean, "fz_peak_N": fz_peak,
            "status": ST_SKIN_STALE if stale else ST_OK,
        })
        return ST_SKIN_STALE if stale else ST_OK

    # ------------------------------------------------------------------
    def _retract(self, wp: dict):
        try:
            self.s.move_pose(wp["above"], speed=config.SPEED_TRANSIT)
        except Exception as e:
            print(f"[press] retract 失败: {e}")
