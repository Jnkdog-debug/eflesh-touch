"""
安全监控线程 —— 力限幅 / 工作空间盒 / 皮肤断流。

违规时: abort_event.set() + arm.stop()（打断阻塞中的 movej_p）。
皮肤断流只置 stale（数据无效但运动安全），不打断机械臂。
"""

import threading
import time

from . import config
from .calib_frame import SkinFrame
from .util import Latest, rate_limited_loop


class SafetyMonitor(threading.Thread):
    def __init__(self, arm, latest_ft: Latest, latest_pose: Latest, sf: SkinFrame,
                 abort_event: threading.Event, stop_event: threading.Event,
                 skin_stats=None):
        super().__init__(daemon=True, name="safety")
        self.arm = arm
        self.latest_ft = latest_ft
        self.latest_pose = latest_pose
        self.sf = sf
        self.abort_event = abort_event
        self.stop_event = stop_event
        self.skin_stats = skin_stats
        self.reason = None
        self.violations = []

    def _violation(self, reason: str, hard: bool = True):
        self.violations.append((time.time(), reason))
        print(f"[SAFETY] {'ABORT' if hard else 'WARN'}: {reason}")
        if hard:
            self.reason = reason
            self.abort_event.set()
            try:
                self.arm.stop()
            except Exception:
                pass

    def _check_once(self):
        # --- 力限幅 ---
        v = self.latest_ft.get()
        if v is not None:
            _, w, ret = v
            if ret == 0:
                fz = float(w[2])
                f_any = max(abs(float(x)) for x in w[:3])
                if abs(fz) > config.ABORT_FZ_N:
                    self._violation(f"|Fz|={fz:.1f}N > {config.ABORT_FZ_N}N")
                    return
                if f_any > config.ABORT_F_ANY_N:
                    self._violation(f"|F|max={f_any:.1f}N > {config.ABORT_F_ANY_N}N")
                    return

        # --- 工作空间盒（皮肤系坐标） ---
        p = self.latest_pose.get()
        if p is not None:
            _, xyz, _rpy = p
            x_mm, y_mm, z_mm = self.sf.base_to_skin(xyz)
            lim = config.GRID_HALF_MM + config.WS_MARGIN_MM
            if abs(x_mm) > lim or abs(y_mm) > lim:
                self._violation(f"超出平面范围 xy=({x_mm:.1f},{y_mm:.1f})mm > ±{lim:.0f}mm")
                return
            if z_mm < -config.WS_Z_BELOW_MM or z_mm > config.WS_Z_ABOVE_MM:
                self._violation(f"超出 Z 范围 z={z_mm:.1f}mm "
                                f"∉ [-{config.WS_Z_BELOW_MM}, +{config.WS_Z_ABOVE_MM}]")
                return

        # --- 皮肤断流（软） ---
        if self.skin_stats is not None and self.skin_stats.age_s() > config.SKIN_STALE_S:
            self._violation(f"皮肤流断流 >{config.SKIN_STALE_S}s", hard=False)

    def run(self):
        rate_limited_loop(config.SAFETY_RATE_HZ, self.stop_event, self._check_once)
