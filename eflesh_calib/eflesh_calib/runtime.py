"""
运行时组装 —— 简化版（v2）。

教训: SDK 的并发轮询线程 + 运动指令会互相卡死（共享 TCP）。
新架构: 所有 SDK 调用只在主线程顺序执行（同 go_home.py 的用法）；
只有皮肤串口 reader 一个后台线程（不碰 SDK，已验证 184Hz 稳定）。

用法:
    with Session(arm_ip=..., skin=True, record="data/batch_xxx.h5") as s:
        s.move_pose(pose, speed=45)          # 阻塞，像 go_home
        w = s.read_ft()                       # 主线程直读六维力 zero_force_data
        p = s.read_pose()                     # 主线程直读位姿（都自动入 HDF5）
"""

import signal
import threading
from collections import deque
from pathlib import Path

import numpy as np

from . import config
from .clock import SessionClock, now_ns
from .recorder import Hdf5Recorder
from .skin_stream import SkinStats, skin_reader
from .util import Latest


def resolve_skin_port() -> str:
    """优先用配置的口；不存在则退回第一个可用的 ttyUSB*（防重插后变号）."""
    import glob
    import os
    if os.path.exists(config.SKIN_PORT):
        return config.SKIN_PORT
    for cand in sorted(glob.glob("/dev/ttyUSB*")):
        print(f"[skin] 配置的 {config.SKIN_PORT} 不存在，改用 {cand}")
        return cand
    return config.SKIN_PORT


class Session:
    def __init__(self, arm_ip: str = config.ARM_IP, skin: bool = True,
                 record: str | Path | None = None, attrs: dict | None = None):
        self.clock = SessionClock()
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.int_count = 0
        self.arm = None
        self.skin_stats = None
        self.rec = None
        self._old_int = None

        # ---- 机械臂（只在主线程用）----
        if arm_ip:
            from curobo_bridge.rm75_robot import RM75Robot   # 来自 ~/rm75_curobo_demo
            self.arm = RM75Robot(arm_ip, config.ARM_PORT)
            if not self.arm.connect():
                raise RuntimeError(f"无法连接机械臂 {arm_ip}")
            # 不调用 rm_set_force_sensor —— 那是重心标定会动臂（demo 封装误名）

        # ---- 皮肤串口线程（唯一后台线程，纯 serial，不碰 SDK）----
        self.latest_skin = Latest()
        self.skin_stats = SkinStats() if skin else None
        self.q_skin = deque(maxlen=1_000_000) if skin else None

        # ---- HDF5 录制（写文件线程，不碰 SDK）----
        if record is not None:
            self.rec = Hdf5Recorder(Path(record), self.clock, attrs or {})
            self.rec_sink_skin = self.rec.q("skin") if skin else None
        else:
            self.rec = None
            self.rec_sink_skin = None

        if skin:
            q_skin = self.q_skin                       # 闭包捕获本地引用
            rec_sink = self.rec_sink_skin

            class _Sink:
                def append(self, item):
                    q_skin.append(item)
                    if rec_sink is not None:
                        rec_sink.append(item)

            self._thread = threading.Thread(
                target=skin_reader,
                args=(resolve_skin_port(), config.SKIN_BAUD, _Sink(),
                      self.skin_stats, self.latest_skin, self.stop_event),
                daemon=True, name="skin")

    # ------------------------------------------------------------------
    # 主线程 SDK 直读（同时喂 HDF5）
    # ------------------------------------------------------------------
    def read_ft(self):
        """读六维力 zero_force_data → (t_rel_s, w f4(6), ret)。喂 ft 流."""
        if self.arm is None:
            return None
        ret, d = self.arm.robot.rm_get_force_data()
        w6 = (d.get("zero_force_data", d.get("force_data", [0] * 6))
              if isinstance(d, dict) else list(d))
        t = now_ns()
        w = np.asarray(w6, dtype=np.float32)
        if self.rec is not None:
            self.rec.q("ft").append((t, w, int(ret)))
        return self.clock.to_rel(t), w, int(ret)

    def read_pose(self):
        """读位姿 → (t_rel_s, xyz, rpy, joints)。喂 pose 流."""
        if self.arm is None:
            return None
        st = self.arm.robot.rm_get_current_arm_state()
        if st[0] != 0:
            return None
        pose, joints = st[1]["pose"], st[1]["joint"]
        t = now_ns()
        xyz = np.asarray(pose[:3], dtype=np.float64)
        rpy = np.asarray(pose[3:6], dtype=np.float64)
        j = np.asarray(joints, dtype=np.float64)
        if self.rec is not None:
            self.rec.q("pose").append((t, xyz, rpy, j))
        return self.clock.to_rel(t), xyz, rpy, j

    def read_fz(self) -> float | None:
        """便捷: 当前 Fz (N, zero_force_data)。失败返回 None."""
        r = self.read_ft()
        if r is None or r[2] != 0:
            return None
        return float(r[1][2])

    def move_pose(self, pose, speed: int = config.SPEED_TRANSIT) -> int:
        """阻塞笛卡尔运动（同 go_home 的用法，block=1）。"""
        return self.arm.move_pose(list(pose), speed=speed, block=1)

    # ------------------------------------------------------------------
    def __enter__(self) -> "Session":
        if self.q_skin is not None:
            self._thread.start()
        main = self

        def _int(sig, frm):
            main.int_count += 1
            if main.int_count == 1:
                print("\n[Session] Ctrl+C → 慢停（再按升级）", flush=True)
                main.abort_event.set()
                if main.arm:
                    main.arm.stop()
            elif main.int_count == 2:
                print("\n[Session] 第二次 Ctrl+C → 急停", flush=True)
                if main.arm:
                    main.arm.hard_stop()
            else:
                print("\n[Session] 第三次 Ctrl+C → 强制退出", flush=True)
                if main.arm:
                    try:
                        main.arm.hard_stop()
                    except Exception:
                        pass
                import os
                os._exit(130)

        self._old_int = signal.signal(signal.SIGINT, _int)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.q_skin is not None:
            self._thread.join(timeout=3.0)
        if self.rec is not None:
            extra = {}
            if self.skin_stats is not None:
                extra.update(self.skin_stats.summary_attrs())
            self.rec.close(extra_attrs=extra)
        if self.arm is not None:
            try:
                self.arm.stop()
                self.arm.disconnect()
            except Exception:
                pass
        if self._old_int is not None:
            signal.signal(signal.SIGINT, self._old_int)
