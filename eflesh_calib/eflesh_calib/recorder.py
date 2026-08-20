"""
HDF5 录制器 —— 单写线程 owns h5py 文件（h5py 非线程安全），
读线程只往 deque 里塞，写线程周期性批量落盘。

Schema (v1):
  attrs:  schema_version, t0_monotonic_ns, t0_utc, arm_ip, skin_port, skin_baud,
          calib_json(+sha), grid, depths_mm, speeds, hold_s, thr_touch_N,
          abort limits, ft_rate_measured_hz, pose_rate_measured_hz, ...
  skin/   t(f8,N)  B(f4,N×5×3 µT)  seq(i4)
  ft/     t(f8,M)  wrench(f4,M×6)  ret(i4,M)
  pose/   t(f8,K)  xyz(f8,K×3)  rpy(f8,K×3)  joints(f8,K×7)
  phases/ t(f8,P)  press_idx(i4,P)  phase_code(i1,P)
  events/ 逐压一行（平行 1-D datasets）

时间全部为相对会话 t0 的秒（f8）。
"""

import hashlib
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from . import config
from .clock import SessionClock

# 相位码
PH_APPROACH, PH_TARE, PH_DESCEND, PH_TOUCHOFF, PH_PRESS, PH_HOLD, PH_RETRACT = range(1, 8)
PH_DRAG, PH_TWIST = 8, 9          # 剪切/扭转批扩展相位（PRESS 之后、HOLD 之前）
PHASE_NAMES = {1: "approach", 2: "tare", 3: "descend", 4: "touchoff",
               5: "press", 6: "hold", 7: "retract", 8: "drag", 9: "twist"}

# events 列定义: (名字, dtype, 说明)
EVENT_COLS = [
    ("press_id", "S32"), ("batch", "S16"),
    ("ix", "i2"), ("iy", "i2"), ("gx_mm", "f4"), ("gy_mm", "f4"),
    ("x_meas_mm", "f4"), ("y_meas_mm", "f4"),
    ("depth_cmd_mm", "f4"), ("z_zero_skin_mm", "f4"), ("z_hold_meas_mm", "f4"),
    ("t_tare", "f8"), ("t_above", "f8"), ("t_descend", "f8"),
    ("t_touchoff", "f8"), ("t_press", "f8"),
    ("t_hold_start", "f8"), ("t_hold_end", "f8"), ("t_retract", "f8"),
    ("fz_touchoff_N", "f4"), ("fz_hold_mean_N", "f4"), ("fz_peak_N", "f4"),
    # 剪切/扭转批扩展（法向批为 NaN，老文件无这些列 → build_dataset 容错读取）
    ("x_hold_meas_mm", "f4"), ("y_hold_meas_mm", "f4"),
    ("drag_dx_cmd_mm", "f4"), ("drag_dy_cmd_mm", "f4"),
    ("twist_cmd_deg", "f4"), ("twist_meas_deg", "f4"),
    ("fx_hold_mean_N", "f4"), ("fy_hold_mean_N", "f4"), ("mz_hold_mean_N", "f4"),
    ("status", "i1"),
]

# 状态码
ST_OK, ST_ABORT_FORCE, ST_ABORT_WS, ST_SKIPPED, ST_SKIN_STALE, ST_ABORT_SHEAR = range(1, 7)
STATUS_NAMES = {1: "ok", 2: "abort_force", 3: "abort_workspace",
                4: "skipped", 5: "skin_stale", 6: "abort_shear"}

_CHUNK = 4096


class Hdf5Recorder:
    def __init__(self, path: Path, clock: SessionClock, attrs: dict):
        self.path = Path(path)
        self.clock = clock
        self._events = []           # list[dict]，close 时一次性写
        self._phases = []           # list[(t, press_idx, code)]
        self._closed = False
        self._ev_lock = threading.Lock()

        self.h5 = h5py.File(self.path, "w")
        base = {"schema_version": config.SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "t0_monotonic_ns": clock.t0_ns, "t0_utc": clock.t0_utc}
        base.update(attrs)
        for k, v in base.items():
            self.h5.attrs[k] = v

        # 三个流（可增长 dataset）
        g = self.h5.create_group("skin")
        g.create_dataset("t", (0,), maxshape=(None,), dtype="f8", chunks=(_CHUNK,))
        g.create_dataset("B", (0, 5, 3), maxshape=(None, 5, 3), dtype="f4", chunks=(_CHUNK, 5, 3))
        g.create_dataset("seq", (0,), maxshape=(None,), dtype="i4", chunks=(_CHUNK,))

        g = self.h5.create_group("ft")
        g.create_dataset("t", (0,), maxshape=(None,), dtype="f8", chunks=(_CHUNK,))
        g.create_dataset("wrench", (0, 6), maxshape=(None, 6), dtype="f4", chunks=(_CHUNK, 6))
        g.create_dataset("ret", (0,), maxshape=(None,), dtype="i4", chunks=(_CHUNK,))

        g = self.h5.create_group("pose")
        g.create_dataset("t", (0,), maxshape=(None,), dtype="f8", chunks=(_CHUNK,))
        g.create_dataset("xyz", (0, 3), maxshape=(None, 3), dtype="f8", chunks=(_CHUNK, 3))
        g.create_dataset("rpy", (0, 3), maxshape=(None, 3), dtype="f8", chunks=(_CHUNK, 3))
        g.create_dataset("joints", (0, 7), maxshape=(None, 7), dtype="f8", chunks=(_CHUNK, 7))

        g = self.h5.create_group("phases")
        g.create_dataset("t", (0,), maxshape=(None,), dtype="f8", chunks=(_CHUNK,))
        g.create_dataset("press_idx", (0,), maxshape=(None,), dtype="i4", chunks=(_CHUNK,))
        g.create_dataset("phase_code", (0,), maxshape=(None,), dtype="i1", chunks=(_CHUNK,))

        self.h5.create_group("events")   # close 时写入

        self._qs = {name: deque() for name in ("skin", "ft", "pose")}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    # ---------------- 各流线程调用 ----------------
    def q(self, stream: str) -> deque:
        return self._qs[stream]

    def log_phase(self, t_ns: int, press_idx: int, phase_code: int):
        self._phases.append((self.clock.to_rel(t_ns), press_idx, phase_code))

    def log_press(self, row: dict):
        with self._ev_lock:
            self._events.append(row)

    # ---------------- 写线程 ----------------
    def _writer(self):
        while not self._stop.is_set():
            for name in ("skin", "ft", "pose"):
                items = []
                q = self._qs[name]
                while True:
                    try:
                        items.append(q.popleft())
                    except IndexError:
                        break
                if items:
                    self._append_stream(name, items)
            try:
                self.h5.flush()
            except Exception:
                pass
            self._stop.wait(1.0)

    def _append_stream(self, name, items):
        g = self.h5[name]
        if name == "skin":
            arrays = {
                "t": np.array([self.clock.to_rel(it[0]) for it in items], dtype="f8"),
                "B": np.stack([it[2] for it in items]),
                "seq": np.array([it[1] for it in items], dtype="i4"),
            }
        elif name == "ft":
            arrays = {
                "t": np.array([self.clock.to_rel(it[0]) for it in items], dtype="f8"),
                "wrench": np.stack([it[1] for it in items]),
                "ret": np.array([it[2] for it in items], dtype="i4"),
            }
        else:  # pose
            arrays = {
                "t": np.array([self.clock.to_rel(it[0]) for it in items], dtype="f8"),
                "xyz": np.stack([it[1] for it in items]),
                "rpy": np.stack([it[2] for it in items]),
                "joints": np.stack([it[3] for it in items]),
            }

        n0 = g["t"].shape[0]
        n = len(items)
        for key, arr in arrays.items():
            ds = g[key]
            ds.resize(n0 + n, axis=0)
            ds[n0:] = arr

    # ---------------- 收尾 ----------------
    def set_attrs(self, attrs: dict):
        for k, v in attrs.items():
            self.h5.attrs[k] = v

    def close(self, extra_attrs: dict | None = None):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=5.0)
        # 写空残留
        for name in ("skin", "ft", "pose"):
            items = []
            q = self._qs[name]
            while True:
                try:
                    items.append(q.popleft())
                except IndexError:
                    break
            if items:
                self._append_stream(name, items)
        # phases
        if self._phases:
            arr = np.array(self._phases, dtype="f8")
            del self.h5["phases"]
            g = self.h5.create_group("phases")
            g.create_dataset("t", data=arr[:, 0].astype("f8"))
            g.create_dataset("press_idx", data=arr[:, 1].astype("i4"))
            g.create_dataset("phase_code", data=arr[:, 2].astype("i1"))
        # events
        eg = self.h5["events"]
        rows = self._events
        for name, dtype in EVENT_COLS:
            if name in ("press_id", "batch"):
                col = np.array([str(r.get(name, "")) for r in rows], dtype=dtype)
            else:
                col = np.array([r.get(name, np.nan) for r in rows], dtype=dtype)
            eg.create_dataset(name, data=col)
        if extra_attrs:
            self.set_attrs(extra_attrs)
        self.h5.close()
        print(f"[recorder] 写入完成: {self.path}  ({len(rows)} presses)")


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
