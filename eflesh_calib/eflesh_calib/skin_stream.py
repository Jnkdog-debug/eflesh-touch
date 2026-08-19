"""
皮肤数据流 reader 线程 —— 移植自 eflesh-touch/eflesh-upper/logger.py。

64 字节帧: AA 55 | seq(u8) | 5×(float32 x,y,z LE) = 60B | XOR(bytes[2:63])
同步/校验/丢帧逻辑与 logger.py 字节级一致。

Usage (自检):
  python -m eflesh_calib.skin_stream --port /dev/ttyUSB0 -t 30
"""

import argparse
import struct
import threading
import time
from collections import deque

import numpy as np

from .clock import now_ns
from .config import SKIN_BAUD, SKIN_PORT
from .util import Latest, rate_limited_loop

FRAME_SIZE = 64
N_SENSORS = 5
SYNC = b"\xaa\x55"
BUF_TRIM = 4096


class SkinStats:
    """链路健康计数（reader 线程更新，主线程只读打印/写 attrs）."""

    def __init__(self):
        self.frames = 0
        self.lost = 0
        self.badck = 0
        self.skipB = 0
        self.resyncs = 0
        self.last_seq = -1
        self.t_start_ns = None
        self.t_last_frame_ns = None
        self.max_gap_s = 0.0
        self._lock = threading.RLock()   # 可重入：防持锁调方法时自死锁

    def hz(self) -> float:
        if self.t_start_ns is None:
            return 0.0
        dur = max((now_ns() - self.t_start_ns) * 1e-9, 1e-6)
        with self._lock:
            return self.frames / dur

    def observe_frame(self):
        t = now_ns()
        with self._lock:
            if self.t_start_ns is None:
                self.t_start_ns = t
            if self.t_last_frame_ns is not None:
                self.max_gap_s = max(self.max_gap_s, (t - self.t_last_frame_ns) * 1e-9)
            self.t_last_frame_ns = t
            self.frames += 1

    def age_s(self) -> float:
        with self._lock:
            if self.t_last_frame_ns is None:
                return float("inf")
            return (now_ns() - self.t_last_frame_ns) * 1e-9

    def summary_attrs(self) -> dict:
        hz = self.hz()   # 先算（内部也拿锁），避免持锁重入死锁
        with self._lock:
            return {
                "skin_lost": self.lost,
                "skin_badck": self.badck,
                "skin_skipB": self.skipB,
                "skin_resync": self.resyncs,
                "skin_max_gap_ms": self.max_gap_s * 1e3,
                "skin_mean_hz": hz,
                "skin_frames": self.frames,
            }

    def line(self) -> str:
        a = self.summary_attrs()
        return (f"f={a['skin_mean_hz']:6.1f}Hz lost={a['skin_lost']} "
                f"badck={a['skin_badck']} skipB={a['skin_skipB']} "
                f"resync={a['skin_resync']} maxgap={a['skin_max_gap_ms']:.0f}ms")


def parse_frame(frame: bytes) -> np.ndarray:
    """解 64B 帧的 5×3 磁场值 (float32, µT)。调用方已验校验和."""
    out = np.empty((N_SENSORS, 3), dtype=np.float32)
    for i in range(N_SENSORS):
        out[i] = struct.unpack_from("<fff", frame, 3 + i * 12)
    return out


def skin_reader(port: str, baud: int, out_q: deque, stats: SkinStats,
                latest: Latest, stop: threading.Event):
    """reader 线程主体：读串口 → 解帧 → (t_ns, seq, B) 入队 + 更新 latest."""

    import serial  # 延迟 import，纯训练环境不用装

    def setup():
        nonlocal ser
        ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(0.2)
        ser.reset_input_buffer()   # 丢掉开串口时 ESP32 复位产生的垃圾

    ser = None
    buf = b""

    def body():
        nonlocal buf
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        while True:
            if len(buf) < FRAME_SIZE:
                if len(buf) > BUF_TRIM:   # 很久找不到同步头
                    stats.skipB += len(buf) - 1
                    buf = buf[-1:]
                break
            idx = buf.find(SYNC)
            if idx < 0:
                stats.skipB += len(buf) - 1
                buf = buf[-1:]
                continue
            if idx:
                stats.skipB += idx
                buf = buf[idx:]
            if len(buf) < FRAME_SIZE:
                break
            frame = buf[:FRAME_SIZE]
            ck = 0
            for b in frame[2:63]:
                ck ^= b
            if ck != frame[63]:
                stats.badck += 1
                buf = buf[2:]
                continue

            # 帧有效
            seq = frame[2]
            with stats._lock:
                if stats.last_seq >= 0:
                    d = (seq - (stats.last_seq + 1)) & 0xFF
                    if 0 < d < 128:
                        stats.lost += d
                    elif d >= 128:
                        stats.resyncs += 1
                stats.last_seq = seq
            stats.observe_frame()

            B = parse_frame(frame)
            t = now_ns()
            out_q.append((t, seq, B))
            latest.set((t, B))
            buf = buf[FRAME_SIZE:]

    try:
        rate_limited_loop(500.0, stop, body, setup=setup)  # 串口按到达驱动，上限轮询即可
    finally:
        if ser is not None:
            ser.close()


# ----------------------------------------------------------------------
# 自检 CLI（与 logger.py 输出风格一致）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="eFlesh 皮肤链路自检")
    ap.add_argument("--port", default=SKIN_PORT)
    ap.add_argument("--baud", type=int, default=SKIN_BAUD)
    ap.add_argument("-t", "--duration", type=float, default=30.0)
    args = ap.parse_args()

    q = deque(maxlen=100000)
    st = SkinStats()
    lat = Latest()
    stop_ev = threading.Event()
    th = threading.Thread(target=skin_reader,
                          args=(args.port, args.baud, q, st, lat, stop_ev), daemon=True)
    th.start()

    t_end = time.time() + args.duration
    next_stat = time.time() + 2.0
    try:
        while time.time() < t_end:
            if time.time() >= next_stat:
                el = (now_ns() - st.t_start_ns) * 1e-9 if st.t_start_ns else 0.0
                print(f"[{el:5.1f}s] {st.line()}  queued={len(q)}")
                next_stat += 2.0
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n(Ctrl+C)")
    finally:
        stop_ev.set()
        th.join(timeout=2.0)
        print("\n================ SUMMARY ================")
        print(st.line())
        print(f"queued frames: {len(q)}")
        ok = st.lost == 0 and st.badck == 0
        print("验收: 掉帧=0, 误码=0 →", "PASS" if ok else "FAIL")
