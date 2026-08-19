#!/usr/bin/env python3
"""
eFlesh 5-Sensor 3D Magnetic Field Visualizer
==============================================
Reads MLX90393 magnetic field data from an ESP32 over serial and renders
5 real-time 3D vector arrows in an OpenGL scene via pyqtgraph.

Supports two firmware protocols:
  TEXT:   /home/xyz/Arduino/eflash/eflash.ino
          FRAME: [S1:x,y,z] [S2:x,y,z] ... [S5:x,y,z]  @ 115200 baud
  BINARY: /home/xyz/Arduino/eflash/eflash_binary/eflash_binary.ino
          64-byte frames (sync + seq + 5x3 floats LE + checksum) @ 921600 baud
          (921600 = CP2102 USB-UART bridge limit; 2M baud garbles on it)

Usage:
  python visualize_3d.py                        # text protocol
  python visualize_3d.py --binary               # binary protocol (fast)
  python visualize_3d.py --sim                  # simulated data
  python visualize_3d.py --port /dev/ttyUSB0 --baud 921600 --binary
"""

import sys
import re
import math
import struct
import time
import argparse
import numpy as np

# ============================================================
# Parse CLI args early (before Qt imports)
# ============================================================
parser = argparse.ArgumentParser(description='eFlesh 3D Tactile Visualizer')
parser.add_argument('--sim', action='store_true',
                    help='Run with simulated data (no hardware needed)')
parser.add_argument('--binary', action='store_true',
                    help='Use binary protocol (eflash_binary firmware, default 2M baud)')
parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                    help='Serial port (default: /dev/ttyUSB0)')
parser.add_argument('--baud', type=int, default=None,
                    help='Serial baud rate (default: 115200 text, 921600 binary)')
parser.add_argument('--debug', action='store_true',
                    help='Print raw serial data for diagnostics')
args = parser.parse_args()

BINARY_MODE = args.binary
if args.baud is None:
    args.baud = 921600 if BINARY_MODE else 115200

# ============================================================
# Serial setup (or simulated data mode)
# ============================================================
ser = None
serial_buffer = b''            # accumulate partial serial reads
SIM_MODE = args.sim

if not SIM_MODE:
    try:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        proto = "BINARY" if BINARY_MODE else "TEXT"
        print(f"[OK] Serial connected: {args.port} @ {args.baud}  [{proto}]")
    except Exception as e:
        print(f"[WARN] Cannot open {args.port}: {e}")
        print("[INFO] Falling back to --sim mode")
        SIM_MODE = True

# ============================================================
# Imports (after Qt env is ready)
# ============================================================
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ============================================================
# Sensor configuration

# Physical layout on PCB (mm) — 四角 + 中心的方形排布（五点骰子形）:
#     S1 ──── S2
#        [S5]          (S5 在正方形中心)
#     S3 ──── S4
# 坐标约定：俯视皮肤，+Y 是"上"、+X 是"右"（对应视图里的绿/红轴）。
# 如果和实物转向对不上（差 90°/180°），把对应坐标的符号对调即可。
# ============================================================
PITCH = 12.0  # 相邻角传感器的间距 (mm)，如 S1-S2 的距离 —— 【用卡尺实测后修改！】
OFF = PITCH / 2.0

SENSOR_POS = {
    'S1': np.array([-OFF, +OFF, 0]),   # 左上
    'S2': np.array([+OFF, +OFF, 0]),   # 右上
    'S3': np.array([-OFF, -OFF, 0]),   # 左下
    'S4': np.array([+OFF, -OFF, 0]),   # 右下
    'S5': np.array([0, 0, 0]),         # 中心
}

SENSOR_IDS = ['S1', 'S2', 'S3', 'S4', 'S5']
NUM_SENSORS = len(SENSOR_IDS)

# Each sensor's latest 3D magnetic field reading (x, y, z) in µT
sensor_data = {sid: np.array([0.0, 0.0, 0.0]) for sid in SENSOR_IDS}

# ============================================================
# Regex: matches [S1:-1959.8,-2912.7,4918.2]
# ============================================================
PATTERN = re.compile(r'\[S(\d+):([^,]+),([^,]+),([^\]]+)\]')

# ============================================================
# Simulated data generator (for --sim mode)
# ============================================================
class SimDataGenerator:
    """Generates fake sinusoidal magnetic field data for demos."""

    def __init__(self):
        self._t0 = time.time()
        # Each sensor gets a unique phase / axis bias
        self._phases = {
            'S1': (0.0, 1.3, 2.1),
            'S2': (1.5, 0.7, 3.0),
            'S3': (2.8, 1.9, 0.4),
            'S4': (0.3, 2.5, 1.7),
            'S5': (1.0, 1.0, 1.0),
        }

    def generate(self):
        t = time.time() - self._t0
        for sid in SENSOR_IDS:
            px, py, pz = self._phases[sid]
            x = 2500 * math.sin(t * 1.8 + px) * math.cos(t * 0.6 + py)
            y = 2500 * math.cos(t * 1.3 + py) * math.sin(t * 0.9 + pz)
            z = 3000 * math.sin(t * 2.0 + pz) + 500 * math.cos(t * 1.1 + px)
            sensor_data[sid] = np.array([x, y, z])

sim_gen = SimDataGenerator() if SIM_MODE else None

# ============================================================
# Qt Application & 3D Viewport
# ============================================================
app = QApplication(sys.argv)
app.setApplicationName('eFlesh 3D Visualizer')

w = gl.GLViewWidget()
w.setWindowTitle('eFlesh — 5-Sensor Magnetic Field (SPI)')
w.resize(1200, 800)
w.setCameraPosition(distance=80, elevation=35, azimuth=-55)
w.show()

# ---- Grid floor ----
grid = gl.GLGridItem()
grid.setSize(60, 60)
grid.setSpacing(2, 2)
grid.translate(0, 0, -15)
w.addItem(grid)

# ---- Sensor origin dots (white spheres) ----
origin_pts = np.array([SENSOR_POS[sid] for sid in SENSOR_IDS])
scatter = gl.GLScatterPlotItem(pos=origin_pts, color=(1, 1, 1, 0.9),
                               size=8, pxMode=False)
w.addItem(scatter)

# ---- Sensor labels: S1..S5 浮在各原点上方，方便核对 通道-物理位置 映射 ----
if hasattr(gl, 'GLTextItem'):          # pyqtgraph >= 0.13
    for sid in SENSOR_IDS:
        w.addItem(gl.GLTextItem(pos=SENSOR_POS[sid] + np.array([0, 0, 3]),
                                text=sid, color=(1, 1, 1, 0.9)))

# ---- Sensor labels (GLLabelItem = not available; use TextItem at each pos) ----
# Simple axis-aligned text labels are tricky in OpenGL.  We'll label in legend style
# by adding small coloured spheres at each sensor origin.
# Better: use a 2D overlay for the legend.

# ---- Direction lines (origin → tip) ----
arrows_item = gl.GLLinePlotItem(mode='lines', width=2.5, antialias=True)
w.addItem(arrows_item)

# ---- Coordinate axes helper ----
def add_axes(view):
    """Draw simple R/G/B XYZ axes at origin."""
    ax_len = 15
    pts = np.array([
        [0, 0, 0], [ax_len, 0, 0],   # X = red
        [0, 0, 0], [0, ax_len, 0],   # Y = green
        [0, 0, 0], [0, 0, ax_len],   # Z = blue
    ])
    cols = np.array([
        (1, 0.3, 0.3, 1), (1, 0.3, 0.3, 1),
        (0.3, 1, 0.3, 1), (0.3, 1, 0.3, 1),
        (0.3, 0.3, 1, 1), (0.3, 0.3, 1, 1),
    ])
    axes = gl.GLLinePlotItem(pos=pts, color=cols, width=2.0, antialias=True)
    view.addItem(axes)

add_axes(w)

# ============================================================
# Colour mapping: magnitude → blue(weak) → cyan → green → yellow → red(strong)
# ============================================================
def magnitude_to_color(mag, max_mag=3000.0, min_mag=0.0):
    """
    Map a scalar magnitude to an RGBA colour.
    Low  → blue   (0, 0, 1)
    Mid  → green  (0, 1, 0)
    High → red    (1, 0, 0)
    """
    # Clamp and normalise
    norm = (abs(mag) - min_mag) / (max_mag - min_mag)
    norm = max(0.0, min(norm, 1.0))

    # Blue → Cyan → Green → Yellow → Red  gradient
    if norm < 0.25:
        t = norm / 0.25
        r, g, b = 0.0, t, 1.0         # blue → cyan
    elif norm < 0.5:
        t = (norm - 0.25) / 0.25
        r, g, b = 0.0, 1.0, 1.0 - t   # cyan → green
    elif norm < 0.75:
        t = (norm - 0.5) / 0.25
        r, g, b = t, 1.0, 0.0         # green → yellow
    else:
        t = (norm - 0.75) / 0.25
        r, g, b = 1.0, 1.0 - t, 0.0   # yellow → red

    return (r, g, b, 1.0)


# ============================================================
# Serial frame readers — text and binary protocols
# ============================================================

# Text protocol
PATTERN = re.compile(r'\[S(\d+):([^,]+),([^,]+),([^\]]+)\]')

def read_text_frames():
    """Read lines, return complete FRAME: lines."""
    global serial_buffer
    frames = []
    try:
        if ser.in_waiting:
            serial_buffer += ser.read(ser.in_waiting)
        while b'\n' in serial_buffer:
            raw_line, serial_buffer = serial_buffer.split(b'\n', 1)
            line = raw_line.decode('utf-8', errors='ignore').strip()
            if line.startswith('FRAME:'):
                frames.append(line)
            elif args.debug and line:
                print(f"[DBG] {line[:100]}")
    except Exception:
        pass
    return frames


def parse_text_frame(line):
    """Parse a FRAME line, update sensor_data. Returns number of sensors."""
    matches = PATTERN.findall(line)
    count = 0
    for match in matches:
        s_id = f"S{match[0]}"
        try:
            x = float(match[1])
            y = float(match[2])
            z = float(match[3])
        except ValueError:
            continue
        if s_id in sensor_data:
            sensor_data[s_id] = np.array([x, y, z], dtype=np.float32)
            count += 1
    return count


# Binary protocol: 64-byte frames
# [0xAA] [0x55] [seq:u8] [5×(float32 x,y,z) = 60B] [xor_checksum:u8]
BIN_FRAME_SIZE = 64
BIN_SYNC0 = 0xAA
BIN_SYNC1 = 0x55
_bin_last_seq = -1
_bin_lost_frames = 0
_bin_badck = 0        # checksum 失败次数 —— 链路误码/丢字节会推高它
_bin_skip_bytes = 0   # 找不到同步头丢弃的字节数 —— 正常≈0；固件重启时会蹦出一批
_bin_resync = 0       # seq 大跳(≥128, mod 256 环绕) —— 多半意味着固件重启了
_frames_parsed = 0
_frames_last = 0

def read_binary_frames():
    """
    Accumulate bytes until we have at least 64.  Scan for sync bytes
    and extract valid frames.  Returns number of frames parsed.
    """
    global serial_buffer, _bin_last_seq, _bin_lost_frames
    global _bin_badck, _bin_skip_bytes, _bin_resync, _frames_parsed
    n_parsed = 0

    try:
        if ser.in_waiting:
            serial_buffer += ser.read(ser.in_waiting)

        # Search for sync in the buffer
        while len(serial_buffer) >= BIN_FRAME_SIZE:
            # Find sync
            idx = serial_buffer.find(bytes([BIN_SYNC0, BIN_SYNC1]))
            if idx < 0:
                # No sync found — keep last byte (might be part of 0xAA)
                _bin_skip_bytes += len(serial_buffer) - 1
                if args.debug:
                    print(f"[DBG] no sync in {len(serial_buffer)}B, discarding")
                serial_buffer = serial_buffer[-1:]
                break

            # Discard bytes before sync
            if idx > 0:
                if args.debug:
                    print(f"[DBG] skipped {idx} bytes to sync")
                serial_buffer = serial_buffer[idx:]

            # Need at least one full frame
            if len(serial_buffer) < BIN_FRAME_SIZE:
                break

            # Extract candidate frame
            frame = serial_buffer[:BIN_FRAME_SIZE]

            # Verify checksum: XOR of bytes [2..62] must match byte [63]
            ck = 0
            for i in range(2, 63):
                ck ^= frame[i]
            if ck != frame[63]:
                # Bad checksum — skip the sync bytes and try again
                _bin_badck += 1
                if args.debug:
                    print(f"[DBG] bad cksum, skipping 2 bytes")
                serial_buffer = serial_buffer[2:]
                continue

            # Frame is valid — unpack
            seq = frame[2]

            # Check for dropped frames
            if _bin_last_seq >= 0:
                expected = (_bin_last_seq + 1) & 0xFF
                dropped = (seq - expected) & 0xFF
                if dropped > 0 and dropped < 128:
                    _bin_lost_frames += dropped
                    if args.debug:
                        print(f"[DBG] lost {dropped} frame(s), seq jump {_bin_last_seq}->{seq}")
                elif dropped >= 128:
                    # 大跳变 —— mod 256 环绕统计不到具体数量，单独计次。
                    # 出现它基本可以断定固件中间重启过（seq 从 0 重新计数）。
                    _bin_resync += 1
                    if args.debug:
                        print(f"[DBG] big seq jump {_bin_last_seq}->{seq} (firmware reboot?)")

            _bin_last_seq = seq

            # Unpack 15 floats from bytes [3..62]
            for i in range(5):
                off = 3 + i * 12
                x, y, z = struct.unpack_from('<fff', frame, off)
                sid = SENSOR_IDS[i]
                sensor_data[sid] = np.array([x, y, z], dtype=np.float32)

            # Advance past this frame
            serial_buffer = serial_buffer[BIN_FRAME_SIZE:]
            n_parsed += 1
            _frames_parsed += 1

    except Exception as e:
        if args.debug:
            print(f"[DBG] binary read error: {e}")

    return n_parsed


# ============================================================
# Main update loop (called by QTimer every 20 ms)
# ============================================================
SCALE_FACTOR = 600.0       # µT→mm (5000µT → ~8.3mm arrow)
MAX_MAG_DISPLAY = 5000.0   # µT at which colour saturates to red
DELTA_SCALE_FACTOR = 60.0  # ΔB 模式: µT→mm（ΔB 通常几十~几百 µT，放大 10 倍才看得见）
DELTA_MAX_DISPLAY = 2000.0 # ΔB 模式颜色饱和阈值 (µT)

# 基线与显示模式：箭头默认画总磁场 B（磁铁静态基线主导，方向反映磁铁几何，
# 与接触无关）。按 Z 采集当前基线 B0，按 B 切到 ΔB=B−B0 —— 这才是形变/接触
# 信息，也是 6 维力模型的实际输入。Z 在每次开跑/零点漂移后按一下即可。
baseline = {sid: np.zeros(3) for sid in SENSOR_IDS}
show_delta = False
_frame_count = 0
_last_status_print = 0


def update():
    global _frame_count, _last_status_print, _frames_last

    # 1. Acquire data
    if SIM_MODE:
        sim_gen.generate()
    elif ser:
        if BINARY_MODE:
            read_binary_frames()
        else:
            frames = read_text_frames()
            for frame_line in frames:
                n = parse_text_frame(frame_line)
                if n > 0 and _frame_count == 0:
                    print(f"[DEBUG] First text frame: {n} sensors  raw={frame_line[:120]}...")

    # 2. Build direction lines (origin → tip per sensor)
    line_pts = []
    line_cols = []

    for sid in SENSOR_IDS:
        origin = SENSOR_POS[sid]
        raw = sensor_data[sid]
        if show_delta:
            vec = raw - baseline[sid]
            scale, maxmag = DELTA_SCALE_FACTOR, DELTA_MAX_DISPLAY
        else:
            vec = raw
            scale, maxmag = SCALE_FACTOR, MAX_MAG_DISPLAY
        mag = float(np.linalg.norm(vec))
        tip = origin + (vec / scale)
        color = magnitude_to_color(mag, max_mag=maxmag)
        line_pts.extend([origin, tip])
        line_cols.extend([color, color])

    arrows_item.setData(pos=np.array(line_pts), color=np.array(line_cols))

    # 3. Print magnitude summary every ~2 s
    _frame_count += 1
    now = time.time()
    if now - _last_status_print > 2.0:
        if show_delta:
            mags = [f"{np.linalg.norm(sensor_data[sid] - baseline[sid]):.0f}" for sid in SENSOR_IDS]
            mtag = "dB(µT)"
        else:
            mags = [f"{np.linalg.norm(sensor_data[sid]):.0f}" for sid in SENSOR_IDS]
            mtag = "mags(µT)"
        if BINARY_MODE:
            dt = max(now - _last_status_print, 1e-6)
            fps = (_frames_parsed - _frames_last) / dt
            _frames_last = _frames_parsed
            tag = (f"BIN f={fps:5.1f}Hz lost={_bin_lost_frames} badck={_bin_badck} "
                   f"resync={_bin_resync} skipB={_bin_skip_bytes}")
        else:
            tag = "TEXT"
        mode_str = "SIM" if SIM_MODE else tag
        print(f"[{mode_str}] frame={_frame_count}  {mtag}=[{', '.join(mags)}]")
        _last_status_print = now


# ============================================================
# Start
# ============================================================
# 丢弃 Qt 初始化期间串口积压的旧数据，避免开场统计出虚假丢帧
if ser and ser.is_open:
    ser.reset_input_buffer()

_last_status_print = time.time()   # fps 统计的基准时间
# ---- 键盘快捷键：Z=采集基线 B0（去零），B=切换 RAW / ΔB 显示 ----
def _on_key(event):
    global show_delta
    if event.key() == Qt.Key_Z:
        for sid in SENSOR_IDS:
            baseline[sid] = sensor_data[sid].copy()
        b0 = ', '.join(f"{np.linalg.norm(baseline[s]):.0f}" for s in SENSOR_IDS)
        print(f"[ZERO] 基线已采集  B0(µT)=[{b0}]  （再按 B 可看 ΔB）")
    elif event.key() == Qt.Key_B:
        show_delta = not show_delta
        mode = "ΔB = B − B0（形变/接触信息，模型实际输入）" if show_delta \
            else "RAW 总磁场（磁铁基线主导）"
        print(f"[MODE] 显示切换 → {mode}")
    else:
        event.ignore()

w.keyPressEvent = _on_key
w.setFocusPolicy(Qt.StrongFocus)

timer = QTimer()
timer.timeout.connect(update)
timer.start(20)  # 50 Hz refresh

if SIM_MODE:
    mode_label = "--sim (synthetic data)"
else:
    proto = "BINARY" if BINARY_MODE else "TEXT"
    mode_label = f"{args.port} @ {args.baud} [{proto}]"
print(f"eFlesh 3D Visualizer started — {mode_label}")
print("Controls: Left-drag=rotate  Right-drag=zoom  Middle-drag=pan")
print("          Z=capture baseline B0 (zero)   B=toggle RAW / ΔB display")
print("Close the window or press Ctrl+C to exit.")

try:
    sys.exit(app.exec_())
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial closed.")
