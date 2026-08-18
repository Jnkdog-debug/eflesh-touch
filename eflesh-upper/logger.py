#!/usr/bin/env python3
"""
eFlesh binary-frame logger — 闪烁/全零帧诊断工具（无 Qt，纯终端）。
读取 64 字节二进制协议，统计每个传感器的读失败（帧内数值恰为 0,0,0）：
失败率、连发次数、最长连发、几个传感器同时失败 —— 用于区分
单传感器接线问题 vs 公共总线/时序问题 vs 固件重启。

Usage:
  python3 logger.py                          # 30s @ /dev/ttyUSB0 921600
  python3 logger.py -t 60 --port /dev/ttyUSB0
"""

import sys
import time
import struct
import argparse
import serial

ap = argparse.ArgumentParser()
ap.add_argument('--port', default='/dev/ttyUSB0')
ap.add_argument('--baud', type=int, default=921600)
ap.add_argument('-t', '--duration', type=float, default=30.0)
args = ap.parse_args()

N = 5
ser = serial.Serial(args.port, args.baud, timeout=0.05)
time.sleep(0.2)
ser.reset_input_buffer()   # 丢掉开串口时复位产生的启动垃圾

buf = b''
zero_count = [0] * N       # 该传感器读数恰为 0,0,0 的帧数
cur_run = [0] * N          # 当前连续失败长度
max_run = [0] * N          # 最长连续失败
bursts = [0] * N           # 失败连发次数（进入失败的次数）
k_dist = [0] * (N + 1)     # 每帧恰有 k 个传感器失败的帧数分布
all_zero = 0               # 5 个全失败的帧数
frames = 0
badck = 0
skipB = 0
lost = 0
resyncs = 0
last_seq = -1
t_start = None
last_frame_t = None
max_gap = 0.0
mag_last = [None] * N


def handle(frame):
    global frames, last_seq, lost, resyncs, all_zero
    global t_start, last_frame_t, max_gap
    now = time.time()
    if t_start is None:
        t_start = now
    if last_frame_t is not None:
        max_gap = max(max_gap, now - last_frame_t)
    last_frame_t = now

    seq = frame[2]
    if last_seq >= 0:
        d = (seq - (last_seq + 1)) & 0xFF
        if 0 < d < 128:
            lost += d
        elif d >= 128:
            resyncs += 1
    last_seq = seq
    frames += 1

    k = 0
    for i in range(N):
        x, y, z = struct.unpack_from('<fff', frame, 3 + i * 12)
        if x == 0.0 and y == 0.0 and z == 0.0:
            zero_count[i] += 1
            k += 1
            cur_run[i] += 1
            if cur_run[i] == 1:
                bursts[i] += 1
            max_run[i] = max(max_run[i], cur_run[i])
            mag_last[i] = 0.0
        else:
            cur_run[i] = 0
            mag_last[i] = (x * x + y * y + z * z) ** 0.5
    k_dist[k] += 1
    if k == N:
        all_zero += 1


def summary():
    dur = max(time.time() - t_start, 1e-6) if t_start else 1.0
    print("\n================ SUMMARY ================")
    print(f"时长 {dur:.1f}s   帧数 {frames}   平均 {frames / dur:.1f} Hz")
    print(f"lost={lost}  resync(大跳变/重启)={resyncs}  badck={badck}  "
          f"skipB={skipB}  最大帧间隔={max_gap * 1000:.0f}ms")
    print(f"5 个全失败的帧: {all_zero} ({100 * all_zero / max(frames, 1):.1f}%)")
    print("各传感器失败率 / 连发次数 / 最长连发(帧):")
    for i in range(N):
        print(f"  S{i + 1}: {100 * zero_count[i] / max(frames, 1):5.1f}%   "
              f"{bursts[i]:4d} 次   最长 {max_run[i]} 帧")
    print("每帧失败传感器个数分布 (k=同时失败的个数):")
    print("  " + "  ".join(f"k={k}:{k_dist[k]}" for k in range(N + 1)))
    print("""
怎么读这份结果：
  * 某一个传感器失败率高、其余≈0   → 那一路 CS/接线/焊点问题
  * k=5 占主导（全体一起失败）且 resync/skipB 不涨 → 公共 SPI 总线/固件时序问题
  * k=5 伴随着 resync 或 skipB 暴增   → 固件在重启（供电/复位），查 USB 供电
  * badck > 0                        → 串口链路误码（波特率/线材）
""")


try:
    t_end = time.time() + args.duration
    next_stat = time.time() + 2.0
    while time.time() < t_end:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        while True:
            if len(buf) < 64:
                if len(buf) > 4096:      # 很久找不到同步头
                    skipB += len(buf) - 1
                    buf = buf[-1:]
                break
            idx = buf.find(b'\xaa\x55')
            if idx < 0:
                skipB += len(buf) - 1
                buf = buf[-1:]
                continue
            if idx:
                skipB += idx
                buf = buf[idx:]
            if len(buf) < 64:
                break
            frame = buf[:64]
            ck = 0
            for b in frame[2:63]:
                ck ^= b
            if ck != frame[63]:
                badck += 1
                buf = buf[2:]
                continue
            handle(frame)
            buf = buf[64:]
        if t_start and time.time() >= next_stat:
            el = time.time() - t_start
            zpc = ', '.join(f'{100 * z / max(frames, 1):.0f}' for z in zero_count)
            mgs = ', '.join(f'{m:.0f}' if m is not None else '-' for m in mag_last)
            print(f"[{el:5.1f}s] f={frames / el:5.1f}Hz lost={lost} badck={badck} "
                  f"skipB={skipB}  失败%=[{zpc}]  mags=[{mgs}]")
            next_stat += 2.0
except KeyboardInterrupt:
    print("\n(Ctrl+C)")
finally:
    summary()
    ser.close()
