#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手指按压分区测试 v2: 回车后开始压,结果带符号自动存 data/live_probe_last.txt"""
import glob, sys, pathlib, threading, time
from collections import deque
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from eflesh_calib.skin_stream import skin_reader, SkinStats
from eflesh_calib.util import Latest
from eflesh_calib.config import SKIN_BAUD

port = (sorted(glob.glob('/dev/ttyUSB*')) + sorted(glob.glob('/dev/ttyACM*')) or [None])[0]
if not port: raise SystemExit('没找到串口')
print(f'皮肤串口: {port}')
q = deque(maxlen=200000); st = SkinStats(); lat = Latest(); stop = threading.Event()
threading.Thread(target=skin_reader, args=(port, SKIN_BAUD, q, st, lat, stop), daemon=True).start()
time.sleep(1.5); print(st.line(), '\n')

def collect(dur):
    out, t0 = [], time.time()
    while time.time() - t0 < dur:
        while q: out.append(q.popleft())
        time.sleep(0.05)
    while q: out.append(q.popleft())
    return np.array([o[2] for o in out])

POS = ['中心', '左上角', '右上角', '左下角', '右下角']
sig, mag = {}, {}
for name in POS:
    input(f'>>> 手指悬在【{name}】上方,按回车后【立刻压实按住 6 秒】...')
    q.clear()
    f = collect(6.0)
    n = len(f)
    base = np.median(f[:max(n//4,1)], axis=0)       # 前1.5s=未压
    press = np.median(f[n//2:], axis=0)              # 后3s=按压稳态
    dB = press - base
    sig[name] = dB.copy(); mag[name] = np.abs(dB).sum(axis=1)
    print(f'  [{name}] {n}帧  Σ|ΔB|: ' + '  '.join(f'S{s+1}={mag[name][s]:.0f}' for s in range(5)))

print('\n====== Σ|ΔB| (µT) ======')
print('位置    ' + ''.join(f'S{s+1:>7d}' for s in range(5)))
for name in POS:
    print(f'{name:5s} ' + ''.join(f'{mag[name][s]:7.0f}' for s in range(5)))
print('\n====== ΔBz 符号 (极性判定: 压哪颗哪颗的主响应z为负=磁体反装) ======')
for name in POS:
    print(f'{name:5s} ' + ''.join(f'{sig[name][s,2]:+8.0f}' for s in range(5)))

out = pathlib.Path(__file__).resolve().parent / 'data' / 'live_probe_last.txt'
with open(out, 'w') as fp:
    fp.write('pos ' + ' '.join(f'S{s+1}_x S{s+1}_y S{s+1}_z' for s in range(5)) + '\n')
    for name in POS:
        fp.write(name + ' ' + ' '.join(f'{v:.1f}' for v in sig[name].ravel()) + '\n')
print(f'\n已存: {out}')
stop.set()
