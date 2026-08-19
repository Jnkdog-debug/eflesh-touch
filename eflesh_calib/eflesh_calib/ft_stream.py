"""
六维力轮询线程 —— 复用 rm75_curobo_demo 的 RM75Robot.get_force_data()。

输出: (t_ns, wrench f4(6), ret) —— 原始 FT 帧（力坐标系），不做任何变换，
皮肤系变换留到 pipeline（可随标定更新重算）。
"""

import threading
from collections import deque

import numpy as np

from .clock import now_ns
from .config import FT_RATE_HZ
from .util import Latest, rate_limited_loop


def ft_poller(arm, out_q: deque, latest: Latest, stop: threading.Event,
              rate_hz: float = FT_RATE_HZ):
    """
    FT 轮询线程主体。arm: 已连接的 RM75Robot。

    取 zero_force_data（系统外受力, N/Nm）—— force_data 是原始数据
    （含工具重量/内部偏置, 清零对它无效）。ret != 0 的帧也记录（可诊断）。
    注意: 绝不调用 rm_set_force_sensor —— 那是重心标定, 会移动机械臂!
    """

    n_ok = [0]
    n_err = [0]

    def body():
        result = arm.robot.rm_get_force_data()
        ret = result[0]
        d = result[1]
        w6 = d.get("zero_force_data", d.get("force_data", [0] * 6)) \
            if isinstance(d, dict) else list(d)
        t = now_ns()
        w = np.asarray(w6, dtype=np.float32)
        if ret != 0:
            n_err[0] += 1
        else:
            n_ok[0] += 1
        out_q.append((t, w, int(ret)))
        latest.set((t, w, ret))

    rate_limited_loop(rate_hz, stop, body)
    return n_ok[0], n_err[0]


def read_fz(latest_ft: Latest):
    """便捷：从 latest 槽取当前 Fz（N）。无数据返回 None."""
    v = latest_ft.get()
    if v is None:
        return None
    _, w, ret = v
    if ret != 0:
        return None
    return float(w[2])
