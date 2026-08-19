"""
位姿轮询线程 —— 一次 rm_get_current_arm_state() 同时拆出 pose/joints，
避免 get_pose()+get_joints() 两次往返。

输出: (t_ns, xyz f8(3) [m], rpy f8(3) [rad], joints f8(7) [deg]) —— 臂基坐标系。
"""

import threading
from collections import deque

import numpy as np

from .clock import now_ns
from .config import POSE_RATE_HZ
from .util import Latest, rate_limited_loop


def pose_poller(arm, out_q: deque, latest: Latest, stop: threading.Event,
                rate_hz: float = POSE_RATE_HZ):
    """位姿轮询线程主体。arm: 已连接的 RM75Robot（直接用其 robot 句柄）."""

    def body():
        state = arm.robot.rm_get_current_arm_state()
        if state[0] != 0:
            return
        d = state[1]
        pose = d["pose"]      # [x,y,z,rx,ry,rz] m + rad
        joints = d["joint"]   # deg
        t = now_ns()
        xyz = np.asarray(pose[:3], dtype=np.float64)
        rpy = np.asarray(pose[3:6], dtype=np.float64)
        j = np.asarray(joints, dtype=np.float64)
        out_q.append((t, xyz, rpy, j))
        latest.set((t, xyz, rpy))

    rate_limited_loop(rate_hz, stop, body)
