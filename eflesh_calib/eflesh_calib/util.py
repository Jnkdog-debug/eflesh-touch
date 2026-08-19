"""线程共享工具：Latest<T> 槽、限频循环、时间匹配."""

import threading
import time
from collections import deque

import numpy as np


class FanoutQueue:
    """一个入口多个出口：reader 线程 append 一次，本地队列和 recorder 同时收到."""

    def __init__(self, queues):
        self.queues = list(queues)

    def append(self, item):
        for q in self.queues:
            q.append(item)


class Latest:
    """单值共享槽：写者覆盖、读者取最近值（带锁，值不可变时安全）."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._t_ns = None

    def set(self, value):
        t = time.monotonic_ns()
        with self._lock:
            self._value = value
            self._t_ns = t

    def get(self):
        with self._lock:
            return self._value

    def age_s(self) -> float:
        with self._lock:
            if self._t_ns is None:
                return float("inf")
            return (time.monotonic_ns() - self._t_ns) * 1e-9


def rate_limited_loop(rate_hz: float, stop: threading.Event, body, setup=None):
    """
    固定频率循环（deadline 调度，不累积漂移）:
      next_t += 1/rate; body(); sleep(max(0, next_t - now))
    body 抛异常时打印并继续（轮询线程不能死）.
    """
    period = 1.0 / rate_hz
    if setup:
        setup()
    next_t = time.monotonic()
    while not stop.is_set():
        try:
            body()
        except Exception as e:  # noqa: BLE001 — 轮询线程要活着
            print(f"[loop@{rate_hz:.0f}Hz] error: {e}")
        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            stop.wait(sleep)
        else:
            next_t = time.monotonic()  # 落后太多则重新锚定


def drain(q: deque):
    """非阻塞取空 deque."""
    items = []
    while True:
        try:
            items.append(q.popleft())
        except IndexError:
            return items


def match_nearest(ts_query: np.ndarray, ts_ref: np.ndarray) -> np.ndarray:
    """对每个 ts_query 返回最近的 ts_ref 下标（两者都已升序）."""
    idx = np.searchsorted(ts_ref, ts_query)
    idx = np.clip(idx, 1, len(ts_ref) - 1)
    left, right = idx - 1, idx
    pick = np.where(
        np.abs(ts_query - ts_ref[left]) <= np.abs(ts_ref[right] - ts_query), left, right
    )
    return pick


def interp_to(ts_query: np.ndarray, ts_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """把 (ts_ref, values) 线性插值到 ts_query；越界取端点值."""
    out = np.empty((len(ts_query), values.shape[1]), dtype=values.dtype)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(ts_query, ts_ref, values[:, c])
    return out
