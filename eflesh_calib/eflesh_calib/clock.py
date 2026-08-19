"""统一时钟：所有流共用一个单调时钟（time.monotonic_ns）."""

import time
from datetime import datetime, timezone


class SessionClock:
    """会话时钟：t0 锚定后，to_rel() 把单调时间转为相对会话起点的秒."""

    def __init__(self):
        self.t0_ns = time.monotonic_ns()
        self.t0_utc = datetime.now(timezone.utc).isoformat()

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def to_rel(self, t_ns: int) -> float:
        """单调 ns → 相对 t0 的秒（HDF5 里存的就是这个）."""
        return (t_ns - self.t0_ns) * 1e-9

    def rel_now(self) -> float:
        return self.to_rel(self.now_ns())

    def attrs(self) -> dict:
        return {"t0_monotonic_ns": self.t0_ns, "t0_utc": self.t0_utc}


def now_ns() -> int:
    """模块级便捷函数（线程里直接用）."""
    return time.monotonic_ns()
