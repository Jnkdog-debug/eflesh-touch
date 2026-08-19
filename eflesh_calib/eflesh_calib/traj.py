"""
按压轨迹生成 —— 网格 + 按压计划 + 皮肤系→基系路点。

默认 9×9 @5mm 覆盖 ±20mm（40×40 皮肤）；--pitch 2 → 21×21（论文级，notes §5.1）。
按压顺序默认随机（打散位置相关的温漂），可选蛇形（连续采集更省时）。
"""

import random
from dataclasses import dataclass

import numpy as np

from .calib_frame import SkinFrame
from .config import GRID_HALF_MM, GRID_PITCH_MM, HOVER_MM


@dataclass
class PressTarget:
    press_id: str
    ix: int
    iy: int
    gx_mm: float          # 皮肤系网格坐标
    gy_mm: float
    depth_mm: float
    batch: str

    def label_xy(self) -> tuple[float, float]:
        return self.gx_mm, self.gy_mm


def make_grid(pitch_mm: float = GRID_PITCH_MM, half_mm: float = GRID_HALF_MM,
              subset_n: int | None = None) -> list[tuple[int, int, float, float]]:
    """
    网格点列表 [(ix, iy, x_mm, y_mm)]，中心对称。
    subset_n=3 → 3×3 中心子集（dry-run / M1 用）。
    """
    if subset_n is not None:
        k = (subset_n - 1) / 2
        xs = np.linspace(-k * pitch_mm, k * pitch_mm, subset_n)
    else:
        n = int(2 * half_mm / pitch_mm) + 1
        xs = np.linspace(-half_mm, half_mm, n)
    idx = np.arange(len(xs)) - (len(xs) - 1) // 2
    pts = []
    for ix, x in zip(idx, xs):
        for iy, y in zip(idx, xs):
            pts.append((int(ix), int(iy), float(x), float(y)))
    return pts


def press_plan(grid=None, depths_mm=(0.5, 1.0, 1.5), batch: str = "b1",
               order: str = "random", seed: int = 0) -> list[PressTarget]:
    """网格 × 深度 → 按压计划。order: random | serpentine."""
    if grid is None:
        grid = make_grid()
    targets = []
    for (ix, iy, gx, gy) in grid:
        for d in depths_mm:
            pid = f"{ix:+03d}{iy:+03d}_d{d:.1f}"
            targets.append(PressTarget(pid, ix, iy, gx, gy, float(d), batch))

    if order == "random":
        rng = random.Random(seed)
        rng.shuffle(targets)
    elif order == "serpentine":
        targets.sort(key=lambda t: (t.depth_mm, t.iy, t.ix if t.iy % 2 == 0 else -t.ix))
    else:
        raise ValueError(f"unknown order: {order}")
    return targets


def waypoints(target: PressTarget, sf: SkinFrame, hover_mm: float = HOVER_MM) -> dict:
    """
    路点（基系 pose 列表，可直接送 move_pose）:
      above: 点位上方 hover_mm（每压重算，touch-off 前的粗略目标）
    下探/按压的 z 由 PressMachine 按 touch-off 实测动态生成，不在此定死。
    """
    p = sf.skin_to_base(target.gx_mm, target.gy_mm, hover_mm)
    return {"above": [float(p[0]), float(p[1]), float(p[2])] + list(sf.rpy_probe)}
