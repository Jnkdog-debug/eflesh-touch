"""
皮肤平面标定 —— 4 角 + 中心示教点拟合平面，建立皮肤坐标系 {S}。

约定:
  - 示教顺序: TL(左上), TR(右上), BR(右下), BL(左下), C(中心)
    （"上/左"是操作者视角看皮肤的方向，只要 4 角按逆/顺时针连续给即可）
  - x̂_S = normalize(TR − TL)，n = 平面法向（翻到 n·ẑ_base > 0，即朝上），
    ŷ_S = n × x̂_S（右手系）
  - 原点 = 4 角几何中心（中心示教点仅作校验）
  - rpy_probe: 第一次标记时的末端姿态，之后所有 move_pose 原样复用
    （控制路径零欧拉运算，规避 CALIBRATION.md 里的 rpy/四元数坑）

变换: p_base = R_bs @ p_skin + t_bs，p 单位 m；skin_to_base 接口用 mm 方便调用。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CORNER_NAMES = ("TL", "TR", "BR", "BL", "C")


def fit_plane(points_m: np.ndarray):
    """SVD 平面拟合 → (单位法向, 质心, RMS 残差 m)。法向翻到 z 分量 ≥ 0."""
    c = points_m.mean(axis=0)
    u, s, vt = np.linalg.svd(points_m - c)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    resid = (points_m - c) @ n
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return n, c, rms


@dataclass
class SkinFrame:
    R_bs: np.ndarray                 # (3,3) p_base = R @ p_skin + t
    t_bs: np.ndarray                 # (3,) 皮肤中心在臂基系 [m]
    rpy_probe: list                  # 示教姿态 [rad]，move_pose 原样用
    corners_base: np.ndarray         # (5,3) 示教点（TL,TR,BR,BL,C）[m]
    residual_mm: float               # 平面拟合 RMS
    meta: dict = field(default_factory=dict)

    # ---------- 变换 ----------
    def skin_to_base(self, x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
        p = np.array([x_mm, y_mm, z_mm]) * 1e-3
        return self.R_bs @ p + self.t_bs

    def base_to_skin(self, p_base_m: np.ndarray) -> np.ndarray:
        """臂基系 → [x_mm, y_mm, z_mm]（z 沿法向朝上为正）."""
        p = np.asarray(p_base_m, dtype=np.float64)
        return (self.R_bs.T @ (p - self.t_bs)) * 1e3

    # ---------- 校验 ----------
    def check_geometry(self, expect_side_mm: float | None = None) -> dict:
        """边长/对角线/中心偏差 —— 抓示教错角."""
        tl, tr, br, bl, c = self.corners_base
        side_top = np.linalg.norm(tr - tl) * 1e3
        side_right = np.linalg.norm(br - tr) * 1e3
        side_bot = np.linalg.norm(br - bl) * 1e3
        side_left = np.linalg.norm(tl - bl) * 1e3
        diag1 = np.linalg.norm(br - tl) * 1e3
        diag2 = np.linalg.norm(tr - bl) * 1e3
        sides = [side_top, side_right, side_bot, side_left]
        center_fit = (tl + tr + br + bl) / 4
        center_dev = np.linalg.norm(c - center_fit) * 1e3
        d = {"sides_mm": sides, "diag_mm": [diag1, diag2],
             "center_dev_mm": center_dev,
             "side_spread_mm": max(sides) - min(sides),
             "diag_diff_mm": abs(diag1 - diag2)}
        if expect_side_mm is not None:
            d["side_err_mm"] = [s - expect_side_mm for s in sides]
        return d

    # ---------- 存取 ----------
    def to_json_dict(self) -> dict:
        return {
            "R_bs": self.R_bs.tolist(),
            "t_bs_m": self.t_bs.tolist(),
            "rpy_probe_rad": list(self.rpy_probe),
            "corners_base_m": self.corners_base.tolist(),
            "residual_mm": self.residual_mm,
            "meta": self.meta,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "SkinFrame":
        return cls(
            R_bs=np.array(d["R_bs"], dtype=np.float64),
            t_bs=np.array(d["t_bs_m"], dtype=np.float64),
            rpy_probe=list(d["rpy_probe_rad"]),
            corners_base=np.array(d["corners_base_m"], dtype=np.float64),
            residual_mm=float(d["residual_mm"]),
            meta=d.get("meta", {}),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path) -> "SkinFrame":
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def build_skin_frame(corners_base: np.ndarray, rpy_probe, meta: dict | None = None) -> SkinFrame:
    """
    corners_base: (5,3) TL,TR,BR,BL,C（臂基系，m）→ SkinFrame。
    x̂ = TR−TL 归一化投影到平面，ŷ = n × x̂，原点 = 4 角中心。
    meta 里自动记录示教实际半幅宽 half_x_mm/half_y_mm（网格生成用，
    防止网格超出标记范围）。
    """
    corners_base = np.asarray(corners_base, dtype=np.float64)
    tl, tr, br, bl, c = corners_base
    n, centroid, rms = fit_plane(corners_base[:4])

    x_raw = tr - tl
    x_hat = x_raw - (x_raw @ n) * n
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(n, x_hat)

    R = np.column_stack([x_hat, y_hat, n])   # 列 = 皮肤轴在基系的表示
    t = (tl + tr + br + bl) / 4.0            # 原点 = 4 角中心（不是 SVD 质心）

    # 示教实际范围（取对边较小者，保守）
    side_top = float(np.linalg.norm(tr - tl))
    side_bot = float(np.linalg.norm(br - bl))
    side_right = float(np.linalg.norm(br - tr))
    side_left = float(np.linalg.norm(tl - bl))
    m = dict(meta or {})
    m.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    m["half_x_mm"] = min(side_top, side_bot) * 0.5e3
    m["half_y_mm"] = min(side_right, side_left) * 0.5e3

    return SkinFrame(R_bs=R, t_bs=t, rpy_probe=list(rpy_probe),
                     corners_base=corners_base, residual_mm=rms * 1e3, meta=m)
