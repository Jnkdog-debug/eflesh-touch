"""模型 —— L0 Ridge 基线 + L1 MLP（官方 characterization/train.py 同构，多任务单头）.

Ridge 优先用 sklearn；环境没有 sklearn 时退回 numpy 正规方程实现（等价，
仅少了些数值优化，量级 15 维毫无差别）—— 避免为训个基线污染共享 conda 环境。
"""

import numpy as np
import torch
import torch.nn as nn


def _ridge_fit(X, y, alpha):
    """闭式 Ridge: W = (XᵀX + αI)⁻¹ Xᵀy，带截距（先中心化）."""
    xm, ym = X.mean(0), y.mean(0)
    Xc, Yc = X - xm, y - ym
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, xm, ym


def _ridge_predict(W, xm, ym, X):
    return (X - xm) @ W + ym


class _NumpyRidge:
    """逐输出 5 折 CV 选 alpha（sklearn 缺席时的等价实现）."""

    ALPHAS = (0.1, 1.0, 10.0, 100.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        folds = np.array_split(idx, 5)
        self.models = []
        for c in range(y.shape[1]):
            best_mse, best_alpha = np.inf, 1.0
            for alpha in self.ALPHAS:
                mses = []
                for f in folds:
                    va = f
                    tr = np.setdiff1d(idx, va)
                    W, xm, ym = _ridge_fit(X[tr], y[tr, c:c + 1], alpha)
                    pred = _ridge_predict(W, xm, ym, X[va])
                    mses.append(float(np.mean((pred[:, 0] - y[va, c]) ** 2)))
                mse = float(np.mean(mses))
                if mse < best_mse:
                    best_mse, best_alpha = mse, alpha
            W, xm, ym = _ridge_fit(X, y[:, c:c + 1], best_alpha)
            self.models.append((W, xm, ym, best_alpha))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cols = [_ridge_predict(W, xm, ym, X)[:, 0] for W, xm, ym, _ in self.models]
        return np.stack(cols, axis=1)


class RidgeMultiOutput:
    """L0 线性基线: 逐输出 5 折 CV 选 alpha。物理下限/可解释参照."""

    def __init__(self):
        try:
            import sklearn.linear_model  # noqa: F401
            self._impl = "sklearn"
        except ImportError:
            self._impl = "numpy"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self._impl == "sklearn":
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import KFold
            best = []
            for c in range(y.shape[1]):
                best_alpha, best_mse = 1.0, np.inf
                for alpha in (0.1, 1.0, 10.0, 100.0):
                    mses = []
                    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
                        m = Ridge(alpha=alpha).fit(X[tr], y[tr, c])
                        mses.append(np.mean((m.predict(X[va]) - y[va, c]) ** 2))
                    mse = float(np.mean(mses))
                    if mse < best_mse:
                        best_mse, best_alpha = mse, alpha
                best.append(Ridge(alpha=best_alpha).fit(X, y[:, c]))
            self.models = best
        else:
            self._m = _NumpyRidge().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl == "sklearn":
            return np.stack([m.predict(X) for m in self.models], axis=1)
        return self._m.predict(X)


class MLP(nn.Module):
    """15(+历史) → 128 → 128 → out（与官方 train.py 结构一致，ReLU）."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
