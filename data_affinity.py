"""PDBbind 亲和力任务的数据侧组件。

与 data.py 的 EnzymeDataset 并列，不复用 KineticsScaler：
kcat/Km 需要 log10 变换与 ratio 分位数裁剪，而亲和力标签已经是 -log10 空间，
只需一次单列标准化。
"""
from __future__ import annotations

import json

import numpy as np


class AffinityScaler:
    """单列标准化。标签已是 -log10(Kd/Ki/IC50)，不做 log 变换、不做分位数裁剪。"""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, y) -> "AffinityScaler":
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("AffinityScaler.fit 收到的标签里没有任何有限值")
        self.mean_ = float(finite.mean())
        std = float(finite.std())
        self.scale_ = std if std > self.eps else 1.0
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("AffinityScaler 尚未 fit，请先调用 fit()")

    def transform(self, y) -> np.ndarray:
        self._check_fitted()
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        out = np.full(values.shape, np.nan, dtype=np.float32)
        finite = np.isfinite(values)
        out[finite] = ((values[finite] - self.mean_) / self.scale_).astype(np.float32)
        return out

    def inverse_transform(self, z) -> np.ndarray:
        self._check_fitted()
        values = np.asarray(z, dtype=np.float64).reshape(-1)
        return (values * self.scale_ + self.mean_).astype(np.float32)

    def save(self, path: str) -> None:
        self._check_fitted()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"mean": self.mean_, "scale": self.scale_, "eps": self.eps},
                      handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "AffinityScaler":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        scaler = cls(eps=payload.get("eps", 1e-8))
        scaler.mean_ = float(payload["mean"])
        scaler.scale_ = float(payload["scale"])
        return scaler
