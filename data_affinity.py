"""PDBbind 亲和力任务的数据侧组件。

与 data.py 的 EnzymeDataset 并列，不复用 KineticsScaler：
kcat/Km 需要 log10 变换与 ratio 分位数裁剪，而亲和力标签已经是 -log10 空间，
只需一次单列标准化。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


class AffinityDataset(Dataset):
    """PDBbind 亲和力数据集：单标签列，可选 source（亲和力类型）。"""

    SEQ_COLUMNS = ("Sequence", "sequence", "Protein Sequence", "protein_sequence")
    SMI_COLUMNS = ("Smiles", "SMILES", "smiles", "Substrate SMILES", "Substrate_SMILES")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        label_col: str = "pKaffinity",
        scaler: AffinityScaler | None = None,
        supervised: bool = True,
        return_source: bool = False,
        source_col: str = "affinity_type",
    ):
        if label_col not in df.columns:
            raise ValueError(
                f"AffinityDataset 缺少标签列 {label_col!r}；"
                f"实际列: {sorted(df.columns)}"
            )

        labels = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(labels)
        if not finite.any():
            raise ValueError(
                f"标签列 {label_col!r} 里没有任何有限值，共 {len(df)} 行。"
            )

        self.df = df.loc[finite].reset_index(drop=True)
        raw_labels = labels[finite]

        self.scaler = scaler
        if scaler is not None:
            self.labels = scaler.transform(raw_labels).astype(np.float32)
        else:
            self.labels = raw_labels.astype(np.float32)

        self.seq_col = self._first_present(df, self.SEQ_COLUMNS, "蛋白序列")
        self.smi_col = self._first_present(df, self.SMI_COLUMNS, "配体 SMILES")

        self.return_source = bool(return_source)
        self.source_col = source_col
        if self.return_source and source_col not in df.columns:
            raise ValueError(f"return_source=True 但缺少列 {source_col!r}")

        self.supervised = bool(supervised)

    @staticmethod
    def _first_present(df: pd.DataFrame, candidates, what: str) -> str:
        for name in candidates:
            if name in df.columns:
                return name
        raise ValueError(f"找不到{what}列，尝试过 {candidates}；实际列: {sorted(df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        seq = str(row[self.seq_col])
        smi = str(row[self.smi_col])
        label = torch.tensor(float(self.labels[index]), dtype=torch.float32)

        if self.return_source:
            return seq, smi, label, str(row[self.source_col])
        return seq, smi, label
