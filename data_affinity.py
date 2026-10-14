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
    """PDBbind 亲和力数据集：单标签列，可选 source（亲和力类型）。

    ``return_source`` 仅供离线分析（例如按 Kd/Ki/IC50 分组统计）。置 True 时
    ``__getitem__`` 返回 4 元组 ``(seq, smi, y, source)``，而这个 4 元组落进
    ``utils.unpack_batch`` 时会走 ``len(batch) == 4`` 分支，把第 4 个元素当成
    **row id** 绑定（utils.py:774-776），即把亲和力类型字符串静默错绑成
    ``row_ids``。因此训练/推理 loader 必须保持默认的 ``return_source=False``
    （3 元组 ``(seq, smi, y)``，走 utils.py:771-773）。

    ``supervised`` 只是接受并保存，affinity 通路始终携带标签，不会因为
    ``supervised=False`` 而省略 ``y``。

    构造期会过滤两处坏行：标签非有限的行，以及 ``Sequence``/``Smiles`` 缺失的行
    （后者经 ``str()`` 会变成字面量 ``"nan"``，见 ``__init__`` 里的注释）。过滤是
    带告警的，不是静默的；``len(self)`` 与 ``self.df`` 始终一致。
    """

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

        # 列名要在过滤之前解析：下面的输入缺失过滤依赖它们。
        self.seq_col = self._first_present(df, self.SEQ_COLUMNS, "蛋白序列")
        self.smi_col = self._first_present(df, self.SMI_COLUMNS, "配体 SMILES")

        # 输入缺失与标签缺失同源地过滤。`__getitem__` 用 str(row[col]) 取值，所以 NaN
        # 会变成**字面量字符串 "nan"** 送进编码器——既不是真实序列也不是真实分子，
        # 而且不会抛错，只会静默污染训练（AffinityScaler.fit 对标签里的 NaN 已经是
        # 这个策略：告警 + 过滤）。放在构造期过滤可以让 len(self) 与 self.df 保持一致；
        # 注意 PredictionRecorder 的容量来自 args.batch_size * n_batches 而不是
        # len(dataset)，所以少掉的这几行不会让 recorder 越界。
        inputs_present = df[self.seq_col].notna() & df[self.smi_col].notna()
        dropped = int((finite & ~inputs_present.to_numpy()).sum())
        keep = finite & inputs_present.to_numpy()
        if dropped:
            print(
                f"[AffinityDataset] 丢弃 {dropped} 行：{self.seq_col!r} 或 "
                f"{self.smi_col!r} 缺失（str(NaN) 会变成字面量 \"nan\" 送进编码器）。"
            )

        self.df = df.loc[keep].reset_index(drop=True)
        raw_labels = labels[keep]

        self.scaler = scaler
        if scaler is not None:
            self.labels = scaler.transform(raw_labels).astype(np.float32)
        else:
            self.labels = raw_labels.astype(np.float32)

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
