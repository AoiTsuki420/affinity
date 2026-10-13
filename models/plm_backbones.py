"""PLM 骨干的加载、形状解析与 LoRA 注入。

设计约束：纯解析函数只依赖 torch，使无 GPU / 无 peft / 无 esm 的环境也能对它们做单元测试。
peft 与 esm 一律在函数内部 lazy import。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

KIND_ESMC = "esmc"
KIND_MOLFORMER = "molformer"
KIND_ESM2 = "esm2"
KIND_T5 = "t5"
SUPPORTED_KINDS = (KIND_ESMC, KIND_MOLFORMER, KIND_ESM2, KIND_T5)

_HIDDEN_SIZE_FIELDS = ("hidden_size", "d_model", "embedding_dim", "n_embd", "dim")
_NUM_LAYER_FIELDS = ("num_hidden_layers", "num_layers", "n_layers", "n_layer", "num_blocks")


def resolve_hidden_size(config: Any) -> int:
    """按已知的 HF config 字段名依次尝试，全部失败时抛出带字段清单的错误。"""
    for field in _HIDDEN_SIZE_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    available = sorted(vars(config)) if hasattr(config, "__dict__") else []
    raise ValueError(
        f"无法从 config 推断 hidden size，依次尝试过 {_HIDDEN_SIZE_FIELDS}；"
        f"config 实际字段: {available}"
    )


def resolve_num_layers(config: Any, model: nn.Module | None = None) -> int:
    """先看 config 字段，再回退到模型结构里的层容器。"""
    for field in _NUM_LAYER_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, int) and value > 0:
            return int(value)

    if model is not None:
        candidates = (
            ("encoder", "layer"),
            ("encoder", "block"),
            (None, "layers"),
            (None, "h"),
        )
        for holder_name, attr in candidates:
            holder = model if holder_name is None else getattr(model, holder_name, None)
            if holder is None:
                continue
            container = getattr(holder, attr, None)
            if container is not None and hasattr(container, "__len__") and len(container) > 0:
                return int(len(container))

    raise ValueError(
        f"无法推断层数：config 无 {_NUM_LAYER_FIELDS}，"
        "模型也无 encoder.layer / encoder.block / layers / h"
    )


def resolve_lora_layers(
    num_layers: int,
    policy: str,
    last_k: int,
    explicit: Sequence[int] | None = None,
) -> list[int]:
    """把 LoRA 层选择策略解析成显式的层号列表。"""
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers 必须是正整数，得到 {num_layers!r}")

    if policy in ("all", "full", "everything"):
        return list(range(num_layers))

    if policy == "last_k":
        k = max(1, min(int(last_k), num_layers))
        return list(range(num_layers - k, num_layers))

    if policy == "explicit":
        if not explicit:
            raise ValueError("policy='explicit' 时必须提供 explicit 层号列表")
        picked = sorted({int(i) for i in explicit if 0 <= int(i) < num_layers})
        if not picked:
            raise ValueError(
                f"explicit 层号 {list(explicit)} 在 num_layers={num_layers} 范围内无有效项"
            )
        return picked

    raise ValueError(f"未知的 LoRA 层策略: {policy!r}（支持 all / last_k / explicit）")
