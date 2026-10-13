"""PLM 骨干的加载、形状解析与 LoRA 注入。

设计约束：纯解析函数只依赖 torch，使无 GPU / 无 peft / 无 esm 的环境也能对它们做单元测试。
peft 与 esm 一律在函数内部 lazy import。
"""
from __future__ import annotations

import re
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


# 覆盖 ESM-2 / BERT(MoLFormer 系) / T5 三套已知命名。
# ESMC-6B 的 remote code 命名由 tools/probe_backbone_modules.py 确认；
# 若出现未覆盖的名字，调用方通过 suffixes 参数注入，不要改这里。
DEFAULT_LORA_SUFFIXES: tuple[str, ...] = (
    # ESM-2 / BERT 风格
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
    # T5 风格
    "SelfAttention.q",
    "SelfAttention.k",
    "SelfAttention.v",
    "SelfAttention.o",
    "DenseReluDense.wi_0",
    "DenseReluDense.wi_1",
    "DenseReluDense.wo",
    # LLaMA / 通用 projector 风格
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "Wqkv",
)

# 前导的 (?:^|\.) 不可省：层容器可能是根级属性（如 `layers.0.mixer`），
# 此时限定名里没有前导点，只写 `\.layer\.` 会解析不出层号而被 layers 过滤掉。
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layer|layers|block|blocks|h)\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    """从限定名里取最后一处层号；取不到返回 None。"""
    hits = _LAYER_INDEX_RE.findall(name)
    return int(hits[-1]) if hits else None


def discover_lora_targets(
    model: nn.Module,
    *,
    layers: Sequence[int] | None = None,
    suffixes: Sequence[str] | None = None,
    min_hits: int = 1,
) -> list[str]:
    """在任意骨干上发现可注入 LoRA 的线性层名。

    不假定具体架构：先按后缀白名单筛出候选 Linear，再按层号过滤。
    命中不足时抛出断言，并把模型里真实出现的 Linear 末段名打印出来。
    """
    suffix_tuple = tuple(suffixes) if suffixes else DEFAULT_LORA_SUFFIXES
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n]

    matched = [n for n in linear_names if n.endswith(suffix_tuple)]

    if layers is not None:
        wanted = {int(i) for i in layers}
        matched = [n for n in matched if (_layer_index(n) in wanted)]

    targets = sorted(set(matched))

    if len(targets) < min_hits:
        seen_leaves = sorted({n.rsplit(".", 1)[-1] for n in linear_names})
        sample = sorted(linear_names)[:30]
        raise AssertionError(
            f"LoRA target 发现失败：命中 {len(targets)} 个，要求 >= {min_hits}。\n"
            f"  匹配后缀 = {suffix_tuple}\n"
            f"  限定层号 = {layers}\n"
            f"  模型内实际出现的 Linear 末段名 = {seen_leaves}\n"
            f"  前 30 个 Linear 限定名 = {sample}\n"
            "若这是 ESMC-6B，请先运行 tools/probe_backbone_modules.py，"
            "再用 suffixes=（或 --lora_target_suffixes）注入正确的名字。"
        )

    return targets
