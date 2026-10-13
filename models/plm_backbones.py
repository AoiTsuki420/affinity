"""PLM 骨干的加载、形状解析与 LoRA 注入。

设计约束：纯解析函数只依赖 torch，使无 GPU / 无 peft / 无 esm 的环境也能对它们做单元测试。
peft 与 esm 一律在函数内部 lazy import。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import torch
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


# 覆盖 ESM-2 / BERT(MoLFormer 系) / T5 三套已知命名，外加 spec §6.1 点名的裸末段形式。
# ESMC-6B 的 remote code 命名由 tools/probe_backbone_modules.py 确认；
# 若出现未覆盖的名字，调用方通过 suffixes 参数注入，不要改这里。
#
# spec §6.1 的草图还列了裸 q / k / v / o 与裸 dense，这里是**故意不加**的：单字母末段名
# 与裸 dense 会命中无关模块，而 SelfAttention.q / q_proj / attention.self.query 和
# out_proj / attention.output.dense 已经覆盖真实的 ESM-2 / BERT / T5 路径。若探针在
# 集群上发现确实需要它们，用 --lora_target_suffixes 注入，不要往这里加。
DEFAULT_LORA_SUFFIXES: tuple[str, ...] = (
    # ESM-2 / BERT 风格
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
    # 裸末段形式：backbone 把 qkv 直接挂在 attn 下（layers.0.attn.query）时，
    # 上面的 attention.self.* 复合名够不着。
    "query",
    "key",
    "value",
    # T5 风格
    "SelfAttention.q",
    "SelfAttention.k",
    "SelfAttention.v",
    "SelfAttention.o",
    "DenseReluDense.wi_0",
    "DenseReluDense.wi_1",
    "DenseReluDense.wo",
    # 裸末段形式：T5 系 FFN 挂在别的前缀下（layer.1.mlp.wi_0）时用。
    "wi_0",
    "wi_1",
    "wo",
    # LLaMA / 通用 projector 风格
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "Wqkv",
    # 通用 MLP 命名（spec §6.1 列入）。
    "fc1",
    "fc2",
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


_KIND_KEYWORDS = (
    ("esmc", KIND_ESMC),
    ("molformer", KIND_MOLFORMER),
    ("mol_former", KIND_MOLFORMER),
    ("molt5", KIND_T5),
    ("esm2", KIND_ESM2),
)

_CONFIG_MODEL_TYPE_MAP = {
    "esmc": KIND_ESMC,
    "molformer": KIND_MOLFORMER,
    "t5": KIND_T5,
    "esm": KIND_ESM2,
}

_DEFAULT_MAX_LENGTH = {
    KIND_ESMC: 1024,
    KIND_MOLFORMER: 202,
    KIND_ESM2: 640,
    KIND_T5: 384,
}

_TRUST_REMOTE_CODE_DEFAULT = {
    KIND_ESMC: True,
    KIND_MOLFORMER: True,
    KIND_ESM2: False,
    KIND_T5: False,
}


def default_max_length(kind: str) -> int:
    if kind not in _DEFAULT_MAX_LENGTH:
        raise ValueError(f"未知骨干类型: {kind!r}（支持 {SUPPORTED_KINDS}）")
    return _DEFAULT_MAX_LENGTH[kind]


def detect_kind(name: str, config: Any = None) -> str:
    """按名字关键字判定骨干类型；失败时回退到 config.model_type。"""
    lowered = (name or "").lower()
    for keyword, kind in _KIND_KEYWORDS:
        if keyword in lowered:
            return kind

    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type in _CONFIG_MODEL_TYPE_MAP:
        return _CONFIG_MODEL_TYPE_MAP[model_type]

    raise ValueError(
        f"无法判定骨干类型：name={name!r}，config.model_type={model_type!r}。"
        f"支持的类型: {SUPPORTED_KINDS}"
    )


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str
    trust_remote_code: bool
    max_length: int
    hidden_size: int
    num_layers: int
    prefer_hidden_states: bool = False


def _load_esmc(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    """ESMC 走 AutoModelForMaskedLM + trust_remote_code，tokenizer 来自 esm 包。"""
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(name, **kwargs)
    try:
        from esm.models.esmc import EsmcTokenizer
    except ImportError as error:
        raise ImportError(
            "加载 ESMC 需要 esm 包来提供 EsmcTokenizer（transformers 未注册该名字）。"
            "请先安装：pip install esm"
        ) from error
    return model, EsmcTokenizer(), True


def _load_molformer(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    """MoLFormer 是 encoder-only，没有 get_encoder()。"""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, **kwargs)
    return model, tokenizer, False


def _load_esm2(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    from transformers import EsmModel, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name, **kwargs)
    return model, tokenizer, False


def _load_t5(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(name)
    model = T5EncoderModel.from_pretrained(name, **kwargs)
    return model, tokenizer, False


_LOADERS = {
    KIND_ESMC: _load_esmc,
    KIND_MOLFORMER: _load_molformer,
    KIND_ESM2: _load_esm2,
    KIND_T5: _load_t5,
}


def _load(name: str, *, kind=None, trust_remote_code=None, max_length=None,
          revision=None) -> tuple[nn.Module, Any, BackboneSpec]:
    resolved_kind = kind or detect_kind(name)
    if resolved_kind not in _LOADERS:
        raise ValueError(f"未知骨干类型: {resolved_kind!r}（支持 {SUPPORTED_KINDS}）")

    if trust_remote_code is None:
        trust_remote_code = _TRUST_REMOTE_CODE_DEFAULT[resolved_kind]

    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if revision:
        kwargs["revision"] = revision

    model, tokenizer, prefer_hidden_states = _LOADERS[resolved_kind](name, **kwargs)

    spec = BackboneSpec(
        name=name,
        kind=resolved_kind,
        trust_remote_code=bool(trust_remote_code),
        max_length=int(max_length or default_max_length(resolved_kind)),
        hidden_size=resolve_hidden_size(model.config),
        num_layers=resolve_num_layers(model.config, model),
        prefer_hidden_states=prefer_hidden_states,
    )
    return model, tokenizer, spec


def load_protein_backbone(name: str, *, kind=None, trust_remote_code=None,
                          max_length=None, revision=None) -> tuple[nn.Module, Any, BackboneSpec]:
    return _load(name, kind=kind, trust_remote_code=trust_remote_code,
                 max_length=max_length, revision=revision)


def load_ligand_backbone(name: str, *, kind=None, trust_remote_code=None,
                         max_length=None, revision=None) -> tuple[nn.Module, Any, BackboneSpec]:
    return _load(name, kind=kind, trust_remote_code=trust_remote_code,
                 max_length=max_length, revision=revision)


def forward_hidden_states(backbone: nn.Module, inputs: dict,
                          *, output_hidden_states: bool = False) -> torch.Tensor:
    """取 token 级 hidden state。单次前向，不重复计算。"""
    if output_hidden_states:
        outputs = backbone(**inputs, output_hidden_states=True)
        states = getattr(outputs, "hidden_states", None)
        if states:
            return torch.nan_to_num(states[-1])
        hidden = getattr(outputs, "last_hidden_state", None)
    else:
        outputs = backbone(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            states = getattr(outputs, "hidden_states", None)
            if states:
                hidden = states[-1]

    if hidden is None:
        attrs = sorted(vars(outputs)) if hasattr(outputs, "__dict__") else []
        raise ValueError(
            "骨干输出里既没有 last_hidden_state 也没有可用的 hidden_states；"
            f"实际属性: {attrs}。若该模型需要 output_hidden_states=True，请显式传入。"
        )
    return torch.nan_to_num(hidden)


def _special_token_ids(tokenizer) -> set[int]:
    ids = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(int(value))
    return ids


def build_token_mask(tokenizer, inputs: dict, *, strip_special: bool = True) -> torch.Tensor:
    """从 attention_mask 出发；strip_special 时额外剔除 BOS/EOS 等特殊 token。

    ESMC 的输出保留 2 个特殊 token，不剔除会污染下游的 masked_mean 池化。
    """
    mask = inputs["attention_mask"].bool().clone()
    if not strip_special:
        return mask

    ids = inputs["input_ids"]
    special_ids = _special_token_ids(tokenizer)
    if not special_ids:
        return mask

    special = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= (ids == token_id)

    per_row = special.sum(dim=1)
    if per_row.numel() > 0:
        typical = int(per_row.float().median().item())
        if typical not in (0, 2):
            warnings.warn(
                f"特殊 token 数的中位数为 {typical}（预期 0 或 2）。"
                "tokenizer 的特殊 token 集合可能不完整，请检查 build_token_mask 的输入。",
                stacklevel=2,
            )

    return mask & ~special
