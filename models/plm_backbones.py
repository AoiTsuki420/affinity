"""PLM 骨干的加载、形状解析与 LoRA 注入。

设计约束：纯解析函数只依赖 torch，使无 GPU / 无 peft / 无 esm 的环境也能对它们做单元测试。
peft 与 esm 一律在函数内部 lazy import。
"""
from __future__ import annotations

import functools
import inspect
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

# 层容器的候选与顺序，被 resolve_num_layers 与 _layer_containers 共用：
# 层数解析和梯度检查点必须落在同一个容器上，否则两者会对不同的层集合生效。
_LAYER_CONTAINER_CANDIDATES: tuple[tuple[str | None, str], ...] = (
    ("encoder", "layer"),
    ("encoder", "block"),
    (None, "layers"),
    (None, "h"),
)


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
        for holder_name, attr in _LAYER_CONTAINER_CANDIDATES:
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

    # tokenizer 与权重共用同一份 kwargs：trust_remote_code 由调用方决定（默认 True），
    # revision 也必须一致，否则 tokenizer 会解析 main 而与钉住的权重版本漂移。
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    model = AutoModel.from_pretrained(name, **kwargs)
    return model, tokenizer, False


def _load_esm2(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    from transformers import EsmModel, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(name, **kwargs)
    model = EsmModel.from_pretrained(name, **kwargs)
    return model, tokenizer, False


def _load_t5(name: str, **kwargs) -> tuple[nn.Module, Any, bool]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(name, **kwargs)
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

    # trust_remote_code=True 时 hub shim 可能返回 esm 自己的 ESMC 对象：那是普通 nn.Module，
    # 没有 .config。此时按 tools/probe_backbone_modules.py 的做法单独取一份 config，
    # 并把 model 一并交给 resolve_num_layers 走结构回退。
    config = getattr(model, "config", None)
    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(name, **kwargs)

    spec = BackboneSpec(
        name=name,
        kind=resolved_kind,
        trust_remote_code=bool(trust_remote_code),
        max_length=int(max_length or default_max_length(resolved_kind)),
        hidden_size=resolve_hidden_size(config),
        num_layers=resolve_num_layers(config, model),
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


def _available_states(outputs) -> Any:
    """取逐层 hidden states 序列；没有可用序列时返回 None。

    不能写 `if states:`：esm 包的 EsmcOutput.hidden_states 是**张量**
    （(n_layers+1, B, L, H)，esm/models/esmc/model.py 文档写明），对多元素张量
    做布尔运算会抛 "Boolean value of Tensor with more than one value is
    ambiguous"（集群 smoke 实测）。transformers 系的元组输出则一直是这么判空的，
    len() 对元组和张量都成立，且 states[-1] 在两种布局下都取最后一层。
    """
    states = getattr(outputs, "hidden_states", None)
    if states is None:
        return None
    try:
        if len(states) == 0:
            return None
    except TypeError:
        return None
    return states


def forward_hidden_states(backbone: nn.Module, inputs: dict,
                          *, output_hidden_states: bool = False) -> torch.Tensor:
    """取 token 级 hidden state。单次前向，不重复计算。"""
    if output_hidden_states:
        outputs = backbone(**inputs, output_hidden_states=True)
        states = _available_states(outputs)
        if states is not None:
            return torch.nan_to_num(states[-1])
        hidden = getattr(outputs, "last_hidden_state", None)
    else:
        outputs = backbone(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            states = _available_states(outputs)
            if states is not None:
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

    # 只看非 padding 位：padding 位塞满 pad_token_id，若一并计入，正常的 padding 批次
    # （50..400 补齐到 400）中位数会是几百，于是每个批次都误报，真信号反而被淹没。
    # 返回的 mask 不受影响：padding 位在 attention_mask 里本来就是 False。
    special = special & mask

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


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练 LoRA 参数总数, LoRA 张量数)。"""
    n_params = 0
    n_tensors = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            n_tensors += 1
            if param.requires_grad:
                n_params += param.numel()
    return n_params, n_tensors


def _layer_containers(model: nn.Module):
    """产出 (层号, 层模块) 的可迭代序列，用于定位逐层属性。"""
    for holder_name, attr in _LAYER_CONTAINER_CANDIDATES:
        holder = model if holder_name is None else getattr(model, holder_name, None)
        if holder is None:
            continue
        seq = getattr(holder, attr, None)
        if seq is not None and hasattr(seq, "__len__") and len(seq) > 0:
            for index, module in enumerate(seq):
                yield index, module
            return


def _absorb_peft_inputs_embeds(model: nn.Module) -> nn.Module:
    """为不接受 inputs_embeds 的骨干补一个吸收该关键字的实例级 forward。

    背景（集群 smoke 实测，2026-10-14）：peft 的 PeftModelForFeatureExtraction.forward
    会**无条件**把 inputs_embeds（包括 None）显式传给底层模型
    （peft_model.py: self.base_model(input_ids=..., inputs_embeds=inputs_embeds, ...)），
    而 esm 包的 EsmcForMaskedLM.forward 没有这个形参——LoRA 包装后的第一次前向就
    TypeError: "EsmcForMaskedLM.forward() got an unexpected keyword argument
    'inputs_embeds'"。单测夹具的 forward 带 **kwargs，覆盖不到这条真实签名差异。

    规则：
    - 骨干 forward 已声明 inputs_embeds（ESM-2 / T5 等）→ 原样返回，不做任何改动；
    - 未声明 → 只在**实例**上替换 forward：吸收 inputs_embeds=None；真张量仍然显式
      TypeError（本仓库没有任何调用方直接喂 embedding，静默丢弃会改变语义）。
    - 不套 nn.Module 壳、不改类：named_modules() 限定名与 state_dict 键名保持逐字不变，
      LoRA 目标发现与既有 checkpoint 都按名字工作，任何包装层都会破坏它们。
    """
    original = model.forward
    if "inputs_embeds" in inspect.signature(original).parameters:
        return model

    def forward(*args, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise TypeError(
                f"{type(model).__name__}.forward() 只接受 token 输入，不支持真正的 "
                "inputs_embeds：适配器只为吸收 peft 无条件传入的 inputs_embeds=None "
                "而存在。"
            )
        return original(*args, **kwargs)

    model.forward = forward  # type: ignore[method-assign]
    return model


def inject_lora(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    policy: str,
    last_k: int,
    explicit_layers: Sequence[int] | None = None,
    suffixes: Sequence[str] | None = None,
) -> tuple[nn.Module, list[str], list[int]]:
    """按策略注入 LoRA，返回 (peft 模型, 命中的目标名, 命中的层号)。

    peft 在函数内 lazy import，使本模块在未安装 peft 的环境里仍可导入与测试纯函数。
    """
    from peft import LoraConfig, get_peft_model

    num_layers = resolve_num_layers(getattr(model, "config", None) or _NullConfig(), model)
    layers = resolve_lora_layers(num_layers, policy, last_k, explicit=explicit_layers)
    targets = discover_lora_targets(model, layers=layers, suffixes=suffixes)

    lora_config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        target_modules=targets,
        lora_dropout=float(dropout),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    # peft 的 FEATURE_EXTRACTION 包装会无条件显式传 inputs_embeds（含 None），
    # 在包装之前吸收掉与严格签名骨干的差异（见 _absorb_peft_inputs_embeds）。
    model = _absorb_peft_inputs_embeds(model)
    peft_model = get_peft_model(model, lora_config)

    n_params, n_tensors = count_lora_parameters(peft_model)
    if n_tensors == 0 or n_params == 0:
        raise AssertionError(
            f"LoRA 注入后没有可训练张量：tensors={n_tensors}, params={n_params}。"
            f"targets={targets[:10]}..."
        )
    return peft_model, targets, layers


class _NullConfig:
    """占位 config，使 resolve_num_layers 走模型结构回退路径。"""


def set_backbone_gradient_checkpointing(
    model: nn.Module,
    mode: str,
    *,
    layers: Sequence[int] | None = None,
) -> int:
    """按模式开关骨干的逐层梯度检查点，返回真正接受了开关的层数。

    mode='active' 只对 LoRA 命中的层开启：前段层因无 LoRA 而不建图，
    对它们开检查点只会白白增加重算开销。

    契约（调用方必读）：
    - 本函数只切换**逐层**的 gradient_checkpointing / _gradient_checkpointing_func 属性，
      也就是 transformers 5.x `GradientCheckpointingLayer` 真正读取的那一对；
    - 若某个骨干把开关放在**模型级或 encoder 级**的 gradient_checkpointing 上
      （自写的 remote code 可能如此），本函数覆盖不到：这类骨干请改用它自带的
      `gradient_checkpointing_enable()`。此时返回值可能是正数，但那**不是**检查点已生效的
      证据——层上带着类属性、模型却不读，本函数无从分辨；
    - 非 off 模式下，一层都没真正接受开关就抛 ValueError，所以返回 0 永远不表示成功；
      off 是唯一允许什么都不做并原样返回 0 的模式。

    逐层属性成对设置，镜像 transformers 的
    `PreTrainedModel._set_gradient_checkpointing`：布尔量是唯一的开关，
    `_gradient_checkpointing_func` 始终指向可用的检查点实现。二者必须同时赋值——
    `GradientCheckpointingLayer.__call__` 在开关为真时会调用
    `self._gradient_checkpointing_func`，而基类并不提供该属性，只设布尔量会让真实骨干
    （EsmLayer 等）的第一次前向抛 AttributeError。
    这里刻意传 use_reentrant=False：HF 传裸 checkpoint 会触发 torch 的 use_reentrant
    弃用警告。
    """
    if mode not in ("off", "active", "all"):
        raise ValueError(f"未知的梯度检查点模式: {mode!r}（支持 off / active / all）")

    wanted = None if layers is None else {int(i) for i in layers}
    checkpoint_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    enabled = 0
    touched = 0
    capable = 0
    for index, module in _layer_containers(model):
        touched += 1
        if not hasattr(module, "gradient_checkpointing"):
            continue

        capable += 1
        if mode == "off":
            should_enable = False
        elif mode == "all":
            should_enable = True
        else:
            should_enable = wanted is not None and index in wanted

        module._gradient_checkpointing_func = checkpoint_func
        module.gradient_checkpointing = should_enable
        if should_enable:
            # 只有真正接受了开关的层才计数：返回值是调用方唯一的成功信号，
            # 不能被"选中的层数"冒名顶替。
            enabled += 1

    if mode != "off" and enabled == 0:
        candidates = " / ".join(
            f"{holder}.{attr}" if holder else attr for holder, attr in _LAYER_CONTAINER_CANDIDATES
        )
        requested = (
            "全部层（mode='all'）"
            if mode == "all"
            else (f"layers={sorted(wanted)}" if wanted is not None else "layers=None（未指定）")
        )
        container_desc = (
            f"找到 {touched} 个层模块" if touched else f"未找到任何层容器（候选：{candidates}）"
        )
        raise ValueError(
            f"梯度检查点未生效：mode={mode!r}，请求的层={requested}，{container_desc}，"
            f"带 gradient_checkpointing 属性的层={capable} 个，实际开启 {enabled} 个。"
            "排查顺序：① 层容器是否为 "
            f"{candidates}；"
            "② 层模块是否带 gradient_checkpointing 属性"
            "（远程代码骨干常常没有，请先在集群上跑 tools/probe_backbone_modules.py 确认）；"
            "③ mode='active' 时 layers 是否传入了 LoRA 命中的层号。"
        )
    return enabled
