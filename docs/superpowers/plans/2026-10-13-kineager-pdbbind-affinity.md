# KinEAGER PDBbind v2020 亲和力通路 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 KinEAGER 框架内新增一条单任务蛋白-小分子亲和力回归通路（`--task affinity`），使用 ESMC-6B + MoLFormer 在线前向与 last-30 层 LoRA 训练 PDBbind v2020，并让训练可在 k8s 上按"每 pod 1 卡"多卡并行。

**Architecture:** 保留现有 `task=kinetics` 三任务通路逐字节不变，新增一条并列分支。骨干与 LoRA 的骨干相关性全部收敛到新的 `models/plm_backbones.py`（取代 `models/predictor.py` 里按 ESM-2 / T5 模块名硬编码的两段）。亲和力头复用现有 `MultiHead` + `MLPHead(out_dim=2)`，损失从已有的单任务 `kcat_expert_loss` 泛化。数据侧新增 `data_affinity.py`（单列标签），DDP 沿用现有 `DistributedDataParallel`，只补齐 xtrimofold3 式启动契约。

**Tech Stack:** Python 3.10+、PyTorch、HuggingFace Transformers 5.x、PEFT、RDKit、pytest。ESMC 额外依赖 `esm` 包（提供 `EsmcTokenizer`）。

**Spec:** `docs/superpowers/specs/2026-10-13-kineager-pdbbind-affinity-design.md`

## Global Constraints

- **本地环境**：无 GPU、无 CUDA、无 `peft`、无 `esm`。`torch 2.13.0+cpu`、`transformers 5.14.1`、`numpy 2.5.1`、`pandas 3.0.5`、`rdkit 2026.3.5`、`pytest 9.1.1`。**所有本地测试必须是 CPU-only 且不下载模型权重。**
- **`task=kinetics` 路径逐字节不变。** 每个改动该文件的 task 结束时必须跑一次 kinetics 回归。
- **affinity 通路不得构造** `s_kcat`、`s_Km`、`s_ratio`、`s_domain_task`、`kcat_source_residual`（DDP 在 `find_unused_parameters=False` 下会因"有图无梯度"的参数报错）。
- **LoRA 策略**：`policy="last_k"`，`last_k=30`，按骨干实际层数 `min(last_k, num_layers)` clamp。ESMC-6B（80 层）→ 第 50..79 层；MoLFormer（12 层）→ 第 0..11 层。
- **ESMC-6B**：HF id `biohub/ESMC-6B`，`trust_remote_code=True`，通过 `AutoModelForMaskedLM` 加载，hidden = **2560**，transformer 层数 = **80**，tokenizer 来自 `esm.models.esmc.EsmcTokenizer`，缓存/输出保留 2 个特殊 token，必须从 attention mask 剔除。
- **MoLFormer**：HF id `ibm-research/MoLFormer-XL-both-10pct`，revision **`7b12d946c181a37f6012b9dc3b002275de070314`**，`trust_remote_code=True`，通过 `AutoModel` 加载（encoder-only，**没有 `get_encoder()`**），hidden = **768**，层数 = **12**，`max_length` = **202**。
- **结构分支关闭**：不传 `--use_struct_branch`、不传 `--final_contact_npz`。其代码一行不改。
- **不引入** DeepSpeed / Lightning / FSDP。
- **提交但不 push**（用户明确要求）。
- 新模块顶部不要引入 `peft` / `esm` 的顶层 import；需要时在函数内 lazily import，保证纯函数部分可在无这些包的环境里被测试。

## 文件结构

**新增**

| 文件 | 职责 |
|---|---|
| `models/plm_backbones.py` | 骨干加载、config 字段解析、LoRA 层选择与目标发现、梯度检查点开关。纯函数部分只依赖 torch |
| `models/affinity_loss.py` | 单任务亲和力损失（Huber/Barron + ranking + var_reg + NLL），从 `kcat_expert_loss` 泛化 |
| `data_affinity.py` | `AffinityScaler`（单列）+ `AffinityDataset` |
| `tools/build_pdbbind_affinity_csv.py` | PDBbind v2020 → 训练 CSV |
| `tools/probe_backbone_modules.py` | P0 探针：打印真实模块名 / config 字段 / 层数 / hidden |
| `bin/train/kineager_k8s.sh` | k8s 启动脚本（对齐 xtrimofold3 rendezvous 契约） |
| `yamls/kubeflow/kineager_affinity_k8s.yaml` | PyTorchJob 模板 |
| `tests/test_plm_backbones.py` | `plm_backbones` 纯函数单测 |
| `tests/test_affinity_loss.py` | 损失单测 |
| `tests/test_affinity_data.py` | Scaler / Dataset 单测 |
| `tests/test_affinity_forward.py` | `forward(task="affinity")` 形状与参数计数 |
| `tests/test_predictor_import.py` | `models.predictor` 无 peft 可导入 |
| `tests/test_pdbbind_csv.py` | CSV 构建逻辑单测 |

**修改**

| 文件 | 改动 |
|---|---|
| `models/encoders.py` | 改走 `plm_backbones`，去掉骨干硬编码 |
| `models/predictor.py` | LoRA 段落替换、`get_encoder` 分支、`task` 开关、affinity forward 分支 |
| `main_train_predictor_multigpu.py` | `--task` 分支、新 CLI、环境变量化、梯度累积 |
| `main_infer_predictor.py` | affinity 单路推理 |
| `path_config.py` | `k8s` profile + 骨干路径键 |
| `requirements.txt` | 补 `peft`、`esm`，删除已不存在的 `loralib` 依赖问题 |
| `README.md` | 补 affinity 用法 |

---

## Task 1: P0 探针脚本

**为什么第一个做：** ESMC-6B 是 `trust_remote_code` 加载的，其模块命名未知；LoRA 的 `target_modules` 必须命中真实层名，否则 `inject_lora` 必然失败。探针输出决定 Task 5 的默认后缀白名单是否够用。

**Files:**
- Create: `tools/probe_backbone_modules.py`
- Create: `tests/test_probe_backbone_modules.py`

**Interfaces:**
- Consumes: 无
- Produces: `probe_backbone_modules.describe_backbone(model, config) -> dict`，返回键 `{"hidden_size": int, "num_layers": int, "linear_names": list[str], "leaf_names": list[str], "config_fields": list[str], "layer_names_sample": list[str]}`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_probe_backbone_modules.py`：

```python
import torch.nn as nn

from tools.probe_backbone_modules import describe_backbone


class _Cfg:
    hidden_size = 768
    num_hidden_layers = 2


def _fake_bert_like():
    """构造一个 BERT 命名风格的极小模型，用于离线验证探针输出格式。"""
    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.key = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = _SelfAttn()
            self.output = nn.Module()
            self.output.dense = nn.Linear(8, 8)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _Attn()

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([_Layer() for _ in range(2)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    return _Model()


def test_describe_backbone_reports_hidden_and_layers():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    assert info["hidden_size"] == 768
    assert info["num_layers"] == 2


def test_describe_backbone_lists_only_linear_modules():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    names = info["linear_names"]
    assert "encoder.layer.0.attention.self.query" in names
    assert "encoder.layer.1.attention.output.dense" in names
    # 非 Linear 模块不得出现
    assert "encoder.layer.0.attention.output" not in names


def test_describe_backbone_reports_leaf_names_sorted_unique():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    assert info["leaf_names"] == sorted(set(info["leaf_names"]))
    assert "query" in info["leaf_names"]


def test_describe_backbone_reports_config_fields():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    assert "hidden_size" in info["config_fields"]
    assert "num_hidden_layers" in info["config_fields"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_probe_backbone_modules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.probe_backbone_modules'`

- [ ] **Step 3: 实现探针**

创建 `tools/probe_backbone_modules.py`：

```python
"""打印一个 PLM 骨干的真实结构，用于确定 LoRA 的 target_modules。

ESMC-6B 通过 trust_remote_code 加载，其模块命名不可先验假定；
本脚本在目标环境上运行一次，把输出作为 models/plm_backbones.py 后缀白名单的依据。

用法（集群上，需要权重可访问）::

    python tools/probe_backbone_modules.py --model biohub/ESMC-6B --kind esmc
    python tools/probe_backbone_modules.py \
        --model ibm-research/MoLFormer-XL-both-10pct --kind molformer \
        --revision 7b12d946c181a37f6012b9dc3b002275de070314
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch.nn as nn


def describe_backbone(model: nn.Module, config: Any) -> dict:
    """收集 LoRA 目标发现所需的全部信息。纯读取，不修改模型。"""
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n]
    leaf_names = sorted({n.rsplit(".", 1)[-1] for n in linear_names})
    config_fields = sorted(vars(config)) if hasattr(config, "__dict__") else []

    layer_names_sample = [
        n for n, _ in model.named_modules()
        if n.count(".") <= 2 and any(k in n for k in ("layer", "block", "layers"))
    ][:40]

    try:
        from models.plm_backbones import resolve_hidden_size, resolve_num_layers

        hidden_size = resolve_hidden_size(config)
        num_layers = resolve_num_layers(config, model)
    except Exception as error:  # 探针的职责就是把失败原因报出来，而不是中断
        hidden_size = None
        num_layers = None
        layer_names_sample.append(f"<resolve failed: {error}>")

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "linear_names": sorted(linear_names),
        "leaf_names": leaf_names,
        "config_fields": config_fields,
        "layer_names_sample": layer_names_sample,
    }


def _load(model_name: str, kind: str, revision: str | None, trust_remote_code: bool):
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    kwargs = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    if kind == "esmc":
        model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
    return model, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", required=True, choices=["esmc", "molformer", "esm2", "t5"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, config = _load(
        args.model, args.kind, args.revision,
        trust_remote_code=not args.no_trust_remote_code,
    )
    info = describe_backbone(model, config)
    info["model_name"] = args.model
    info["kind"] = args.kind
    info["revision"] = args.revision

    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)

    print("\n=== 建议的 LoRA 后缀白名单（取 Linear 末段名，人工筛选后填入 plm_backbones） ===")
    print(json.dumps(info["leaf_names"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_probe_backbone_modules.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/probe_backbone_modules.py tests/test_probe_backbone_modules.py
git commit -m "feat(tools): add backbone module probe for LoRA target discovery"
```

**集群出口条件（阻塞 Task 5 的默认白名单定稿，但不阻塞 Task 2-4 的编写）：** 在能访问权重的环境上跑 `--model biohub/ESMC-6B --kind esmc` 与 MoLFormer，产出 `leaf_names`。若其中出现白名单外的名字，在 Task 5 里通过 `--lora_target_suffixes` 注入，而不是改代码。

---

## Task 2: `resolve_hidden_size` / `resolve_num_layers`

**Files:**
- Create: `models/plm_backbones.py`
- Create: `tests/test_plm_backbones.py`

**Interfaces:**
- Consumes: 无
- Produces: `resolve_hidden_size(config) -> int`、`resolve_num_layers(config, model=None) -> int`、常量 `KIND_ESMC/KIND_MOLFORMER/KIND_ESM2/KIND_T5`、`SUPPORTED_KINDS`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_plm_backbones.py`：

```python
from types import SimpleNamespace

import pytest
import torch.nn as nn

from models.plm_backbones import resolve_hidden_size, resolve_num_layers


def test_hidden_size_esm2_style():
    assert resolve_hidden_size(SimpleNamespace(hidden_size=1280)) == 1280


def test_hidden_size_t5_style():
    assert resolve_hidden_size(SimpleNamespace(d_model=768)) == 768


def test_hidden_size_esmc_style():
    assert resolve_hidden_size(SimpleNamespace(embedding_dim=2560)) == 2560


def test_hidden_size_prefers_hidden_size_over_d_model():
    assert resolve_hidden_size(SimpleNamespace(hidden_size=768, d_model=512)) == 768


def test_hidden_size_raises_with_field_listing():
    with pytest.raises(ValueError) as excinfo:
        resolve_hidden_size(SimpleNamespace(weird_field=1))
    assert "weird_field" in str(excinfo.value)


def test_num_layers_from_config():
    assert resolve_num_layers(SimpleNamespace(num_hidden_layers=33)) == 33
    assert resolve_num_layers(SimpleNamespace(num_layers=12)) == 12
    assert resolve_num_layers(SimpleNamespace(n_layers=80)) == 80


def test_num_layers_falls_back_to_encoder_layer_modulelist():
    model = SimpleNamespace(encoder=SimpleNamespace(layer=nn.ModuleList([nn.Identity()] * 6)))
    assert resolve_num_layers(SimpleNamespace(), model) == 6


def test_num_layers_falls_back_to_layers_modulelist():
    model = SimpleNamespace(layers=nn.ModuleList([nn.Identity()] * 4))
    assert resolve_num_layers(SimpleNamespace(), model) == 4


def test_num_layers_raises_when_undeterminable():
    with pytest.raises(ValueError):
        resolve_num_layers(SimpleNamespace(), SimpleNamespace())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.plm_backbones'`

- [ ] **Step 3: 实现最小版本**

创建 `models/plm_backbones.py`：

```python
"""PLM 骨干的加载、形状解析与 LoRA 注入。

设计约束：纯解析函数只依赖 torch，使无 GPU / 无 peft / 无 esm 的环境也能对它们做单元测试。
peft 与 esm 一律在函数内部 lazy import。
"""
from __future__ import annotations

from typing import Any

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git add models/plm_backbones.py tests/test_plm_backbones.py
git commit -m "feat(models): add backbone config field resolution helpers"
```

---

## Task 3: `resolve_lora_layers`

**Files:**
- Modify: `models/plm_backbones.py`
- Modify: `tests/test_plm_backbones.py`

**Interfaces:**
- Consumes: `resolve_num_layers`（不直接调用）
- Produces: `resolve_lora_layers(num_layers: int, policy: str, last_k: int, explicit: Sequence[int] | None = None) -> list[int]`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_plm_backbones.py`：

```python
from models.plm_backbones import resolve_lora_layers


def test_lora_layers_all():
    assert resolve_lora_layers(80, "all", 30) == list(range(80))


def test_lora_layers_last_k_esmc_6b():
    """ESMC-6B 有 80 层，last-30 -> 第 50..79 层。"""
    assert resolve_lora_layers(80, "last_k", 30) == list(range(50, 80))


def test_lora_layers_last_k_clamped_for_molformer():
    """MoLFormer 只有 12 层，last_k=30 必须 clamp 到全部 12 层。"""
    assert resolve_lora_layers(12, "last_k", 30) == list(range(12))


def test_lora_layers_last_k_equal_to_num_layers():
    assert resolve_lora_layers(8, "last_k", 8) == list(range(8))


def test_lora_layers_last_k_larger_than_num_layers():
    assert resolve_lora_layers(4, "last_k", 100) == list(range(4))


def test_lora_layers_explicit_sorted_and_deduped():
    assert resolve_lora_layers(80, "explicit", 30, explicit=[79, 50, 79, 51]) == [50, 51, 79]


def test_lora_layers_explicit_out_of_range_dropped():
    assert resolve_lora_layers(10, "explicit", 30, explicit=[-1, 3, 99]) == [3]


def test_lora_layers_explicit_requires_argument():
    with pytest.raises(ValueError):
        resolve_lora_layers(80, "explicit", 30, explicit=None)


def test_lora_layers_rejects_unknown_policy():
    with pytest.raises(ValueError):
        resolve_lora_layers(80, "banana", 30)


def test_lora_layers_rejects_non_positive_num_layers():
    with pytest.raises(ValueError):
        resolve_lora_layers(0, "all", 30)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_plm_backbones.py -k lora_layers -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_lora_layers'`

- [ ] **Step 3: 实现**

追加到 `models/plm_backbones.py`（并在文件顶部的 typing import 中补 `Sequence`）：

```python
from typing import Any, Sequence

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: PASS（19 passed）

- [ ] **Step 5: 提交**

```bash
git add models/plm_backbones.py tests/test_plm_backbones.py
git commit -m "feat(models): add LoRA layer selection policy resolution"
```

---

## Task 4: `discover_lora_targets`（本计划的技术核心）

**取代** `models/predictor.py:568-587` 的 `_build_esm_targets_exact` 与 `models/predictor.py:594-615` 的 `_build_t5_encoder_targets_exact`。

**Files:**
- Modify: `models/plm_backbones.py`
- Modify: `tests/test_plm_backbones.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `DEFAULT_LORA_SUFFIXES: tuple[str, ...]`
  - `discover_lora_targets(model, *, layers=None, suffixes=None, min_hits=1) -> list[str]`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_plm_backbones.py`：

```python
from models.plm_backbones import discover_lora_targets


def _fake_esm2(num_layers: int = 3):
    """ESM-2 命名：encoder.layer.{i}.attention.self.{query,key,value} 等。"""
    class _Self(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.key = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)

    class _AttnOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(8, 8)

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = _Self()
            self.output = _AttnOut()

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _Attention()
            self.intermediate = nn.Module()
            self.intermediate.dense = nn.Linear(8, 32)
            self.output = nn.Module()
            self.output.dense = nn.Linear(32, 8)

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([_Layer() for _ in range(num_layers)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    return _Model()


def _fake_t5(num_layers: int = 2):
    """T5 命名：encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o} 等。"""
    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.v = nn.Linear(8, 8)
            self.o = nn.Linear(8, 8)

    class _DenseRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.wi_0 = nn.Linear(8, 32)
            self.wi_1 = nn.Linear(8, 32)
            self.wo = nn.Linear(32, 8)

    class _BlockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.SelfAttention = _SelfAttn()
            self.DenseReluDense = _DenseRelu()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([_BlockLayer()])

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([_Block() for _ in range(num_layers)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    return _Model()


def test_discover_esm2_all_layers():
    targets = discover_lora_targets(_fake_esm2(3))
    assert len(targets) == 3 * 6  # query/key/value/attn.output.dense/intermediate/output
    assert "encoder.layer.0.attention.self.query" in targets
    assert "encoder.layer.2.output.dense" in targets


def test_discover_esm2_restricted_to_layers():
    targets = discover_lora_targets(_fake_esm2(3), layers=[1, 2])
    assert all(".layer.0." not in t for t in targets)
    assert len(targets) == 2 * 6


def test_discover_t5_all_layers():
    targets = discover_lora_targets(_fake_t5(2))
    assert len(targets) == 2 * 7
    assert "encoder.block.0.layer.0.SelfAttention.q" in targets
    assert "encoder.block.1.layer.1.DenseReluDense.wo" in targets


def test_discover_returns_sorted_unique():
    targets = discover_lora_targets(_fake_esm2(3))
    assert targets == sorted(set(targets))


def test_discover_raises_and_lists_real_leaf_names_when_nothing_matches():
    class _Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_custom_proj = nn.Linear(4, 4)

    with pytest.raises(AssertionError) as excinfo:
        discover_lora_targets(_Weird(), layers=[0])
    message = str(excinfo.value)
    assert "my_custom_proj" in message


def test_discover_respects_custom_suffixes_from_probe():
    """探针如果在 ESMC-6B 上发现白名单外的名字，调用方应能直接注入。"""
    class _Esmscish(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module() for _ in range(2)])
            self.layers[0].mixer = nn.Linear(4, 4)

    targets = discover_lora_targets(_Esmscish(), layers=[0, 1], suffixes=("mixer",))
    assert targets == ["layers.0.mixer"]


def test_discover_min_hits_enforced():
    class _Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()

    with pytest.raises(AssertionError):
        discover_lora_targets(_Empty(), layers=[0], min_hits=1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_plm_backbones.py -k discover -v`
Expected: FAIL — `ImportError: cannot import name 'discover_lora_targets'`

- [ ] **Step 3: 实现**

追加到 `models/plm_backbones.py`（顶部补 `import re`）：

```python
import re

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

_LAYER_INDEX_RE = re.compile(r"\.(?:layer|layers|block|blocks|h)\.(\d+)\.")


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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: PASS（26 passed）

- [ ] **Step 5: 提交**

```bash
git add models/plm_backbones.py tests/test_plm_backbones.py
git commit -m "feat(models): add architecture-agnostic LoRA target discovery

Replaces the ESM-2/T5 hardcoded module-name builders in models/predictor.py
so the encoders can be swapped for ESMC-6B and MoLFormer."
```

---

## Task 5: 骨干识别、加载与 hidden state 取值

**Files:**
- Modify: `models/plm_backbones.py`
- Modify: `tests/test_plm_backbones.py`

**Interfaces:**
- Consumes: `resolve_hidden_size`、`resolve_num_layers`
- Produces:
  - `BackboneSpec`（frozen dataclass：`name, kind, trust_remote_code, max_length, hidden_size, num_layers, prefer_hidden_states`）
  - `detect_kind(name, config=None) -> str`
  - `default_max_length(kind) -> int`
  - `load_protein_backbone(name, *, trust_remote_code=None, max_length=None) -> tuple[nn.Module, Any, BackboneSpec]`
  - `load_ligand_backbone(name, *, trust_remote_code=None, max_length=None, revision=None) -> tuple[nn.Module, Any, BackboneSpec]`
  - `forward_hidden_states(backbone, inputs, *, output_hidden_states=False) -> torch.Tensor`
  - `build_token_mask(tokenizer, inputs, *, strip_special=True) -> torch.Tensor`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_plm_backbones.py`：

```python
import torch

from models.plm_backbones import (
    KIND_ESM2,
    KIND_ESMC,
    KIND_MOLFORMER,
    KIND_T5,
    build_token_mask,
    default_max_length,
    detect_kind,
    forward_hidden_states,
)


def test_detect_kind_esmc():
    assert detect_kind("biohub/ESMC-6B") == KIND_ESMC


def test_detect_kind_molformer():
    assert detect_kind("ibm-research/MoLFormer-XL-both-10pct") == KIND_MOLFORMER


def test_detect_kind_t5():
    assert detect_kind("/models/molt5-base-smiles2caption") == KIND_T5


def test_detect_kind_esm2():
    assert detect_kind("/models/esm2_t33_650M_UR50D") == KIND_ESM2


def test_detect_kind_falls_back_to_config_model_type():
    assert detect_kind("/some/local/dir", SimpleNamespace(model_type="molformer")) == KIND_MOLFORMER


def test_detect_kind_raises_for_unknown():
    with pytest.raises(ValueError):
        detect_kind("/some/local/dir", SimpleNamespace(model_type="llama"))


def test_default_max_length_matches_reference():
    """MoLFormer 的 token 上限是 202（xtrimofold3 docs/pdbbind_molformer.md:127）。"""
    assert default_max_length(KIND_MOLFORMER) == 202
    assert default_max_length(KIND_ESMC) == 1024
    assert default_max_length(KIND_ESM2) == 640
    assert default_max_length(KIND_T5) == 384


def test_forward_hidden_states_prefers_last_hidden_state():
    class _Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(2, 3, 4))

    out = forward_hidden_states(_Backbone(), {})
    assert out.shape == (2, 3, 4)


def test_forward_hidden_states_uses_hidden_states_when_requested():
    class _Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            assert kwargs.get("output_hidden_states") is True
            states = (torch.zeros(2, 3, 4), torch.full((2, 3, 4), 7.0))
            return SimpleNamespace(hidden_states=states, last_hidden_state=None)

    out = forward_hidden_states(_Backbone(), {}, output_hidden_states=True)
    assert torch.all(out == 7.0)


def test_forward_hidden_states_nan_to_num():
    class _Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[float("nan")]]]))

    out = forward_hidden_states(_Backbone(), {})
    assert torch.isfinite(out).all()


def test_forward_hidden_states_raises_with_attr_listing():
    class _Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1))

    with pytest.raises(ValueError) as excinfo:
        forward_hidden_states(_Backbone(), {})
    assert "logits" in str(excinfo.value)


class _FakeTokenizer:
    all_special_ids = [0, 2]


def test_build_token_mask_strips_special_tokens():
    inputs = {
        "input_ids": torch.tensor([[0, 5, 6, 2], [0, 7, 2, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }
    mask = build_token_mask(_FakeTokenizer(), inputs)
    assert mask.tolist() == [[False, True, True, False], [False, True, False, False]]


def test_build_token_mask_keeps_special_when_disabled():
    inputs = {
        "input_ids": torch.tensor([[0, 5, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mask = build_token_mask(_FakeTokenizer(), inputs, strip_special=False)
    assert mask.tolist() == [[True, True, True]]


def test_build_token_mask_without_special_ids_is_identity():
    class _NoSpecial:
        all_special_ids = []

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mask = build_token_mask(_NoSpecial(), inputs)
    assert mask.tolist() == [[True, True, True]]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_plm_backbones.py -k "detect or default_max or forward_hidden or token_mask" -v`
Expected: FAIL — `ImportError: cannot import name 'detect_kind'`

- [ ] **Step 3: 实现**

追加到 `models/plm_backbones.py`（顶部补 `import warnings`、`from dataclasses import dataclass`）：

```python
import warnings
from dataclasses import dataclass

import torch

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: PASS（40 passed）

- [ ] **Step 5: 提交**

```bash
git add models/plm_backbones.py tests/test_plm_backbones.py
git commit -m "feat(models): add backbone loading and hidden-state extraction"
```

---

## Task 6: `inject_lora` 与骨干梯度检查点

**Files:**
- Modify: `models/plm_backbones.py`
- Modify: `tests/test_plm_backbones.py`

**Interfaces:**
- Consumes: `discover_lora_targets`、`resolve_lora_layers`、`resolve_num_layers`
- Produces:
  - `inject_lora(model, *, r, alpha, dropout, policy, last_k, explicit_layers=None, suffixes=None) -> tuple[nn.Module, list[str], list[int]]`
  - `count_lora_parameters(model) -> tuple[int, int]`
  - `set_backbone_gradient_checkpointing(model, mode, *, layers=None) -> int`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_plm_backbones.py`：

```python
peft = pytest.importorskip("peft", reason="本地未安装 peft，集群上执行")

from models.plm_backbones import (  # noqa: E402
    count_lora_parameters,
    inject_lora,
    set_backbone_gradient_checkpointing,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.self = torch.nn.Module()
        self.attention.self.query = torch.nn.Linear(8, 8)
        self.attention.self.key = torch.nn.Linear(8, 8)
        self.attention.self.value = torch.nn.Linear(8, 8)
        self.gradient_checkpointing = False


class _PeftableBackbone(torch.nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([_Block() for _ in range(num_layers)])
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=num_layers)


def test_inject_lora_last_k_hits_expected_layers():
    model, targets, layers = inject_lora(
        _PeftableBackbone(6), r=4, alpha=8, dropout=0.0,
        policy="last_k", last_k=2,
    )
    assert layers == [4, 5]
    assert all((".layer.4." in t) or (".layer.5." in t) for t in targets)
    assert len(targets) == 2 * 3


def test_inject_lora_produces_trainable_lora_parameters():
    model, _, _ = inject_lora(
        _PeftableBackbone(6), r=4, alpha=8, dropout=0.0,
        policy="last_k", last_k=2,
    )
    n_params, n_tensors = count_lora_parameters(model)
    assert n_tensors == 6
    assert n_params > 0


def test_inject_lora_r_zero_style_smaller_rank_is_cheaper():
    small, _, _ = inject_lora(_PeftableBackbone(6), r=2, alpha=8, dropout=0.0,
                              policy="last_k", last_k=2)
    large, _, _ = inject_lora(_PeftableBackbone(6), r=8, alpha=8, dropout=0.0,
                              policy="last_k", last_k=2)
    assert count_lora_parameters(small)[0] < count_lora_parameters(large)[0]


def test_set_gradient_checkpointing_active_only_touches_selected_layers():
    model = _PeftableBackbone(6)
    n = set_backbone_gradient_checkpointing(model, "active", layers=[4, 5])
    assert n == 2
    flags = [model.encoder.layer[i].gradient_checkpointing for i in range(6)]
    assert flags == [False, False, False, False, True, True]


def test_set_gradient_checkpointing_all_sets_every_layer():
    model = _PeftableBackbone(6)
    n = set_backbone_gradient_checkpointing(model, "all", layers=[4, 5])
    assert n == 6
    assert all(model.encoder.layer[i].gradient_checkpointing for i in range(6))


def test_set_gradient_checkpointing_off_clears_flags():
    model = _PeftableBackbone(6)
    set_backbone_gradient_checkpointing(model, "all", layers=[4, 5])
    n = set_backbone_gradient_checkpointing(model, "off", layers=[4, 5])
    assert n == 0
    assert not any(model.encoder.layer[i].gradient_checkpointing for i in range(6))


def test_set_gradient_checkpointing_rejects_unknown_mode():
    with pytest.raises(ValueError):
        set_backbone_gradient_checkpointing(_PeftableBackbone(2), "banana")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_plm_backbones.py -k "inject_lora or gradient_checkpointing" -v`
Expected: SKIPPED（本地无 peft）或在集群上 FAIL — `ImportError: cannot import name 'inject_lora'`

- [ ] **Step 3: 实现**

追加到 `models/plm_backbones.py`：

```python
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
    containers = (
        ("encoder", "layer"),
        ("encoder", "block"),
        (None, "layers"),
        (None, "h"),
    )
    for holder_name, attr in containers:
        holder = model if holder_name is None else getattr(model, holder_name, None)
        if holder is None:
            continue
        seq = getattr(holder, attr, None)
        if seq is not None and hasattr(seq, "__len__") and len(seq) > 0:
            for index, module in enumerate(seq):
                yield index, module
            return


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
    """按模式开关骨干的逐层梯度检查点，返回被开启的层数。

    mode='active' 只对 LoRA 命中的层开启：前段层因无 LoRA 而不建图，
    对它们开检查点只会白白增加重算开销。
    """
    if mode not in ("off", "active", "all"):
        raise ValueError(f"未知的梯度检查点模式: {mode!r}（支持 off / active / all）")

    wanted = None if layers is None else {int(i) for i in layers}

    enabled = 0
    touched = 0
    for index, module in _layer_containers(model):
        touched += 1
        if mode == "off":
            should_enable = False
        elif mode == "all":
            should_enable = True
        else:
            should_enable = wanted is not None and index in wanted

        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = should_enable
        if should_enable:
            enabled += 1

    if mode != "off" and touched == 0:
        raise ValueError(
            "无法定位骨干的层容器（encoder.layer / encoder.block / layers / h），"
            "梯度检查点未生效。"
        )
    return enabled
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_plm_backbones.py -v`
Expected: 本地 40 passed + 7 skipped；集群上 47 passed

- [ ] **Step 5: 提交**

```bash
git add models/plm_backbones.py tests/test_plm_backbones.py
git commit -m "feat(models): add LoRA injection and scoped backbone checkpointing"
```

---

## Task 7: 解除 `models/predictor.py` 的 peft 顶层依赖，清理死 import

**为什么先做：** `models/predictor.py:10` 的 `import loralib as lora` 全仓零调用（死 import），而 `loralib` 根本不在 `requirements.txt` 里；`models/predictor.py:11` 的 `from peft import ...` 是顶层 import，导致**没有 peft 的环境连 `import models.predictor` 都做不到**，本地无法对 `forward` 做任何单测。Task 12 的测试依赖这一点。

**Files:**
- Modify: `models/predictor.py:10-13`
- Create: `tests/test_predictor_import.py`

**Interfaces:**
- Consumes: 无
- Produces: `models.predictor` 在无 `peft` / 无 `loralib` 时仍可被 import

- [ ] **Step 1: 写失败测试**

创建 `tests/test_predictor_import.py`：

```python
"""models.predictor 必须能在没有 peft / loralib 的环境里被 import。

KinEAGER 原先在模块顶层 import peft 与 loralib，导致任何不装 peft 的环境
（含本项目的 CPU 单测环境）连导入模型定义都做不到。
"""
import sys


def _drop_modules():
    for name in list(sys.modules):
        if name == "peft" or name == "loralib" or name.startswith("models"):
            del sys.modules[name]


def test_predictor_import_does_not_pull_peft():
    _drop_modules()
    import models.predictor  # noqa: F401

    assert "peft" not in sys.modules, "models.predictor 仍在模块顶层 import peft"


def test_predictor_import_does_not_pull_loralib():
    _drop_modules()
    import models.predictor  # noqa: F401

    assert "loralib" not in sys.modules, "models.predictor 仍在模块顶层 import loralib"


def test_predictor_exposes_public_names():
    _drop_modules()
    import models.predictor as module

    for name in ("KineticsPredictor", "PredictorConfig", "MultiHead", "MLPHead"):
        assert hasattr(module, name), f"models.predictor 缺少 {name}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predictor_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'loralib'`（或 `peft`）

- [ ] **Step 3: 实现**

编辑 `models/predictor.py`，删除第 10-13 行这四个顶层 import：

```python
import loralib as lora
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from transformers import AutoModelForSeq2SeqLM
```

`loralib` 直接删除（全仓零调用）。`peft` 的使用点当前只在 `KineticsPredictor.__init__` 的 `if self.use_lora:` 分支内（`models/predictor.py:632-664`），把 import 移到该分支开头：

```python
        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            def _count_lora_params(model):
```

`transformers` 的两个 AutoModel import 暂时保留在顶层（transformers 本地已装，且 Task 8 会把这两个调用点整体移走）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_predictor_import.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 跑现有测试确认没打破什么**

Run: `python -m pytest tests/ test_scripts/ -v`
Expected: 与改动前一致（`tests/smoke_moe_kcat.py` 不是 `test_` 前缀，pytest 不收集；若报 `peft` 缺失则是既有问题，不是本次引入）

- [ ] **Step 6: 提交**

```bash
git add models/predictor.py tests/test_predictor_import.py
git commit -m "refactor(models): make peft import lazy and drop dead loralib import

models/predictor.py imported peft and loralib at module scope, so the model
definition could not be imported without them. loralib was never called."
```

---

## Task 8: 改造 `models/encoders.py` 走骨干抽象

**Files:**
- Modify: `models/encoders.py`（全文重写，117 行）
- Create: `tests/test_encoders.py`

**Interfaces:**
- Consumes: `BackboneSpec`、`forward_hidden_states`、`build_token_mask`、`load_protein_backbone`、`load_ligand_backbone`
- Produces:
  - `ProteinEncoder(backbone_name=None, d_model=768, device="cuda", backbone_obj=None, train_backbone=False, tokenizer_obj=None, spec=None, max_length=None)`
  - `LigandEncoder(...)` 同签名
  - 两者 `forward(x) -> tuple[Tensor, BoolTensor]`，形状 `(B, L, d_model)` / `(B, L)`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_encoders.py`：

```python
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.encoders import LigandEncoder, ProteinEncoder
from models.plm_backbones import KIND_ESM2, KIND_MOLFORMER, BackboneSpec


class _FakeTokenizer:
    all_special_ids = [0, 2]

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        batch = len(texts)
        length = 4
        return {
            "input_ids": torch.tensor([[0, 5, 6, 2]] * batch),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
        }


class _FakeBackbone(nn.Module):
    def __init__(self, hidden=16, layers=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Identity() for _ in range(layers)])

    def forward(self, **kwargs):
        batch, length = kwargs["input_ids"].shape
        hidden = torch.arange(
            batch * length * self.config.hidden_size, dtype=torch.float32
        ).reshape(batch, length, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden)


def _spec(kind, hidden, layers, max_length=64):
    return BackboneSpec(
        name="fake", kind=kind, trust_remote_code=False,
        max_length=max_length, hidden_size=hidden, num_layers=layers,
    )


def test_protein_encoder_projects_to_d_model():
    encoder = ProteinEncoder(
        d_model=32,
        device="cpu",
        backbone_obj=_FakeBackbone(hidden=16),
        tokenizer_obj=_FakeTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_ESM2, 16, 3),
    )
    tokens, mask = encoder(["MKTAYIAK"])
    assert tokens.shape == (1, 4, 32)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[False, True, True, False]]


def test_protein_encoder_uses_spec_max_length():
    seen = {}

    class _RecordingTokenizer(_FakeTokenizer):
        def __call__(self, texts, **kwargs):
            seen.update(kwargs)
            return super().__call__(texts, **kwargs)

    encoder = ProteinEncoder(
        d_model=8, device="cpu",
        backbone_obj=_FakeBackbone(),
        tokenizer_obj=_RecordingTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_ESM2, 16, 3, max_length=777),
    )
    encoder(["MKTAYIAK"])
    assert seen["max_length"] == 777


def test_ligand_encoder_reads_hidden_size_not_d_model():
    """MoLFormer 是 BERT 系，config 里没有 d_model；旧代码会在这里崩。"""
    encoder = LigandEncoder(
        d_model=32, device="cpu",
        backbone_obj=_FakeBackbone(hidden=16),
        tokenizer_obj=_FakeTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_MOLFORMER, 16, 3, max_length=202),
    )
    tokens, mask = encoder(["CCO"])
    assert tokens.shape == (1, 4, 32)


def test_ligand_encoder_sanitizes_nan():
    class _NanBackbone(_FakeBackbone):
        def forward(self, **kwargs):
            out = super().forward(**kwargs)
            out.last_hidden_state[0, 0, 0] = float("nan")
            return out

    encoder = LigandEncoder(
        d_model=8, device="cpu",
        backbone_obj=_NanBackbone(),
        tokenizer_obj=_FakeTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_MOLFORMER, 16, 3),
    )
    tokens, _ = encoder(["CCO"])
    assert torch.isfinite(tokens).all()


def test_encoders_require_spec_when_backbone_injected():
    with pytest.raises(ValueError):
        ProteinEncoder(d_model=8, device="cpu",
                       backbone_obj=_FakeBackbone(), tokenizer_obj=_FakeTokenizer(),
                       train_backbone=True, spec=None)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_encoders.py -v`
Expected: FAIL — `TypeError: ProteinEncoder.__init__() got an unexpected keyword argument 'spec'`

- [ ] **Step 3: 实现**

用以下内容替换 `models/encoders.py` 全文：

```python
"""蛋白与配体编码器。

骨干的加载、hidden 维度解析、特殊 token 处理全部委托给 models.plm_backbones，
本模块只负责「骨干输出 → 投影到统一的 d_model」以及冻结/LoRA 参数的可见性策略。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .plm_backbones import (
    BackboneSpec,
    build_token_mask,
    forward_hidden_states,
    load_ligand_backbone,
    load_protein_backbone,
)


class _BackboneEncoder(nn.Module):
    """蛋白与配体编码器的共同实现。"""

    def __init__(
        self,
        *,
        backbone_name: str | None,
        d_model: int,
        device: str,
        backbone_obj: nn.Module | None,
        tokenizer_obj: Any | None,
        train_backbone: bool,
        spec: BackboneSpec | None,
        max_length: int | None,
        loader,
    ):
        super().__init__()
        self.device = device

        if backbone_obj is not None and spec is None:
            raise ValueError(
                "注入 backbone_obj 时必须同时提供 spec，"
                "否则无法确定 hidden 维度与最大长度。"
            )

        if backbone_obj is None:
            self.backbone, self.tokenizer, self.spec = loader(
                backbone_name, max_length=max_length
            )
        else:
            self.backbone = backbone_obj
            self.tokenizer = tokenizer_obj
            self.spec = spec

        if self.tokenizer is None:
            raise ValueError("缺少 tokenizer：请提供 tokenizer_obj 或可解析的 backbone_name。")

        self.max_length = int(max_length or self.spec.max_length)
        self.train_backbone = train_backbone
        self._apply_trainability()
        self.proj = nn.Linear(self.spec.hidden_size, d_model).to(device)

    def _apply_trainability(self) -> None:
        raise NotImplementedError

    def encode_tokens(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        enable_grad = any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(enable_grad):
            tokens = forward_hidden_states(
                self.backbone, encoded, output_hidden_states=self.spec.prefer_hidden_states
            )

        mask = build_token_mask(self.tokenizer, encoded, strip_special=True)
        return tokens, mask

    def forward(self, inputs):
        tokens, mask = self.encode_tokens(inputs)
        tokens = self.proj(tokens.float())
        return torch.nan_to_num(tokens), mask


class ProteinEncoder(_BackboneEncoder):
    def __init__(self, backbone_name: str | None = None, d_model: int = 768,
                 device: str = "cuda", backbone_obj: nn.Module | None = None,
                 train_backbone: bool = False, tokenizer_obj: Any | None = None,
                 spec: BackboneSpec | None = None, max_length: int | None = None):
        super().__init__(
            backbone_name=backbone_name, d_model=d_model, device=device,
            backbone_obj=backbone_obj, tokenizer_obj=tokenizer_obj,
            train_backbone=train_backbone, spec=spec, max_length=max_length,
            loader=load_protein_backbone,
        )

    def _apply_trainability(self) -> None:
        """蛋白侧：非 LoRA 参数按 train_backbone 决定；LoRA 参数始终可训练。"""
        if self.train_backbone:
            self.backbone.train()
            return
        for name, param in self.backbone.named_parameters():
            param.requires_grad_("lora_" in name)
        if any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.train()
        else:
            self.backbone.eval()


class LigandEncoder(_BackboneEncoder):
    def __init__(self, backbone_name: str | None = None, d_model: int = 768,
                 device: str = "cuda", backbone_obj: nn.Module | None = None,
                 train_backbone: bool = False, tokenizer_obj: Any | None = None,
                 spec: BackboneSpec | None = None, max_length: int | None = None):
        super().__init__(
            backbone_name=backbone_name, d_model=d_model, device=device,
            backbone_obj=backbone_obj, tokenizer_obj=tokenizer_obj,
            train_backbone=train_backbone, spec=spec, max_length=max_length,
            loader=load_ligand_backbone,
        )

    def _apply_trainability(self) -> None:
        """配体侧：与蛋白侧同策略。"""
        if self.train_backbone:
            self.backbone.train()
            return
        for name, param in self.backbone.named_parameters():
            param.requires_grad_("lora_" in name)
        if any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.train()
        else:
            self.backbone.eval()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_encoders.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add models/encoders.py tests/test_encoders.py
git commit -m "refactor(models): route encoders through the backbone abstraction

Removes the EsmTokenizer / config.d_model / max_length=384 assumptions that
blocked swapping ESM-2 and MolT5 for ESMC-6B and MoLFormer."
```

---

## Task 9: `AffinityScaler`

**Files:**
- Create: `data_affinity.py`
- Create: `tests/test_affinity_data.py`

**Interfaces:**
- Consumes: 无
- Produces: `AffinityScaler`，方法 `fit(y) -> self`、`transform(y) -> np.ndarray`、`inverse_transform(z) -> np.ndarray`、`save(path)`、`load(path)`（classmethod）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_affinity_data.py`：

```python
import numpy as np
import pytest

from data_affinity import AffinityScaler


def test_fit_transform_standardizes_to_zero_mean_unit_std():
    y = np.array([-2.0, 0.0, 4.0, 6.0, 8.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(y)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std()) - 1.0) < 1e-4


def test_inverse_transform_round_trips():
    y = np.array([-1.5, 2.25, 7.0, 11.5], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    restored = scaler.inverse_transform(scaler.transform(y))
    np.testing.assert_allclose(restored, y, atol=1e-4)


def test_deterministic_scale_for_constant_input():
    """常量输入不能让 scale 变成 0，否则后续除零。"""
    y = np.full(5, 5.0, dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(y)
    assert np.isfinite(z).all()


def test_fit_ignores_non_finite_values():
    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    assert np.isfinite(scaler.mean_)
    assert np.isfinite(scaler.scale_)


def test_fit_raises_when_no_finite_values():
    y = np.array([np.nan, np.inf], dtype=np.float32)
    with pytest.raises(ValueError):
        AffinityScaler().fit(y)


def test_transform_preserves_nan_for_invalid_rows():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(np.array([1.0, np.nan, 3.0], dtype=np.float32))
    assert np.isnan(z[1])
    assert np.isfinite(z[0]) and np.isfinite(z[2])


def test_save_and_load_round_trip(tmp_path):
    y = np.array([-1.0, 0.5, 3.0, 9.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    path = tmp_path / "scaler_affinity.json"
    scaler.save(str(path))

    reloaded = AffinityScaler.load(str(path))
    np.testing.assert_allclose(reloaded.transform(y), scaler.transform(y), atol=1e-6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_affinity_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_affinity'`

- [ ] **Step 3: 实现**

创建 `data_affinity.py` 的 Scaler 部分：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_affinity_data.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add data_affinity.py tests/test_affinity_data.py
git commit -m "feat(data): add single-column affinity scaler"
```

---

## Task 10: `AffinityDataset`

**Files:**
- Modify: `data_affinity.py`
- Modify: `tests/test_affinity_data.py`

**Interfaces:**
- Consumes: `AffinityScaler`
- Produces: `AffinityDataset(df, *, label_col="pKaffinity", scaler=None, supervised=True, return_source=False, source_col="affinity_type")`，`__getitem__` 返回 **3 元组 `(seq: str, smi: str, y: torch.Tensor)`**

**为什么是 3 元组：** `utils.py:771-773` 的 `unpack_batch` 已支持 `len(batch) == 3`，解出 `row_ids=None, source_ids=None, p_s=None, l_s=None`。affinity 通路不需要它们，零改动即可对接。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_affinity_data.py`：

```python
import pandas as pd
import torch

from data_affinity import AffinityDataset


def _frame(**overrides):
    data = {
        "pdb_id": ["1a", "1b", "1c"],
        "Sequence": ["MKTAYIAK", "MKTAYIAL", "MKTAYIAM"],
        "Smiles": ["CCO", "CCC", "CCN"],
        "pKaffinity": [4.0, 6.0, 8.0],
        "affinity_type": ["Kd", "Ki", "IC50"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_dataset_returns_three_tuple():
    dataset = AffinityDataset(_frame())
    item = dataset[0]
    assert len(item) == 3
    seq, smi, y = item
    assert seq == "MKTAYIAK"
    assert smi == "CCO"
    assert isinstance(y, torch.Tensor)


def test_dataset_drops_rows_with_non_finite_labels():
    frame = _frame()
    frame.loc[1, "pKaffinity"] = float("nan")
    dataset = AffinityDataset(frame)
    assert len(dataset) == 2
    assert list(dataset.df["pdb_id"]) == ["1a", "1c"]


def test_dataset_raises_when_label_column_missing():
    with pytest.raises(ValueError) as excinfo:
        AffinityDataset(_frame().drop(columns=["pKaffinity"]))
    assert "pKaffinity" in str(excinfo.value)


def test_dataset_raises_when_no_valid_rows():
    frame = _frame()
    frame["pKaffinity"] = float("nan")
    with pytest.raises(ValueError):
        AffinityDataset(frame)


def test_dataset_accepts_sequence_and_smiles_column_aliases():
    frame = _frame().rename(columns={"Sequence": "sequence", "Smiles": "SMILES"})
    dataset = AffinityDataset(frame)
    seq, smi, _ = dataset[0]
    assert seq == "MKTAYIAK"
    assert smi == "CCO"


def test_dataset_applies_scaler():
    frame = _frame()
    scaler = AffinityScaler().fit(frame["pKaffinity"].to_numpy())
    dataset = AffinityDataset(frame, scaler=scaler)
    _, _, y = dataset[0]
    expected = scaler.transform(frame["pKaffinity"].to_numpy())[0]
    assert abs(float(y) - float(expected)) < 1e-5


def test_dataset_without_scaler_returns_raw_label():
    dataset = AffinityDataset(_frame())
    _, _, y = dataset[2]
    assert abs(float(y) - 8.0) < 1e-5


def test_dataset_return_source_flag():
    dataset = AffinityDataset(_frame(), return_source=True)
    item = dataset[0]
    assert len(item) == 4
    assert item[3] == "Kd"


def test_dataset_respects_custom_label_column():
    frame = _frame().rename(columns={"pKaffinity": "pKd"})
    dataset = AffinityDataset(frame, label_col="pKd")
    _, _, y = dataset[1]
    assert abs(float(y) - 6.0) < 1e-5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_affinity_data.py -k dataset -v`
Expected: FAIL — `ImportError: cannot import name 'AffinityDataset'`

- [ ] **Step 3: 实现**

追加到 `data_affinity.py`（顶部补 `import pandas as pd`、`import torch`、`from torch.utils.data import Dataset`）：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_affinity_data.py -v`
Expected: PASS（16 passed）

- [ ] **Step 5: 提交**

```bash
git add data_affinity.py tests/test_affinity_data.py
git commit -m "feat(data): add single-label affinity dataset"
```

---

## Task 11: `tools/build_pdbbind_affinity_csv.py`

**Files:**
- Create: `tools/build_pdbbind_affinity_csv.py`
- Create: `tests/test_pdbbind_csv.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `AFFINITY_PRIORITY = ("Kd", "Ki", "IC50")`
  - `EXCLUDED_PDB_IDS: frozenset[str]`
  - `pick_affinity(record: Mapping[str, Any], *, priority=AFFINITY_PRIORITY) -> tuple[float, str] | None`
  - `build_affinity_table(df, *, id_col, affinity_columns, priority=AFFINITY_PRIORITY, excluded=EXCLUDED_PDB_IDS) -> pd.DataFrame`

- [ ] **Step 1: 在集群上确认真实列名**

PDBbind 索引 CSV（`pdbbind_2020_feats.csv`）的确切列名未在本地确认。先执行：

```bash
head -1 /pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020/pdbbind_2020_feats.csv
```

把 Kd / Ki / IC50 三列的真实列名记录下来，作为后续 `--kd-col` / `--ki-col` / `--ic50-col` 的默认值写入脚本常量 `DEFAULT_AFFINITY_COLUMNS`。**未确认前不要臆造列名。**

- [ ] **Step 2: 写失败测试**

创建 `tests/test_pdbbind_csv.py`：

```python
import math

import pandas as pd
import pytest

from tools.build_pdbbind_affinity_csv import (
    AFFINITY_PRIORITY,
    EXCLUDED_PDB_IDS,
    build_affinity_table,
    pick_affinity,
)


def test_priority_is_kd_then_ki_then_ic50():
    assert AFFINITY_PRIORITY == ("Kd", "Ki", "IC50")


def test_pick_affinity_prefers_kd():
    value, kind = pick_affinity({"Kd": 1e-9, "Ki": 1e-6, "IC50": 1e-5})
    assert kind == "Kd"
    assert abs(value - 9.0) < 1e-6


def test_pick_affinity_falls_back_to_ki():
    value, kind = pick_affinity({"Kd": None, "Ki": 1e-6, "IC50": 1e-5})
    assert kind == "Ki"
    assert abs(value - 6.0) < 1e-6


def test_pick_affinity_falls_back_to_ic50():
    value, kind = pick_affinity({"Kd": None, "Ki": float("nan"), "IC50": 1e-5})
    assert kind == "IC50"
    assert abs(value - 5.0) < 1e-6


def test_pick_affinity_returns_none_when_nothing_usable():
    assert pick_affinity({"Kd": None, "Ki": 0.0, "IC50": float("nan")}) is None


def test_excluded_ids_are_the_documented_set():
    assert EXCLUDED_PDB_IDS == frozenset(
        {"2pll", "2vgc", "3d45", "3pjg", "4b70", "6ajz", "6e5b", "6fmp"}
    )


def _index_frame():
    return pd.DataFrame(
        {
            "pdb_id": ["1abc", "1abc", "1def", "2pll"],
            "Kd": [1e-9, 1e-9, None, 1e-9],
            "Ki": [None, None, 1e-7, None],
            "IC50": [None, None, None, None],
            "Sequence": ["MKT", "MKT", "MKA", "MKB"],
            "Smiles": ["CCO", "CCO", "CCC", "CCN"],
        }
    )


def test_build_table_collapses_duplicate_pdb_ids():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert list(out["pdb_id"]) == ["1abc", "1def"]


def test_build_table_excludes_known_bad_ids():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert "2pll" not in set(out["pdb_id"])


def test_build_table_emits_expected_columns():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    for column in ("pdb_id", "Sequence", "Smiles", "pKaffinity", "affinity_type"):
        assert column in out.columns


def test_build_table_label_is_negative_log10():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    row = out[out["pdb_id"] == "1def"].iloc[0]
    assert row["affinity_type"] == "Ki"
    assert abs(float(row["pKaffinity"]) - 7.0) < 1e-6


def test_build_table_drops_rows_without_usable_affinity():
    frame = _index_frame()
    frame.loc[2, "Ki"] = None
    out = build_affinity_table(
        frame, id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert "1def" not in set(out["pdb_id"])


def test_build_table_raises_when_column_missing():
    frame = _index_frame().drop(columns=["IC50"])
    with pytest.raises(ValueError) as excinfo:
        build_affinity_table(
            frame, id_col="pdb_id",
            affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
        )
    assert "IC50" in str(excinfo.value)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_pdbbind_csv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.build_pdbbind_affinity_csv'`

- [ ] **Step 4: 实现**

创建 `tools/build_pdbbind_affinity_csv.py`：

```python
"""把 PDBbind v2020 索引表转换成单标签亲和力训练 CSV。

标签口径：-log10(Kd/Ki/IC50)，按 Kd > Ki > IC50 的优先级取第一个可用值。
同一 pdb_id 的多行（链/界面级索引）折叠为一行。
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Mapping

import pandas as pd

AFFINITY_PRIORITY = ("Kd", "Ki", "IC50")

# 8 个在 xtrimofold3 的 MoLFormer 缓存里缺失的样本，见
# docs/HANDOFF_molformer_esmc.md:74-81。这里一并排除，保证两边样本集可比。
EXCLUDED_PDB_IDS = frozenset(
    {"2pll", "2vgc", "3d45", "3pjg", "4b70", "6ajz", "6e5b", "6fmp"}
)

# 由 Step 1 在集群上确认后填写真实列名。
DEFAULT_AFFINITY_COLUMNS = {"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"}


def _as_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def pick_affinity(
    record: Mapping[str, Any],
    *,
    priority: tuple[str, ...] = AFFINITY_PRIORITY,
) -> tuple[float, str] | None:
    """返回 (-log10(value), 类型)；全都不可用时返回 None。"""
    for kind in priority:
        value = _as_positive_float(record.get(kind))
        if value is not None:
            return -math.log10(value), kind
    return None


def build_affinity_table(
    df: pd.DataFrame,
    *,
    id_col: str,
    affinity_columns: Mapping[str, str],
    priority: tuple[str, ...] = AFFINITY_PRIORITY,
    excluded: frozenset[str] = EXCLUDED_PDB_IDS,
    sequence_col: str = "Sequence",
    smiles_col: str = "Smiles",
) -> pd.DataFrame:
    missing = [name for name in affinity_columns.values() if name not in df.columns]
    if missing:
        raise ValueError(
            f"索引表缺少亲和力列 {missing}；实际列: {sorted(df.columns)}"
        )
    if id_col not in df.columns:
        raise ValueError(f"索引表缺少 id 列 {id_col!r}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pdb_id = str(row[id_col]).strip().lower()
        if not pdb_id or pdb_id in excluded:
            continue

        source = {kind: row.get(column) for kind, column in affinity_columns.items()}
        picked = pick_affinity(source, priority=priority)
        if picked is None:
            continue
        label, affinity_type = picked

        records.append(
            {
                "pdb_id": pdb_id,
                "Sequence": row.get(sequence_col),
                "Smiles": row.get(smiles_col),
                "pKaffinity": float(label),
                "affinity_type": affinity_type,
            }
        )

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(columns=["pdb_id", "Sequence", "Smiles", "pKaffinity", "affinity_type"])

    table["_priority"] = table["affinity_type"].map(
        {kind: rank for rank, kind in enumerate(priority)}
    )
    table = (
        table.sort_values(["pdb_id", "_priority"])
        .drop_duplicates(subset="pdb_id", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", required=True,
                        help="PDBbind v2020 索引表，例如 pdbbind_2020_feats.csv")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--id-col", default="pdb_id")
    parser.add_argument("--sequence-col", default="Sequence")
    parser.add_argument("--smiles-col", default="Smiles")
    parser.add_argument("--kd-col", default=DEFAULT_AFFINITY_COLUMNS["Kd"])
    parser.add_argument("--ki-col", default=DEFAULT_AFFINITY_COLUMNS["Ki"])
    parser.add_argument("--ic50-col", default=DEFAULT_AFFINITY_COLUMNS["IC50"])
    parser.add_argument("--split-csv", default=None,
                        help="可选：既有划分文件，仅用于统计覆盖情况")
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    table = build_affinity_table(
        df,
        id_col=args.id_col,
        affinity_columns={"Kd": args.kd_col, "Ki": args.ki_col, "IC50": args.ic50_col},
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
    )
    table.to_csv(args.out_csv, index=False)

    print(f"[PDBbind] 输入行数={len(df)} 输出复合物数={len(table)}")
    print(f"[PDBbind] 标签类型分布={table['affinity_type'].value_counts().to_dict()}")
    print(f"[PDBbind] pKaffinity 统计: min={table['pKaffinity'].min():.3f} "
          f"max={table['pKaffinity'].max():.3f} mean={table['pKaffinity'].mean():.3f} "
          f"std={table['pKaffinity'].std():.3f}")
    if args.split_csv:
        split = pd.read_csv(args.split_csv)
        split_ids = set(split.iloc[:, 0].astype(str).str.strip().str.lower())
        covered = len(split_ids & set(table["pdb_id"]))
        print(f"[PDBbind] 与 {args.split_csv} 的交集={covered} / {len(split_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_pdbbind_csv.py -v`
Expected: PASS（11 passed）

- [ ] **Step 6: 提交**

```bash
git add tools/build_pdbbind_affinity_csv.py tests/test_pdbbind_csv.py
git commit -m "feat(tools): add PDBbind v2020 to single-label affinity CSV builder"
```

---

## Task 12: `models/affinity_loss.py`

从 `models/kcat_expert.py:276-332` 的 `kcat_expert_loss` 泛化而来 —— **该函数已经是单任务形式**，不需要重新设计损失。

**Files:**
- Create: `models/affinity_loss.py`
- Create: `tests/test_affinity_loss.py`

**Interfaces:**
- Consumes: `main_train_predictor_multigpu.py:247-261` 的 Barron 损失思路（在下方内联实现，避免从训练脚本反向 import）
- Produces: `affinity_loss(mu, log_var, y, *, loss_type="huber", huber_beta=1.0, barron_alpha=1.0, barron_scale=1.0, nll_weight=0.05, rank_weight=0.0, rank_pairs=256, rank_margin=0.0, var_reg_weight=0.0, var_reg_mode="abs") -> tuple[Tensor, dict[str, Tensor]]`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_affinity_loss.py`：

```python
import math

import pytest
import torch

from models.affinity_loss import affinity_loss


def _inputs(n=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, generator=generator)
    log_var = torch.randn(n, generator=generator) * 0.5
    y = torch.randn(n, generator=generator)
    return mu, log_var, y


def test_loss_is_finite_and_scalar():
    mu, log_var, y = _inputs()
    loss, logs = affinity_loss(mu, log_var, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "total" in logs and "huber" in logs and "nll" in logs


def test_loss_backpropagates_to_both_outputs():
    mu, log_var, y = _inputs()
    mu = mu.clone().requires_grad_(True)
    log_var = log_var.clone().requires_grad_(True)
    loss, _ = affinity_loss(mu, log_var, y, nll_weight=0.1)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert log_var.grad is not None and torch.isfinite(log_var.grad).all()


def test_nan_labels_are_masked_not_propagated():
    mu, log_var, y = _inputs()
    y[0] = float("nan")
    y[1] = float("inf")
    loss, _ = affinity_loss(mu, log_var, y)
    assert torch.isfinite(loss)


def test_all_labels_invalid_returns_finite_zero_loss():
    mu, log_var, _ = _inputs()
    y = torch.full((8,), float("nan"))
    loss, _ = affinity_loss(mu, log_var, y)
    assert torch.isfinite(loss)


def test_extreme_log_var_is_clamped():
    mu, _, y = _inputs()
    log_var = torch.tensor([1e6, -1e6] * 4, dtype=torch.float32)
    loss, _ = affinity_loss(mu, log_var, y, nll_weight=1.0)
    assert torch.isfinite(loss)


def test_nll_weight_zero_removes_nll_term():
    mu, log_var, y = _inputs()
    _, logs_zero = affinity_loss(mu, log_var, y, nll_weight=0.0)
    _, logs_used = affinity_loss(mu, log_var, y, nll_weight=1.0)
    assert abs(float(logs_zero["total"]) - float(logs_used["total"])) > 1e-6


def test_rank_weight_zero_skips_ranking():
    mu, log_var, y = _inputs()
    _, logs = affinity_loss(mu, log_var, y, rank_weight=0.0)
    assert float(logs["rank"]) == 0.0


def test_rank_weight_positive_enables_ranking():
    mu, log_var, y = _inputs()
    _, logs = affinity_loss(mu, log_var, y, rank_weight=1.0, rank_margin=0.5)
    assert float(logs["rank"]) >= 0.0


def test_single_sample_batch_does_not_crash():
    mu = torch.tensor([1.0])
    log_var = torch.tensor([0.0])
    y = torch.tensor([1.5])
    loss, _ = affinity_loss(mu, log_var, y, rank_weight=1.0, var_reg_weight=1.0)
    assert torch.isfinite(loss)


def test_barron_loss_type_runs():
    mu, log_var, y = _inputs()
    loss, logs = affinity_loss(mu, log_var, y, loss_type="barron", barron_alpha=1.0)
    assert torch.isfinite(loss)
    assert "barron" in logs or "huber" in logs


def test_unknown_loss_type_raises():
    mu, log_var, y = _inputs()
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, loss_type="banana")


def test_var_reg_mode_validated():
    mu, log_var, y = _inputs()
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, var_reg_weight=1.0, var_reg_mode="banana")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_affinity_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.affinity_loss'`

- [ ] **Step 3: 实现**

创建 `models/affinity_loss.py`：

```python
"""单任务亲和力回归损失。

由 models/kcat_expert.py:276-332 的 kcat_expert_loss 泛化而来：该函数本就是单任务形式
（Huber + ranking + var_reg + NLL），只是命名与 kcat 绑定。这里去掉 CCC 项，
补上 Barron 稳健损失选项（对齐 main_train_predictor_multigpu.py:247-261）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0


def _barron_loss(error: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
    """Barron 稳健损失（CVPR 2019），alpha=2 退化为 L2，alpha=0 近似 L1。"""
    scaled = (error / scale) ** 2
    if abs(alpha - 2.0) < 1e-6:
        return 0.5 * scaled
    if abs(alpha) < 1e-6:
        return torch.log1p(0.5 * scaled)
    factor = abs(alpha - 2.0) / alpha
    return (torch.pow(1.0 + scaled / factor, factor) - 1.0) * factor


def affinity_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_beta: float = 1.0,
    barron_alpha: float = 1.0,
    barron_scale: float = 1.0,
    nll_weight: float = 0.05,
    rank_weight: float = 0.0,
    rank_pairs: int = 256,
    rank_margin: float = 0.0,
    var_reg_weight: float = 0.0,
    var_reg_mode: str = "abs",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (总损失, 分项日志)。所有分项均为标量张量，便于直接写日志。"""
    if loss_type not in ("huber", "barron"):
        raise ValueError(f"未知 loss_type: {loss_type!r}（支持 huber / barron）")
    if var_reg_mode not in ("abs", "ratio"):
        raise ValueError(f"未知 var_reg_mode: {var_reg_mode!r}（支持 abs / ratio）")

    mu = torch.nan_to_num(mu.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_var = torch.nan_to_num(
        log_var.float(), nan=0.0, posinf=_LOG_VAR_MAX, neginf=_LOG_VAR_MIN
    ).clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
    y = torch.nan_to_num(y.float(), nan=float("nan"), posinf=float("nan"),
                         neginf=float("nan"))

    valid = torch.isfinite(y)
    device = mu.device
    logs: dict[str, torch.Tensor] = {}

    if not valid.any():
        zero = torch.zeros((), device=device, dtype=mu.dtype)
        logs.update({"huber": zero, "barron": zero, "rank": zero,
                     "var_reg": zero, "nll": zero, "total": zero})
        return zero, logs

    mu_v = mu[valid]
    y_v = y[valid]
    log_var_v = log_var[valid]

    if loss_type == "barron":
        regression = _barron_loss(mu_v - y_v, barron_alpha, barron_scale).mean()
        logs["barron"] = regression.detach()
        logs["huber"] = regression.detach()
    else:
        regression = F.smooth_l1_loss(mu_v, y_v, beta=huber_beta)
        logs["huber"] = regression.detach()
        logs["barron"] = torch.zeros((), device=device, dtype=mu.dtype)

    nll = 0.5 * (log_var_v + (y_v - mu_v) ** 2 / log_var_v.exp().clamp_min(1e-6))
    nll = nll.mean()
    logs["nll"] = nll.detach()

    rank = torch.zeros((), device=device, dtype=mu.dtype)
    n_valid = int(y_v.numel())
    if rank_weight > 0.0 and n_valid >= 2:
        n_pairs = min(int(rank_pairs), n_valid * (n_valid - 1))
        if n_pairs > 0:
            i = torch.randint(0, n_valid, (n_pairs,), device=device)
            j = torch.randint(0, n_valid, (n_pairs,), device=device)
            keep = i != j
            i, j = i[keep], j[keep]
            sign = torch.sign(y_v[i] - y_v[j])
            nonzero = sign != 0
            if nonzero.any():
                rank = F.margin_ranking_loss(
                    mu_v[i[nonzero]], mu_v[j[nonzero]], sign[nonzero],
                    margin=float(rank_margin),
                )
    logs["rank"] = rank.detach()

    var_reg = torch.zeros((), device=device, dtype=mu.dtype)
    if var_reg_weight > 0.0 and n_valid >= 2:
        std_pred = mu_v.std(unbiased=False)
        std_true = y_v.std(unbiased=False)
        if var_reg_mode == "ratio":
            var_reg = torch.relu(std_true - std_pred)
        else:
            var_reg = (std_pred - std_true).abs()
    logs["var_reg"] = var_reg.detach()

    total = (
        regression
        + float(nll_weight) * nll
        + float(rank_weight) * rank
        + float(var_reg_weight) * var_reg
    )
    logs["total"] = total.detach()
    return total, logs
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_affinity_loss.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: 提交**

```bash
git add models/affinity_loss.py tests/test_affinity_loss.py
git commit -m "feat(models): add single-task affinity loss generalized from kcat_expert_loss"
```

---

## Task 13: `PredictorConfig.for_task` 与 affinity forward 分支

**关键约束：** affine 分支必须在 `forward` 的**末端**分叉（trunk 完全共用），并且**不得构造** `s_kcat`/`s_Km`/`s_ratio`/`s_domain_task`/`kcat_source_residual`。kinetics 分支的代码一行不改。

**Files:**
- Modify: `models/predictor.py`（`PredictorConfig` 类、`KineticsPredictor.__init__`、`KineticsPredictor.forward`）
- Create: `tests/test_affinity_forward.py`

**Interfaces:**
- Consumes: `models.encoders.ProteinEncoder` / `LigandEncoder`（Task 8）
- Produces:
  - `PredictorConfig(task="kinetics"|"affinity", ...)`
  - `PredictorConfig.for_task(task: str, **overrides) -> PredictorConfig`
  - `KineticsPredictor(..., protein_encoder=None, ligand_encoder=None)` —— 注入时跳过骨干加载与 LoRA 注入，用于测试
  - `KineticsPredictor.forward(...) -> tuple[Tensor, Tensor]`（affinity，两项形状 `(B,)`）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_affinity_forward.py`：

```python
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.encoders import LigandEncoder, ProteinEncoder
from models.plm_backbones import KIND_ESM2, KIND_MOLFORMER, BackboneSpec
from models.predictor import KineticsPredictor, PredictorConfig


class _FakeTokenizer:
    all_special_ids = [0, 2]

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        batch = len(texts)
        return {
            "input_ids": torch.tensor([[0, 5, 6, 2]] * batch),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


class _FakeBackbone(nn.Module):
    def __init__(self, hidden=16, layers=2):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Identity() for _ in range(layers)])

    def forward(self, **kwargs):
        batch, length = kwargs["input_ids"].shape
        return SimpleNamespace(
            last_hidden_state=torch.randn(batch, length, self.config.hidden_size)
        )


def _spec(kind, hidden, layers):
    return BackboneSpec(name="fake", kind=kind, trust_remote_code=False,
                        max_length=64, hidden_size=hidden, num_layers=layers)


def _fake_encoders(d_model=32):
    protein = ProteinEncoder(d_model=d_model, device="cpu",
                             backbone_obj=_FakeBackbone(16, 2),
                             tokenizer_obj=_FakeTokenizer(),
                             train_backbone=True, spec=_spec(KIND_ESM2, 16, 2))
    ligand = LigandEncoder(d_model=d_model, device="cpu",
                           backbone_obj=_FakeBackbone(16, 2),
                           tokenizer_obj=_FakeTokenizer(),
                           train_backbone=True, spec=_spec(KIND_MOLFORMER, 16, 2))
    return protein, ligand


def _small_cfg(**overrides):
    base = dict(d_model=32, num_heads=4, num_interaction_layers=1,
                num_token_transformer_layers=0, head_hidden=16,
                transformer_ffn_dim=64, fusion_proj_dim=16,
                use_attention_pool=False, use_checkpoint=False)
    base.update(overrides)
    return base


def test_for_task_affinity_selects_single_two_output_head():
    cfg = PredictorConfig.for_task("affinity", **_small_cfg())
    assert cfg.task == "affinity"
    assert cfg.active_heads == ["affinity"]
    assert cfg.head_specs["affinity"]["out_dim"] == 2


def test_for_task_kinetics_preserves_original_three_heads():
    cfg = PredictorConfig.for_task("kinetics", **_small_cfg())
    assert cfg.task == "kinetics"
    assert cfg.active_heads == ["kcat", "Km", "ratio"]


def test_for_task_rejects_unknown_task():
    with pytest.raises(ValueError):
        PredictorConfig.for_task("banana", **_small_cfg())


def test_affinity_forward_returns_mu_and_log_var():
    cfg = PredictorConfig.for_task("affinity", **_small_cfg())
    protein, ligand = _fake_encoders()
    model = KineticsPredictor(device="cpu", cfg=cfg, use_lora=False,
                              protein_encoder=protein, ligand_encoder=ligand)
    model.eval()

    with torch.no_grad():
        mu, log_var = model(["MKTAYIAK", "MKTAYIAL"], ["CCO", "CCC"])

    assert mu.shape == (2,)
    assert log_var.shape == (2,)


def test_affinity_model_does_not_construct_kinetics_parameters():
    """DDP 在 find_unused_parameters=False 下会因「有图无梯度」的参数报错。"""
    cfg = PredictorConfig.for_task("affinity", **_small_cfg())
    protein, ligand = _fake_encoders()
    model = KineticsPredictor(device="cpu", cfg=cfg, use_lora=False,
                              protein_encoder=protein, ligand_encoder=ligand)

    for name in ("s_kcat", "s_Km", "s_ratio", "s_domain_task", "kcat_source_residual"):
        assert not hasattr(model, name), f"affinity 通路不应构造 {name}"


def test_kinetics_model_still_constructs_kinetics_parameters():
    cfg = PredictorConfig.for_task("kinetics", **_small_cfg())
    protein, ligand = _fake_encoders()
    model = KineticsPredictor(device="cpu", cfg=cfg, use_lora=False,
                              protein_encoder=protein, ligand_encoder=ligand)

    for name in ("s_kcat", "s_Km", "s_ratio"):
        assert hasattr(model, name), f"kinetics 通路必须保留 {name}"
        assert isinstance(getattr(model, name), nn.Parameter)


def test_kinetics_forward_return_arity_unchanged():
    """Task 13 不得改变 kinetics 的返回形状。"""
    cfg = PredictorConfig.for_task("kinetics", **_small_cfg())
    protein, ligand = _fake_encoders()
    model = KineticsPredictor(device="cpu", cfg=cfg, use_lora=False,
                              protein_encoder=protein, ligand_encoder=ligand)
    model.eval()

    with torch.no_grad():
        out = model(["MKTAYIAK"], ["CCO"])

    assert isinstance(out, tuple) and len(out) == 7
    pred_kcat, pred_Km, pred_ratio, gate_p, gate_s, s_raw, s_eff = out
    assert pred_kcat.shape == (1,)
    assert len(s_raw) == 3 and len(s_eff) == 3


def test_affinity_trainable_parameter_count_is_smaller():
    """affinity 通路不应带三个任务的异方差参数。"""
    def n_params(task):
        cfg = PredictorConfig.for_task(task, **_small_cfg())
        protein, ligand = _fake_encoders()
        model = KineticsPredictor(device="cpu", cfg=cfg, use_lora=False,
                                  protein_encoder=protein, ligand_encoder=ligand)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n_params("affinity") < n_params("kinetics")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_affinity_forward.py -v`
Expected: FAIL — `AttributeError: type object 'PredictorConfig' has no attribute 'for_task'`

- [ ] **Step 3: 给 `PredictorConfig` 加 `task` 与工厂方法**

在 `models/predictor.py` 的 `PredictorConfig.__init__` 参数表末尾（`use_gate_l=True` 之后）加 `task="kinetics"`，并在函数体末尾加 `self.task = task`。

在 `PredictorConfig` 类内追加：

```python
    KINETICS_HEADS = ["kcat", "Km", "ratio"]

    @classmethod
    def for_task(cls, task: str, **overrides) -> "PredictorConfig":
        """按任务产出预设配置，避免在训练脚本里硬编码 active_heads / head_specs。"""
        if task == "kinetics":
            preset = {
                "task": "kinetics",
                "active_heads": list(cls.KINETICS_HEADS),
                "head_specs": {
                    "kcat":  {"type": "mlp", "hidden_dim": 768, "depth": 3,
                              "dropout": 0.10, "out_dim": 1},
                    "Km":    {"type": "mlp", "hidden_dim": 640, "depth": 3,
                              "dropout": 0.10, "out_dim": 1},
                    "ratio": {"type": "mlp", "out_dim": 1},
                },
            }
        elif task == "affinity":
            preset = {
                "task": "affinity",
                "active_heads": ["affinity"],
                "head_specs": {
                    "affinity": {"type": "mlp", "hidden_dim": 768, "depth": 3,
                                 "dropout": 0.10, "out_dim": 2},
                },
            }
        else:
            raise ValueError(f"未知 task: {task!r}（支持 kinetics / affinity）")

        preset.update(overrides)
        return cls(**preset)
```

- [ ] **Step 4: 给 `KineticsPredictor.__init__` 加注入点并守护 kinetics 参数**

1. 在 `__init__` 签名末尾加 `protein_encoder=None, ligand_encoder=None`。
2. 把 `self.esm = AutoModel.from_pretrained(...)` 到 `self.ligand = LigandEncoder(...)` 这整段（当前 `models/predictor.py:480-481` 与 `533-680`）替换为：

```python
        if protein_encoder is not None and ligand_encoder is not None:
            # 测试注入路径：跳过骨干加载与 LoRA 注入。
            self.protein = protein_encoder
            self.ligand = ligand_encoder
            self.esm = getattr(protein_encoder, "backbone", None)
            self.molt5 = getattr(ligand_encoder, "backbone", None)
        else:
            self.esm, self._prot_spec = None, None
            self.molt5, self._lig_spec = None, None
            self._build_backbones_and_lora(esm_model, molt5_model)   # 见 Task 13b

        self.protein = self.protein if protein_encoder is not None else ProteinEncoder(
            backbone_name=esm_model, backbone_obj=self.esm, tokenizer_obj=self._prot_tokenizer,
            spec=self._prot_spec, d_model=C, device=self.device, train_backbone=True,
        )
        self.ligand = self.ligand if ligand_encoder is not None else LigandEncoder(
            backbone_name=molt5_model, backbone_obj=self.molt5, tokenizer_obj=self._lig_tokenizer,
            spec=self._lig_spec, d_model=C, device=self.device, train_backbone=True,
        )
```

> **实现说明（Task 13b，与 Task 13 同一步完成）：** 抽出 `_build_backbones_and_lora()`，内部用 Task 5 的 `load_*_backbone` 与 Task 6 的 `inject_lora` 取代原 `models/predictor.py:480-481` 的 `AutoModel.from_pretrained` / `AutoModelForSeq2SeqLM.from_pretrained`、以及 `533-664` 的 `_build_esm_targets_exact` / `_build_t5_encoder_targets_exact`。**`self.molt5.get_encoder()` 只在 `spec.kind == KIND_T5` 时调用**；MoLFormer 直接把 backbone 传给 `LigandEncoder`。

3. 把 `models/predictor.py:741-750` 的 `kcat_source_residual` 与 `766-772` 的 `s_kcat`/`s_Km`/`s_ratio`/`s_domain_task` 用条件包裹：

```python
        self.kcat_source_residual = None
        if cfg.task == "kinetics" and bool(getattr(cfg, "use_source_residual_head", False)):
            tasks = set(getattr(cfg, "source_head_tasks", ["kcat"]))
            if "kcat" in tasks:
                self.kcat_source_residual = SourceResidualHead(...)

        self.s_kcat = self.s_Km = self.s_ratio = None
        self.s_domain_task = None
        if cfg.task == "kinetics":
            self.s_kcat = nn.Parameter(torch.zeros(1))
            self.s_Km = nn.Parameter(torch.zeros(1))
            self.s_ratio = nn.Parameter(torch.zeros(1))
            if bool(getattr(cfg, "domain_uncertainty", False)):
                self.s_domain_task = nn.Parameter(
                    torch.zeros(int(getattr(cfg, "num_domains", 3)), 3)
                )
```

- [ ] **Step 5: 给 `forward` 加 affinity 末端分支**

把 `models/predictor.py:959-965` 替换为：

```python
        outs_list, outs_dict = self.multi_head(x, cond=x)
        self.last_pred_dict = outs_dict

        if getattr(self.cfg, "task", "kinetics") == "affinity":
            affinity = outs_dict["affinity"]
            return affinity[..., 0], affinity[..., 1]

        heads = self.cfg.active_heads  # ["kcat", "Km", "ratio"]
        outs_list = [outs_dict[name].squeeze(-1) for name in heads]

        pred_kcat_log, pred_Km_log, pred_ratio_log = outs_list[:3]
```

`saliency()`（`models/predictor.py:1019-1044`）在开头加：

```python
        if getattr(self.cfg, "task", "kinetics") == "affinity":
            raise NotImplementedError("saliency 目前只支持 kinetics 通路")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_affinity_forward.py -v`
Expected: PASS（8 passed）

- [ ] **Step 7: 全量单测回归**

Run: `python -m pytest tests/ -v`
Expected: 全绿（peft 相关 skip）

- [ ] **Step 8: 提交**

```bash
git add models/predictor.py tests/test_affinity_forward.py
git commit -m "feat(models): add affinity task branch to KineticsPredictor

Adds PredictorConfig.for_task and a single-head affinity forward path.
The affinity branch must not construct s_kcat/s_Km/s_ratio/s_domain_task
or kcat_source_residual: DDP with find_unused_parameters=False rejects
parameters that participate in the graph without receiving gradients."
```

---

## Task 14: 训练脚本 — `--task` 与配置/数据分支

**Files:**
- Modify: `main_train_predictor_multigpu.py`
- Modify: `tests/test_predictor_import.py`（追加参数解析测试）

**Interfaces:**
- Consumes: `PredictorConfig.for_task`、`AffinityScaler`、`AffinityDataset`
- Produces: `build_arg_parser() -> argparse.ArgumentParser`（把 `main()` 里的 `p.add_argument(...)` 群抽成函数，便于测试）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_predictor_import.py`：

```python
def test_arg_parser_has_task_defaulting_to_kinetics():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(["-csv", "x.csv", "-esm", "a", "-molt5", "b"])
    assert args.task == "kinetics"


def test_arg_parser_accepts_affinity_task():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(
        ["-csv", "x.csv", "-esm", "a", "-molt5", "b", "--task", "affinity"]
    )
    assert args.task == "affinity"


def test_arg_parser_lora_defaults_to_last_k_30():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(["-csv", "x.csv", "-esm", "a", "-molt5", "b"])
    assert args.lora_layer_policy == "last_k"
    assert args.lora_last_k == 30


def test_arg_parser_accepts_lora_target_suffixes_override():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(
        ["-csv", "x.csv", "-esm", "a", "-molt5", "b",
         "--lora_target_suffixes", "mixer", "gate_proj"]
    )
    assert args.lora_target_suffixes == ["mixer", "gate_proj"]


def test_arg_parser_rejects_unknown_task():
    from main_train_predictor_multigpu import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["-csv", "x.csv", "-esm", "a", "-molt5", "b", "--task", "banana"]
        )
```

（该文件顶部补 `import pytest`。）

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predictor_import.py -k arg_parser -v`
Expected: FAIL — `ImportError: cannot import name 'build_arg_parser'`

- [ ] **Step 3: 抽出 `build_arg_parser()` 并加新参数**

把 `main_train_predictor_multigpu.py:2245-2440` 的 `p = argparse.ArgumentParser(...)` 到 `args = p.parse_args()` 之间整段移入新的模块级函数：

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # ... 原有全部 add_argument 原样搬入 ...
    return p
```

`main()` 内改为 `args = build_arg_parser().parse_args()`。

在 `build_arg_parser()` 里追加：

```python
    # ---- PDBbind 亲和力通路 ----
    p.add_argument("--task", type=str, default="kinetics",
                   choices=["kinetics", "affinity"],
                   help="kinetics=原有 kcat/Km/ratio 三任务；affinity=单任务亲和力")
    p.add_argument("--affinity_label_col", type=str, default="pKaffinity")
    p.add_argument("--affinity_loss_type", type=str, default="huber",
                   choices=["huber", "barron"])
    p.add_argument("--affinity_nll_weight", type=float, default=0.05)
    p.add_argument("--affinity_rank_weight", type=float, default=0.0)
    p.add_argument("--affinity_rank_margin", type=float, default=0.0)
    p.add_argument("--affinity_var_reg_weight", type=float, default=0.0)
    p.add_argument("--affinity_var_reg_mode", type=str, default="abs",
                   choices=["abs", "ratio"])

    # ---- LoRA 与骨干 ----
    p.add_argument("--lora_layer_policy", type=str, default="last_k",
                   choices=["all", "last_k", "explicit"])
    p.add_argument("--lora_last_k", type=int, default=30)
    p.add_argument("--lora_explicit_layers", type=int, nargs="+", default=None)
    p.add_argument("--lora_target_suffixes", type=str, nargs="+", default=None,
                   help="覆盖 LoRA 目标后缀白名单（由 tools/probe_backbone_modules.py 的"
                        "leaf_names 决定）")
    p.add_argument("--plm_max_len_prot", type=int, default=None)
    p.add_argument("--plm_max_len_lig", type=int, default=None)
    p.add_argument("--grad_checkpoint_backbone", type=str, default="off",
                   choices=["off", "active", "all"])

    # ---- 分布式与吞吐 ----
    p.add_argument("--iters_to_accumulate", type=int,
                   default=int(os.environ.get("KINEAGER_ITERS_TO_ACCUMULATE", "1")))
    p.add_argument("--num_workers", type=int,
                   default=int(os.environ.get("KINEAGER_NUM_WORKERS", "8")))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_predictor_import.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 在 `main()` 里按 task 分支配置与数据**

把 `main_train_predictor_multigpu.py:2610-2651` 的 `cfg = PredictorConfig(...)` + `model = KineticsPredictor(...)` 替换为：

```python
    if args.task == "affinity":
        cfg = PredictorConfig.for_task(
            "affinity",
            d_model=768, num_heads=8, num_interaction_layers=4,
            num_mamba_layers=1, rate=0.1, head_hidden=768,
            use_attention_pool=True, use_checkpoint=True,
            num_stages=2, num_interaction_layers_per_stage=3,
            use_struct_branch=args.use_struct_branch,
            struct_in_dim_prot=args.struct_in_dim_prot,
            struct_in_dim_lig=args.struct_in_dim_lig,
            struct_fusion_mode=getattr(args, "struct_fusion_mode", "concat"),
            enable_avail_gate=bool(getattr(args, "enable_avail_gate", False)),
            avail_gate_hidden=int(getattr(args, "avail_gate_hidden", 256)),
            use_phys_branch=use_phys_evidence,
            phys_in_dim=len(phys_cols) if use_phys_evidence else 0,
            phys_hidden=128,
            domain_uncertainty=False,      # affinity 通路关闭域不确定性
            num_domains=1,
            use_interactions=args.use_interactions,
            use_gate_p=args.use_gate_p,
            use_gate_l=args.use_gate_l,
            fusion_norm=True,
            fusion_proj_dim=512,
            use_moe_in_head=False,
            num_heads_in_head=1,
            head_hidden_dim=768,
            head_depth=3,
        )
    else:
        cfg = PredictorConfig(
            # ... 原有 2610-2649 的内容逐行保留 ...
        )
```

> **注意：** `active_heads` / `head_specs` 由 `for_task` 提供；kinetics 分支保留原有显式写法，确保逐字节不变。

把 `main_train_predictor_multigpu.py:2535-2538` 的 scaler 与 `2542-2559` 的 Dataset 改为：

```python
    if args.task == "affinity":
        kscaler = AffinityScaler()
        kscaler.fit(train_df[args.affinity_label_col].to_numpy())
        if is_main_process():
            kscaler.save(os.path.join(run_dir, "scaler_affinity.json"))

        def _make_affinity_dataset(frame, source_ids):
            return AffinityDataset(
                frame, label_col=args.affinity_label_col, scaler=kscaler,
                supervised=True, return_source=False,
            )

        train_dataset = _make_affinity_dataset(train_df, train_source_ids)
        val_dataset = _make_affinity_dataset(val_df, val_source_ids)
    else:
        kscaler = KineticsScaler()
        kscaler.fit(train_df["kcat(s^-1)"], train_df["Km(M)"])
        # ... 原有 2537-2559 逐行保留 ...
```

把 `train_loader` / `val_loader` 的 `num_workers=8` / `num_workers=4` 改为 `args.num_workers` / `max(1, args.num_workers // 2)`。

把 `main_train_predictor_multigpu.py:2597-2605` 的 `esm_layers` / `molt5_layers` 推断改为在 affinity 下从 `spec.num_layers` 取（`esm_layers = self._prot_spec.num_layers`），避免 remote-code config 缺字段时崩溃。

- [ ] **Step 6: 干跑验证参数解析与分支不炸**

Run: `python main_train_predictor_multigpu.py --help | Select-String "task|affinity|lora_last_k"`
Expected: 新参数出现在帮助里

- [ ] **Step 7: 提交**

```bash
git add main_train_predictor_multigpu.py tests/test_predictor_import.py
git commit -m "feat(train): add --task switch and affinity data path

Extracts build_arg_parser for testability and routes config/scaler/dataset
by task. The kinetics branch keeps its original explicit construction."
```

---

## Task 15: 训练脚本 — 损失、记录器与指标分支

**Files:**
- Modify: `main_train_predictor_multigpu.py`（`train_step` 与 `val_epoch*`）
- Modify: `tests/test_predictor_import.py`

**Interfaces:**
- Consumes: `models.affinity_loss.affinity_loss`、`PredictorConfig.for_task`
- Produces: `compute_affinity_step_loss(model_out, labels, args, device) -> tuple[Tensor, dict]`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_predictor_import.py`：

```python
def test_compute_affinity_step_loss_is_finite():
    import torch

    from main_train_predictor_multigpu import compute_affinity_step_loss

    class _Args:
        affinity_loss_type = "huber"
        affinity_nll_weight = 0.05
        affinity_rank_weight = 0.0
        affinity_rank_margin = 0.0
        affinity_var_reg_weight = 0.0
        affinity_var_reg_mode = "abs"

    mu = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    log_var = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    labels = torch.tensor([1.5, 1.0, 4.0])

    loss, logs = compute_affinity_step_loss((mu, log_var), labels, _Args(), torch.device("cpu"))
    assert torch.isfinite(loss)
    assert "total" in logs
    loss.backward()
    assert mu.grad is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predictor_import.py -k affinity_step -v`
Expected: FAIL — `ImportError: cannot import name 'compute_affinity_step_loss'`

- [ ] **Step 3: 实现分派函数**

在 `main_train_predictor_multigpu.py` 的 `compute_multihead_loss`（当前 921 行）之后追加：

```python
def compute_affinity_step_loss(model_out, labels, args, device):
    """affinity 通路的单任务损失。model_out = (mu, log_var)，labels = (B,)。"""
    from models.affinity_loss import affinity_loss

    mu, log_var = model_out
    if mu.ndim > 1:
        mu = mu.squeeze(-1)
    if log_var.ndim > 1:
        log_var = log_var.squeeze(-1)

    y = labels.reshape(-1).to(device=device, dtype=torch.float32)
    return affinity_loss(
        mu, log_var, y,
        loss_type=str(getattr(args, "affinity_loss_type", "huber")),
        nll_weight=float(getattr(args, "affinity_nll_weight", 0.05)),
        rank_weight=float(getattr(args, "affinity_rank_weight", 0.0)),
        rank_margin=float(getattr(args, "affinity_rank_margin", 0.0)),
        var_reg_weight=float(getattr(args, "affinity_var_reg_weight", 0.0)),
        var_reg_mode=str(getattr(args, "affinity_var_reg_mode", "abs")),
    )
```

- [ ] **Step 4: 在 `train_step` 里分支**

在 `main_train_predictor_multigpu.py:1237-1258` 处，把 forward + 解包改为：

```python
            with amp.autocast('cuda'):
                if str(getattr(args, "task", "kinetics")) == "affinity":
                    mu, log_var = model(
                        seqs_mb, smiles_mb,
                        prot_struct=p_mb, lig_struct=l_mb,
                        struct_avail_mask=struct_mb,
                        phys_feat=phys_mb, phys_mask=phys_m_mb, phys_quality=phys_q_mb,
                        source_ids=source_mb,
                        use_mask=use_mask,
                    )
                    loss, loss_logs = compute_affinity_step_loss(
                        (mu, log_var), labels_mb, args, device
                    )
                    loss = loss / max(int(getattr(args, "iters_to_accumulate", 1)), 1)
                else:
                    # ... 原有 1239-1500 逐行保留，包在这个 else 里 ...
```

`train_step` 的返回值与日志结构保持 `(loss, loss_detail)` 形状不变，affinity 的 `loss_detail` 直接复用 `loss_logs`（补齐 `"total"` 键）。

`PredictionRecorder` 的构造（当前 1554 行）改：

```python
    n_out = 2 if str(getattr(args, "task", "kinetics")) == "affinity" else 3
    recorder = PredictionRecorder(n_samples, output_dim=n_out)
```

并在更新处对 affinity 传入 `torch.stack([mu, log_var], dim=-1)`。

- [ ] **Step 5: 加 affinity 验证指标**

新增模块级函数：

```python
def _compute_affinity_metrics(y_true, y_pred, sigma=None):
    """亲和力任务指标：MAE / RMSE / Pearson / Spearman / R2，可选 NLL。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    metrics = {"n": int(y_true.size)}
    if y_true.size == 0:
        metrics.update({"mae": float("nan"), "rmse": float("nan"),
                        "pearson": float("nan"), "spearman": float("nan"),
                        "r2": float("nan")})
        return metrics

    metrics["mae"] = float(np.abs(y_true - y_pred).mean())
    metrics["rmse"] = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    metrics["pearson"] = float(_pearson_r_safe(y_true, y_pred))
    metrics["spearman"] = float(_spearman_r_safe(y_true, y_pred))
    denominator = float(((y_true - y_true.mean()) ** 2).sum())
    metrics["r2"] = float(1.0 - ((y_true - y_pred) ** 2).sum() / denominator) \
        if denominator > 1e-12 else float("nan")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)[valid]
        metrics["nll"] = float(
            0.5 * (np.log(2 * np.pi * np.maximum(sigma, 1e-8))
                   + (y_true - y_pred) ** 2 / np.maximum(sigma, 1e-8)).mean()
        )
    return metrics
```

（`_pearson_r_safe` / `_spearman_r_safe` 已在 `main_train_predictor_multigpu.py:263-282` 存在，直接复用。）

在 `val_epoch`（1666 行起）与 `val_epoch_with_ema`（1682 行起）的指标计算处按 `args.task` 分派到 `_compute_affinity_metrics`，跳过 `_add_macro_metrics`。

- [ ] **Step 6: 运行测试与 kinetics 回归**

Run: `python -m pytest tests/ -v`
Expected: 全绿

Run: `python -c "import ast,sys; ast.parse(open('main_train_predictor_multigpu.py',encoding='utf-8').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 7: 提交**

```bash
git add main_train_predictor_multigpu.py tests/test_predictor_import.py
git commit -m "feat(train): route loss, recorder and metrics by task

Adds compute_affinity_step_loss and affinity regression metrics
(MAE/RMSE/Pearson/Spearman/R2/NLL) alongside the existing three-task path."
```

---

## Task 16: 梯度累积与 k8s 友好的环境变量化

**Files:**
- Modify: `main_train_predictor_multigpu.py`（`train_step` 的 `no_sync` 判定与 `optimizer.step()` 门控、`train_one_stage` 的 checkpoint/日志）
- Modify: `main_train_predictor_multigpu.py:2446-2470`（DDP 初始化处加环境变量说明与 `broadcast_buffers`）
- Modify: `tests/test_predictor_import.py`

**Interfaces:**
- Consumes: `args.iters_to_accumulate`
- Produces: `should_step(step_index: int, iters_to_accumulate: int, steps_in_epoch: int) -> bool`、`should_sync_grad(step_index: int, iters_to_accumulate: int, steps_in_epoch: int, is_ddp: bool) -> bool`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_predictor_import.py`：

```python
def test_should_step_without_accumulation():
    from main_train_predictor_multigpu import should_step

    assert all(should_step(i, 1, 5) for i in range(5))


def test_should_step_with_accumulation():
    from main_train_predictor_multigpu import should_step

    # 累积 3 步：第 2、5 步（0-based）才真正 step
    flags = [should_step(i, 3, 7) for i in range(7)]
    assert flags == [False, False, True, False, False, True, False]


def test_should_step_flushes_last_partial_window():
    from main_train_predictor_multigpu import should_step

    # 7 步、累积 3：最后一步必须 flush，否则梯度被丢弃
    assert should_step(6, 3, 7) is True


def test_should_sync_grad_is_false_inside_window_when_ddp():
    from main_train_predictor_multigpu import should_sync_grad

    assert should_sync_grad(0, 3, 7, is_ddp=True) is False
    assert should_sync_grad(2, 3, 7, is_ddp=True) is True


def test_should_sync_grad_is_always_true_when_not_ddp():
    from main_train_predictor_multigpu import should_sync_grad

    assert should_sync_grad(0, 3, 7, is_ddp=False) is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predictor_import.py -k "should_step or should_sync" -v`
Expected: FAIL — `ImportError: cannot import name 'should_step'`

- [ ] **Step 3: 实现两个纯函数**

在 `main_train_predictor_multigpu.py` 的 `train_step` 之前追加：

```python
def should_step(step_index: int, iters_to_accumulate: int, steps_in_epoch: int) -> bool:
    """是否在本步执行 optimizer.step()。

    窗口末尾与 epoch 最后一步都要 flush，否则尾部梯度被静默丢弃。
    """
    accum = max(int(iters_to_accumulate), 1)
    if accum == 1:
        return True
    is_window_end = ((step_index + 1) % accum) == 0
    is_last_step = (step_index + 1) >= int(steps_in_epoch)
    return bool(is_window_end or is_last_step)


def should_sync_grad(step_index: int, iters_to_accumulate: int,
                     steps_in_epoch: int, is_ddp: bool) -> bool:
    """DDP 下在累积窗口内部用 no_sync() 跳过梯度同步。"""
    if not is_ddp:
        return True
    return should_step(step_index, iters_to_accumulate, steps_in_epoch)
```

- [ ] **Step 4: 接入 `train_step`**

把 `main_train_predictor_multigpu.py:1233-1234` 的：

```python
        is_ddp = dist.is_available() and dist.is_initialized()
        sync_ctx = (model.no_sync() if (is_ddp and (t + 1) < steps) else nullcontext())
```

改为：

```python
        is_ddp = dist.is_available() and dist.is_initialized()
        do_sync = should_sync_grad(t, args.iters_to_accumulate, steps, is_ddp=is_ddp)
        sync_ctx = nullcontext() if do_sync else model.no_sync()
```

把 `main_train_predictor_multigpu.py:1496-1510` 的 `scaler.scale(loss).backward()` / `scaler.step(optimizer)` 段包进门控：

```python
            scaler.scale(loss).backward()

        if not should_step(t, args.iters_to_accumulate, steps):
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

- [ ] **Step 5: 给 DDP 构造加 `broadcast_buffers=False`**

把 `main_train_predictor_multigpu.py:2698-2704` 的 `DDP(...)` 调用加一个参数（6B 骨干每卡一份，广播 buffer 无意义且拖慢启动）：

```python
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
            static_graph=(not find_unused) and (not has_cons),
            broadcast_buffers=False,
        )
```

并在 `is_main_process()` 下打印可训练参数分组，便于诊断 DDP 未使用参数问题：

```python
        if is_main_process():
            real_model = getattr(model, "module", model)
            groups = {"lora": 0, "proj": 0, "interactions": 0, "head": 0, "other": 0}
            for name, param in real_model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_" in name:
                    groups["lora"] += param.numel()
                elif "proj" in name:
                    groups["proj"] += param.numel()
                elif "interaction" in name:
                    groups["interactions"] += param.numel()
                elif "multi_head" in name:
                    groups["head"] += param.numel()
                else:
                    groups["other"] += param.numel()
            total = sum(groups.values())
            print(f"[DDP] trainable params total={total:,} breakdown={groups}")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_predictor_import.py -v`
Expected: PASS（13 passed）

- [ ] **Step 7: 提交**

```bash
git add main_train_predictor_multigpu.py tests/test_predictor_import.py
git commit -m "feat(train): add gradient accumulation and DDP tuning for large frozen backbones

Flushes the final partial accumulation window, skips gradient sync inside
windows, and disables buffer broadcast for the per-rank ESMC-6B copy."
```

---

## Task 17: `main_infer_predictor.py` 的 affinity 推理

**Files:**
- Modify: `main_infer_predictor.py:295-330`
- Modify: `tests/test_predictor_import.py`

**Interfaces:**
- Consumes: `KineticsPredictor.forward` 的 affinity 返回
- Produces: `format_affinity_prediction(mu: float, log_var: float) -> dict`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_predictor_import.py`：

```python
def test_format_affinity_prediction_round_trips_sigma():
    import math

    from main_infer_predictor import format_affinity_prediction

    out = format_affinity_prediction(mu=7.5, log_var=math.log(0.25))
    assert abs(out["pred_pKaffinity"] - 7.5) < 1e-6
    assert abs(out["sigma"] - 0.5) < 1e-6
    assert abs(out["sigma2"] - 0.25) < 1e-6


def test_format_affinity_prediction_clamps_negative_variance():
    from main_infer_predictor import format_affinity_prediction

    out = format_affinity_prediction(mu=5.0, log_var=50.0)
    assert out["sigma"] > 0.0
    assert out["sigma2"] > 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predictor_import.py -k format_affinity -v`
Expected: FAIL — `ImportError: cannot import name 'format_affinity_prediction'`

- [ ] **Step 3: 实现**

在 `main_infer_predictor.py` 模块级追加：

```python
def format_affinity_prediction(mu: float, log_var: float) -> dict:
    """把 (mu, log_sigma2) 转成可写 CSV 的字段。sigma2 下限防止 log_var 溢出。"""
    import math

    log_var = float(min(max(float(log_var), -20.0), 20.0))
    sigma2 = math.exp(log_var)
    return {
        "pred_pKaffinity": float(mu),
        "sigma": math.sqrt(sigma2),
        "sigma2": sigma2,
    }
```

在 `main_infer_predictor.py:303-305` 的三路解包处按 `args.task` 分支：

```python
                if str(getattr(args, "task", "kinetics")) == "affinity":
                    mu, log_var = out
                    preds_list.append(
                        torch.stack([mu.reshape(-1), log_var.reshape(-1)], dim=-1).cpu().numpy()
                    )
                else:
                    pred_kcat, pred_km, _pred_ratio = out[:3]
                    pred_ratio = pred_kcat - pred_km + 3.0
                    preds_list.append(
                        torch.stack([pred_kcat, pred_km, pred_ratio], dim=-1).cpu().numpy()
                    )
```

并把输出行构造（`main_infer_predictor.py:311-328`）按 task 分派：

```python
            if str(getattr(args, "task", "kinetics")) == "affinity":
                mu_mean = preds_mean[:, 0]
                log_var_mean = preds_mean[:, 1]
                fields = format_affinity_prediction(float(mu_mean[0]), float(log_var_mean[0]))
                row = {"pdb_id": pdb_id, **fields}
            else:
                # ... 原有 311-328 逐行保留 ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_predictor_import.py -v`
Expected: PASS（15 passed）

- [ ] **Step 5: 提交**

```bash
git add main_infer_predictor.py tests/test_predictor_import.py
git commit -m "feat(infer): add single-head affinity inference path"
```

---

## Task 18: k8s 启动脚本

**Files:**
- Create: `bin/train/kineager_k8s.sh`

**Interfaces:**
- Consumes: `main_train_predictor_multigpu.py` 的 CLI
- Produces: 一个 `exec` 到 `torch.distributed.run` 的启动脚本

- [ ] **Step 1: 写脚本**

创建 `bin/train/kineager_k8s.sh`：

```bash
#!/usr/bin/env bash
# KinEAGER 亲和力训练的 k8s 启动入口。
#
# 与 xtrimofold3 的 bin/train/run_k8s_*.sh 保持同一套 rendezvous 契约：
# 每 pod 一张卡（NPROC_PER_NODE=1），由 k8s 编排注入 NNODES / NODE_RANK /
# MASTER_ADDR，本脚本负责校验并 exec 到 torch.distributed.run。
#
# 训练超参（-epochs / -batch_size / --split_* 等）一律由调用方通过 "$@" 追加。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

for integer_var in NNODES NODE_RANK NPROC_PER_NODE MASTER_PORT; do
    if ! [[ "${!integer_var}" =~ ^[0-9]+$ ]]; then
        echo "${integer_var} must be a non-negative integer (got ${!integer_var})" >&2
        exit 1
    fi
done
if [ "${NNODES}" -lt 1 ]; then
    echo "NNODES (${NNODES}) must be at least 1" >&2
    exit 1
fi
if [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "NODE_RANK (${NODE_RANK}) must be in [0, $((NNODES - 1))]" >&2
    exit 1
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must be at least 1" >&2
    exit 1
fi

if [ "${NNODES}" -gt 1 ]; then
    if [ -z "${MASTER_ADDR:-}" ]; then
        echo "MASTER_ADDR must be set when NNODES > 1" >&2
        exit 1
    fi
    master_addr="${MASTER_ADDR}"
else
    master_addr="${MASTER_ADDR:-127.0.0.1}"
fi

export TASK="${TASK:-affinity}"
export TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV must point to the PDBbind affinity CSV}"
export KINEAGER_ESMC_PATH="${KINEAGER_ESMC_PATH:?KINEAGER_ESMC_PATH must point to a local ESMC-6B directory}"
export KINEAGER_MOLFORMER_PATH="${KINEAGER_MOLFORMER_PATH:?KINEAGER_MOLFORMER_PATH must point to a local MoLFormer snapshot}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[kineager] task=${TASK} nnodes=${NNODES} node_rank=${NODE_RANK} nproc=${NPROC_PER_NODE} master=${master_addr}:${MASTER_PORT}"

exec python3 -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${master_addr}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT}/main_train_predictor_multigpu.py" \
        --task "${TASK}" \
        -csv "${TRAIN_CSV}" \
        -esm "${KINEAGER_ESMC_PATH}" \
        -molt5 "${KINEAGER_MOLFORMER_PATH}" \
        --lora_layer_policy "${LORA_LAYER_POLICY:-last_k}" \
        --lora_last_k "${LORA_LAST_K:-30}" \
        --iters_to_accumulate "${ITERS_TO_ACCUMULATE:-1}" \
        "$@"
```

- [ ] **Step 2: 赋可执行位并做静态检查**

Run:
```bash
chmod +x bin/train/kineager_k8s.sh
bash -n bin/train/kineager_k8s.sh && echo "syntax ok"
```
Expected: `syntax ok`

- [ ] **Step 3: 校验必填变量缺失时报错清晰**

Run (Windows Git Bash 或 WSL): `TRAIN_CSV= bash bin/train/kineager_k8s.sh 2>&1 | head -3`
Expected: 输出含 `TRAIN_CSV must point to the PDBbind affinity CSV` 且退出码非 0

- [ ] **Step 4: 校验非法拓扑被拒**

Run: `NNODES=2 bash bin/train/kineager_k8s.sh 2>&1 | head -3`
Expected: 输出含 `MASTER_ADDR must be set when NNODES > 1`

- [ ] **Step 5: 提交**

```bash
git add bin/train/kineager_k8s.sh
git commit -m "feat(train): add k8s launch script aligned with xtrimofold3 rendezvous contract"
```

---

## Task 19: k8s PyTorchJob 模板

**Files:**
- Create: `yamls/kubeflow/kineager_affinity_k8s.yaml`

**Interfaces:**
- Consumes: `bin/train/kineager_k8s.sh`
- Produces: 一个 `kubeflow.org/v1 PyTorchJob`，`Master.replicas: 1` + `Worker.replicas: N-1`

- [ ] **Step 1: 写模板**

创建 `yamls/kubeflow/kineager_affinity_k8s.yaml`，结构对齐 `D:\project\xtrimofold3-affinity-plm\yamls\kubeflow\xf3_pl_multitask_k8s.yaml`：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: kineager-affinity
  namespace: default
spec:
  runPolicy:
    cleanPodPolicy: None
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: <训练镜像>
              command: ["/bin/bash", "-lc"]
              args:
                - |
                  set -euo pipefail
                  : "${MASTER_ADDR:?PyTorchJob did not inject MASTER_ADDR}"
                  export NNODES="${PET_NNODES:-2}"
                  export NODE_RANK="${PET_NODE_RANK:-${RANK:?PyTorchJob did not inject RANK}}"
                  export NPROC_PER_NODE=1
                  export MASTER_PORT="${MASTER_PORT:-29500}"
                  export NCCL_SOCKET_IFNAME=eth0
                  export NCCL_SOCKET_FAMILY=AF_INET
                  export TRAIN_CSV="${TRAIN_CSV:?}"
                  export KINEAGER_ESMC_PATH="${KINEAGER_ESMC_PATH:?}"
                  export KINEAGER_MOLFORMER_PATH="${KINEAGER_MOLFORMER_PATH:?}"
                  bash bin/train/kineager_k8s.sh \
                    -epochs "${EPOCHS:-30}" \
                    -batch_size "${BATCH_SIZE:-8}" \
                    --split_mode group_pair \
                    --split_seed 42
              resources:
                limits:
                  nvidia.com/gpu: 1
                  rdma/hca: 1
                  cpu: "16"
                  memory: 200Gi
          nodeSelector:
            gpu.device: a100
          tolerations:
            - key: NBTopic
              operator: Exists
              effect: NoSchedule
    Worker:
      replicas: 1          # NNODES - 1
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: <训练镜像>
              command: ["/bin/bash", "-lc"]
              args:
                - |
                  set -euo pipefail
                  export NNODES="${PET_NNODES:-2}"
                  export NODE_RANK="${PET_NODE_RANK:-${RANK:?PyTorchJob did not inject RANK}}"
                  export NPROC_PER_NODE=1
                  export MASTER_PORT="${MASTER_PORT:-29500}"
                  export NCCL_SOCKET_IFNAME=eth0
                  export NCCL_SOCKET_FAMILY=AF_INET
                  bash bin/train/kineager_k8s.sh \
                    -epochs "${EPOCHS:-30}" \
                    -batch_size "${BATCH_SIZE:-8}" \
                    --split_mode group_pair \
                    --split_seed 42
              resources:
                limits:
                  nvidia.com/gpu: 1
                  rdma/hca: 1
                  cpu: "16"
                  memory: 200Gi
          nodeSelector:
            gpu.device: a100
          tolerations:
            - key: NBTopic
              operator: Exists
              effect: NoSchedule
```

- [ ] **Step 2: 语法校验**

Run: `python -c "import yaml,sys; yaml.safe_load(open('yamls/kubeflow/kineager_affinity_k8s.yaml',encoding='utf-8')); print('yaml ok')"`
Expected: `yaml ok`

- [ ] **Step 3: 提交**

```bash
git add yamls/kubeflow/kineager_affinity_k8s.yaml
git commit -m "feat(k8s): add PyTorchJob template for per-pod single-GPU affinity training"
```

---

## Task 20: `path_config.py`、`requirements.txt` 与文档

**Files:**
- Modify: `path_config.py`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: 无
- Produces: `PATH_PROFILES["k8s"]`、`CATA_ESMC_MODEL_PATH` / `CATA_MOLFORMER_MODEL_PATH` 覆盖

- [ ] **Step 1: 加 k8s profile**

在 `path_config.py` 的 `PATH_PROFILES` 中追加：

```python
    "k8s": {
        "project_root": "/workspace/kineager",
        "runs_dir": "/pfs_beijing/${USER}/kineager_runs",
        "weights_dir": "/pfs_beijing/${USER}/kineager_runs",
        "train_data_dir": "/pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020",
        "train_csv_path": "/pfs_beijing/${USER}/kineager_data/pdbbindv2020_affinity.csv",
        "val_csv_path": "/pfs_beijing/${USER}/kineager_data/pdbbindv2020_affinity_val.csv",
        "esm_model_path": "/pfs_beijing/xinran_2026/models/biohub--ESMC-6B",
        "molt5_model_path": "/pfs_beijing/xinran_2026/models/molformer/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314",
    },
```

在 `ENV_OVERRIDES` 中追加：

```python
    "esm_model_path": "CATA_ESM_MODEL_PATH",
    "molt5_model_path": "CATA_MOLT5_MODEL_PATH",
```

（若同名的 `CATA_ESM_MODEL_PATH` / `CATA_MOLT5_MODEL_PATH` 已存在，则复用现有键，不要重复添加。）

在 `FILESYSTEM_PATH_KEYS` 中把 `esm_model_path` / `molt5_model_path` 纳入（当前它们不在集合里，会被当作字符串而不是 `Path`）。

- [ ] **Step 2: 写测试**

创建或追加 `tests/test_path_config.py`：

```python
import os

import pytest

from path_config import resolve_runtime_paths


def test_k8s_profile_resolves():
    paths = resolve_runtime_paths(profile="k8s")
    assert paths["profile"] == "k8s"
    assert paths["train_csv_path"] is not None
    assert paths["esm_model_path"] is not None
    assert paths["molt5_model_path"] is not None


def test_k8s_profile_backbone_paths_are_path_objects():
    from pathlib import Path

    paths = resolve_runtime_paths(profile="k8s")
    assert isinstance(paths["esm_model_path"], Path)
    assert isinstance(paths["molt5_model_path"], Path)


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("CATA_ESM_MODEL_PATH", "/tmp/esmc-local")
    paths = resolve_runtime_paths(profile="k8s")
    assert str(paths["esm_model_path"]).replace("\\", "/") == "/tmp/esmc-local"


def test_existing_profiles_still_resolve():
    for profile in ("autodl", "windows"):
        assert resolve_runtime_paths(profile=profile)["profile"] == profile
```

- [ ] **Step 3: 更新 `requirements.txt`**

替换为：

```text
torch
transformers
peft
esm
pandas
numpy
scikit-learn
scipy
tqdm
rdkit
sentencepiece
protobuf
wandb
```

**说明：** `peft` 是 `models/predictor.py` 的 LoRA 依赖，原 `requirements.txt` 漏了；`loralib` 从不需要（Task 7 已删除该死 import）；`esm` 提供 ESMC 的 tokenizer。

- [ ] **Step 4: 补 README**

在 `README.md` 的 "Main Multi-task Training" 之后插入一节：

````markdown
## PDBbind v2020 Protein-Ligand Affinity Training

Single-task affinity regression over `-log10(Kd/Ki/IC50)` using ESMC-6B and
MoLFormer with last-30-layer LoRA. The kinetics three-task recipe is unchanged
and still selected by default.

Build the training table (on a host that can read the PDBbind index):

```bash
python tools/build_pdbbind_affinity_csv.py \
  --index-csv /path/to/PDBbind_v2020/pdbbind_2020_feats.csv \
  --out-csv data/pdbbindv2020_affinity.csv \
  --kd-col Kd --ki-col Ki --ic50-col IC50
```

Confirm the LoRA target names against the real backbones before the first run:

```bash
python tools/probe_backbone_modules.py --model "$KINEAGER_ESMC_PATH" --kind esmc \
  --out probe_esmc.json
python tools/probe_backbone_modules.py --model "$KINEAGER_MOLFORMER_PATH" --kind molformer \
  --out probe_molformer.json
```

Train on k8s (one GPU per pod):

```bash
NNODES=8 NODE_RANK=0 MASTER_ADDR=<master-pod-ip> MASTER_PORT=29500 \
TRAIN_CSV=data/pdbbindv2020_affinity.csv \
KINEAGER_ESMC_PATH=/models/ESMC-6B \
KINEAGER_MOLFORMER_PATH=/models/MoLFormer-XL-both-10pct \
bash bin/train/kineager_k8s.sh -epochs 30 -batch_size 8 --split_mode group_pair
```

Do **not** pass `--use_struct_branch` or `--final_contact_npz`: the structure
branch is disabled for this task.
````

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/test_path_config.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: 提交**

```bash
git add path_config.py requirements.txt README.md tests/test_path_config.py
git commit -m "chore: add k8s path profile, fix requirements, document affinity task

requirements.txt was missing peft even though models/predictor.py imported it."
```

---

## Task 21: 全量回归与集群 smoke 验收

**Files:**
- 无新增；产出 `docs/superpowers/plans/smoke-report-2026-10-13.md`（记录实测数据）

- [ ] **Step 1: 本地全量单测**

Run: `python -m pytest tests/ -v`
Expected: 全绿。跳过项仅为需要 `peft` 的 LoRA 注入测试。

- [ ] **Step 2: 确认 kinetics 路径未被改动**

Run: `git diff 0c86c64 --stat -- models/predictor.py main_train_predictor_multigpu.py`
Expected: 改动仅出现在 Task 7/13/14/15/16 记录的位置；人工核对 `active_heads=["kcat","Km","ratio"]` 与三任务损失段**未变**。

Run: `git log --oneline 0c86c64..HEAD -- main_train_predictor_multigpu.py`
Expected: 只有 Task 14 / 15 / 16 三个提交

- [ ] **Step 3: 集群上跑探针并定稿后缀白名单**

```bash
python tools/probe_backbone_modules.py --model "$KINEAGER_ESMC_PATH" --kind esmc --out probe_esmc.json
python tools/probe_backbone_modules.py --model "$KINEAGER_MOLFORMER_PATH" --kind molformer \
  --revision 7b12d946c181a37f6012b9dc3b002275de070314 --out probe_molformer.json
```

核对：ESMC-6B `hidden_size == 2560`、`num_layers == 80`；MoLFormer `hidden_size == 768`、`num_layers == 12`。
把 `leaf_names` 与 `DEFAULT_LORA_SUFFIXES` 求差集；若有差集，用 `--lora_target_suffixes` 传入，**不要改代码默认值**。

- [ ] **Step 4: 集群上跑 2 卡 smoke**

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29517 \
TRAIN_CSV=<小样本子集 csv> \
KINEAGER_ESMC_PATH=/models/ESMC-6B \
KINEAGER_MOLFORMER_PATH=/models/MoLFormer-XL-both-10pct \
ITERS_TO_ACCUMULATE=2 \
bash bin/train/kineager_k8s.sh -epochs 1 -batch_size 2 --task affinity
```

（`MASTER_PORT` 用 29517 而非 29300：xtrimofold3 记录过 29300 被占用。）

- [ ] **Step 5: 记录验收数据**

在 `docs/superpowers/plans/smoke-report-2026-10-13.md` 中记录：

| 项 | 记录内容 |
|---|---|
| LoRA 命中层 | ESMC 实际层号区间与目标数；MoLFormer 实际层号区间与目标数 |
| 可训练参数量 | `[DDP] trainable params total=... breakdown={...}` 的完整输出 |
| 单步耗时 | 秒/步（含与不含梯度累积） |
| 峰值显存 | `torch.cuda.max_memory_allocated()` 或 `nvidia-smi` 采样 |
| 梯度同步 | 两个 rank 上同一个 LoRA 张量的 `allclose` 结果 |
| checkpoint | `predictor_best_ema.pt` 落盘且可被 `strict=True` 载入 |
| bf16 | 无 NaN/Inf 损失；若有，记录具体模块并考虑 `autocast(enabled=False)` |

- [ ] **Step 6: 依据实测决定 `--grad_checkpoint_backbone` 默认值**

若峰值显存超过单卡容量的一半，把默认值从 `off` 改为 `active`，并在 spec 的 §5 D2 与 §11 更新记录。这是一个**实测驱动的决定**，不要凭猜测改。

- [ ] **Step 7: 最终提交**

```bash
git add docs/superpowers/plans/smoke-report-2026-10-13.md
git commit -m "docs: record k8s smoke results for the affinity path"
```

**注意：全部任务完成后不要 `git push`（用户明确要求）。**

---

## Self-Review

**1. Spec 覆盖检查**

| Spec 章节 | 对应任务 |
|---|---|
| §5 D1 last-30 层 LoRA | Task 3、Task 6、Task 14（CLI 默认值） |
| §5 D2 骨干梯度检查点 | Task 6（实现）、Task 21 Step 6（实测定默认值） |
| §5 D3 复用 `MultiHead` | Task 13 Step 3 |
| §5 D4 不构造 kinetics 参数 | Task 13 Step 4/5、Task 13 Step 1 的 `test_affinity_model_does_not_construct_kinetics_parameters` |
| §5 D5 结构分支运行期关闭 | Task 14 Step 5（`use_struct_branch` 透传而不默认打开）、Task 20 Step 4（README 明确禁用） |
| §5 D6 泛化 `kcat_expert_loss` | Task 12 |
| §5 D7 单列 Scaler/Dataset | Task 9、Task 10 |
| §5 D8 DDP 只改启动层 | Task 16、Task 18、Task 19 |
| §6.1 `plm_backbones` | Task 2、3、4、5、6 |
| §6.2 `affinity_loss` | Task 12 |
| §6.3 `data_affinity` | Task 9、Task 10 |
| §6.4 `encoders` | Task 8 |
| §6.5 `predictor` | Task 7、Task 13 |
| §6.6 训练脚本 | Task 14、Task 15、Task 16 |
| §6.7 k8s 启动脚本 | Task 18 |
| §6.8 PyTorchJob yaml | Task 19 |
| §6.9 PDBbind CSV 构建 | Task 11 |
| §6.10 推理 / path_config / requirements / README | Task 17、Task 20 |
| §8 错误处理 | 分散在各 Task 的失败测试中（零命中报错、字段清单、标签列缺失、DDP 参数诊断） |
| §9 测试策略 | 每个 Task 的 Step「运行测试」 |
| §10 P0 探针 | Task 1、Task 21 Step 3 |

**2. 占位符扫描**

- 无 "TBD" / "TODO" / "implement later"。
- `yamls/kubeflow/kineager_affinity_k8s.yaml` 中的 `<训练镜像>` 与 `<master-pod-ip>` 是**部署环境相关的值**，不是未完成的设计项；镜像名无法从代码库推断。
- Task 11 Step 1 的「确认真实列名」是一个**必须执行的验证步骤**，而不是遗漏：PDBbind 索引 CSV 的列名在本地无法确认，脚本把列名做成 CLI 参数正是为此。

**3. 类型与命名一致性**

- `AffinityScaler.transform` 返回 `(N,)`，`AffinityDataset` 按下标取标量 → 一致。
- `AffinityDataset.__getitem__` 返回 3 元组 → `unpack_batch` 的 `len(batch) == 3` 分支 → 一致。
- `affinity_loss(mu, log_var, y, ...) -> (Tensor, dict)`；`compute_affinity_step_loss(model_out, labels, args, device) -> (Tensor, dict)`，其中 `model_out = (mu, log_var)` → 一致。
- `KineticsPredictor.forward` affinity 返回 `(mu, log_sigma2)`，每项 `(B,)`；`compute_affinity_step_loss` 对 `ndim > 1` 做 `squeeze(-1)` 兜底 → 一致。
- `BackboneSpec.prefer_hidden_states` 由 `load_*_backbone` 设置，被 `_BackboneEncoder.encode_tokens` 读取 → 一致。
- `inject_lora(...) -> (peft_model, targets, layers)`，`layers` 供 `set_backbone_gradient_checkpointing(mode="active", layers=...)` 使用 → 一致。
- `should_step` / `should_sync_grad` 的 `steps_in_epoch` 语义统一为「当前 epoch 的批次数」 → 一致。
- 新 CLI 参数名在 Task 14 定义、Task 15/16/18 引用：`--task`、`--affinity_label_col`、`--affinity_loss_type`、`--affinity_nll_weight`、`--affinity_rank_weight`、`--affinity_rank_margin`、`--affinity_var_reg_weight`、`--affinity_var_reg_mode`、`--lora_layer_policy`、`--lora_last_k`、`--lora_explicit_layers`、`--lora_target_suffixes`、`--plm_max_len_prot`、`--plm_max_len_lig`、`--grad_checkpoint_backbone`、`--iters_to_accumulate`、`--num_workers` → 一致。

**4. 未在计划内、需要执行者注意的两点**

- `--lora_target_suffixes` 与 `--plm_max_len_*` 在 Task 14 里只解析成参数，**接入 `_build_backbones_and_lora()`（Task 13b）的传递路径需要在实现时一并补上**；两者在 Task 13b 的 `inject_lora` / `load_*_backbone` 调用点各多传一个实参。
- `models/predictor.py` 的 `_build_backbones_and_lora`（Task 13b）没有独立测试，因为它需要真实权重；它的正确性由 Task 21 的集群 smoke 覆盖。这是**有意的取舍**：本地无法加载 6B 权重，为它编造 mock 只会测试 mock。

