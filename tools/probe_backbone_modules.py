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
