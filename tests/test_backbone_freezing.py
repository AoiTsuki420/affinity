"""freeze_backbone 命名匹配的回归测试（集群 smoke 2026-10-14 实测事故）。

本分支的 KineticsPredictor 把骨干模块同时注册为 protein.backbone 与 esm/molt5
别名（models/predictor.py:573-574），named_parameters() 对共享模块只按其中一个
路径枚举——实际枚举出的是 ``esm.`` / ``molt5.`` 前缀。旧实现只认
``protein.backbone`` / ``ligand.backbone``，一个参数都匹配不到：6.5B 骨干全部
落进 else 分支被解冻（实测 freeze 后 requires_grad 总数 ≈ 6.5B），LoRA 设计
名存实亡，训练显存与科学结论同时失控。
"""
import pytest
import torch
import torch.nn as nn

from main_train_predictor_multigpu import freeze_backbone


def _build_model(prefix: str) -> nn.Module:
    """最小模型：骨干参数挂在 prefix 路径下，另有 head/proj 两个非骨干层。"""
    model = nn.Module()
    holder = model
    for part in prefix.split("."):
        setattr(holder, part, nn.Module())
        holder = getattr(holder, part)
    holder.weight = nn.Parameter(torch.ones(4, 4))
    holder.lora_A_default = nn.Parameter(torch.ones(2, 4))
    holder.lora_B_default = nn.Parameter(torch.zeros(4, 2))
    model.head = nn.Linear(4, 1)
    model.proj = nn.Linear(4, 4)
    return model


@pytest.mark.parametrize("prefix", ["esm", "module.esm", "protein.backbone"])
def test_freeze_backbone_keeps_lora_and_freezes_rest_of_backbone(prefix):
    """两代命名（esm. 别名 / 旧 protein.backbone）与 DDP 的 module. 前缀都必须：
    骨干冻结、LoRA 保留、非骨干可训练。"""
    model = _build_model(prefix)
    freeze_backbone(model)
    params = dict(model.named_parameters())

    assert params[f"{prefix}.weight"].requires_grad is False, "骨干权重必须被冻结"
    assert params[f"{prefix}.lora_A_default"].requires_grad is True, "LoRA 必须保持可训练"
    assert params[f"{prefix}.lora_B_default"].requires_grad is True
    assert params["head.weight"].requires_grad is True, "非骨干参数必须保持可训练"
    assert params["proj.weight"].requires_grad is True


def test_freeze_backbone_raises_when_stack_params_stay_trainable():
    """命名漂移时宁可显式报错，也不能静默把整个堆栈留给全量训练。"""
    model = _build_model("esm")
    model.renamed_tower = nn.Module()
    model.renamed_tower.inner = nn.Module()
    model.renamed_tower.inner.transformer = nn.Module()
    model.renamed_tower.inner.transformer.weight = nn.Parameter(torch.ones(4, 4))

    with pytest.raises(AssertionError) as excinfo:
        freeze_backbone(model)
    assert "renamed_tower" in str(excinfo.value)
