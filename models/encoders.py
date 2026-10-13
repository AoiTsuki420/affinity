"""蛋白与配体编码器。

骨干的加载、hidden 维度解析、特殊 token 处理全部委托给 models.plm_backbones，
本模块只负责「骨干输出 → 投影到统一的 d_model」以及冻结/LoRA 参数的可见性策略。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .plm_backbones import (
    KIND_ESMC,
    BackboneSpec,
    build_token_mask,
    forward_hidden_states,
    load_ligand_backbone,
    load_protein_backbone,
)


def _resolve_backbone_name(backbone_name, legacy_name, legacy_keyword):
    """把 legacy 关键字（esm_model / molt5_model）折叠进 backbone_name。

    R24：旧的调用点（main_train_kcat_expert.py:112-113、main_infer_ensemble.py:48-49）
    仍然用 esm_model= / molt5_model= 传名字，而设计 spec §12 禁止修改这两个文件，
    所以兼容层放在这里。两个名字都给就是调用方的错误，必须显式报出来，
    不能静默挑一个——静默会让"以为加载了 A、实际加载了 B"变成不可见的 bug。
    """
    if backbone_name is not None and legacy_name is not None:
        raise ValueError(
            f"backbone_name 与 legacy 别名 {legacy_keyword} 不能同时提供："
            f"backbone_name={backbone_name!r}, {legacy_keyword}={legacy_name!r}。"
            f"请只传 backbone_name（新写法），或只传 {legacy_keyword}（legacy 写法）。"
        )
    return backbone_name if backbone_name is not None else legacy_name


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

        # 骨干和 proj 一样必须落到 device：旧实现是 backbone_obj.to(device) /
        # from_pretrained(...).to(device)，漏掉这一步会让 ProteinEncoder(device="cuda")
        # 把骨干留在 CPU，直到第一次 forward 才炸。两条分支共用这一次赋值。
        self.backbone = self.backbone.to(device)

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

        # 梯度开关只能**收窄**环境上下文，绝不能把它打开：main_infer_ensemble.py:56-62
        # 注入 LoRA 后从不设 train_backbone=True，所以 any(...) 为真；而它的前向跑在
        # models/moe_kcat.py:103 的 @torch.no_grad() 里（:111 调 encode_tokens）。
        # 若这里写成 set_grad_enabled(True)，650M 骨干会在 ensemble 路由时重新建图并
        # 一直挂在返回的 tokens 上——数值不变，但峰值显存和延迟退化，且静默推翻了
        # 调用方的意图。
        # main_train_kcat_expert.py 那条路径不受影响：它在 :128 注入 LoRA 之后设了
        # train_backbone=True，旧表达式本来就是 True，LoRA 一直在正常训练——这里
        # 不存在「曾经没训练」的 bug。main kinetics 训练路径与新的 affinity 路径同理，
        # 环境允许且参数可训练时仍然建图。
        enable_grad = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        with torch.set_grad_enabled(enable_grad):
            tokens = forward_hidden_states(
                self.backbone, encoded, output_hidden_states=self.spec.prefer_hidden_states
            )

        # R25：剥离边界特殊 token 只对 ESMC 成立（spec 约束 7：ESMC 的输出保留 2 个
        # 特殊 token，不剔除会污染 masked_mean 池化）。ESM-2 / T5 走的旧代码是从
        # attention_mask 直接取掩码，<cls>/<eos> 是**计入**池化的；这里若无条件剥离，
        # kinetics 的 masked_mean（models/moe_kcat.py:114-115）就会变，等于改动所有
        # kinetics 数值。所以由 kind 决定，而不是常量 True。
        # 也不要把条件放宽到别的 kind：EsmTokenizer.all_special_ids 可能含 <unk>，
        # 剥离未知残基位会丢掉真实信号。
        mask = build_token_mask(
            self.tokenizer, encoded, strip_special=(self.spec.kind == KIND_ESMC)
        )
        return tokens, mask

    def forward(self, inputs):
        tokens, mask = self.encode_tokens(inputs)
        tokens = self.proj(tokens.float())
        return torch.nan_to_num(tokens), mask


class ProteinEncoder(_BackboneEncoder):
    def __init__(self, backbone_name: str | None = None, d_model: int = 768,
                 device: str = "cuda", backbone_obj: nn.Module | None = None,
                 train_backbone: bool = False, tokenizer_obj: Any | None = None,
                 spec: BackboneSpec | None = None, max_length: int | None = None,
                 esm_model: str | None = None):
        """esm_model 是 backbone_name 的 **legacy 别名**（R24：main_train_kcat_expert.py
        与 main_infer_ensemble.py 仍在用它，且那两个文件不允许改）。两者都给会抛
        ValueError，不会静默取其一。
        """
        super().__init__(
            backbone_name=_resolve_backbone_name(backbone_name, esm_model, "esm_model"),
            d_model=d_model, device=device,
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
                 spec: BackboneSpec | None = None, max_length: int | None = None,
                 molt5_model: str | None = None):
        """molt5_model 是 backbone_name 的 **legacy 别名**（R24：main_train_kcat_expert.py
        与 main_infer_ensemble.py 仍在用它，且那两个文件不允许改）。两者都给会抛
        ValueError，不会静默取其一。
        """
        super().__init__(
            backbone_name=_resolve_backbone_name(backbone_name, molt5_model, "molt5_model"),
            d_model=d_model, device=device,
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
