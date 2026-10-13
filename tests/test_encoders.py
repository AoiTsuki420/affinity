from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import models.encoders as encoders
from models.encoders import LigandEncoder, ProteinEncoder
from models.plm_backbones import (
    KIND_ESM2,
    KIND_ESMC,
    KIND_MOLFORMER,
    KIND_T5,
    BackboneSpec,
)


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


class _TrainableFakeBackbone(_FakeBackbone):
    """带可训练参数的 fake：让「有没有建图」在输出 requires_grad 上可见。

    _FakeBackbone 自身没有参数，光看它无法区分 set_grad_enabled 的真假。
    """

    def __init__(self, hidden=16, layers=3):
        super().__init__(hidden=hidden, layers=layers)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        out.last_hidden_state = out.last_hidden_state * self.scale
        return out


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
    # R25：ESM-2 保持旧的 attention_mask 语义，<cls>/<eos> 位**保留**（原 brief 在此
    # 期望剥离，被 R25 裁定推翻：剥离会改变 kinetics 池化）。ESMC 的剥离见
    # test_strip_special_follows_backbone_kind。
    assert mask.tolist() == [[True, True, True, True]]


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


def test_encoder_moves_backbone_to_device():
    """R28-fix：骨干必须和 proj 一样被搬到 device。

    旧实现是 backbone_obj.to(device)，重写时漏了这一步：单独构造
    ProteinEncoder(esm_model=..., device="cuda") 会把骨干留在 CPU，第一次 forward 才炸。
    这里用 meta device 做 CPU-only 的等价断言。
    """
    encoder = ProteinEncoder(
        d_model=8, device="meta",
        backbone_obj=_TrainableFakeBackbone(hidden=16),
        tokenizer_obj=_FakeTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_ESM2, 16, 3),
    )
    assert encoder.proj.weight.device.type == "meta"
    assert next(encoder.backbone.parameters()).device.type == "meta"


def test_encoder_grad_gate_narrows_but_never_overrides_ambient_context():
    """R28：编码器只能收窄环境梯度上下文，不能把调用方的 no_grad 顶掉。

    main_infer_ensemble.py:56-62 在注入 LoRA 后从不设 train_backbone=True，所以 gate
    会是 True；而它的前向跑在 models/moe_kcat.py:103 的 @torch.no_grad() 里
    （:111 调 encode_tokens）。若这里用 torch.set_grad_enabled(True)，650M 骨干会在
    ensemble 路由时重新建图：数值一样，但峰值显存与延迟退化，且静默推翻了调用方意图。
    """
    encoder = ProteinEncoder(
        d_model=8, device="cpu",
        backbone_obj=_TrainableFakeBackbone(hidden=16),
        tokenizer_obj=_FakeTokenizer(),
        train_backbone=True,
        spec=_spec(KIND_ESM2, 16, 3),
    )

    # 断言必须落在 encode_tokens 的返回值上，不能落在 forward 的输出上：
    # forward 里的 self.proj(...) 跑在 set_grad_enabled 块**之外**，处于环境上下文，
    # 会把 no_grad 里的图剥掉，于是 forward 输出永远 requires_grad=False —— 那样断言
    # 是空的。encode_tokens 才既是被顶掉的那一层，也是 models/moe_kcat.py:111
    # 在 ensemble 路由里真正调用的那一层。
    # 反向也要钉住：环境允许 + 参数可训练时必须建图，否则训练路径会被"一律关梯度"骗过。
    outside, _ = encoder.encode_tokens(["MKTAYIAK"])
    assert outside.requires_grad is True

    with torch.no_grad():
        inside, _ = encoder.encode_tokens(["MKTAYIAK"])
    assert inside.requires_grad is False


def test_strip_special_follows_backbone_kind():
    """R25：只有 ESMC 剔除边界特殊 token，ESM-2 / T5 保持旧的 attention_mask 语义。

    fake tokenizer 的 all_special_ids=[0,2] 正好落在 [0,5,6,2] 的两端，两个方向都要断言：
    只测 ESMC 会漏掉「ESM-2 被误改成剥离」这个回归（kinetics 的 masked_mean 池化会变）。
    """
    tokenizer = _FakeTokenizer()

    def _mask(cls, kind):
        encoder = cls(
            d_model=8, device="cpu",
            backbone_obj=_FakeBackbone(hidden=16),
            tokenizer_obj=tokenizer,
            train_backbone=True,
            spec=_spec(kind, 16, 3),
        )
        _, mask = encoder(["MKTAYIAK"])
        return mask.tolist()

    assert _mask(ProteinEncoder, KIND_ESMC) == [[False, True, True, False]]
    assert _mask(ProteinEncoder, KIND_ESM2) == [[True, True, True, True]]
    assert _mask(LigandEncoder, KIND_T5) == [[True, True, True, True]]


def test_protein_encoder_legacy_esm_model_alias(monkeypatch):
    """R24：esm_model= 是 backbone_name= 的 legacy 别名。

    fake 骨干经 monkeypatch 掉的 loader 注入（与 main_train_kcat_expert.py /
    main_infer_ensemble.py 一样只传名字、不传 backbone_obj），这样既能观察到别名真的
    解析成同一个 backbone_name 并原样传给 loader，又不下载任何权重。
    """
    calls = []
    spec = _spec(KIND_ESM2, 16, 3)
    backbone = _FakeBackbone(hidden=16)
    tokenizer = _FakeTokenizer()

    def _recording_loader(name, *, max_length=None):
        calls.append((name, max_length))
        return backbone, tokenizer, spec

    monkeypatch.setattr(encoders, "load_protein_backbone", _recording_loader)

    legacy = ProteinEncoder(esm_model="legacy-esm", d_model=32, device="cpu",
                            train_backbone=True)
    current = ProteinEncoder(backbone_name="legacy-esm", d_model=32, device="cpu",
                             train_backbone=True)

    assert calls == [("legacy-esm", None), ("legacy-esm", None)]
    assert legacy.backbone is backbone and legacy.spec is spec
    assert legacy.max_length == current.max_length == spec.max_length

    # 两条构造路径产出等价的编码器。proj 是随机初始化的，先对齐权重再逐位比较，
    # 这样比较的是构造路径（spec / max_length / mask / 输出形状）而不是 RNG。
    current.proj.load_state_dict(legacy.proj.state_dict())
    legacy_tokens, legacy_mask = legacy(["MKTAYIAK"])
    current_tokens, current_mask = current(["MKTAYIAK"])
    assert torch.equal(legacy_mask, current_mask)
    torch.testing.assert_close(legacy_tokens, current_tokens)


def test_ligand_encoder_legacy_molt5_model_alias(monkeypatch):
    """R24：molt5_model= 是 backbone_name= 的 legacy 别名。"""
    calls = []
    spec = _spec(KIND_T5, 16, 3)
    backbone = _FakeBackbone(hidden=16)
    tokenizer = _FakeTokenizer()

    def _recording_loader(name, *, max_length=None):
        calls.append((name, max_length))
        return backbone, tokenizer, spec

    monkeypatch.setattr(encoders, "load_ligand_backbone", _recording_loader)

    legacy = LigandEncoder(molt5_model="legacy-molt5", d_model=32, device="cpu",
                           train_backbone=True)
    current = LigandEncoder(backbone_name="legacy-molt5", d_model=32, device="cpu",
                            train_backbone=True)

    assert calls == [("legacy-molt5", None), ("legacy-molt5", None)]
    assert legacy.backbone is backbone and legacy.spec is spec
    assert legacy.max_length == current.max_length == spec.max_length

    current.proj.load_state_dict(legacy.proj.state_dict())
    legacy_tokens, legacy_mask = legacy(["CCO"])
    current_tokens, current_mask = current(["CCO"])
    assert torch.equal(legacy_mask, current_mask)
    torch.testing.assert_close(legacy_tokens, current_tokens)


def test_protein_encoder_rejects_backbone_name_with_legacy_alias():
    with pytest.raises(ValueError) as excinfo:
        ProteinEncoder(
            backbone_name="new-name", esm_model="old-name",
            d_model=32, device="cpu",
            backbone_obj=_FakeBackbone(hidden=16),
            tokenizer_obj=_FakeTokenizer(),
            train_backbone=True,
            spec=_spec(KIND_ESM2, 16, 3),
        )
    message = str(excinfo.value)
    assert "backbone_name" in message and "esm_model" in message


def test_ligand_encoder_rejects_backbone_name_with_legacy_alias():
    with pytest.raises(ValueError) as excinfo:
        LigandEncoder(
            backbone_name="new-name", molt5_model="old-name",
            d_model=32, device="cpu",
            backbone_obj=_FakeBackbone(hidden=16),
            tokenizer_obj=_FakeTokenizer(),
            train_backbone=True,
            spec=_spec(KIND_T5, 16, 3),
        )
    message = str(excinfo.value)
    assert "backbone_name" in message and "molt5_model" in message
