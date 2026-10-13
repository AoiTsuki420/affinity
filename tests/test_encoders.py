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
