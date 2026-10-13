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
        """真 T5 的 self-attention 子层位于 block.layer.0。"""

        def __init__(self):
            super().__init__()
            self.SelfAttention = _SelfAttn()

    class _FfnLayer(nn.Module):
        """真 T5 的 FFN 子层位于 block.layer.1。

        旧实现 _build_t5_encoder_targets_exact 用的正是
        "layer.1.DenseReluDense.wi_0"，故两个子层必须分开挂载。
        """

        def __init__(self):
            super().__init__()
            self.DenseReluDense = _DenseRelu()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([_BlockLayer(), _FfnLayer()])

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
