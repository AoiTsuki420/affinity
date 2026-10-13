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
