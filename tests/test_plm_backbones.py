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
