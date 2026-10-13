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
    leaf_lines = [line for line in message.splitlines() if "Linear 末段名" in line]
    assert len(leaf_lines) == 1, f"诊断信息缺少唯一的末段名清单行:\n{message}"
    assert "my_custom_proj" in leaf_lines[0]


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


from models.plm_backbones import DEFAULT_LORA_SUFFIXES


def test_default_suffixes_cover_spec_required_leaf_forms():
    """spec §6.1 要求默认白名单覆盖这些末段名；集合包含式断言让任何一条被删都立刻变红。"""
    required = {"query", "key", "value", "wi_0", "wi_1", "wo", "fc1", "fc2"}
    assert required <= set(DEFAULT_LORA_SUFFIXES)


def test_discover_matches_spec_required_bare_leaf_forms():
    """上面那 8 条必须真的能被 endswith 命中，而不只是躺在元组里。"""
    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.query = nn.Linear(8, 8)
            self.attn.key = nn.Linear(8, 8)
            self.attn.value = nn.Linear(8, 8)
            self.mlp = nn.Module()
            self.mlp.wi_0 = nn.Linear(8, 32)
            self.mlp.wi_1 = nn.Linear(8, 32)
            self.mlp.wo = nn.Linear(32, 8)
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(2)])

    targets = discover_lora_targets(_Model(), layers=[0, 1])
    assert targets == sorted(
        f"layers.{i}.{leaf}"
        for i in range(2)
        for leaf in (
            "attn.query", "attn.key", "attn.value",
            "mlp.wi_0", "mlp.wi_1", "mlp.wo",
            "fc1", "fc2",
        )
    )


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


from models.plm_backbones import load_protein_backbone


def test_load_esmc_without_esm_package_raises_install_hint(monkeypatch):
    """spec §8 错误表：`esm` 包缺失时，ESMC 通路必须在加载期抛出带安装提示的 ImportError。

    只桩掉权重加载那一步（本机无网络也无 ESMC-6B 权重），使流程真正走到 tokenizer 的
    `from esm.models.esmc import EsmcTokenizer`，避免测试因为更早的失败而假通过。
    """
    import transformers

    loaded = []

    def _fake_from_pretrained(cls, name, **kwargs):
        loaded.append((name, kwargs))
        return object()

    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    with pytest.raises(ImportError) as excinfo:
        load_protein_backbone("biohub/ESMC-6B", max_length=1024)

    message = str(excinfo.value)
    assert "esm" in message, message
    assert "pip install esm" in message, message

    # 证明这个 ImportError 确实源自缺失的 esm 包，而不是任何更早的 import 失败。
    assert loaded, "模型加载步骤没有被执行，流程没走到 tokenizer 导入"
    cause = excinfo.value.__cause__
    assert isinstance(cause, ImportError), cause
    assert getattr(cause, "name", None) == "esm", cause


import warnings


def test_build_token_mask_ignores_special_ids_in_padding():
    """padding 里的 pad_token_id 不能计入特殊 token 诊断。

    真实批次会把 50..400 长度的序列 pad 到 400，padding 位全是 pad_token_id；
    整行统计的中位数会是几百，于是每个正常批次都误报"特殊 token 集合可能不完整"。
    simplefilter("error") 让任何这类误报直接变成测试失败。
    """

    class _PaddedTokenizer:
        all_special_ids = [0, 2]
        pad_token_id = 0

    inputs = {
        "input_ids": torch.tensor([
            [0, 5, 6, 7, 2, 0, 0, 0],
            [0, 8, 2, 0, 0, 0, 0, 0],
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
        ]),
    }
    # 整行统计得到 [5, 7]（中位数 6，会误报）；只看非 padding 位则是 [2, 2]（中位数 2）。
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = build_token_mask(_PaddedTokenizer(), inputs)

    assert mask.tolist() == [
        [False, True, True, True, False, False, False, False],
        [False, True, False, False, False, False, False, False],
    ]


from models import plm_backbones


def test_load_resolves_config_when_model_has_no_config(monkeypatch):
    """trust_remote_code=True 时 hub shim 可能返回 esm 自己的对象（普通 nn.Module，无 .config）。

    此时必须像 tools/probe_backbone_modules.py 那样回退到 AutoConfig.from_pretrained，
    而不是让 model.config 抛 AttributeError。
    """
    import transformers

    class _NoConfig(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([torch.nn.Identity()] * 5)

    def _fake_loader(name, **kwargs):
        return _NoConfig(), object(), True

    config_calls = []

    def _fake_from_pretrained(cls, name, **kwargs):
        config_calls.append((name, kwargs))
        return SimpleNamespace(embedding_dim=2560)  # 不带层数字段，逼出结构回退

    monkeypatch.setitem(plm_backbones._LOADERS, KIND_ESMC, _fake_loader)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(_fake_from_pretrained)
    )

    model, _tokenizer, spec = plm_backbones.load_protein_backbone(
        "biohub/ESMC-6B", max_length=512
    )

    assert isinstance(model, _NoConfig)
    assert config_calls == [("biohub/ESMC-6B", {"trust_remote_code": True})]
    assert spec.hidden_size == 2560  # 来自 AutoConfig 兜底
    assert spec.num_layers == 5  # 来自模型结构的回退分支
    assert spec.kind == KIND_ESMC
    assert spec.max_length == 512
    assert spec.prefer_hidden_states is True


def test_load_threads_revision_and_trust_remote_code_to_tokenizer(monkeypatch):
    """MoLFormer 权重被钉在固定 revision（spec §4 约束 4），tokenizer 必须同 revision。

    同时校验 tokenizer 用的是调用方解析出的 trust_remote_code，而不是硬编码 True。
    """
    import transformers

    revision = "7b12d946c181a37f6012b9dc3b002275de070314"
    tokenizer_calls = []
    model_calls = []

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            tokenizer_calls.append((name, kwargs))
            return cls()

    class _FakeModel:
        config = SimpleNamespace(hidden_size=768, num_hidden_layers=12)

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append((name, kwargs))
            return cls()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", _FakeTokenizer.from_pretrained
    )
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", _FakeModel.from_pretrained)

    name = "ibm-research/MoLFormer-XL-both-10pct"
    _model, _tokenizer, spec = plm_backbones.load_ligand_backbone(name, revision=revision)

    # 默认 trust_remote_code=True（见 _TRUST_REMOTE_CODE_DEFAULT），权重与 tokenizer 同参同 revision。
    assert tokenizer_calls == [(name, {"trust_remote_code": True, "revision": revision})]
    assert model_calls == tokenizer_calls
    assert spec.kind == KIND_MOLFORMER
    assert spec.trust_remote_code is True

    # 调用方显式关掉 trust_remote_code 时，tokenizer 不能偷偷改回 True。
    tokenizer_calls.clear()
    model_calls.clear()
    plm_backbones.load_ligand_backbone(name, trust_remote_code=False, revision=revision)

    assert tokenizer_calls == [(name, {"revision": revision})]
    assert model_calls == tokenizer_calls
