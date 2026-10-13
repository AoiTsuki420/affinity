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
