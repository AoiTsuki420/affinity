"""R38：KineticsPredictor 在**不注入** encoders 时的真实构造路径。

Task 13 让 KineticsPredictor 重新可以自行加载骨干（models/predictor.py 的
`_build_backbones_and_lora`），但当时的测试文件被逐字节固定，只覆盖了
`protein_encoder=` / `ligand_encoder=` 的注入路径。这里把两个骨干加载器替换成
config 构造的 EsmModel / BertModel（不 from_pretrained、不联网、不读权重），
再走真实构造函数，确保这条路径不能被静默改坏。
"""
import pytest
import torch

from models.plm_backbones import KIND_ESMC, KIND_MOLFORMER, BackboneSpec
from models.predictor import KineticsPredictor, PredictorConfig

HIDDEN = 32
LAYERS = 2
VOCAB = 32


class _FakeTokenizer:
    """只满足 _BackboneEncoder.encode_tokens 与 build_token_mask 的接口。"""

    all_special_ids = ()

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        batch = len(texts)
        return {
            "input_ids": torch.tensor([[2, 5, 6, 3]] * batch),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


def _spec(kind, name):
    return BackboneSpec(name=name, kind=kind, trust_remote_code=False,
                        max_length=16, hidden_size=HIDDEN, num_layers=LAYERS)


def _loader_calls():
    """返回 (protein_loader, ligand_loader, calls)。两个 loader 都记录被调用的名字。"""
    calls = []

    def protein_loader(name, **kwargs):
        from transformers import EsmConfig, EsmModel

        calls.append(("protein", name))
        config = EsmConfig(vocab_size=VOCAB, hidden_size=HIDDEN,
                           num_hidden_layers=LAYERS, num_attention_heads=2,
                           intermediate_size=64,
                           pad_token_id=1, mask_token_id=2)
        return EsmModel(config), _FakeTokenizer(), _spec(KIND_ESMC, name)

    def ligand_loader(name, **kwargs):
        from transformers import BertConfig, BertModel

        calls.append(("ligand", name))
        config = BertConfig(vocab_size=VOCAB, hidden_size=HIDDEN,
                            num_hidden_layers=LAYERS, num_attention_heads=2,
                            intermediate_size=64)
        return BertModel(config), _FakeTokenizer(), _spec(KIND_MOLFORMER, name)

    return protein_loader, ligand_loader, calls


def _small_head(out_dim):
    return {"type": "mlp", "hidden_dim": 16, "depth": 2, "dropout": 0.0,
            "out_dim": out_dim}


def _small_cfg(task):
    return dict(
        d_model=32, num_heads=2, num_interaction_layers=1, num_stages=1,
        num_interaction_layers_per_stage=1, num_token_transformer_layers=0,
        head_hidden=16, head_hidden_dim=16, transformer_ffn_dim=32,
        dropout=0.0, use_attention_pool=False, use_checkpoint=False,
        head_specs={"affinity": _small_head(2)} if task == "affinity" else {
            "kcat": _small_head(1), "Km": _small_head(1), "ratio": _small_head(1),
        },
    )


def _build(task, monkeypatch):
    protein_loader, ligand_loader, calls = _loader_calls()
    monkeypatch.setattr("models.plm_backbones.load_protein_backbone", protein_loader)
    monkeypatch.setattr("models.plm_backbones.load_ligand_backbone", ligand_loader)

    cfg = PredictorConfig.for_task(task, **_small_cfg(task))
    # 不传 protein_encoder= / ligand_encoder=，强制走骨干加载分支。
    model = KineticsPredictor(device="cpu", esm_model="esm-local",
                              molt5_model="molt5-local", cfg=cfg, use_lora=False)
    return model, calls


def test_kinetics_predictor_constructs_without_injected_encoders(monkeypatch):
    model, calls = _build("kinetics", monkeypatch)
    assert calls == [("protein", "esm-local"), ("ligand", "molt5-local")]
    assert model._prot_spec.hidden_size == HIDDEN and model._lig_spec.hidden_size == HIDDEN

    model.eval()
    with torch.no_grad():
        out = model(["MKTAYIAK", "MKTAYIAL"], ["CCO", "CCC"], use_mask=False)

    assert isinstance(out, tuple) and len(out) == 7
    pred_kcat, pred_Km, pred_ratio, gate_p, gate_s, s_raw, s_eff = out
    assert pred_kcat.shape == (2,) and pred_Km.shape == (2,) and pred_ratio.shape == (2,)
    assert len(s_raw) == 3 and len(s_eff) == 3


def test_affinity_predictor_constructs_without_injected_encoders(monkeypatch):
    model, calls = _build("affinity", monkeypatch)
    assert calls == [("protein", "esm-local"), ("ligand", "molt5-local")]

    model.train()
    mu, log_var = model(["MKTAYIAK", "MKTAYIAL"], ["CCO", "CCC"], use_mask=False)
    assert mu.shape == (2,) and log_var.shape == (2,)

    (mu.mean() + log_var.mean()).backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads, "affinity 通路反传后没有任何参数拿到梯度"


def test_config_rejects_affinity_task_without_affinity_head_spec():
    """R40：直接构造 PredictorConfig(task="affinity") 不再等到 forward 才抛裸 KeyError。"""
    with pytest.raises(ValueError, match="affinity"):
        PredictorConfig(task="affinity")


def test_config_rejects_affinity_head_spec_without_active_affinity_head():
    """R43：head_specs 里有 affinity 但 active_heads 停在 kinetics 缺省值时也必须报错。

    MultiHead 按 cfg.active_heads 构建输出，forward 才读 outs_dict["affinity"]。
    只查 head_specs 会让这个组合通过构造，然后死在裸 KeyError: 'affinity'。
    """
    # head_specs 有 affinity、active_heads 用缺省值 —— 正是漏检的那个组合
    slipped = PredictorConfig(head_specs={"affinity": _small_head(2)})
    assert slipped.active_heads == ["kcat", "Km", "ratio"]

    with pytest.raises(ValueError, match="affinity"):
        PredictorConfig(task="affinity", head_specs={"affinity": _small_head(2)})


def test_config_for_task_affinity_keeps_spec_and_active_head_in_sync():
    """R43 的反面：for_task 同时设置 active_heads 与 head_specs，不能被误伤。"""
    cfg = PredictorConfig.for_task("affinity", **_small_cfg("affinity"))
    assert cfg.active_heads == ["affinity"]
    assert set(cfg.head_specs) == {"affinity"}


def test_config_allows_partial_head_specs_for_kinetics():
    """R42：head_specs 允许只覆盖部分 head。

    main_infer_predictor.py:90-102 按 checkpoint 里实际存在的 head 构造 head_specs，
    只含 kcat 是合法输入；MultiHead 对缺失项走 cfg.head_specs.get(name, {}) 回退，
    所以「active_heads 里每个 head 都必须有 spec」的通用校验会误伤这条推理路径。
    """
    cfg = PredictorConfig(head_specs={"kcat": _small_head(1)})
    assert cfg.active_heads == ["kcat", "Km", "ratio"]
    assert set(cfg.head_specs) == {"kcat"}


def test_predictor_rejects_half_injected_encoders():
    """R39b：只注入一个编码器必须报错，不能静默改去加载真实骨干权重。"""
    cfg = PredictorConfig.for_task("kinetics", **_small_cfg("kinetics"))
    with pytest.raises(ValueError, match="ligand_encoder"):
        KineticsPredictor(device="cpu", esm_model="esm-local", molt5_model="molt5-local",
                          cfg=cfg, use_lora=False, protein_encoder=object())


def test_predictor_rejects_gradient_checkpointing_without_lora():
    """R39a：use_lora=False 时梯度检查点开关会被静默忽略，必须提前报错。"""
    cfg = PredictorConfig.for_task("kinetics", **_small_cfg("kinetics"))
    with pytest.raises(ValueError, match="grad_checkpoint_backbone"):
        KineticsPredictor(device="cpu", esm_model="esm-local", molt5_model="molt5-local",
                          cfg=cfg, use_lora=False, grad_checkpoint_backbone="active")
