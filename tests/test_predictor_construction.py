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


# ---------------------------------------------------------------------------
# R69：真实 affinity state_dict 必须能走**真实** build_model_from_ckpt
# ---------------------------------------------------------------------------

# depth/dropout 无法从 state_dict 反推，推理侧（_infer_predictor_cfg_from_ckpt）沿用
# 与 kinetics 头相同的固定值 3 / 0.10；训练侧 for_task("affinity") 用的也正是这两个值，
# 所以夹具必须一致，否则重建出来的是另一个形状的头。
_AFFINITY_CKPT_HEAD = {"type": "mlp", "hidden_dim": 16, "depth": 3, "dropout": 0.10,
                       "out_dim": 2}


def _affinity_ckpt_source_model(monkeypatch):
    """造一个「可以存成 checkpoint」的真实亲和力模型（骨干仍是 config 构造的）。"""
    protein_loader, ligand_loader, _ = _loader_calls()
    monkeypatch.setattr("models.plm_backbones.load_protein_backbone", protein_loader)
    monkeypatch.setattr("models.plm_backbones.load_ligand_backbone", ligand_loader)

    cfg_kwargs = _small_cfg("affinity")
    cfg_kwargs["head_specs"] = {"affinity": dict(_AFFINITY_CKPT_HEAD)}
    cfg = PredictorConfig.for_task("affinity", **cfg_kwargs)
    return KineticsPredictor(device="cpu", esm_model="esm-local",
                             molt5_model="molt5-local", cfg=cfg, use_lora=False)


def test_affinity_checkpoint_rebuilds_as_an_affinity_model(monkeypatch):
    """R69：真实 state_dict 走**真实** build_model_from_ckpt，不能打桩。

    这条测试刻意不 monkeypatch build_model_from_ckpt：tests/test_predictor_import.py 的
    lambda 替身正是让这个缺陷溜过去的原因。修复前 `PredictorConfig(**cfg_updates)` 建的是
    kinetics 三头配置——active_heads 变成 ["kcat","Km","ratio"]，训练好的
    multi_head.heads.affinity.* 成了 unexpected keys，forward 返回 7 元组，
    main() 里的两路解包随即 ValueError：整条 affinity 推理路径实际不可用。
    """
    import pathlib
    import shutil

    import torch

    import main_infer_predictor as infer

    scratch = pathlib.Path("_scratch_affinity_ckpt")
    scratch.mkdir(exist_ok=True)
    try:
        saved = _affinity_ckpt_source_model(monkeypatch).state_dict()
        ckpt_path = scratch / "affinity.pt"
        torch.save(saved, ckpt_path)

        rebuilt, cfg, _ckpt = infer.build_model_from_ckpt(
            str(ckpt_path), device="cpu",
            esm_model="esm-local", molt5_model="molt5-local",
        )

        assert cfg.task == "affinity"
        assert cfg.active_heads == ["affinity"]

        rebuilt.eval()
        with torch.no_grad():
            out = rebuilt(["MKTAYIAK", "MKTAYIAL"], ["CCO", "CCC"], use_mask=False)
        assert isinstance(out, tuple) and len(out) == 2, (
            f"affinity 必须返回 (mu, log_sigma2) 两路，实际得到 "
            f"{len(out) if isinstance(out, tuple) else type(out).__name__}"
        )
        mu, log_var = out
        assert mu.shape == (2,) and log_var.shape == (2,)

        # 训练好的亲和力头必须被真正用上，而不是重建一个同名的随机头。
        head_keys = sorted(k for k in saved if k.startswith(infer.AFFINITY_HEAD_PREFIX))
        assert head_keys, "夹具没有造出亲和力头，这条测试就没有意义了"
        rebuilt_state = rebuilt.state_dict()
        for key in head_keys:
            assert key in rebuilt_state, f"重建后的模型里没有 {key}"
            assert rebuilt_state[key].shape == saved[key].shape, f"{key} 形状不一致"
            assert torch.allclose(rebuilt_state[key], saved[key]), f"{key} 没有被加载回训练值"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
