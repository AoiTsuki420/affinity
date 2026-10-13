"""models.predictor 必须能在没有 peft / loralib 的环境里被 import。

KinEAGER 原先在模块顶层 import peft 与 loralib，导致任何不装 peft 的环境
（含本项目的 CPU 单测环境）连导入模型定义都做不到。
"""
import sys

import pytest


def _drop_modules():
    for name in list(sys.modules):
        if name == "peft" or name == "loralib" or name.startswith("models"):
            del sys.modules[name]


def test_predictor_import_does_not_pull_peft():
    _drop_modules()
    import models.predictor  # noqa: F401

    assert "peft" not in sys.modules, "models.predictor 仍在模块顶层 import peft"


def test_predictor_import_does_not_pull_loralib():
    _drop_modules()
    import models.predictor  # noqa: F401

    assert "loralib" not in sys.modules, "models.predictor 仍在模块顶层 import loralib"


def test_predictor_exposes_public_names():
    _drop_modules()
    import models.predictor as module

    for name in ("KineticsPredictor", "PredictorConfig", "MultiHead", "MLPHead"):
        assert hasattr(module, name), f"models.predictor 缺少 {name}"


def test_arg_parser_has_task_defaulting_to_kinetics():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(["-csv", "x.csv", "-esm", "a", "-molt5", "b"])
    assert args.task == "kinetics"


def test_arg_parser_accepts_affinity_task():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(
        ["-csv", "x.csv", "-esm", "a", "-molt5", "b", "--task", "affinity"]
    )
    assert args.task == "affinity"


def test_arg_parser_lora_defaults_to_last_k_30():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(["-csv", "x.csv", "-esm", "a", "-molt5", "b"])
    assert args.lora_layer_policy == "last_k"
    assert args.lora_last_k == 30


def test_arg_parser_accepts_lora_target_suffixes_override():
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(
        ["-csv", "x.csv", "-esm", "a", "-molt5", "b",
         "--lora_target_suffixes", "mixer", "gate_proj"]
    )
    assert args.lora_target_suffixes == ["mixer", "gate_proj"]


def test_arg_parser_rejects_unknown_task():
    from main_train_predictor_multigpu import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["-csv", "x.csv", "-esm", "a", "-molt5", "b", "--task", "banana"]
        )


def test_compute_affinity_step_loss_is_finite():
    import torch

    from main_train_predictor_multigpu import compute_affinity_step_loss

    class _Args:
        affinity_loss_type = "huber"
        affinity_nll_weight = 0.05
        affinity_rank_weight = 0.0
        affinity_rank_margin = 0.0
        affinity_var_reg_weight = 0.0
        affinity_var_reg_mode = "abs"

    mu = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    log_var = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    labels = torch.tensor([1.5, 1.0, 4.0])

    loss, logs = compute_affinity_step_loss((mu, log_var), labels, _Args(), torch.device("cpu"))
    assert torch.isfinite(loss)
    assert "total" in logs
    loss.backward()
    assert mu.grad is not None


def test_compute_affinity_metrics_scores_and_skips_non_finite_rows():
    from main_train_predictor_multigpu import _compute_affinity_metrics

    perfect = _compute_affinity_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],
                                        sigma=[0.5] * 4)
    assert perfect["n"] == 4
    assert perfect["mae"] == 0.0 and perfect["rmse"] == 0.0
    assert perfect["pearson"] == 1.0 and perfect["spearman"] == 1.0
    assert perfect["r2"] == 1.0
    assert perfect["nll"] > 0.0

    masked = _compute_affinity_metrics([1.0, float("nan"), 3.0], [1.0, 5.0, 3.0])
    assert masked["n"] == 2
    assert masked["mae"] == 0.0


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_step_affinity_branch_fills_two_column_recorder():
    """R37：affinity 的 labels 必须以 (B, 2) 显式传入 recorder，两列同为标签。"""
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from data_affinity import AffinityDataset, AffinityScaler
    from main_train_predictor_multigpu import train_step
    from utils import PredictionRecorder

    class _Args:
        task = "affinity"
        affinity_loss_type = "huber"
        affinity_nll_weight = 0.05
        affinity_rank_weight = 0.0
        affinity_rank_margin = 0.0
        affinity_var_reg_weight = 0.0
        affinity_var_reg_mode = "abs"
        iters_to_accumulate = 1

    class _DummyAffinity(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_mu = nn.Parameter(torch.tensor(0.1))
            self.p_log_var = nn.Parameter(torch.tensor(0.0))

        def forward(self, seqs, smiles, **kwargs):
            one = torch.ones(len(seqs))
            return self.p_mu * one, self.p_log_var * one

    class _NoOpScaler:
        """GradScaler(enabled=False) 的等价替身，避免触发 torch.cuda.amp 弃用告警。"""

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            return None

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

    frame = pd.DataFrame({
        "Sequence": ["MKT"] * 4,
        "Smiles": ["CCO"] * 4,
        "pKaffinity": [1.0, 2.0, 3.0, 4.0],
    })
    scaler = AffinityScaler().fit(frame["pKaffinity"].to_numpy())
    dataset = AffinityDataset(frame, label_col="pKaffinity", scaler=scaler)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    model = _DummyAffinity()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    recorder = PredictionRecorder(4, output_dim=2)

    loss = train_step(model, batch, optimizer, _NoOpScaler(),
                      recorder=recorder, args=_Args())

    assert np.isfinite(float(loss))
    preds, labels = recorder.get()
    assert preds.shape == (4, 2) and labels.shape == (4, 2)
    assert np.allclose(labels[:, 0], labels[:, 1])


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_val_epoch_affinity_branch_returns_finite_loss():
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from data_affinity import AffinityDataset, AffinityScaler
    from main_train_predictor_multigpu import val_epoch

    class _Args:
        task = "affinity"
        epochs = 1
        phys_ablation = "real"
        gate_on_margin = 0.2
        lambda_gate_off = 0.0
        lambda_gate_on = 0.0
        affinity_loss_type = "huber"
        affinity_nll_weight = 0.05
        affinity_rank_weight = 0.0
        affinity_rank_margin = 0.0
        affinity_var_reg_weight = 0.0
        affinity_var_reg_mode = "abs"

    class _DummyAffinity(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_mu = nn.Parameter(torch.tensor(0.2))
            self.p_log_var = nn.Parameter(torch.tensor(-0.5))

        def forward(self, seqs, smiles, **kwargs):
            one = torch.ones(len(seqs))
            return self.p_mu * one, self.p_log_var * one

    frame = pd.DataFrame({
        "Sequence": ["MKT"] * 3,
        "Smiles": ["CCO"] * 3,
        "pKaffinity": [1.0, 2.0, 3.0],
    })
    scaler = AffinityScaler().fit(frame["pKaffinity"].to_numpy())
    loader = DataLoader(AffinityDataset(frame, label_col="pKaffinity", scaler=scaler),
                        batch_size=3, shuffle=False)

    loss = val_epoch(_DummyAffinity(), loader, 0, _Args())
    assert np.isfinite(float(loss))
