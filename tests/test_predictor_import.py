"""models.predictor 必须能在没有 peft / loralib 的环境里被 import。

KinEAGER 原先在模块顶层 import peft 与 loralib，导致任何不装 peft 的环境
（含本项目的 CPU 单测环境）连导入模型定义都做不到。
"""
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from utils import PredictionRecorder

_AFFINITY_SAMPLES = {
    "Sequence": ["MKT"] * 4,
    "Smiles": ["CCO"] * 4,
    "pKaffinity": [1.0, 2.0, 3.0, 4.0],
}


class _AffinityArgs:
    task = "affinity"
    epochs = 1
    batch_size = 4
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
    iters_to_accumulate = 1


class _VaryingAffinity(nn.Module):
    """mu / log_var 都逐样本变化：常量 log_var 会被 recorder 的零初始化掩盖。"""

    def __init__(self):
        super().__init__()
        self.p_mu = nn.Parameter(torch.tensor(0.1))
        self.p_log_var = nn.Parameter(torch.tensor(0.75))

    def forward(self, seqs, smiles, **kwargs):
        idx = torch.arange(len(seqs), dtype=torch.float32)
        return self.p_mu + 0.1 * idx, self.p_log_var + 0.5 * idx


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


class _ShapeSpyRecorder(PredictionRecorder):
    """记录 update() 实际收到的形状：labels 必须已经是 (B, 2)，不能靠 numpy 广播。"""

    def __init__(self, n_samples, output_dim=3):
        super().__init__(n_samples, output_dim=output_dim)
        self.seen = []

    def update(self, preds, labels):
        self.seen.append((np.shape(preds), np.shape(labels)))
        super().update(preds=preds, labels=labels)


def _affinity_batch():
    """返回 (scaler, frame, batch)：batch 的标签已按 scaler 标准化。"""
    from data_affinity import AffinityDataset, AffinityScaler
    from torch.utils.data import DataLoader

    frame = pd.DataFrame(_AFFINITY_SAMPLES)
    scaler = AffinityScaler().fit(frame["pKaffinity"].to_numpy())
    dataset = AffinityDataset(frame, label_col="pKaffinity", scaler=scaler)
    batch = next(iter(DataLoader(dataset, batch_size=len(frame), shuffle=False)))
    return scaler, frame, batch


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
    """R37：affinity 的 preds 是 (B, 2) = [mu, log_var]，labels 是显式 (B, 2) 标签副本。

    三个断言都必须是可证伪的：log_var 逐样本变化（只把 mu 写进 recorder 会失败）、
    labels 是标准化后的标签（写原始标签或全零会失败）、update() 收到的 labels 已经是
    二维 (B, 2)（传 (B,) 时 numpy 会在 update 内部静默广播，形状断言会失败）。
    """
    scaler, frame, batch = _affinity_batch()
    from main_train_predictor_multigpu import train_step

    model = _VaryingAffinity()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    recorder = _ShapeSpyRecorder(4, output_dim=2)

    loss = train_step(model, batch, optimizer, _NoOpScaler(),
                      recorder=recorder, args=_AffinityArgs())

    assert np.isfinite(float(loss))
    preds, labels = recorder.get()
    assert preds.shape == (4, 2) and labels.shape == (4, 2)
    assert recorder.seen == [((4, 2), (4, 2))]

    idx = np.arange(4, dtype=np.float32)
    assert np.allclose(preds[:, 0], 0.1 + 0.1 * idx, atol=1e-6)
    assert np.allclose(preds[:, 1], 0.75 + 0.5 * idx, atol=1e-6)

    raw = frame["pKaffinity"].to_numpy()
    expected = scaler.transform(raw)
    assert np.allclose(labels[:, 0], expected, atol=1e-6)
    assert not np.allclose(labels[:, 0], raw)


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_step_affinity_reports_sub_losses_to_the_sink():
    """affinity_loss 的分项必须能被调用方拿到：train_epoch 用它写日志。"""
    _, _, batch = _affinity_batch()
    from main_train_predictor_multigpu import train_step

    model = _VaryingAffinity()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    sink = {}

    loss = train_step(model, batch, optimizer, _NoOpScaler(),
                      recorder=_ShapeSpyRecorder(4, output_dim=2),
                      args=_AffinityArgs(), loss_log_sink=sink)

    assert np.isfinite(float(loss))
    for key in ("huber", "barron", "nll", "rank", "var_reg", "total"):
        assert key in sink, f"affinity 分项 {key} 没有进入日志 sink"
        assert np.isfinite(sink[key])


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_val_epoch_affinity_branch_reports_sub_losses(monkeypatch):
    from data_affinity import AffinityDataset
    from main_train_predictor_multigpu import val_epoch
    from torch.utils.data import DataLoader

    import main_train_predictor_multigpu as training

    scaler, frame, _ = _affinity_batch()
    loader = DataLoader(AffinityDataset(frame, label_col="pKaffinity", scaler=scaler),
                        batch_size=4, shuffle=False)
    logged = {}
    monkeypatch.setattr(training.swanlab, "log", lambda payload: logged.update(payload))

    loss = val_epoch(_VaryingAffinity(), loader, 0, _AffinityArgs())

    assert np.isfinite(float(loss))
    assert "affinity_val_total" in logged and "affinity_val_huber" in logged
    assert np.isfinite(logged["affinity_val_nll"])


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_epoch_affinity_routes_sub_losses_to_swanlab(monkeypatch):
    import pathlib
    import shutil

    from data_affinity import AffinityDataset
    from main_train_predictor_multigpu import train_epoch
    from torch.utils.data import DataLoader

    import main_train_predictor_multigpu as training

    scaler, frame, _ = _affinity_batch()
    loader = DataLoader(AffinityDataset(frame, label_col="pKaffinity", scaler=scaler),
                        batch_size=4, shuffle=False)
    logged = {}
    monkeypatch.setattr(training.swanlab, "log", lambda payload: logged.update(payload))

    class _NoOpEMA:
        def update(self, module):
            return None

    scratch = pathlib.Path("_scratch_epoch_loss_logs")
    scratch.mkdir(exist_ok=True)
    monkeypatch.setitem(training.RUNTIME_PATHS, "train_data_dir", scratch)
    try:
        model = _VaryingAffinity()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        train_epoch(model, loader, optimizer, None, _NoOpScaler(), _NoOpEMA(),
                    0, _AffinityArgs(), None, None, 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    assert "affinity_total" in logged and "affinity_huber" in logged
    assert np.isfinite(logged["affinity_nll"])


def test_compute_affinity_metrics_ignores_non_finite_sigma():
    """R43 对位：非有限 sigma 不能污染 NLL，kinetics 路径同样先过滤 sigma。"""
    from main_train_predictor_multigpu import _compute_affinity_metrics

    poisoned = _compute_affinity_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                                        sigma=[0.5, float("nan"), 0.5])
    assert np.isfinite(poisoned["nll"])
    assert poisoned["nll"] > 0.0

    dead = _compute_affinity_metrics([1.0, 2.0], [1.0, 2.0],
                                     sigma=[float("nan"), float("inf")])
    assert np.isnan(dead["nll"])
