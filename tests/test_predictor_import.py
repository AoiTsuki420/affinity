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
                                        std=[0.5] * 4)
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
    二维 (B, 2)（传 (B,) 时 numpy 直接抛 could not broadcast input array from
    shape (4,) into shape (4,2)；真正会静默广播的是 (B,1) 被广播进第二列。
    两种形状都会被 :226 的形状断言抓住）。
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
                                        std=[0.5, float("nan"), 0.5])
    assert np.isfinite(poisoned["nll"])
    assert poisoned["nll"] > 0.0

    dead = _compute_affinity_metrics([1.0, 2.0], [1.0, 2.0],
                                     std=[float("nan"), float("inf")])
    assert np.isnan(dead["nll"])


def test_compute_affinity_metrics_nll_uses_standard_deviation():
    """R44：第三个参数是**标准差** σ = exp(0.5·log_var)，不是方差。

    用 (mu, log_var, y) = (1.0, ln 4, 3.0) 把口径钉在数字上：σ = 2、err = 2，
    nll = 0.5·(ln 2π + 2·ln σ + err²/σ²) = 0.5·(1.8378770664 + 1.3862943611 + 1)
        = 2.112085713764618。
    把 σ 当方差（旧公式 0.5·(ln(2πσ) + err²/σ)）得到 2.2655121234846454，正是
    修复前的记录值；把方差 exp(log_var) = 4 传给现在的公式则得到 2.430232894324563。
    （旧公式配方差与新公式配标准差在代数上等价，所以重点是两边配套：都当 σ 或都当 σ²。）
    """
    import math

    from main_train_predictor_multigpu import _compute_affinity_metrics

    mu, y = 1.0, 3.0
    log_var = math.log(4.0)
    std = math.exp(0.5 * log_var)

    metrics = _compute_affinity_metrics([y], [mu], [std])

    assert std == 2.0
    assert metrics["nll"] == pytest.approx(2.112085713764618, rel=1e-12)

    # 记录出来的 nll 与训练用 NLL（models/affinity_loss.py:86，log_var 为 log-方差）
    # 只差高斯常数 0.5·ln(2π)，与 kinetics 的 _compute_task_metrics_with_sigma 同口径。
    err = y - mu
    train_nll = 0.5 * (log_var + err ** 2 / math.exp(log_var))
    assert metrics["nll"] - train_nll == pytest.approx(0.5 * math.log(2 * math.pi), rel=1e-12)


def test_should_step_without_accumulation():
    from main_train_predictor_multigpu import should_step

    assert all(should_step(i, 1, 5) for i in range(5))


def test_should_step_with_accumulation():
    from main_train_predictor_multigpu import should_step

    # 累积 3 步：第 2、5 步（0-based）是窗口末尾；第 6 步是 epoch 最后一批，
    # 必须 flush（brief 里同一段的两条断言自相矛盾：列表写 False，而
    # test_should_step_flushes_last_partial_window 断言 should_step(6, 3, 7) is True。
    # 按要求「最后一个不满的窗口也要 step，不能静默丢梯度」取 True）。
    flags = [should_step(i, 3, 7) for i in range(7)]
    assert flags == [False, False, True, False, False, True, True]


def test_should_step_flushes_last_partial_window():
    from main_train_predictor_multigpu import should_step

    # 7 步、累积 3：最后一步必须 flush，否则梯度被丢弃
    assert should_step(6, 3, 7) is True


def test_should_sync_grad_is_false_inside_window_when_ddp():
    from main_train_predictor_multigpu import should_sync_grad

    assert should_sync_grad(0, 3, 7, is_ddp=True) is False
    assert should_sync_grad(2, 3, 7, is_ddp=True) is True


def test_should_sync_grad_is_always_true_when_not_ddp():
    from main_train_predictor_multigpu import should_sync_grad

    assert should_sync_grad(0, 3, 7, is_ddp=False) is True


# ---------------------------------------------------------------------------
# Task 16：梯度累积的窗口判定在 train_epoch（epoch 级批次数），不在 train_step
# （那里的 t/steps 是微批索引）。下面用 step 次数与 step 时的梯度把窗口钉死。
# ---------------------------------------------------------------------------

class _StepCountingAdamW(AdamW):
    """记录 step() 次数，以及每次 step 时被跟踪参数的梯度。"""

    def __init__(self, params, tracked, lr=1e-3):
        super().__init__(params, lr=lr)
        self.tracked = tracked
        self.step_count = 0
        self.tracked_grads = []

    def step(self, closure=None):
        self.step_count += 1
        grad = self.tracked.grad
        self.tracked_grads.append(None if grad is None else float(grad.detach()))
        return super().step(closure)


class _NullEMA:
    def update(self, module):
        return None


class _NoSyncAffinity(_VaryingAffinity):
    """带 no_sync() 的 affinity 假模型：记录 DDP 窗口内部是否跳过了梯度同步。"""

    def __init__(self):
        super().__init__()
        self.no_sync_calls = 0

    def no_sync(self):
        model = self

        class _Ctx:
            def __enter__(self):
                model.no_sync_calls += 1
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()


class _KineticsArgs:
    task = "kinetics"
    batch_size = 4
    phys_ablation = "real"
    gate_on_margin = 0.2
    lambda_gate_off = 0.0
    lambda_gate_on = 0.0
    iters_to_accumulate = 1


class _KineticsDummy(nn.Module):
    """kinetics 通路的最小可微替身（与 test_scripts 的同名替身同构）。"""

    def __init__(self):
        super().__init__()
        self.p_k = nn.Parameter(torch.tensor(0.1))
        self.p_m = nn.Parameter(torch.tensor(0.2))
        self.p_r = nn.Parameter(torch.tensor(0.3))
        self.s_kcat = nn.Parameter(torch.tensor(0.0))
        self.s_Km = nn.Parameter(torch.tensor(0.0))
        self.s_ratio = nn.Parameter(torch.tensor(0.0))
        self.last_gate_struct = None
        self.last_struct_avail = None
        self._mse_ema = torch.zeros(3)

    def forward(self, seqs, smiles, use_mask=True, bias_tuple=(0.0, 0.0, 0.0),
                caps=(0.5, 0.5, 0.4), floor_m=-1e9, prot_struct=None,
                lig_struct=None, struct_avail_mask=None, phys_feat=None,
                phys_mask=None, phys_quality=None):
        bsz = len(seqs)
        device = self.p_k.device
        one = torch.ones(bsz, device=device)
        pred_k = self.p_k * one
        pred_m = self.p_m * one
        pred_r = self.p_r * one
        gate_e = torch.sigmoid(torch.stack([pred_k, pred_k], dim=1))
        gate_s = torch.sigmoid(torch.stack([pred_m, pred_m], dim=1))
        if struct_avail_mask is None:
            avail = torch.zeros(bsz, device=device)
        else:
            avail = struct_avail_mask.to(device=device, dtype=pred_k.dtype)
        self.last_gate_struct = torch.zeros_like(avail)
        self.last_struct_avail = avail
        s_raw = (self.s_kcat, self.s_Km, self.s_ratio)
        return pred_k, pred_m, pred_r, gate_e, gate_s, s_raw, s_raw


def _kinetics_batch():
    from data import EnzymeDataset
    from torch.utils.data import DataLoader
    from utils import KineticsScaler

    frame = pd.DataFrame({
        "Sequence": ["MKT"] * 4,
        "Smiles": ["CCO"] * 4,
        "kcat(s^-1)": [1.0, 1.5, 2.0, 1.3],
        "Km(M)": [0.01, 0.02, 0.03, 0.04],
        "source_id": [0, 1, 2, 0],
    })
    scaler = KineticsScaler()
    scaler.fit(frame["kcat(s^-1)"].to_numpy(), frame["Km(M)"].to_numpy())
    dataset = EnzymeDataset(frame, scaler=scaler, supervised=True,
                            return_row_id=True, source_col="source_id")
    return next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))


def _task_fixture(task, accum):
    """返回 (model, batch, args, tracked_param)，两个 task 的初值都确定可复现。"""
    if task == "affinity":
        _, _, batch = _affinity_batch()
        model = _VaryingAffinity()
        args = _AffinityArgs()
        tracked = model.p_mu
    else:
        batch = _kinetics_batch()
        model = _KineticsDummy()
        args = _KineticsArgs()
        tracked = model.p_k
    args.iters_to_accumulate = accum
    return model, batch, args, tracked


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
@pytest.mark.filterwarnings("ignore:.*torch.cuda.amp.autocast.*is deprecated:FutureWarning")
@pytest.mark.parametrize("task", ["kinetics", "affinity"])
def test_train_step_scales_loss_by_accumulation_for_both_tasks(task):
    """R41：iters_to_accumulate 的缩放对两个任务必须一致（每批 1/accum）。

    旧实现里 affinity 除以 accum 而 kinetics 完全不看这个参数：同一个
    --iters_to_accumulate 2 下两条通路的有效学习率差一倍。
    """
    from main_train_predictor_multigpu import train_step

    grads = {}
    for accum in (1, 2):
        model, batch, args, tracked = _task_fixture(task, accum)
        optimizer = _StepCountingAdamW(model.parameters(), tracked)
        train_step(model, batch, optimizer, _NoOpScaler(), args=args)
        grads[accum] = optimizer.tracked_grads[0]

    assert grads[1] is not None and grads[2] is not None
    assert grads[2] == pytest.approx(grads[1] / 2, rel=1e-5)


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_step_do_step_false_accumulates_without_stepping():
    """R31：do_step=False 既不能 step，也不能清梯度；窗口内梯度必须叠加。"""
    from main_train_predictor_multigpu import train_step

    _, _, batch = _affinity_batch()
    args = _AffinityArgs()
    args.iters_to_accumulate = 2
    model = _VaryingAffinity()
    optimizer = _StepCountingAdamW(model.parameters(), model.p_mu)

    train_step(model, batch, optimizer, _NoOpScaler(), args=args, do_step=False)
    first = float(model.p_mu.grad)
    train_step(model, batch, optimizer, _NoOpScaler(), args=args, do_step=False)
    second = float(model.p_mu.grad)

    assert optimizer.step_count == 0
    assert second == pytest.approx(2.0 * first, rel=1e-6)


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_step_do_step_true_closes_the_accumulation_window():
    """R31：真正 step 的那一次调用就地清零，下一个窗口从干净状态开始。"""
    from main_train_predictor_multigpu import train_step

    _, _, batch = _affinity_batch()
    args = _AffinityArgs()
    args.iters_to_accumulate = 2
    model = _VaryingAffinity()
    optimizer = _StepCountingAdamW(model.parameters(), model.p_mu)

    train_step(model, batch, optimizer, _NoOpScaler(), args=args, do_step=False)
    assert model.p_mu.grad is not None
    train_step(model, batch, optimizer, _NoOpScaler(), args=args, do_step=True)

    assert optimizer.step_count == 1
    assert model.p_mu.grad is None


def _affinity_scaler(n_rows):
    from data_affinity import AffinityScaler

    labels = np.array([1.0 + 0.1 * (k % 4) for k in range(n_rows)])
    return AffinityScaler().fit(labels)


def _affinity_loader(n_batches, batch_size, scaler):
    """标签按 batch_size 周期重复：每个 batch 的数据完全相同，梯度才可比。"""
    from data_affinity import AffinityDataset
    from torch.utils.data import DataLoader

    n_rows = n_batches * batch_size
    frame = pd.DataFrame({
        "Sequence": ["MKT"] * n_rows,
        "Smiles": ["CCO"] * n_rows,
        "pKaffinity": [1.0 + 0.1 * (k % batch_size) for k in range(n_rows)],
    })
    dataset = AffinityDataset(frame, label_col="pKaffinity", scaler=scaler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _run_affinity_epoch(monkeypatch, loader, accum, model):
    """跑一个 train_epoch，返回记录 step 次数 / step 时梯度的 optimizer。"""
    import pathlib
    import shutil

    import main_train_predictor_multigpu as training
    from main_train_predictor_multigpu import train_epoch

    args = _AffinityArgs()
    args.iters_to_accumulate = accum
    args.batch_size = int(loader.batch_size)

    monkeypatch.setattr(training.swanlab, "log", lambda payload: None)
    scratch = pathlib.Path("_scratch_accum_epoch")
    scratch.mkdir(exist_ok=True)
    monkeypatch.setitem(training.RUNTIME_PATHS, "train_data_dir", scratch)

    optimizer = _StepCountingAdamW(model.parameters(), model.p_mu)
    try:
        train_epoch(model, loader, optimizer, None, _NoOpScaler(), _NullEMA(),
                    0, args, None, None, 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return optimizer


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_epoch_steps_once_per_accumulation_window(monkeypatch):
    """R31+R41：4 批 / 累积 2 只 step 两次，且窗口末尾的梯度是两批之和。

    基线（累积 1、单批）step 时的梯度就是单批梯度 g；累积 2 的窗口里每批贡献
    g/2，两批叠加后仍是 g。窗口被中途清零（每次 train_step 开头 zero_grad）会
    只拿到 g/2，缩放写成 1/steps 会拿到 2g。
    """
    # 两条通路共用同一个 scaler：否则标签标准化不同，梯度没有可比性。
    scaler = _affinity_scaler(16)
    baseline = _run_affinity_epoch(monkeypatch, _affinity_loader(1, 4, scaler), 1,
                                   _VaryingAffinity())
    assert baseline.step_count == 1
    reference = baseline.tracked_grads[0]

    model = _NoSyncAffinity()
    opt = _run_affinity_epoch(monkeypatch, _affinity_loader(4, 4, scaler), 2, model)

    assert opt.step_count == 2
    assert opt.tracked_grads[0] == pytest.approx(reference, rel=1e-5)
    assert model.no_sync_calls == 0, "非 DDP 通路不该调用 no_sync()"


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
@pytest.mark.parametrize("n_batches,accum,expected", [
    (1, 1, 1),
    (4, 1, 4),
    (4, 2, 2),
    (5, 2, 3),
    (7, 3, 3),
])
def test_train_epoch_step_count_flushes_last_partial_window(monkeypatch, n_batches,
                                                            accum, expected):
    """最后一个不满的窗口必须 flush：否则尾部梯度被静默丢弃。"""
    opt = _run_affinity_epoch(monkeypatch, _affinity_loader(n_batches, 2, _affinity_scaler(16)),
                              accum, _VaryingAffinity())
    assert opt.step_count == expected


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_train_epoch_skips_ddp_sync_inside_accumulation_window(monkeypatch):
    """R31：DDP 下窗口内部走 no_sync()，只在窗口末尾同步一次。"""
    import main_train_predictor_multigpu as training

    class _FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setattr(training, "dist", _FakeDist)
    model = _NoSyncAffinity()
    opt = _run_affinity_epoch(monkeypatch, _affinity_loader(4, 2, _affinity_scaler(8)), 2, model)

    assert opt.step_count == 2
    assert model.no_sync_calls == 2, "窗口内部的两批（i=0/2）必须跳过梯度同步"
