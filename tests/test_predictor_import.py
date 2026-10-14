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

# 生产代码里的两处 CPU 环境告警：跑 kinetics 通路就会触发，与本任务的断言无关。
_NO_CUDA_WARNING = "ignore:CUDA is not available:UserWarning"
_NO_AUTOCAST_DEPRECATION = (
    "ignore:.*torch.cuda.amp.autocast.*is deprecated:FutureWarning"
)


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


def test_arg_parser_lora_policy_default_is_neutral():
    """R70：argparse 的默认值必须是 None，由 main() 按 task 解析。

    固定成 "last_k" 会覆盖 models/predictor.py 自己的 "all" 缺省值，把同一条 kinetics
    命令的 LoRA 层从 0..32 改成 3..32（198 个目标模块少 18 个），基线行为就变了。
    """
    from main_train_predictor_multigpu import build_arg_parser

    args = build_arg_parser().parse_args(["-csv", "x.csv", "-esm", "a", "-molt5", "b"])
    assert args.lora_layer_policy is None
    assert args.lora_last_k == 30


@pytest.mark.parametrize("task,expected", [("kinetics", "all"), ("affinity", "last_k")])
def test_arg_parser_lora_policy_default_resolves_per_task(task, expected):
    """R70：不传 flag 时 kinetics 回到 "all"（基线），affinity 是设计要求的 last_k。

    这条断言在修复前必然失败：既没有 resolve_lora_layer_policy，默认值也停在 "last_k"。
    """
    from main_train_predictor_multigpu import build_arg_parser, resolve_lora_layer_policy

    args = build_arg_parser().parse_args(
        ["-csv", "x.csv", "-esm", "a", "-molt5", "b", "--task", task]
    )
    assert resolve_lora_layer_policy(args.lora_layer_policy, args.task) == expected


@pytest.mark.parametrize("explicit", ["all", "last_k", "explicit"])
def test_explicit_lora_policy_overrides_the_task_default(explicit):
    """R70：显式传值（消融用）永远优先于任务默认值。"""
    from main_train_predictor_multigpu import resolve_lora_layer_policy

    for task in ("kinetics", "affinity"):
        assert resolve_lora_layer_policy(explicit, task) == explicit


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


class _CountingEMA:
    """只统计 update() 次数：R47 要求它与 optimizer step 同频。"""

    def __init__(self):
        self.updates = 0

    def update(self, module):
        self.updates += 1


class _CountingScheduler:
    """只实现 train_epoch 调用的 step_and_clip()：统计 LR 调度推进次数。"""

    def __init__(self):
        self.steps = 0

    def step_and_clip(self):
        self.steps += 1


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
    epochs = 1
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


def _kinetics_loader(n_batches, batch_size):
    """kinetics 版的确定性 loader；标签按 batch_size 周期重复。"""
    from data import EnzymeDataset
    from torch.utils.data import DataLoader
    from utils import KineticsScaler

    n_rows = n_batches * batch_size
    frame = pd.DataFrame({
        "Sequence": ["MKT"] * n_rows,
        "Smiles": ["CCO"] * n_rows,
        "kcat(s^-1)": [1.0 + 0.1 * (k % batch_size) for k in range(n_rows)],
        "Km(M)": [0.01 + 0.001 * (k % batch_size) for k in range(n_rows)],
        "source_id": [k % 3 for k in range(n_rows)],
    })
    scaler = KineticsScaler()
    scaler.fit(frame["kcat(s^-1)"].to_numpy(), frame["Km(M)"].to_numpy())
    dataset = EnzymeDataset(frame, scaler=scaler, supervised=True,
                            return_row_id=True, source_col="source_id")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _kinetics_batch():
    return next(iter(_kinetics_loader(1, 4)))


class _MseEmaCountingKinetics(_KineticsDummy):
    """统计 `_mse_ema` 被赋值的次数（train_step 每次更新一次）。

    train_step 用的是普通赋值 `model._mse_ema = ...`（不是 in-place），
    所以重载 __setattr__ 就能精确数出更新次数。
    """

    def __init__(self):
        super().__init__()
        self.mse_ema_updates = 0  # __init__ 里那次赋值不计入

    def __setattr__(self, name, value):
        if name == "_mse_ema":
            self.__dict__["mse_ema_updates"] = self.__dict__.get("mse_ema_updates", 0) + 1
        super().__setattr__(name, value)


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


def _run_epoch(monkeypatch, loader, accum, model, args, tracked,
               scheduler=None, ema=None):
    """跑一个 train_epoch，返回 (optimizer, scheduler, ema, global_step)。

    scheduler / ema 缺省都是计数器，用来验证 R47 的「记账按 optimizer step 推进」。
    """
    import pathlib
    import shutil

    import main_train_predictor_multigpu as training
    from main_train_predictor_multigpu import train_epoch

    args.iters_to_accumulate = accum
    args.batch_size = int(loader.batch_size)
    scheduler = _CountingScheduler() if scheduler is None else scheduler
    ema = _CountingEMA() if ema is None else ema

    monkeypatch.setattr(training.swanlab, "log", lambda payload: None)
    scratch = pathlib.Path("_scratch_accum_epoch")
    scratch.mkdir(exist_ok=True)
    monkeypatch.setitem(training.RUNTIME_PATHS, "train_data_dir", scratch)

    optimizer = _StepCountingAdamW(model.parameters(), tracked)
    try:
        global_step, _ = train_epoch(model, loader, optimizer, scheduler, _NoOpScaler(),
                                     ema, 0, args, None, None, 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return optimizer, scheduler, ema, global_step


def _run_affinity_epoch(monkeypatch, loader, accum, model):
    """跑一个 train_epoch，返回记录 step 次数 / step 时梯度的 optimizer。"""
    optimizer, _, _, _ = _run_epoch(monkeypatch, loader, accum, model,
                                    _AffinityArgs(), model.p_mu)
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


@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
def test_optimizer_steps_per_epoch_matches_the_flush_contract():
    """R47：LR 计划长度 = 每 epoch 的 optimizer step 数 = ceil(B / accum)。

    尾窗会 flush，所以不满一个窗口的那一批仍然产生一次 step；accum=1 时
    退化成旧的 len(train_loader)。
    """
    from main_train_predictor_multigpu import optimizer_steps_per_epoch

    assert optimizer_steps_per_epoch(7, 1) == 7      # 默认路径与旧式相同
    assert optimizer_steps_per_epoch(4, 1) == 4
    assert optimizer_steps_per_epoch(4, 2) == 2
    assert optimizer_steps_per_epoch(5, 2) == 3      # 尾窗 flush
    assert optimizer_steps_per_epoch(7, 3) == 3
    assert optimizer_steps_per_epoch(1, 4) == 1      # 一批也要 step
    assert optimizer_steps_per_epoch(0, 3) == 0


@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
@pytest.mark.parametrize("n_batches,accum,expected", [
    (1, 1, 1),
    (4, 1, 4),
    (4, 2, 2),
    (5, 2, 3),
    (7, 3, 3),
])
def test_train_epoch_step_count_flushes_last_partial_window(monkeypatch, n_batches,
                                                            accum, expected):
    """最后一个不满的窗口必须 flush：否则尾部梯度被静默丢弃。

    同时把「LR 计划长度公式」与真实 step 次数对上：train_one_stage 用
    optimizer_steps_per_epoch() 算 OneCycle/余弦的 total_steps，公式错了
    LR 计划就会与实际步数脱节。
    """
    from main_train_predictor_multigpu import optimizer_steps_per_epoch

    opt = _run_affinity_epoch(monkeypatch, _affinity_loader(n_batches, 2, _affinity_scaler(16)),
                              accum, _VaryingAffinity())
    assert opt.step_count == expected
    assert opt.step_count == optimizer_steps_per_epoch(n_batches, accum)


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


# ---------------------------------------------------------------------------
# R47：LR 调度 / global_step / ema.update() / _mse_ema 必须与 optimizer step 同频。
# 累积 k 时它们若按 batch 推进，LR 计划会提前 k 倍走完、EMA 会被多更新 k 倍。
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
@pytest.mark.filterwarnings(_NO_AUTOCAST_DEPRECATION)
def test_train_epoch_accumulation_advances_bookkeeping_per_optimizer_step(monkeypatch):
    """R47：7 批 / 累积 3 → 五个计数全部是 3（optimizer step 次数），不是 7。"""
    model = _MseEmaCountingKinetics()
    opt, sched, ema, global_step = _run_epoch(
        monkeypatch, _kinetics_loader(7, 2), 3, model, _KineticsArgs(), model.p_k)

    assert opt.step_count == 3
    assert sched.steps == 3, "LR 调度器必须按 optimizer step 推进"
    assert global_step == 3, "global_step 必须按 optimizer step 推进"
    assert ema.updates == 3, "ema.update() 必须按 optimizer step 推进"
    assert model.mse_ema_updates == 3, "_mse_ema 必须按 optimizer step 推进"


@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
@pytest.mark.filterwarnings(_NO_AUTOCAST_DEPRECATION)
def test_train_epoch_default_path_advances_bookkeeping_every_batch(monkeypatch):
    """R47 反面：iters_to_accumulate=1（默认）时五个计数仍然各推进 7 次。

    默认路径每批都 step，所以新门控是 no-op —— 这里把「没有变化」钉住。
    """
    model = _MseEmaCountingKinetics()
    opt, sched, ema, global_step = _run_epoch(
        monkeypatch, _kinetics_loader(7, 2), 1, model, _KineticsArgs(), model.p_k)

    assert opt.step_count == 7
    assert sched.steps == 7
    assert global_step == 7
    assert ema.updates == 7
    assert model.mse_ema_updates == 7


@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
def test_train_epoch_affinity_accumulation_advances_bookkeeping_per_optimizer_step(monkeypatch):
    """R47：affinity 通路走同一个门控（affinity 没有 _mse_ema，其余四个在列）。"""
    model = _VaryingAffinity()
    opt, sched, ema, global_step = _run_epoch(
        monkeypatch, _affinity_loader(7, 2, _affinity_scaler(8)), 3, model,
        _AffinityArgs(), model.p_mu)

    assert opt.step_count == 3
    assert sched.steps == 3
    assert global_step == 3
    assert ema.updates == 3


# ---------------------------------------------------------------------------
# R54：OneCycleLR 的构造函数把**每个**参数组都设成 max_lr / div_factor（默认
# div_factor=10，即 base_lr/10），而 onecycle.step() 只在累积窗口末尾调用。
# 非调度组（tag 不在 head/lora*）的 base_lr 钉住必须在每个 batch 都执行。
# ---------------------------------------------------------------------------

class _TaggedLrRecorder(AdamW):
    """记录每次 step() 时被跟踪 tag 的参数组 lr。"""

    def __init__(self, param_groups, watched_tag):
        super().__init__(param_groups, lr=1e-3)
        self.watched_tag = watched_tag
        self.watched_lrs = []

    def step(self, closure=None):
        for group in self.param_groups:
            if group.get("tag") == self.watched_tag:
                self.watched_lrs.append(float(group["lr"]))
        return super().step(closure)


@pytest.mark.filterwarnings(_NO_CUDA_WARNING)
def test_train_epoch_pins_non_scheduled_lr_every_batch_under_accumulation(monkeypatch):
    """R54：累积 2 时第一个窗口末尾的 optimizer.step() 必须已经用回 base_lr。

    OneCycleLR 构造后非调度组是 base_lr/10；钉住在 do_step 门控里时，窗口内部
    的第 0..k-1 批都不会恢复它，于是第一个窗口末尾那次 step 仍按 base_lr/10 走
    （旧实现的 watched_lrs 是 [base/10, base]），门控外执行后是 [base, base]。
    """
    import pathlib
    import shutil

    from torch.optim.lr_scheduler import OneCycleLR

    import main_train_predictor_multigpu as training

    base_head, base_backbone = 1e-3, 3e-2
    model = _VaryingAffinity()
    optimizer = _TaggedLrRecorder(
        [
            {"params": [model.p_mu], "lr": base_head, "tag": "head", "base_lr": base_head},
            {"params": [model.p_log_var], "lr": base_backbone, "tag": "backbone",
             "base_lr": base_backbone},
        ],
        watched_tag="backbone",
    )

    args = _AffinityArgs()
    args.iters_to_accumulate = 2
    args.batch_size = 2
    # 4 个 optimizer step（8 批 / 累积 2），pct_start=0.3 让第 0 步落在 warmup 里，
    # 于是构造后的初值正好是 max_lr/div_factor —— 与生产默认值同形。
    onecycle = OneCycleLR(optimizer, max_lr=[base_head, base_backbone], epochs=1,
                          steps_per_epoch=4, div_factor=10.0, final_div_factor=1e4,
                          cycle_momentum=False)
    assert [float(g["lr"]) for g in optimizer.param_groups] == [
        pytest.approx(base_head / 10), pytest.approx(base_backbone / 10),
    ]

    monkeypatch.setattr(training.swanlab, "log", lambda payload: None)
    scratch = pathlib.Path("_scratch_onecycle_pin")
    scratch.mkdir(exist_ok=True)
    monkeypatch.setitem(training.RUNTIME_PATHS, "train_data_dir", scratch)
    try:
        training.train_epoch(model, _affinity_loader(8, 2, _affinity_scaler(16)), optimizer,
                             None, _NoOpScaler(), _CountingEMA(), 0, args, None, None, 0,
                             use_onecycle=True, onecycle=onecycle)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    assert optimizer.watched_lrs == [pytest.approx(base_backbone)] * 4


# ---------------------------------------------------------------------------
# Task 17：推理侧的 affinity 单路输出。--task affinity 走 (mu, log_var) 两路解包，
# 其余仍走 kcat/Km/ratio 三路；format_affinity_prediction 是 CSV 字段的唯一出口。
# ---------------------------------------------------------------------------

class _FakeAffinityInferModel(nn.Module):
    """main_infer_predictor.main() 的亲和力替身：常量 (mu, log_var)，不碰权重。"""

    def forward(self, seqs, smiles, source_ids=None, use_mask=False):
        import math

        batch = len(seqs)
        return (torch.full((batch,), 7.5),
                torch.full((batch,), math.log(0.25)))


class _FakeKineticsInferModel(nn.Module):
    """kinetics 替身：脚本只取 out[:3]，ratio 由 pred_kcat - pred_km + 3 重算。"""

    def forward(self, seqs, smiles, source_ids=None, use_mask=False):
        batch = len(seqs)
        return (torch.full((batch,), 1.0),
                torch.full((batch,), -3.0),
                torch.zeros(batch))


def _cfg_for_task(task):
    """替身也要给出**正确的 task cfg**（R69）。

    真实 build_model_from_ckpt 现在按 checkpoint 决定 cfg.task，main() 会用
    cfg.task 与 args.task 对齐（构造与循环必须一致）。替身返回 cfg=None 时那三个
    e2e 测试会因为「--task affinity 但 checkpoint 没有亲和力头」而失败——那正是
    守卫在正确工作，所以这里让替身说实话。
    """
    from main_infer_predictor import PredictorConfig

    return PredictorConfig.for_task("affinity" if task == "affinity" else "kinetics")


def _run_infer_main(monkeypatch, model, frame, task):
    """在一次性的顶层目录里跑 main_infer_predictor.main()，返回写出的 CSV。

    **build_model_from_ckpt 在这里被替身顶掉**，所以脚本不读真实权重、不下载任何东西；
    这是有意的：这三个测试断言的是循环侧的输出形状（列名、两路/三路解包、真值列门），
    不是加载器。加载器本身的真实覆盖由
    `tests/test_predictor_construction.py::test_affinity_checkpoint_rebuilds_as_an_affinity_model`
    提供——R69 正是被这个 lambda 模式漏掉的，所以那条测试**不**打桩。
    脚本按 cwd 写 predictions_<task>.csv，因此必须在一次性目录里跑。
    """
    import os
    import pathlib
    import shutil

    import main_infer_predictor as infer

    scratch = pathlib.Path(f"_scratch_infer_{task}")
    scratch.mkdir(exist_ok=True)
    original_cwd = pathlib.Path.cwd()
    try:
        frame.to_csv(scratch / "input.csv", index=False)
        ckpt_cfg = _cfg_for_task(task)
        monkeypatch.setattr(infer, "build_model_from_ckpt",
                            lambda *a, **k: (model, ckpt_cfg, {}))
        monkeypatch.setattr(infer, "check_model_weights", lambda m, ckpt: None)
        monkeypatch.setattr(sys, "argv", [
            "main_infer_predictor.py", "-csv", "input.csv", "-weights", "fake.pt",
            "--task", task,
        ])
        monkeypatch.chdir(scratch)
        infer.main()
        return pd.read_csv(f"predictions_{task}.csv")
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def test_format_affinity_prediction_round_trips_sigma():
    import math

    from main_infer_predictor import format_affinity_prediction

    out = format_affinity_prediction(mu=7.5, log_var=math.log(0.25))
    assert abs(out["pred_pKaffinity"] - 7.5) < 1e-6
    assert abs(out["sigma"] - 0.5) < 1e-6
    assert abs(out["sigma2"] - 0.25) < 1e-6


def test_format_affinity_prediction_clamps_log_variance_overflow():
    """R48：夹取守的是 math.exp 的**上溢**，不是「负方差」——exp 永远为正。

    未夹取时 math.exp(1000.0) 抛 OverflowError（math range error），
    math.exp(-1000.0) 则下溢成 0.0；所以下面两组断言在删掉夹取后都会失败。
    夹取上界取 ±10（= 训练期 models/affinity_loss.py 的 _LOG_VAR_MIN/_LOG_VAR_MAX），
    见 AFFINITY_LOG_VAR_BOUND 的注释。
    """
    import math

    from main_infer_predictor import AFFINITY_LOG_VAR_BOUND, format_affinity_prediction

    high = format_affinity_prediction(mu=5.0, log_var=1000.0)
    assert high["sigma2"] == pytest.approx(math.exp(AFFINITY_LOG_VAR_BOUND))
    assert high["sigma"] == pytest.approx(math.exp(AFFINITY_LOG_VAR_BOUND / 2.0))
    assert math.isfinite(high["sigma2"]) and math.isfinite(high["sigma"])

    low = format_affinity_prediction(mu=5.0, log_var=-1000.0)
    assert low["sigma2"] == pytest.approx(math.exp(-AFFINITY_LOG_VAR_BOUND))
    assert low["sigma"] > 0.0


def test_affinity_inference_clamp_matches_the_training_loss():
    """推理端的夹取必须与训练期的同一约定一致，否则 sigma 会跑到标定之外。

    两边各自持有一个常量（推理端 AFFINITY_LOG_VAR_BOUND，损失里 _LOG_VAR_MIN/MAX），
    这条断言把二者钉在一起：谁改了一边而没改另一边，它就会红。
    """
    from main_infer_predictor import AFFINITY_LOG_VAR_BOUND
    from models.affinity_loss import _LOG_VAR_MAX, _LOG_VAR_MIN

    assert _LOG_VAR_MIN == -_LOG_VAR_MAX        # 损失自己的对称性
    assert AFFINITY_LOG_VAR_BOUND == _LOG_VAR_MAX


def test_lr_find_is_refused_for_the_affinity_task():
    """Important 3：--lr_find 走三头损失，affinity 的单头输出会在解包处 ValueError。

    拒绝比跑到第 50 个 batch 才炸更有用；kinetics 下不受影响。
    """
    from main_train_predictor_multigpu import validate_task_option_compatibility

    with pytest.raises(ValueError, match=r"--lr_find"):
        validate_task_option_compatibility("affinity", True)

    # kinetics 用同一个开关是合法的，affinity 不打开 lr_find 也合法
    validate_task_option_compatibility("kinetics", True)
    validate_task_option_compatibility("affinity", False)


def test_main_infer_predictor_affinity_task_writes_pkabind_columns(monkeypatch):
    """Task 17 的两个 affinity 分支：解析 --task affinity、两路解包、写 4 列。"""
    out = _run_infer_main(
        monkeypatch,
        _FakeAffinityInferModel(),
        pd.DataFrame({
            "pdb_id": ["1abc", "2def"],
            "sequence": ["MKT", "MKT"],
            "smiles": ["CCO", "CCO"],
        }),
        "affinity",
    )

    assert list(out.columns) == ["pdb_id", "pred_pKaffinity", "sigma", "sigma2"]
    assert out["pdb_id"].tolist() == ["1abc", "2def"]
    assert np.allclose(out["pred_pKaffinity"], 7.5)
    assert np.allclose(out["sigma"], 0.5)
    assert np.allclose(out["sigma2"], 0.25)


@pytest.mark.parametrize("injected", [{"value": [6.0, 6.0]}, {"log10_value": [0.8, 0.8]}])
def test_main_infer_predictor_affinity_never_writes_kinetics_truth_columns(monkeypatch,
                                                                          injected):
    """R59：真值提取是 kinetics 专用的。

    affinity 的真值列是 pKaffinity（已是 -log10 空间，由 pdb_id 在外部对齐）。
    没有任务门时，带 value / log10_value 列的 affinity CSV 会被 else 分支按 Km 的
    线性列名读走，然后写出 true_Km_log / true_Km —— 下游读者无法察觉的错标签。
    """
    frame = pd.DataFrame({
        "pdb_id": ["1abc", "2def"],
        "sequence": ["MKT", "MKT"],
        "smiles": ["CCO", "CCO"],
        **injected,
    })
    out = _run_infer_main(monkeypatch, _FakeAffinityInferModel(), frame, "affinity")

    assert list(out.columns) == ["pdb_id", "pred_pKaffinity", "sigma", "sigma2"]
    assert [c for c in out.columns if c.startswith("true")] == []


def test_main_infer_predictor_kinetics_task_keeps_three_head_columns(monkeypatch):
    """kinetics 路径的对外形状不变：三头解包 + 10 的幂还原 + true_kcat 字段。"""
    out = _run_infer_main(
        monkeypatch,
        _FakeKineticsInferModel(),
        pd.DataFrame({
            "sequence": ["MKT", "MKT"],
            "smiles": ["CCO", "CCO"],
            "kcat(s^-1)": [1.0, 1.0],
        }),
        "kcat",
    )

    assert "sigma" not in out.columns and "sigma2" not in out.columns
    assert np.allclose(out["pred_kcat"], 10.0)
    assert np.allclose(out["pred_Km"], 1e-6)
    assert np.allclose(out["pred_kcat_div_Km"], 1e7)
    assert np.allclose(out["true_kcat"], 1.0)
    assert out["source_id"].tolist() == [0, 0]
