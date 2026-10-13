import math

import pytest
import torch

from models.affinity_loss import _barron_loss, affinity_loss


def _inputs(n=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, generator=generator)
    log_var = torch.randn(n, generator=generator) * 0.5
    y = torch.randn(n, generator=generator)
    return mu, log_var, y


def test_loss_is_finite_and_scalar():
    mu, log_var, y = _inputs()
    loss, logs = affinity_loss(mu, log_var, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "total" in logs and "huber" in logs and "nll" in logs


def test_loss_backpropagates_to_both_outputs():
    mu, log_var, y = _inputs()
    mu = mu.clone().requires_grad_(True)
    log_var = log_var.clone().requires_grad_(True)
    loss, _ = affinity_loss(mu, log_var, y, nll_weight=0.1)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert log_var.grad is not None and torch.isfinite(log_var.grad).all()


def test_nan_labels_are_masked_not_propagated():
    """无效标签既不进入回归项，也不产生梯度。"""
    mu, log_var, y = _inputs()
    y[0] = float("nan")
    y[1] = float("inf")
    loss, logs = affinity_loss(mu, log_var, y)
    assert torch.isfinite(loss)

    # 回归项必须与其他有效标签上的平滑 L1 完全一致。
    keep = torch.isfinite(y)
    expected = torch.nn.functional.smooth_l1_loss(mu[keep], y[keep], beta=1.0)
    assert abs(float(logs["huber"]) - float(expected)) < 1e-6

    # 被 mask 的位置不得拿到梯度（nan_to_num 已把 mu 置零，梯度应为 0）。
    mu = mu.clone().requires_grad_(True)
    affinity_loss(mu, log_var, y)[0].backward()
    assert float(mu.grad[0]) == 0.0 and float(mu.grad[1]) == 0.0


def test_all_labels_invalid_returns_finite_zero_loss():
    mu, log_var, _ = _inputs()
    y = torch.full((8,), float("nan"))
    loss, logs = affinity_loss(mu, log_var, y)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert float(logs["total"]) == 0.0
    assert float(logs["huber"]) == 0.0
    assert float(logs["nll"]) == 0.0


def test_extreme_log_var_is_clamped():
    mu, _, y = _inputs()
    log_var = torch.tensor([1e6, -1e6] * 4, dtype=torch.float32)
    loss, _ = affinity_loss(mu, log_var, y, nll_weight=1.0)
    assert torch.isfinite(loss)
    # 反正 +-1e6 会被夹到 +-10：NLL 必须等于用夹过的 log_var 手算的结果。
    clamped = log_var.clamp(-10.0, 10.0)
    expected = 0.5 * (clamped + (y - mu) ** 2 / clamped.exp().clamp_min(1e-6))
    reg = torch.nn.functional.smooth_l1_loss(mu, y, beta=1.0)
    assert abs(float(loss) - float(reg + expected.mean())) < 1e-4


def test_nll_weight_zero_removes_nll_term():
    mu, log_var, y = _inputs()
    _, logs_zero = affinity_loss(mu, log_var, y, nll_weight=0.0)
    _, logs_used = affinity_loss(mu, log_var, y, nll_weight=1.0)
    assert abs(float(logs_zero["total"]) - float(logs_used["total"])) > 1e-6


def test_rank_weight_zero_skips_ranking():
    mu, log_var, y = _inputs()
    # 故意给出完全反序的预测：若 ranking 被激活，rank 项必然非零。
    _, logs = affinity_loss(-mu, log_var, y, rank_weight=0.0)
    assert float(logs["rank"]) == 0.0


def test_rank_weight_positive_enables_ranking():
    """预测完全反序时，margin ranking 必须给出非零值并进入 total。

    rank 项按对随机采样（rank_pairs），因此这里固定全局种子保证可复现。
    margin(1.0) > |预测间隔|(1.0) 使 (0,1)/(1,0) 两种抽法都落在 hinge 的非零
    分支（2.0 / 1.0），抽到的顺序不会让断言靠运气通过。rank_pairs 取 4 相对
    2 个样本，使"至少抽到一个有效对"的概率为 15/16，真正走到采样路径。
    """
    torch.manual_seed(0)
    mu = torch.tensor([1.0, 2.0])
    log_var = torch.zeros(2)
    y = torch.tensor([2.0, 1.0])
    total_zero, logs_zero = affinity_loss(mu, log_var, y, rank_weight=0.0)
    total_used, logs_used = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=1.0, rank_pairs=4
    )
    rank_zero = float(logs_zero["rank"])

    # ranking 未开启时该分项必须精确为 0（否则等于永远给了一个免费正则）。
    assert rank_zero == 0.0
    # 开了权重就必须产生非零 hinge；若采样出的对全被 i==j 丢掉，这里会是 0。
    assert float(logs_used["rank"]) > 0.0
    # 该项确实以 rank_weight 进入 total。
    assert abs(
        float(total_used) - float(total_zero) - float(logs_used["rank"])
    ) < 1e-5


def test_var_reg_modes_differ_on_one_sided_deviation():
    """var_reg 只在非默认权重下生效：abs 惩罚任意偏差，ratio 只惩罚低估。"""
    mu = torch.zeros(4)
    log_var = torch.zeros(4)
    y = torch.tensor([1.0, 1.0, 1.0, 2.0])

    total_zero, logs_zero = affinity_loss(mu, log_var, y, var_reg_weight=0.0)
    total_abs, logs_abs = affinity_loss(
        mu, log_var, y, var_reg_weight=1.0, var_reg_mode="abs"
    )
    _, logs_ratio = affinity_loss(
        mu, log_var, y, var_reg_weight=1.0, var_reg_mode="ratio"
    )
    assert float(logs_zero["var_reg"]) == 0.0
    # 预测方差为 0、真值方差 > 0 → 两种模式都应给出正的正则项。
    assert float(logs_abs["var_reg"]) > 0.0
    assert abs(float(logs_abs["var_reg"]) - float(y.std(unbiased=False))) < 1e-6
    assert float(logs_ratio["var_reg"]) > 0.0
    # 该项确实被计入了 total（权重 1.0 → total 恰好增加 var_reg）。
    assert abs(float(total_abs) - float(total_zero) - float(logs_abs["var_reg"])) < 1e-6

    # 反向的单侧偏差：预测方差大于真值方差 → ratio 归零，abs 仍非零。
    mu_flat = torch.tensor([0.0, 0.0, 0.0, 2.0])
    y_flat = torch.zeros(4)
    _, logs_abs2 = affinity_loss(
        mu_flat, log_var, y_flat, var_reg_weight=1.0, var_reg_mode="abs")
    _, logs_ratio2 = affinity_loss(
        mu_flat, log_var, y_flat, var_reg_weight=1.0, var_reg_mode="ratio")
    assert float(logs_abs2["var_reg"]) > 0.0
    assert float(logs_ratio2["var_reg"]) == 0.0


def test_single_sample_batch_does_not_crash():
    mu = torch.tensor([1.0])
    log_var = torch.tensor([0.0])
    y = torch.tensor([1.5])
    loss, logs = affinity_loss(mu, log_var, y, rank_weight=1.0, var_reg_weight=1.0)
    assert torch.isfinite(loss)
    # 单样本无法构造对/方差 → 两个分支必须安全归零。
    assert float(logs["rank"]) == 0.0
    assert float(logs["var_reg"]) == 0.0


def test_barron_loss_type_runs():
    mu, log_var, y = _inputs()
    loss, logs = affinity_loss(mu, log_var, y, loss_type="barron", barron_alpha=1.0)
    assert torch.isfinite(loss)
    assert "barron" in logs or "huber" in logs


def test_barron_degenerates_to_l2_and_l1_limits():
    """alpha=2 必须精确等于 L2，alpha=0 必须精确等于 log1p(r^2/2)。"""
    error = torch.tensor([0.3, -1.7, 2.5])
    assert torch.allclose(_barron_loss(error, 2.0, 1.0), 0.5 * error ** 2)
    assert torch.allclose(
        _barron_loss(error, 0.0, 1.0), torch.log1p(0.5 * error ** 2)
    )
    # alpha=1（默认）落在两者之间，且 scale 真的参与缩放。
    mid = _barron_loss(error, 1.0, 1.0)
    assert bool((mid > 0.5 * error ** 2).all())
    scaled = _barron_loss(error * 3.0, 1.0, 3.0)
    assert torch.allclose(scaled, mid, atol=1e-6)


def test_barron_regression_term_reaches_total_and_gradients():
    mu, log_var, y = _inputs()
    mu = mu.clone().requires_grad_(True)
    total, logs = affinity_loss(
        mu, log_var, y, loss_type="barron", barron_alpha=1.0, nll_weight=0.0,
    )
    total.backward()
    assert torch.isfinite(mu.grad).all()
    assert float(torch.abs(mu.grad).sum()) > 0.0
    # barron 模式下 huber 日志与 barron 日志同为回归项。
    assert abs(float(logs["barron"]) - float(logs["huber"])) < 1e-6
    assert abs(float(logs["total"]) - float(logs["barron"])) < 1e-6
    # huber 模式下 barron 分项必须为 0。
    _, logs_huber = affinity_loss(mu.detach(), log_var, y)
    assert float(logs_huber["barron"]) == 0.0


def test_gradients_flow_through_ranking_and_var_reg():
    """非默认权重开启的分支必须真的可训练，而不是被 detach 掉。"""
    mu, log_var, y = _inputs()
    mu_r = mu.clone().requires_grad_(True)
    affinity_loss(mu_r, log_var, y, rank_weight=1.0,
                  rank_margin=0.5, loss_type="barron")[0].backward()
    assert mu_r.grad is not None and torch.isfinite(mu_r.grad).all()
    assert float(torch.abs(mu_r.grad).sum()) > 0.0

    mu_v = mu.clone().requires_grad_(True)
    affinity_loss(mu_v, log_var, y, var_reg_weight=1.0)[0].backward()
    assert mu_v.grad is not None and torch.isfinite(mu_v.grad).all()


def test_unknown_loss_type_raises():
    mu, log_var, y = _inputs()
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, loss_type="banana")


def test_var_reg_mode_validated():
    mu, log_var, y = _inputs()
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, var_reg_weight=1.0, var_reg_mode="banana")
