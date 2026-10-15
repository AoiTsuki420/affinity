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


def _canonical_barron(r, alpha, scale=1.0):
    """Barron (CVPR 2019) 的独立实现，逐项对应

    main_train_predictor_multigpu.py:247-260 的 _barron_loss_per_sample。
    这里用纯 Python 标量重写一遍，使测试不依赖被测模块自身的公式。
    """
    r2 = (r / max(scale, 1e-8)) ** 2
    if abs(alpha - 2.0) < 1e-6:
        return 0.5 * r2
    if abs(alpha) < 1e-6:
        return math.log1p(0.5 * r2)
    beta = abs(alpha - 2.0)
    return (beta / alpha) * ((r2 / beta + 1.0) ** (alpha / 2.0) - 1.0)


def test_barron_degenerates_to_l2_and_l1_limits():
    """alpha=2 必须精确等于 L2，alpha=0 必须精确等于 log1p(r^2/2)。"""
    error = torch.tensor([0.3, -1.7, 2.5])
    assert torch.allclose(_barron_loss(error, 2.0, 1.0), 0.5 * error ** 2)
    assert torch.allclose(
        _barron_loss(error, 0.0, 1.0), torch.log1p(0.5 * error ** 2)
    )


def test_barron_alpha_one_is_sqrt_not_squared_error():
    """默认 alpha=1 必须是 sqrt(1+r^2)-1（次二次），而不是退化成 r^2。

    判别性断言：beta=alpha=1 时 canonical 化简为 sqrt(1+r^2)-1。
    """
    for r in (0.5, -1.3, 2.0):
        error = torch.tensor([r])
        got = float(_barron_loss(error, 1.0, 1.0))
        expected = math.sqrt(1.0 + r * r) - 1.0
        assert abs(got - expected) < 1e-6
        # 明确排除 r^2 —— 旧公式 f=1 时恰好返回 r^2。
        assert abs(got - r * r) > 1e-3


def test_barron_matches_canonical_form_for_non_special_alpha():
    """非特殊 alpha 必须逐值等于 canonical 形式，防止再次出现公式分歧。"""
    error = torch.tensor([0.5, -1.3, 2.0])
    for alpha in (1.5, 3.0, -1.0):
        expected = torch.tensor(
            [_canonical_barron(float(r), alpha) for r in error]
        )
        got = _barron_loss(error, alpha, 1.0)
        assert torch.allclose(got, expected, atol=1e-6), (alpha, got, expected)
        # 稳健损失必须非负。
        assert bool((got >= 0.0).all())


def test_barron_scale_invariance_and_large_error_growth():
    """scale 参与缩放；alpha=1 在大误差下渐近线性（这才是稳健性的来源）。

    注意 alpha=0 分支是 log1p，只按对数增长；真正渐近 |r| 的是 alpha=1 的
    sqrt(1+r^2)-1。alpha=2 才是二次增长。
    """
    error = torch.tensor([0.3, -1.7, 2.5])
    mid = _barron_loss(error, 1.0, 1.0)
    scaled = _barron_loss(error * 3.0, 1.0, 3.0)
    assert torch.allclose(scaled, mid, atol=1e-6)

    # alpha=1：大误差下约等于 |r|（线性增长 = 稳健）。
    # 容差取 10.0：float32 在 r=1e6 处只能给到约 1e-6 的相对精度，
    # 但该容差已足以区分线性（~1e6）与二次（~1e12）增长。
    big = torch.tensor([1.0e6])
    assert abs(float(_barron_loss(big, 1.0, 1.0)) - 1.0e6) < 10.0
    # alpha=2：同样的大误差会给出二次爆炸。
    assert float(_barron_loss(big, 2.0, 1.0)) > 1.0e11
    # alpha=0：只按对数增长，远小于 |r|。
    assert float(_barron_loss(big, 0.0, 1.0)) < 30.0


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


# ---------------------------------------------------------------------------
# ProtAff margin-ranking：最小间隔过滤（δ_min）+ gap 加权
# ---------------------------------------------------------------------------

def _pair_inputs(y, mu):
    """把 (y, mu) 钉死成显式张量，其余分项交给默认值，便于手算 rank 项。"""
    return (
        torch.tensor(mu, dtype=torch.float32),
        torch.zeros(len(y), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def _clear_degenerate_rank_log():
    """清空退化日志计数器，好断言"第一次退化一定打印"。

    计数器是模块级的（否则一个 epoch 会刷几万行），所以测试必须显式重置，
    否则断言会依赖测试执行顺序。
    """
    import models.affinity_loss as affinity_module

    affinity_module._DEGENERATE_RANK_COUNTS.clear()


def test_rank_weight_zero_is_exactly_zero_whatever_the_new_parameters_say():
    """rank_weight=0 仍是"完全没有 rank 项"：新参数不得漏进 total。

    rank_weight=0 是今天所有在跑的命令的取值，新增的 δ_min / gap 加权 / clamp
    只要有一处忘了被 rank_weight 门控（或把新参数加进 total），这条就会失败。
    """
    mu, log_var, y = _inputs()
    total_off, logs_off = affinity_loss(mu, log_var, y, rank_weight=0.0)
    total_new, logs_new = affinity_loss(
        mu, log_var, y,
        rank_weight=0.0,
        rank_pairs=4,
        rank_min_gap=1.0,
        rank_gap_weight=True,
        rank_gap_clamp=(0.1, 5.0),
    )
    assert float(logs_new["rank"]) == 0.0
    assert float(logs_new["rank_pairs"]) == 0.0
    assert float(logs_new["rank_gap_mean"]) == 0.0
    # 与一个完全不传新参数的调用逐位相同。
    assert float(total_new) == float(total_off)


def test_rank_min_gap_zero_enumerates_every_pair_at_the_n_pair_ceiling():
    """δ=0（不过滤）时，天花板 n(n-1) 个有序对全部进入 rank 项。

    手算：y=[0,2,4,6]、mu=y、margin=3.0。有序对 (i,j) 的 hinge =
    relu(margin - sign(Δy)·Δmu) = relu(3 - |Δy|)（sign 与 Δmu 同时变号，两个方向
    等值）。|Δy|=2 有 6 对、=4 有 4 对、=6 有 2 对，共 n(n-1)=12 对：
        rank = (6·1.0 + 4·0 + 2·0) / 12 = 0.5
        rank_gap_mean = (6·2 + 4·4 + 2·6) / 12 = 40/12
    一个"抽 12 次但有放回"的实现会给出更少的**不同**对，rank 与 rank_pairs
    都会偏离这两个数（旧实现对 batch=4 期望只有约 6.5 个不同对）。
    """
    mu, log_var, y = _pair_inputs([0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0])
    total_zero, logs_zero = affinity_loss(mu, log_var, y, rank_weight=0.0)
    total_used, logs = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=3.0, rank_pairs=256,
        rank_min_gap=0.0,
    )

    assert float(logs["rank_pairs"]) == 12.0
    assert abs(float(logs["rank"]) - 0.5) < 1e-6
    assert abs(float(logs["rank_gap_mean"]) - 40.0 / 12.0) < 1e-6
    # 两个新日志项都是标量 float 张量（_accumulate_affinity_loss_logs 直接 .item()）。
    for key in ("rank_pairs", "rank_gap_mean"):
        assert torch.is_tensor(logs[key])
        assert logs[key].ndim == 0
        assert logs[key].dtype == torch.float32
    # rank 项按 rank_weight 进入 total，且新日志项本身不改变 total。
    assert abs(float(total_used) - float(total_zero) - float(logs["rank"])) < 1e-6
    # 天花板是硬的：要再多也拿不到第 13 对。
    _, logs_more = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=3.0, rank_pairs=100_000,
        rank_min_gap=0.0,
    )
    assert float(logs_more["rank_pairs"]) == 12.0


def test_gap_filter_zeroes_the_term_and_says_so_when_no_pair_clears_the_gap(capsys):
    """标签两两相差 < 1.0 的 batch：rank 恰好 0、rank_pairs 恰好 0，并打印原因。

    "批量太小 / 间隔太严 → 静默为 0"是这次改动最容易出的错：run 看起来在用
    ranking，实际上一对都没有。这里同时钉住数值（精确 0）和日志（必须打印）。
    """
    _clear_degenerate_rank_log()
    mu, log_var, y = _pair_inputs([0.0, 0.4, 0.9, 0.2], [1.0, -1.0, 1.0, -1.0])
    total_zero, logs_zero = affinity_loss(mu, log_var, y, rank_weight=0.0)
    total_gap, logs_gap = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_pairs=256, rank_min_gap=1.0
    )

    assert float(logs_gap["rank"]) == 0.0
    assert float(logs_gap["rank_pairs"]) == 0.0
    assert float(logs_gap["rank_gap_mean"]) == 0.0
    assert float(total_gap) == float(total_zero)
    printed = capsys.readouterr().out
    assert "[affinity_loss] rank term is 0" in printed
    # 原因必须带上被哪个 δ_min 挡掉的，否则运维只能猜是 batch 太小还是间隔太严。
    assert "rank_min_gap=1.0" in printed


def test_fewer_than_two_valid_labels_logs_zero_pairs(capsys):
    """只有 1 个有效标签时同样必须留下"0 对"的痕迹，而不是静默。"""
    _clear_degenerate_rank_log()
    mu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    log_var = torch.zeros(3, dtype=torch.float32)
    y = torch.tensor([1.5, float("nan"), float("inf")], dtype=torch.float32)

    _, logs = affinity_loss(mu, log_var, y, rank_weight=1.0, rank_min_gap=1.0)

    assert float(logs["rank"]) == 0.0
    assert float(logs["rank_pairs"]) == 0.0
    assert float(logs["rank_gap_mean"]) == 0.0
    assert "[affinity_loss] rank term is 0" in capsys.readouterr().out


def test_small_microbatch_pair_ceiling_is_reported_even_when_pairs_survive(capsys):
    """batch=4 不是"退化"（12 对 > 0），但天花板照样必须被说出来。

    这条防的是最危险的静默情形：没有报错、rank_pairs 也不是 0，可是请求的 256 对
    被 n_valid*(n_valid-1)=12 截断，一个 step 只有 12 个（其中一半还是彼此反向的
    重复对）。run 看上去完全正常，实验却不成立。
    """
    _clear_degenerate_rank_log()
    mu, log_var, y = _pair_inputs([0.0, 2.0, 4.0, 6.0], [1.0, -1.0, 1.0, -1.0])
    _, logs = affinity_loss(mu, log_var, y, rank_weight=1.0, rank_margin=0.1,
                            rank_pairs=256, rank_min_gap=1.0)

    assert float(logs["rank_pairs"]) == 12.0
    printed = capsys.readouterr().out
    assert "[affinity_loss] rank_pairs ceiling" in printed
    assert "n_valid=4" in printed and "12" in printed


def test_large_microbatch_does_not_print_the_ceiling_advice(capsys):
    """天花板没被撞到时不许刷日志（batch=64 的天花板 4032 > 请求的 256）。

    没有这条，"每次调用都打印"的实现也能让上一条通过。
    """
    _clear_degenerate_rank_log()
    labels = torch.arange(64, dtype=torch.float32) * 0.5
    mu = torch.zeros(64, dtype=torch.float32)
    log_var = torch.zeros(64, dtype=torch.float32)
    _, logs = affinity_loss(mu, log_var, labels, rank_weight=1.0, rank_margin=0.1,
                            rank_pairs=256, rank_min_gap=0.0)

    assert float(logs["rank_pairs"]) == 256.0
    assert "[affinity_loss] rank_pairs ceiling" not in capsys.readouterr().out


def test_gap_filter_drops_the_near_tie_the_unfiltered_call_keeps():
    """δ_min 必须真的把"近邻对"剔除，而不只是记一个数。

    y=[0.0, 0.4, 3.0]，mu=[2.0, 0.0, 0.5]（模型把 0↔1 这对近邻反序预测），
    margin=0.1。两个方向等值，逐对手算：
        (0,1)/(1,0): |Δy|=0.4, 反序 → relu(0.1 + 2.0) = 2.1
        (0,2)/(2,0): |Δy|=3.0, Δmu=1.5 → relu(0.1 + 1.5) = 1.6
        (1,2)/(2,1): |Δy|=2.6, Δmu=-0.5 → relu(0.1 - 0.5) = 0.0
    不过滤：6 对 → (2.1·2 + 1.6·2 + 0·2)/6 = 7.4/6
    δ_min=1.0：近邻对被剔除 → 4 对 → (1.6·2 + 0·2)/4 = 0.8
    也就是"最不可信的那对反序"正是被 δ_min 剔掉的那对，rank 从 1.2333 掉到 0.8。
    """
    mu, log_var, y = _pair_inputs([0.0, 0.4, 3.0], [2.0, 0.0, 0.5])

    _, logs_raw = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=0.1, rank_pairs=6,
        rank_min_gap=0.0,
    )
    _, logs_gap = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=0.1, rank_pairs=6,
        rank_min_gap=1.0,
    )

    assert float(logs_raw["rank_pairs"]) == 6.0
    assert abs(float(logs_raw["rank"]) - 7.4 / 6.0) < 1e-6
    assert float(logs_gap["rank_pairs"]) == 4.0
    assert abs(float(logs_gap["rank"]) - 0.8) < 1e-6
    assert float(logs_gap["rank_gap_mean"]) >= 1.0


def test_gap_filter_includes_a_pair_at_exactly_the_minimum_gap():
    """边界语义是 |Δy| >= δ_min（不是 >）：δ 是"低于它才排除"。

    y=[0,1,4]、δ=1.0 → (0,1) 的 |Δy| 恰好 = 1.0 必须保留：n(n-1)=6 对全在，
    rank_gap_mean 正好 = (1+4+3)·2/6 = 8/3。写成严格 > 的实现只有 4 对。
    """
    mu, log_var, y = _pair_inputs([0.0, 1.0, 4.0], [0.0, 1.0, 4.0])
    _, logs = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=0.0, rank_pairs=6,
        rank_min_gap=1.0,
    )
    assert float(logs["rank_pairs"]) == 6.0
    assert abs(float(logs["rank_gap_mean"]) - 8.0 / 3.0) < 1e-6


def test_gap_weight_matches_the_hand_computed_weighted_mean():
    """gap 加权 = Σ w·hinge / Σ w，w = clamp(|Δy|, lo, hi)，ProtAff 的 clamp。

    y=[0.0, 1.0, 3.0]、δ_min=2.0、mu=[0.0, 1.0, -2.0]、margin=0.1。
    δ_min=2.0 下的有效有序对只有 4 个（(0,1) 的 |Δy|=1.0 被剔除）：
        (0,2)/(2,0): |Δy|=3.0 → w=3.0，hinge = relu(0.1 + (0-(-2))) = 2.1
        (1,2)/(2,1): |Δy|=2.0 → w=2.0，hinge = relu(0.1 + (1-(-2))) = 3.1
    加权：  (3·2.1 + 2·3.1 + 3·2.1 + 2·3.1) / 10 = 25.0/10 = 2.5
    不加权： (2.1 + 3.1 + 2.1 + 3.1) / 4         = 10.4/4  = 2.6
    clamp 退化到 (2.0, 2.0)（所有 w 相同）时又必须回到 2.6 —— 这一条把"权值确实
    来自每个 pair 的 |Δy|"和"分母是 Σw 而不是 pair 数"同时钉住：忘了归一化会得到
    25.0，忽略权值会得到 2.6，用 1/|Δy| 反加权会得到 > 2.6。
    """
    mu, log_var, y = _pair_inputs([0.0, 1.0, 3.0], [0.0, 1.0, -2.0])
    common = dict(rank_weight=1.0, rank_margin=0.1, rank_pairs=256, rank_min_gap=2.0)

    _, logs_plain = affinity_loss(mu, log_var, y, **common)
    _, logs_weighted = affinity_loss(mu, log_var, y, rank_gap_weight=True, **common)
    _, logs_flat = affinity_loss(
        mu, log_var, y, rank_gap_weight=True, rank_gap_clamp=(2.0, 2.0), **common
    )

    assert float(logs_weighted["rank_pairs"]) == 4.0
    assert abs(float(logs_weighted["rank_gap_mean"]) - 2.5) < 1e-6
    assert abs(float(logs_plain["rank"]) - 2.6) < 1e-6
    assert abs(float(logs_weighted["rank"]) - 2.5) < 1e-6
    assert abs(float(logs_flat["rank"]) - 2.6) < 1e-6


def test_gap_clamp_bounds_the_weight_at_both_ends():
    """clamp 的两端都必须真的生效：上界 5.0 夹住大间隔，下界 0.1 抬起近邻对。

    上界：y=[0.0, 1.0, 9.0]、δ_min=2.0、mu=[0.0, 3.0, -2.0]、margin=0.1 →
    有效有序对 4 个，|Δy| = 9, 8, 9, 8，hinge = 2.1, 5.1, 2.1, 5.1。
        clamp (0.1, 5.0)   → w 全为 5   → (2.1+5.1+2.1+5.1)/4 = 14.4/4 = 3.6
        clamp (0.1, 100.0) → w = 9, 8   → (9·2.1 + 8·5.1)·2/34 = 119.4/34
    下界：y=[0.0, 0.05, 3.0]、δ_min=0.0、同一 mu/margin → 6 个有序对，
    |Δy| = 0.05,0.05,3,3,2.95,2.95，hinge = 0,0,2.1,2.1,3.1,3.1。
        clamp lo=0.1 → Σw = 12.1，Σw·hinge = 30.89 → 2.55289256…
        clamp lo=0.0 → Σw = 12.0，Σw·hinge = 30.89 → 2.57416666…
    """
    mu_hi, log_var_hi, y_hi = _pair_inputs([0.0, 1.0, 9.0], [0.0, 3.0, -2.0])
    _, logs_hi = affinity_loss(
        mu_hi, log_var_hi, y_hi, rank_weight=1.0, rank_margin=0.1, rank_pairs=256,
        rank_min_gap=2.0, rank_gap_weight=True, rank_gap_clamp=(0.1, 5.0),
    )
    _, logs_uncapped = affinity_loss(
        mu_hi, log_var_hi, y_hi, rank_weight=1.0, rank_margin=0.1, rank_pairs=256,
        rank_min_gap=2.0, rank_gap_weight=True, rank_gap_clamp=(0.1, 100.0),
    )
    assert float(logs_hi["rank_pairs"]) == 4.0
    assert abs(float(logs_hi["rank"]) - 3.6) < 1e-5
    assert abs(float(logs_uncapped["rank"]) - 119.4 / 34.0) < 1e-5

    mu_lo, log_var_lo, y_lo = _pair_inputs([0.0, 0.05, 3.0], [0.0, 1.0, -2.0])
    _, logs_lo = affinity_loss(
        mu_lo, log_var_lo, y_lo, rank_weight=1.0, rank_margin=0.1, rank_pairs=256,
        rank_min_gap=0.0, rank_gap_weight=True, rank_gap_clamp=(0.1, 5.0),
    )
    _, logs_lo_off = affinity_loss(
        mu_lo, log_var_lo, y_lo, rank_weight=1.0, rank_margin=0.1, rank_pairs=256,
        rank_min_gap=0.0, rank_gap_weight=True, rank_gap_clamp=(0.0, 5.0),
    )
    assert float(logs_lo["rank_pairs"]) == 6.0
    assert abs(float(logs_lo["rank"]) - 30.89 / 12.1) < 1e-5
    assert abs(float(logs_lo_off["rank"]) - 30.89 / 12.0) < 1e-5


def test_gradient_flows_through_the_rank_term_when_it_is_active():
    """rank 项必须真的可训练，而不是只把日志填成非零。

    y=[0, 3]、mu 初值 = y（回归项与 NLL 对 mu 的梯度恰好为 0），margin=5.0 >
    |Δy|=3 → 两个有序对 (0,1)/(1,0) 的 hinge 都等于 relu(5 + mu0 - mu1) = 2.0，
    rank = 2.0，∂rank/∂mu = [2·(1/2), 2·(-1/2)] = [1.0, -1.0]。
    rank_weight=0 时 mu 的梯度必须精确为 0（证明上面那 1.0 只能来自 rank 项）。
    """
    mu_off = torch.tensor([0.0, 3.0], requires_grad=True)
    log_var = torch.zeros(2, dtype=torch.float32)
    y = torch.tensor([0.0, 3.0], dtype=torch.float32)
    affinity_loss(mu_off, log_var, y, rank_weight=0.0, nll_weight=0.0,
                  rank_min_gap=0.0)[0].backward()
    assert float(mu_off.grad.abs().sum()) == 0.0

    mu_on = torch.tensor([0.0, 3.0], requires_grad=True)
    total, logs = affinity_loss(mu_on, log_var, y, rank_weight=1.0, rank_margin=5.0,
                                nll_weight=0.0, rank_pairs=2, rank_min_gap=0.0)
    total.backward()

    assert float(logs["rank_pairs"]) == 2.0
    assert abs(float(logs["rank"]) - 2.0) < 1e-6
    assert mu_on.grad is not None and torch.isfinite(mu_on.grad).all()
    assert torch.allclose(mu_on.grad, torch.tensor([1.0, -1.0]), atol=1e-6)


def test_invalid_rank_gap_parameters_are_refused():
    """clamp 反向 / δ_min 为负必须报错，而不是静默给出一个假的 rank 项。"""
    mu, log_var, y = _inputs()
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, rank_weight=1.0, rank_gap_clamp=(5.0, 0.1))
    with pytest.raises(ValueError):
        affinity_loss(mu, log_var, y, rank_weight=1.0, rank_min_gap=-1.0)


def test_rank_min_gap_in_force_is_logged_in_label_units():
    """生效的 δ_min 必须出现在日志里，而且是**损失实际使用的标签单位**（z 单位）。

    只看 affinity_rank_pairs 分不清"是 δ 太大"还是"这个 batch 全是近邻对"；
    rank_min_gap_z 把"当前按哪个间隔在过滤"直接摆出来。三条语义：
    - rank 开着且一对都没留下时**照实记 δ**（那正是最需要看到 δ 的时候，记 0 会掩盖
      真正的原因）；
    - rank 关掉时没有过滤器 → 0.0；
    - 早退路径（没有有效标签）也必须带这个键，否则 sink 的 key 集合会漂。
    """
    mu, log_var, y = _pair_inputs([0.0, 0.4, 0.9, 0.2], [1.0, -1.0, 1.0, -1.0])

    _, logs_degenerate = affinity_loss(mu, log_var, y, rank_weight=1.0,
                                       rank_min_gap=1.0, rank_pairs=256)
    assert float(logs_degenerate["rank_pairs"]) == 0.0
    assert torch.is_tensor(logs_degenerate["rank_min_gap_z"])
    assert logs_degenerate["rank_min_gap_z"].ndim == 0
    assert logs_degenerate["rank_min_gap_z"].dtype == torch.float32
    assert abs(float(logs_degenerate["rank_min_gap_z"]) - 1.0) < 1e-6

    _, logs_off = affinity_loss(mu, log_var, y, rank_weight=0.0, rank_min_gap=1.0)
    assert float(logs_off["rank_min_gap_z"]) == 0.0
    assert float(logs_off["rank_pairs"]) == 0.0

    _, logs_default = affinity_loss(mu, log_var, y)
    assert float(logs_default["rank_min_gap_z"]) == 0.0

    nan_labels = torch.full((4,), float("nan"), dtype=torch.float32)
    _, logs_none = affinity_loss(mu, log_var, nan_labels, rank_weight=1.0,
                                 rank_min_gap=0.588)
    assert abs(float(logs_none["rank_min_gap_z"]) - 0.588) < 1e-6


def test_protaff_preset_constants_are_the_paper_values_and_usable_as_arguments():
    """论文取值只写一处：δ_min=1.0、gap clamp=(0.1, 5.0)。

    这条是"预设常量"的钉子（不是新行为的钉子）：常量漂移、或常量不再能直接当参数
    用时失败。跑一遍 y=[0,1,3]、mu=[0,1,-2]、margin=0.1：δ_min=1.0 时 6 个有序对
    全在（|Δy|=1.0 的那对正好落在边界上，按"低于 δ_min 才排除"保留），hinge =
    0,0,2.1,2.1,3.1,3.1，权重 = 1,1,3,3,2,2 → Σw=12、Σw·hinge=25 → rank = 25/12，
    rank_pairs = 6。（δ_min=2.0 的那个手算例子见 gap 加权那条测试，两个 δ 下的
    pair 数 6 与 4 正是"边界是否保留"的差别。）
    """
    from models.affinity_loss import PROTAFF_RANK_GAP_CLAMP, PROTAFF_RANK_MIN_GAP

    assert PROTAFF_RANK_MIN_GAP == 1.0
    assert PROTAFF_RANK_GAP_CLAMP == (0.1, 5.0)

    mu, log_var, y = _pair_inputs([0.0, 1.0, 3.0], [0.0, 1.0, -2.0])
    _, logs = affinity_loss(
        mu, log_var, y, rank_weight=1.0, rank_margin=0.1, rank_pairs=256,
        rank_min_gap=PROTAFF_RANK_MIN_GAP,
        rank_gap_weight=True,
        rank_gap_clamp=PROTAFF_RANK_GAP_CLAMP,
    )
    assert float(logs["rank_pairs"]) == 6.0
    assert abs(float(logs["rank"]) - 25.0 / 12.0) < 1e-6
