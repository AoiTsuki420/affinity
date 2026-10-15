"""单任务亲和力回归损失。

由 models/kcat_expert.py:276-332 的 kcat_expert_loss 泛化而来：该函数本就是单任务形式
（Huber + ranking + var_reg + NLL），只是命名与 kcat 绑定。这里去掉 CCC 项，
补上 Barron 稳健损失选项（对齐 main_train_predictor_multigpu.py:247-261）。

排序项可选地采用 ProtAff（Chu, Vogt, Chungyoun, Gray; bioRxiv 2026,
doi 10.64898/2026.06.13.732058）的 margin-ranking 配方：**最小间隔** δ_min 只留
|Δy| >= δ_min 的 pair（默认关，δ_min=0.0 与旧行为一致），以及按 |Δy| 加权的
可选变体（权重 clamp 到 [0.1, 5.0]）。论文用 δ_min=1.0（10 倍 K_D 差）、
margin=0.1、gap 加权。

注意：本函数的 ``rank_min_gap`` 与 ``y`` **同单位**。本仓库里 y 是 AffinityScaler
标准化后的 z 值，而 CLI 的 ``--affinity_rank_min_gap`` 以 **log10(K)** 计（ProtAff 的
口径，1.0 = 10 倍 K_D 差）。换算在调用点完成：
``main_train_predictor_multigpu.resolve_affinity_rank_min_gap()`` 除以 scaler.scale_；
拿不到 scale_ 时**报错，不按字面值猜**——猜错会让过滤过严、rank 项长期贴近 0，把一次
4-GPU 实验误读成"排序没用"。实际生效的 δ_min 会以 ``rank_min_gap_z`` 写进日志。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0

# ProtAff 的取值（论文 §Methods）。默认不改行为，需要显式开启：
#   --affinity_rank_min_gap 1.0 --affinity_rank_gap_weight --affinity_rank_margin 0.1
PROTAFF_RANK_MIN_GAP = 1.0
PROTAFF_RANK_GAP_CLAMP = (0.1, 5.0)

# 退化的 rank 项（batch 太小 / 间隔太严 / 根本没开）会静默变成 0.0，而 run 看上去
# 仍然"在用 ranking 训练"。这里把它写进日志：计数器放在模块级（一个 epoch 有上万次
# 调用，不能每次都打印），第一次和之后每 _DEGENERATE_RANK_LOG_EVERY 次各打一行。
_DEGENERATE_RANK_COUNTS: dict[str, int] = {}
_DEGENERATE_RANK_LOG_EVERY = 500
_RANK_LOG_PREFIX = "[affinity_loss] rank term is 0"
_RANK_CEILING_PREFIX = "[affinity_loss] rank_pairs ceiling"


def _distributed_rank() -> int:
    """DDP 时只让 rank 0 打印，避免 N 个 rank 刷同一行。"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _log_rank_once(reason: str, message: str) -> None:
    """按 reason 限流地打印一条 rank 相关提示（rank 0 才打，首次 + 每 N 次）。"""
    if _distributed_rank() != 0:
        return
    count = _DEGENERATE_RANK_COUNTS.get(reason, 0) + 1
    _DEGENERATE_RANK_COUNTS[reason] = count
    if count == 1 or count % _DEGENERATE_RANK_LOG_EVERY == 0:
        print(f"{message} [occurrence {count}]")


def _note_rank_zero(reason: str) -> None:
    """记录"这一批的 rank 项是 0"以及原因。"""
    _log_rank_once(
        reason,
        f"{_RANK_LOG_PREFIX} ({reason}); logged rank_pairs=0. If this run is "
        f"meant to train with the ranking loss, raise -batch_size (pairs are "
        f"drawn inside one microbatch, so n_valid*(n_valid-1) is the hard "
        f"ceiling) or lower --affinity_rank_min_gap.",
    )


def _note_rank_ceiling(n_valid: int, n_pairs: int, requested: int) -> None:
    """请求的 pair 数被 microbatch 天花板截断时提示一次。

    这不是错误（rank_pairs > 0），但 batch=4 这种情形下 12 对里一半还是彼此反向的
    重复对，排序项几乎不携带信号：不说出来的话，run 看不出任何异常。
    """
    reason = (f"requested={requested}, n_valid={n_valid} -> {n_pairs} "
              f"(hard ceiling n_valid*(n_valid-1))")
    _log_rank_once(
        reason,
        f"{_RANK_CEILING_PREFIX} reached: {reason}; the ranking term only sees "
        f"{n_pairs} ordered pairs per microbatch. Raise -batch_size for the "
        f"ranking arm of the experiment (-iters_to_accumulate does NOT help: "
        f"pairs are built inside one microbatch).",
    )


def _validate_rank_gap_params(
    rank_min_gap: float, rank_gap_clamp
) -> tuple[float, tuple[float, float]]:
    """把两个新参数校验并规范化成 (δ_min, (lo, hi))。"""
    try:
        gap_lo, gap_hi = float(rank_gap_clamp[0]), float(rank_gap_clamp[1])
    except (TypeError, IndexError) as exc:
        raise ValueError(
            f"rank_gap_clamp 需要 (lo, hi) 两个浮点数，收到 {rank_gap_clamp!r}"
        ) from exc
    if not math.isfinite(gap_lo) or not math.isfinite(gap_hi) or gap_lo < 0.0 \
            or gap_hi < gap_lo:
        raise ValueError(
            f"rank_gap_clamp 需要 0 <= lo <= hi 的有限值，收到 "
            f"({gap_lo}, {gap_hi})"
        )
    min_gap = float(rank_min_gap)
    if not math.isfinite(min_gap) or min_gap < 0.0:
        raise ValueError(f"rank_min_gap 需要 >= 0 的有限值，收到 {rank_min_gap!r}")
    return min_gap, (gap_lo, gap_hi)


def _rank_pair_indices(y_v: torch.Tensor, n_pairs: int, min_gap: float):
    """按 ProtAff 的 anchored 方式抽 (anchor, partner) 下标。

    与旧实现（两个独立随机下标 + i!=j）的差别：
      * anchor 轮流覆盖整个有效池（c % n_valid），所以只要 rank_pairs >= n_valid，
        每个样本至少当一次 anchor —— batch_size 很小时这一点决定有没有信号；
      * partner 只从该 anchor 的**间隔窗口**（|Δy| >= min_gap，且不含自身与并列值）
        里取，窗口内**无放回**：rank_pairs >= n_valid*(n_valid-1) 时正好枚举全部
        有效有序对，不再像旧实现那样"抽 12 次但有放回"。
    返回 (i, j, reason)：reason 非 None 表示没有可用 pair，rank 项必须恰好为 0。
    """
    n_valid = int(y_v.numel())
    if n_pairs <= 0:
        return None, None, f"rank_pairs={n_pairs} -> no pair requested"
    # 每个 anchor 的候选数（行=第几个 partner，列=anchor）。
    rows = (n_pairs + n_valid - 1) // n_valid
    device = y_v.device
    anchor_grid = torch.arange(n_valid, device=device).unsqueeze(0).expand(rows, n_valid)
    slot_grid = torch.arange(rows, device=device).unsqueeze(1).expand(rows, n_valid)
    stair = slot_grid * n_valid + anchor_grid < n_pairs

    # 每个 anchor 的窗口边界：排序后 [0, left) ∪ [right, n_valid)。
    # min_gap=0 时要**严格**排除并列值（零间隔对），min_gap>0 时边界值 |Δy|==min_gap
    # 属于合法 pair（"|Δy| 低于 min_gap 才排除"）。
    order = torch.argsort(y_v)
    ys = y_v[order]
    zero_gap = min_gap <= 0.0
    left = torch.searchsorted(ys, y_v - min_gap, right=not zero_gap)
    right = torch.searchsorted(ys, y_v + min_gap, right=zero_gap)
    window = left + (n_valid - right)

    # 无放回：给候选一个随机 key，同一 anchor 内的 key 名次就是窗口内的位置；
    # 名次 >= 窗口大小 的候选丢弃。padding 候选的 key 置 +inf，否则它们会偷走名次。
    keys = torch.rand(rows, n_valid, device=device, dtype=torch.float32)
    keys = keys.masked_fill(~stair, float("inf"))
    by_key = torch.argsort(keys, dim=0)
    rank_in_anchor = torch.empty_like(by_key)
    rank_in_anchor.scatter_(0, by_key, slot_grid)

    keep = stair & (rank_in_anchor < window.unsqueeze(0))
    if not bool(keep.any()):
        return None, None, (
            f"no pair satisfies |dy| >= rank_min_gap={min_gap} "
            f"(n_valid={n_valid}, rank_pairs={n_pairs})"
        )

    anchor = anchor_grid[keep]
    ordinal = rank_in_anchor[keep]
    left_k = left[anchor]
    # 窗口位置 -> 排序后位置：左半段原样，右半段整体右移 right-left。
    pos = torch.where(ordinal < left_k, ordinal, right[anchor] + ordinal - left_k)
    return anchor, order[pos], None


def _barron_loss(error: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
    """Barron 稳健损失（CVPR 2019），alpha=2 退化为 L2，alpha=0 近似 L1。

    逐行对应 main_train_predictor_multigpu.py:247-260 的 _barron_loss_per_sample：
    分母先取 max(scale, 1e-8)，alpha=2 / alpha=0 两个特例单独处理，其余情况用
    (beta/alpha) * ((r2/beta + 1)^(alpha/2) - 1)，其中 beta = |alpha - 2|。
    """
    c = max(float(scale), 1e-8)
    a = float(alpha)
    r2 = (error / c) ** 2
    if abs(a - 2.0) < 1e-6:
        return 0.5 * r2
    if abs(a) < 1e-6:
        return torch.log1p(0.5 * r2)
    beta = abs(a - 2.0)
    return (beta / a) * (torch.pow(r2 / beta + 1.0, a / 2.0) - 1.0)


def affinity_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_beta: float = 1.0,
    barron_alpha: float = 1.0,
    barron_scale: float = 1.0,
    nll_weight: float = 0.05,
    rank_weight: float = 0.0,
    rank_pairs: int = 256,
    rank_margin: float = 0.0,
    rank_min_gap: float = 0.0,
    rank_gap_weight: bool = False,
    rank_gap_clamp: tuple[float, float] = PROTAFF_RANK_GAP_CLAMP,
    var_reg_weight: float = 0.0,
    var_reg_mode: str = "abs",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (总损失, 分项日志)。所有分项均为标量张量，便于直接写日志。

    rank 项（默认关闭）：
      * ``rank_min_gap``（ProtAff 的 δ_min）：只保留 ``|Δy| >= rank_min_gap`` 的
        pair；``0.0`` 表示不过滤（只排除零间隔对），与旧行为一致。**单位与 ``y``
        相同**（本仓库是标准化标签的 z 单位）；CLI 的 log10(K) → z 换算在调用点做，
        见模块 docstring。
      * ``rank_gap_weight``：为真时每个 pair 按 ``clamp(|Δy|, lo, hi)`` 加权，
        归一化用 Σw（加权平均，不是加权求和），所以量级不随 pair 数漂移。
      * ``rank_gap_clamp``：加权变体的 clamp 上下界（ProtAff 取 (0.1, 5.0)，
        作用在与 ``y`` 同单位的 |Δy| 上）。
    无论哪条路径，``rank_weight=0.0`` 都让 rank 项精确为 0。

    另返回三个诊断项（与 rank 项同生命周期）：
      * ``rank_pairs``：真正进入 rank 项的 pair 数。上限是 microbatch 内的
        ``n_valid*(n_valid-1)``，所以它把"batch 太小 → 排序项没信号"这件事直接
        摆到 swanlab 上。
      * ``rank_gap_mean``：这些 pair 的 |Δy| 均值，用来看间隔过滤是否真的在选
        有意义的分隔，而不是被近邻对淹没。
      * ``rank_min_gap_z``：**实际生效**的 δ_min（与 ``y`` 同单位）。ranking 开着时
        即使一对都没留下也照实记录（那正是需要看清 δ 的时候）；ranking 关掉时没有
        过滤器，记 0.0。运维靠它确认 CLI 的 log10 值被换算成了什么。
    """
    if loss_type not in ("huber", "barron"):
        raise ValueError(f"未知 loss_type: {loss_type!r}（支持 huber / barron）")
    if var_reg_mode not in ("abs", "ratio"):
        raise ValueError(f"未知 var_reg_mode: {var_reg_mode!r}（支持 abs / ratio）")
    min_gap, gap_clamp = _validate_rank_gap_params(rank_min_gap, rank_gap_clamp)
    rank_weight = float(rank_weight)
    # 生效中的 δ_min（与 y 同单位）；ranking 关掉 ⇒ 没有过滤器 ⇒ 0.0。
    min_gap_z = torch.as_tensor(
        min_gap if rank_weight > 0.0 else 0.0, device=mu.device, dtype=torch.float32)

    mu = torch.nan_to_num(mu.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_var = torch.nan_to_num(
        log_var.float(), nan=0.0, posinf=_LOG_VAR_MAX, neginf=_LOG_VAR_MIN
    ).clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
    y = torch.nan_to_num(y.float(), nan=float("nan"), posinf=float("nan"),
                         neginf=float("nan"))

    valid = torch.isfinite(y)
    device = mu.device
    logs: dict[str, torch.Tensor] = {}

    if not valid.any():
        zero = torch.zeros((), device=device, dtype=mu.dtype)
        _note_rank_zero(
            f"rank_weight={rank_weight} -> ranking disabled"
            if rank_weight <= 0.0
            else "no finite label in this microbatch (n_valid=0)"
        )
        logs.update({"huber": zero, "barron": zero, "rank": zero,
                     "rank_pairs": zero, "rank_gap_mean": zero,
                     "rank_min_gap_z": min_gap_z,
                     "var_reg": zero, "nll": zero, "total": zero})
        return zero, logs

    mu_v = mu[valid]
    y_v = y[valid]
    log_var_v = log_var[valid]

    if loss_type == "barron":
        regression = _barron_loss(mu_v - y_v, barron_alpha, barron_scale).mean()
        logs["barron"] = regression.detach()
        logs["huber"] = regression.detach()
    else:
        regression = F.smooth_l1_loss(mu_v, y_v, beta=huber_beta)
        logs["huber"] = regression.detach()
        logs["barron"] = torch.zeros((), device=device, dtype=mu.dtype)

    nll = 0.5 * (log_var_v + (y_v - mu_v) ** 2 / log_var_v.exp().clamp_min(1e-6))
    nll = nll.mean()
    logs["nll"] = nll.detach()

    rank = torch.zeros((), device=device, dtype=mu.dtype)
    rank_pairs_log = torch.zeros((), device=device, dtype=mu.dtype)
    rank_gap_mean_log = torch.zeros((), device=device, dtype=mu.dtype)
    n_valid = int(y_v.numel())
    if rank_weight <= 0.0:
        # 明确写出"排序项被关掉了"：否则一个忘了传 --affinity_rank_weight 的
        # "排序变体" run 会以 rank=0 / rank_pairs=0 静默跑完。
        _note_rank_zero(f"rank_weight={rank_weight} -> ranking disabled")
    elif n_valid < 2:
        _note_rank_zero(f"fewer than 2 valid labels (n_valid={n_valid})")
    else:
        requested_pairs = int(rank_pairs)
        n_pairs = min(requested_pairs, n_valid * (n_valid - 1))
        if n_pairs < requested_pairs:
            # 天花板被撞到：不是错误，但 batch 太小时排序项几乎没有信号。
            _note_rank_ceiling(n_valid, n_pairs, requested_pairs)
        i_idx, j_idx, reason = _rank_pair_indices(y_v, n_pairs, min_gap)
        if reason is not None:
            _note_rank_zero(reason)
        else:
            delta = (y_v[i_idx] - y_v[j_idx]).abs()
            # 窗口构造保证 sign != 0：自身与并列值都已被排除。
            sign = torch.sign(y_v[i_idx] - y_v[j_idx])
            pair_loss = F.margin_ranking_loss(
                mu_v[i_idx], mu_v[j_idx], sign,
                margin=float(rank_margin), reduction="none",
            )
            if rank_gap_weight:
                weights = delta.clamp(gap_clamp[0], gap_clamp[1])
                rank = (weights * pair_loss).sum() / weights.sum().clamp_min(1e-12)
            else:
                rank = pair_loss.mean()
            rank_pairs_log = torch.as_tensor(
                float(pair_loss.numel()), device=device, dtype=mu.dtype)
            rank_gap_mean_log = delta.mean().detach()
    logs["rank"] = rank.detach()
    logs["rank_pairs"] = rank_pairs_log
    logs["rank_gap_mean"] = rank_gap_mean_log
    logs["rank_min_gap_z"] = min_gap_z

    var_reg = torch.zeros((), device=device, dtype=mu.dtype)
    if var_reg_weight > 0.0 and n_valid >= 2:
        std_pred = mu_v.std(unbiased=False)
        std_true = y_v.std(unbiased=False)
        if var_reg_mode == "ratio":
            var_reg = torch.relu(std_true - std_pred)
        else:
            var_reg = (std_pred - std_true).abs()
    logs["var_reg"] = var_reg.detach()

    total = (
        regression
        + float(nll_weight) * nll
        + rank_weight * rank
        + float(var_reg_weight) * var_reg
    )
    logs["total"] = total.detach()
    return total, logs
