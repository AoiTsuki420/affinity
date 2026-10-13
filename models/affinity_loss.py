"""单任务亲和力回归损失。

由 models/kcat_expert.py:276-332 的 kcat_expert_loss 泛化而来：该函数本就是单任务形式
（Huber + ranking + var_reg + NLL），只是命名与 kcat 绑定。这里去掉 CCC 项，
补上 Barron 稳健损失选项（对齐 main_train_predictor_multigpu.py:247-261）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0


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
    var_reg_weight: float = 0.0,
    var_reg_mode: str = "abs",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (总损失, 分项日志)。所有分项均为标量张量，便于直接写日志。"""
    if loss_type not in ("huber", "barron"):
        raise ValueError(f"未知 loss_type: {loss_type!r}（支持 huber / barron）")
    if var_reg_mode not in ("abs", "ratio"):
        raise ValueError(f"未知 var_reg_mode: {var_reg_mode!r}（支持 abs / ratio）")

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
        logs.update({"huber": zero, "barron": zero, "rank": zero,
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
    n_valid = int(y_v.numel())
    if rank_weight > 0.0 and n_valid >= 2:
        n_pairs = min(int(rank_pairs), n_valid * (n_valid - 1))
        if n_pairs > 0:
            i = torch.randint(0, n_valid, (n_pairs,), device=device)
            j = torch.randint(0, n_valid, (n_pairs,), device=device)
            keep = i != j
            i, j = i[keep], j[keep]
            sign = torch.sign(y_v[i] - y_v[j])
            nonzero = sign != 0
            if nonzero.any():
                rank = F.margin_ranking_loss(
                    mu_v[i[nonzero]], mu_v[j[nonzero]], sign[nonzero],
                    margin=float(rank_margin),
                )
    logs["rank"] = rank.detach()

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
        + float(rank_weight) * rank
        + float(var_reg_weight) * var_reg
    )
    logs["total"] = total.detach()
    return total, logs
