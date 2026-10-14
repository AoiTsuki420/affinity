"""val_epoch_with_ema 备份范围的回归测试（集群 smoke 2026-10-14 实测）。

原实现把整份 state_dict（含 6.5B 冻结骨干）clone 到 GPU 做备份（+24GB），把验证
步抬到 ~60GB 触顶；而 ema.copy_to 只会写 ema.shadow 里的参数（即全部可训练参数），
其余参数在验证期间不会被动过，无需备份。
"""
import pytest
import torch
import torch.nn as nn

import main_train_predictor_multigpu as M


def _tiny_model() -> nn.Module:
    model = nn.Module()
    model.esm = nn.Module()
    model.esm.backbone = nn.Module()
    model.esm.backbone.weight = nn.Parameter(torch.full((2, 2), 1.0), requires_grad=False)
    model.head = nn.Linear(2, 1)
    return model


def test_ema_backup_covers_only_shadow_params():
    """备份必须精确等于 ema.shadow 覆盖的参数集合，而不是全部 state_dict。"""
    model = _tiny_model()
    ema = M.EMA(model)
    assert set(ema.shadow) == {"head.weight", "head.bias"}

    backup = M._ema_backup_params(model, ema)
    assert set(backup) == {"head.weight", "head.bias"}


def test_val_epoch_with_ema_uses_ema_then_restores(monkeypatch):
    model = _tiny_model()
    ema = M.EMA(model)
    with torch.no_grad():
        ema.shadow["head.weight"].fill_(5.0)
        ema.shadow["head.bias"].fill_(-3.0)

    before_head_w = dict(model.named_parameters())["head.weight"].detach().clone()
    seen = {}

    def fake_val_epoch(model, val_loader, ep, args, final_contact_store=None):
        params = dict(model.named_parameters())
        seen["head_w"] = float(params["head.weight"][0, 0])
        seen["esm_w"] = float(params["esm.backbone.weight"][0, 0])
        return 0.0

    monkeypatch.setattr(M, "val_epoch", fake_val_epoch)
    M.val_epoch_with_ema(model, ema, None, 0, None)

    assert seen["head_w"] == 5.0, "验证期间应使用 EMA 权重"
    assert seen["esm_w"] == 1.0, "冻结骨干参数全程不应被触碰"

    params = dict(model.named_parameters())
    assert torch.equal(params["head.weight"].detach(), before_head_w), "验证后必须还原原权重"
