import torch.nn as nn

from tools.probe_backbone_modules import describe_backbone


class _Cfg:
    # 真实 HF config（_load 的返回值）把字段放在实例 __dict__ 上，探针据此用 vars(config)
    # 收集 config_fields；这里必须同样用实例属性，否则 vars() 返回空 dict，与真实对象不符。
    def __init__(self):
        self.hidden_size = 768
        self.num_hidden_layers = 2


def _fake_bert_like():
    """构造一个 BERT 命名风格的极小模型，用于离线验证探针输出格式。"""
    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.key = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = _SelfAttn()
            self.output = nn.Module()
            self.output.dense = nn.Linear(8, 8)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _Attn()

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([_Layer() for _ in range(2)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    return _Model()


def test_describe_backbone_reports_hidden_and_layers():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    # 契约：brief 规定的 6 个键必须齐全（Task 2/4 依赖这些键名）
    assert set(info) == {
        "hidden_size", "num_layers", "linear_names",
        "leaf_names", "config_fields", "layer_names_sample",
    }
    # hidden_size/num_layers 的取值由 models/plm_backbones.py 的 resolve_* 决定；
    # 该模块由 Task 2 创建，此前探针按 brief 设计把失败原因写进 layer_names_sample
    # 而不是中断。等值断言（768 / 2）属于 T1×T2 集成测试，落在 Task 2 的测试文件里。
    if info["hidden_size"] is None:
        assert any("resolve failed" in item for item in info["layer_names_sample"]), (
            "resolve 失败时必须留下失败原因，不能静默返回 None"
        )
    else:
        assert info["hidden_size"] == 768
        assert info["num_layers"] == 2


def test_describe_backbone_lists_only_linear_modules():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    names = info["linear_names"]
    assert "encoder.layer.0.attention.self.query" in names
    assert "encoder.layer.1.attention.output.dense" in names
    # 非 Linear 模块不得出现
    assert "encoder.layer.0.attention.output" not in names


def test_describe_backbone_reports_leaf_names_sorted_unique():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    assert info["leaf_names"] == sorted(set(info["leaf_names"]))
    assert "query" in info["leaf_names"]


def test_describe_backbone_reports_config_fields():
    info = describe_backbone(_fake_bert_like(), _Cfg())
    assert "hidden_size" in info["config_fields"]
    assert "num_hidden_layers" in info["config_fields"]
