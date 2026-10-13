import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch.nn as nn

from tools.probe_backbone_modules import describe_backbone

REPO_ROOT = Path(__file__).resolve().parents[1]

# 子进程（离线 transformers 替身）用它把真实的 sys.path 回传给测试
_SYS_PATH_MARKER = "PROBE_TEST_SYSPATH="

# 离线 transformers 替身：只替换 _load 的网络依赖（HuggingFace），
# 不触碰被测的 sys.path 条件；探针必须能像在集群上一样跑完 describe_backbone。
_STUB_TRANSFORMERS_SOURCE = '''\
import json
import sys

import torch.nn as nn

print("PROBE_TEST_SYSPATH=" + json.dumps(list(sys.path)))


class _Cfg:
    def __init__(self):
        self.hidden_size = 768
        self.num_hidden_layers = 2


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(8, 8)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(8, 8)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Layer() for _ in range(2)])


class AutoConfig:
    @staticmethod
    def from_pretrained(name, **kwargs):
        return _Cfg()


class AutoModel:
    @staticmethod
    def from_pretrained(name, **kwargs):
        return _Model()


class AutoModelForMaskedLM(AutoModel):
    pass
'''


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


@pytest.fixture()
def probe_scratch():
    """仓库内的临时目录。

    本机沙箱下 pytest 的 tmp_path 与 tempfile.mkdtemp 都不可用：它们以 0o700 建目录，
    受限进程随后无法写入或列举该目录（PermissionError）。这里用默认权限建目录，
    并复用 .gitignore 已覆盖的 pytest-cache-files-*/ 前缀，清理失败也不会弄脏工作树。
    """
    path = REPO_ROOT / f"pytest-cache-files-probe-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_direct_script_invocation_puts_repo_root_on_sys_path(probe_scratch):
    """回归：按文件路径直接运行探针时，仓库根必须出现在子进程的 sys.path 上。

    启动方式 `python tools/probe_backbone_modules.py ...` 会把 sys.path[0] 设为
    tools/，仓库根不在 sys.path 上；此时 describe_backbone 内的
    `from models.plm_backbones import ...` 以 "No module named 'models'" 失败，
    hidden_size / num_layers 静默变成 null —— 而集群验收正是检查这两个字段
    （ESMC-6B: hidden_size == 2560, num_layers == 80）。

    本测试用真实子进程复现该启动方式，读的是子进程真实的 sys.path，不是对
    sys.path 的模拟；只有网络依赖 transformers 被换成了离线替身。
    """
    stub_dir = probe_scratch / "stubs"
    stub_dir.mkdir()
    (stub_dir / "transformers.py").write_text(_STUB_TRANSFORMERS_SOURCE, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_dir)  # 只留替身，避免外层 PYTHONPATH 掩盖缺陷
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable, "tools/probe_backbone_modules.py",
            "--model", "stub/local", "--kind", "esm2", "--no-trust-remote-code",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, f"探针未正常退出:\n{result.stdout}\n{result.stderr}"

    lines = result.stdout.splitlines()
    reported = [line for line in lines if line.startswith(_SYS_PATH_MARKER)]
    assert len(reported) == 1, f"未拿到子进程回传的 sys.path:\n{result.stdout}"
    child_sys_path = json.loads(reported[0][len(_SYS_PATH_MARKER):])
    normalized = {os.path.normcase(os.path.abspath(entry)) for entry in child_sys_path if entry}
    assert os.path.normcase(str(REPO_ROOT)) in normalized, (
        f"仓库根不在子进程 sys.path 上，models 包无法解析: {child_sys_path}"
    )

    json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
    info, _ = json.JSONDecoder().raw_decode("\n".join(lines[json_start:]))
    sample = "\n".join(info["layer_names_sample"])
    # 断言串带右引号：Task 2 落地前退化为 "No module named 'models.plm_backbones'"
    # 是预期结果，不应命中这里；命中的只能是 'models 包整个解析不到'。
    assert "No module named 'models'" not in sample, (
        f"describe_backbone 仍以 'models 包不存在' 失败，"
        f"hidden_size={info['hidden_size']!r}, num_layers={info['num_layers']!r}:\n{sample}"
    )
