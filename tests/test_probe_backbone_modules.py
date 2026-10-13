import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

from tools.probe_backbone_modules import describe_backbone

REPO_ROOT = Path(__file__).resolve().parents[1]

# 子进程（离线 transformers 替身）用它把真实的 sys.path 回传给测试
_SYS_PATH_MARKER = "PROBE_TEST_SYSPATH="

# 离线 transformers 替身：只替换 _load 的网络依赖（HuggingFace），
# 不触碰被测的 sys.path 条件；探针必须能像在集群上一样跑完 describe_backbone。
# __CONFIG_ASSIGNMENTS__ 由 _stub_source() 注入，用来切换 config 是否可解析。
_STUB_TRANSFORMERS_TEMPLATE = '''\
import json
import sys

import torch.nn as nn

print("PROBE_TEST_SYSPATH=" + json.dumps(list(sys.path)))


class _Cfg:
    def __init__(self):
__CONFIG_ASSIGNMENTS__


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

# 真实 HF config 的字段名：resolve_hidden_size / resolve_num_layers 应当直接命中。
_STUB_CONFIG_RESOLVABLE = """\
        self.hidden_size = 768
        self.num_hidden_layers = 2"""

# 不含任何已知 hidden size 字段：resolve_hidden_size 必然抛错，describe_backbone
# 只能把 hidden_size / num_layers 置为 None 并把原因写进 layer_names_sample。
_STUB_CONFIG_UNRESOLVABLE = """\
        self.unrecognised_field = 1"""


def _stub_source(config_assignments: str) -> str:
    return _STUB_TRANSFORMERS_TEMPLATE.replace("__CONFIG_ASSIGNMENTS__", config_assignments)


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
    # 取值由 models/plm_backbones.py 的 resolve_* 决定。配置里就写着
    # hidden_size=768 / num_hidden_layers=2（真实 HF config 的字段名），
    # 解析必须直接命中，无条件成立。
    assert info["hidden_size"] == 768
    assert info["num_layers"] == 2


def test_describe_backbone_records_resolve_failure_reason():
    """resolve_* 失败时探针必须留下原因，而不是静默返回 None。

    用一份 resolve_hidden_size 无论如何都满足不了的 config（不含任何已知
    hidden size 字段）确定性地触发失败分支；不依赖 models/plm_backbones.py
    是否存在，因此模块缺失时本测试依然在测同一件事。
    """
    info = describe_backbone(_fake_bert_like(), SimpleNamespace(unrecognised_field=1))
    assert info["hidden_size"] is None
    reasons = [item for item in info["layer_names_sample"] if "resolve failed" in item]
    assert len(reasons) == 1, f"未记录 resolve 失败原因: {info['layer_names_sample']}"
    # 原因串里必须带上 config 的真实字段名，才能证明失败来自 resolve_hidden_size
    # 的字段清单逻辑（而不是 import 失败之类的其它异常）。
    assert "unrecognised_field" in reasons[0]


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


def _run_probe(scratch: Path, stub_source: str) -> subprocess.CompletedProcess:
    """按文件路径在真实子进程里运行探针，只把网络依赖 transformers 换成离线替身。

    启动方式与集群上一致（python tools/probe_backbone_modules.py），因此拿到的
    sys.path / 退出码 / stdout / stderr 都是真实行为，不是对探针的模拟。
    """
    stub_dir = scratch / "stubs"
    stub_dir.mkdir()
    (stub_dir / "transformers.py").write_text(stub_source, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_dir)  # 只留替身，避免外层 PYTHONPATH 掩盖缺陷
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    return subprocess.run(
        [
            sys.executable, "tools/probe_backbone_modules.py",
            "--model", "stub/local", "--kind", "esm2", "--no-trust-remote-code",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _parse_probe_json(stdout: str) -> dict:
    """取出 stdout 里探针打印的 JSON（替身的标记行可能排在它前面）。"""
    lines = stdout.splitlines()
    json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
    info, _ = json.JSONDecoder().raw_decode("\n".join(lines[json_start:]))
    return info


def test_direct_script_invocation_puts_repo_root_on_sys_path(probe_scratch):
    """回归：按文件路径直接运行探针时，仓库根必须出现在子进程的 sys.path 上。

    启动方式 `python tools/probe_backbone_modules.py ...` 会把 sys.path[0] 设为
    tools/，仓库根不在 sys.path 上；此时 describe_backbone 内的
    `from models.plm_backbones import ...` 以 "No module named 'models'" 失败，
    hidden_size / num_layers 变成 null（R13 之后退出码为 2），而集群验收正是检查
    这两个字段（ESMC-6B: hidden_size == 2560, num_layers == 80）。

    本测试用真实子进程复现该启动方式，读的是子进程真实的 sys.path，不是对
    sys.path 的模拟；只有网络依赖 transformers 被换成了离线替身。
    """
    result = _run_probe(probe_scratch, _stub_source(_STUB_CONFIG_RESOLVABLE))
    assert result.returncode == 0, f"探针未正常退出:\n{result.stdout}\n{result.stderr}"

    lines = result.stdout.splitlines()
    reported = [line for line in lines if line.startswith(_SYS_PATH_MARKER)]
    assert len(reported) == 1, f"未拿到子进程回传的 sys.path:\n{result.stdout}"
    child_sys_path = json.loads(reported[0][len(_SYS_PATH_MARKER):])
    normalized = {os.path.normcase(os.path.abspath(entry)) for entry in child_sys_path if entry}
    assert os.path.normcase(str(REPO_ROOT)) in normalized, (
        f"仓库根不在子进程 sys.path 上，models 包无法解析: {child_sys_path}"
    )

    info = _parse_probe_json(result.stdout)
    sample = "\n".join(info["layer_names_sample"])
    # 断言串带右引号：命中的只能是 'models 包整个解析不到'。
    assert "No module named 'models'" not in sample, (
        f"describe_backbone 仍以 'models 包不存在' 失败，"
        f"hidden_size={info['hidden_size']!r}, num_layers={info['num_layers']!r}:\n{sample}"
    )


def test_probe_exits_2_and_keeps_leaf_names_when_resolve_fails(probe_scratch):
    """回归：resolve 失败时探针必须以退出码 2 失败，同时 stdout 仍保留 leaf_names。

    集群验收（Task 21 Step 3）检查的正是 hidden_size == 2560 / num_layers == 80。
    如果 config 里没有任何已知 hidden size 字段而探针照样退出 0，一次什么都没解析
    出来的探测会被验收当成成功。反过来，leaf_names（模块命名 dump）是探针的主要
    价值，即使解析失败也必须留在 stdout 上供人工确定 LoRA target_modules。
    """
    result = _run_probe(probe_scratch, _stub_source(_STUB_CONFIG_UNRESOLVABLE))

    assert result.returncode == 2, (
        f"resolve 失败时探针未以退出码 2 结束:\n{result.stdout}\n{result.stderr}"
    )
    info = _parse_probe_json(result.stdout)
    # 先证明这次退出 2 确实来自 resolve 失败，而不是替身构建失败之类的其它原因。
    assert info["hidden_size"] is None

    # 主产物必须存活：JSON 里的 leaf_names，以及末尾的模块命名 dump。
    assert "leaf_names" in result.stdout
    assert info["leaf_names"], "resolve 失败时 leaf_names 不能为空"
    assert "query" in info["leaf_names"]

    # stderr 上要有带明确前缀的失败块，并点名解析不出来的字段。
    assert "[PROBE-FAILURE]" in result.stderr, result.stderr
    assert "hidden_size" in result.stderr, result.stderr


_IMPORT_SYS_PATH_MARKER = "PROBE_TEST_IMPORT_SYSPATH="

# 在全新解释器里只做一件事：记录 import 探针模块前后的 sys.path 快照。
# 用 `-c` 启动，sys.path[0] 是 ''（当前目录）而不是仓库根的绝对路径，
# 因此旧代码在 import 期 `sys.path.insert(0, 仓库根绝对路径)` 会立刻显形。
_IMPORT_PROBE_SOURCE = '''\
import json
import sys

before = list(sys.path)
import tools.probe_backbone_modules  # noqa: F401  这里测的就是它的导入期副作用

print("PROBE_TEST_IMPORT_SYSPATH=" + json.dumps({"before": before, "after": list(sys.path)}))
'''


def test_importing_probe_leaves_sys_path_untouched():
    """回归（R14）：import 探针模块不得改动 sys.path。

    测试模块在收集期 import 本文件。若文件在 import 期执行 sys.path.insert(0, 仓库根)，
    整个 pytest 会话的导入优先级都会被改变，仓库根的 data.py / utils.py / tracing.py /
    path_config.py 会遮蔽同名的已安装发行包。仓库根 bootstrap 只允许发生在脚本模式下。
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # 不让外层 PYTHONPATH 把仓库根预先塞进 sys.path
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE_SOURCE],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, f"子进程未能 import 探针模块:\n{result.stderr}"

    reported = [
        line for line in result.stdout.splitlines()
        if line.startswith(_IMPORT_SYS_PATH_MARKER)
    ]
    assert len(reported) == 1, f"未拿到子进程回传的 sys.path:\n{result.stdout}"
    snapshots = json.loads(reported[0][len(_IMPORT_SYS_PATH_MARKER):])
    assert snapshots["after"] == snapshots["before"], (
        f"import 探针模块改动了 sys.path（R14）："
        f"{snapshots['before']} -> {snapshots['after']}"
    )
