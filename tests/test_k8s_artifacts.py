"""k8s 交付物的结构检查（本机起不了 bash，所以这是结构断言，不是语法解析）。

R58 的两半必须一起成立，否则 k8s run 会**静默**用错 profile：
`path_config.resolve_runtime_paths()` 在 `CATA_PROFILE` 未设时回落到平台默认
（Linux 上是 `autodl`），于是 `train_csv_path` / `train_data_dir` / 骨干路径全部
指向另一个 profile。所以：

1. k8s 入口脚本必须自己 export `CATA_PROFILE`（默认 `k8s`，仍可被环境覆盖），
   并且要在 exec 训练进程之前；
2. manifest 的 env 块必须显式列出 `CATA_PROFILE`，操作者不必读脚本就能改，
   头部注释里也要出现这个名字。

其余断言是上一轮临时结构 checker 的固化版本：manifest 的锚点身份、exec 唯一且在
最后、校验先于 exec、rendezvous 参数各一次、`"$@"` 透传一次、不硬编码超参、
launcher 传给训练脚本的 flag 都能在训练脚本的 argparse 里找到、两个 shell 面的
编码卫生（纯 LF / 纯 ASCII / 单个结尾换行）。

注意：本文件用 PyYAML 解析 manifest（无锚点身份检查就只能靠文本比对）。
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "bin" / "train" / "kineager_k8s.sh"
MANIFEST = REPO_ROOT / "yamls" / "kubeflow" / "kineager_affinity_k8s.yaml"
TRAINER = REPO_ROOT / "main_train_predictor_multigpu.py"

# torchrun 自己的参数（不是训练脚本的 argparse 参数）
_TORCHRUN_FLAGS = {"-m", "--nnodes", "--node_rank", "--nproc_per_node",
                   "--master_addr", "--master_port"}

# 训练超参：launcher 一个都不许硬编码，必须由调用方通过 "$@" 传进来
_HYPERPARAMETER_FLAGS = ("-epochs", "-batch_size", "--lr", "--split_mode", "--split_seed",
                         "--val_ratio", "--test_ratio", "--eval_only", "--weights",
                         "--use_struct_branch", "--final_contact_npz")


def _launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def _strip_shell_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _header_comment(text):
    """文件开头的连续注释块（含空行为止）。"""
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _replicas():
    return _manifest()["spec"]["pytorchReplicaSpecs"]


def _container():
    return _replicas()["Master"]["template"]["spec"]["containers"][0]


def _trainer_argparse_flags():
    """训练脚本 add_argument 的名字集合（AST，不执行业务代码）。"""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("-"):
                        flags.add(arg.value)
    return flags


def _launcher_trainer_flags():
    """exec 块里传给训练脚本的 flag（去掉 torchrun 自己的参数）。

    先删掉 `${VAR:-default}` 参数展开：其中的 `:-off` / `:-last_k` 会被 flag 正则
    误当成 `-off` / `-last_k` 这样的短选项。
    """
    exec_block = _strip_shell_comments(_launcher_text().split("exec python3", 1)[1])
    exec_block = re.sub(r"\$\{[^{}]*\}", " EXP ", exec_block)
    found = set(re.findall(r"(?<![\w-])(--?[A-Za-z][A-Za-z0-9_]*)", exec_block))
    return found - _TORCHRUN_FLAGS


# ---------------------------------------------------------------------------
# R58：profile 必须被真正激活
# ---------------------------------------------------------------------------

def test_launcher_exports_the_k8s_profile_before_exec():
    text = _launcher_text()
    match = re.search(r'^export CATA_PROFILE="\$\{CATA_PROFILE:-([A-Za-z0-9_]+)\}"[ \t]*$',
                      text, re.M)
    assert match, (
        "launcher 没有 export CATA_PROFILE：CATA_PROFILE 未设时 "
        "resolve_runtime_paths() 会回落到 autodl，k8s run 静默用错 profile"
    )
    assert match.group(1) == "k8s", "k8s 入口的默认 profile 必须是 k8s"
    assert match.start() < text.index("exec python3"), "profile 必须在 exec 之前导出"


def test_manifest_env_lists_the_cata_profile():
    env = {entry["name"]: entry.get("value") for entry in _container()["env"]}
    assert "CATA_PROFILE" in env, "manifest 的 env 块没有列出 CATA_PROFILE"
    assert env["CATA_PROFILE"] == "k8s"


def test_manifest_header_lists_the_cata_profile():
    assert "CATA_PROFILE" in _header_comment(MANIFEST.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# manifest：仍然可解析、锚点身份仍然成立
# ---------------------------------------------------------------------------

def test_manifest_parses_and_keeps_the_single_pod_spec_anchor():
    spec = _manifest()
    assert spec["apiVersion"] == "kubeflow.org/v1"
    assert spec["kind"] == "PyTorchJob"
    assert spec["metadata"]["name"] == "kineager-affinity"

    replicas = _replicas()
    master, worker = replicas["Master"], replicas["Worker"]
    assert set(master) == set(worker) == {"replicas", "restartPolicy", "template"}
    assert master["template"] is worker["template"], "Worker.template 不再是 Master.template 的别名"
    assert master["replicas"] == 1


def test_manifest_container_runs_the_launcher():
    args = _container()["args"]
    assert len(args) == 1
    assert "bash bin/train/kineager_k8s.sh" in args[0]


# ---------------------------------------------------------------------------
# launcher：结构契约（上一轮临时 checker 的固化版本）
# ---------------------------------------------------------------------------

def test_launcher_has_exactly_one_exec_and_it_is_last():
    lines = _launcher_text().splitlines()
    exec_lines = [i for i, line in enumerate(lines) if line.startswith("exec ")]
    assert len(exec_lines) == 1, "launcher 必须只有一条 exec"
    guards = [i for i, line in enumerate(lines) if line.strip().startswith("exit 1")]
    assert guards, "launcher 的拓扑/输入校验不见了"
    assert max(guards) < exec_lines[0], "所有校验都必须先于 exec"


def test_launcher_passes_each_rendezvous_flag_exactly_once():
    text = _launcher_text()
    for flag in ("--nnodes=", "--node_rank=", "--nproc_per_node=",
                 "--master_addr=", "--master_port="):
        assert text.count(flag) == 1, f"{flag} 出现 {text.count(flag)} 次"
    code = _strip_shell_comments(text)          # 注释里解释过 "$@"，不能算进次数
    assert code.count('${1+"$@"}') == 1, "调用方参数必须恰好透传一次"
    assert code.count(' -m torch.distributed.run') == 1


def test_launcher_hardcodes_no_training_hyperparameters():
    code = _strip_shell_comments(_launcher_text())
    found = [flag for flag in _HYPERPARAMETER_FLAGS if flag in code]
    assert found == [], f"launcher 硬编码了训练超参: {found}"


def test_launcher_trainer_flags_exist_in_the_trainer_argparse():
    known = _trainer_argparse_flags()
    passed = _launcher_trainer_flags()
    assert passed, "exec 块里没有解析到任何训练脚本 flag"
    unknown = sorted(flag for flag in passed if flag not in known)
    assert unknown == [], f"launcher 传了训练脚本不认识的 flag: {unknown}"


def test_k8s_artifacts_are_ascii_with_one_trailing_newline_and_consistent_eol():
    """源码卫生：纯 ASCII、恰好一个结尾换行、没有孤立 CR（不混用换行）。

    这里不要求工作区是 LF：本仓库 `core.autocrlf=true`，checkout 出来就是 CRLF，
    文件工作区字节数会随 git 操作变化（断言「工作区无 CR」会变成 flaky）。真正要
    保证的是 blob 是 LF——提交时 git 会归一化，报告里另用 `git cat-file blob`
    逐字节核对过 CR=0。
    """
    for path in (LAUNCHER, MANIFEST):
        raw = path.read_bytes()
        assert raw.decode("utf-8").isascii(), f"{path.name} 含非 ASCII 字节"
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{path.name} 结尾换行不规范"
        assert b"\r" not in raw.replace(b"\r\n", b""), f"{path.name} 含孤立 CR（混用换行）"
