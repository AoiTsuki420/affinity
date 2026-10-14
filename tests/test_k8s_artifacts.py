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

R61 的两半也必须一起成立，否则共享目录会被静默写入：k8s profile 的
`runs_dir` / `train_data_dir` 都含 `${USER}`，pod 里既没有 `USER` 也没有
`USERNAME`，于是 `path_config` 回退到固定的 `kineager`，两条路径一起变成**共享**的
`/pfs_beijing/kineager/kineager_runs`；而 `train_data_dir` 是写入目标
（`train_epoch` 往里 `np.save` 每个 epoch 的 `preds_epoch*` / `labels_epoch*`）。
所以 1. manifest 的 env 块必须声明操作者填写的 `USER`，头部注释也要说明；
2. launcher 必须在 `k8s` profile 且 `USER` 未设时拒绝启动（先于 exec）。

R62：本模块对 `yaml` 保持硬 import，**不要**改成 `importorskip`。R60 已让 PyYAML
成为声明依赖，而 `importorskip` 会静默跳过这个负责校验 k8s 交付物的模块——正好把
它存在的意义（发现交付物损坏）藏起来。收集期报「装依赖」才是正确的失败方式。

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
REQUIREMENTS = REPO_ROOT / "requirements.txt"

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
# R61：k8s profile 的写入路径必须落在用户自己的目录下
# ---------------------------------------------------------------------------

def _launcher_user_guard():
    """launcher 里那个「k8s profile 且 USER 未设」的守卫块（含条件行与 fi）。

    先去掉 shell 注释再定位：`#` 后面的说明文字里也出现了 CATA_PROFILE / USER，
    不剥掉就会把注释行也当成候选。返回 (起始行号, 块文本)。
    """
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("if ")
              and "CATA_PROFILE" in line and "USER" in line]
    assert len(starts) == 1, f"USER 守卫必须恰好一处，找到 {len(starts)} 处"
    ends = [i for i in range(starts[0] + 1, len(lines)) if lines[i].strip() == "fi"]
    assert ends, "USER 守卫没有 fi 收尾"
    return starts[0], "\n".join(lines[starts[0]:ends[0] + 1])


def test_launcher_refuses_the_k8s_profile_without_user():
    _, block = _launcher_user_guard()
    condition = block.splitlines()[0]
    assert '"${USER:-}"' in condition, "守卫必须测 USER 是否为空（${USER:-} 才不会触发 set -u）"
    assert "-z" in condition, "守卫必须测「为空」而不是「非空」"
    assert re.search(r"^\s*exit 1\s*$", block, re.M), "守卫必须 exit 1：拒绝启动才是正确的取舍"
    assert "USER" in block, "守卫的消息必须点名 USER"
    lowered = block.lower()
    assert "shared" in lowered and "kineager_runs" in lowered, (
        "守卫的消息必须说明理由：回退目录是共享的，并发 run 的产物会互相覆盖"
    )


def test_launcher_user_guard_precedes_the_exec():
    start, _ = _launcher_user_guard()
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    exec_lines = [i for i, line in enumerate(lines) if line.startswith("exec ")]
    assert exec_lines, "launcher 的 exec 不见了"
    assert start < exec_lines[0], "USER 守卫必须与其它校验一起先于 exec"


def test_manifest_env_declares_user():
    env = {entry["name"]: entry.get("value") for entry in _container()["env"]}
    assert "USER" in env, (
        "manifest 的 env 块没有列出 USER：pod 里 USER/USERNAME 都没设，"
        "写入路径会回退到共享目录"
    )
    value = env["USER"]
    assert isinstance(value, str) and value.strip(), "USER 不能是空值"
    assert value.startswith("<REPLACE_ME") and value.endswith(">"), (
        f"USER 必须写成 <REPLACE_ME: …>，与其它操作者填写的值风格一致（实际 {value!r}）"
    )


def test_manifest_header_lists_user():
    header = _header_comment(MANIFEST.read_text(encoding="utf-8"))
    assert "${USER}" in header, "manifest 头部注释没有说明 USER 必须填写及其原因"


# ---------------------------------------------------------------------------
# R60 / R62：PyYAML 是声明依赖；这个模块的 yaml import 保持硬失败
# ---------------------------------------------------------------------------

def _declared_requirements():
    """requirements.txt 里声明的发行包名（小写、下划线归一为连字符）。"""
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec:
            continue
        names.add(re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].strip().lower().replace("_", "-"))
    return names


def test_requirements_declares_pyyaml():
    """R60：本模块 import yaml，所以 PyYAML 必须在 requirements.txt 里声明。

    不声明的话，集群上 `pip install -r requirements.txt` 之后 pytest 会在**收集期**
    报 ModuleNotFoundError，而不是报任何真实缺陷。
    """
    declared = _declared_requirements()
    assert "pyyaml" in declared, f"requirements.txt 未声明 PyYAML：{sorted(declared)}"


def test_yaml_import_stays_hard_and_is_not_importorskip():
    """R62：不要把 `import yaml` 改成 `importorskip`，否则本模块会被静默跳过。

    这条断言防的是「后来的读者按 reviewer 的建议把 import 软化」——缺依赖时本模块
    根本收集不起来，这正是想要的失败方式（见模块 docstring）。用 AST 判断实际调用，
    这样模块里说明这条 ruling 的注释不会被自己误判。
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    imported = {alias.name for node in tree.body
                if isinstance(node, ast.Import) for alias in node.names}
    assert "yaml" in imported, "yaml 必须是模块顶层的硬 import"

    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    called |= {node.func.id for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "importorskip" not in called, "本模块的 PyYAML 依赖不允许被软化跳过"


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
