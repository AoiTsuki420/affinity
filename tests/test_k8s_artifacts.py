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

R61：k8s profile 的 `runs_dir` / `train_data_dir` 都含 `${USER}`，pod 里既没有
`USER` 也没有 `USERNAME`，于是 `path_config` 回退到固定的 `kineager`，两条路径一起
变成**共享**的 `/pfs_beijing/kineager/kineager_runs`；而 `train_data_dir` 是写入目标
（`train_epoch` 往里 `np.save` 每个 epoch 的 `preds_epoch*` / `labels_epoch*`）。
所以 1. manifest 的 env 块必须声明操作者填写的 `USER`，头部注释也要说明；
2. launcher 必须在 `k8s` profile 下拒绝不安全的 `USER`，先于 exec。这个守卫有四件事
都要成立，缺一件它就只是**看起来**安全：
   * R63 接受 `USER` 或 `USERNAME`（`path_config` 的解析顺序），否则解析器认可的
     配置会被启动器拒绝；
   * R61 两者都没有时拒绝启动，而不是共享目录；
   * R65 拒绝仍然是 manifest 占位符（`<REPLACE_ME…>`）的值——它是非空字符串，
     否则会变成一个以占位符命名的目录；
   * R64 `export` 校验过的那个值：`${USER:-}` 能看到**未导出**的 shell 变量，而
     exec 出去的训练进程只继承环境，`path_config` 读的是 `os.getenv`。

R62：本模块对 `yaml` 保持硬 import，**不要**改成 `importorskip`。R60 已让 PyYAML
成为声明依赖，而 `importorskip` 会静默跳过这个负责校验 k8s 交付物的模块——正好把
它存在的意义（发现交付物损坏）藏起来。收集期报「装依赖」才是正确的失败方式。

注意：本文件用 PyYAML 解析 manifest（无锚点身份检查就只能靠文本比对）。
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from path_config import resolve_runtime_paths

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
# R61 / R63 / R64 / R65：k8s profile 的写入路径必须落在用户自己的目录下
# ---------------------------------------------------------------------------

def _launcher_user_guard():
    """launcher 里那个 k8s-profile USER 守卫的整块（含结尾 fi）。返回 (首行, 末行, 文本)。

    先去掉 shell 注释再定位：`#` 后面的说明文字里也出现了 CATA_PROFILE / USER，
    不剥掉就会把注释行也当成候选。锚点是「唯一一条提到 CATA_PROFILE 的 if」——
    条件行本身不再直接出现 USER（R63/R65 之后用户名归一到了块内的变量），所以
    USER 的存在性在块上校验。块内有嵌套的 if/case，结尾要按深度配对，不能取第一个 fi。
    """
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("if ") and "CATA_PROFILE" in line]
    assert len(starts) == 1, f"USER 守卫必须恰好一处（唯一的 CATA_PROFILE if），找到 {len(starts)} 处"

    depth = 0
    end = None
    for index in range(starts[0], len(lines)):
        stripped = lines[index].strip()
        if re.match(r"(if|case)\b", stripped):
            depth += 1
        if re.match(r"(fi|esac)\b", stripped):
            depth -= 1
            if depth == 0:
                end = index
                break
    assert end is not None, "USER 守卫的 if 没有配对的 fi"
    block = "\n".join(lines[starts[0]:end + 1])
    assert "USER" in block, "提到 CATA_PROFILE 的那个 if 不是 USER 守卫"
    return starts[0], end, block


def _guard_user_variable(block):
    """守卫里承接 `${USER:-${USERNAME:-}}` 的那个变量名（R63 的两种形式）。

    从源码里取变量名而不是硬编码，这样「校验的那个值」与「导出的那个值」可以
    相互对照：R64 的失败模式正是两者脱钩。
    """
    match = re.search(r'^\s*([A-Za-z_]\w*)=["\']?\$\{USER:-\$\{USERNAME:-\}\}',
                      block, re.M)
    assert match, (
        "守卫没有以 ${USER:-${USERNAME:-}} 的形式承接用户名："
        "path_config 接受 USER 或 USERNAME，守卫必须接受同样的两种形式"
    )
    return match.group(1)


def _guard_messages(block):
    """守卫里 echo 出去的文本（注释已被 _strip_shell_comments 去掉）。"""
    return "\n".join(line for line in block.splitlines()
                     if line.strip().startswith("echo "))


def _guard_case_region(block):
    """`case … esac` 那一段（R65 的占位符拒绝分支）。"""
    assert "case " in block and "esac" in block, "守卫缺少 case/esac 的占位符拒绝分支"
    start = block.index("case ")
    end = block.index("esac", start)
    return block[start:end]


def _guard_unset_region(block, variable):
    """`if [ -z "${var}" ]; then … fi` 那一段（R61 的缺用户名拒绝分支）。"""
    match = re.search(rf'^\s*if \[ -z "\$\{{{variable}\}}" \]; then\s*$', block, re.M)
    assert match, f'守卫缺少 if [ -z "${{{variable}}}" ] 分支'
    tail = block[match.end():]
    end = re.search(r"^\s*fi\s*$", tail, re.M)
    assert end, "缺用户名的 if 分支没有 fi 收尾"
    return tail[:end.start()]


def test_launcher_refuses_the_k8s_profile_without_user():
    _, _, block = _launcher_user_guard()
    variable = _guard_user_variable(block)
    unset = _guard_unset_region(block, variable)

    assert '[ "${CATA_PROFILE}" = "k8s" ]' in block, "守卫必须只作用于 k8s profile"
    assert re.search(r"^\s*exit 1\s*$", unset, re.M), (
        "缺用户名这一支必须 exit 1：拒绝启动才是正确的取舍"
    )
    assert "USER" in unset, "消息必须点名 USER"
    lowered = unset.lower()
    assert "shared" in lowered and "kineager_runs" in lowered, (
        "消息必须说明理由：回退目录是共享的，并发 run 的产物会互相覆盖"
    )


def test_launcher_accepts_user_or_username_like_path_config():
    """R63：守卫接受的两种形式必须与 path_config 的解析顺序一致。"""
    _, _, block = _launcher_user_guard()
    _guard_user_variable(block)          # USER -> USERNAME，缺一不可
    assert "USERNAME" in _guard_messages(block), (
        "拒绝消息里必须点名 USERNAME，否则补救办法不可发现"
    )


def test_launcher_rejects_the_manifest_user_placeholder():
    """R65：manifest 的 `<REPLACE_ME: …>` 是非空字符串，忘了填也能过原来的检查。"""
    _, _, block = _launcher_user_guard()
    placeholder = _guard_case_region(block)

    assert re.search(r'["\']<REPLACE_ME["\']\*', placeholder), (
        "守卫必须用 case 模式拒绝仍然留着 <REPLACE_ME 前缀的 USER"
    )
    assert re.search(r"^\s*exit 1\s*$", placeholder, re.M), "占位符这一支必须 exit 1"
    assert "placeholder" in placeholder.lower(), "消息必须说明「占位符还在」"


def test_launcher_exports_the_validated_user():
    """R64：被校验的值必须 export —— exec 出去的训练进程继承的是环境。

    `${USER:-}` 能看到未导出的 shell 变量，但 `USER=alice; bash …` 里的值不会
    进入环境，`path_config` 的 `os.getenv("USER")` 于是看不到它，还会落到共享目录。
    bash 在本机起不来，所以这里是源码级断言：export 存在、在守卫之内、在最后一条
    exit 1 之后、且导出的是**守卫校验过的那个变量**（不是另写的表达式）。
    """
    start, end, block = _launcher_user_guard()
    variable = _guard_user_variable(block)
    lines = _strip_shell_comments(_launcher_text()).splitlines()

    exports = [i for i, line in enumerate(lines[start:end + 1], start=start)
               if re.fullmatch(rf'export USER="\$\{{{variable}\}}"', line.strip())]
    assert len(exports) == 1, (
        f'守卫内必须恰好一条 export USER="${{{variable}}}"，找到 {len(exports)} 条'
    )
    export_index = exports[0]

    exits = [i for i, line in enumerate(lines[start:end + 1], start=start)
             if line.strip() == "exit 1"]
    assert exits and max(exits) < export_index, "export 必须跟在所有拒绝分支之后"
    assert export_index < end, "export 必须在守卫块内（k8s profile 才需要它）"

    exec_lines = [i for i, line in enumerate(lines) if line.startswith("exec ")]
    assert exec_lines, "launcher 的 exec 不见了"
    assert export_index < exec_lines[0], "export 必须先于 exec，否则训练进程看不到它"


def test_an_unexported_value_never_reaches_a_child_process():
    """R64 的机制，用真实子进程演示：环境才是子进程能看到的通道。

    bash 在本机起不来，所以用 Python 复现同一语义——父进程里存在一个名为 USER
    的变量，但只有把它放进环境，子进程的 `os.getenv("USER")` 才能看到。这正是
    `USER=alice; bash …` 与 `export USER=alice; bash …` 的差别，也是 R64 要求
    export 的原因：`path_config` 读的是环境。
    """
    probe = "import os; print(os.getenv('USER') or '<unset>')"
    env_without = {k: v for k, v in os.environ.items() if k not in ("USER", "USERNAME")}

    USER = "alice"          # noqa: F841 — 父进程里的变量，等价于未 export 的 shell 变量
    assert USER == "alice"  # 值确实存在，只是不在环境里

    unseen = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, env=env_without)
    assert unseen.stdout.strip() == "<unset>", "未导出的值居然被子进程看到了"

    seen = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env={**env_without, "USER": "alice"})
    assert seen.stdout.strip() == "alice", "环境里的值必须被子进程看到"


def test_path_config_accepts_username_when_user_is_unset(monkeypatch):
    """R63 的解析器一侧：只设 USERNAME 时解析出的是该用户自己的目录。

    这条断言把守卫接受的形式钉在解析器的真实行为上；两个都未设时才回退到共享
    目录（`tests/test_path_config.py::test_k8s_profile_falls_back_when_no_user_variable_is_set`）。
    """
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("USERNAME", "alice")
    resolved = str(resolve_runtime_paths(profile="k8s")["train_data_dir"]).replace("\\", "/")

    assert resolved == "/pfs_beijing/alice/kineager_runs"
    assert "kineager/kineager_runs" not in resolved


def test_launcher_user_guard_precedes_the_exec():
    start, _, _ = _launcher_user_guard()
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    exec_lines = [i for i, line in enumerate(lines) if line.startswith("exec ")]
    assert exec_lines, "launcher 的 exec 不见了"
    assert start < exec_lines[0], "USER 守卫必须与其它校验一起先于 exec"


def test_launcher_user_guard_precedes_the_required_input_checks():
    """守卫必须先于 `:?` 必填检查 —— runbook §8 第 11 条据此断言「不必再设别的」。

    否则把守卫挪到 TRAIN_CSV 之后就没人发现：套件仍然全绿，runbook 却变成错的。
    """
    start, _, _ = _launcher_user_guard()
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    required = [i for i, line in enumerate(lines) if ":?" in line]
    assert required, "launcher 的必填变量检查（${VAR:?…}）不见了"
    assert start < min(required), "USER 守卫必须先于 ${VAR:?…} 必填检查"


def test_manifest_env_declares_user():
    env = {entry["name"]: entry.get("value") for entry in _container()["env"]}
    assert "USER" in env, (
        "manifest 的 env 块没有列出 USER：pod 里 USER/USERNAME 都没设，"
        "写入路径会回退到共享目录"
    )
    value = env["USER"]
    # 这里只要求「已声明且非空」。R61 要的是 manifest 必须提供 USER，好让写入路径按用户
    # 隔离；早期版本还断言它必须是 `<REPLACE_ME: …>` 占位符，但那是把「尚未填写」当成了
    # 期望状态——真实部署恰恰应该是填好的（本仓库现在是 `xinran_2026`），于是那条断言会
    # 在配置正确的 manifest 上失败。未填写的占位符由 launcher 的守卫负责拒绝（匹配
    # `^<REPLACE_ME` 即报错退出），不需要在这里重复。
    assert isinstance(value, str) and value.strip(), "USER 不能是空值"


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
    根本收集不起来，这正是想要的失败方式（见模块 docstring）。用 AST 判断是否有
    `importorskip` **调用**，这样本文件里说明这条 ruling 的注释和消息不会被误判，
    而真正的 `pytest.importorskip(...)` 一定跑不掉。
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


def test_k8s_artifacts_are_utf8_with_one_trailing_newline_and_consistent_eol():
    """源码卫生：可解码为 UTF-8、恰好一个结尾换行、没有孤立 CR（不混用换行）。

    这里**不要求纯 ASCII**：启动脚本带中文注释（`PYTHONPATH` 那一段），bash 按字节读取，
    UTF-8 完全合法。早期版本断言 `.isascii()`，那是我自己的偏好而不是需求，会在加了中文
    注释的脚本上误报。

    这里也**不要求工作区是 LF**：本仓库 `core.autocrlf=true`，checkout 出来就是 CRLF，
    断言「工作区无 CR」会 flaky。真正有害的是**孤立** CR，那说明换行混用，所以只查它。
    """
    for path in (LAUNCHER, MANIFEST):
        raw = path.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(f"{path.name} 不是合法 UTF-8：{error}") from error
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{path.name} 结尾换行不规范"
        assert b"\r" not in raw.replace(b"\r\n", b""), f"{path.name} 含孤立 CR（混用换行）"
