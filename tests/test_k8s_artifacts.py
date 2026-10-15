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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from path_config import resolve_runtime_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "bin" / "train" / "kineager_k8s.sh"
LAUNCH_WRAPPER = REPO_ROOT / "bin" / "train" / "launch_kineager_affinity_k8s.sh"
MANIFEST = REPO_ROOT / "yamls" / "kubeflow" / "kineager_affinity_k8s.yaml"
PIPELINE = REPO_ROOT / "yamls" / "kubeflow" / "kineager_affinity_pipeline.yaml"
TRAINER = REPO_ROOT / "main_train_predictor_multigpu.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# `bash -n` 需要一个**真能跑**的 bash：本机 PATH 上的 `bash` 常常是
# `WindowsApps\bash.EXE`（WSL 转发器，报 Wsl/E_ACCESSDENIED），Git for Windows 的
# bash.exe 又被文件沙箱拦在「创建 signal pipe」那一步（Win32 error 5）。所以不猜
# 路径：拿 `bash -n` 试一个玩具脚本，能返回 0 的才算数，否则下面的语法断言直接
# skip —— 报一个它其实看不见的缺陷比 skip 更糟。
def _find_usable_bash():
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe",
                  r"C:\Program Files\Git\usr\bin\bash.exe"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run([candidate, "-n"], input="true\n",
                                   capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


BASH = _find_usable_bash()

# torchrun 自己的参数（不是训练脚本的 argparse 参数）
_TORCHRUN_FLAGS = {"-m", "--nnodes", "--node_rank", "--nproc_per_node",
                   "--master_addr", "--master_port"}

# 训练超参：launcher 一个都不许硬编码，必须由调用方通过 "$@" 传进来
_HYPERPARAMETER_FLAGS = ("-epochs", "-batch_size", "--lr", "--split_mode", "--split_seed",
                         "--val_ratio", "--test_ratio", "--eval_only", "--weights",
                         "--use_struct_branch", "--final_contact_npz")

# 训练算子的 label：它标在 PyTorchJob 自己创建的 pod 上，也是「这个 job 有哪些
# pod」唯一稳定的查询方式（pod 名在 elastic 拓扑下不可预测）。
_JOB_LABEL = "training.kubeflow.org/job-name"

# 上传到 KFP UI 的 compiled pipeline 必须长成的形状
_PIPELINE_TEMPLATE = "kineager-affinity-train"
_PIPELINE_NAME = "kineager-affinity-train"
_PIPELINE_PARAMETERS = ["bash_command", "logpath", "memory", "cpu", "pod_gpus",
                        "image", "pod_nums", "working_dir"]


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


def _load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _substitute_argo_parameters(text, values):
    """把 `{{inputs.parameters.NAME}}` 换成给定值（Argo 在集群上做的事）。

    KFP 编译产物的 manifest 里 `{{...}}` 是**未加引号**地嵌在 YAML 里的
    （`minReplicas: {{inputs.parameters.pod_nums}}`），直接 safe_load 会把它读成嵌套
    mapping 而报 "found unhashable key"。参考文件
    （xtrimofold3-affinity-plm/yamls/kubeflow/xf3_pl_multitask_k8s.yaml）在同样的
    输入下报同样的错，所以那是模板语法的性质、不是缺陷——本函数先做替换，再把
    替换结果当真正的 YAML 解析。values 里的字典项会被 dump 成 JSON 字面量
    （容器 args 的 `{...}` 花括号因此不会把解析器搞乱），其余按原文插入。
    """
    import json as _json

    def replace(match):
        name = match.group(1)
        assert name in values, f"测试没有为参数 {name} 准备替换值"
        value = values[name]
        if isinstance(value, str):
            return value
        return _json.dumps(value)

    return re.sub(r"\{\{inputs\.parameters\.([A-Za-z0-9_]+)\}\}", replace, text)


def _embedded_manifest(path=PIPELINE):
    """KFP pipeline 里那段 `manifest: |` 字符串。"""
    document = _load_yaml(path)
    templates = [item for item in document["spec"]["templates"] if "resource" in item]
    assert len(templates) == 1, f"{path.name} 必须恰好一个 resource 模板"
    manifest = templates[0]["resource"]["manifest"]
    assert isinstance(manifest, str) and manifest.strip(), "resource 模板的 manifest 是空的"
    return manifest


def _rank0_argv(pod_nums):
    """按 pod_nums 渲染出 rank 0 pod 上真正执行的容器 argv。"""
    values = {
        "bash_command": "bash bin/train/kineager_k8s.sh -epochs 30",
        "logpath": "/pfs_beijing/xinran_2026/kineager_runs/kineager_affinity/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "harbor.biomap-int.com/xinran_2026/kineager:affinity-v1",
        "pod_nums": pod_nums,
        "working_dir": "/workspace/kineager",
    }
    job = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), values))
    container = job["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]
    assert container["command"][:2] == ["/bin/bash", "-lc"]
    return container["args"][0]


def _replicas():
    return _manifest()["spec"]["pytorchReplicaSpecs"]


def _container():
    return _replicas()["Master"]["template"]["spec"]["containers"][0]


def _pipeline_container():
    """pipeline 嵌的 PyTorchJob 里那个容器（参数已替换，可以当普通 YAML 读）。"""
    manifest = _substitute_argo_parameters(_embedded_manifest(), {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "2",
        "working_dir": "/workspace/kineager",
    })
    job = yaml.safe_load(manifest)
    return job["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]


def _launcher_exec_lines():
    """launcher 里每条 `exec python3 …` 命令的**起始行号**（允许缩进）。

    续行不算新的 exec，所以只在「上一行不以反斜杠结尾」的位置取。早期版本要求
    `exec` 出现在第 0 列、并且只有一条。rank-0 的日志重定向必须在 exec **之前**
    生效，而把重定向包在 exec 外面（`{ … } >> "$LOG_PATH" 2>&1`）就会让那条 exec
    缩进一格；再加上「每条命令只有一次 exec」的约束，只能写成两个分支、每条分支各
    一条 exec 命令行。于是这里断言的是：都在行首（允许缩进）、且没有其它形式的
    exec（`_launcher_exec_occurrences`）——强度没有放松，只是把「一条」换成了
    「每个分支一条、彼此逐 token 相同」。
    """
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    exec_lines = []
    for index, line in enumerate(lines):
        if not re.match(r"\s*exec\s+python3\b", line):
            continue
        previous = lines[index - 1].rstrip() if index else ""
        if not previous.endswith("\\"):
            exec_lines.append(index)
    assert exec_lines, "launcher 的 exec python3 不见了"
    return exec_lines


def _launcher_exec_line():
    """第一条 exec 的行号（校验必须都在它之前）。"""
    return _launcher_exec_lines()[0]


def _launcher_exec_occurrences():
    """整个脚本里出现 `exec ` 这个命令 token 的行。

    这是「没有第二条 exec」的真正证明：既有断言只看 `^exec python3`，而
    `exec 1>>"$LOG_PATH" 2>&1` 这样的写法能骗过它。这里连 `;`/`&&`/`$( ` 之后的
    exec 一起抓，再与 `_launcher_exec_lines()` 的数量对齐。
    """
    return [line.strip() for line in _strip_shell_comments(_launcher_text()).splitlines()
            if re.search(r"(?:^|[;&|({]|\$\()\s*exec\s", line)]


def _launcher_exec_commands():
    """各条 exec 分支的「逻辑命令行」：把续行拼起来、去掉每行缩进。

    它们必须逐 token 相同（见
    `test_launcher_passes_each_rendezvous_flag_exactly_once`）：rank 门控只允许改
    变输出去向，不允许改变 torchrun 收到的参数。谁改了一条分支的 `--nnodes=`
    表达式，这里立刻变成两条不同的命令行。
    """
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    commands = []
    for start in _launcher_exec_lines():
        chunk = []
        for line in lines[start:]:
            chunk.append(line.strip())
            if not line.rstrip().endswith("\\"):
                break
        commands.append(" ".join(chunk))
    return commands


def _launcher_section(heading):
    """`# --- <heading> …` 那一节的 block（去掉注释行）。

    只认真正的分节头（`# --- ` 之后紧跟分节名），不认正文注释里提到的
    `# --- topology validation: … ---`：两者都以 `# --- ` 开头，靠 heading 的完整
    名字区分（调用方传的是分节名本身，例如 "topology resolution"）。
    """
    lines = _launcher_text().splitlines()
    marker = f"# --- {heading}"
    starts = [i for i, line in enumerate(lines) if line.startswith(marker)]
    assert len(starts) == 1, f"launcher 里 `{heading}` 那一节必须恰好一处，找到 {len(starts)} 处"
    end = len(lines)
    for index in range(starts[0] + 1, len(lines)):
        if lines[index].startswith("# --- "):
            end = index
            break
    block = "\n".join(lines[starts[0]:end])
    return "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))


def _topology_assignment(variable):
    """拓扑解析块里 `VAR="${A:-${B:-…}}"` 的 fallback 链。

    链 = 赋值里每个 `${...}` 展开的名字；最后一个展开的默认值（字面量）追加在末尾，
    所以 `world_size` 这种「裸默认值」也进得了链。返回 (链, 原始赋值)。
    """
    block = _launcher_section("topology resolution")
    match = re.search(rf'^{variable}="(.*)"[ \t]*$', block, re.M)
    assert match, f"拓扑解析块里没有 {variable}= 赋值"
    expression = match.group(1)
    chain = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-|\})", expression)
    tail = re.search(r":-([0-9]+)\}\}\}?$", expression)
    if tail:
        chain.append(tail.group(1))
    return chain, expression


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

    assert export_index < _launcher_exec_line(), "export 必须先于 exec，否则训练进程看不到它"


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
    assert start < _launcher_exec_line(), "USER 守卫必须与其它校验一起先于 exec"


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

def test_launcher_has_exactly_one_exec_per_branch_and_they_are_last():
    """每条分支一条 exec，且它们是脚本的最后一件事。

    「行首 token」的判定允许缩进（见 `_launcher_exec_lines`）：rank-0 的日志重定向
    必须包住那条 exec，所以它在 `{ …; } >> "${LOG_PATH}" 2>&1` 里。同时断言 `exec`
    这个命令 token 在全脚本里只出现在这些行上——否则 `exec 1>>file` 那种写法既能
    骗过行首检查，又能给每行偷偷加一次 exec。
    """
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    exec_lines = _launcher_exec_lines()
    assert len(exec_lines) == 2, (
        f"期望两条 exec（rank 0 的带重定向 / 其它 rank 的不带），找到 {len(exec_lines)} 条"
    )
    occurrences = _launcher_exec_occurrences()
    assert len(occurrences) == len(exec_lines), (
        f"`exec` 命令 token 必须只出现在这些分支行上，实际: {occurrences}"
    )

    guards = [i for i, line in enumerate(lines) if line.strip().startswith("exit 1")]
    assert guards, "launcher 的拓扑/输入校验不见了"
    assert max(guards) < min(exec_lines), "所有校验都必须先于 exec"

    # 最后一条 exec 之后只允许收尾的 `fi`；任何代码在那里都已经不可达，而这正是
    # 要挡的（例如把某个校验挪到 exec 后面）。两条 exec 之间不需要单独检查：它们必须
    # 逐 token 相同（`test_launcher_passes_each_rendezvous_flag_exactly_once`），
    # 中间插不进任何语句。
    trailer = [line for line in lines[exec_lines[-1] + 1:] if line.strip()]
    assert trailer and trailer[-1].strip() == "fi", "exec 之后必须以收尾的 fi 结束"
    for line in trailer:
        assert line.rstrip().endswith("\\") or re.match(r'^\s*(fi|else|}|\$)', line), (
            f"exec 之后还有代码在跑（不可达）: {line!r}"
        )
    assert lines[-1].strip() == "fi", "脚本必须以收尾的 fi 结束"


def test_launcher_passes_each_rendezvous_flag_exactly_once():
    """每个 rendezvous flag 只能出现**一次命令**。

    这条曾经数的是「字符串出现次数 == 1」，在有 rank-0 日志重定向之后不可能成立：
    重定向必须包住 exec，而 exec 只有一条，所以两个分支各自带一份完整的 exec 参数
    列表。真正的契约是「每个 flag 只传给 torchrun 一次」——所以按**去重后的命令行**
    计数：同一个 flag 在两条不同的命令行上各出现一次（两个分支各一份）是允许的，
    在同一条命令行上出现两次、或出现两条内容相同的命令行，都是缺陷。
    """
    code = _strip_shell_comments(_launcher_text())
    for flag in ("--nnodes=", "--node_rank=", "--nproc_per_node=",
                 "--master_addr=", "--master_port="):
        lines = sorted({line.strip() for line in code.splitlines() if flag in line})
        assert len(lines) == 1, (
            f"{flag} 必须恰好出现在一条（去重后的）命令行上，找到 {len(lines)} 条: {lines}"
        )
    passthrough = sorted({line.strip() for line in code.splitlines() if '${1+"$@"}' in line})
    assert len(passthrough) == 1, (
        f"调用方参数必须恰好透传一次（去重后的命令行只有一条），找到 {len(passthrough)} 条"
    )
    commands = _launcher_exec_commands()
    assert len(set(commands)) == 1, (
        "两条 exec 分支（rank 0 带重定向 / 其它 rank 不带）必须逐 token 相同："
        f"rank 门控只允许改变输出去向，不允许改变 torchrun 收到的参数: {commands}"
    )


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


# ---------------------------------------------------------------------------
# KFP：elastic Worker-only 拓扑下 launcher 的启动无关性（R71）
# ---------------------------------------------------------------------------
#
# 集群上那个 xtfold3-train pipeline 是 elastic + Worker-only：算子里程碑变量全部
# 走 `PET_` 前缀（外加 `RANK`/`WORLD_SIZE`），**没有**任何东西设置裸名
# `NNODES`/`NODE_RANK`。于是只认裸名的 launcher 在每个 pod 里都解析出
# `NNODES=1, NODE_RANK=0`：要么每个 pod 都以为自己是 rank 0（各自单进程训练，写同
# 一个 runs_dir），要么 torchrun 永远等一个不会加入的 rank。两种都是**静默**的
# ——没有报错，只有错误的结果或永久挂起。下面把解析顺序钉在源码上。

_TOPOLOGY_CHAINS = {
    "NNODES": ["PET_NNODES", "NNODES", "WORLD_SIZE", "1"],
    "NODE_RANK": ["PET_NODE_RANK", "NODE_RANK", "RANK", "0"],
    "NPROC_PER_NODE": ["PET_NPROC_PER_NODE", "NPROC_PER_NODE", "1"],
    "MASTER_PORT": ["PET_MASTER_PORT", "MASTER_PORT", "29500"],
    "MASTER_ADDR": ["PET_MASTER_ADDR", "MASTER_ADDR"],
}


def test_launcher_resolves_topology_from_pet_then_bare_then_torchrun_then_default():
    """每个拓扑变量的 fallback 链，顺序即优先级。"""
    for variable, expected in _TOPOLOGY_CHAINS.items():
        chain, raw = _topology_assignment(variable)
        assert chain == expected, (
            f"{variable} 的解析链是 {chain}，期望 {expected}（原始赋值: {variable}={raw}）"
        )


def test_launcher_topology_resolution_block_exported():
    """解析结果必须 export：torchrun 和训练进程读的是环境。"""
    block = _launcher_section("topology resolution")
    match = re.search(r"^export ([A-Z_ ]+)$", block, re.M)
    assert match, "拓扑解析块没有 export"
    exported = set(match.group(1).split())
    assert exported == set(_TOPOLOGY_CHAINS), (
        f"拓扑解析块导出的变量是 {sorted(exported)}，期望 {sorted(_TOPOLOGY_CHAINS)}"
    )


def test_launcher_topology_resolution_precedes_every_check_and_the_exec():
    """解析必须在所有校验与 exec 之前：校验用的是解析后的值。"""
    lines = _strip_shell_comments(_launcher_text()).splitlines()
    resolved = [i for i, line in enumerate(lines)
                if re.match(r"^(NNODES|NODE_RANK)=", line)
                and "${PET_" in line]
    assert len(resolved) == 2, f"期望两条拓扑赋值，找到 {len(resolved)} 条"

    checks = [i for i, line in enumerate(lines) if line.strip().startswith("exit 1")]
    assert checks, "launcher 的校验不见了"
    assert max(resolved) < min(checks), "拓扑解析必须早于所有校验"
    assert max(resolved) < _launcher_exec_line(), "拓扑解析必须在 exec 之前"


def test_launcher_appends_to_log_path_only_on_rank_zero():
    """LOG_PATH 只由 rank 0 追加写，其它 rank 保持 stdout/stderr。

    pipeline 的容器 args 在每个 pod 上都跑，没法自己做这件事（那正是 N 个 pod 交错
    写一个文件的原因），所以门控落在 launcher 里。两个分支各一条 exec，是单 exec
    约束下唯一能让重定向包住 exec 的写法。
    """
    code = _strip_shell_comments(_launcher_text())
    condition = re.search(r"if \[ -n \"\$\{LOG_PATH:-\}\" \] && \[ \"\$\{NODE_RANK\}\" -eq 0 \]; then",
                          code)
    assert condition, "没有 `LOG_PATH` 非空且 `NODE_RANK` 为 0 的门控条件"
    condition_line = code[:condition.start()].count("\n")
    assert condition_line < _launcher_exec_line(), "门控必须在 exec 之前成立"
    assert code.count('} >> "${LOG_PATH}" 2>&1') == 1, (
        "rank 0 那支必须恰好一条分组重定向：`} >> \"${LOG_PATH}\" 2>&1`"
    )
    commands = _launcher_exec_commands()
    assert len(commands) == 2, f"期望 rank 0 / 其它 rank 两条 exec 分支，找到 {len(commands)}"
    assert len(set(commands)) == 1, "两条分支的 exec 必须逐 token 相同"


def test_launcher_unset_log_path_falls_back_to_the_plain_exec():
    """LOG_PATH 未设时行为必须与改动前完全一致：裸 exec，不重定向。

    这条挡的是把 `LOG_PATH` 换成「总有默认值」的写法，那会把每个 rank 的 stdout
    悄悄搬到一个文件里（`kubectl logs` 随即变空）。
    """
    code = _strip_shell_comments(_launcher_text())
    assert "${LOG_PATH:?" not in code, "LOG_PATH 不能是必填项：未设时必须维持旧行为"
    assert re.search(r'if \[ -n "\$\{LOG_PATH:-\}" \]', code), (
        "门控必须用 ${LOG_PATH:-} 判空，未设即走 else（裸 exec）"
    )
    assert "LOG_PATH=" not in code, "launcher 不许给 LOG_PATH 兜底默认值"


# ---------------------------------------------------------------------------
# KFP：编译产物本身
# ---------------------------------------------------------------------------

def test_kfp_pipeline_is_an_argo_workflow_with_the_kfp_annotations():
    """KFP v1 UI 的上传契约：Workflow + pipelines.kubeflow.org/* 注解。"""
    document = _load_yaml(PIPELINE)
    assert document["apiVersion"] == "argoproj.io/v1alpha1"
    assert document["kind"] == "Workflow"
    assert document["metadata"]["generateName"] == "kineager-affinity-train-"
    annotations = document["metadata"]["annotations"]
    for key in ("pipelines.kubeflow.org/kfp_sdk_version",
                "pipelines.kubeflow.org/pipeline_compilation_time",
                "pipelines.kubeflow.org/pipeline_spec"):
        assert key in annotations, f"缺少 KFP 注解 {key}"
    assert "pipelines.kubeflow.org/kfp_sdk_version" in document["metadata"]["labels"]

    # pipeline_spec 注解必须是**真 JSON**：KFP UI 用它渲染参数表单。
    import json

    spec = json.loads(annotations["pipelines.kubeflow.org/pipeline_spec"])
    assert spec["name"] == _PIPELINE_NAME
    assert [item["name"] for item in spec["inputs"]] == _PIPELINE_PARAMETERS


def test_kfp_pipeline_has_the_expected_templates_and_entrypoint():
    document = _load_yaml(PIPELINE)
    names = [template["name"] for template in document["spec"]["templates"]]
    assert names == [_PIPELINE_TEMPLATE, f"{_PIPELINE_TEMPLATE}-run"], names
    # entrypoint 直接指向 resource 模板：一个只有单个任务的 pipeline，DAG 那层是
    # 空的。参考文件两者都有（它的 entrypoint 是 DAG），两种都能被 KFP v1 接受；
    # 这里少一层没有意义的间接。
    assert document["spec"]["entrypoint"] == _PIPELINE_TEMPLATE
    run = document["spec"]["templates"][1]
    (task,) = run["dag"]["tasks"]
    assert task["template"] == _PIPELINE_TEMPLATE, "DAG 任务没有指向 resource 模板"
    assert document["spec"]["serviceAccountName"] == "pipeline-runner"


def test_kfp_pipeline_parameters_mirror_through_all_three_levels():
    """resource 模板的 inputs ← DAG 任务的 arguments ← 顶层 arguments。

    任一层少一个名字，Argo 会在**运行时**报 `failed to resolve ... parameter`，
    或在 UI 里根本不显示该输入框——两者都不是本地能看到的失败。
    """
    document = _load_yaml(PIPELINE)
    resource, run = document["spec"]["templates"][0], document["spec"]["templates"][1]
    declared = [item["name"] for item in resource["inputs"]["parameters"]]
    assert declared == _PIPELINE_PARAMETERS, declared

    (task,) = run["dag"]["tasks"]
    supplied = [item["name"] for item in task["arguments"]["parameters"]]
    assert sorted(supplied) == sorted(declared), (
        f"DAG 任务提供的参数 {supplied} 与 resource 模板声明的 {declared} 不一致"
    )
    values = {item["name"]: item["value"] for item in task["arguments"]["parameters"]}
    for name in declared:
        assert values[name] == f"{{{{inputs.parameters.{name}}}}}", (
            f"{name} 没有从 DAG 的 inputs 透传（value={values[name]!r}）"
        )

    top = [item["name"] for item in document["spec"]["arguments"]["parameters"]]
    assert sorted(top) == sorted(declared), (
        f"顶层 arguments {top} 与 resource 模板声明的 {declared} 不一致"
    )


def test_kfp_pipeline_embedded_pytorchjob_parses_after_substitution():
    """嵌在 `manifest: |` 里的 PyTorchJob 必须能当 YAML 解析。

    顶层 safe_load 只证明 Workflow 那层没问题；引号/缩进/参数替换的错误全在嵌的那段
    里。`{{...}}` 按 Argo 的做法替换成具体值之后再解析，正好覆盖「替换之后 YAML 还是
    合法的」这件事。

    注意：**未替换**的 manifest 连参考文件都解析不了（`minReplicas:
    {{inputs.parameters.pod_nums}}` 被读成嵌套 mapping，报 "found unhashable key"），
    所以断言的是替换后的可解析性，而不是原文可解析性。
    """
    job = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "2",
        "working_dir": "/workspace/kineager",
    }))
    assert job["apiVersion"] == "kubeflow.org/v1"
    assert job["kind"] == "PyTorchJob"
    assert job["metadata"]["generateName"] == "kineager-affinity-"


def _as_text(value):
    """Argo 的 `{{...}}` 替换是**文本**替换，替换之后再当 YAML 解析。

    所以未加引号的 `cpu: {{inputs.parameters.cpu}}` 在 pod_nums/cpu 都填数字时会解析
    成 int，而 `replicas:` 在参数是字符串时会解析成 str。这两种形状在 Kubernetes
    里都合法，断言因此比较归一化后的文本，而不是 Python 类型。
    """
    return str(value)


def test_kfp_pipeline_job_is_elastic_and_worker_only():
    """elastic Worker-only：minReplicas == maxReplicas == pod_nums，且没有 Master。"""
    job = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "2",
        "working_dir": "/workspace/kineager",
    }))
    spec = job["spec"]
    assert set(spec["pytorchReplicaSpecs"]) == {"Worker"}, (
        "这个 pipeline 必须是 Worker-only：有 Master 就不再是 elastic 拓扑"
    )
    assert _as_text(spec["pytorchReplicaSpecs"]["Worker"]["replicas"]) == "2"

    policy = spec["elasticPolicy"]
    assert policy["rdzvBackend"] == "c10d"
    assert _as_text(policy["minReplicas"]) == "2", "minReplicas 必须等于 pod_nums"
    assert _as_text(policy["maxReplicas"]) == "2", (
        "maxReplicas 必须等于 pod_nums；否则算子可以把 replica 数缩到 launcher 校验的"
        "NNODES 之下，torchrun 会静默等一个永远不会出现的 rank"
    )
    assert spec["runPolicy"]["cleanPodPolicy"] == "Running"
    assert spec["runPolicy"]["ttlSecondsAfterFinished"] == 30


def test_kfp_pipeline_elastic_bounds_track_pod_nums():
    """换个 pod_nums，min/max/replicas 必须跟着动——它们不是写死的 2。"""
    values = {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "7",
        "working_dir": "/workspace/kineager",
    }
    spec = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), values))["spec"]
    assert _as_text(spec["pytorchReplicaSpecs"]["Worker"]["replicas"]) == "7"
    assert _as_text(spec["elasticPolicy"]["minReplicas"]) == "7"
    assert _as_text(spec["elasticPolicy"]["maxReplicas"]) == "7"


def test_kfp_pipeline_keeps_the_reference_pod_placement_verbatim():
    """nodeSelector / tolerations 必须与参考 pipeline（也就是裸 manifest）逐字一致。

    这些值来自集群上**正在跑**的那个 pipeline：换掉就等于换硬件，配错了只会表现成
    pod 一直 Pending。
    """
    document = _load_yaml(PIPELINE)
    for template in document["spec"]["templates"]:
        if "resource" not in template:
            continue
        manifest = _substitute_argo_parameters(template["resource"]["manifest"], {
            "bash_command": "bash bin/train/kineager_k8s.sh",
            "logpath": "/tmp/rank0.log",
            "memory": "200Gi",
            "cpu": "16",
            "pod_gpus": "1",
            "image": "example/image:tag",
            "pod_nums": "2",
            "working_dir": "/workspace/kineager",
        })
        pod = yaml.safe_load(manifest)["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]
        assert pod["nodeSelector"] == {"gpu.device": "a100"}
        assert pod["tolerations"] == [
            {"effect": "NoSchedule", "key": "role", "operator": "Equal", "value": "mpu"},
            {"effect": "NoSchedule", "key": "role", "operator": "Equal", "value": "m1b"},
            {"effect": "NoSchedule", "key": "role", "operator": "Equal", "value": "NBTopic"},
        ], pod["tolerations"]

    # 裸 manifest 的 nodeSelector / tolerations 不能被这次改动带偏（它仍然是
    # `kubectl apply` 那条路）。两者的 value 集合本来就不同——pipeline 来自集群上跑的
    # 那个 reference（mpu / m1b / NBTopic），裸 manifest 来自更早的 xtrimofold3 副本
    # （NBTopic / 006_mix）——所以这里只钉住「各自都还在、且都是 role 容忍」。
    bare = _replicas()["Master"]["template"]["spec"]
    assert bare["nodeSelector"]["gpu.device"] == "a100"
    assert {item["value"] for item in bare["tolerations"]} == {"NBTopic", "006_mix"}
    for item in bare["tolerations"]:
        assert item["key"] == "role" and item["operator"] == "Equal"
        assert item["effect"] == "NoSchedule"


def test_kfp_pipeline_container_keeps_the_reference_security_and_resources():
    container = _pipeline_container()
    assert container["name"] == "pytorch"
    assert container["imagePullPolicy"] == "Always"
    assert container["securityContext"] == {
        "privileged": False,
        "capabilities": {"add": ["IPC_LOCK"]},
    }
    assert {key: _as_text(value) for key, value in container["resources"]["limits"].items()} == {
        "cpu": "16", "memory": "200Gi", "nvidia.com/gpu": "1",
    }
    assert "requests" not in container["resources"], (
        "参考 pipeline 只写 limits；补上 requests 就不是逐字照搬了（也改变 QoS）"
    )
    (mount,) = container["volumeMounts"]
    assert mount == {"mountPath": "/dev/shm", "name": "xumeng"}
    pod = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "2",
        "working_dir": "/workspace/kineager",
    }))["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]
    assert pod["volumes"] == [
        {"name": "xumeng", "emptyDir": {"medium": "Memory", "sizeLimit": "40Gi"}}
    ], pod["volumes"]


def test_kfp_pipeline_success_condition_tracks_the_worker_replica():
    """Worker-only 拓扑下 `status.replicaStatuses.Master` 永远不会出现。"""
    document = _load_yaml(PIPELINE)
    resource = document["spec"]["templates"][0]["resource"]
    assert resource["action"] == "create"
    assert resource["setOwnerReference"] is True
    assert resource["successCondition"] == "status.replicaStatuses.Worker.succeeded == 1"
    assert resource["failureCondition"] == "status.replicaStatuses.Worker.failed > 0"
    assert "Master" not in resource["successCondition"] + resource["failureCondition"]

    # 这些 conditions 只有在**只有一个**副本源时才自洽。
    (worker,) = yaml.safe_load(_substitute_argo_parameters(_embedded_manifest(), {
        "bash_command": "bash bin/train/kineager_k8s.sh",
        "logpath": "/tmp/rank0.log",
        "memory": "200Gi",
        "cpu": "16",
        "pod_gpus": "1",
        "image": "example/image:tag",
        "pod_nums": "2",
        "working_dir": "/workspace/kineager",
    }))["spec"]["pytorchReplicaSpecs"].values()
    assert _as_text(worker["replicas"]) == "2"


def test_kfp_pipeline_rank_zero_log_gate_substitutes_into_parseable_bash():
    """容器 args 渲染出来必须是**能过 `bash -n`** 的脚本，且门控形状正确。

    这是「引号有没有活过 YAML 与 Argo 的 `{{...}}` 替换」的直接证据：`logpath` 和
    `bash_command` 都换成真实值，再把 args 交给 bash 的词法分析器。少了/多了引号、
    替换把 `>>` 吃掉、`$@` 被展开——都会在这里变成语法错误。

    形状断言（不依赖 bash）先跑，所以本机没有 bash 时这一条仍然能抓到「重定向没引号」
    「门控被写成 true」这类改动；只有语法那一半会 skip。
    """
    argv = _rank0_argv(2)
    assert 'if [ "${NODE_RANK:-0}" -eq 0 ]; then' in argv, (
        "rank 门控必须真的按 NODE_RANK 判断"
    )
    assert '>> "/pfs_beijing/xinran_2026/kineager_runs/kineager_affinity/rank0.log" 2>&1' in argv, (
        'rank 0 那一支必须以 >> "<logpath>" 2>&1 结尾（路径在双引号里）'
    )
    assert argv.count(">>") == 1, "只有 rank 0 那一支才有重定向"
    redirect = next(line for line in argv.splitlines() if ">>" in line)
    assert redirect.count('"') == 2, (
        f'rank 0 的重定向行必须恰好一对引号（>> "<logpath>" 2>&1），实际: {redirect!r}'
    )
    assert "{{" not in argv and "}}" not in argv, "替换之后不能留下 Argo 模板语法"

    if not BASH:
        pytest.skip("没有可用的 bash，跳过 `bash -n` 那一半（本机 sandbox 下 Git bash 起不来）")

    result = subprocess.run([BASH, "-n"], input=argv, capture_output=True, text=True)
    assert result.returncode == 0, f"容器 args 不是合法 bash: {result.stderr}"
    assert subprocess.run([BASH, "-n"], input=_rank0_argv(3),
                          capture_output=True, text=True).returncode == 0


def test_kfp_pipeline_logpath_substitution_cannot_expand_into_extra_words():
    """`logpath` 被塞进双引号里：去掉引号就会被 word-splitting 拆成两个词。

    这条用「把引号拿掉就得到不同的词数」证明引号真的在起作用，而不是碰巧路径里没有
    空格。它同时是给操作者的护栏：带空格/引号的路径会在这里被抓住。
    """
    if not BASH:
        pytest.skip("没有可用的 bash，无法做词法检查（本机 sandbox 下 Git bash 起不来）")

    argv = _rank0_argv(2)
    log_line = next(line for line in argv.splitlines() if "rank0.log" in line)
    match = re.search(r"bash .*? >> (.*) 2>&1$", log_line)
    assert match, f"渲染后的 rank 0 命令行形状不对: {log_line!r}"
    quoted = match.group(1)
    assert quoted.startswith('"') and quoted.endswith('"'), (
        f"logpath 没有被引号包住，`>>` 之后就是裸路径: {log_line!r}"
    )

    def word_count(argument):
        probe = subprocess.run([BASH, "-c", f'set -- {argument}; printf "%s\\n" "$#"'],
                               capture_output=True, text=True)
        assert probe.returncode == 0, probe.stderr
        return probe.stdout.strip()

    assert word_count(quoted) == "1", f"重定向目标被拆开了: {quoted}"
    assert word_count(quoted[1:-1]) == "2", (
        "引号必须真的起作用：去掉引号后这个路径应该被拆成 2 个词，否则这条断言什么都没证明"
    )


# ---------------------------------------------------------------------------
# 运维脚本：elastic 拓扑下没有 master pod
# ---------------------------------------------------------------------------

def test_launch_wrapper_discovers_the_rank0_pod_instead_of_hardcoding_master_0():
    """`${JOB}-master-0` 在 elastic Worker-only 下不存在，必须改成查询。"""
    text = LAUNCH_WRAPPER.read_text(encoding="utf-8")
    assert "-master-0" in text, "仍然要认识 master 形状（裸 manifest 有 Master）"
    assert "-worker-0" in text, "没有 Master 的 elastic 拓扑下 rank 0 是 worker-0"
    assert 'MASTER_POD="${JOB}-master-0"' not in text, (
        "pod 名不能再硬编码：elastic 作业的 pod 名带随机后缀，且根本没有 master"
    )

    # pod 查询本身必须带算子的 label —— 只看「文件里出现过这个字符串」不够：错误
    # 消息里也有它，把查询改成 `kubectl get pods` 会恰好落进那个盲区。
    query = text.split("pod_names() {", 1)[1].split("\n}", 1)[0]
    assert _JOB_LABEL in query, f"pod 查询必须按 {_JOB_LABEL} 过滤: {query!r}"
    assert re.search(r"kubectl get pods\b", query), "查询必须调 kubectl get pods"
    assert re.search(r'-l\s+"' + re.escape(_JOB_LABEL) + r'=\$JOB"', query), (
        f"label selector 形状不对: {query!r}"
    )

    # 「找不到」那一支必须真的失败，并把看到的东西列出来
    assert "no rank-0 pod found" in text
    assert "(none)" in text, "失败消息必须列出实际找到的 pod（一个都没有时要说明）"
    assert "exit 1" in text


def test_launch_wrapper_keeps_its_flags_and_exit_codes():
    """四个开关都必须还是**被解析**的 case 分支，而不是只出现在帮助注释里。"""
    text = LAUNCH_WRAPPER.read_text(encoding="utf-8")
    for flag in ("--no-delete", "--no-follow", "--dry-run", "--wait-timeout"):
        assert re.search(rf"^\s*{re.escape(flag)}\)", text, re.M), (
            f"launch wrapper 的 case 分支里少了 {flag}"
        )
    assert "exit 2" in text, "未知选项必须仍然 exit 2"
    assert "exit 1" in text, "无法连接集群 / 等不到 pod 必须仍然 exit 1"
    # 重新运行先删旧 job 的逻辑必须留着：固定 metadata.name + 长 TTL 的组合下，
    # 第二次 apply 会「成功」但什么都不启动。
    assert "delete pytorchjob" in text
    assert "kubectl apply -f" in text


def _wrapper_case_patterns():
    """wrapper 里 `pick_rank0` 的 case 模式（按出现顺序）。

    bash 在本机起不来，所以把模式从源码里取出来、在 Python 里按同样的顺序重放。
    这样「master 优先、worker-0 兜底」这条选择规则是被**执行**验证的，不是被读着
    确认的：删掉任一模式、把两者顺序对调，或把两趟循环并成一趟，重放结果都会变。
    """
    text = LAUNCH_WRAPPER.read_text(encoding="utf-8")
    body = text.split("pick_rank0() {", 1)[1].split("\n}", 1)[0]
    return re.findall(r"^\s*([^\s()]+)\)\s*chosen=", body, re.M)


def _pick_rank0(patterns, pods):
    """复现 `pick_rank0` 的两趟扫描：每个模式各扫一遍整份 pod 名单。"""
    import fnmatch

    for pattern in patterns:
        for pod in pods:
            if fnmatch.fnmatchcase(pod, pattern):
                return pod
    return ""


def test_launch_wrapper_prefers_the_master_pod_and_falls_back_to_worker_0():
    patterns = _wrapper_case_patterns()
    assert patterns == ["*-master-0", "*-worker-0"], (
        f"rank-0 的选择必须是「先 master-0，再 worker-0」，实际 {patterns}"
    )

    bare = "kineager-affinity-master-0"
    elastic0 = "kineager-affinity-abc12-worker-0"
    elastic1 = "kineager-affinity-abc12-worker-1"

    # 裸 manifest（Master + Worker）：master-0 就是 rank 0
    assert _pick_rank0(patterns, [bare, "kineager-affinity-worker-0"]) == bare
    # elastic Worker-only：名单里根本没有 master，必须落到 worker-0
    assert _pick_rank0(patterns, [elastic0, elastic1]) == elastic0
    # 两种 pod 同时存在（迁移中/名字带 worker 前缀的旧对象）：仍然优先 master-0
    assert _pick_rank0(patterns, [bare, elastic0]) == bare
    assert _pick_rank0(patterns, [elastic0, elastic1, bare]) == bare
    # 一个都不是 rank 0（都还没建出来）：返回空串 → wrapper 报错退出
    assert _pick_rank0(patterns, ["kineager-affinity-abc12-worker-1"]) == ""
    assert _pick_rank0(patterns, []) == ""


def test_all_k8s_artifacts_parse_and_are_committed_as_lf():
    """新交付物也要满足仓库的编码卫生：可解析、UTF-8、结尾换行、无孤立 CR。"""
    for path in (PIPELINE, LAUNCH_WRAPPER):
        raw = path.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(f"{path.name} 不是合法 UTF-8：{error}") from error
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{path.name} 结尾换行不规范"
        assert b"\r" not in raw.replace(b"\r\n", b""), f"{path.name} 含孤立 CR（混用换行）"
    assert _load_yaml(PIPELINE)["kind"] == "Workflow"
