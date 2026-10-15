#!/usr/bin/env bash
#
# KinEAGER PDBbind-v2020 affinity training - k8s / PyTorchJob entrypoint.
#
# Same rendezvous contract as xtrimofold3's
# bin/train/run_k8s_protenix_v2_multitask_PL_k8s.sh:
#   * one GPU per pod (NPROC_PER_NODE=1),
#   * the orchestrator injects the topology (PET_* from the training-operator,
#     or NNODES / NODE_RANK / MASTER_ADDR; see the resolution block below),
#   * this script validates the topology and execs torch.distributed.run.
#
# The topology comes from the environment; training hyperparameters (-epochs,
# -batch_size, -lr, --split_*, ...) are NEVER hardcoded here. The caller appends
# them through "$@" - see yamls/kubeflow/kineager_affinity_k8s.yaml, which is the
# k8s caller. Because argparse keeps the last occurrence of a flag, anything in
# "$@" also overrides the launcher defaults below.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# 与 xtrimofold3 同类脚本对齐：显式把仓库根塞进 PYTHONPATH。
# 训练脚本经 `python3 -m torch.distributed.run` 启动时，sys.path[0] 是**当前
# 工作目录**而不是脚本目录——不设 PYTHONPATH 的话只有恰好在仓库根下执行才能
# import models / data_affinity（k8s 靠 workingDir 保证，独立运行很容易踩坑）。
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# --- topology resolution, launch-agnostic -----------------------------------
# One resolution for all three ways this script is started, so a run cannot
# silently degrade to a single rank depending on the *launcher*:
#
#   1. the training-operator's own variables. They come FIRST because the
#      elastic, Worker-only KFP pipeline (yamls/kubeflow/kineager_affinity_pipeline.yaml)
#      injects topology ONLY under this prefix (plus RANK/WORLD_SIZE): nothing in
#      that pipeline sets the bare names, so with the old bare-name-only
#      resolution every pod fell back to NNODES=1 / NODE_RANK=0. That is not an
#      error the run can notice - DDP just wraps with world_size=1, every pod
#      runs an independent replica writing the SAME runs_dir/train_data_dir, and
#      the launcher's own `NODE_RANK < NNODES` check still passes.
#   2. the bare, Kubernetes-style names: what
#      yamls/kubeflow/kineager_affinity_k8s.yaml exports and what the operator
#      leaves in place for the non-elastic (Master + Worker) topology.
#   3. RANK / WORLD_SIZE: torchrun's own names - what a `torchrun` or
#      `torch.distributed.run` parent passes down. Their values are 0-based, the
#      same convention as NODE_RANK, so they map straight across.
#   4. the historical defaults: a single local process.
#
# Every expansion carries a default, so all five names are always set and the
# rest of the script stays safe under `set -u`.
NNODES="${PET_NNODES:-${NNODES:-${WORLD_SIZE:-1}}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-${RANK:-0}}}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
export NNODES NODE_RANK NPROC_PER_NODE MASTER_PORT MASTER_ADDR

# The elastic training-operator reports the node count as torchrun's own range
# syntax "MIN:MAX" (observed as PET_NNODES=1:1 on the cluster, 2026-10-15).
# This deployment runs a FIXED topology, so only a collapsed range (MIN == MAX)
# is meaningful: use the number. An open range cannot be served by the static
# rendezvous below - guessing a count would either hang the run or, far worse,
# let every pod train standalone into the same runs_dir - so it must stop.
if [[ "${NNODES}" == *:* ]]; then
    nnodes_min="${NNODES%%:*}"
    nnodes_max="${NNODES##*:}"
    if [[ "${nnodes_min}" =~ ^[0-9]+$ && "${nnodes_max}" =~ ^[0-9]+$ && "${nnodes_min}" == "${nnodes_max}" ]]; then
        echo "[kineager] NNODES=${NNODES} (elastic range) -> fixed ${nnodes_max}"
        NNODES="${nnodes_max}"
    else
        echo "NNODES='${NNODES}' is an open elastic range; a fixed topology is required - set min_replicas == max_replicas == pod_nums" >&2
        exit 1
    fi
fi

# --- topology validation: all of it runs before the exec at the bottom -------
for integer_var in NNODES NODE_RANK NPROC_PER_NODE MASTER_PORT; do
    if ! [[ "${!integer_var}" =~ ^[0-9]+$ ]]; then
        echo "${integer_var} must be a non-negative integer (got ${!integer_var})" >&2
        exit 1
    fi
done
if [ "${NNODES}" -lt 1 ]; then
    echo "NNODES (${NNODES}) must be at least 1" >&2
    exit 1
fi
if [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "NODE_RANK (${NODE_RANK}) must be in [0, $((NNODES - 1))]" >&2
    exit 1
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must be at least 1" >&2
    exit 1
fi
if [ "${MASTER_PORT}" -lt 1 ] || [ "${MASTER_PORT}" -gt 65535 ]; then
    echo "MASTER_PORT (${MASTER_PORT}) must be in [1, 65535]" >&2
    exit 1
fi

if [ "${NNODES}" -gt 1 ]; then
    # multi-node rendezvous: the PyTorchJob must have injected MASTER_ADDR
    if [ -z "${MASTER_ADDR:-}" ]; then
        echo "MASTER_ADDR must be set when NNODES > 1" >&2
        exit 1
    fi
    master_addr="${MASTER_ADDR}"
else
    master_addr="${MASTER_ADDR:-127.0.0.1}"
fi

# --- path profile ------------------------------------------------------------
# path_config.resolve_runtime_paths() falls back to the platform default
# ("autodl" on Linux) when CATA_PROFILE is unset, and *nothing else* would select
# the k8s profile: an unset profile silently reads the training CSV, the run
# directories and the backbone snapshots from the wrong profile. This script is
# the k8s entrypoint by name, so default it here; the ${VAR:-...} form keeps it
# overridable for a run that deliberately wants another profile.
export CATA_PROFILE="${CATA_PROFILE:-k8s}"

# --- write-path scoping: ${USER} must be real, exported and filled in -------
# Only the k8s profile uses ${USER}, and path_config._current_user_name()
# resolves it as USER -> USERNAME -> the literal "kineager" (the R63 order, so
# this guard accepts exactly the same two forms). A pod sets neither variable, so
# an unset value silently resolves runs_dir AND train_data_dir to the SHARED
# /pfs_beijing/kineager/kineager_runs. train_data_dir is a write target
# (train_epoch mkdirs it and np.saves the per-epoch preds_epoch*/labels_epoch*.npy
# there), so two concurrent users would overwrite each other's artifacts while
# both runs look healthy -- far harder to notice than a refusal to start.
#
# Two ways past a naive check, both closed here:
#   * this guard reads a SHELL VARIABLE (`${USER:-}` sees one whether or not it is
#     exported), while the exec'd trainer inherits only the ENVIRONMENT and
#     path_config reads os.getenv("USER"). Any route that makes those two disagree
#     defeats the check: a SOURCED invocation (`USER=alice; source
#     bin/train/kineager_k8s.sh`, where the caller's shell variables are visible to
#     the guard but not to the exec'd child), or a BASH_ENV/startup file that sets
#     USER without exporting it. (A plain `USER=alice; bash ...` is a different
#     case: the child bash inherits only the environment, so the guard sees
#     nothing and refuses.) The validated value is therefore exported below --
#     the value that was checked becomes the value the trainer receives.
#   * the manifest's own placeholder is a non-empty string, so an operator who
#     forgets to fill it in would otherwise create a directory named after the
#     placeholder. Kubernetes accepts any string as an env value; this rejects it.
if [ "${CATA_PROFILE}" = "k8s" ]; then
    kineager_user="${USER:-${USERNAME:-}}"
    case "${kineager_user}" in
        "<REPLACE_ME"*)
            echo "USER is still the manifest placeholder (${kineager_user}) but CATA_PROFILE=k8s: fill in the USER entry in the PyTorchJob's env block. The placeholder is a non-empty string, so it would otherwise become a literal path component under /pfs_beijing/." >&2
            exit 1
            ;;
    esac
    if [ -z "${kineager_user}" ]; then
        echo "Neither USER nor USERNAME is set but CATA_PROFILE=k8s: path_config expands \${USER} in runs_dir/train_data_dir, and with neither set it falls back to the SHARED /pfs_beijing/kineager/kineager_runs, where concurrent runs overwrite each other's per-epoch .npy artifacts. Export USER=<your username> (or USERNAME), or fill in the manifest's USER entry." >&2
        exit 1
    fi
    export USER="${kineager_user}"
fi

# --- required inputs (fail fast, with the variable name in the message) ------
export TASK="${TASK:-affinity}"
export TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV must point to the PDBbind affinity CSV}"
export KINEAGER_ESMC_PATH="${KINEAGER_ESMC_PATH:?KINEAGER_ESMC_PATH must point to a local ESMC-6B directory}"
export KINEAGER_MOLFORMER_PATH="${KINEAGER_MOLFORMER_PATH:?KINEAGER_MOLFORMER_PATH must point to a local MoLFormer snapshot}"

# --- allocator and NCCL ------------------------------------------------------
# The names and defaults of the NCCL block match the PyTorchJob manifest
# (yamls/kubeflow/kineager_affinity_k8s.yaml) and the xtrimofold3 reference, so
# this script is also usable standalone; every value stays overridable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-5}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[kineager] task=${TASK} profile=${CATA_PROFILE} nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${master_addr}:${MASTER_PORT}"

# --- launch ------------------------------------------------------------------
# LOG_PATH: when it is set, ONLY rank 0 appends to it and the other ranks stay on
# stdout/stderr (the same single-writer pattern as the xtrimofold3 reference).
# The elastic pipeline's container args cannot express this by themselves - a
# plain `bash ... >> ${logpath} 2>&1` wraps EVERY pod, so N pods interleave into
# one file and the result is unreadable. Doing it here also covers the bare
# manifest and a hand-run shell.
#
# The redirect is composed with the exec instead of opening the file separately,
# so two properties survive:
#   * rank 0's stdout/stderr are the log file BEFORE the exec, therefore torchrun
#     and the trainer inherit the redirect (writing to fd 1/2 needs no LOG_PATH
#     knowledge downstream). `exec 1>>file 2>&1` would leak the open descriptors
#     into every child; this does not rely on that at all.
#   * there is still exactly ONE exec and it is still the last thing the script
#     does, so the trainer - not this wrapper - is the container's PID 1 and
#     k8s SIGTERM / preemption reaches torchrun (see the long note below).
# `{ ...; }` needs the semicolon before the closing brace; the `2>&1` applies to
# the group, i.e. after the append redirect has already moved fd 1.
#
# Unset LOG_PATH => the `else` branch => behaviour is exactly what it was before
# this block existed (both streams inherited from the container).
#
# exec: this wrapper is replaced, so k8s SIGTERM / preemption reaches torchrun
# instead of being swallowed here. The PyTorchJob also calls this script through
# `exec bash bin/train/kineager_k8s.sh`, which is what puts torchrun - not the
# outer bash - at the container's PID 1; a PID-namespace init ignores
# default-action signals, so a plain child would never see SIGTERM.
#
# The caller's arguments are appended with ${1+"$@"} rather than "$@": under
# `set -u`, bash older than 4.4 reports "$@" as unbound when there are no
# positional parameters, while ${1+"$@"} expands to zero words in that case and
# to the (quoted) arguments otherwise - so a bare invocation still works.
# -esm / -molt5 are the training script's legacy aliases for the two backbone
# snapshot paths.
if [ -n "${LOG_PATH:-}" ] && [ "${NODE_RANK}" -eq 0 ]; then
    {
        exec python3 -m torch.distributed.run \
            --nnodes="${NNODES}" \
            --node_rank="${NODE_RANK}" \
            --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${master_addr}" \
            --master_port="${MASTER_PORT}" \
            "${ROOT}/main_train_predictor_multigpu.py" \
                --task "${TASK}" \
                -csv "${TRAIN_CSV}" \
                -esm "${KINEAGER_ESMC_PATH}" \
                -molt5 "${KINEAGER_MOLFORMER_PATH}" \
                --lora_layer_policy "${LORA_LAYER_POLICY:-last_k}" \
                --lora_last_k "${LORA_LAST_K:-30}" \
                --grad_checkpoint_backbone "${GRAD_CHECKPOINT_BACKBONE:-off}" \
                --iters_to_accumulate "${ITERS_TO_ACCUMULATE:-${KINEAGER_ITERS_TO_ACCUMULATE:-1}}" \
                --num_workers "${NUM_WORKERS:-${KINEAGER_NUM_WORKERS:-8}}" \
                ${1+"$@"}
    } >> "${LOG_PATH}" 2>&1
else
    exec python3 -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${master_addr}" \
        --master_port="${MASTER_PORT}" \
        "${ROOT}/main_train_predictor_multigpu.py" \
            --task "${TASK}" \
            -csv "${TRAIN_CSV}" \
            -esm "${KINEAGER_ESMC_PATH}" \
            -molt5 "${KINEAGER_MOLFORMER_PATH}" \
            --lora_layer_policy "${LORA_LAYER_POLICY:-last_k}" \
            --lora_last_k "${LORA_LAST_K:-30}" \
            --grad_checkpoint_backbone "${GRAD_CHECKPOINT_BACKBONE:-off}" \
            --iters_to_accumulate "${ITERS_TO_ACCUMULATE:-${KINEAGER_ITERS_TO_ACCUMULATE:-1}}" \
            --num_workers "${NUM_WORKERS:-${KINEAGER_NUM_WORKERS:-8}}" \
            ${1+"$@"}
fi
