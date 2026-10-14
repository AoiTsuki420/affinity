#!/usr/bin/env bash
#
# KinEAGER PDBbind-v2020 affinity training - k8s / PyTorchJob entrypoint.
#
# Same rendezvous contract as xtrimofold3's
# bin/train/run_k8s_protenix_v2_multitask_PL_k8s.sh:
#   * one GPU per pod (NPROC_PER_NODE=1),
#   * the orchestrator injects NNODES / NODE_RANK / MASTER_ADDR,
#   * this script validates the topology and execs torch.distributed.run.
#
# The topology comes from the environment; training hyperparameters (-epochs,
# -batch_size, -lr, --split_*, ...) are NEVER hardcoded here. The caller appends
# them through "$@" - see yamls/kubeflow/kineager_affinity_k8s.yaml, which is the
# k8s caller. Because argparse keeps the last occurrence of a flag, anything in
# "$@" also overrides the launcher defaults below.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

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
