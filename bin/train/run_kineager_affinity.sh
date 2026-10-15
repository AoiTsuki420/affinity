#!/usr/bin/env bash
#
# One-line Kubeflow Pipelines entry for this deployment.
#
# The generic xtfold3-train pipeline only forwards bash_command; everything else
# this training needs - the deployment's env block, the hyperparameters of the
# current experiment, and the multi-rank prewarm - is wrapped here so the
# pipeline's bash_command can be just:
#
#     bash /pfs_beijing/xinran_2026/project/affinity/bin/train/run_kineager_affinity.sh
#
# Topology (NNODES / NODE_RANK / MASTER_ADDR, PET_* spellings included) is NOT
# set here: it comes from the pipeline and is resolved by kineager_k8s.sh, which
# fails fast when a multi-node launch has no MASTER_ADDR. Use the launcher's
# `[kineager] ... nnodes=... node_rank=...` line to confirm the world size
# (nnodes must equal pod_nums; each pod must report a different node_rank).
#
# The world size is NPROC_PER_NODE x pod_nums. NPROC_PER_NODE is auto-detected
# by default from the GPUs visible inside this container - K8s mounts exactly
# the pod_gpus GPUs it allocated, so it always matches the form's pod_gpus
# (8, 4, 6, ...); no script edit is needed when the GPU count changes. Export
# NPROC_PER_NODE to pin it explicitly - on the shared local node always run
# with `NPROC_PER_NODE=1 bash ...` because the whole node's GPUs are visible.
#
# Every value below can be overridden by exporting it before this script runs.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNS_DIR=/pfs_beijing/xinran_2026/general_protein_dir/kineager_runs

# One process per GPU: default to the GPU count visible in this container
# (== pod_gpus in the submission form); an explicit NPROC_PER_NODE wins.
if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    if [ "${NPROC_PER_NODE}" -lt 1 ]; then
        echo "cannot detect a GPU count via nvidia-smi; set NPROC_PER_NODE explicitly (= pod_gpus)" >&2
        exit 1
    fi
fi
export NPROC_PER_NODE
export USER="${USER:-xinran_2026}"
export CATA_RUNS_DIR="${CATA_RUNS_DIR:-$RUNS_DIR}"
export CATA_WEIGHTS_DIR="${CATA_WEIGHTS_DIR:-$RUNS_DIR}"
export CATA_TRAIN_DATA_DIR="${CATA_TRAIN_DATA_DIR:-$RUNS_DIR}"
export TRAIN_CSV="${TRAIN_CSV:-/pfs_beijing/xinran_2026/kineager_data/pdbbindv2020_affinity.csv}"
export KINEAGER_ESMC_PATH="${KINEAGER_ESMC_PATH:-/pfs_beijing/xinran_2026/models/ESMC-6B2026.9.15}"
export KINEAGER_MOLFORMER_PATH="${KINEAGER_MOLFORMER_PATH:-/pfs_beijing/xinran_2026/models/molformer/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-LjZwDjUAxKIzS86yP9mhG}"

# Several ranks share this container's HF module cache; cache the remote-code
# modules once, single-process, before torchrun starts (the copy is not atomic
# and two racing ranks can import a half-written module - reproduced with just
# two local ranks on 2026-10-15).
python "${ROOT}/bin/train/prewarm_remote_code.py"

exec bash "${ROOT}/bin/train/kineager_k8s.sh" \
    -epochs "${EPOCHS:-30}" \
    -batch_size "${BATCH_SIZE:-4}" \
    --split_mode group_pair \
    --split_seed 42 \
    --val_ratio 0.05 \
    --test_ratio 0.01 \
    --run_name "${RUN_NAME:-affinity_local_v1}"
