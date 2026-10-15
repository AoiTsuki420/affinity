#!/usr/bin/env bash
#
# Operator wrapper: launch the KinEAGER affinity PyTorchJob and follow rank 0.
#
# It exists because of the two operational footguns documented in
# docs/kineager_affinity_runbook.md §6:
#   * metadata.name is fixed (kineager-affinity) with cleanPodPolicy=None and a
#     6000s TTL, so a re-run MUST delete the finished PyTorchJob first -
#     otherwise `kubectl apply` of the unchanged manifest reports "unchanged",
#     exits 0, and launches nothing;
#   * only rank 0 appends to LOG_PATH, so the master pod's log is the one worth
#     following.
#
# Usage:
#   bash bin/train/launch_kineager_affinity_k8s.sh [options]
#
# Options:
#   --no-delete          do not delete an existing PyTorchJob first
#   --no-follow          do not stream the master pod log
#   --dry-run            print the kubectl commands without running them
#   --wait-timeout N     seconds to wait for the master pod (default 1800)
#   -h | --help          show this help
#
# Environment overrides:
#   MANIFEST   path to the manifest (default yamls/kubeflow/kineager_affinity_k8s.yaml)
#   JOB        PyTorchJob / pod prefix (default kineager-affinity)
#   NAMESPACE  kubernetes namespace (default default)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MANIFEST="${MANIFEST:-$ROOT/yamls/kubeflow/kineager_affinity_k8s.yaml}"
JOB="${JOB:-kineager-affinity}"
NAMESPACE="${NAMESPACE:-default}"

DELETE_OLD=1
FOLLOW=1
DRY_RUN=0
WAIT_TIMEOUT=1800

while [ $# -gt 0 ]; do
    case "$1" in
        --no-delete)    DELETE_OLD=0 ;;
        --no-follow)    FOLLOW=0 ;;
        --dry-run)      DRY_RUN=1 ;;
        --wait-timeout) WAIT_TIMEOUT="${2:?--wait-timeout needs seconds}"; shift ;;
        -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

run() {
    echo "+ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

if [ "$DRY_RUN" -eq 0 ]; then
    command -v kubectl >/dev/null || { echo "kubectl not on PATH" >&2; exit 1; }
    if ! kubectl version --request-timeout=5s >/dev/null 2>&1; then
        # Different kubectl releases disagree on --request-timeout support;
        # a plain namespace lookup is the portable fallback.
        if ! kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
            echo "kubectl cannot reach the cluster (missing kubeconfig or no access)." >&2
            echo "Configure kubectl on this host, or run this script where cluster access exists." >&2
            exit 1
        fi
    fi
fi
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }

echo "[launch] manifest=$MANIFEST job=$JOB namespace=$NAMESPACE"

# --- 1) the fixed job name means a re-run must delete the old object ---------
if [ "$DELETE_OLD" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ (dry-run) kubectl get pytorchjob $JOB -n $NAMESPACE   # 存在则先执行下面这步"
        echo "+ (dry-run) kubectl delete pytorchjob $JOB -n $NAMESPACE --wait=true"
    elif kubectl get pytorchjob "$JOB" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "[launch] existing PyTorchJob '$JOB' found - deleting it first"
        run kubectl delete pytorchjob "$JOB" -n "$NAMESPACE" --wait=true
    else
        echo "[launch] no existing PyTorchJob '$JOB'"
    fi
fi

# --- 2) apply ---------------------------------------------------------------
run kubectl apply -f "$MANIFEST"

if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$FOLLOW" -eq 1 ]; then
        echo "+ (dry-run) kubectl get pod ${JOB}-master-0 ...   # 轮询直到 Running"
        echo "+ (dry-run) kubectl logs -f ${JOB}-master-0 -c pytorch -n $NAMESPACE"
    fi
    exit 0
fi

if [ "$FOLLOW" -eq 0 ]; then
    echo "[launch] submitted; not following (--no-follow)."
    echo "[launch] pod: ${JOB}-master-0  |  logs: kubectl logs -f ${JOB}-master-0 -c pytorch -n $NAMESPACE"
    exit 0
fi

# --- 3) wait for the master pod, then stream its log -------------------------
MASTER_POD="${JOB}-master-0"
echo "[launch] waiting for pod ${MASTER_POD} (timeout ${WAIT_TIMEOUT}s; first run may pull the image)..."
phase=""
deadline=$((SECONDS + WAIT_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
    phase=$(kubectl get pod "$MASTER_POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    case "$phase" in
        Running|Succeeded|Failed)
            break ;;
    esac
    sleep 10
done

case "$phase" in
    Running)
        echo "[launch] ${MASTER_POD} is Running - streaming rank-0 log (Ctrl-C stops following, the job keeps running)"
        run kubectl logs -f "$MASTER_POD" -c pytorch -n "$NAMESPACE"
        ;;
    Succeeded|Failed)
        echo "[launch] ${MASTER_POD} finished with phase=$phase before we could follow; dumping log"
        run kubectl logs "$MASTER_POD" -c pytorch -n "$NAMESPACE" --tail=200
        ;;
    *)
        echo "[launch] timed out after ${WAIT_TIMEOUT}s waiting for ${MASTER_POD}." >&2
        echo "         Inspect: kubectl describe pod $MASTER_POD -n $NAMESPACE" >&2
        exit 1
        ;;
esac

echo "[launch] final status:"
run kubectl get pytorchjob "$JOB" -n "$NAMESPACE" -o wide || true
echo "[launch] rank-0 log file (per the manifest's LOG_PATH): see yamls/kubeflow/kineager_affinity_k8s.yaml"
