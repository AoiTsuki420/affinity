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
#   * only rank 0 appends to LOG_PATH, so rank 0's pod is the one worth following.
#     Which pod that is depends on the launcher: the bare manifest puts rank 0 in
#     `<job>-master-0`, the elastic Worker-only pipeline has no master pod at all
#     and rank 0 is `<job>-...-worker-0`. The pod is therefore DISCOVERED from the
#     training-operator's labels (master-0 preferred when it exists) instead of
#     being hardcoded.
#
# Usage:
#   bash bin/train/launch_kineager_affinity_k8s.sh [options]
#
# Options:
#   --no-delete          do not delete an existing PyTorchJob first
#   --no-follow          do not stream the rank-0 pod log
#   --dry-run            print the kubectl commands without running them
#   --wait-timeout N     seconds to wait for the rank-0 pod (default 1800)
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
        echo "+ (dry-run) kubectl get pods -n $NAMESPACE -l training.kubeflow.org/job-name=$JOB   # 轮询直到出现 *-master-0 / *-worker-0 且 Running"
        echo "+ (dry-run) kubectl logs -f <rank-0 pod> -c pytorch -n $NAMESPACE"
    fi
    exit 0
fi

if [ "$FOLLOW" -eq 0 ]; then
    echo "[launch] submitted; not following (--no-follow)."
    echo "[launch] rank-0 pod: kubectl get pods -n $NAMESPACE -l training.kubeflow.org/job-name=$JOB"
    echo "[launch] logs:       kubectl logs -f <that pod> -c pytorch -n $NAMESPACE"
    exit 0
fi

# --- 3) wait for the rank-0 pod, then stream its log -------------------------
# The pod name is DISCOVERED, not hardcoded. The bare manifest
# (yamls/kubeflow/kineager_affinity_k8s.yaml) has a Master replica, so rank 0
# runs in `${JOB}-master-0`; the elastic, Worker-only pipeline
# (yamls/kubeflow/kineager_affinity_pipeline.yaml) has no Master at all and rank 0
# is whichever worker holds NODE_RANK=0 - in practice `${JOB}-...-worker-0`. (The
# elastic job's pods also carry a generated suffix, so even the worker's name is
# not predictable.) Ask the cluster instead of assuming either shape.
#
# The user-supplied JOB may be a prefix (kineager-affinity) or the exact
# metadata.name of the bare manifest, so match on the training-operator's own
# label and filter client-side: kubectl label selectors have no prefix match.
pod_names() {
    kubectl get pods -n "$NAMESPACE" \
        -l "training.kubeflow.org/job-name=$JOB" \
        --no-headers -o custom-columns=':metadata.name' 2>/dev/null || true
}

# Two passes, deliberately: `kubectl get pods` sorts by name, and an elastic job's
# generated name can sort *before* a master pod's, so a single loop that takes the
# first match can pick worker-0 while a master exists. A master pod wins outright;
# worker-0 is only the fallback for the Worker-only pipeline.
pick_rank0() {
    chosen=""
    for candidate in $(pod_names); do
        case "$candidate" in
            *-master-0) chosen="$candidate" ;;
        esac
        [ -n "$chosen" ] && break
    done
    if [ -z "$chosen" ]; then
        for candidate in $(pod_names); do
            case "$candidate" in
                *-worker-0) chosen="$candidate" ;;
            esac
            [ -n "$chosen" ] && break
        done
    fi
    printf '%s' "$chosen"
}

echo "[launch] waiting for the rank-0 pod of '$JOB' (timeout ${WAIT_TIMEOUT}s; first run may pull the image)..."
phase=""
MASTER_POD=""
last_seen=""
deadline=$((SECONDS + WAIT_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
    MASTER_POD=$(pick_rank0)
    if [ -n "$MASTER_POD" ]; then
        phase=$(kubectl get pod "$MASTER_POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    else
        phase=""
    fi
    case "$phase" in
        Running|Succeeded|Failed)
            break ;;
    esac
    # One line per pod-set change (not one per 10 s poll), so a slow start is
    # still visible: which pods exist and which phase rank 0 is in.
    if [ "$MASTER_POD" != "$last_seen" ]; then
        if [ -n "$MASTER_POD" ]; then
            echo "[launch] rank-0 candidate ${MASTER_POD} is in phase '${phase:-unknown}'"
        else
            echo "[launch] no *-master-0 / *-worker-0 pod yet"
        fi
        last_seen="$MASTER_POD"
    fi
    sleep 10
done

if [ -z "${MASTER_POD:-}" ]; then
    echo "[launch] no rank-0 pod found for job '$JOB' in namespace '$NAMESPACE' after ${WAIT_TIMEOUT}s." >&2
    echo "         Looked for a pod labelled training.kubeflow.org/job-name=$JOB with a name ending in -master-0 or -worker-0." >&2
    echo "         Pods actually found for that job (may be none, or all still Pending/ContainerCreating):" >&2
    listed=$(pod_names)
    if [ -n "$listed" ]; then
        printf '%s\n' "$listed" | sed 's/^/           /' >&2
    else
        echo "           (none)" >&2
    fi
    echo "         Inspect:" >&2
    echo "           kubectl get pods -n $NAMESPACE -l training.kubeflow.org/job-name=$JOB -o wide" >&2
    echo "           kubectl get pytorchjob -n $NAMESPACE" >&2
    exit 1
fi

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
        echo "[launch] timed out after ${WAIT_TIMEOUT}s waiting for ${MASTER_POD} to leave phase '${phase}'." >&2
        echo "         Inspect: kubectl describe pod $MASTER_POD -n $NAMESPACE" >&2
        exit 1
        ;;
esac

echo "[launch] final status:"
run kubectl get pytorchjob "$JOB" -n "$NAMESPACE" -o wide || true
echo "[launch] rank-0 log file (per the manifest's LOG_PATH): see yamls/kubeflow/kineager_affinity_k8s.yaml"
