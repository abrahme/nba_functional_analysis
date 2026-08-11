#!/usr/bin/env bash
set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline_pipelined.sh — keep the GPUs saturated across a multi-scheme run.
#
# The linear pipeline runs, per scheme:  MAP+MCMC (GPU) → Export+Diag+Combine (CPU).
# Measured on holdout_last_k: ~6h20 GPU work, then ~7h50 where the GPUs sit at 0%
# while export + R diagnostics churn on CPU. Across 5 schemes that is ~40 GPU-hours
# wasted.
#
# This orchestrator splits those into two concurrent lanes over the SAME
# run_pipeline.sh (no sampling/analysis logic is duplicated):
#
#   GPU lane (producer): phases 1–2 for each scheme, back-to-back. The instant a
#                        scheme's sampling finishes it is enqueued and the next
#                        scheme's sampling starts immediately — GPUs never idle.
#   CPU lane (consumer): a single SERIAL worker that drains the queue, running
#                        phases 3–5 for each completed scheme. Serial on purpose:
#                        phase-3 export loads full posteriors into host RAM (the
#                        reason run_pipeline.sh defaults ANALYSIS_PARALLEL=0).
#                        The GPU lane is what we keep busy; the CPU lane may trail.
#
# Because the CPU leg (~8h/scheme) is longer than the GPU leg (~6h20/scheme), the
# consumer becomes the bottleneck and the queue backs up — the good case: GPUs
# finish all schemes while the CPU worker catches up. Expected wall-clock for 5
# schemes: ~72h linear → ~46h pipelined.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   ./run_pipeline_pipelined.sh "<schemes>" "<model numbers>" [scripts]
#     schemes        : quoted space-separated list, or 'all' (the 5 valid schemes)
#     model numbers  : same encoding as run_pipeline.sh (e.g. "1 2 9 10")
#     scripts        : phase-4 R scripts, run_pipeline.sh encoding (default: all)
#
#   Env overrides:
#     GPU_START_PHASE : first producer phase (default 1)   # 0 to include prior checks
#     GPU_END_PHASE   : last  producer phase (default 2)   # 1 for MAP-only producer
#     CPU_START_PHASE : first consumer phase (default 3)
#     CPU_END_PHASE   : last  consumer phase (default 5)
#     LOGDIR          : per-lane log dir (default: logs)
#     POLL_SECS       : consumer queue poll interval (default 15)
#
# ── Examples ─────────────────────────────────────────────────────────────────
#   ./run_pipeline_pipelined.sh \
#       "holdout_last_k holdout_first_k random_interior holdout_peak" "1 2 9 10"
#   ./run_pipeline_pipelined.sh all "1 2 9 10" "1 2"
#   nohup ./run_pipeline_pipelined.sh all "1 2 9 10" >/dev/null 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUN_PIPELINE="${RUN_PIPELINE:-$HERE/run_pipeline.sh}"   # overridable for testing
[[ -x "$RUN_PIPELINE" ]] || { echo "ERROR: $RUN_PIPELINE not found/executable"; exit 1; }

VALID_SCHEMES=(holdout_last_k holdout_first_k random_interior holdout_peak stratified_next_k)

# ── Args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<schemes | all>\" \"<model numbers>\" [scripts]"
    echo "       e.g. $0 \"holdout_last_k holdout_first_k\" \"1 2 9 10\""
    exit 1
fi

SCHEME_ARG="$1"
MODELS="$2"
SCRIPTS="${3:-all}"

GPU_START_PHASE="${GPU_START_PHASE:-1}"
GPU_END_PHASE="${GPU_END_PHASE:-2}"
CPU_START_PHASE="${CPU_START_PHASE:-3}"
CPU_END_PHASE="${CPU_END_PHASE:-5}"
LOGDIR="${LOGDIR:-$HERE/logs}"
POLL_SECS="${POLL_SECS:-15}"

# Expand / validate schemes.
if [[ "$SCHEME_ARG" == "all" ]]; then
    SCHEMES=("${VALID_SCHEMES[@]}")
else
    read -ra SCHEMES <<< "$SCHEME_ARG"
fi
for s in "${SCHEMES[@]}"; do
    printf '%s\n' "${VALID_SCHEMES[@]}" | grep -qx "$s" || {
        echo "ERROR: invalid scheme '$s'. Choose from: all ${VALID_SCHEMES[*]}"; exit 1; }
done

mkdir -p "$LOGDIR"
# A run tag derived from PID (Date/rand-free) keeps concurrent invocations apart.
TAG="pipelined_$$"
QUEUE="$LOGDIR/${TAG}.queue"
GPU_LOG="$LOGDIR/${TAG}_gpu.log"
CPU_LOG="$LOGDIR/${TAG}_cpu.log"
: > "$QUEUE"; : > "$GPU_LOG"; : > "$CPU_LOG"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] [orchestrator] $*"; }

say "schemes : ${SCHEMES[*]}"
say "models  : $MODELS   scripts: $SCRIPTS"
say "GPU lane: phases $GPU_START_PHASE-$GPU_END_PHASE  →  $GPU_LOG"
say "CPU lane: phases $CPU_START_PHASE-$CPU_END_PHASE  (serial)  →  $CPU_LOG"
say "queue   : $QUEUE"
echo

# ── CPU consumer: serial drain of the queue ──────────────────────────────────
# Polls the queue file for newly-enqueued schemes and runs phases 3–5 for each,
# one at a time. Exits on the __DONE__ sentinel.
consumer() {
    local processed=0 line
    while true; do
        line="$(sed -n "$((processed + 1))p" "$QUEUE" 2>/dev/null)"
        if [[ -z "$line" ]]; then
            sleep "$POLL_SECS"
            continue
        fi
        processed=$((processed + 1))
        [[ "$line" == "__DONE__" ]] && { echo "[$(ts)] [consumer] sentinel — done."; break; }
        echo "=== [$(ts)] [consumer] post-processing (phases $CPU_START_PHASE-$CPU_END_PHASE): $line ==="
        if bash "$RUN_PIPELINE" "$line" "$MODELS" "$CPU_START_PHASE" "$SCRIPTS" "$CPU_END_PHASE"; then
            echo "=== [$(ts)] [consumer] done: $line ==="
        else
            echo "!!! [$(ts)] [consumer] FAILED (rc=$?): $line — continuing ===" >&2
        fi
    done
}

consumer >> "$CPU_LOG" 2>&1 &
CONSUMER_PID=$!
# If we die, don't leave the consumer polling forever.
trap 'echo "__DONE__" >> "$QUEUE" 2>/dev/null; wait "$CONSUMER_PID" 2>/dev/null' EXIT INT TERM

# ── GPU producer: sample each scheme, enqueue on completion ──────────────────
{
    for s in "${SCHEMES[@]}"; do
        echo "=== [$(ts)] [producer] sampling (phases $GPU_START_PHASE-$GPU_END_PHASE): $s ==="
        if bash "$RUN_PIPELINE" "$s" "$MODELS" "$GPU_START_PHASE" "$SCRIPTS" "$GPU_END_PHASE"; then
            echo "$s" >> "$QUEUE"    # hand off to CPU lane; next scheme's sampling starts now
            echo "=== [$(ts)] [producer] enqueued for post-processing: $s ==="
        else
            echo "!!! [$(ts)] [producer] sampling FAILED (rc=$?): $s — NOT enqueued, skipping ===" >&2
        fi
    done
    echo "__DONE__" >> "$QUEUE"
    echo "=== [$(ts)] [producer] all schemes sampled; sentinel enqueued ==="
} 2>&1 | tee -a "$GPU_LOG"

say "GPU lane finished; waiting on CPU lane to drain the queue…"
trap - EXIT INT TERM
wait "$CONSUMER_PID"
say "CPU lane drained. Pipelined run complete."
