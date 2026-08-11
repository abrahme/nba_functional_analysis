#!/usr/bin/env bash
set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# handoff_holdout_peak.sh — one-shot GPU hand-off watcher.
#
# The linear driver was stopped after `random_interior` so it will NOT auto-run
# `holdout_peak`. This watcher waits until `random_interior` finishes sampling
# (enters Phase 3 Export, i.e. GPUs freed) and then immediately launches
# `holdout_peak`'s full pipeline on the freed GPUs — so holdout_peak's MAP+MCMC
# (GPU) overlaps random_interior's export+diagnostics (CPU).
#
# Trigger: the "PHASE 3: Export" marker appearing AFTER the last
#          "scheme: random_interior" marker in holdouts_pipeline.log, confirmed
#          by both GPUs going idle (no random_interior sampler left).
# ─────────────────────────────────────────────────────────────────────────────

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
LOG="$HERE/logs/holdouts_pipeline.log"
OUT="$HERE/logs/holdout_peak.log"          # holdout_peak's own pipeline log
WLOG="$HERE/logs/handoff_watcher.log"      # this watcher's log
MODELS="1 2 9 10"
POLL=60

ts() { date '+%F %T'; }
say() { echo "[$(ts)] [handoff] $*" >> "$WLOG"; }

# Anchor: only count Phase-3 markers that come after the CURRENT random_interior run.
anchor=$(grep -n "scheme: random_interior" "$LOG" | tail -1 | cut -d: -f1)
say "watcher started; random_interior anchor line = ${anchor:-<none>}"
[[ -z "$anchor" ]] && { say "ERROR: no random_interior marker found — aborting"; exit 1; }

gpus_idle() {
    # true when every GPU reports <10% utilization
    ! nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ if ($1+0 >= 10) exit 0 } END { exit 1 }'
}

sampler_running() {
    pgrep -af "main.py.*random_interior.*inference_method=mcmc" >/dev/null 2>&1
}

# ── Wait for Phase 2 (sampling) of random_interior to finish ─────────────────
while true; do
    phase3=$(awk -v a="$anchor" 'NR>a && /PHASE 3: Export/{print NR; exit}' "$LOG")
    if [[ -n "$phase3" ]] && ! sampler_running && gpus_idle; then
        say "random_interior entered Phase 3 (line $phase3); GPUs idle; sampler gone → firing hand-off"
        break
    fi
    sleep "$POLL"
done

# ── Launch holdout_peak's full pipeline on the freed GPUs ────────────────────
say "launching: run_pipeline.sh holdout_peak \"$MODELS\"  → $OUT"
cd "$HERE"
setsid nohup bash "$HERE/run_pipeline.sh" holdout_peak "$MODELS" >> "$OUT" 2>&1 &
peak_pid=$!
say "holdout_peak launched (pid $peak_pid). Watcher done."
