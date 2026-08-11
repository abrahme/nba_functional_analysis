#!/usr/bin/env bash
set -euo pipefail

# ── Ad-hoc resume after the 2026-06-10 server crash ───────────────────────────
# Stages:
#   1  MAP + MCMC   — the 3 unstarted schemes                       (GPU-bound)
#   2  Export       — ALL 5 schemes, serial (one model at a time)   (host RAM)
#   3  Diagnostics  — ALL 5 schemes; per-model R scripts run in     (host RAM)
#                     PARALLEL, but models & schemes stay serial
#   4  Combine      — once, across all schemes
#
# Usage:  ./resume_pipeline.sh [start_stage]     # start_stage default 1
#         ./resume_pipeline.sh 3                 # diagnostics + combine only (1 & 2 done)
#         nohup ./resume_pipeline.sh 3 > resume_pipeline.log 2>&1 &

START_STAGE=${1:-1}

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUN="$HERE/run_pipeline.sh"

MODELS="1 2 4 5 6"                     # tvlvm ar naive tvlinearlvm tvlinearlvm_AR
NEW_SCHEMES=(random_interior holdout_peak stratified_next_k)
ALL_SCHEMES=(holdout_last_k holdout_first_k random_interior holdout_peak stratified_next_k)

banner() { echo; echo "########## [$(date '+%F %T')] $* ##########"; echo; }

# ── Stage 1: MAP + MCMC for the unstarted schemes (phases 1-2) ────────────────
# GPU phases are safe to overlap: pair the first two across GPUs, run the rest solo.
if (( START_STAGE <= 1 )); then
banner "STAGE 1: MAP + MCMC (${NEW_SCHEMES[*]})"
GPU_BASE=0 bash "$RUN" "${NEW_SCHEMES[0]}" "$MODELS" 1 all 2 &
GPU_BASE=1 bash "$RUN" "${NEW_SCHEMES[1]}" "$MODELS" 1 all 2 &
wait
for sch in "${NEW_SCHEMES[@]:2}"; do
    bash "$RUN" "$sch" "$MODELS" 1 all 2
done
fi

# ── Stage 2: Export only (phase 3) for ALL schemes, serial ───────────────────
# Export stays one-model-at-a-time (ANALYSIS_PARALLEL unset → 0) to bound host RAM.
if (( START_STAGE <= 2 )); then
banner "STAGE 2: EXPORT (all schemes, serial)"
for sch in "${ALL_SCHEMES[@]}"; do
    bash "$RUN" "$sch" "$MODELS" 3 all 3
done
fi

# ── Stage 3: Diagnostics (phase 4) for ALL schemes ───────────────────────────
# ANALYSIS_PARALLEL=1 → the per-model R scripts (coverage.r, model_diagnostics.r,
# latent_space.r, team_window.r) run in PARALLEL. Phase 4's per-model `wait` keeps
# models serial; this loop keeps schemes serial. So only ONE (model, scheme)'s
# scripts are ever in flight at once -- i.e. up to ~4 concurrent R processes,
# each loading posterior_ar. Watch host RAM; drop to serial (remove ANALYSIS_PARALLEL=1)
# if it gets tight.
if (( START_STAGE <= 3 )); then
banner "STAGE 3: DIAGNOSTICS (per-model scripts parallel; models & schemes serial)"
for sch in "${ALL_SCHEMES[@]}"; do
    ANALYSIS_PARALLEL=1 bash "$RUN" "$sch" "$MODELS" 4 all 4
done
fi

# ── Stage 4: Combine coverage tables once ────────────────────────────────────
if (( START_STAGE <= 4 )); then
banner "STAGE 4: COMBINE coverage tables"
bash "$RUN" "${ALL_SCHEMES[0]}" "$MODELS" 5 all 5
fi

banner "RESUME COMPLETE"
