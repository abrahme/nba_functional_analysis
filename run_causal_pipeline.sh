#!/usr/bin/env bash
set -euo pipefail

# ── Two-Stage Causal Injury Pipeline ─────────────────────────────────────────
# Stage 1 MAP  : non-injury model, healthy-only metric likelihood, survival
#                censored at injury onset → clean counterfactual plug-ins.
# Stage 2 MCMC : injury model with all Stage 1 params fixed; samples only the
#                injury-specific parameters. Run sequentially to avoid GPU
#                memory contention.
# Stage 3 Export: model_export.py on Stage 2 MCMC output (parallel).
# Stage 4 R     : injury_two_stage_causal.r on Stage 2 export (parallel).
#
# Usage (interactive):
#   ./run_causal_pipeline.sh
# Non-interactive:
#   ./run_causal_pipeline.sh <base_model> [start_phase]
#     base_model  : tvlvm | ar | all
#     start_phase : 1=Stage1_MAP  2=Stage2_MCMC  3=Export  4=Diagnostics
#                   (default: 1)
#
# When base_model=all:
#   Phase 1 MAP  — tvlvm on GPU 0 and ar on GPU 1, in parallel
#   Phase 2 MCMC — tvlvm then ar, sequentially
#   Phase 3/4    — both in parallel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CONFIG="config/model_config.yaml"
CTR_MCMC="mcmc"
CTR_ANALYSIS="mcmc-analysis"
CTR_R="r-new"
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"
VALIDATION_YEAR="2021"

# ── Model name helpers ────────────────────────────────────────────────────────
stage1_name() {
    case $1 in
        tvlvm) echo "nba_convex_max_tvlinearlvm_causal_prefit" ;;
        ar)    echo "nba_convex_max_tvlinearlvm_AR_causal_prefit" ;;
    esac
}

stage2_name() {
    case $1 in
        tvlvm) echo "nba_convex_max_tvlinearlvm_injury_causal" ;;
        ar)    echo "nba_convex_max_tvlinearlvm_AR_injury_causal" ;;
    esac
}

stage2_dir() {
    case $1 in
        tvlvm) echo "model_output/nba_convex_max_tvlinearlvm_injury_causal/mcmc" ;;
        ar)    echo "model_output/nba_convex_max_tvlinearlvm_AR_injury_causal/mcmc" ;;
    esac
}

# ── Parse base_model ──────────────────────────────────────────────────────────
VALID_MODELS=(tvlvm ar all)

if [[ $# -ge 1 ]]; then
    BASE_MODEL=$1
    if ! printf '%s\n' "${VALID_MODELS[@]}" | grep -qx "$BASE_MODEL"; then
        echo "ERROR: invalid base_model '$BASE_MODEL'. Choose from: ${VALID_MODELS[*]}"
        exit 1
    fi
else
    echo "Select base model(s):"
    select BASE_MODEL in "${VALID_MODELS[@]}"; do
        [[ -n $BASE_MODEL ]] && break
    done
fi

[[ $BASE_MODEL == "all" ]] && MODELS=(tvlvm ar) || MODELS=("$BASE_MODEL")

START_PHASE=${2:-1}

echo ""
echo "Models      : ${MODELS[*]}"
echo "Start phase : $START_PHASE"
echo ""

# ── Container health check ────────────────────────────────────────────────────
for ctr in "$CTR_MCMC" "$CTR_ANALYSIS" "$CTR_R"; do
    docker inspect --format='{{.State.Running}}' "$ctr" 2>/dev/null \
        | grep -q true \
        || { echo "ERROR: container '$ctr' is not running"; exit 1; }
done
echo "All containers running."
echo ""

# ── Phase 1: Stage 1 MAP — parallel across GPUs ───────────────────────────────
if [[ $START_PHASE -le 1 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 1: Stage 1 MAP ==="
gpu=0
for m in "${MODELS[@]}"; do
    s1=$(stage1_name "$m")
    echo "  [GPU $gpu] map: $s1"
    $EXEC -e "CUDA_VISIBLE_DEVICES=$gpu" "$CTR_MCMC" python main.py \
        --model_name="$s1" \
        --model_config="$MODEL_CONFIG" \
        --inference_method=map &
    gpu=$(( (gpu + 1) % 2 ))
done
wait
echo "=== [$(date '+%H:%M:%S')] Stage 1 MAP done ==="
echo ""
fi

# ── Phase 2: Stage 2 MCMC — sequential ───────────────────────────────────────
if [[ $START_PHASE -le 2 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 2: Stage 2 MCMC ==="
for m in "${MODELS[@]}"; do
    s2=$(stage2_name "$m")
    echo "  mcmc: $s2"
    $EXEC "$CTR_MCMC" python main.py \
        --model_name="$s2" \
        --model_config="$MODEL_CONFIG" \
        --inference_method=mcmc
done
echo "=== [$(date '+%H:%M:%S')] Stage 2 MCMC done ==="
echo ""
fi

# ── Phase 3: Export — parallel ────────────────────────────────────────────────
if [[ $START_PHASE -le 3 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 3: Export ==="
for m in "${MODELS[@]}"; do
    s2=$(stage2_name "$m")
    echo "  export: $s2"
    $EXEC "$CTR_ANALYSIS" python model_export.py \
        --model_name="$s2" \
        --model_config="$MODEL_CONFIG" &
done
wait
echo "=== [$(date '+%H:%M:%S')] Export done ==="
echo ""
fi

# ── Phase 4: R diagnostics — parallel ────────────────────────────────────────
if [[ $START_PHASE -le 4 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 4: R diagnostics ==="
for m in "${MODELS[@]}"; do
    s2dir=$(stage2_dir "$m")
    echo "  diagnostics: $s2dir"
    $EXEC "$CTR_R" Rscript data_analysis/model_diagnostics.r \
        "$s2dir" "$VALIDATION_YEAR" &
    $EXEC "$CTR_R" Rscript data_causal/injury_two_stage_causal.r \
        "$s2dir" &
done
wait
echo "=== [$(date '+%H:%M:%S')] Diagnostics done ==="
echo ""
fi

echo "=== [$(date '+%H:%M:%S')] Causal pipeline complete ==="
