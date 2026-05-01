#!/usr/bin/env bash
set -euo pipefail

# ── Config (mirrors Makefile) ─────────────────────────────────────────────────
MODEL_CONFIG="config/model_config.yaml"
CTR_MCMC="mcmc"
CTR_ANALYSIS="mcmc-analysis"
CTR_R="r-new"
VALIDATION_YEAR="2021"
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"

# ── Model name / dir helpers ──────────────────────────────────────────────────
model_name() {
    local model=$1 scheme=$2
    case $model in
        tvlvm)  echo "nba_convex_max_tvlinearlvm_${scheme}" ;;
        ar)     echo "nba_convex_max_tvlinearlvm_AR_${scheme}" ;;
        injury) echo "nba_convex_max_tvlinearlvm_injury_${scheme}" ;;
        naive)  echo "nba_naive_${scheme}" ;;
    esac
}

model_dir() {
    local model=$1 scheme=$2
    case $model in
        tvlvm)  echo "model_output/nba_convex_max_tvlinearlvm/${scheme}/mcmc" ;;
        ar)     echo "model_output/nba_convex_max_tvlinearlvm_AR/${scheme}/mcmc" ;;
        injury) echo "model_output/nba_convex_max_tvlinearlvm_injury/${scheme}/mcmc" ;;
        naive)  echo "model_output/nba_naive/${scheme}/mcmc" ;;
    esac
}

# ── Usage ─────────────────────────────────────────────────────────────────────
# Interactive:    ./run_pipeline.sh
# Non-interactive: ./run_pipeline.sh <scheme> "<model numbers>"
#   scheme         : holdout_last_k | holdout_first_k | random_interior | holdout_peak
#   model numbers  : space-separated subset of 1=tvlvm 2=ar 3=injury 4=naive
#   e.g.           : ./run_pipeline.sh holdout_last_k "1 2 4"

VALID_SCHEMES=(holdout_last_k holdout_first_k random_interior holdout_peak)

parse_model_choices() {
    local -a choices=($1)
    SELECTED_MODELS=()
    for c in "${choices[@]}"; do
        case $c in
            1) SELECTED_MODELS+=(tvlvm) ;;
            2) SELECTED_MODELS+=(ar) ;;
            3) SELECTED_MODELS+=(injury) ;;
            4) SELECTED_MODELS+=(naive) ;;
            *) echo "Warning: unrecognised option '$c' — skipping" ;;
        esac
    done
}

# ── Scheme: arg or prompt ─────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    SCHEME=$1
    if ! printf '%s\n' "${VALID_SCHEMES[@]}" | grep -qx "$SCHEME"; then
        echo "ERROR: invalid scheme '$SCHEME'. Choose from: ${VALID_SCHEMES[*]}"
        exit 1
    fi
else
    echo "Select holdout scheme:"
    select SCHEME in "${VALID_SCHEMES[@]}"; do
        [[ -n $SCHEME ]] && break
    done
fi

# ── Models: arg or prompt ─────────────────────────────────────────────────────
SELECTED_MODELS=()

if [[ $# -ge 2 ]]; then
    parse_model_choices "$2"
else
    echo ""
    echo "Select models to include (space-separated numbers):"
    echo "  1) tvlvm"
    echo "  2) ar"
    echo "  3) injury"
    echo "  4) naive"
    read -rp "? " -a choices
    parse_model_choices "${choices[*]}"
fi

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    echo "No models selected — exiting."
    exit 1
fi

echo ""
echo "Scheme : $SCHEME"
echo "Models : ${SELECTED_MODELS[*]}"
echo ""

# ── Container health check ────────────────────────────────────────────────────
for ctr in "$CTR_MCMC" "$CTR_ANALYSIS" "$CTR_R"; do
    docker inspect --format='{{.State.Running}}' "$ctr" 2>/dev/null \
        | grep -q true \
        || { echo "ERROR: container '$ctr' is not running"; exit 1; }
done
echo "All containers running."
echo ""

# ── Phase 1: MAP — two sequential GPU chains running in parallel ──────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 1: MAP ==="

gpu0_models=()
gpu1_models=()
for i in "${!SELECTED_MODELS[@]}"; do
    (( i % 2 == 0 )) && gpu0_models+=("${SELECTED_MODELS[$i]}") \
                     || gpu1_models+=("${SELECTED_MODELS[$i]}")
done

(
    for m in "${gpu0_models[@]}"; do
        mname=$(model_name "$m" "$SCHEME")
        echo "  [GPU 0] map: $mname"
        $EXEC -e "CUDA_VISIBLE_DEVICES=0" "$CTR_MCMC" python main.py \
            --model_name="$mname" \
            --model_config="$MODEL_CONFIG" \
            --inference_method=map
    done
) &

(
    for m in "${gpu1_models[@]}"; do
        mname=$(model_name "$m" "$SCHEME")
        echo "  [GPU 1] map: $mname"
        $EXEC -e "CUDA_VISIBLE_DEVICES=1" "$CTR_MCMC" python main.py \
            --model_name="$mname" \
            --model_config="$MODEL_CONFIG" \
            --inference_method=map
    done
) &

wait
echo "=== [$(date '+%H:%M:%S')] MAP done ==="
echo ""

# ── Phase 2: MCMC — sequential to avoid GPU memory contention ────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 2: MCMC ==="
for m in "${SELECTED_MODELS[@]}"; do
    mname=$(model_name "$m" "$SCHEME")
    echo "  mcmc: $mname"
    $EXEC "$CTR_MCMC" python main.py \
        --model_name="$mname" \
        --model_config="$MODEL_CONFIG" \
        --inference_method=mcmc
done
echo "=== [$(date '+%H:%M:%S')] MCMC done ==="
echo ""

# ── Phase 3: Export — parallel in mcmc-analysis container ────────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 3: Export ==="
for m in "${SELECTED_MODELS[@]}"; do
    mname=$(model_name "$m" "$SCHEME")
    echo "  export: $mname"
    $EXEC "$CTR_ANALYSIS" python model_export.py \
        --model_name="$mname" \
        --model_config="$MODEL_CONFIG" &
done
wait
echo "=== [$(date '+%H:%M:%S')] Export done ==="
echo ""

# ── Phase 4: R diagnostics — parallel in r-new container ─────────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 4: R diagnostics ==="
for m in "${SELECTED_MODELS[@]}"; do
    mdir=$(model_dir "$m" "$SCHEME")
    echo "  diagnostics: $mdir"
    $EXEC "$CTR_R" Rscript data_analysis/model_diagnostics.r \
        "$mdir" "$VALIDATION_YEAR" &
done

# injury causal script runs alongside if injury is selected
if printf '%s\n' "${SELECTED_MODELS[@]}" | grep -q '^injury$'; then
    $EXEC "$CTR_R" Rscript data_causal/injury_causal.r \
        "$(model_dir injury "$SCHEME")" &
fi

wait
echo "=== [$(date '+%H:%M:%S')] Diagnostics done ==="
echo ""

# ── Phase 5: Combine coverage tables ─────────────────────────────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 5: Combine coverage tables ==="
$EXEC "$CTR_ANALYSIS" python model_output/model_plots/coverage/combine_holdout_tables.py
echo "=== [$(date '+%H:%M:%S')] Pipeline complete ==="
