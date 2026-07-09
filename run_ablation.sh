#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CONFIG="config/model_config.yaml"
CTR_MCMC="mcmc"
CTR_ANALYSIS="mcmc-analysis"
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"

ALL_Q=(5 10 15 20)
ALL_SCHEMES=(holdout_last_k holdout_first_k holdout_peak random_interior)

# ── Usage ─────────────────────────────────────────────────────────────────────
# Interactive:     ./run_ablation.sh
# Non-interactive: ./run_ablation.sh "<q values>" ["<schemes>"]
#   q values : space-separated subset of 5 10 15 20, or "all"  (default: all)
#   schemes  : space-separated subset of the four schemes, or "all" (default: all)
#   e.g.: ./run_ablation.sh "5 10 15 20"
#         ./run_ablation.sh "10 20" "holdout_last_k holdout_peak"

# ── Parse q values ────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    [[ $1 == "all" ]] && SELECTED_Q=("${ALL_Q[@]}") || read -ra SELECTED_Q <<< "$1"
else
    echo "Select q values (space-separated, e.g. '5 10 15 20', or 'all'):"
    read -rp "? " raw
    [[ $raw == "all" ]] && SELECTED_Q=("${ALL_Q[@]}") || read -ra SELECTED_Q <<< "$raw"
fi

# ── Parse schemes ─────────────────────────────────────────────────────────────
if [[ $# -ge 2 ]]; then
    [[ $2 == "all" ]] && SELECTED_SCHEMES=("${ALL_SCHEMES[@]}") || read -ra SELECTED_SCHEMES <<< "$2"
else
    echo "Select schemes (space-separated, or 'all'):"
    printf '  %s\n' "${ALL_SCHEMES[@]}"
    read -rp "? " raw
    [[ $raw == "all" ]] && SELECTED_SCHEMES=("${ALL_SCHEMES[@]}") || read -ra SELECTED_SCHEMES <<< "$raw"
fi

if [[ ${#SELECTED_Q[@]} -eq 0 ]]; then
    echo "No q values selected — exiting."; exit 1
fi
if [[ ${#SELECTED_SCHEMES[@]} -eq 0 ]]; then
    echo "No schemes selected — exiting."; exit 1
fi

# Build flat list of all (q, scheme) model names
RUNS=()
for q in "${SELECTED_Q[@]}"; do
    for scheme in "${SELECTED_SCHEMES[@]}"; do
        RUNS+=("nba_convex_max_tvlinearlvm_q${q}_${scheme}")
    done
done

echo ""
echo "Q values : ${SELECTED_Q[*]}"
echo "Schemes  : ${SELECTED_SCHEMES[*]}"
echo "Total    : ${#RUNS[@]} MAP runs"
echo ""

# ── Container health check ────────────────────────────────────────────────────
for ctr in "$CTR_MCMC" "$CTR_ANALYSIS"; do
    docker inspect --format='{{.State.Running}}' "$ctr" 2>/dev/null \
        | grep -q true \
        || { echo "ERROR: container '$ctr' is not running"; exit 1; }
done
echo "All containers running."
echo ""

# ── Phase 1: MAP — split runs across two GPUs ─────────────────────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 1: MAP ==="

gpu0_runs=(); gpu1_runs=()
for i in "${!RUNS[@]}"; do
    (( i % 2 == 0 )) && gpu0_runs+=("${RUNS[$i]}") \
                     || gpu1_runs+=("${RUNS[$i]}")
done

(
    for mname in "${gpu0_runs[@]}"; do
        echo "  [GPU 0] map: $mname"
        $EXEC -e "CUDA_VISIBLE_DEVICES=0" "$CTR_MCMC" python main.py \
            --model_name="$mname" \
            --model_config="$MODEL_CONFIG" \
            --inference_method=map
    done
) &

(
    for mname in "${gpu1_runs[@]}"; do
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

# ── Phase 2: Combine ablation tables ──────────────────────────────────────────
echo "=== [$(date '+%H:%M:%S')] PHASE 2: Combine ablation tables ==="
$EXEC "$CTR_ANALYSIS" python model_output/model_plots/coverage/combine_ablation_tables.py
echo "=== [$(date '+%H:%M:%S')] Ablation complete ==="
