#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MAP fit for the RFF and RFF+AR models on the stratified_next_k regime.
#
#   nba_convex_max_tvrflvm_stratified_next_k       -> GPU 0
#   nba_convex_max_tvrflvm_AR_stratified_next_k    -> GPU 1
#
# Runs both MAPs in parallel (one per GPU), directly via main.py. At MAP the kernel
# hyperparameters are learned (incl. the RFF `lengthscale` and `sigma_exit_scale`);
# `W` is learned here too and then sampled at MCMC, while `lengthscale` is plugged in.
#
# Outputs:
#   model_output/nba_convex_max_tvrflvm/stratified_next_k/map/{samples.pkl,state.pkl}
#   model_output/nba_convex_max_tvrflvm_AR/stratified_next_k/map/{samples.pkl,state.pkl}
#
# Usage:
#   ./run_rff_stratified.sh                 # both models, full MAP (config map_num_steps)
#   MAP_STEPS=200 ./run_rff_stratified.sh   # quick smoke-test MAP (fewer Adam steps)
#   GPU_A=1 GPU_B=0 ./run_rff_stratified.sh # swap GPU assignment
#
# (Pipeline equivalent, if you use the docker containers instead:
#    ./run_pipeline.sh stratified_next_k "9 10" 1 all 1 )
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=config/model_config.yaml
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
EXTRA=()
[[ -n "${MAP_STEPS:-}" ]] && EXTRA+=(--set "map_num_steps=${MAP_STEPS}")

run_map() {
  local gpu=$1 name=$2
  echo "[GPU ${gpu}] MAP: ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
    --model_name="${name}" \
    --model_config="${CONFIG}" \
    --inference_method=map "${EXTRA[@]}"
}

echo "=== MAP: RFF + RFF_AR | stratified_next_k | GPUs ${GPU_A},${GPU_B}${MAP_STEPS:+ | map_num_steps=${MAP_STEPS}} ==="
run_map "${GPU_A}" nba_convex_max_tvrflvm_stratified_next_k    &
run_map "${GPU_B}" nba_convex_max_tvrflvm_AR_stratified_next_k &
wait
echo "=== MAP done for both RFF models (stratified_next_k) ==="
