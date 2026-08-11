#!/usr/bin/env bash
# Post-restart recovery: (1) remaining 5 legacy cells + combine, (2) sym chain remainder.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
run_cell() {
  local base=$1 scheme=$2
  local name="${base}_${scheme}" mdir="model_output/${base}/${scheme}/mcmc" log="logs/covfan/legacy_${base}_${scheme}.log"
  echo "[$(date +%H:%M)] RETRY EXPORT $name" >> "$log"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only >>"$log" 2>&1 \
    || { echo "EXPORT FAILED $name" | tee -a "$log"; return 1; }
  docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 >>"$log" 2>&1 \
    || { echo "COVERAGE FAILED $name" | tee -a "$log"; return 1; }
  echo "[$(date +%H:%M)] DONE $name" >> "$log"
}
export -f run_cell
{
  printf '%s\n' "nba_tvlinearlvm random_interior" \
                "nba_tvlinearlvm_AR holdout_first_k" "nba_tvlinearlvm_AR holdout_peak" \
                "nba_tvlinearlvm_AR random_interior" "nba_tvlinearlvm_AR stratified_next_k"
} | xargs -P 3 -L1 bash -c 'run_cell "$0" "$1"'
echo "[$(date +%H:%M)] legacy remainder done — combine"
docker exec -w /home/joyvan/work mcmc-analysis python model_output/model_plots/coverage/combine_holdout_tables.py
echo "[$(date +%H:%M)] LEGACY+COMBINE DONE"

# sym chain remainder (sequential after legacy pool to bound memory)
docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
  model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope_sym/mcmc > logs/slope_sym_causal_r.log 2>&1 \
  && echo "[$(date +%H:%M)] slope_sym R done"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym \
  --model_config=config/model_config.yaml --inference_method=mcmc > logs/placebo_sym_export.log 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
     model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym/mcmc > logs/placebo_sym_causal_r.log 2>&1 \
  && echo "[$(date +%H:%M)] placebo_sym export+R done"
echo "[$(date +%H:%M)] RECOVERY COMPLETE"
