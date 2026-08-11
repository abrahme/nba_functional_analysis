#!/usr/bin/env bash
# Re-export the two legacy centered-X models (all 5 schemes) with the normalization fix,
# re-run coverage.r, then re-run the combine so the paper tables pick up corrected columns.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
run_cell() {
  local base=$1 scheme=$2
  local name="${base}_${scheme}" mdir="model_output/${base}/${scheme}/mcmc" log="logs/covfan/legacy_${base}_${scheme}.log"
  echo "[$(date +%H:%M)] EXPORT $name" > "$log"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only >>"$log" 2>&1 \
    || { echo "EXPORT FAILED $name" | tee -a "$log"; return 1; }
  docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 >>"$log" 2>&1 \
    || { echo "COVERAGE FAILED $name" | tee -a "$log"; return 1; }
  echo "[$(date +%H:%M)] DONE $name" >> "$log"
}
export -f run_cell
for b in nba_tvlinearlvm nba_tvlinearlvm_AR; do
  for s in holdout_last_k holdout_first_k random_interior holdout_peak stratified_next_k; do echo "$b $s"; done
done | xargs -P 5 -L1 bash -c 'run_cell "$0" "$1"'
echo "[$(date +%H:%M)] legacy cells done — re-running combine"
docker exec -w /home/joyvan/work mcmc-analysis python model_output/model_plots/coverage/combine_holdout_tables.py
echo "[$(date +%H:%M)] LEGACY RE-EXPORT + COMBINE DONE"
