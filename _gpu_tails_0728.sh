#!/usr/bin/env bash
# Recreate the lost post-MCMC tails for both GPU chains: when samples.pkl appears,
# run coverage+elppd export + coverage.r.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
tail_for() {  # $1 model_name  $2 mdir  $3 logstem
  until [ -f "$2/samples.pkl" ]; do sleep 300; done
  sleep 60
  echo "[$(date +%H:%M)] $3: MCMC done — export"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$1" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
    > "logs/$3_export.log" 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$2" 2021 \
       > "logs/$3_coverage.log" 2>&1 \
  && echo "[$(date +%H:%M)] $3: export+coverage DONE" || echo "[$(date +%H:%M)] $3: FAILED"
}
tail_for nba_convex_max_tvrflvm_AR_re_on_stratified_next_k \
  model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc re_on_strat &
tail_for nba_convex_max_tvrflvm_AR_linear_holdout_last_k \
  model_output/nba_convex_max_tvrflvm_AR_linear/holdout_last_k/mcmc linear_pilot &
wait
echo "[$(date +%H:%M)] BOTH GPU TAILS COMPLETE"
