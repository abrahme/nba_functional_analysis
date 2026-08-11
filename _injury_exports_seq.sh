#!/usr/bin/env bash
# Sequential injury exports with a memory gate: wait until >260GB available before each.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
wait_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 260 ]; do sleep 120; done; }
run_pair() {  # $1 = model_name, $2 = log stem
  wait_mem
  echo "[$(date +%H:%M)] EXPORT $1"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
    --model_name="$1" --model_config=config/model_config.yaml --inference_method=mcmc \
    > "logs/$2_export.log" 2>&1 || { echo "[$(date +%H:%M)] EXPORT FAILED $1"; return 1; }
  echo "[$(date +%H:%M)] R $1"
  docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
    "model_output/$1/mcmc" > "logs/$2_causal_r.log" 2>&1 || echo "[$(date +%H:%M)] R FAILED $1"
  echo "[$(date +%H:%M)] DONE $1"
}
run_pair nba_convex_max_tvrflvm_AR_injury_causal_slope       slope
run_pair nba_convex_max_tvrflvm_AR_injury_causal_slope_only  slope_only
# placebo: wait for its samples.pkl to exist first (fit still running on GPU 0)
until [ -f model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo/mcmc/samples.pkl ]; do sleep 180; done
run_pair nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo placebo
echo "[$(date +%H:%M)] ALL INJURY EXPORTS DONE"
