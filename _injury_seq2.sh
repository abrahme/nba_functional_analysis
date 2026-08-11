#!/usr/bin/env bash
# Injury export orchestration v2. Adopts the in-flight slope export (PID arg), then runs
# ablation + placebo sequentially. Gate: >300GB available AND no other model_export running.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
SLOPE_PID=${1:?need slope export pid}
gate() {
  while :; do
    avail=$(free -g | awk '/^Mem:/{print $7}')
    nexp=$(pgrep -fc "model_export.py" || true)
    [ "$avail" -ge 300 ] && [ "${nexp:-0}" -eq 0 ] && break
    sleep 120
  done
}
run_r() { docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
          "model_output/$1/mcmc" > "logs/$2_causal_r.log" 2>&1; }

# 1) adopt the running slope export
while kill -0 "$SLOPE_PID" 2>/dev/null; do sleep 60; done
if grep -q "injury horizon\|Parquet files written\|injury effect" logs/slope_export.log 2>/dev/null \
   || [ -f model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope/mcmc/posterior_injury_samples.parquet ]; then
  echo "[$(date +%H:%M)] slope export OK — running R"
  run_r nba_convex_max_tvrflvm_AR_injury_causal_slope slope && echo "[$(date +%H:%M)] slope R done"
else
  echo "[$(date +%H:%M)] SLOPE EXPORT DID NOT COMPLETE — rerunning"
  gate
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
    --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope \
    --model_config=config/model_config.yaml --inference_method=mcmc > logs/slope_export.log 2>&1 \
    && run_r nba_convex_max_tvrflvm_AR_injury_causal_slope slope
fi

# 2) ablation
gate
echo "[$(date +%H:%M)] ablation export"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope_only \
  --model_config=config/model_config.yaml --inference_method=mcmc > logs/slope_only_export.log 2>&1 \
  && run_r nba_convex_max_tvrflvm_AR_injury_causal_slope_only slope_only \
  && echo "[$(date +%H:%M)] ablation done"

# 3) placebo
gate
echo "[$(date +%H:%M)] placebo export"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo \
  --model_config=config/model_config.yaml --inference_method=mcmc > logs/placebo_export.log 2>&1 \
  && run_r nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo placebo \
  && echo "[$(date +%H:%M)] placebo done"
echo "[$(date +%H:%M)] INJURY SEQUENCE COMPLETE"
