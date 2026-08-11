#!/usr/bin/env bash
# Re-run the slope export (its first run was killed mid-write during the memory crisis;
# posterior_counterfactual_ar.parquet et al. missing) after the current injury chain finishes.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
until grep -q "INJURY SEQUENCE COMPLETE" logs/injury_seq2.log 2>/dev/null; do sleep 300; done
while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ]; do sleep 120; done
echo "[$(date +%H:%M)] slope export FULL re-run"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope \
  --model_config=config/model_config.yaml --inference_method=mcmc > logs/slope_export.log 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
       model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope/mcmc > logs/slope_causal_r.log 2>&1 \
  && echo "[$(date +%H:%M)] SLOPE EXPORT+R COMPLETE" || echo "[$(date +%H:%M)] SLOPE RERUN FAILED"
