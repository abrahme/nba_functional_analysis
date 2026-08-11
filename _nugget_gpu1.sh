#!/usr/bin/env bash
# Nugget rollout, GPU 1 leg: adopt the in-flight holdout_first_k MCMC (its driver was replaced),
# then run holdout_peak end to end. GPU 0 leg handles random_interior in parallel.
# Final combine is run by whichever leg finishes last (guarded by the marker files).
set -u
cd /home/abhijitbrahme/nba_functional_analysis
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }

post() {  # $1 scheme
  local scheme=$1
  local name="nba_convex_max_tvrflvm_AR_re_on_${scheme}"
  local mdir="model_output/nba_convex_max_tvrflvm_AR_re_on/${scheme}/mcmc"
  gate
  echo "[$(date +%H:%M)] ${scheme}: export + coverage"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
    > "logs/re_on_${scheme}_export.log" 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 \
    > "logs/re_on_${scheme}_coverage.log" 2>&1 \
  && echo "[$(date +%H:%M)] ${scheme} DONE" || echo "[$(date +%H:%M)] ${scheme} export/coverage FAILED"
}

# 1) adopt the running holdout_first_k MCMC
echo "[$(date +%H:%M)] waiting on in-flight holdout_first_k MCMC"
until [ -f model_output/nba_convex_max_tvrflvm_AR_re_on/holdout_first_k/mcmc/samples.pkl ]; do sleep 300; done
sleep 60
echo "[$(date +%H:%M)] holdout_first_k MCMC done"
post holdout_first_k

# 2) holdout_peak end to end on GPU 1
scheme=holdout_peak
name="nba_convex_max_tvrflvm_AR_re_on_${scheme}"
echo "[$(date +%H:%M)] ${scheme}: MAP (GPU 1)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=1 mcmc python main.py \
  --model_name="$name" --model_config=config/model_config.yaml --inference_method=map \
  > "logs/re_on_${scheme}_map.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MAP FAILED"; exit 1; }
echo "[$(date +%H:%M)] ${scheme}: MCMC (GPU 1)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=1 mcmc python main.py \
  --model_name="$name" --model_config=config/model_config.yaml --inference_method=mcmc \
  > "logs/re_on_${scheme}_mcmc.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MCMC FAILED"; exit 1; }
post holdout_peak
echo "[$(date +%H:%M)] GPU1 LEG COMPLETE"
