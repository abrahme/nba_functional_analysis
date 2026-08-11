#!/usr/bin/env bash
# Nugget rollout, GPU 0 leg: random_interior (the GPU 1 leg keeps holdout_first_k, then peak).
set -u
cd /home/abhijitbrahme/nba_functional_analysis
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }
scheme=random_interior
name="nba_convex_max_tvrflvm_AR_re_on_${scheme}"
mdir="model_output/nba_convex_max_tvrflvm_AR_re_on/${scheme}/mcmc"
echo "[$(date +%H:%M)] ${scheme}: MAP (GPU 0)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$name" --model_config=config/model_config.yaml --inference_method=map \
  > "logs/re_on_${scheme}_map.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MAP FAILED"; exit 1; }
echo "[$(date +%H:%M)] ${scheme}: MCMC (GPU 0)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$name" --model_config=config/model_config.yaml --inference_method=mcmc \
  > "logs/re_on_${scheme}_mcmc.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MCMC FAILED"; exit 1; }
echo "[$(date +%H:%M)] ${scheme}: export + coverage"
gate
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
  --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
  > "logs/re_on_${scheme}_export.log" 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 \
  > "logs/re_on_${scheme}_coverage.log" 2>&1 \
&& echo "[$(date +%H:%M)] ${scheme} DONE" || echo "[$(date +%H:%M)] ${scheme} export/coverage FAILED"
