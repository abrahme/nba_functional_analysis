#!/usr/bin/env bash
# Adopt-the-nugget rollout: fit the REs-on (c/t peak random effect) variant on the three
# remaining holdout schemes, then coverage-only+ELPPD export and coverage.r for each, then a
# final combine so the paper tables carry the adopted spec. Runs on GPU 1 (GPU 0 = CT campaign).
set -u
cd /home/abhijitbrahme/nba_functional_analysis

gate() {
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do
    sleep 120
  done
}

for scheme in holdout_first_k random_interior holdout_peak; do
  name="nba_convex_max_tvrflvm_AR_re_on_${scheme}"
  mdir="model_output/nba_convex_max_tvrflvm_AR_re_on/${scheme}/mcmc"
  echo "[$(date +%H:%M)] ${scheme}: MAP"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=1 mcmc python main.py \
    --model_name="$name" --model_config=config/model_config.yaml --inference_method=map \
    > "logs/re_on_${scheme}_map.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MAP FAILED"; continue; }
  echo "[$(date +%H:%M)] ${scheme}: MCMC"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=1 mcmc python main.py \
    --model_name="$name" --model_config=config/model_config.yaml --inference_method=mcmc \
    > "logs/re_on_${scheme}_mcmc.log" 2>&1 || { echo "[$(date +%H:%M)] ${scheme} MCMC FAILED"; continue; }
  echo "[$(date +%H:%M)] ${scheme}: MCMC done -> export + coverage"
  gate
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
    > "logs/re_on_${scheme}_export.log" 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 \
    > "logs/re_on_${scheme}_coverage.log" 2>&1 \
  && echo "[$(date +%H:%M)] ${scheme} DONE" || echo "[$(date +%H:%M)] ${scheme} export/coverage FAILED"
done

echo "[$(date +%H:%M)] all schemes done -> combine"
docker exec -w /home/joyvan/work mcmc-analysis python model_output/model_plots/coverage/combine_holdout_tables.py \
  > logs/combine_nugget.log 2>&1 && echo "[$(date +%H:%M)] COMBINE DONE"
echo "[$(date +%H:%M)] NUGGET ROLLOUT COMPLETE"
