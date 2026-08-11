#!/usr/bin/env bash
# Restore the one legacy cell I re-exported under the (worse) normal reconstruction during the
# bias test, then recombine so the paper tables are internally consistent again.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }
gate
echo "[$(date +%H:%M)] re-export nba_tvlinearlvm_AR_holdout_last_k (pinned reconstruction)"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_tvlinearlvm_AR_holdout_last_k --model_config=config/model_config.yaml \
  --inference_method=mcmc --coverage_only > logs/legacy_restore_export.log 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r \
     model_output/nba_tvlinearlvm_AR/holdout_last_k/mcmc 2021 > logs/legacy_restore_cov.log 2>&1 \
&& echo "[$(date +%H:%M)] cell restored" || { echo "[$(date +%H:%M)] RESTORE FAILED"; exit 1; }
echo "[$(date +%H:%M)] combine"
docker exec -w /home/joyvan/work mcmc-analysis python model_output/model_plots/coverage/combine_holdout_tables.py \
  > logs/combine_restore.log 2>&1 && echo "[$(date +%H:%M)] COMBINE DONE"
