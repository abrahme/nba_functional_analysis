#!/usr/bin/env bash
# Test the RE-fallthrough fix on the legacy GP+AR cell: does disabling REs whose sites are
# absent from the samples restore the published ~87.5% coverage?
set -u
cd /home/abhijitbrahme/nba_functional_analysis
while pgrep -f "model_export.py" >/dev/null || [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ]; do sleep 120; done
echo "[$(date +%H:%M)] export nba_tvlinearlvm_AR_holdout_last_k (RE-fallthrough fix)"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py \
  --model_name=nba_tvlinearlvm_AR_holdout_last_k --model_config=config/model_config.yaml \
  --inference_method=mcmc --coverage_only > logs/legacy_retest_export.log 2>&1 \
  || { echo "[$(date +%H:%M)] EXPORT FAILED"; exit 1; }
grep -E "^\[export\]|^\[legacy\]" logs/legacy_retest_export.log
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pyarrow.dataset as ds, pandas as pd, numpy as np
raw=pd.read_csv('data/injury_player_cleaned.csv')[['id','age','usg']].dropna()
raw['player']=raw['id'].astype(str); raw['usg_model']=raw['usg']/100+0.01
t=ds.dataset('model_output/nba_tvlinearlvm_AR/holdout_last_k/mcmc/posterior_ar.parquet').to_table(
    filter=ds.field('metric')=='usg', columns=['player','age','value']).to_pandas()
g=t.groupby(['player','age'])['value'].mean().reset_index()
m=g.merge(raw,on=['player','age']); r=m['value']-m['usg_model']
print(f'USG bias with RE fix: {r.mean():+.4f}  (normal-recon 0.459 | W_proj-pin 0.043 | healthy ref -0.006)')
"
echo "[$(date +%H:%M)] coverage.r"
docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r \
  model_output/nba_tvlinearlvm_AR/holdout_last_k/mcmc 2021 > logs/legacy_retest_cov.log 2>&1 \
  && grep -E "^All " model_output/nba_tvlinearlvm_AR/holdout_last_k/mcmc/plots/coverage/coverage_basic.tex \
  || echo "coverage FAILED"
echo "[$(date +%H:%M)] RETEST DONE (published GP+AR = 87.5%)"
