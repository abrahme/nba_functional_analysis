#!/usr/bin/env bash
# REs + loosened peak-value amplitude prior (sigma_c ~ InverseGamma(2,4) instead of (2,1)).
#
# Diagnosis being tested: the OBPM peak-value process is saturated at its amplitude cap
# (sigma_c[obpm]=2.09 vs a modeled shared peak SD of 2.06 -- 98%), and it is the only metric in
# that state. A process capped at +/-2.09 cannot represent players whose true peaks are 6-8, which
# is the measured compression (slope of shared ceiling on observed best = 0.31; RE-only 0.49).
# sigma_c is frozen from MAP, so loosening its prior is the way to raise it.
#
# Acceptance tests vs the RE-only fit at the same scheme:
#   1. sigma_c[obpm] rises materially above 2.09 and the shared peak SD is no longer ~98% of it
#   2. compression slope rises above 0.49
#   3. Wembanyama's ceiling lifts (RE-only MAP peak_val 3.35) and his peak age moves later
#   4. no regression: star-tercile coverage >= 95.6%, pooled ELPPD not worse
#   5. the nugget does LESS work (mean |c_offset_re| falls) -- the shared component should now
#      carry the level, relieving the badly-mixed (ESS 5-12) component
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_sc_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_sc/stratified_next_k/mcmc
MAPP=model_output/nba_convex_max_tvrflvm_AR_re_sc/stratified_next_k/map/samples.pkl
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }

echo "[$(date +%H:%M)] MAP (GPU 0)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=map \
  > logs/re_sc_map.log 2>&1 || { echo "[$(date +%H:%M)] MAP FAILED"; exit 1; }

# Early read: did the amplitude actually rise, and did Wembanyama's ceiling lift?
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, numpy as np, pandas as pd
s=pickle.load(open('$MAPP','rb')); s={k.replace('__loc',''):v for k,v in s.items()}
sc=np.asarray(s['sigma_c'],dtype=float).ravel()
print('[MAP] sigma_c[obpm] = %.3f   (RE-only was 2.093)' % sc[3])
pv=pd.read_parquet('model_output/nba_convex_max_tvrflvm_AR_re_sc/stratified_next_k/map/map_peak_vals.parquet')
w=pv[(pv.id.astype(str)=='wembavi01')&(pv.metric=='obpm')]['peak_value']
if len(w): print('[MAP] Wembanyama peak_value = %.2f   (RE-only was 3.35)' % w.mean())
o=pv[pv.metric=='obpm'].set_index(pv[pv.metric=='obpm'].id.astype(str))['peak_value']
print('[MAP] shared+RE peak SD = %.2f' % o.std())
p=pd.read_csv('data/injury_player_cleaned.csv'); p=p[p.year<=2025]
b=p.groupby('id').agg(best=('obpm','max'), tot=('minutes','sum')); b.index=b.index.astype(str)
j=pd.concat([o.rename('peak'),b],axis=1).dropna(); j=j[j.tot>=8000]
print('[MAP] compression slope = %.2f   (RE-only MAP was 0.50)' % np.polyfit(j.best,j.peak,1)[0])
" 2>&1 | grep "^\[MAP\]"

echo "[$(date +%H:%M)] -> MCMC"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_sc_mcmc.log 2>&1 || { echo "[$(date +%H:%M)] MCMC FAILED"; exit 1; }
echo "[$(date +%H:%M)] MCMC done -> export + coverage"
gate
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
  > logs/re_sc_export.log 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$MDIR" 2021 \
  > logs/re_sc_coverage.log 2>&1 \
&& echo "[$(date +%H:%M)] SIGMA_C PILOT COMPLETE" || echo "[$(date +%H:%M)] export/coverage FAILED"
