#!/usr/bin/env bash
# REs + linear peak-value kernel (k_SE + k_lin + nugget), stratified_next_k, on GPU 0 (idle).
#
# Motivation: the stationary RFF/SE kernel compresses elite peak values. Regressing the shared
# (RE-free) ceiling on observed best OBPM for well-measured players gives slope 0.31 (1.0 = no
# compression); the top sextile's ceiling sits 4.14 below what those players already produced.
# The nugget only lifts the slope to 0.49 because it is unstructured -- it memorises seen players
# but cannot extrapolate that a high-X player should project higher. Adding a linear component to
# the peak-value process should let the ceiling scale with latent quality.
#
# Acceptance tests (vs the RE-only fit at the same scheme):
#   1. slope of shared ceiling on observed best rises materially above 0.31
#   2. star-tercile OBPM coverage stays >= the RE-only 95.6% and pooled ELPPD does not regress
#   3. Wembanyama's peak-value ceiling and peak age move up / later
# NOTE: an earlier pilot tested linear INSTEAD of the nugget and lost badly (ELPPD -4462). This
# run is linear IN ADDITION to it, which is the complement the earlier result argued for.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_lin_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_lin/stratified_next_k/mcmc
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }

echo "[$(date +%H:%M)] MAP (GPU 0)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=map \
  > logs/re_lin_map.log 2>&1 || { echo "[$(date +%H:%M)] MAP FAILED"; exit 1; }
# verify the linear head actually engaged before spending ~19h on MCMC
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle,sys
s=pickle.load(open('model_output/nba_convex_max_tvrflvm_AR_re_lin/stratified_next_k/map/samples.pkl','rb'))
s={k.replace('__loc',''):v for k,v in s.items()}
ok='lambda_c' in s and 'c_offset_re' in s
print('[verify] lambda_c present:', 'lambda_c' in s, '| c_offset_re present:', 'c_offset_re' in s)
sys.exit(0 if ok else 1)" || { echo "[$(date +%H:%M)] ABORT: linear head or RE not in MAP samples"; exit 1; }
echo "[$(date +%H:%M)] MAP ok (linear + RE engaged) -> MCMC"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_lin_mcmc.log 2>&1 || { echo "[$(date +%H:%M)] MCMC FAILED"; exit 1; }
echo "[$(date +%H:%M)] MCMC done -> export + coverage"
gate
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
  > logs/re_lin_export.log 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$MDIR" 2021 \
  > logs/re_lin_coverage.log 2>&1 \
&& echo "[$(date +%H:%M)] RE+LINEAR PILOT COMPLETE" || echo "[$(date +%H:%M)] export/coverage FAILED"
