#!/usr/bin/env bash
# Peak-VALUE RE only (c_offset_re on, t_offset_re OFF), stratified_next_k, GPU 0.
#
# Rationale (measured on the RE-on fit, OBPM peak age):
#   shared RBF only : within-player SD 1.01 vs between-player SD 2.08  -> ratio 0.48 (identified)
#   RBF + age RE    : within-player SD 2.06 vs between-player SD 2.26  -> ratio 0.92 (not)
# The age RE adds variance without discrimination; Jokic (10 seasons, peak already observed) still
# gets a 9-year 95% CI. The VALUE RE scores 0.44 on the same measure and is what fixes the
# star-coverage bias, so it stays.
#
# Acceptance tests vs the RE-on fit at the same scheme:
#   1. peak-age within/between ratio returns toward ~0.5, and Jokic's peak-age CI narrows
#   2. star-tercile OBPM coverage stays >= 95.6%  (the result we are protecting)
#   3. pooled holdout ELPPD does not regress vs -48408
#   4. Wembanyama's peak age stops being pinned at the edge of his observed data
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_c_only_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_c_only/stratified_next_k/mcmc
MAPD=model_output/nba_convex_max_tvrflvm_AR_re_c_only/stratified_next_k/map
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }

echo "[$(date +%H:%M)] MAP (GPU 0)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=map \
  > logs/re_c_only_map.log 2>&1 || { echo "[$(date +%H:%M)] MAP FAILED"; exit 1; }

# confirm the age RE really is gone and the value RE really is present
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle,sys
s=pickle.load(open('$MAPD/samples.pkl','rb')); s={k.replace('__loc',''):v for k,v in s.items()}
c='c_offset_re' in s; t='t_offset_re' in s
print('[verify] c_offset_re present:', c, '| t_offset_re present:', t)
sys.exit(0 if (c and not t) else 1)" || { echo "[$(date +%H:%M)] ABORT: RE knobs not as intended"; exit 1; }

echo "[$(date +%H:%M)] -> MCMC"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_c_only_mcmc.log 2>&1 || { echo "[$(date +%H:%M)] MCMC FAILED"; exit 1; }
echo "[$(date +%H:%M)] MCMC done -> export + coverage"
gate
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
  > logs/re_c_only_export.log 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$MDIR" 2021 \
  > logs/re_c_only_coverage.log 2>&1 \
&& echo "[$(date +%H:%M)] PEAK-VALUE-RE-ONLY PILOT COMPLETE" || echo "[$(date +%H:%M)] export/coverage FAILED"
