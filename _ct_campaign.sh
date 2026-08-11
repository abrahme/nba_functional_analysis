#!/usr/bin/env bash
# Continuous-time injury campaign: waits for GPU 0 (stratified nugget MCMC) to finish, then
# runs sym_v2 prefit MAP -> ct_sym MCMC -> ct_placebo_sym MCMC, then CPU exports + R for both.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
until [ -f model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc/samples.pkl ]; do sleep 300; done
sleep 120
run_gpu() { docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$1" --model_config=config/model_config.yaml --inference_method="$2" \
  > "logs/$3.log" 2>&1; }
echo "[$(date +%H:%M)] GPU 0 free — sym_v2 prefit MAP"
run_gpu nba_convex_max_tvrflvm_AR_causal_sym_v2_prefit map ct_prefit_map || { echo "PREFIT FAILED"; exit 1; }
echo "[$(date +%H:%M)] ct_sym MCMC"
run_gpu nba_convex_max_tvrflvm_AR_injury_causal_ct_sym mcmc ct_sym_mcmc || { echo "CT_SYM FAILED"; exit 1; }
echo "[$(date +%H:%M)] ct_placebo_sym MCMC"
run_gpu nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym mcmc ct_placebo_mcmc || { echo "CT_PLACEBO FAILED"; exit 1; }
echo "[$(date +%H:%M)] GPU done — exports (memory-gated)"
for m in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym; do
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$m" \
    --model_config=config/model_config.yaml --inference_method=mcmc > "logs/${m##*_AR_}_export.log" 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
       "model_output/$m/mcmc" > "logs/${m##*_AR_}_causal_r.log" 2>&1 \
  && echo "[$(date +%H:%M)] $m export+R done" || echo "[$(date +%H:%M)] $m export/R FAILED"
done
echo "[$(date +%H:%M)] CT CAMPAIGN COMPLETE"
