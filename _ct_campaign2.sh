#!/usr/bin/env bash
# CT campaign (GPU 0 free now): prefit MAP -> ct_sym MCMC -> ct_placebo MCMC -> exports + R.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }
run_gpu() { docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
  --model_name="$1" --model_config=config/model_config.yaml --inference_method="$2" > "logs/$3.log" 2>&1; }
echo "[$(date +%H:%M)] prefit MAP (sym v2 union panel)"
run_gpu nba_convex_max_tvrflvm_AR_causal_sym_v2_prefit map ct_prefit_map || { echo "[$(date +%H:%M)] PREFIT FAILED"; exit 1; }
echo "[$(date +%H:%M)] prefit done -> ct_sym MCMC"
run_gpu nba_convex_max_tvrflvm_AR_injury_causal_ct_sym mcmc ct_sym_mcmc || { echo "[$(date +%H:%M)] CT_SYM FAILED"; exit 1; }
echo "[$(date +%H:%M)] ct_sym done -> ct_placebo MCMC"
run_gpu nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym mcmc ct_placebo_mcmc || { echo "[$(date +%H:%M)] CT_PLACEBO FAILED"; exit 1; }
echo "[$(date +%H:%M)] both CT fits done -> exports"
for m in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym; do
  gate 300
  s=${m##*causal_}
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$m" \
    --model_config=config/model_config.yaml --inference_method=mcmc > "logs/${s}_export.log" 2>&1 \
  && docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
     "model_output/$m/mcmc" > "logs/${s}_causal_r.log" 2>&1 \
  && echo "[$(date +%H:%M)] $s export+R DONE" || echo "[$(date +%H:%M)] $s export/R FAILED"
done
echo "[$(date +%H:%M)] CT CAMPAIGN COMPLETE"
