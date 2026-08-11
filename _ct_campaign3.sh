#!/usr/bin/env bash
# CT campaign v3 — rerun after fixing main.py's missing injury_w/dt_lo/dt_hi offsets wiring.
# In v2 the continuous clock + decay silently no-op'd (offsets.get("injury_w") was None), so the
# fits were really "discrete slope + mechanism4 + v2 panel". The stage-1 prefit is a NON-injury
# model and never reads the timing arrays, so it is reused as-is.
set -u
cd /home/abhijitbrahme/nba_functional_analysis

gate() {
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do
    sleep 120
  done
}

run_gpu() {
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=0 mcmc python main.py \
    --model_name="$1" --model_config=config/model_config.yaml --inference_method=mcmc \
    > "logs/$2.log" 2>&1
}

verify_acute() {  # $1 = model_name ; fails loudly if the continuous path is still inactive
  docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, sys
s = pickle.load(open('model_output/$1/mcmc/samples.pkl','rb'))
ok = any('acute' in k for k in s)
print('[verify] acute sites present:', ok)
print('[verify] injury sites:', sorted(k for k in s if 'injury' in k))
sys.exit(0 if ok else 1)"
}

for pair in "nba_convex_max_tvrflvm_AR_injury_causal_ct_sym ct_sym_mcmc2" \
            "nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym ct_placebo_mcmc2"; do
  set -- $pair
  echo "[$(date +%H:%M)] MCMC $1"
  if ! run_gpu "$1" "$2"; then echo "[$(date +%H:%M)] $1 MCMC FAILED"; exit 1; fi
  if verify_acute "$1" >> "logs/$2.log" 2>&1; then
    echo "[$(date +%H:%M)] $1 OK — continuous clock + decay engaged"
  else
    echo "[$(date +%H:%M)] WARNING: $1 has NO acute sites — continuous path still inactive"
  fi
done

echo "[$(date +%H:%M)] fits done -> exports + R"
for m in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym; do
  gate
  s=${m##*causal_}
  if docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$m" \
       --model_config=config/model_config.yaml --inference_method=mcmc > "logs/${s}_export2.log" 2>&1 \
     && docker exec -w /home/joyvan/work r-new Rscript data_causal/injury_two_stage_causal.r \
       "model_output/$m/mcmc" > "logs/${s}_causal_r2.log" 2>&1; then
    echo "[$(date +%H:%M)] $s export+R DONE"
  else
    echo "[$(date +%H:%M)] $s export/R FAILED"
  fi
done
echo "[$(date +%H:%M)] CT CAMPAIGN v3 COMPLETE"
