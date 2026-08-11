#!/usr/bin/env bash
# Flagship figures for the adopted spec (Concave RFF + AR + peak REs), stratified_next_k.
# v4, correctly ordered and idempotent:
#   1. main FULL export (phi_X, peak_vals, loadings, derivs, plot assets)  [runs now]
#   2. wait for the archetypal --peaks_no_re pass already in flight (writes *_shared.parquet)
#   3. latent_space.r (archetypes, neighbours, draft-cohort breakout) + model_diagnostics.r
#      (player plots, peaks PCA from the ARCHETYPAL peaks) in parallel
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_on_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc
memgate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do sleep 180; done; }

memgate 400
echo "[$(date +%H:%M)] main FULL export"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_flagship_export.log 2>&1 || { echo "[$(date +%H:%M)] MAIN EXPORT FAILED"; exit 1; }
echo "[$(date +%H:%M)] main export done"

echo "[$(date +%H:%M)] waiting for archetypal peaks (--peaks_no_re)"
for _ in $(seq 1 240); do
  [ -f "$MDIR/posterior_peaks_ar_shared.parquet" ] && break
  sleep 60
done
if [ -f "$MDIR/posterior_peaks_ar_shared.parquet" ]; then
  echo "[$(date +%H:%M)] archetypal peaks present"
else
  echo "[$(date +%H:%M)] archetypal peaks MISSING -> running --peaks_no_re now"
  memgate 350
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
    --model_config=config/model_config.yaml --inference_method=mcmc --peaks_no_re \
    > logs/shared_peaks_export.log 2>&1 || { echo "[$(date +%H:%M)] SHARED PEAKS FAILED"; exit 1; }
fi

memgate 500
echo "[$(date +%H:%M)] latent_space.r + model_diagnostics.r (parallel)"
docker exec -w /home/joyvan/work r-new Rscript data_analysis/latent_space.r "$MDIR" \
  > logs/re_flagship_latent.log 2>&1 &
P1=$!
docker exec -w /home/joyvan/work r-new Rscript data_analysis/model_diagnostics.r "$MDIR" 2021 \
  > logs/re_flagship_diag.log 2>&1 &
P2=$!
wait $P1 && echo "[$(date +%H:%M)] latent_space DONE" || echo "[$(date +%H:%M)] latent_space FAILED"
wait $P2 && echo "[$(date +%H:%M)] diagnostics DONE" || echo "[$(date +%H:%M)] diagnostics FAILED"
grep -q "peaks PCA: using archetypal" logs/re_flagship_diag.log 2>/dev/null \
  && echo "[verify] PCA used ARCHETYPAL (RE-free) peaks" \
  || echo "[verify] WARNING: PCA did NOT use archetypal peaks"
echo "[$(date +%H:%M)] RE FLAGSHIP FIGURES COMPLETE"
