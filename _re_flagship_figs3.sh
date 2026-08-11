#!/usr/bin/env bash
# Flagship figures for the adopted spec, correctly ordered:
#   (main export, already running) -> --peaks_no_re pass -> latent_space.r + model_diagnostics.r
# v2 raced: it started the R scripts as soon as the main export finished, so model_diagnostics.r
# could read RE-inclusive peaks for the PCA before the archetypal file existed.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_on_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc
memgate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do sleep 180; done; }

echo "[$(date +%H:%M)] waiting for the main flagship export to finish"
until [ -f "$MDIR/posterior_peaks_ar.parquet" ] && ! pgrep -f "model_export.py.*re_on_stratified" >/dev/null; do
  sleep 180
done
echo "[$(date +%H:%M)] main export complete"

memgate 350
echo "[$(date +%H:%M)] --peaks_no_re pass (archetypal peaks for the PCA figures)"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --peaks_no_re \
  > logs/shared_peaks_export.log 2>&1 || { echo "[$(date +%H:%M)] SHARED PEAKS FAILED"; exit 1; }
[ -f "$MDIR/posterior_peaks_ar_shared.parquet" ] \
  && echo "[$(date +%H:%M)] archetypal peaks written" \
  || { echo "[$(date +%H:%M)] ABORT: shared peaks missing"; exit 1; }

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
grep -c "peaks PCA: using archetypal" logs/re_flagship_diag.log 2>/dev/null | sed 's/^/[verify] archetypal-PCA message count: /'
echo "[$(date +%H:%M)] RE FLAGSHIP FIGURES COMPLETE"
