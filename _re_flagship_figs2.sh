#!/usr/bin/env bash
# Flagship figures for the adopted spec (Concave RFF + AR + peak REs), stratified_next_k.
# v2: gates on MEMORY ONLY. v1 also waited for "no other model_export", which put the figures
# behind ~3 queued exports; with >1.2 TB free two concurrent exports (~250 GB each) fit easily.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_on_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc

memgate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do sleep 180; done; }

memgate 450
echo "[$(date +%H:%M)] FULL export (figures + loadings + derivs + phi_X)"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_flagship_export.log 2>&1 || { echo "[$(date +%H:%M)] EXPORT FAILED"; exit 1; }
echo "[$(date +%H:%M)] export done"

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
echo "[$(date +%H:%M)] RE FLAGSHIP FIGURES COMPLETE"
ls "$MDIR/plots" 2>/dev/null | tr '\n' ' '
