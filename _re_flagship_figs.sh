#!/usr/bin/env bash
# Flagship figure regeneration for the ADOPTED spec (Concave RFF + AR + peak REs), on the
# stratified_next_k scheme that the main text's figures use.
#
# The RE contribution flows into every downstream figure automatically: compute_curves resolves
# c_offset/t_offset through _resolve_c_offset/_resolve_t_offset, so peaks, mu, and the predictive
# draws all carry the per-player peak-value/peak-age offsets. That includes the draft-cohort
# breakout table/JSON, which latent_space.r builds from the PREDICTIVE OBPM at the posterior peak
# age -- i.e. curve + AR + RE, as required.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_on_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc

gate() {  # $1 = GB required
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ] || pgrep -f "model_export.py" >/dev/null; do
    sleep 180
  done
}

# Sanity: the REs must actually be in the samples, else the export would silently disable them.
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, sys
s = pickle.load(open('$MDIR/samples.pkl','rb'))
ok = ('c_offset_re' in s) and ('t_offset_re' in s)
print('[check] c_offset_re/t_offset_re in samples:', ok)
sys.exit(0 if ok else 1)" || { echo "[$(date +%H:%M)] ABORT: RE sites missing from samples"; exit 1; }

gate 320
echo "[$(date +%H:%M)] FULL export (figures + loadings + derivs + phi_X)"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/re_flagship_export.log 2>&1 || { echo "[$(date +%H:%M)] EXPORT FAILED"; exit 1; }
echo "[$(date +%H:%M)] export done"

# latent_space.r (archetypes, neighbours, PCA, draft-cohort breakout) and model_diagnostics.r
# (player trajectory plots, curvature players, rhat) are independent -> run concurrently.
gate 500
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
