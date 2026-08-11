#!/usr/bin/env bash
# Regenerate latent_space.r outputs for the adopted RE spec after adding the cohort
# minimum-minutes eligibility filter (>= 500 minutes in a season).
#
# Why: the peak random effect is unidentified at low exposure (posterior SD rises toward the
# prior) AND under-sampled (ESS ~5-12 of 500 draws, Rhat 1.15-1.32), so a data-poor player's
# posterior mean is dominated by Monte Carlo error. Ranking top-5 within an all-data-poor draft
# cohort therefore selected the largest upward chain excursion (a 297-minute player outranking
# lottery picks). The filter removes ineligible players from the cohort ranking only; nothing
# else about the model or its predictive summaries changes.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc
while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 400 ]; do sleep 180; done
echo "[$(date +%H:%M)] latent_space.r on the RE flagship (cohort filter active)"
docker exec -w /home/joyvan/work r-new Rscript data_analysis/latent_space.r "$MDIR" \
  > logs/cohort_regen_latent.log 2>&1 \
  && echo "[$(date +%H:%M)] latent_space DONE" || { echo "[$(date +%H:%M)] latent_space FAILED"; exit 1; }
grep -E "cohort table: .* players eligible" logs/cohort_regen_latent.log || true
echo "[$(date +%H:%M)] regenerated cohort artifacts:"
ls -la --time-style=+"%m-%d_%H:%M" \
  "$MDIR"/plots/latent_space/map/breakout_by_cohort.json \
  "$MDIR"/plots/latent_space/map/breakout_by_cohort.tex \
  "$MDIR"/plots/latent_space/map/breakout_cohort_distributions.png 2>/dev/null | awk '{print "  "$6, $7}'
echo "[$(date +%H:%M)] COHORT REGEN COMPLETE"
