#!/usr/bin/env bash
# Re-export the injury runs with the data-prep fixes applied. NO re-sampling: the MCMC samples are
# fine, only the export was inconsistent with how the model was TRAINED.
#
# Three defects fixed in model_export.py, all the same species -- the export replicated a SUBSET of
# main.py's data prep:
#   1. panel path hardcoded to injury_player_cleaned.csv, ignoring cfg["injury_data_csv"]
#   2. mechanism4 grouping omitted -> the model was FIT on 4 mechanism groups while the export
#      built categories from the 8 ungrouped names, so every per-type effect carried the wrong
#      injury label (code 1 = "Axial", 19 players, was being written as "ACL")
#   3. games_exposure not rectified by season_available_fraction -> 590 rows (2.9%, all
#      post-injury) differ by a mean of 28.5 games; a rehab season fit with denominator 51 was
#      exported as 82
# Verified after the fix: games_exposure / pct_minutes / usg / injury_code are identical across
# both prep paths, and labels resolve to Axial(19) / Fracture(157) / Knee Structural(117) /
# Tendon Rupture(47).
set -u
cd /home/abhijitbrahme/nba_functional_analysis
CFG=config/model_config.yaml
log(){ echo "[$(date +%m-%d\ %H:%M)] $*"; }

# A single export peaks around 192 GB RSS, and _injury_queue.sh launches its own exports
# independently. Gate before EVERY run (not just once at the top) so at most one export exists
# across both scripts -- otherwise the second and later iterations here would fire straight into
# whatever the queue happened to start.
MIN_FREE_GB=400
gate() {
  while pgrep -f "model_export.py --model_name" >/dev/null \
     || [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MIN_FREE_GB" ]; do
    sleep 120
  done
}

for n in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_slope_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo; do
  t=$(echo "$n" | sed 's/nba_convex_max_tvrflvm_AR_//')
  gate
  log "re-export $t  (free $(free -g | awk '/^Mem:/{print $7}')G)"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$n" \
    --model_config=$CFG --inference_method=mcmc > "logs/fx_${t}_export.log" 2>&1 \
    && log "  OK $t  (labels: $(grep -m1 -o 'panel: .*' logs/fx_${t}_export.log || echo n/a))" \
    || log "  !! FAILED $t"
done
log "===== CORRECTED RE-EXPORTS COMPLETE ====="
