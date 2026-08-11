#!/usr/bin/env bash
# Athleticism-panel model: MAP -> MCMC -> export -> coverage, on GPU 1.
#
# GPU 1 ONLY. GPU 0 was running an unrelated job (`main.py --experiment snn_ablation`) when this
# was written -- check `nvidia-smi` before assuming either card is free.
#
# What this fits: the adopted RE spec (ConvexMax RFF + AR + peak REs), unchanged, but reading the
# PBPStats athleticism panel instead of the box-score panel. 10 heads, all count/binomial, all on
# possession exposures. obpm/dbpm/usg/pct_minutes are deliberately absent so that "good at
# basketball" cannot load onto X -- see the config entry for the measured justification
# (panel's first factor holds 48.5% of variance raw, 49.6% after residualising on quality).
#
# What to look at when it lands:
#   1. Does X encode athleticism? Nearest neighbours should be athletic archetypes (rim-runners
#      grouped together) rather than the production archetypes the box-score model returns.
#   2. Peak ages: rim pressure should peak EARLIER than OBPM does (~24-25 vs ~26-27) if the panel
#      is really tracking explosiveness rather than skill.
#   3. Coverage/ELPPD: sanity only. This panel is not competing with the box-score model on
#      prediction; it is a different measurement of a different construct.
#   4. THEN the injury stage-2 -- but NOT before the stage-2 sampling bug is fixed (metric-side
#      injury sites froze at MAP init, ESS 5-12, R-hat ~1e5). A new panel inherits that bug.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_athl_stratified_next_k
BASE=model_output/nba_convex_max_tvrflvm_AR_athl/stratified_next_k
MDIR=$BASE/mcmc
GPU=1

gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 300 ] || pgrep -f "model_export.py" >/dev/null; do sleep 120; done; }

echo "[$(date +%H:%M)] rebuild panel"
docker exec -w /home/joyvan/work mcmc-analysis python data/build_athleticism_panel.py \
  > logs/athl_panel.log 2>&1 || { echo "[$(date +%H:%M)] PANEL BUILD FAILED"; exit 1; }

# Resume-friendly: MAP is deterministic given the panel, so a completed MAP is reused. Delete
# $BASE/map/samples.pkl to force a refit (required if the panel or priors change).
if [ -f "$BASE/map/samples.pkl" ] && [ -z "${FORCE_MAP:-}" ]; then
  echo "[$(date +%H:%M)] MAP already present ($(date -r "$BASE/map/samples.pkl" +%m-%d\ %H:%M)) -- reusing"
else
  echo "[$(date +%H:%M)] MAP (GPU $GPU)"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=$GPU mcmc python main.py \
    --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=map \
    > logs/athl_map.log 2>&1 || { echo "[$(date +%H:%M)] MAP FAILED"; exit 1; }
fi

# Confirm the fit is real before spending a night on MCMC: latent X present and not collapsed,
# and the peak-value/peak-age sites populated for all 10 heads.
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, numpy as np, sys
s=pickle.load(open('$BASE/map/samples.pkl','rb')); s={k.replace('__loc',''):v for k,v in s.items()}
X=np.asarray(s['X']); cm=np.asarray(s['c_max']); tm=np.asarray(s['t_max_raw'])
print('[verify] X', X.shape, 'RMS %.3f' % np.sqrt((X**2).sum(1)).mean())
print('[verify] c_max', cm.shape, 'finite', bool(np.isfinite(cm).all()))
print('[verify] t_max_raw', tm.shape, 'finite', bool(np.isfinite(tm).all()))
print('[verify] c_offset_re present:', 'c_offset_re' in s)
ok = np.isfinite(X).all() and np.sqrt((X**2).sum(1)).mean() > 1e-3 and np.isfinite(cm).all()
sys.exit(0 if ok else 1)" || { echo "[$(date +%H:%M)] ABORT: MAP fit degenerate"; exit 1; }

echo "[$(date +%H:%M)] -> MCMC (GPU $GPU)"
docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=$GPU mcmc python main.py \
  --model_name="$NAME" --model_config=config/model_config.yaml --inference_method=mcmc \
  > logs/athl_mcmc.log 2>&1 || { echo "[$(date +%H:%M)] MCMC FAILED"; exit 1; }

echo "[$(date +%H:%M)] MCMC done -> export + coverage"
gate
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --with_elppd \
  > logs/athl_export.log 2>&1 \
&& docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$MDIR" 2021 \
  > logs/athl_coverage.log 2>&1 \
&& echo "[$(date +%H:%M)] ATHLETICISM RUN COMPLETE" || echo "[$(date +%H:%M)] export/coverage FAILED"
