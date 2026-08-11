#!/usr/bin/env bash
# Serial injury queue, GPU 1. Order is deliberate: the ORIGINAL box-score suite first (that is
# what the paper claims), then the export repairs, then the athleticism mirror.
#
# Two independent defects motivate this, both found 08-08:
#
#  (1) STAGE-2 FREEZE. `init_path` is the injury-free prefit, so a map init supplies only 2 of the
#      10 free sites; the 8 injury sites fall through to init_to_uniform. On the continuous-time
#      path the chains then never move -- within-chain SD ~1e-6, R-hat ~1e5, i.e. a "posterior"
#      that is just its initialisation. ct_sym and ct_placebo_sym are the ONLY two runs affected
#      (verified: the eight slope_*/injury_causal runs share the pattern and sample fine at
#      SD 0.05-0.30). Fix = mcmc_init: median, measured to lift within-chain SD to ~5e-2.
#
#  (2) EXPORT PANEL. model_export.py hardcoded data/injury_player_cleaned.csv and ignored
#      cfg["injury_data_csv"], so any run on an alternate panel was FIT on one dataframe and
#      EXPORTED against another. Six entries affected -- including both placebo runs, whose whole
#      point is that their onsets are pseudo-onsets. Fix is in; the artifacts need regenerating.
#
# Stage 4 (athleticism) is LAST on purpose: running it before the box-score suite is known good
# would just reproduce an unknown bug on new data.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
GPU=1
CFG=config/model_config.yaml
log(){ echo "[$(date +%m-%d\ %H:%M)] $*"; }
gate(){ while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 250 ]; do sleep 120; done; }

run_mcmc(){ # name logtag
  log "MCMC $1"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=$GPU mcmc python main.py \
    --model_name="$1" --model_config=$CFG --inference_method=mcmc > "logs/q_$2_mcmc.log" 2>&1 \
    || { log "  !! MCMC FAILED $1"; return 1; }
}
run_map(){ log "MAP  $1"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=$GPU mcmc python main.py \
    --model_name="$1" --model_config=$CFG --inference_method=map > "logs/q_$2_map.log" 2>&1 \
    || { log "  !! MAP FAILED $1"; return 1; }
}
run_export(){ gate; log "export $1"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$1" \
    --model_config=$CFG --inference_method=mcmc > "logs/q_$2_export.log" 2>&1 \
    || { log "  !! EXPORT FAILED $1"; return 1; }
}
# Confirm the injury sites actually moved -- the whole point of the fix. Fails loudly instead of
# silently producing another frozen "posterior".
check_moved(){
  docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle,numpy as np,sys
s=pickle.load(open('model_output/$1/mcmc/samples.pkl','rb'))
ks=[k for k in s if 'injury' in k.lower() and not k.startswith('sigma_injury')]
sd=np.nanmean([np.nanmean(np.nanstd(np.asarray(s[k],float),axis=1)) for k in ks])
print('  [check] mean within-chain SD %.2e over %d injury sites'%(sd,len(ks)))
sys.exit(0 if sd>1e-4 else 1)" || { log "  !! STILL FROZEN: $1"; return 1; }
}

log "===== STAGE 1: original box-score suite, re-run with the init fix ====="
for n in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym; do
  t=$(echo "$n" | sed 's/nba_convex_max_tvrflvm_AR_//')
  run_mcmc "$n" "$t" && check_moved "$n" && run_export "$n" "$t" && log "  OK $t"
done

log "===== STAGE 2: re-export alternate-panel runs against their CORRECT panel ====="
for n in nba_convex_max_tvrflvm_AR_injury_causal_slope_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo; do
  t=$(echo "$n" | sed 's/nba_convex_max_tvrflvm_AR_//')
  [ -f "model_output/$n/mcmc/samples.pkl" ] && run_export "$n" "re_$t" && log "  OK re-export $t"
done

log "===== STAGE 3: athleticism prefit (stage-1 counterfactual) ====="
run_map nba_convex_max_tvrflvm_AR_athl_causal_sym_prefit athl_prefit || exit 1

log "===== STAGE 4: athleticism injury suite ====="
for n in nba_convex_max_tvrflvm_AR_athl_injury_causal_ct_sym \
         nba_convex_max_tvrflvm_AR_athl_injury_causal_ct_placebo_sym; do
  t=$(echo "$n" | sed 's/nba_convex_max_tvrflvm_AR_athl_//')
  run_mcmc "$n" "athl_$t" && check_moved "$n" && run_export "$n" "athl_$t" && log "  OK athl_$t"
done

log "===== QUEUE COMPLETE ====="
