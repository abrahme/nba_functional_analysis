#!/usr/bin/env bash
# Re-run the stage-2 injury fits with the MONOTONE overflow guard and report the diagnostics that
# matter. The previous guard (hard cap on the Gompertz exponent) created a flat, artificially
# high-likelihood plateau: the exit offset drifted to +3.0 (gamma*seg ~ 850 vs clamp 60) and the
# kink froze every other site at its MAP init. Success criteria:
#   1. exit offset returns to a sane range (pre-guard runs sat near -0.6)
#   2. the metric-side sites MOVE (within-chain SD comparable to their posterior scale)
#   3. split-Rhat is O(1), not O(1e5)
set -u
cd /home/abhijitbrahme/nba_functional_analysis

# wait for a GPU to free (nugget rollout owns both until its MCMCs finish)
echo "[$(date +%H:%M)] waiting for a free GPU"
pick_gpu() {
  for g in 0 1; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    [ -n "$u" ] && [ "$u" -lt 2000 ] && { echo $g; return; }
  done
  echo ""
}
G=""
while [ -z "$G" ]; do G=$(pick_gpu); [ -z "$G" ] && sleep 300; done
echo "[$(date +%H:%M)] using GPU $G"

for m in nba_convex_max_tvrflvm_AR_injury_causal_ct_sym \
         nba_convex_max_tvrflvm_AR_injury_causal_ct_placebo_sym; do
  s=${m##*causal_}
  echo "[$(date +%H:%M)] refit $s"
  docker exec -w /home/joyvan/work -e CUDA_VISIBLE_DEVICES=$G mcmc python main.py \
    --model_name="$m" --model_config=config/model_config.yaml --inference_method=mcmc \
    > "logs/${s}_fixedguard.log" 2>&1 || { echo "[$(date +%H:%M)] $s FAILED"; continue; }
  docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, numpy as np
def rhat(a):
    a=np.asarray(a,dtype=float)
    if a.ndim==2: a=a[:,:,None]
    a=a.reshape(a.shape[0],a.shape[1],-1); m,n=a.shape[0],a.shape[1]
    W=a.var(axis=1,ddof=1).mean(0); B=n*a.mean(axis=1).var(axis=0,ddof=1)
    return np.sqrt(np.where(W>0,(((n-1)/n)*W+B/n)/W,1.0))
s=pickle.load(open('model_output/$m/mcmc/samples.pkl','rb'))
print('  --- $s diagnostics (monotone guard) ---')
for k in ['injury_exit_global_offset','injury_global_offset','injury_slope_global_offset','injury_acute_global_offset']:
    if k not in s: continue
    a=np.asarray(s[k],dtype=float)
    sd=a.std(axis=1).mean(); rel=sd/(np.abs(a).mean()+1e-12)
    print(f'  {k:32s} mean={a.mean():+7.3f} within-sd={sd:.2e} rel={rel:.1e} rhat={rhat(a).max():.2f}')
" 2>&1 | tail -8
done
echo "[$(date +%H:%M)] CT RERUN (fixed guard) COMPLETE"
