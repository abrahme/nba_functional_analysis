#!/usr/bin/env bash
# Surgical Figure-11 fix: rewrite posterior_injury_global_offset.parquet with the IDENTIFIED
# per-metric mean effect (mean over injury types) instead of the aliased raw injury_global_offset
# site, then re-run model_diagnostics.r to regenerate the injury_global_offset figure. Waits for the
# current v5 R phase to finish first so it doesn't race on the mcmc/ dir.
set -uo pipefail
cd "$(dirname "$0")"
MDIR=model_output/nba_convex_max_tvrflvm_AR_injury_causal/mcmc
echo "[regen] waiting for current v5 R phase to finish..."
while pgrep -f "Rscript.*nba_convex_max_tvrflvm_AR_injury_causal/mcmc" >/dev/null; do sleep 120; done
echo "[regen] R done; rewriting global-offset parquet with identified mean effect"
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pickle, numpy as np, pandas as pd, os
d=pickle.load(open('$MDIR/samples.pkl','rb')); d={k.replace('__loc',''):v for k,v in d.items()}
mets=['games','usg','pct_minutes','obpm','dbpm','blk','stl','ast','dreb','oreb','tov','fta','fg2a','fg3a','ftm','fg2m','fg3m']
go=np.asarray(d['injury_global_offset']); ir=np.asarray(d['injury_raw']); si=np.asarray(d['sigma_injury'])
me=(go[...,None]+si[...,None]*ir).mean(-1)   # (c,d,k) identified per-metric mean effect
c,n,k=me.shape; ci,sj,ki=np.meshgrid(np.arange(c),np.arange(n),np.arange(k),indexing='ij')
df=pd.DataFrame({'chain':ci.ravel(),'sample':sj.ravel(),'metric':np.array(mets)[ki.ravel()],'value':me.ravel()})
ego=np.asarray(d['injury_exit_global_offset']); er=np.asarray(d['injury_exit_raw']); se=np.asarray(d['sigma_injury_exit'])
hz=(ego[...,None]+se[...,None]*er).mean(-1)   # (c,d) identified hazard mean
c2,s2=np.meshgrid(np.arange(c),np.arange(n),indexing='ij')
df=pd.concat([df,pd.DataFrame({'chain':c2.ravel(),'sample':s2.ravel(),'metric':'exit_hazard','value':hz.ravel()})],ignore_index=True)
df.to_parquet('$MDIR/posterior_injury_global_offset.parquet',index=False)
print('  rewrote', len(df), 'rows; per-metric identified means:')
print(df[df.metric!='exit_hazard'].groupby('metric')['value'].mean().round(3).to_string())
"
echo "[regen] re-running model_diagnostics.r (regenerates injury_global_offset figure)"
docker exec -w /home/joyvan/work r-new Rscript data_analysis/model_diagnostics.r "$MDIR" 2021 >> logs/regen_fig11.log 2>&1
echo "[regen] done -> $MDIR/plots/injury/injury_global_offset.png"
