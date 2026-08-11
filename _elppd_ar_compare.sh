#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
REF=1784845087   # only fire on a coverage csv written AFTER this watcher was armed
VS=model_output/model_plots/coverage/nba_convex_max_tvrflvm_split_AR_stratified_next_k_vs_stratified_next_k_f20_k2_s42.csv
echo "[elppd $(date '+%F %T')] waiting for FRESH split_AR coverage csv"
while [ ! -f "$VS" ] || [ "$(stat -c %Y "$VS" 2>/dev/null || echo 0)" -lt "$REF" ]; do
  sleep 120
done
sleep 20   # settle
echo "[elppd $(date '+%F %T')] split_AR coverage landed -> AR split vs non-split ELPPD"
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pandas as pd, numpy as np
G='model_output/model_plots/coverage/'
def load(name):
    df=pd.read_csv(G+name+'_stratified_next_k_vs_stratified_next_k_f20_k2_s42.csv')
    return df[df['split']=='holdout'][['metric','avg_log_loss','n_obs']].set_index('metric')
sp=load('nba_convex_max_tvrflvm_split_AR'); ns=load('nba_convex_max_tvrflvm_AR')
m=sp.join(ns,lsuffix='_split',rsuffix='_nonsplit'); m['delta']=m.avg_log_loss_split-m.avg_log_loss_nonsplit
print('=== AR: HOLDOUT avg log-loss (lower better). delta<0 => split_AR better ===')
print('  %-12s %10s %10s %8s'%('metric','split_AR','nonsplit','delta'))
for k in m.sort_values('delta').index:
    r=m.loc[k]; print('  %-12s %10.3f %10.3f %+8.3f'%(k,r.avg_log_loss_split,r.avg_log_loss_nonsplit,r.delta))
for lbl,c,n in [('split_AR','avg_log_loss_split','n_obs_split'),('nonsplit_AR','avg_log_loss_nonsplit','n_obs_nonsplit')]:
    w=np.average(m[c],weights=m[n]); tot=-(m[c]*m[n]).sum()
    print('  OVERALL %-11s weighted mean log-loss=%.4f  total ELPPD=%.0f'%(lbl,w,tot))
print('  split_AR wins %d/%d metrics'%((m.delta<0).sum(),len(m)))
"
echo "[elppd $(date '+%F %T')] DONE"
