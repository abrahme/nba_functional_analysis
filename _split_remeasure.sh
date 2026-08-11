#!/usr/bin/env bash
# Re-measure the split-RFF variants (per-modality RFF kernels: independent frequency draw + ARD
# lengthscale for peak age / peak value / curvature) on the SAME footing as the other seven
# variants. Their fits are from 07-23; their exports/coverage predate the A1 predictive-noise fix
# (07-26 12:30), so the existing numbers (split OBPM 65.4 / All 83.5 ; split_AR OBPM 90.3 /
# All 90.3) are not comparable to anything measured this session.
#
# CPU only -- does not touch the GPUs running the nugget rollout. Both variants run concurrently;
# memory is ample (>1 TB free) and each export peaks ~250 GB.
set -u
cd /home/abhijitbrahme/nba_functional_analysis

run_one() {
  local name=$1 base=$2
  local mdir="model_output/${base}/stratified_next_k/mcmc"
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 350 ]; do sleep 180; done
  echo "[$(date +%H:%M)] $base: export (+elppd)"
  docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$name" \
    --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only --with_elppd \
    > "logs/split_${base}_export.log" 2>&1 || { echo "[$(date +%H:%M)] $base EXPORT FAILED"; return 1; }
  echo "[$(date +%H:%M)] $base: coverage.r"
  docker exec -w /home/joyvan/work r-new Rscript data_analysis/coverage.r "$mdir" 2021 \
    > "logs/split_${base}_coverage.log" 2>&1 || { echo "[$(date +%H:%M)] $base COVERAGE FAILED"; return 1; }
  echo "[$(date +%H:%M)] $base DONE"
  grep -E "^OBPM |^DBPM |^All " "$mdir/plots/coverage/coverage_basic.tex" 2>/dev/null | sed "s/^/  [$base] /"
}

run_one nba_convex_max_tvrflvm_split_stratified_next_k    nba_convex_max_tvrflvm_split    &
P1=$!
run_one nba_convex_max_tvrflvm_split_AR_stratified_next_k nba_convex_max_tvrflvm_split_AR &
P2=$!
wait $P1; wait $P2

echo "[$(date +%H:%M)] ===== SPLIT RE-MEASUREMENT COMPLETE ====="
docker exec -w /home/joyvan/work mcmc-analysis python -c "
import pandas as pd, re, os
def cov(p):
    d={}
    if not os.path.exists(p): return d
    for line in open(p):
        m=re.match(r'\s*([A-Za-z0-9%\\\\]+)\s*&\s*([\d.]+)\\\\%', line)
        if m: d[m.group(1).replace('\\\\','')]=float(m.group(2))
    return d
rows=[('split',        'model_output/nba_convex_max_tvrflvm_split/stratified_next_k/mcmc'),
      ('split_AR',     'model_output/nba_convex_max_tvrflvm_split_AR/stratified_next_k/mcmc'),
      ('flagship RFF+AR','model_output/nba_convex_max_tvrflvm_AR/stratified_next_k/mcmc'),
      ('adopted RE+AR', 'model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc')]
print(f\"{'model':18s} {'OBPM':>7s} {'DBPM':>7s} {'All':>7s} {'holdout ELPPD':>15s}\")
for lbl,d in rows:
    c=cov(f'{d}/plots/coverage/coverage_basic.tex')
    e=''
    f=f'{d}/posterior_elppd.parquet'
    if os.path.exists(f):
        t=pd.read_parquet(f); h=t[t.split=='holdout'].set_index('metric')
        if 'all' in h.index: e=f\"{h.loc['all','elppd']:.0f}\"
    print(f\"{lbl:18s} {c.get('OBPM',float('nan')):6.1f}% {c.get('DBPM',float('nan')):6.1f}% {c.get('All',float('nan')):6.1f}% {e:>15s}\")
"
