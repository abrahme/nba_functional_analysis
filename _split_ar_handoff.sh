#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
AR=model_output/nba_convex_max_tvrflvm_split_AR/stratified_next_k/mcmc/samples.pkl
echo "[handoff $(date '+%F %T')] waiting for split_AR MCMC samples.pkl"
while [ ! -f "$AR" ] || docker exec mcmc bash -c "pgrep -f 'main.py.*split_AR.*mcmc'" >/dev/null 2>&1; do
  sleep 120
done
sleep 30
echo "[handoff $(date '+%F %T')] split_AR MCMC done -> export + coverage/diagnostics/latent"
bash ./run_pipeline.sh stratified_next_k "15" 3 "1 2 4" 4
echo "[handoff $(date '+%F %T')] combine coverage tables (both split models)"
bash ./run_pipeline.sh stratified_next_k "14 15" 5 1 5
echo "[handoff $(date '+%F %T')] DONE"
