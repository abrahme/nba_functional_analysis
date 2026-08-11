#!/usr/bin/env bash
# Parallel coverage re-export + measurement driver (replaces the serial run_pipeline fan-out).
# For each (model, scheme): coverage-only re-export (CPU, only if the parquet predates the
# noise-scale fix) then coverage.r. Runs POOL units concurrently. Idempotent + resumable:
# skips export when the parquet is already newer than the fix, skips coverage when the tex is
# already newer than the parquet. Finishes with the phase-5 combine.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
mkdir -p logs/covfan

POOL=6
YEAR=2021
FIX_REF=model/inference_utils.py          # the file carrying the A1 noise-scale fix
CTR_EXPORT=mcmc-analysis
CTR_R=r-new
EXEC="docker exec -w /home/joyvan/work"

MODELS=(tvlvm ar naive tvlinearlvm tvlinearlvm_AR rflvm rflvm_AR)
SCHEMES=(holdout_last_k holdout_first_k random_interior holdout_peak stratified_next_k)

model_name() { case $1 in
  tvlvm)          echo "nba_convex_max_tvlinearlvm_$2" ;;
  ar)             echo "nba_convex_max_tvlinearlvm_AR_$2" ;;
  naive)          echo "nba_naive_$2" ;;
  tvlinearlvm)    echo "nba_tvlinearlvm_$2" ;;
  tvlinearlvm_AR) echo "nba_tvlinearlvm_AR_$2" ;;
  rflvm)          echo "nba_convex_max_tvrflvm_$2" ;;
  rflvm_AR)       echo "nba_convex_max_tvrflvm_AR_$2" ;;
esac; }
model_base() { case $1 in
  tvlvm) echo nba_convex_max_tvlinearlvm ;; ar) echo nba_convex_max_tvlinearlvm_AR ;;
  naive) echo nba_naive ;; tvlinearlvm) echo nba_tvlinearlvm ;;
  tvlinearlvm_AR) echo nba_tvlinearlvm_AR ;; rflvm) echo nba_convex_max_tvrflvm ;;
  rflvm_AR) echo nba_convex_max_tvrflvm_AR ;;
esac; }

run_one() {
  local key=$1 scheme=$2
  local name mdir par tex log
  name=$(model_name "$key" "$scheme")
  mdir="model_output/$(model_base "$key")/$scheme/mcmc"
  par="$mdir/posterior_ar.parquet"
  tex="$mdir/plots/coverage/coverage_basic.tex"
  log="logs/covfan/${key}_${scheme}.log"
  : > "$log"

  if [[ ! -f "$mdir/samples.pkl" ]]; then echo "SKIP(no samples): $mdir" | tee -a "$log"; return; fi

  # Re-export unless the parquet is already newer than the fix.
  if [[ ! -f "$par" || "$FIX_REF" -nt "$par" ]]; then
    echo "[$(date +%H:%M)] EXPORT $name" | tee -a "$log"
    $EXEC "$CTR_EXPORT" python model_export.py --model_name="$name" \
      --model_config=config/model_config.yaml --inference_method=mcmc --coverage_only >>"$log" 2>&1 \
      || { echo "EXPORT FAILED $name" | tee -a "$log"; return; }
  else
    echo "[$(date +%H:%M)] export up-to-date $name" | tee -a "$log"
  fi

  # Run coverage.r unless the tex is already newer than the (possibly just-rewritten) parquet.
  if [[ ! -f "$tex" || "$par" -nt "$tex" ]]; then
    echo "[$(date +%H:%M)] COVERAGE $mdir" | tee -a "$log"
    $EXEC "$CTR_R" Rscript data_analysis/coverage.r "$mdir" "$YEAR" >>"$log" 2>&1 \
      || { echo "COVERAGE FAILED $mdir" | tee -a "$log"; return; }
  else
    echo "[$(date +%H:%M)] coverage up-to-date $mdir" | tee -a "$log"
  fi
  echo "[$(date +%H:%M)] DONE $name" | tee -a "$log"
}
export -f run_one model_name model_base
export EXEC CTR_EXPORT CTR_R YEAR FIX_REF

# Emit the 35 (key scheme) pairs; xargs runs POOL at a time.
for k in "${MODELS[@]}"; do for s in "${SCHEMES[@]}"; do echo "$k $s"; done; done \
  | xargs -P "$POOL" -L1 bash -c 'run_one "$0" "$1"'

echo "=== [$(date +%H:%M)] all coverage jobs done — running phase-5 combine ==="
$EXEC "$CTR_EXPORT" python model_output/model_plots/coverage/combine_holdout_tables.py
echo "=== [$(date +%H:%M)] COMBINE DONE — paper coverage tables regenerated ==="
