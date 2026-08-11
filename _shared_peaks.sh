#!/usr/bin/env bash
# Archetypal (RE-free) peaks for the peak-age / peak-value PCA figures on the adopted spec.
# Waits for the main flagship export to finish (they write to the same model_dir), then runs the
# --peaks_no_re pass, which zeroes c/t_offset_re + curve_re and writes ONLY:
#   posterior_peaks_ar_shared.parquet, posterior_peak_vals_ar_shared.parquet
# model_diagnostics.r picks those up automatically for the PCA blocks and leaves every other
# figure (and coverage) on the RE-inclusive peaks.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
NAME=nba_convex_max_tvrflvm_AR_re_on_stratified_next_k
MDIR=model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc

# wait for the main flagship export to complete (it writes posterior_peaks_ar.parquet last-ish)
until grep -q "export done" logs/re_flagship_figs2.log 2>/dev/null; do sleep 180; done
while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 350 ]; do sleep 180; done

echo "[$(date +%H:%M)] --peaks_no_re pass"
docker exec -w /home/joyvan/work mcmc-analysis python model_export.py --model_name="$NAME" \
  --model_config=config/model_config.yaml --inference_method=mcmc --peaks_no_re \
  > logs/shared_peaks_export.log 2>&1 \
  || { echo "[$(date +%H:%M)] SHARED PEAKS EXPORT FAILED"; exit 1; }
grep -E "^\[export\] --peaks_no_re|^--peaks_no_re" logs/shared_peaks_export.log
ls -la --time-style=+"%H:%M" "$MDIR"/posterior_peaks_ar_shared.parquet \
   "$MDIR"/posterior_peak_vals_ar_shared.parquet 2>/dev/null | awk '{print "  "$6, $7}'
echo "[$(date +%H:%M)] SHARED PEAKS DONE"
