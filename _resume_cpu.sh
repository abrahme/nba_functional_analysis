#!/usr/bin/env bash
# Stranded CPU work after session teardown: two GPU-chain verdicts, legacy tail + combine,
# sym injury R. Each job memory-gated; pool width 3.
set -u
cd /home/abhijitbrahme/nba_functional_analysis
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 320 ]; do sleep 120; done; }
EXP="docker exec -w /home/joyvan/work mcmc-analysis python model_export.py"
RSC="docker exec -w /home/joyvan/work r-new Rscript"

job_verdict() {  # $1 model_name  $2 mdir  $3 stem
  gate; echo "[$(date +%H:%M)] $3 export (+elppd)"
  $EXP --model_name="$1" --model_config=config/model_config.yaml --inference_method=mcmc \
       --coverage_only --with_elppd > "logs/$3_export.log" 2>&1 || { echo "[$(date +%H:%M)] $3 EXPORT FAILED"; return 1; }
  gate; echo "[$(date +%H:%M)] $3 coverage.r"
  $RSC data_analysis/coverage.r "$2" 2021 > "logs/$3_coverage.log" 2>&1 \
    && echo "[$(date +%H:%M)] $3 VERDICT READY" || echo "[$(date +%H:%M)] $3 COVERAGE FAILED"
}
job_legacy() {
  for pair in "nba_tvlinearlvm_AR random_interior" "nba_tvlinearlvm_AR stratified_next_k"; do
    set -- $pair; b=$1; s=$2
    gate; echo "[$(date +%H:%M)] legacy $b/$s"
    $EXP --model_name="${b}_${s}" --model_config=config/model_config.yaml --inference_method=mcmc \
         --coverage_only >> "logs/covfan/legacy_${b}_${s}.log" 2>&1 \
    && $RSC data_analysis/coverage.r "model_output/$b/$s/mcmc" 2021 >> "logs/covfan/legacy_${b}_${s}.log" 2>&1 \
    && echo "[$(date +%H:%M)] DONE ${b}_${s}" >> "logs/covfan/legacy_${b}_${s}.log" \
    || { echo "[$(date +%H:%M)] legacy ${b}_${s} FAILED"; }
  done
  echo "[$(date +%H:%M)] legacy done -> COMBINE"
  docker exec -w /home/joyvan/work mcmc-analysis python model_output/model_plots/coverage/combine_holdout_tables.py \
    > logs/combine_final.log 2>&1 && echo "[$(date +%H:%M)] COMBINE DONE — paper tables regenerated"
}
job_sym() {
  gate; echo "[$(date +%H:%M)] slope_sym R"
  $RSC data_causal/injury_two_stage_causal.r \
    model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope_sym/mcmc > logs/slope_sym_causal_r.log 2>&1 \
    && echo "[$(date +%H:%M)] slope_sym R done" || echo "[$(date +%H:%M)] slope_sym R FAILED"
  gate; echo "[$(date +%H:%M)] placebo_sym export"
  $EXP --model_name=nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym \
       --model_config=config/model_config.yaml --inference_method=mcmc > logs/placebo_sym_export.log 2>&1 \
  && $RSC data_causal/injury_two_stage_causal.r \
       model_output/nba_convex_max_tvrflvm_AR_injury_causal_slope_placebo_sym/mcmc > logs/placebo_sym_causal_r.log 2>&1 \
  && echo "[$(date +%H:%M)] placebo_sym export+R done" || echo "[$(date +%H:%M)] placebo_sym FAILED"
}
export -f job_verdict job_legacy job_sym gate; export EXP RSC
{
  echo "verdict nba_convex_max_tvrflvm_AR_linear_holdout_last_k model_output/nba_convex_max_tvrflvm_AR_linear/holdout_last_k/mcmc linear_pilot"
  echo "verdict nba_convex_max_tvrflvm_AR_re_on_stratified_next_k model_output/nba_convex_max_tvrflvm_AR_re_on/stratified_next_k/mcmc re_on_strat"
  echo "legacy - - -"
  echo "sym - - -"
} | xargs -P 3 -L1 bash -c 'case "$0" in verdict) job_verdict "$1" "$2" "$3";; legacy) job_legacy;; sym) job_sym;; esac'
echo "[$(date +%H:%M)] CPU RESUME COMPLETE"
