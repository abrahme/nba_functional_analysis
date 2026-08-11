#!/usr/bin/env bash
set -euo pipefail

# ── Config (mirrors Makefile) ─────────────────────────────────────────────────
MODEL_CONFIG="config/model_config.yaml"
CTR_MCMC="mcmc"
CTR_ANALYSIS="mcmc-analysis"
CTR_R="r-new"
VALIDATION_YEAR="2021"
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"

# ── Memory-safe job runner ────────────────────────────────────────────────────
# Phases 3 (export) and 4 (R diagnostics) load full posteriors into host RAM.
# Fanning them out across all models — and especially running two schemes at once
# via the `all` wrapper — can OOM the machine. Default here is SERIAL: one process
# at a time. Set ANALYSIS_PARALLEL=1 to restore the old all-at-once fan-out.
run_job() {
    if [[ "${ANALYSIS_PARALLEL:-0}" == "1" ]]; then
        "$@" &
    else
        "$@"
    fi
}

# ── Model name / dir helpers ──────────────────────────────────────────────────
model_name() {
    local model=$1 scheme=$2
    case $model in
        tvlvm)          echo "nba_convex_max_tvlinearlvm_${scheme}" ;;
        ar)             echo "nba_convex_max_tvlinearlvm_AR_${scheme}" ;;
        injury)         echo "nba_convex_max_tvlinearlvm_injury_${scheme}" ;;
        naive)          echo "nba_naive_${scheme}" ;;
        tvlinearlvm)    echo "nba_tvlinearlvm_${scheme}" ;;
        tvlinearlvm_AR) echo "nba_tvlinearlvm_AR_${scheme}" ;;
        cosine)         echo "nba_convex_max_cosine_tvlinearlvm_${scheme}" ;;
        cosine_AR)      echo "nba_convex_max_cosine_tvlinearlvm_AR_${scheme}" ;;
        rflvm)          echo "nba_convex_max_tvrflvm_${scheme}" ;;
        rflvm_AR)       echo "nba_convex_max_tvrflvm_AR_${scheme}" ;;
        rflvm_split)    echo "nba_convex_max_tvrflvm_split_${scheme}" ;;
        rflvm_split_AR) echo "nba_convex_max_tvrflvm_split_AR_${scheme}" ;;
        injury_rff)     echo "nba_convex_max_tvrflvm_injury_${scheme}" ;;
        injury_cut)     echo "nba_convex_max_tvlinearlvm_injury_${scheme}" ;;
        injury_rff_cut) echo "nba_convex_max_tvrflvm_injury_${scheme}" ;;
    esac
}

model_dir() {
    local model=$1 scheme=$2
    case $model in
        tvlvm)          echo "model_output/nba_convex_max_tvlinearlvm/${scheme}/mcmc" ;;
        ar)             echo "model_output/nba_convex_max_tvlinearlvm_AR/${scheme}/mcmc" ;;
        injury)         echo "model_output/nba_convex_max_tvlinearlvm_injury/${scheme}/mcmc" ;;
        naive)          echo "model_output/nba_naive/${scheme}/mcmc" ;;
        tvlinearlvm)    echo "model_output/nba_tvlinearlvm/${scheme}/mcmc" ;;
        tvlinearlvm_AR) echo "model_output/nba_tvlinearlvm_AR/${scheme}/mcmc" ;;
        cosine)         echo "model_output/nba_convex_max_cosine_tvlinearlvm/${scheme}/mcmc" ;;
        cosine_AR)      echo "model_output/nba_convex_max_cosine_tvlinearlvm_AR/${scheme}/mcmc" ;;
        rflvm)          echo "model_output/nba_convex_max_tvrflvm/${scheme}/mcmc" ;;
        rflvm_AR)       echo "model_output/nba_convex_max_tvrflvm_AR/${scheme}/mcmc" ;;
        rflvm_split)    echo "model_output/nba_convex_max_tvrflvm_split/${scheme}/mcmc" ;;
        rflvm_split_AR) echo "model_output/nba_convex_max_tvrflvm_split_AR/${scheme}/mcmc" ;;
        injury_rff)     echo "model_output/nba_convex_max_tvrflvm_injury/${scheme}/mcmc" ;;
        injury_cut)     echo "model_output/nba_convex_max_tvlinearlvm_injury/${scheme}/cut_mcmc" ;;
        injury_rff_cut) echo "model_output/nba_convex_max_tvrflvm_injury/${scheme}/cut_mcmc" ;;
    esac
}

# Sampler regime per model key: the *_cut injury keys run the cut-posterior Gibbs
# (counterfactual latents on the injury-free likelihood, injury params on the full
# likelihood); everything else runs plain NUTS. Used by phases 2 (sampling) and 3
# (export reads the matching samples dir).
inference_method_for() {
    case $1 in
        injury_cut|injury_rff_cut) echo "cut_mcmc" ;;
        *)                         echo "mcmc" ;;
    esac
}

# ── Usage ─────────────────────────────────────────────────────────────────────
# Interactive:    ./run_pipeline.sh
# Non-interactive: ./run_pipeline.sh <scheme> "<model numbers>" [start_phase] [scripts] [end_phase]
#   scheme         : a single scheme, a quoted space-separated LIST of schemes (run sequentially), or 'all'
#                    (valid: holdout_last_k | holdout_first_k | random_interior | holdout_peak | stratified_next_k)
#   model numbers  : space-separated subset of 1=tvlvm 2=ar 3=injury 4=naive 5=tvlinearlvm 6=tvlinearlvm_AR 7=cosine 8=cosine_AR 9=rflvm 10=rflvm_AR
#                    11=injury_rff 12=injury_cut 13=injury_rff_cut
#                    (12/13 run the same models as 3/11 but sample via cut-posterior Gibbs and write to
#                     .../cut_mcmc; they share the MAP with 3/11, so don't select both for phase 1)
#   start_phase    : 0=PriorCheck 1=MAP 2=MCMC 3=Export 4=Diagnostics 5=Combine (default: 1)
#                    (phase 0 is opt-in: run prior-predictive checks only with `... 0 all 0`)
#   scripts        : phase-4 R scripts, selected like models — space-separated subset of
#                      1=coverage 2=diagnostics 3=team_window 4=latent 5=injury_causal
#                      (default: all). Legacy words all|latent|coverage still accepted.
#                      Phase 5 (combine) only runs when 1=coverage is selected.
#   end_phase      : last phase to run inclusive (default: 5)
#   GPU_BASE       : env var, 0 or 1 — which physical GPU the first model slot uses (default: 0)
#   e.g.           : ./run_pipeline.sh holdout_last_k "1 2 4"
#                  : ./run_pipeline.sh holdout_last_k "1 2" 3        # export+diag only
#                  : ./run_pipeline.sh holdout_last_k "2" 4 "2 4" 4      # diagnostics+latent, skip coverage
#                  : ./run_pipeline.sh holdout_last_k "1 2 3" 4 4 4      # latent_space.r only
#                  : ./run_pipeline.sh holdout_last_k "1 2 3" 4 1 5      # coverage.r (+combine) only
#                  : ./run_pipeline.sh holdout_last_k "5" 1 all 1   # MAP only
#                  : ./run_pipeline.sh all "5" 1 all 1              # MAP only, all schemes
#                  : ./run_pipeline.sh "holdout_last_k holdout_first_k random_interior holdout_peak" "1 2 9 10"
#                                                                   # 4 schemes (all but stratified), run sequentially
#                  : GPU_BASE=1 ./run_pipeline.sh holdout_first_k "5" 1 all 1 &  # MAP on GPU 1

VALID_SCHEMES=(holdout_last_k holdout_first_k random_interior holdout_peak stratified_next_k)

parse_model_choices() {
    local -a choices=($1)
    SELECTED_MODELS=()
    for c in "${choices[@]}"; do
        case $c in
            1) SELECTED_MODELS+=(tvlvm) ;;
            2) SELECTED_MODELS+=(ar) ;;
            3) SELECTED_MODELS+=(injury) ;;
            4) SELECTED_MODELS+=(naive) ;;
            5) SELECTED_MODELS+=(tvlinearlvm) ;;
            6) SELECTED_MODELS+=(tvlinearlvm_AR) ;;
            7) SELECTED_MODELS+=(cosine) ;;
            8) SELECTED_MODELS+=(cosine_AR) ;;
            9)  SELECTED_MODELS+=(rflvm) ;;
            10) SELECTED_MODELS+=(rflvm_AR) ;;
            14) SELECTED_MODELS+=(rflvm_split) ;;
            15) SELECTED_MODELS+=(rflvm_split_AR) ;;
            11) SELECTED_MODELS+=(injury_rff) ;;
            12) SELECTED_MODELS+=(injury_cut) ;;
            13) SELECTED_MODELS+=(injury_rff_cut) ;;
            *) echo "Warning: unrecognised option '$c' — skipping" ;;
        esac
    done
}

# Phase-4 R scripts, selected like models (space-separated numbers or names).
# Legacy single-word filters (all | latent | coverage) are still honoured so
# existing callers (resume_pipeline.sh, README examples) keep working.
parse_script_choices() {
    SELECTED_SCRIPTS=()
    case "$1" in
        all|"")   SELECTED_SCRIPTS=(coverage diagnostics team_window latent injury_causal); return ;;
        latent)   SELECTED_SCRIPTS=(latent);   return ;;
        coverage) SELECTED_SCRIPTS=(coverage); return ;;
    esac
    local -a choices=($1)
    for c in "${choices[@]}"; do
        case $c in
            1|coverage)              SELECTED_SCRIPTS+=(coverage) ;;
            2|diagnostics|diag)      SELECTED_SCRIPTS+=(diagnostics) ;;
            3|team_window|team)      SELECTED_SCRIPTS+=(team_window) ;;
            4|latent|latent_space)   SELECTED_SCRIPTS+=(latent) ;;
            5|injury_causal|causal)  SELECTED_SCRIPTS+=(injury_causal) ;;
            *) echo "Warning: unrecognised script option '$c' — skipping" ;;
        esac
    done
}

# Membership test against the selected phase-4 scripts.
has_script() { printf '%s\n' "${SELECTED_SCRIPTS[@]}" | grep -qx "$1"; }

# ── Scheme: arg or prompt ─────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    # A quoted, space-separated list of schemes in $1 (e.g. "holdout_last_k holdout_first_k")
    # runs each scheme sequentially through the full pipeline. A single scheme or the literal
    # "all" fall through to their existing handling below.
    read -ra _SCHEME_LIST <<< "$1"
    if [[ "$1" != "all" && ${#_SCHEME_LIST[@]} -gt 1 ]]; then
        SCRIPT="$(realpath "$0")"
        REST_ARGS=("${@:2}")  # everything after the scheme list
        for s in "${_SCHEME_LIST[@]}"; do
            printf '%s\n' "${VALID_SCHEMES[@]}" | grep -qx "$s" || {
                echo "ERROR: invalid scheme '$s'. Choose from: ${VALID_SCHEMES[*]}"; exit 1; }
        done
        echo "=== Running ${#_SCHEME_LIST[@]} schemes sequentially: ${_SCHEME_LIST[*]} ==="
        for s in "${_SCHEME_LIST[@]}"; do
            echo "=== [$(date '+%H:%M:%S')] scheme: $s ==="
            bash "$SCRIPT" "$s" "${REST_ARGS[@]}"
        done
        echo "=== All requested schemes complete ==="
        exit 0
    fi
    SCHEME=$1
    if [[ "$SCHEME" == "all" ]]; then
        # Re-invoke this script for each scheme, pairing them across GPUs
        SCRIPT="$(realpath "$0")"
        REST_ARGS=("${@:2}")  # everything after scheme
        echo "=== Running all 5 schemes (2 at a time across GPUs) ==="
        GPU_BASE=0 bash "$SCRIPT" holdout_last_k  "${REST_ARGS[@]}" &
        GPU_BASE=1 bash "$SCRIPT" holdout_first_k "${REST_ARGS[@]}" &
        wait
        GPU_BASE=0 bash "$SCRIPT" random_interior "${REST_ARGS[@]}" &
        GPU_BASE=1 bash "$SCRIPT" holdout_peak    "${REST_ARGS[@]}" &
        wait
        GPU_BASE=0 bash "$SCRIPT" stratified_next_k "${REST_ARGS[@]}" &
        wait
        echo "=== All schemes complete ==="
        exit 0
    fi
    if ! printf '%s\n' "${VALID_SCHEMES[@]}" | grep -qx "$SCHEME"; then
        echo "ERROR: invalid scheme '$SCHEME'. Choose from: all ${VALID_SCHEMES[*]}"
        exit 1
    fi
else
    echo "Select holdout scheme:"
    select SCHEME in "all" "${VALID_SCHEMES[@]}"; do
        [[ -n $SCHEME ]] && break
    done
    if [[ "$SCHEME" == "all" ]]; then
        SCRIPT="$(realpath "$0")"
        echo "=== Running all 5 schemes (2 at a time across GPUs) ==="
        GPU_BASE=0 bash "$SCRIPT" holdout_last_k  &
        GPU_BASE=1 bash "$SCRIPT" holdout_first_k &
        wait
        GPU_BASE=0 bash "$SCRIPT" random_interior &
        GPU_BASE=1 bash "$SCRIPT" holdout_peak    &
        wait
        GPU_BASE=0 bash "$SCRIPT" stratified_next_k &
        wait
        echo "=== All schemes complete ==="
        exit 0
    fi
fi

# ── Models: arg or prompt ─────────────────────────────────────────────────────
SELECTED_MODELS=()

if [[ $# -ge 2 ]]; then
    parse_model_choices "$2"
else
    echo ""
    echo "Select models to include (space-separated numbers):"
    echo "  1) tvlvm"
    echo "  2) ar"
    echo "  3) injury"
    echo "  4) naive"
    echo "  5) tvlinearlvm"
    echo "  6) tvlinearlvm_AR"
    echo "  7) cosine"
    echo "  8) cosine_AR"
    echo "  9) rflvm"
    echo "  10) rflvm_AR"
    echo "  11) injury_rff"
    echo "  12) injury_cut      (cut-posterior Gibbs on 3)"
    echo "  13) injury_rff_cut  (cut-posterior Gibbs on 11)"
    read -rp "? " -a choices
    parse_model_choices "${choices[*]}"
fi

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    echo "No models selected — exiting."
    exit 1
fi

START_PHASE=${3:-1}
END_PHASE=${5:-5}
GPU_BASE=${GPU_BASE:-0}  # which physical GPU the first model slot uses

# ── Phase-4 scripts: arg or prompt (selected like models) ─────────────────────
if [[ $# -ge 4 ]]; then
    parse_script_choices "$4"
elif [[ $# -eq 0 ]]; then
    echo ""
    echo "Select phase-4 scripts (space-separated numbers, blank = all):"
    echo "  1) coverage"
    echo "  2) diagnostics"
    echo "  3) team_window"
    echo "  4) latent"
    echo "  5) injury_causal"
    read -rp "? " -a schoices
    parse_script_choices "${schoices[*]:-all}"
else
    parse_script_choices all
fi

if [[ ${#SELECTED_SCRIPTS[@]} -eq 0 ]]; then
    echo "No valid scripts selected — exiting."
    exit 1
fi

echo ""
echo "Scheme      : $SCHEME"
echo "Models      : ${SELECTED_MODELS[*]}"
echo "Start phase : $START_PHASE  End phase: $END_PHASE"
echo "Scripts     : ${SELECTED_SCRIPTS[*]}"
[[ "$GPU_BASE" != "0" ]] && echo "GPU base    : $GPU_BASE"
echo ""

# ── Container health check ────────────────────────────────────────────────────
for ctr in "$CTR_MCMC" "$CTR_ANALYSIS" "$CTR_R"; do
    docker inspect --format='{{.State.Running}}' "$ctr" 2>/dev/null \
        | grep -q true \
        || { echo "ERROR: container '$ctr' is not running"; exit 1; }
done
echo "All containers running."
echo ""

# ── Phase 0: Prior predictive checks (opt-in: START_PHASE=0) ─────────────────
# Pre-fit prior tuning: Python ELPPD/checks/exports in mcmc-analysis (CPU), plots in r-new.
#   PRIOR_CHECK_ARGS        : shared prior_check.py flags applied to every selected model
#                             (e.g. knob overrides, --players all, --sweep ...)
#   PRIOR_CHECK_ARGS_<model> : PER-MODEL override that REPLACES PRIOR_CHECK_ARGS for that model.
#                             Prior names differ across models (e.g. convex uses lengthscale_deriv,
#                             gplvm uses lengthscale, naive uses sigma_ar), so each model can take its
#                             own knobs. <model> is the pipeline key: tvlvm, ar, injury, naive,
#                             tvlinearlvm, tvlinearlvm_AR.
#   PRIOR_PLOT_PLAYERS      : which players prior_check.r renders PNGs for ("all" or a comma list;
#                             default = the model_diagnostics posterior_plot_names set)
# Examples:
#   ./run_pipeline.sh stratified_next_k "1" 0 all 0                         # prior checks only
#   PRIOR_CHECK_ARGS="--players all" \
#     ./run_pipeline.sh stratified_next_k "1" 0 all 0                       # export all players' draws
#   PRIOR_CHECK_ARGS="--set sigma_X=0.3 --set sigma_W_proj=0.5" \
#     ./run_pipeline.sh stratified_next_k "1" 0 all 0                       # prior-tuned config
#   # all models in a row, each with its own knobs:
#   PRIOR_CHECK_ARGS="--num_prior_samples=200" \
#   PRIOR_CHECK_ARGS_tvlvm="--num_prior_samples=200 --set alpha@obpm=0.15 --set alpha@dbpm=0.15" \
#   PRIOR_CHECK_ARGS_tvlinearlvm="--num_prior_samples=200 --set alpha@obpm=0.15" \
#   PRIOR_CHECK_ARGS_naive="--num_prior_samples=200 --set sigma=InverseGamma:5,600" \
#     ./run_pipeline.sh stratified_next_k "1 4 5" 0 all 0
if [[ $START_PHASE -le 0 && $END_PHASE -ge 0 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 0: Prior predictive checks ==="
for m in "${SELECTED_MODELS[@]}"; do
    mname=$(model_name "$m" "$SCHEME")
    pcdir=$(model_dir "$m" "$SCHEME"); pcdir="${pcdir%/cut_mcmc}"; pcdir="${pcdir%/mcmc}"; pcdir="${pcdir}/prior"
    # Per-model knob override (PRIOR_CHECK_ARGS_<model>) wins; else the shared PRIOR_CHECK_ARGS.
    _pc_var="PRIOR_CHECK_ARGS_${m}"
    _pc_args="${!_pc_var-${PRIOR_CHECK_ARGS:-}}"
    echo "  prior_check (py, mcmc-analysis): $mname  [args: ${_pc_args:-<none>}]"
    $EXEC "$CTR_ANALYSIS" python prior_check.py \
        --model_name="$mname" --model_config="$MODEL_CONFIG" ${_pc_args}
    echo "  prior_check (R, r-new):          $pcdir"
    $EXEC "$CTR_R" Rscript data_analysis/prior_check.r "$pcdir" ${PRIOR_PLOT_PLAYERS:-}
done
echo "=== [$(date '+%H:%M:%S')] Prior checks done ==="
echo ""
fi  # end phase 0

# ── Phase 1: MAP — two sequential GPU chains running in parallel ──────────────
if [[ $START_PHASE -le 1 && $END_PHASE -ge 1 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 1: MAP ==="

gpu0_models=()
gpu1_models=()
for i in "${!SELECTED_MODELS[@]}"; do
    (( i % 2 == 0 )) && gpu0_models+=("${SELECTED_MODELS[$i]}") \
                     || gpu1_models+=("${SELECTED_MODELS[$i]}")
done

GPU_A=$GPU_BASE
GPU_B=$(( 1 - GPU_BASE ))

(
    for m in "${gpu0_models[@]}"; do
        mname=$(model_name "$m" "$SCHEME")
        echo "  [GPU $GPU_A] map: $mname"
        $EXEC -e "CUDA_VISIBLE_DEVICES=$GPU_A" "$CTR_MCMC" python main.py \
            --model_name="$mname" \
            --model_config="$MODEL_CONFIG" \
            --inference_method=map
    done
) &

(
    for m in "${gpu1_models[@]}"; do
        mname=$(model_name "$m" "$SCHEME")
        echo "  [GPU $GPU_B] map: $mname"
        $EXEC -e "CUDA_VISIBLE_DEVICES=$GPU_B" "$CTR_MCMC" python main.py \
            --model_name="$mname" \
            --model_config="$MODEL_CONFIG" \
            --inference_method=map
    done
) &

wait
echo "=== [$(date '+%H:%M:%S')] MAP done ==="
echo ""
fi  # end phase 1

# ── Phase 2: MCMC — sequential to avoid GPU memory contention ────────────────
if [[ $START_PHASE -le 2 && $END_PHASE -ge 2 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 2: MCMC ==="
for m in "${SELECTED_MODELS[@]}"; do
    mname=$(model_name "$m" "$SCHEME")
    imethod=$(inference_method_for "$m")
    echo "  ${imethod}: $mname"
    $EXEC "$CTR_MCMC" python main.py \
        --model_name="$mname" \
        --model_config="$MODEL_CONFIG" \
        --inference_method="$imethod"
done
echo "=== [$(date '+%H:%M:%S')] MCMC done ==="
echo ""
fi  # end phase 2

# ── Phase 3: Export — parallel in mcmc-analysis container ────────────────────
if [[ $START_PHASE -le 3 && $END_PHASE -ge 3 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 3: Export ==="
for m in "${SELECTED_MODELS[@]}"; do
    mname=$(model_name "$m" "$SCHEME")
    imethod=$(inference_method_for "$m")
    echo "  export: $mname (${imethod})"
    # EXPORT_FLAGS: optional extra model_export.py flags (e.g. EXPORT_FLAGS=--concave_only
    # to regenerate only the concave-loadings parquets without the full export).
    run_job $EXEC "$CTR_ANALYSIS" python model_export.py \
        --model_name="$mname" \
        --model_config="$MODEL_CONFIG" \
        --inference_method="$imethod" \
        ${EXPORT_FLAGS:-}
done
wait
echo "=== [$(date '+%H:%M:%S')] Export done ==="
echo ""
fi  # end phase 3

# ── Phase 4: R diagnostics — parallel in r-new container ─────────────────────
if [[ $START_PHASE -le 4 && $END_PHASE -ge 4 ]]; then
echo "=== [$(date '+%H:%M:%S')] PHASE 4: R diagnostics ==="
for m in "${SELECTED_MODELS[@]}"; do
    mdir=$(model_dir "$m" "$SCHEME")
    # coverage.r runs for all models (incl. naive)
    if has_script coverage; then
        echo "  coverage:    $mdir"
        run_job $EXEC "$CTR_R" Rscript data_analysis/coverage.r \
            "$mdir" "$VALIDATION_YEAR"
    fi
    if has_script diagnostics; then
        # model_diagnostics.r runs for all models
        echo "  diagnostics: $mdir"
        run_job $EXEC "$CTR_R" Rscript data_analysis/model_diagnostics.r \
            "$mdir" "$VALIDATION_YEAR"
    fi
    # team_window.r and latent_space.r require posterior_ar.parquet — not for naive
    if [[ "$m" != "naive" ]]; then
        if has_script team_window; then
            echo "  team_window:   $mdir"
            run_job $EXEC "$CTR_R" Rscript data_analysis/team_window.r "$mdir"
        fi
        if has_script latent; then
            echo "  latent_space:  $mdir"
            run_job $EXEC "$CTR_R" Rscript data_analysis/latent_space.r "$mdir"
        fi
    fi
    wait  # finish all scripts for this model before starting the next
done

# injury causal script runs alongside for every selected injury key (plain, RFF, and cut runs)
if has_script injury_causal; then
    for m in "${SELECTED_MODELS[@]}"; do
        case $m in
            injury|injury_rff|injury_cut|injury_rff_cut)
                echo "  injury_causal: $(model_dir "$m" "$SCHEME")"
                run_job $EXEC "$CTR_R" Rscript data_causal/injury_causal.r \
                    "$(model_dir "$m" "$SCHEME")" ;;
        esac
    done
fi

wait
echo "=== [$(date '+%H:%M:%S')] Diagnostics done ==="
echo ""
fi  # end phase 4

# ── Phase 5: Combine coverage tables ─────────────────────────────────────────
if [[ $START_PHASE -le 5 && $END_PHASE -ge 5 ]] && has_script coverage; then
echo "=== [$(date '+%H:%M:%S')] PHASE 5: Combine coverage tables ==="
$EXEC "$CTR_ANALYSIS" python model_output/model_plots/coverage/combine_holdout_tables.py
echo "=== [$(date '+%H:%M:%S')] Pipeline complete ==="
fi
