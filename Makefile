MODEL_CONFIG    := config/model_config.yaml
NUM_GPUS        := 2
VALIDATION_YEAR := 2021

# ── Container names (must be running before any target is invoked) ─────────────
CTR_MCMC     := mcmc           # GPU training (MAP + MCMC)
CTR_ANALYSIS := mcmc-analysis  # CPU Python  (eval-only, coverage combine, export)
CTR_R        := r-new          # R diagnostics

# Working directory inside each container (where the project is mounted).
CONTAINER_WORKDIR := /home/joyvan/work

# Base exec prefixes — no volume flags needed, containers are already running.
EXEC         := docker exec -w $(CONTAINER_WORKDIR)
D_GPU        := $(EXEC) $(CTR_MCMC)
D_CPU        := $(EXEC) $(CTR_ANALYSIS)
D_R          := $(EXEC) $(CTR_R)

# ── Container health guard ─────────────────────────────────────────────────────
# Fails fast with a clear message if any required container is not running.
# Depend on this from any top-level target that needs containers.
check-containers:
	@for ctr in $(CTR_MCMC) $(CTR_ANALYSIS) $(CTR_R); do \
	    docker inspect --format='{{.State.Running}}' $$ctr 2>/dev/null \
	        | grep -q true \
	        || { echo "ERROR: container '$$ctr' is not running"; exit 1; }; \
	done
	@echo "All containers running."

# ── Recipe helpers ─────────────────────────────────────────────────────────────
# Each macro emits a timestamped START line before the exec and a DONE line
# after.  If the exec exits non-zero, Make stops and prints the failing target;
# the START line in the log tells you exactly when and what was running.

# MAP on GPU: CUDA_VISIBLE_DEVICES selects which of the two GPUs this job uses.
# $(1) model_name   $(2) job index (0-11)
define run_job
	@echo "=== [$$(date '+%H:%M:%S')] START map/$(1) (gpu $(shell expr $(2) % $(NUM_GPUS))) ==="
	$(EXEC) -e CUDA_VISIBLE_DEVICES=$(shell expr $(2) % $(NUM_GPUS)) \
	    $(CTR_MCMC) python main.py \
	        --model_name=$(1) \
	        --model_config=$(MODEL_CONFIG) \
	        --inference_method=map
	@echo "=== [$$(date '+%H:%M:%S')] DONE  map/$(1) ==="
endef

# Hard-pinned MAP: $(1) = gpu_id, $(2) = model_name, $(3) = inference_method
define run_pinned
	@echo "=== [$$(date '+%H:%M:%S')] START $(3)/$(2) (gpu $(1)) ==="
	$(EXEC) -e CUDA_VISIBLE_DEVICES=$(1) \
	    $(CTR_MCMC) python main.py \
	        --model_name=$(2) \
	        --model_config=$(MODEL_CONFIG) \
	        --inference_method=$(3)
	@echo "=== [$$(date '+%H:%M:%S')] DONE  $(3)/$(2) ==="
endef

# Eval-only on CPU (no GPU needed): $(1) model_name
define recompute_job
	@echo "=== [$$(date '+%H:%M:%S')] START eval/$(1) ==="
	$(EXEC) -e CUDA_VISIBLE_DEVICES="" -e JAX_PLATFORMS=cpu \
	    $(CTR_ANALYSIS) python main.py \
	        --model_name=$(1) \
	        --model_config=$(MODEL_CONFIG) \
	        --inference_method=map \
	        --eval_only
	@echo "=== [$$(date '+%H:%M:%S')] DONE  eval/$(1) ==="
endef

# MCMC training (GPU container) then immediate R diagnostics (r-new container).
# $(1) model_name   $(2) mcmc model_dir
# MCMC training (GPU) then export (mcmc-analysis container).
# $(1) model_name
define run_mcmc_job
	@echo "=== [$$(date '+%H:%M:%S')] START mcmc/$(1) ==="
	$(D_GPU) python main.py \
	    --model_name=$(1) \
	    --model_config=$(MODEL_CONFIG) \
	    --inference_method=mcmc
	@echo "=== [$$(date '+%H:%M:%S')] START export/$(1) ==="
	$(D_CPU) python model_export.py \
	    --model_name=$(1) \
	    --model_config=$(MODEL_CONFIG)
	@echo "=== [$$(date '+%H:%M:%S')] DONE  mcmc+export/$(1) ==="
endef

# MCMC + export then immediate R diagnostics.
# $(1) model_name   $(2) mcmc model_dir
define run_coverage_mcmc_job
	$(call run_mcmc_job,$(1))
	@echo "=== [$$(date '+%H:%M:%S')] START diagnostics/$(2) ==="
	$(D_R) Rscript data_analysis/model_diagnostics.r $(2) $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r $(2) $(VALIDATION_YEAR)
	@echo "=== [$$(date '+%H:%M:%S')] DONE  diagnostics/$(2) ==="
endef

# Per-scheme R diagnostics: $(1) = holdout scheme subdirectory
define run_scheme_diagnostics
	@echo "=== [$$(date '+%H:%M:%S')] START diagnostics/$(1) ==="
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm_AR/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm_AR/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_naive/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_naive/$(1)/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_causal/injury_causal.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/$(1)/mcmc
	@echo "=== [$$(date '+%H:%M:%S')] DONE  diagnostics/$(1) ==="
endef

.PHONY: check-containers \
	all base ar injury naive coverage coverage_naive coverage_mcmc mcmc diagnostics \
	tvlvm_holdout_last_k tvlvm_holdout_first_k \
	tvlvm_random_interior tvlvm_holdout_peak \
	ar_holdout_last_k ar_holdout_first_k \
	ar_random_interior ar_holdout_peak \
	injury_holdout_last_k injury_holdout_first_k \
	injury_random_interior injury_holdout_peak \
	naive_holdout_last_k naive_holdout_first_k \
	naive_random_interior naive_holdout_peak \
	gpu0_map gpu1_map \
	recompute_coverage recompute_coverage_naive \
	mcmc_holdout_last_k mcmc_holdout_first_k mcmc_random_interior mcmc_holdout_peak \
	mcmc_all \
	diagnostics_holdout_last_k diagnostics_holdout_first_k \
	diagnostics_random_interior diagnostics_holdout_peak

# ── Individual MAP targets (mcmc container, GPU) ───────────────────────────────

tvlvm_holdout_last_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_holdout_last_k,0)

tvlvm_holdout_first_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_holdout_first_k,1)

tvlvm_random_interior: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_random_interior,2)

tvlvm_holdout_peak: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_holdout_peak,3)

ar_holdout_last_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_AR_holdout_last_k,4)

ar_holdout_first_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_AR_holdout_first_k,5)

ar_random_interior: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_AR_random_interior,6)

ar_holdout_peak: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_AR_holdout_peak,7)

injury_holdout_last_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_injury_holdout_last_k,8)

injury_holdout_first_k: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_injury_holdout_first_k,9)

injury_random_interior: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_injury_random_interior,10)

injury_holdout_peak: check-containers
	$(call run_job,nba_convex_max_tvlinearlvm_injury_holdout_peak,11)

naive_holdout_last_k: check-containers
	$(call run_job,nba_naive_holdout_last_k,0)

naive_holdout_first_k: check-containers
	$(call run_job,nba_naive_holdout_first_k,1)

naive_random_interior: check-containers
	$(call run_job,nba_naive_random_interior,2)

naive_holdout_peak: check-containers
	$(call run_job,nba_naive_holdout_peak,3)

# ── Individual MCMC targets (mcmc container, all GPUs) ────────────────────────

tvlvm_mcmc: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm)

ar_mcmc: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR)

injury_mcmc: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury)

naive_mcmc: check-containers
	$(call run_mcmc_job,nba_naive)

# Per-scheme MCMC: all 4 model families sequentially so they don't compete for
# GPU memory.  Usage: make mcmc_holdout_last_k

mcmc_holdout_last_k: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_holdout_last_k)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_last_k)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_last_k)
	$(call run_mcmc_job,nba_naive_holdout_last_k)

mcmc_holdout_first_k: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_holdout_first_k)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_first_k)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_first_k)
	$(call run_mcmc_job,nba_naive_holdout_first_k)

mcmc_random_interior: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_random_interior)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR_random_interior)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury_random_interior)
	$(call run_mcmc_job,nba_naive_random_interior)

mcmc_holdout_peak: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_holdout_peak)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_peak)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_peak)
	$(call run_mcmc_job,nba_naive_holdout_peak)

# All 16 scheme×model MCMC jobs sequentially.
# Usage: nohup make mcmc_all > make_mcmc.log 2>&1 & echo $! > make_mcmc.pid
mcmc_all: mcmc_holdout_last_k mcmc_holdout_first_k mcmc_random_interior mcmc_holdout_peak

# ── GPU-pinned sequential chains (for -j2 bulk MAP runs) ──────────────────────
# Each chain pins to one GPU via CUDA_VISIBLE_DEVICES and runs its jobs one
# after another.  Usage: make -j2 all

gpu0_map: check-containers
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_holdout_last_k,map)
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_random_interior,map)
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_AR_holdout_last_k,map)
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_AR_random_interior,map)
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_injury_holdout_last_k,map)
	$(call run_pinned,0,nba_convex_max_tvlinearlvm_injury_random_interior,map)
	$(call run_pinned,0,nba_naive_holdout_last_k,map)
	$(call run_pinned,0,nba_naive_random_interior,map)

gpu1_map: check-containers
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_holdout_first_k,map)
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_holdout_peak,map)
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_AR_holdout_first_k,map)
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_AR_holdout_peak,map)
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_injury_holdout_first_k,map)
	$(call run_pinned,1,nba_convex_max_tvlinearlvm_injury_holdout_peak,map)
	$(call run_pinned,1,nba_naive_holdout_first_k,map)
	$(call run_pinned,1,nba_naive_holdout_peak,map)

# ── Aggregate targets ──────────────────────────────────────────────────────────

# Family-level sequential runs.
base:   tvlvm_holdout_last_k tvlvm_holdout_first_k tvlvm_random_interior tvlvm_holdout_peak
ar:     ar_holdout_last_k ar_holdout_first_k ar_random_interior ar_holdout_peak
injury: injury_holdout_last_k injury_holdout_first_k injury_random_interior injury_holdout_peak
naive:  naive_holdout_last_k naive_holdout_first_k naive_random_interior naive_holdout_peak

# Bulk 2-GPU parallel MAP.  Usage: make -j2 all
all: gpu0_map gpu1_map

# Run all MAP holdout jobs then combine coverage tables (mcmc-analysis container).
# Usage: nohup make -j2 coverage > make.log 2>&1 & echo $! > make.pid
coverage: all
	$(D_CPU) python model_output/model_plots/coverage/combine_holdout_tables.py

# Re-combine after running only the naive family (other families already done).
coverage_naive: naive
	$(D_CPU) python model_output/model_plots/coverage/combine_holdout_tables.py

# ── Eval-only coverage recompute (mcmc-analysis container, no retraining) ─────
# Loads saved samples.pkl for each scheme and re-runs coverage metrics only.
# Usage: make recompute_coverage

recompute_coverage: check-containers
	$(call recompute_job,nba_convex_max_tvlinearlvm_holdout_last_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_holdout_first_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_random_interior)
	$(call recompute_job,nba_convex_max_tvlinearlvm_holdout_peak)
	$(call recompute_job,nba_convex_max_tvlinearlvm_AR_holdout_last_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_AR_holdout_first_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_AR_random_interior)
	$(call recompute_job,nba_convex_max_tvlinearlvm_AR_holdout_peak)
	$(call recompute_job,nba_convex_max_tvlinearlvm_injury_holdout_last_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_injury_holdout_first_k)
	$(call recompute_job,nba_convex_max_tvlinearlvm_injury_random_interior)
	$(call recompute_job,nba_convex_max_tvlinearlvm_injury_holdout_peak)
	$(call recompute_job,nba_naive_holdout_last_k)
	$(call recompute_job,nba_naive_holdout_first_k)
	$(call recompute_job,nba_naive_random_interior)
	$(call recompute_job,nba_naive_holdout_peak)
	$(D_CPU) python model_output/model_plots/coverage/combine_holdout_tables.py

recompute_coverage_naive: check-containers
	$(call recompute_job,nba_naive_holdout_last_k)
	$(call recompute_job,nba_naive_holdout_first_k)
	$(call recompute_job,nba_naive_random_interior)
	$(call recompute_job,nba_naive_holdout_peak)
	$(D_CPU) python model_output/model_plots/coverage/combine_holdout_tables.py

# ── MCMC + diagnostics pipeline ───────────────────────────────────────────────
# Each model trains in mcmc container then R diagnostics run immediately in
# r-new container.  Requires MAP samples.pkl files to exist (used as init).
# Usage: nohup make coverage_mcmc > make.log 2>&1 & echo $! > make.pid

coverage_mcmc: check-containers
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_holdout_last_k,\
		model_output/nba_convex_max_tvlinearlvm/holdout_last_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_holdout_first_k,\
		model_output/nba_convex_max_tvlinearlvm/holdout_first_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_random_interior,\
		model_output/nba_convex_max_tvlinearlvm/random_interior/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_holdout_peak,\
		model_output/nba_convex_max_tvlinearlvm/holdout_peak/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_last_k,\
		model_output/nba_convex_max_tvlinearlvm_AR/holdout_last_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_first_k,\
		model_output/nba_convex_max_tvlinearlvm_AR/holdout_first_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_AR_random_interior,\
		model_output/nba_convex_max_tvlinearlvm_AR/random_interior/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_AR_holdout_peak,\
		model_output/nba_convex_max_tvlinearlvm_AR/holdout_peak/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_last_k,\
		model_output/nba_convex_max_tvlinearlvm_injury/holdout_last_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_first_k,\
		model_output/nba_convex_max_tvlinearlvm_injury/holdout_first_k/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_injury_random_interior,\
		model_output/nba_convex_max_tvlinearlvm_injury/random_interior/mcmc)
	$(call run_coverage_mcmc_job,nba_convex_max_tvlinearlvm_injury_holdout_peak,\
		model_output/nba_convex_max_tvlinearlvm_injury/holdout_peak/mcmc)
	$(call run_coverage_mcmc_job,nba_naive_holdout_last_k,\
		model_output/nba_naive/holdout_last_k/mcmc)
	$(call run_coverage_mcmc_job,nba_naive_holdout_first_k,\
		model_output/nba_naive/holdout_first_k/mcmc)
	$(call run_coverage_mcmc_job,nba_naive_random_interior,\
		model_output/nba_naive/random_interior/mcmc)
	$(call run_coverage_mcmc_job,nba_naive_holdout_peak,\
		model_output/nba_naive/holdout_peak/mcmc)
	$(D_CPU) python model_output/model_plots/coverage/combine_holdout_tables.py

# Full MCMC run (mcmc container) then full R diagnostics (r-new container).
# Usage: make mcmc  /  make diagnostics
mcmc: check-containers
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_AR)
	$(call run_mcmc_job,nba_convex_max_tvlinearlvm_injury)
	$(call run_mcmc_job,nba_naive)

diagnostics: mcmc
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm_AR/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm_AR/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/model_diagnostics.r \
	    model_output/nba_naive/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_analysis/coverage.r \
	    model_output/nba_naive/mcmc $(VALIDATION_YEAR)
	$(D_R) Rscript data_causal/injury_causal.r \
	    model_output/nba_convex_max_tvlinearlvm_injury/mcmc

# ── Per-scheme diagnostics (r-new container) ───────────────────────────────────
# Usage: nohup make -j2 diagnostics_holdout_last_k > diag.log 2>&1 & echo $! > diag.pid

diagnostics_holdout_last_k: check-containers
	$(call run_scheme_diagnostics,holdout_last_k)

diagnostics_holdout_first_k: check-containers
	$(call run_scheme_diagnostics,holdout_first_k)

diagnostics_random_interior: check-containers
	$(call run_scheme_diagnostics,random_interior)

diagnostics_holdout_peak: check-containers
	$(call run_scheme_diagnostics,holdout_peak)
