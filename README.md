# Bayesian Causal Inference for NBA Injury Impact

A probabilistic modeling framework for estimating NBA player career trajectories and quantifying the causal effect of major injuries on performance using Bayesian latent variable models.

---

## Motivation

Injuries are among the most consequential events in a professional athlete's career, yet their causal effect on performance is difficult to isolate. Raw performance after an injury conflates the injury effect with natural aging, selection bias (injured players are more likely to exit the league), and regression to the mean. This project builds a generative model of the full career arc and uses counterfactual reasoning to attribute performance changes directly to injury.

**Key questions:**
- How much does an ACL tear reduce a player's performance, controlling for age and position?
- Does performance return to the pre-injury trajectory, or is there a permanent shift?
- Which metrics (scoring, playmaking, defense) are most sensitive to which injury types?

---

## Data

- **Performance:** NBA advanced statistics from 1997–2026 covering 17 per-season metrics (OBPM, DBPM, FG2, FG3, FT, AST, REB, STL, BLK, TOV, USG%, MPG, GP, and others)
- **Injuries:** Type-coded injury records: ACL, Achilles, Meniscus, Quad/Patellar Tendon, Hip, Foot Fracture, Back/Spine, Lower Body Fracture
- **Career censoring:** Entrance and exit ages for each player, enabling survival analysis

---

## Mathematical Framework

### 1. Latent Variable Model

Each player $n$ is represented by a low-dimensional latent factor $\mathbf{x}_n \in \mathbb{R}^r$. Player-level features are projected into a high-dimensional feature space using **Random Fourier Features** (Rahimi & Recht 2007):

$$\boldsymbol{\phi}(\mathbf{x}_n) = \frac{1}{\sqrt{m}} \left[ \cos(\mathbf{W}\mathbf{x}_n),\ \sin(\mathbf{W}\mathbf{x}_n) \right] \in \mathbb{R}^{2m}$$

where $\mathbf{W}$ is drawn from the spectral density of a Matérn kernel. The expected performance of player $n$ on metric $k$ at age $t$ is then:

$$\mu_{nkt} = \boldsymbol{\phi}(\mathbf{x}_n)^\top \boldsymbol{\beta}_k + \text{time\_basis}_k(t)$$

### 2. Career Trajectory: HSGP Concavity Constraint

Career arcs are **unimodal by construction** — they rise to a peak and then decline. This is enforced via a Hilbert Space Gaussian Process (HSGP) basis with a custom convexity construction, not imposed as a post-hoc constraint.

**HSGP eigenfunctions** on the domain $[-L, L]$:

$$\psi_j(t) = \frac{1}{\sqrt{L}} \sin\!\left(\lambda_j (t + L)\right), \quad \lambda_j = \frac{j\pi}{2L}, \quad j = 1,\ldots,M$$

**Basis matrix** $\boldsymbol{\Phi}(t)$ is the *second-order integral* of the eigenfunction outer product, so its second derivative satisfies:

$$\boldsymbol{\Phi}''(t) = \boldsymbol{\psi}(t) \otimes \boldsymbol{\psi}(t) \succeq 0$$

i.e., the Hessian is positive semi-definite — $\boldsymbol{\Phi}$ is globally convex.

**Peak parametrization:** Each player × metric pair has a peak age $t^*$ and peak value $c^*$. The mean trajectory is the Taylor expansion of a convex function around $t^*$, negated:

$$\mu(t) = c^* + \boldsymbol{\gamma}^\top \!\left[\boldsymbol{\Phi}(t^*) - \boldsymbol{\Phi}(t) + \boldsymbol{\Phi}'(t^*)(t - t^*)\right] \boldsymbol{\gamma}$$

where $\boldsymbol{\gamma} = \mathbf{W}\boldsymbol{\beta}$ projects latent player factors through spectral basis weights. This construction guarantees:

1. $\mu(t^*) = c^*$ — the trajectory passes exactly through the peak
2. $\dot{\mu}(t^*) = 0$ — zero derivative at the peak
3. $\ddot{\mu}(t) \leq 0$ — globally concave (since $\boldsymbol{\Phi}$ is convex, its negation is concave)

Smoothness of $\boldsymbol{\gamma}$ is further enforced via a Matérn spectral density prior on $\boldsymbol{\beta}$, penalizing high-frequency components.

### 3. AR(1) Temporal Process

Beyond the mean trajectory, year-to-year performance fluctuations are captured by an autoregressive process:

$$z_{n,t} = \rho \cdot z_{n,t-1} + \sigma \cdot \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$

The full linear predictor is $\eta_{nkt} = \mu_{nkt} + z_{n,t}$.

### 4. Causal Injury Effect Model

#### 4a. Treatment Definition

Each player-season is assigned an injury status. Let $d_{n,t} \in \{0, 1, \ldots, I\}$ denote the injury type for player $n$ at age $t$, where $d_{n,t} = 0$ means uninjured. Injury status is **forward-filled** from the first occurrence: once a player sustains injury type $i$, all subsequent seasons carry $d_{n,t} = i$. This encodes a permanent-shift assumption — the treatment is the cumulative biological insult, not the acute event alone.

#### 4b. Generative Model for Injury Effects

The injury effect on metric $k$ is a **low-rank hierarchical model** with player-level heterogeneity:

$$\delta_{k,n,t} = g_{k,d_{n,t}} + \underbrace{\sum_{p=1}^{P} \varphi_{d_{n,t},p}\, \eta_{k,p}}_{\text{shared injury structure}} + \underbrace{\sum_{r=1}^{R} \mathbf{x}_{n,r}\, \lambda_{k,d_{n,t},r}}_{\text{player heterogeneity}} + \sigma_k\, \varepsilon_{k,n,t}$$

The components are:

| Term | Dimension | Prior | Role |
|------|-----------|-------|------|
| $g_{k,i}$ | $K \times I$ | $\mathcal{N}(0,1)$ | Global intercept per metric × injury type |
| $\varphi_{i,p}$ | $I \times P$ | $\mathcal{N}(0,1)$ | Injury factor — shared latent structure across injury types |
| $\eta_{k,p}$ | $K \times P$ | $\mathcal{N}(0,1)$ | Injury loading — metric sensitivity to each injury factor |
| $\lambda_{k,i,r}$ | $K \times I \times R$ | $\mathcal{N}(0, 0.04)$ | Personalization — how player phenotype $\mathbf{x}_n$ moderates the effect |
| $\sigma_k$ | $K$ | $\text{HalfNormal}()$ | Residual variation per metric |

The tight prior on $\lambda$ ($\sigma = 0.2$) strongly regularizes player-heterogeneity relative to the shared injury factors, encoding a prior belief that injury effects are largely homogeneous across players with individual variation as second order.

The full linear predictor is:

$$\eta_{k,n,t} = \underbrace{\mu_{k,n,t}^{\text{base}}}_{\text{career trajectory}} + \underbrace{\mathbb{1}[d_{n,t} > 0] \cdot \delta_{k,n,t,d_{n,t}}}_{\text{injury effect}} + \underbrace{z_{n,t}}_{\text{AR(1)}}$$

#### 4c. Causal Assumptions

The causal interpretation of $\delta$ rests on four assumptions:

1. **Consistency (SUTVA):** The observed outcome under injury type $i$ equals the potential outcome $Y^{(i)}$. There are no multiple versions of treatment and no interference between players.

2. **Ignorability conditional on latent type:** Given the latent player factor $\mathbf{x}_n$ (which encodes pre-injury performance level and positional archetype), injury occurrence is assumed independent of the potential outcome trajectory. Formally: $Y^{(0)} \perp \mathbb{1}[d > 0] \mid \mathbf{x}_n$. This is the key identification assumption — it is partially addressed by including $\mathbf{x}_n$ in the injury effect via $\lambda$, and by the joint survival model.

3. **Separability:** Injury effects are additive on the linear predictor. There is no interaction between the injury effect and the shape of the career trajectory (the convex basis is fixed).

4. **Stable career counterfactual:** The uninjured trajectory $\mu_{k,n,t}^{\text{base}}$ correctly represents what performance would have been absent the injury. This is enforced structurally — the convex trajectory model is fit jointly with the injury component, so the base trajectory is estimated from all pre-injury data and the model's prior over the career arc.

#### 4d. Counterfactual Construction and ATT

The counterfactual trajectory for an injured player is obtained by setting all injury indicators to zero in the posterior predictive:

$$\hat{Y}_{k,n,t}^{\text{cf}} \sim \text{Family}_k\!\left(\text{link}^{-1}\!\left(\mu_{k,n,t}^{\text{base}} + z_{n,t}\right),\; \text{exposure}_{k,n,t}\right)$$

The **Average Treatment Effect on the Treated (ATT)** for injury type $i$ on metric $k$ is then:

$$\text{ATT}_{i,k} = \mathbb{E}\!\left[\, Y_{k,n,t}^{\text{obs}} - \hat{Y}_{k,n,t}^{\text{cf}} \;\Big|\; d_{n,t} = i,\ t > t_{\text{injury}}\right]$$

This is computed from posterior samples, with 95% highest density intervals (HDI) reported. Results are additionally stratified by whether the injury occurred **before or after the player's estimated peak age** $t^*$, since an injury that cuts short an ascending trajectory has a different causal signature than one occurring in a player's decline phase.

### 5. Survival / Selection Model

To correct for selection bias (injured or declining players exit the league), career duration is jointly modeled using a **Weibull hazard** for both entry and exit:

$$h(t) = \frac{\alpha}{\lambda} \left(\frac{t}{\lambda}\right)^{\alpha - 1} \exp(\mathbf{x}_n^\top \boldsymbol{\theta})$$

where $\boldsymbol{\theta}$ includes injury history and latent player quality. This ensures that performance estimates are not confounded by differential attrition.

### 6. Observation Model

Each metric is linked to the linear predictor via a family-appropriate likelihood:

| Metric Type | Distribution | Link |
|-------------|-------------|------|
| Box score rates (OBPM, DBPM) | Gaussian | Identity |
| Over-dispersed counts (AST, REB, STL, BLK) | Negative Binomial | Log |
| Shooting proportions (FG%, FT%) | Beta-Binomial | Logit |
| Usage rate, minutes pct. | Beta | Logit |
| Games played | Poisson | Log |

League-average trends are removed before fitting (de-trending array) to focus inference on player-relative performance.

---

## Inference

| Method | Details |
|--------|---------|
| **NUTS (primary)** | No-U-Turn Sampler via NumPyro — 4 chains × 2000 samples (1000 warmup) |
| **SVI** | AutoNormal / AutoLaplaceApproximation guide for fast approximate posteriors |
| **MAP** | AutoDelta guide for point estimates |

All models are implemented in **JAX** with `jit` compilation and `vmap` over players for GPU-accelerated inference.

---

## Running the Pipeline

All runs are driven by `config/model_config.yaml`. Three model families are available:

| Family | Model name prefix | Description |
|--------|------------------|-------------|
| Base | `nba_convex_max_tvlinearlvm` | Latent LVM + concave trajectory |
| AR | `nba_convex_max_tvlinearlvm_AR` | + AR(1) temporal process |
| Injury | `nba_convex_max_tvlinearlvm_injury` | + injury effects + survival |

### Container prerequisites

Jobs are dispatched via `docker exec` into three long-running containers. Start them once before invoking any Make target:

```bash
# GPU training (MAP + MCMC) — 2 GPUs
docker run --name mcmc -dt --rm \
  -v .:/home/joyvan/work -w /home/joyvan/work \
  nba-functional-analysis-mcmc

# CPU Python analysis (eval-only, coverage combine, export)
docker run --name mcmc-analysis -dt --rm \
  -v .:/home/joyvan/work -w /home/joyvan/work \
  nba-functional-analysis-mcmc-analysis

# R diagnostics
docker run --name r-new -dt --rm \
  -v .:/home/joyvan/work -w /home/joyvan/work \
  nba-functional-analysis-r
```

All three containers mount the project root at `/home/joyvan/work` and share the same filesystem, so model outputs written by the `mcmc` container are immediately visible to `r-new` for diagnostics.

Every Make target starts with a `check-containers` guard that fails fast if any container is not running:

```
ERROR: container 'r-new' is not running
make: *** [check-containers] Error 1
```

### Job routing

| Work | Container | Resources |
|------|-----------|-----------|
| MAP training (`run_job`, `run_pinned`) | `mcmc` | GPU 0 or 1 via `CUDA_VISIBLE_DEVICES` |
| MCMC training | `mcmc` | both GPUs |
| Eval-only / coverage combine / export | `mcmc-analysis` | CPU |
| `model_diagnostics.r`, `injury_causal.r` | `r-new` | CPU |

### Makefile shortcuts

Each recipe emits timestamped `START` / `DONE` lines so you can follow progress in a log file:

```
=== [14:03:22] START map/nba_convex_max_tvlinearlvm_AR_holdout_last_k (gpu 0) ===
=== [14:11:47] DONE  map/nba_convex_max_tvlinearlvm_AR_holdout_last_k ===
```

If a job fails, Make stops and prints the failing target; the last `START` line in the log shows what was running and when.

```bash
# MAP holdout runs per family (sequential, GPU 0)
make base        # base LVM, all four holdout schemes
make ar          # + AR(1)
make injury      # + injury effects + survival
make naive       # naive baseline

# All MAP holdout runs across two GPUs in parallel
make -j2 all

# Then combine holdout coverage tables (runs in mcmc-analysis container)
make coverage

# MCMC — individual families
make tvlvm_mcmc
make ar_mcmc
make injury_mcmc
make naive_mcmc

# Full MCMC for all families, then R diagnostics
make mcmc
make diagnostics

# Per-scheme MCMC (all four families, sequential)
make mcmc_holdout_last_k
make mcmc_holdout_first_k
make mcmc_random_interior
make mcmc_holdout_peak

# All 16 scheme × family MCMC jobs in sequence (pipe output to a log)
nohup make mcmc_all > make_mcmc.log 2>&1 & echo $! > make_mcmc.pid

# MCMC + R diagnostics chained per scheme
make coverage_mcmc

# R diagnostics only (after MCMC artifacts exist)
make diagnostics_holdout_last_k
make diagnostics_holdout_first_k
make diagnostics_random_interior
make diagnostics_holdout_peak

# Recompute coverage metrics without retraining (uses mcmc-analysis container)
make recompute_coverage
make recompute_coverage_naive
```

### Interactive per-scheme pipeline (`run_pipeline.sh`)

For a fully guided run over a single holdout scheme and a chosen subset of model families, use the interactive pipeline script. It prompts for scheme and models, then executes all phases in the correct order:

| Phase | Parallelism | Container |
|-------|-------------|-----------|
| 1. MAP | 2 GPU chains in parallel | `mcmc` |
| 2. MCMC | Sequential (avoids GPU memory contention) | `mcmc` |
| 3. Export | All selected models in parallel | `mcmc-analysis` |
| 4. R diagnostics | All selected models in parallel (+ `injury_causal.r` if injury selected) | `r-new` |
| 5. Combine | Single combine script | `mcmc-analysis` |

```bash
# Interactive — prompts for scheme then models
./run_pipeline.sh

# Non-interactive — pass scheme and model numbers as args (required for nohup)
# Model numbers: 1=tvlvm  2=ar  3=injury  4=naive
./run_pipeline.sh holdout_last_k "1 2 4"

# Run in background, keep a log
nohup ./run_pipeline.sh holdout_last_k "1 2 4" > pipeline.log 2>&1 & echo $! > pipeline.pid
```

The interactive prompts look like:

```
Select holdout scheme:
1) holdout_last_k
2) holdout_first_k
3) random_interior
4) holdout_peak
#? 1

Select models to include (space-separated numbers):
  1) tvlvm
  2) ar
  3) injury
  4) naive
? 1 2 4
```

MAP models are split evenly across the two GPUs (even-indexed selections → GPU 0, odd-indexed → GPU 1), each GPU running its chain sequentially to avoid memory contention.

To monitor a long background run:

```bash
# Watch live
tail -f make_mcmc.log

# Quick timeline — what started and finished, and any errors
grep -E "START|DONE|Error|Traceback" make_mcmc.log
```

### Output layout

```
model_output/
  nba_convex_max_tvlinearlvm_AR/
    map/                    ← MAP artifacts
      samples.pkl
      state.pkl
      latent_space.csv
      map_peaks.csv
      map_peak_vals.csv
      plots/
        latent_space/map/
        peaks/map/
        player_plots/map/
    mcmc/                   ← MCMC artifacts
      samples.pkl
      posterior_ar.csv
      posterior_mu_ar.csv
      posterior_peaks_ar.csv
      ...
      plots/
        peaks/mcmc/
        latent_space/mcmc/
        player_plots/mcmc/
        coverage/
    holdout_last_k/map/     ← Holdout variant artifacts
    holdout_first_k/map/
    ...
  model_plots/coverage/     ← Cross-model coverage tables (combine_holdout_tables.py)
```

---

## Repository Structure

```
├── model/
│   ├── models.py          # Bayesian model definitions (RFLVM hierarchy)
│   ├── hsgp.py            # HSGP basis construction (eigenfunctions, φ, φ', φ'')
│   └── data_utils.py      # Data preprocessing and family-specific transforms
├── main.py                # Inference entrypoint (data loading → model → posteriors)
├── causal_analysis.py     # ATT computation from posterior samples
├── plot_causal.py         # Injury effect visualizations
├── model_output/
│   ├── model_plots/causal/  # Per-player and aggregate causal plots
│   └── posterior_*.csv      # Posterior samples and summaries
└── data/                  # NBA stats and injury records
```

### Model Hierarchy

```
RFLVMBase
└── ConvexMaxTVLinearLVM          # Core: latent LVM + HSGP concave trajectory
    ├── ConvexMaxARTVLinearLVM    # + AR(1) temporal process
    └── ConvexMaxInjuryTVLinearLVM  # + injury effects + Weibull survival
```

---

## References

- Rahimi & Recht (2007). *Random Features for Large-Scale Kernel Machines.*
- Riutort-Mayol et al. (2023). *Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming.* Statistics and Computing.
- Phan et al. (2019). *Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro.* NeurIPS.
- Rubin (1974). *Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies.* Journal of Educational Psychology.
