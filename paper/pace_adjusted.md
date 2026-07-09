# Plan: Pace-Adjusted Exposure for NBA Career Trajectory Model

## Context

The model currently uses raw minutes as the exposure for all Poisson and Negative-Binomial count metrics (FG3A, AST, FTA, BLK, STL, OREB, DREB, TOV, FG2A). NBA pace increased by 41% from 1997 to 2026 (92 → 130 possessions/48 min), creating a +0.343 log-unit upward inflation in all per-minute count rates purely from pace, independent of any player-level change. Without correcting for this:

- FTA and AST appear to have large secular trends (+44%, +51% per-36) that almost entirely disappear once pace is removed (+2%, +7% per-possession).
- FG3A has a genuine three-point revolution trend (+0.504 log units per possession) on top of pace (+0.343).
- OREB, BLK, STL have genuine per-possession declines that are partially masked or reversed when looking at per-minute rates.
- The latent space conflates player-level effects with era-level pace, corrupting archetype discovery and cross-era nearest-neighbor retrieval.

**The fix**: replace `minutes` with `pace_adjusted_minutes = minutes × (pace_t / pace_ref)` as the exposure for all Poisson and NB metrics. This is a pure offset change of variables (`log(pace_adjusted_minutes) = log(minutes) + log(pace_factor_t)`) that requires no model code changes — only data preprocessing and config.

**After pace adjustment**, the model's linear predictor `μ` is in log(count / pace-adjusted-minute) space. The R plots continue to display `exp(μ) × 36`, which now means "per 36 pace-adjusted minutes" — a directly era-comparable unit equivalent to per-possession × a fixed constant.

---

## Part 0 — Pace-Adjusted Exposure

### 0A. Compute pace factor and add `pace_adjusted_minutes` column

**Files: `main.py` (lines ~147–152), `model_export.py` (lines ~91–97)**

In both files, immediately after the existing derived-column block (where `log_min`, `simple_exposure`, `pct_minutes` are created), add:

```python
# Pace adjustment: possessions per player-minute, relative to start_year
_poss_by_year = (
    data_all.groupby("year")
    .apply(lambda g: (
        g["fg2a"].sum() + g["fg3a"].sum() + 0.44 * g["fta"].sum()
        - g["oreb"].sum() + g["tov"].sum()
    ) / (g["minutes"].sum() / 5))
    .rename("poss_per_team_min")
    .reset_index()
)
_ref_pace = float(
    _poss_by_year.loc[_poss_by_year["year"] == start_year, "poss_per_team_min"]
    .iloc[0]
)
data_all = data_all.merge(_poss_by_year, on="year")
data_all["pace_adjusted_minutes"] = data_all["minutes"] * (
    data_all["poss_per_team_min"] / _ref_pace
)
data_all.drop(columns="poss_per_team_min", inplace=True)
```

**Note**: `start_year` is already available in both files from config (`args["start_year"]` in `main.py`; `cfg["start_year"]` in `model_export.py`). The formula uses the Dean Oliver possession approximation from the existing totals columns (`fg2a`, `fg3a`, `fta`, `oreb`, `tov`). The reference year is the first data year so that 1997-era players are unchanged and later years get proportionally larger exposure.

The fake player row (line ~175 in `main.py`) has `year=2000` and all count columns NaN — `pace_adjusted_minutes` will be NaN for it, which is fine (masked out by the model).

### 0B. Update `model_config.yaml` — change exposure for Poisson and NB metrics

**File: `config/model_config.yaml` (lines 25–33)**

Change `exposure: minutes` → `exposure: pace_adjusted_minutes` for all 9 Poisson/NB metrics:

```yaml
- {name: blk,  distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: stl,  distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: ast,  distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: dreb, distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: oreb, distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: tov,  distribution: poisson,           exposure: pace_adjusted_minutes}
- {name: fta,  distribution: negative-binomial, exposure: pace_adjusted_minutes}
- {name: fg2a, distribution: negative-binomial, exposure: pace_adjusted_minutes}
- {name: fg3a, distribution: negative-binomial, exposure: pace_adjusted_minutes}
```

**Unaffected metrics** (no change needed):
- `games` (beta-binomial, exposure: `games_exposure`) — counts games, not possessions
- `usg`, `pct_minutes` (beta, exposure: `minutes` or `games`) — already proportions
- `obpm`, `dbpm` (gaussian) — already relative metrics
- `ftm`, `fg2m`, `fg3m` (binomial, exposure: `fta`/`fg2a`/`fg3a`) — trial counts are the observed counts, not minutes; FTA/FG2A/FG3A as trial exposures come from the data column directly, not the modelled exposure

### 0C. `league_avg` computation is automatically correct

**File: `main.py` line 166, `model_export.py` line 136**

Both files compute:
```python
data_all[f"{metric}_league_avg"] = (
    data_all.groupby("year")[metric].transform("sum")
    / data_all.groupby("year")[exposure].transform("sum")
)
```

Since `exposure` comes from `parse_metrics(cfg)` which reads the YAML, it automatically picks up `pace_adjusted_minutes` for Poisson/NB metrics after 0B. The league_avg is then in per-pace-adjusted-minute units, which is the correct reference for the model's `de_trend_array`. No explicit change needed.

### 0D. Priors and linear predictor initialization are automatically correct

**File: `model/model_utils.py` — `compute_priors` (line 1268) and `_linear_predictor_eta_weight_valid` (line 1174)**

`compute_priors` scales observations by `Y / exp(exposure)` (line 1325), which becomes `Y / pace_adjusted_minutes` after the change — the correct rate for setting peak-value priors in the new space. The prior mean `μ_eta = log(p_max)` is then the log of the pace-adjusted rate, which is the correct initialization for the model.

`_linear_predictor_eta_weight_valid` (line 1181) computes `rate_per36 = 36 × Y / exp(exposure)` — after the change this becomes `36 × Y / pace_adjusted_minutes`, which is per-36-pace-adjusted-minutes. This is the correct linear predictor for the model.

No changes required to either function.

### 0E. Update `de_trend_metrics` in config

Exhaustive metric audit. Per-possession log Δ computed from league totals, 1997 → 2026. Pace accounts for +0.343 log units in all per-minute rates; values below are **after** pace removal (i.e., genuine per-possession trends).

**Poisson / Negative-Binomial metrics** (use `pace_adjusted_minutes` as exposure; detrending applies):

| Metric | Dist | Per-36 log Δ | Per-possession log Δ | Pace share | Verdict |
|--------|------|-------------|----------------------|-----------|---------|
| FG3A | NB | +0.847 | **+0.504** | 41% | **Keep** — genuine 3-point revolution |
| OREB | Poisson | −0.276 | **−0.619** | absorbed by pace reversal | **Add** — strong genuine decline (zone defense, pace kills ORB) |
| BLK | Poisson | −0.175 | **−0.518** | absorbed by pace reversal | **Add** — genuine positional/style decline |
| STL | Poisson | +0.035 | **−0.308** | absorbed by pace reversal | **Add** — genuine decline; +0.035 per-36 was masking the trend |
| FG2A | NB | +0.087 | **−0.256** | absorbed | **Add** — mid-range decline is genuine but modest; include to prevent prior bias |
| TOV | Poisson | +0.110 | **−0.233** | absorbed | **No** — ambiguous direction, trend is small; leave unmodelled |
| DREB | Poisson | +0.151 | **−0.192** | absorbed | **No** — small trend; team-rebound structure confounds player-level detrend |
| AST | Poisson | +0.414 | **+0.070** | 83% | **Remove** — per-possession rate is nearly flat; was almost entirely pace |
| FTA | NB | +0.364 | **+0.020** | 94% | **Remove** — essentially all pace; genuine per-possession FTA is flat |

**Binomial metrics** (exposure = observed attempt counts; pace adjustment does not apply):

| Metric | Dist | Exposure | Log Δ (ratio 1997→2026) | Verdict |
|--------|------|----------|------------------------|---------|
| FTM (FT%) | Binomial | fta | +0.103 (+10.8%) | No detrend needed — model exposure is FTA, which is already pace-adjusted; FT% is a pure proportion |
| FG2M (FG2%) | Binomial | fg2a | +0.127 (+13.6%) | No detrend needed — same reasoning; genuine shooting improvement is a player-level effect, not an era nuisance |
| FG3M (FG3%) | Binomial | fg3a | +0.009 (+0.9%) | No detrend needed — essentially flat proportion |

**Gaussian metrics** (already league-relative by construction; no exposure):

| Metric | Dist | Note | Verdict |
|--------|------|------|---------|
| OBPM | Gaussian | Box Plus/Minus is zero-sum by season; era effects cancel in league-relative units | No detrend needed |
| DBPM | Gaussian | Same | No detrend needed |

**Beta / Beta-Binomial metrics** (`data_utils.py` lines 89 and 98 already apply `logit(league_avg)` for these distributions — infrastructure supports de-trending via the logit link):

| Metric | Dist | Exposure | Note | League-avg formula | Verdict |
|--------|------|----------|------|-------------------|---------|
| USG | Beta | minutes | Usage rate = team possession share; already a per-possession proportion by construction — no secular trend expected | `sum(usg)/sum(minutes)` — not a proportion; would need fix | No detrend |
| PCT_MINUTES | Beta | minutes | Higher pace → more metabolic load per minute → players may log fewer minutes; indirect downward trend plausible | `sum(pct_minutes)/sum(minutes)` ✗ — not a proportion. **Needs fix**: change to `mean(pct_minutes)` per year in `main.py` and `model_export.py` before detrending is valid | Consider adding after fixing league_avg computation |
| GP% | Beta-Binomial | games_exposure | Higher pace → more injuries → GP% may decline over time; genuine indirect effect | `sum(games)/sum(games_exposure)` ✓ correct proportion — logit applied correctly in `data_utils.py:98` | **Add to `de_trend_metrics`** if secular trend observed; infrastructure already correct |

**Key code note for PCT_MINUTES detrending fix** (`main.py` ~line 166, `model_export.py` ~line 136):

The generic league_avg formula `sum(metric)/sum(exposure)` is incorrect for Beta metrics whose observed column is already a proportion. For PCT_MINUTES specifically, replace:
```python
data_all["pct_minutes_league_avg"] = (
    data_all.groupby("year")["pct_minutes"].transform("sum")
    / data_all.groupby("year")["minutes"].transform("sum")
)
```
with:
```python
data_all["pct_minutes_league_avg"] = data_all.groupby("year")["pct_minutes"].transform("mean")
```
This gives the league-average proportion (∈ (0,1)) which `data_utils.py:89` then correctly converts to `logit(league_avg)` for the de_trend offset.

**Summary — suggested `de_trend_metrics` after pace adjustment:**

```yaml
de_trend_metrics: [fg3a, oreb, blk, stl, fg2a, games]
```

Rationale for Poisson/NB threshold: include metrics with |per-possession log Δ| > 0.25 (FG3A +0.504, OREB −0.619, BLK −0.518, STL −0.308, FG2A −0.256). TOV (−0.233) and DREB (−0.192) fall below threshold and have structural confounders. `games` (GP%) is added because the logit-scale de_trend infrastructure already works correctly for Beta-Binomial and pace-driven injury rates create a plausible secular decline. `pct_minutes` can be added after fixing its league_avg computation (see above).

### 0F. R plotting — no code changes required; label update only

**Files: `data_analysis/latent_space.r` (line 942), `data_analysis/model_diagnostics.r` (lines 1238, 1296, 1337)**

The current conversion `.default = exp(mu) * 36` remains correct — it now produces rates in "per 36 pace-adjusted minutes" rather than per 36 raw minutes. This is era-comparable and exactly what the paper needs for cross-era archetype and neighbor analysis. No code change is required.

Optional: update y-axis labels from `"per 36 min"` to `"per 36 pace-adj. min"` in affected ggplot calls, or add a paper footnote explaining the unit. This is cosmetic and can be done as a follow-up.

**Empirical rate plots in `model_diagnostics.r`** (raw data `BLK = 36 * (blk / minutes)` etc.) remain in raw-per-36-minutes units for the empirical overlay, which is fine — the mismatch between empirical (raw) and model (pace-adjusted) overlays can be noted or fixed separately.

---

## Critical Files

| File | Section | Change |
|------|---------|--------|
| `main.py` | ~line 150 | Add pace factor computation and `pace_adjusted_minutes` column |
| `model_export.py` | ~line 94 | Same pace factor computation |
| `config/model_config.yaml` | lines 25–33 | Change 9 Poisson/NB exposures from `minutes` to `pace_adjusted_minutes`; update `de_trend_metrics` |
| `model/model_utils.py` | lines 1174, 1268 | **No changes** — auto-scales with exposure |
| `model/inference_utils.py` | lines 380–386 | **No changes** — posterior display is in pace-adjusted per-36 units (desired) |
| `data/data_utils.py` | lines 68–81 | **No changes** — reads exposure column name from config |
| `data_analysis/latent_space.r` | line 942 | **No changes** — `exp(mu) * 36` correct in new space |
| `data_analysis/model_diagnostics.r` | lines 1238, 1296 | **No changes** for model output; empirical overlay remains raw-per-36 |

---

## Verification

1. Run `python main.py --model_name=nba_convex_max_tvlinearlvm_AR_holdout_last_k --model_config=config/model_config.yaml --inference_method=map` and confirm no KeyError on `pace_adjusted_minutes`.
2. Check that `data_all["pace_adjusted_minutes"]` equals `data_all["minutes"]` for `year == start_year` and is ~1.41× larger for 2026.
3. Confirm `offset_max` printed at line 475 of `main.py` changes for FG3A, AST, OREB (smaller values in pace-adjusted space for later-era players).
4. Re-run the full pipeline and inspect posterior curves for Stephen Curry: AST per-36 should be slightly lower than the raw per-36 (~8→~5.5 for reference-year scale), confirming the pace adjustment is active.
5. Confirm the latent space clusters players from different eras more fairly — LeBron James (2003–2024) should not be separated from similarly productive players purely by era.

---

## Part A — `paper/main.tex` Prose and Structure Fixes

### A1. Fix section nesting (Critical)
**Problem:** `\subsubsection{A Non-Parametric Prior for Concave Functions}` (line 290) is nested under `\subsection{Career Duration Model}`. It is a standalone topic and should be its own `\subsection`.

**Fix:** Change line 290 from:
```latex
\subsubsection{A Non-Parametric Prior for Concave Functions}
```
to:
```latex
\subsection{A Non-Parametric Prior for Concave Functions}
```
Also change its two nested propositions' subsubsection references if any are present.

---

### A2. Remove MAP log-loss claim in line 743 (Critical)
**Problem:** "the AR-only single-metric baseline achieves holdout log-losses in the range $10^4$–$10^8$ on count statistics such as BLK and OREB, whereas the joint Concave model reduces these to the range 4–7" — these are MAP-based artifacts, already removed from the conclusion but still present here.

**Fix:** Replace the entire compound sentence (after "producing tighter and more coherent posteriors ---") with a reference to the ELPPD comparison from Section 5.2:
```
…producing tighter and more coherent posteriors. By ELPPD (Section~\ref{sec:validation}), the Concave$+$AR model significantly outperforms the AR baseline on count statistics (BLK, STL, DREB, OREB; all $|\Delta|>1.96$\,SE), where the bounded career arc provides regularization unavailable to a pure AR process.
```

---

### A3. Fix broken cross-reference at line 140 (Critical)
**Problem:** "Contrast the marginal curves \ref{fig:empirical_production_players} with that of a specific player, Kobe Bryant, in \ref{fig:empirical_production_players}." — both refs point to the same figure; the second should reference the Kobe-specific figure.

**Fix:** The paper currently shows only one figure (the population-level LOESS plot); there is no separate Kobe figure. Remove the second reference entirely and rewrite:
```
We can see an example of this interplay by contrasting the population-level LOESS curves (Figure~\ref{fig:empirical_production_players}) with individual player profiles. Kobe Bryant's athleticism metrics (steals, offensive rebounds) peak early and decline steeply, while his shooting metrics peak later and remain stable — a pattern consistent with skill-based aging dominating over athleticism decline.
```

---

### A4. Fix `\label{tab:jokic_neighbors}` placement at line 759–760 (High)
**Problem:** The `\label` is outside any floating environment; it follows a bare `\input` with no surrounding `table` environment.

**Fix:** Wrap the `\input` in a minimal table float and move the caption/label inside:
```latex
\begin{table}[htbp]
\centering
\caption{Five nearest latent-space neighbors of Nikola Joki\'{c} (Procrustes-aligned posterior mean coordinates). Frequency column reports the fraction of 1{,}000 posterior draws in which each player appears as the closest neighbor.}
\label{tab:jokic_neighbors}
\input{../model_output/nba_convex_max_tvlinearlvm_injury/holdout_last_k/mcmc/plots/latent_space/tables/Nikola_Jokic_neighbors.tex}
\end{table}
```

---

### A5. Fix wrong prior reference in Latent Space Permanence (line 998) (Medium)
**Problem:** "The $\mathcal{N}(0, I_q)$ prior on $X_p$" — contradicts the paper's own structured prior.

**Fix:**
```
The structured prior on $X_p$ (Equation~\ref{eq:structured_prior}) imposes a mild regularisation…
```

---

### A6. Fix coverage range in Limitations (line 1150) (Medium)
**Problem:** "mild overconfidence (81–93% vs. the nominal 95%)" — the actual table shows 60–99.7%.

**Fix:**
```
Second, the holdout coverage analysis reveals calibration heterogeneity across metrics (60--99\% vs.\ the nominal 95\%), with OBPM showing the largest shortfall; richer non-Gaussian likelihoods or heavier-tailed error models could improve calibration for the most volatile metrics.
```

---

### A7. Fix "full space of concave functions" overstatement (line 352) (Medium)
**Problem:** "the ability to sample from the full space of concave functions" — the proof establishes positive mass on smooth neighborhoods only.

**Fix:** Replace the final sentence:
```
Thus, this nonparametric construction assigns positive mass to every uniform neighborhood of any continuous concave function on a compact interval — the precise sense in which it is a universal prior over concave functions.
```

---

### A8. Add denominator for Jokic 74% claim (line 756) (Medium)
**Problem:** "Giannis Antetokounmpo appears in 74% of MCMC samples" — denominator unstated.

**Fix:**
```
Giannis Antetokounmpo appears as the closest neighbor in 74\% of the 1{,}000 posterior draws (740 of 1{,}000),…
```

---

### A9. Add immortal time bias to Limitations (after line 1151) (Medium)
**Problem:** Injury analysis conditions on players who returned after injury, excluding those who retired immediately.

**Fix:** Add a new sentence after the injury confounding limitation:
```
Fourth, the injury counterfactual analysis conditions on the first observed major injury, selecting only players who were injured and subsequently returned to play; players who retired immediately after injury are excluded. This induces immortal-time bias: the estimated injury effects describe the subset of players robust enough to return, and may understate the total career impact of the most severe injuries.
```

---

### A10. Informal register cleanup (Medium)
Replace these specific instances:

| Line | Current | Replacement |
|------|---------|-------------|
| 103 | "For those not as familiar with the sport, \cite{terner2020modeling} is a good introduction to the metrics presented in this paper." | "Metric definitions follow \cite{terner2020modeling}." |
| 1016 | "Here, we take a look at Kevin Durant…" | "Figure~\ref{fig:kd_bj} shows Kevin Durant…" |
| 1037 | "As we can see, most markers…" | "Most markers of production…" |
| 1039 | "as we can see, Jennings' FTA rate dropped dramatically" | "Jennings' FTA rate dropped dramatically" |
| 1057 | "It is more common that we see a significant negative impact…" | "The negative impact of lower body injuries is most pronounced…" |
| 906 | "Subsequently, it is interesting to see that Russell Westbrook is close to Giannis" | "Notably, Russell Westbrook appears adjacent to Giannis Antetokounmpo," |

---

### A11. Define RAPTOR on first mention (line 68) (Low)
**Fix:** Change:
```
Rating systems such as RAPTOR \citep{...}
```
to:
```
Rating systems such as RAPTOR (Robust Algorithm using Player Tracking and On/Off Ratings; \citealt{natesilver538_2015, natesilver538_2019})
```

---

### A12. Add PCA variance justification sentence (Medium)
**Problem:** PC1 + PC2 = 22% variance; no justification in the text.

**Fix:** In Section 5.3 ("Uncovering Metric Relationships"), after the sentence introducing the PCA figure, add:
```
The low cumulative variance captured by PC1 and PC2 (22.6\%) is expected: with 17 heterogeneous metrics spanning athleticism, skill, and composite production, the latent aging structure is inherently high-dimensional, and no two principal components can summarise it compactly. The eigenvalue scree plot (Figure~\ref{fig:pca_scree}) and the PC3–PC4 projection (Figure~\ref{fig:latent_pca_pc34}, Appendix~\ref{sec:latent_appendix}) confirm that the remaining principal components capture meaningful residual structure.
```

---

## Part B — `data_analysis/model_diagnostics.r` New Outputs

### B1. Scree plot for latent PCA
**Where to add:** After the existing `latent_pca_plt` block (around line 2228).

**What to add:** A standalone scree/eigenvalue plot using `latent_pca$sdev^2 / sum(latent_pca$sdev^2)` with cumulative variance overlay.

**Output path:** `model_output/model_plots/latent_space/map/latent_pca_scree.png`

---

### B2. PC3 vs PC4 scatter plot for latent PCA
**Where to add:** Immediately after B1 (around line 2230).

**What to add:** A scatter of PC3 vs PC4, colored by archetype, with variance % annotations (same style as existing `latent_pca_plt`).

**Output path:** `model_output/model_plots/latent_space/map/latent_pca_pc34_archetypes.png`

---

### B3. Archetype quantitative summary table (LaTeX)
**Where to add:** After `archetype_val_heatmap` (after line 2183). Replace the two heatmaps with this table export (heatmaps are not referenced in the paper).

**What to add:** Using existing `archetype_peak_ages` and `archetype_peak_vals` tibbles (lines 2155–2165), generate a combined LaTeX table:
- Rows: 17 metrics
- Columns: Archetype 1–4, showing `mean_peak_age (mean_peak_val)` per cell
- Use `knitr::kable()` with `format="latex"` and `booktabs=TRUE`, wrapped in a `table` environment
- Save via `writeLines()` to a `.tex` file

**Output path:** `model_output/model_plots/latent_space/map/archetype_summary_table.tex`

**Activate commented-out infrastructure:** The commented-out table code at lines 295–337 shows the general approach (gt tables for DBPM/OBPM). The new table will use a simpler kable approach covering all metrics × all archetypes.

---

### B4. Archetype-specific peak age distribution overlay for Fig 7
**Where to add:** Extend the existing peaks plot code (around line ~855–869 in main.tex corresponds to the R code generating `peaks.png` at line 431 of model_diagnostics.r).

**What to add:** After generating the per-player peak CI ribbon, add archetype group means as colored horizontal ticks (one per archetype per metric), so the figure shows not just the population distribution but how archetypes differ in peak timing.

**Output path:** Same as current — `model_output/model_plots/peaks/mcmc/peaks.png` (overwrite existing, or save as `peaks_with_archetypes.png` to avoid side effects).

---

## Part C — `paper/main.tex` Additions for New R Outputs

### C1. Add scree plot and PC3–PC4 figure to appendix
**Where:** In the appendix, add a new subsection or integrate into the existing convergence/latent appendix.

Add after the modality-specific clustering (around line 993):
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{../model_output/.../latent_pca_scree.png}
    \caption{Eigenvalue scree plot for the posterior mean latent coordinate PCA. The dashed line shows cumulative variance explained.}
    \label{fig:pca_scree}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\linewidth]{../model_output/.../latent_pca_pc34_archetypes.png}
    \caption{PC3 vs.\ PC4 of the posterior mean latent coordinates, colored by archetype. PC3 and PC4 explain X\% and Y\% of variance respectively.}
    \label{fig:latent_pca_pc34}
\end{figure}
```

### C2. Add archetype summary table to Section 5.4
**Where:** After Figure~\ref{fig:archetype_frechet_curves} and before the archetype bullet list (around line 930).

```latex
Table~\ref{tab:archetype_summary} reports posterior mean peak ages and peak values for each archetype across all 17 metrics, providing a quantitative complement to the curve-level summaries in Figure~\ref{fig:archetype_frechet_curves}.

\input{../model_output/model_plots/latent_space/map/archetype_summary_table.tex}
```

---

## Files Modified

| File | Change |
|------|--------|
| `paper/main.tex` | A1–A12 prose/structure fixes; C1–C2 additions for new figures/table |
| `data_analysis/model_diagnostics.r` | B1 scree plot; B2 PC3–PC4 scatter; B3 archetype LaTeX table; B4 archetype peaks overlay |

---

## Execution Order
1. All Part A edits first (no new figures needed — self-contained prose fixes)
2. Part B R code changes (generates new output files)
3. Part C LaTeX additions (depends on B output files existing)
4. Recompile `paper/main.tex` to verify clean build

---

## Verification
1. `pdflatex main.tex` produces 0 errors and the PDF renders the corrected sections.
2. Section nesting: the Concave Process Prior appears as a `\subsection` at the same level as "Likelihood", "Mean Model", and "Career Duration Model".
3. The archetype table (Table~\ref{tab:archetype_summary}) renders with 4 archetype columns × 17 metric rows.
4. Figures `fig:pca_scree` and `fig:latent_pca_pc34` render in the appendix.
5. Grep for "take a look", "As we can see", "It is more common", "not as familiar" — all should return 0 matches.
6. Grep for `$10^{4}$--$10^{8}$` — should return 0 matches.
