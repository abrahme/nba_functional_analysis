# Paper TODO (JASA Review — Critical Analysis)

---

## Blocking — Must Fix Before Submission

- [x] **16 vs 17 metrics discrepancy** — fixed to 17 throughout (abstract line 70, likelihood section line 170).
- [x] **MCMC sample size / convergence summary** — corrected chain description (2000 post-warmup draws, thinned ×8 to 250 per chain, 1000 total; injury: 50 per chain); added worst-case $\hat{R}$ fractions to main text inference section (7.5–40.8%; AST outlier structural); fixed appendix diagnostics to cite correct metrics and numbers.
- [x] **Notational conflict: $K$ overloaded** — renamed HSGP basis count to $J$ throughout (sum limits, proposition, $\alpha$-vector); covariance kernel renamed $\mathcal{K}$ in theorem and HSGP description.
- [x] **Concavity not preserved through count-metric link functions** — resolved by monotonicity argument: prior is on latent mean; all links are strictly monotone, so $\arg\max_t g^{-1}(\mu_{ptk}) = \arg\max_t \mu_{ptk}$; peak age and peak value ordering exactly preserved. Added to both main body (Section 3.3) and appendix proof section.
- [x] **Indexing inconsistency: $n$ vs $p$** — data section now uses $\mathcal{Y}_{ptk}$, $P$ for player count, "player $p$" throughout.
- [ ] **Injury factor dimension $d$ never specified** — the injury offset model uses $w_m \in \mathbb{R}^d$ and $\tau_k^* \in \mathbb{R}^d$, but $d$ does not appear in the main text or prior table. Specify and justify.
- [x] **Draft position missingness: undrafted players** — added to structured prior section: undrafted players assigned sentinel rank $P+1$, so $-\log d_p$ defaults to minimum after standardization; survival entry time set to first observed NBA season (consistent with `data_utils.py` line 187).
- [x] **Two-step sampling procedure ambiguous** — replaced with explicit three-stage cascade: Stage 1 (GP%), Stage 2 (MPG conditioned on Stage 1), Stage 3 (all metrics using composite minutes exposure); clarified this is a deliberate hierarchical model structure, not an approximation.

---

## Major — Required for Strong Submission

- [ ] **PCA variance explained is low and underreported** — PC1 explains 11.2% and PC2 10.6% of peak-age variance. A scatterplot in 22%-variance space is weak evidence of structure. Justify why this is expected (high-dimensional metric space), or replace with higher-variance representation. Report cumulative variance in the main text.
- [x] **Figures lack observed data overlay** — confirmed: observed data points are already overlaid on all trajectory figures.
- [ ] **Multiple comparisons uncorrected** — ELPPD significance tests ($|\Delta| > 1.96\,\mathrm{SE}$) applied across 17 metrics × 3 model pairs × 4 schemes ≈ 204 comparisons. Apply Benjamini-Hochberg correction or explicitly discuss the false discovery rate.
- [ ] **Immortal time bias in injury analysis** — selecting "first injury occurrence" conditions on players surviving to be injured and having post-injury data. Players who retired immediately after injury are excluded. This is immortal-time bias; must be added to limitations.
- [ ] **Archetype characterization is subjective** — archetypes described by naming a few players per cluster. Replace with a table of archetype-specific posterior means across key metrics.
- [ ] **No posterior predictive checks** — paper evaluates holdout accuracy but never shows whether the model captures in-sample patterns: marginal metric distributions, career length vs. Weibull posterior, player-level mean-variance relationships.
- [ ] **Prior justification absent throughout** — HalfNormal(1) for $\sigma_W$, $\mathcal{N}(\log 11.5, 0.5)$ for $\mu_\text{scale}$, Inverse-Gamma values in appendix stated without motivation. Add one-sentence justifications or a sensitivity analysis for influential priors.
- [ ] **Proposition 1 proof gap** — proof shows the pushforward has full support but does not prove individual sample paths are concave. The claim "almost surely concave" requires a separate argument.

---

## Moderate — Should Address

- [ ] **ESS / convergence reported only in appendix** — $\hat{R}$ plots appear in the appendix but no scalar summary (worst-case $\hat{R}$, minimum ESS) appears in the main text. Add one sentence to the inference section.
- [ ] **Survival model choice unjustified** — Weibull chosen without comparison to alternatives. Add a sentence justifying why Weibull is appropriate for NBA career length.
- [x] **"Roughly concave" in abstract** — abstract no longer uses this phrase; concavity is described as "enforced through a novel nonparametric prior."
- [ ] **Binomial likelihood parameterization unclear** — line 172 writes $\text{Binomial}(n^k_{ptk}, \operatorname{logit}^{-1}(\mu_{ptk}))$. Rewrite as $p_{ptk} = \sigma(\mu_{ptk})$, $y_{ptk} \sim \text{Binomial}(n^k_{ptk}, p_{ptk})$.
- [ ] **Negative-Binomial parameterization ambiguous** — does not specify mean/overdispersion vs. shape/rate parameterization.
- [ ] **NB overdispersion parameter $\mathcal{R}$ undefined in main text** — defined only in appendix; move definition to main text.
- [ ] **Jokic neighbor frequency needs denominator** — "Giannis appears in 74% of MCMC samples" with only 200 total samples = ~148 draws. State denominator explicitly.
- [ ] **Claim all concave functions reachable is unproven** — line 350 says the prior can "sample from the full space of concave functions." Proof shows positive measure on smooth neighborhoods only. Restate precisely.
- [ ] **Identifiability and archetype stability** — archetype cluster assignments depend on the specific Procrustes rotation. Show empirically that archetypes are stable across different reference frames or random restarts.
- [ ] **Computational cost not discussed** — no mention of GPU hours, memory requirements, or scalability.
- [x] **Code references in prose removed** — `\texttt{make\_psi\_gamma}`, `data/data\_utils.py`, `create\_validation\_mask`, `\texttt{nanargmax}`, `\texttt{nba\_convex\_max\_tvlinearlvm}`, `\texttt{obpm}/\texttt{dreb}/\texttt{pct\_minutes}` all removed from paper prose.

---

## Results Narrative

- [ ] **Latent space section needs a synthesis** — the current tour of dendrograms and PCAs never answers "what do we learn about basketball?" Add a paragraph synthesizing what the four archetypes reveal about career development patterns.
- [ ] **PC3/PC4 latent space plot + eigenvalue scree plot** — advisor request; also addresses the low-variance PCA concern.
- [ ] **New 2×4 panel figure** — Curry (top) / Jokic (bottom), metrics FTA / OBPM / MPG / FG2A, with posterior age densities on each panel.
- [ ] **Fig 7 (`fig:peaks`): add archetype-specific curves** — show how peak age and value differ across the four archetypes.
- [ ] **Confirm whether coverage deteriorates for base model (no AR)** — advisor question; numbers already in the coverage comparison table.

---

## Validation / Analysis

- [ ] **Compute ELPPD from MCMC samples (BDA method)** — verify current computation matches $\sum_i \log \frac{1}{S}\sum_s p(y_i|\theta^s)$. Clarify in text.
- [ ] **Remove bias column from holdout RMSE tables** — edit `combine_holdout_tables.py`; bias reported separately in interval tables.
- [ ] **Compute log-loss excluding low-minutes players** — filter consistently with the coverage filter pipeline.

---

## Minor / Cleanup

- [ ] **Bare `\url{}` → `\citep`** — `basketball-reference.com` and `fivethirtyeight.com` need proper citations.
- [ ] **Remove comment macros** — `\cmnt`, `\franks`, `\Terner` must be removed before submission.
- [ ] **Audit bare `\ref{}`** — should be `Figure~\ref{}`, `Table~\ref{}`, `Section~\ref{}` throughout.
- [ ] **Reconcile data year range** — "1997–2021" vs "1996–2021" used inconsistently.
- [ ] **Shorten captions** — many are 3–4 sentences mixing description with interpretation; move interpretation to text.
- [ ] **Remove duplicate `\usepackage` declarations** — `\usepackage[utf8]{inputenc}` and `\usepackage{graphicx}` appear twice in preamble.
- [ ] **Informal register** — "intuitively" (line 105), "interesting," "striking," "we take a look at," "It is more common that we see" should be replaced throughout.
- [ ] **Group vs. Archetype numbering** — metric Groups 1/2/3 (Section 5.3) and latent Archetypes 1–4 (Section 5.4) both numbered from 1; rename one set.
- [ ] **Figure captions: move interpretation to body text** — e.g., injury ATT caption states a result rather than describing the figure.
- [ ] **RAPTOR undefined on first mention** — needs definition for non-sports-statistics readers.
- [x] **Predictive accuracy narrative reframed** — no longer claims the model dominates AR everywhere; now states "broadly comparable with added structure." MAP-based catastrophic log-loss claim removed.
- [x] **Coverage table bolding** — now marks the model closest to nominal 95% per metric (penalises over- and under-coverage equally), with caption updated.
- [x] **Prior equation overflow** — structured prior align block no longer overflows page margins.
