"""Prior-predictive checks & ELPPD for prior-knob tuning.

Draws from the prior predictive distribution over the actual dataset and scores a prior
configuration, primarily via prior-predictive ELPPD (expected log pointwise predictive density of
the observed data under prior parameter draws), plus distributional coverage and peak checks.

Usage:
    python prior_check.py --model_name=nba_convex_max_tvlinearlvm_stratified_next_k \
        --model_config=config/model_config.yaml --num_prior_samples=500 --s_chunk=50
    # override a knob:
    python prior_check.py ... --set sigma_curve=0.2 --set sigma_X=1.5
    # sweep a knob (one run per value -> sweep_summary.csv):
    python prior_check.py ... --sweep sigma_curve=0.25,0.5,1.0

The ELPPD uses the model's OWN likelihood (numpyro.infer.log_likelihood), so it is guaranteed
consistent with the fitted model. Curves are produced via the shared forward model._compute_mu
(no deterministic recordings).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import argparse
import re
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.infer import Predictive, log_likelihood
from numpyro import handlers
jax.config.update("jax_enable_x64", True)

from config.config_utils import resolve_model_config
from model.inference_inputs import build_inference_inputs
from model.inference_utils import get_latent_sites, create_metric_trajectory_all


def parse_args():
    p = argparse.ArgumentParser(description="Prior predictive checks & ELPPD")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--num_prior_samples", type=int, default=500)
    p.add_argument("--s_chunk", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="override a knob, e.g. --set sigma_curve=0.2 (repeatable)")
    p.add_argument("--sweep", default=None,
                   help="sweep a knob, e.g. --sweep sigma_curve=0.25,0.5,1.0")
    p.add_argument("--oat", action="store_true",
                   help="one-at-a-time scan of the built-in OAT_GRID (base + each knob varied alone)")
    p.add_argument("--players", default=None,
                   help="comma-separated player names for per-player prior-draw plots (default: a preset list)")
    p.add_argument("--player_draws", type=int, default=60,
                   help="number of prior draws to export per player for the spaghetti plots")
    return p.parse_args()


def _save(df, out_dir, name):
    """Write a prior-check table as parquet (consistent with model_export.py outputs)."""
    df.to_parquet(os.path.join(out_dir, name + ".parquet"), index=False)


from model.inference_inputs import knobs_from_overrides as _knobs_from_overrides, _parse_scalar  # shared parsers


# ── core: prior latent draws ────────────────────────────────────────────────
def _latent_site_names(model, model_args):
    model._prior_predictive = False
    names = set(get_latent_sites(model.model_fn, model_args))
    return {n for n in names if not n.startswith("likelihood_")}


def _draw_prior_latents(model, model_args, latent_names, s_chunk, key):
    """One chunk of prior latent draws (posterior empty -> all latents from the prior)."""
    model._prior_predictive = False
    draws = Predictive(model.model_fn, posterior_samples={}, num_samples=s_chunk)(key, **model_args)
    return {k: v for k, v in draws.items() if k in latent_names}


# ── ELPPD ───────────────────────────────────────────────────────────────────
def compute_prior_elppd(model, model_args, masks, distribution_indices, metrics, age_min, S, s_chunk, seed):
    """Returns (elppd_total, per_metric_dict, long_df[metric,player,age,lppd])."""
    latent_names = _latent_site_names(model, model_args)
    families = list(distribution_indices.keys())
    fam_partials = {f: [] for f in families}   # list of per-chunk logsumexp_s over FINITE draws
    fam_nfinite = {f: 0 for f in families}     # per-cell count of finite draws (accumulated)
    surv_partials, surv_nfinite = [], 0        # per-player survival (exit-time) log-lik
    key = jax.random.PRNGKey(seed)
    n_done = 0
    while n_done < S:
        sc = min(s_chunk, S - n_done)
        key, k1 = jax.random.split(key)
        latents = _draw_prior_latents(model, model_args, latent_names, sc, k1)
        model._prior_predictive = False  # condition on obs so likelihood is scored
        ll = log_likelihood(model.model_fn, latents, batch_ndims=1, **model_args)
        for f in families:
            site = ll[f"likelihood_{f}"]            # (sc, n_obs_f)
            # A diffuse prior can make a draw's predictive density numerically non-finite
            # (e.g. NegBin mean=exp(log_rate) overflows). Treat such draws as ~zero density
            # (-inf in log space) so they drop out of the predictive mixture average.
            finite = jnp.isfinite(site)
            site_safe = jnp.where(finite, site, -jnp.inf)
            fam_partials[f].append(jnp.asarray(logsumexp(site_safe, axis=0)))
            fam_nfinite[f] = fam_nfinite[f] + np.asarray(finite.sum(axis=0))
        # survival (exit-time) likelihood: factor sites, masked obs vs censored -> per-player log-lik
        if "log_lik_exit_observed" in ll:
            surv = ll["log_lik_exit_observed"] + ll["log_lik_exit_censored"]  # (sc, n); masked term is 0
            sfin = jnp.isfinite(surv)
            surv_partials.append(jnp.asarray(logsumexp(jnp.where(sfin, surv, -jnp.inf), axis=0)))
            surv_nfinite = surv_nfinite + np.asarray(sfin.sum(axis=0))
        n_done += sc

    DEGEN_FLOOR = -100.0   # per-obs lppd when no draw gave finite density (prior excludes the data)
    per_metric = {}
    per_metric_degen = {}
    rows = []
    for f in families:
        idx_f = np.asarray(distribution_indices[f])
        nfin = np.asarray(fam_nfinite[f])                                  # (n_obs_f,) finite-draw count
        with np.errstate(divide="ignore", invalid="ignore"):
            lppd_i = np.asarray(logsumexp(jnp.stack(fam_partials[f], axis=0), axis=0)) - np.log(np.maximum(nfin, 1))
        degenerate = nfin == 0
        lppd_i = np.where(degenerate, DEGEN_FLOOR, lppd_i)                 # finite & comparable
        fam_mask = np.asarray(masks[idx_f]) > 0                            # (n_f, n, j)
        coords = np.argwhere(fam_mask)                                     # row-major == arr[mask] order
        assert coords.shape[0] == lppd_i.shape[0], f"{f}: {coords.shape[0]} vs {lppd_i.shape[0]}"
        for lm, gm in enumerate(idx_f.tolist()):
            sel = coords[:, 0] == lm
            per_metric_degen[metrics[gm]] = int(degenerate[sel].sum())
        gmetric = idx_f[coords[:, 0]]
        for gm, p, a, v in zip(gmetric.tolist(), coords[:, 1].tolist(), coords[:, 2].tolist(), lppd_i.tolist()):
            rows.append((metrics[gm], p, age_min + a, v))
        for lm, gm in enumerate(idx_f.tolist()):
            sel = coords[:, 0] == lm
            per_metric[metrics[gm]] = float(lppd_i[sel].sum())
    # survival ELPPD (per-player); add as its own component + per-player rows (age=-1) for breakdowns
    if surv_partials:
        nfin = np.asarray(surv_nfinite)
        with np.errstate(divide="ignore", invalid="ignore"):
            surv_lppd = np.asarray(logsumexp(jnp.stack(surv_partials, axis=0), axis=0)) - np.log(np.maximum(nfin, 1))
        surv_lppd = np.where(nfin == 0, DEGEN_FLOOR, surv_lppd)            # (n,)
        per_metric["survival"] = float(surv_lppd.sum())
        per_metric_degen["survival"] = int((nfin == 0).sum())
        for p, v in enumerate(surv_lppd.tolist()):
            rows.append(("survival", p, -1, v))
    long_df = pd.DataFrame(rows, columns=["metric", "player_idx", "age", "lppd"])
    return float(sum(per_metric.values())), per_metric, per_metric_degen, long_df


# ── shared-forward curves (peaks) ───────────────────────────────────────────
def _compute_mu_draws_peaks(model, model_args, latents_all, key):
    """Run model._compute_mu under substitute for each prior draw; reduce to peak value/age.
    Uses jax.lax.map (SEQUENTIAL over draws) so peak memory is one draw's forward, not the
    whole batch — the convex core_tensor (n*k*t*M*M) is ~GBs per draw, so vmap would OOM.
    Returns peak_val (S,k,n), peak_age_idx (S,k,n)."""
    hp = model_args["hsgp_params"]; off = model_args["offsets"]
    sfi = model_args["sample_free_indices"]; sxi = model_args["sample_fixed_indices"]
    ami = model_args.get("ar_metric_indices", jnp.array([])); yi = model_args.get("year_indices", jnp.array([]))
    ny = model_args.get("num_years", 1); nd = model_args.get("num_de_trend", 0); rfi = model_args.get("ref_year_idx", 0)
    S = jax.tree_util.tree_leaves(latents_all)[0].shape[0]
    keys = jax.random.split(key, S)

    def one(args):
        latent, k = args
        def f():
            mu, _trend, _X = model._compute_mu(hp, off, sfi, sxi, ami, yi, ny, nd, rfi)
            return mu  # (k, n, j)
        mu = handlers.substitute(handlers.seed(f, k), data=latent)()
        return jnp.max(mu, axis=-1), jnp.argmax(mu, axis=-1)   # (k,n),(k,n)

    pv, pa = jax.lax.map(one, (latents_all, keys))
    return np.asarray(pv), np.asarray(pa)


# ── survival predictive check ───────────────────────────────────────────────
def survival_predictive_check(model, model_args, latents_all, age_min, age_max, key):
    """Simulate exit ages from the prior Gompertz hazard (h(t)=eta*exp(gamma*t)) per draw and
    compare to observed career ends. Returns (curve_df, summary) or (None, None) if no survival
    data. curve_df: prior-predictive survival band + empirical KM by age. summary: exit-age coverage."""
    off = model_args["offsets"]
    if "exit_times" not in off or "entrance_times" not in off:
        return None, None
    hp = off; sfi = model_args["sample_free_indices"]; sxi = model_args["sample_fixed_indices"]
    ami = model_args.get("ar_metric_indices", jnp.array([])); yi = model_args.get("year_indices", jnp.array([]))
    ny = model_args.get("num_years", 1); nd = model_args.get("num_de_trend", 0); rfi = model_args.get("ref_year_idx", 0)
    hsgp = model_args["hsgp_params"]
    S = jax.tree_util.tree_leaves(latents_all)[0].shape[0]
    keys = jax.random.split(key, S)

    def one(args):
        latent, k = args
        def f():
            _mu, _t, X = model._compute_mu(hsgp, off, sfi, sxi, ami, yi, ny, nd, rfi)
            eta, gamma = model._survival_rates(X)
            return eta[:, 0], gamma[:, 0]   # (n,),(n,)
        return handlers.substitute(handlers.seed(f, k), data=latent)()

    eta_all, gamma_all = jax.lax.map(one, (latents_all, keys))
    eta_all = np.clip(np.asarray(eta_all), 1e-8, None)
    gamma_all = np.clip(np.asarray(gamma_all), 1e-3, None)         # avoid /0 (Gompertz->exp limit)
    a = np.asarray(off["entrance_times"]).ravel()                  # entrance age (relative to age_min)
    exit_rel = np.asarray(off["exit_times"]).ravel()
    rc = np.asarray(off["right_censor"]).ravel().astype(bool)
    obs_exit = exit_rel + age_min
    ent_abs = a + age_min

    # inverse-CDF simulate exit age: exp(gamma*T) = exp(gamma*a) + (gamma/eta)*Exp(1); stable via logaddexp
    rng = np.random.default_rng(0)
    e = rng.exponential(1.0, size=eta_all.shape)                   # (S,n) = -log u
    Trel = np.logaddexp(gamma_all * a[None], np.log(gamma_all) - np.log(eta_all) + np.log(e)) / gamma_all
    sim_exit = np.clip(Trel + age_min, age_min, age_max)           # (S,n); >age_max -> survived window

    q05, q50, q95 = np.nanpercentile(sim_exit, [5, 50, 95], axis=0)   # (n,)
    cov_unc = float(np.mean((obs_exit[~rc] >= q05[~rc]) & (obs_exit[~rc] <= q95[~rc]))) if (~rc).any() else float("nan")
    cov_cen = float(np.mean(q95[rc] >= obs_exit[rc])) if rc.any() else float("nan")

    grid = np.arange(age_min, age_max + 1)
    Sdraw = np.full((eta_all.shape[0], grid.size), np.nan)
    for ti, t in enumerate(grid):
        risk = ent_abs <= t
        if risk.sum() > 0:
            Sdraw[:, ti] = (sim_exit[:, risk] > t).mean(axis=1)
    pS = np.nanpercentile(Sdraw, [5, 50, 95], axis=0)
    km = np.ones(grid.size); s = 1.0                              # KM (left-trunc + right-censor)
    for ti, t in enumerate(grid):
        at_risk = int(np.sum((ent_abs <= t) & (obs_exit >= t)))
        events = int(np.sum((~rc) & (obs_exit >= t) & (obs_exit < t + 1) & (ent_abs <= t)))  # exits in [t,t+1)
        if at_risk > 0:
            s *= (1.0 - events / at_risk)
        km[ti] = s
    curve_df = pd.DataFrame({"age": grid, "prior_S_q05": pS[0], "prior_S_q50": pS[1], "prior_S_q95": pS[2], "empirical_S": km})
    summary = {"surv_cover_uncensored": cov_unc, "surv_cover_censored": cov_cen,
               "obs_exit_median": float(np.median(obs_exit)), "prior_exit_median": float(np.nanmedian(sim_exit))}
    return curve_df, summary


# Default players for per-player prior-draw trajectory plots — kept in sync with
# `posterior_plot_names` in data_analysis/diagnostics_utils.r (the canonical set model_diagnostics.r
# uses for its per-player posterior plots) so prior and posterior plots cover the same players.
SELECT_PLAYERS = [
    "Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Dwight Howard",
    "Nikola Jokic", "Kevin Garnett", "Steve Nash",
    "Chris Paul", "Shaquille O'Neal", "Anthony Edwards", "Jamal Murray",
    "Donovan Mitchell", "Ray Allen", "Klay Thompson",
    "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki",
    "Jason Kidd", "Marcus Camby", "Rudy Gobert", "Tim Duncan",
    "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic",
    "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton",
    "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose",
    "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis",
    "Giannis Antetokounmpo", "Jrue Holiday", "No Name",
]


def player_prior_draws(model, model_args, inp, latents_all, select_names, key, n_draws=60):
    """Per-player FULL prior-predictive production curves for select players: draws of the OBSERVABLE
    (not just the latent mean), incl. observational-variance params (sigma / dispersions) and the
    games->minutes->metrics exposure process — reuses create_metric_trajectory_all, the exact sampler
    the posterior export uses, so prior and posterior plots share scale + recipe (rates per-36, etc.).
    Returns (draws_df, obs_df). select_names="all" exports every player (excludes synthetic 'No Name')."""
    names = list(inp.names); metrics = list(inp.metrics); fams = list(inp.metric_output)
    exposure_list = list(inp.exposure_list); basis = np.asarray(inp.basis)
    want_all = (select_names == "all") or (isinstance(select_names, (list, tuple)) and "all" in select_names)
    if want_all:
        sel = [(nm, i) for i, nm in enumerate(names) if nm != "No Name"]
    else:
        sel = [(nm, names.index(nm)) for nm in select_names if nm in names]
    if not sel:
        return None, None
    sel_idx = np.array([i for _, i in sel]); sel_names_arr = np.array([nm for nm, _ in sel])
    hp = model_args["hsgp_params"]; off = model_args["offsets"]
    sfi = model_args["sample_free_indices"]; sxi = model_args["sample_fixed_indices"]
    ami = model_args.get("ar_metric_indices", jnp.array([])); yi = model_args.get("year_indices", jnp.array([]))
    ny = model_args.get("num_years", 1); nd = model_args.get("num_de_trend", 0); rfi = model_args.get("ref_year_idx", 0)
    S = min(n_draws, jax.tree_util.tree_leaves(latents_all)[0].shape[0])
    lat = {k: v[:S] for k, v in latents_all.items()}
    keys = jax.random.split(key, S)
    sel_idx_j = jnp.asarray(sel_idx)

    def one(args):
        latent, k = args
        def f():
            mu, trend_ar, _X = model._compute_mu(hp, off, sfi, sxi, ami, yi, ny, nd, rfi)
            # Full linear predictor = aging curve + per-player AR(1) + calendar trend, matching what
            # the likelihood uses. _compute_player_ar is zero for non-AR models, the sampled AR(1)
            # otherwise, so the AR variants' draws actually show the AR contribution.
            lp = mu + model._compute_player_ar() + trend_ar
            return lp[:, sel_idx_j, :]   # (k, n_sel, j)
        return handlers.substitute(handlers.seed(f, k), data=latent)()

    mu_sel = jax.lax.map(one, (lat, keys))                  # (S, k, n_sel, j)
    mu5 = jnp.asarray(mu_sel)[None]                          # (chains=1, S, k, n_sel, j)
    Ysel = jnp.asarray(np.asarray(inp.Y)[:, sel_idx, :])    # (k, n_sel, j)
    Esel = jnp.asarray(np.asarray(inp.exposures)[:, sel_idx, :])

    def _disp(site):  # prior dispersion draws -> (family, chains=1, S) as the sampler expects
        if site not in lat:
            return None
        d = np.asarray(lat[site])[:S]                       # (S, n_family)
        return jnp.asarray(np.transpose(d[None], (2, 0, 1)))
    var = _disp("sigma"); dbeta = _disp("sigma_beta")
    kappa = _disp("sigma_beta_binomial"); negb = _disp("sigma_negative_binomial")

    # Prior predictive (observable + obs variance), CONDITIONED ON OBSERVED EXPOSURES: condition the
    # games/pct_minutes exposure chain on the player's actual playing time (condition_on_observed=True)
    # instead of sampling games from the prior BetaBinomial. This isolates metric-rate uncertainty
    # (FG2A/36, etc.) from exposure-generation noise, matching the posterior conditional export
    # (model_export.py pos_conditional). Esel/Ysel are the observed exposures/values for these players.
    obs_arr, pred_arr = create_metric_trajectory_all(
        mu5, Ysel, Esel, fams, metrics, exposure_list,
        var, dbeta, posterior_kappa_samples=kappa, posterior_neg_bin_samples=negb,
        condition_on_observed=True)
    yp = np.asarray(pred_arr)[0]         # (chains=1, S, n_sel, T, M) -> (S, n_sel, T, M)
    yo = np.asarray(obs_arr)             # (n_sel, T, M)
    # metric axis is returned in the original `metrics` order (create_metric_trajectory_all L413-416)
    met_names = np.array(metrics)
    Sn, Np, T, M = yp.shape
    draws_df = pd.DataFrame({
        "name": pd.Categorical(sel_names_arr[np.tile(np.repeat(np.arange(Np), T * M), Sn)]),
        "metric": pd.Categorical(met_names[np.tile(np.arange(M), Sn * Np * T)]),
        "age": np.tile(np.repeat(basis, M), Sn * Np).astype(np.int16),
        "draw": np.repeat(np.arange(Sn), Np * T * M).astype(np.int16),
        "value": yp.reshape(-1).astype(np.float32),
    })
    obs_df = pd.DataFrame({
        "name": pd.Categorical(sel_names_arr[np.repeat(np.arange(Np), T * M)]),
        "metric": pd.Categorical(met_names[np.tile(np.arange(M), Np * T)]),
        "age": np.tile(np.repeat(basis, M), Np).astype(np.int16),
        "obs_value": yo.reshape(-1).astype(np.float32),
    })
    obs_df = obs_df[np.isfinite(obs_df["obs_value"])]
    return draws_df, obs_df


# ── distributional PPC (y_rep coverage + per-age ribbon data) ────────────────
def compute_ppc(model, model_args, masks, distribution_indices, metrics, age_min, S, s_chunk, seed):
    """Draw y_rep at observed cells (prior predictive); return
    (coverage_df[metric,family,n_obs,cover90,cover50], by_age_df[metric,age,prior q's,obs stats])."""
    families = list(distribution_indices.keys())
    yrep_partials = {f: [] for f in families}
    key = jax.random.PRNGKey(seed + 12345)
    n_done = 0
    while n_done < S:
        sc = min(s_chunk, S - n_done)
        key, k1 = jax.random.split(key)
        model._prior_predictive = True   # draw likelihood (obs=None) -> y_rep at masked cells
        draws = Predictive(model.model_fn, posterior_samples={}, num_samples=sc)(k1, **model_args)
        for f in families:
            yrep_partials[f].append(np.asarray(draws[f"likelihood_{f}"]))  # (sc, n_obs_f)
        n_done += sc
    cov_rows, age_rows = [], []
    for f in families:
        idx_f = np.asarray(distribution_indices[f])
        yrep = np.concatenate(yrep_partials[f], 0)                  # (S, n_obs_f)
        fam_mask = np.asarray(masks[idx_f]) > 0
        coords = np.argwhere(fam_mask)                              # (n_obs_f, 3) local_metric, player, age_idx
        Yobs = np.asarray(model_args["data_set"][f]["Y"])[fam_mask]  # (n_obs_f,)
        lo, hi = np.nanpercentile(yrep, [5, 95], axis=0)
        lo50, hi50 = np.nanpercentile(yrep, [25, 75], axis=0)
        inside90 = (Yobs >= lo) & (Yobs <= hi)
        inside50 = (Yobs >= lo50) & (Yobs <= hi50)
        for lm, gm in enumerate(idx_f.tolist()):
            sel = coords[:, 0] == lm
            cov_rows.append({"metric": metrics[gm], "family": f, "n_obs": int(sel.sum()),
                             "cover90": float(np.mean(inside90[sel])), "cover50": float(np.mean(inside50[sel]))})
            # per-age ribbon: pool y_rep across draws+players within (metric, age)
            ages_local = coords[sel, 2]
            for a in np.unique(ages_local):
                acol = (coords[:, 0] == lm) & (coords[:, 2] == a)
                pool = yrep[:, acol].reshape(-1)
                obs_a = Yobs[acol[coords[:, 0] == lm]] if False else Yobs[acol]
                qs = np.nanpercentile(pool, [5, 25, 50, 75, 95])
                age_rows.append({"metric": metrics[gm], "age": int(age_min + a),
                                 "prior_q05": qs[0], "prior_q25": qs[1], "prior_q50": qs[2], "prior_q75": qs[3], "prior_q95": qs[4],
                                 "obs_mean": float(np.nanmean(obs_a)), "obs_q50": float(np.nanmedian(obs_a)), "n_obs": int(obs_a.size)})
    return pd.DataFrame(cov_rows), pd.DataFrame(age_rows)


# ── one configuration end-to-end ────────────────────────────────────────────
def run_one_config(args, knobs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    inp = build_inference_inputs({**args, "prior_knobs": knobs})
    model, model_args = inp.model, inp.model_args
    S, s_chunk = args["num_prior_samples"], args["s_chunk"]

    # 1) ELPPD + breakdowns
    elppd_total, per_metric, per_metric_degen, long_df = compute_prior_elppd(
        model, model_args, inp.masks, inp.distribution_indices, inp.metrics, inp.age_min, S, s_chunk, args["seed"])

    fam_of = {inp.metrics[int(i)]: f for f, idxs in inp.distribution_indices.items() for i in np.asarray(idxs)}
    n_obs = long_df.groupby("metric").size()
    pm = pd.DataFrame({"metric": list(per_metric.keys()), "elppd": list(per_metric.values())})
    pm["family"] = pm["metric"].map(fam_of)
    pm["n_obs"] = pm["metric"].map(n_obs).fillna(0).astype(int)
    pm["n_degenerate"] = pm["metric"].map(per_metric_degen).fillna(0).astype(int)  # cells with no finite prior-pred draw
    pm["elppd_per_obs"] = pm["elppd"] / pm["n_obs"].replace(0, np.nan)
    _save(pm.sort_values("elppd"), out_dir, "prior_elppd_per_metric")
    _save(pd.DataFrame([{"elppd_total": elppd_total, "n_obs": int(len(long_df)), "S": S, **{f"knob_{k}": v for k, v in knobs.items()}}]),
          out_dir, "prior_elppd_total")

    # breakdowns: age + (position, minutes, stratum) joined via player metadata
    _save(long_df.groupby("age")["lppd"].agg(["mean", "sum", "size"]).reset_index(), out_dir, "prior_elppd_by_age")
    meta = inp.data.groupby("id").agg(position_group=("position_group", "first"), minutes=("minutes", "sum")).reset_index()
    # player_idx in long_df indexes the model's player axis == build order (== latent_space order)
    id_order = inp.data.groupby("id")["name"].first().reset_index()  # same groupby order as names
    id_order["player_idx"] = np.arange(len(id_order))
    meta = meta.merge(id_order[["id", "player_idx"]], on="id", how="right")
    ld = long_df.merge(meta, on="player_idx", how="left")
    _save(ld.groupby("position_group")["lppd"].agg(["mean", "sum", "size"]).reset_index(), out_dir, "prior_elppd_by_position")
    _mq = np.nanpercentile(meta["minutes"].dropna().values, [33, 67]) if meta["minutes"].notna().any() else [0, 0]
    ld["min_bucket"] = pd.cut(ld["minutes"], [-1, _mq[0], _mq[1], np.inf], labels=["low", "mid", "high"])
    _save(ld.groupby("min_bucket", observed=True)["lppd"].agg(["mean", "sum", "size"]).reset_index(), out_dir, "prior_elppd_by_minutes")
    if inp.holdout_indices_path and os.path.exists(inp.holdout_indices_path):
        ho = pd.read_csv(inp.holdout_indices_path)
        if "stratum" in ho.columns:
            # stratum is a per-player property (constant across a player's tail). ELPPD is scored on
            # TRAINING cells (holdout is masked out), so map each player -> stratum and group on player.
            ho_p = ho.drop_duplicates("player")[["player", "stratum"]].merge(
                id_order[["id", "player_idx"]], left_on="player", right_on="id", how="left")
            lds = long_df.merge(ho_p[["player_idx", "stratum"]], on="player_idx", how="inner")
            _save(lds.groupby("stratum")["lppd"].agg(["mean", "sum", "size"]).reset_index(), out_dir, "prior_elppd_by_stratum")

    # 2) coverage + per-age ribbon data (for R plots)
    cov, by_age = compute_ppc(model, model_args, inp.masks, inp.distribution_indices, inp.metrics, inp.age_min, S, s_chunk, args["seed"])
    _save(cov.sort_values("metric"), out_dir, "prior_ppc_coverage")
    _save(by_age.sort_values(["metric", "age"]), out_dir, "prior_ppc_by_age")

    # 3) peak checks (prior mu-peak vs empirical peak offset c_max, on the linear-predictor scale)
    # Sequential (lax.map) forward is GBs/draw, so cap the peak draws (distribution converges fast).
    # Requires a convex _compute_mu matching the model's forward; skip (with a note) for models
    # whose forward differs (e.g. naive) so ELPPD/coverage still get produced.
    latents_all = None
    try:
        S_peak = min(S, 64)
        latent_names = _latent_site_names(model, model_args)
        key = jax.random.PRNGKey(args["seed"] + 999)
        lat_all = []
        n_done = 0
        while n_done < S_peak:
            sc = min(s_chunk, S_peak - n_done)
            key, k1 = jax.random.split(key)
            lat_all.append(_draw_prior_latents(model, model_args, latent_names, sc, k1))
            n_done += sc
        latents_all = {k: jnp.concatenate([d[k] for d in lat_all], axis=0) for k in lat_all[0]}
        peak_val, peak_age_idx = _compute_mu_draws_peaks(model, model_args, latents_all, key)  # (S_peak,k,n)
        # Empirical per-metric peak (lin-pred scale). Only the convex/linear family stores a (k,)
        # "c_max" anchor; naive/gplvm anchor on "c_mean" (scalar) so there's no per-metric overlay —
        # still emit the prior peak interval, just without the empirical comparison.
        _cmax = model_args["offsets"].get("c_max")
        c_max_emp = np.atleast_1d(np.asarray(_cmax, dtype=float)) if _cmax is not None else None
        have_emp = c_max_emp is not None and c_max_emp.ndim == 1 and c_max_emp.shape[0] == len(inp.metrics)
        basis = np.asarray(inp.basis)
        prows = []
        for ki, m in enumerate(inp.metrics):
            pv = peak_val[:, ki, :].reshape(-1)
            pa = basis[peak_age_idx[:, ki, :].reshape(-1)]
            emp = float(c_max_emp[ki]) if have_emp else np.nan
            prows.append({"metric": m,
                          "prior_peak_val_q05": float(np.nanpercentile(pv, 5)),
                          "prior_peak_val_med": float(np.nanmedian(pv)),
                          "prior_peak_val_q95": float(np.nanpercentile(pv, 95)),
                          "empirical_peak_val": emp,
                          "prior_peak_age_med": float(np.nanmedian(pa)),
                          "empirical_outside_90": bool(have_emp and not (np.nanpercentile(pv, 5) <= emp <= np.nanpercentile(pv, 95)))})
        _save(pd.DataFrame(prows), out_dir, "prior_peak_check")
    except Exception as _e:
        print(f"[prior_check] peak check skipped ({type(_e).__name__}: {_e})")

    # 4) survival predictive check (exit-time hazard) — reuses the peak-draw latents
    surv_sum = {}
    try:
        if latents_all is not None:
            curve_df, surv_sum = survival_predictive_check(
                model, model_args, latents_all, inp.age_min, inp.age_max, jax.random.PRNGKey(args["seed"] + 7))
            if curve_df is not None:
                _save(curve_df, out_dir, "prior_survival_curve")
                _save(pd.DataFrame([surv_sum]), out_dir, "prior_survival_check")
                print(f"[prior_check]   survival: cover(unc)={surv_sum['surv_cover_uncensored']:.3f} "
                      f"cover(cens)={surv_sum['surv_cover_censored']:.3f} "
                      f"exit median obs/prior={surv_sum['obs_exit_median']:.1f}/{surv_sum['prior_exit_median']:.1f}")
            else:
                surv_sum = {}
    except Exception as _e:
        print(f"[prior_check] survival check skipped ({type(_e).__name__}: {_e})")

    # 5) per-player prior-draw production curves for select players (parquet, like model_export.py)
    try:
        if latents_all is not None:
            sel_names = args.get("select_players") or SELECT_PLAYERS
            draws_df, obs_df = player_prior_draws(
                model, model_args, inp, latents_all, sel_names, jax.random.PRNGKey(args["seed"] + 3),
                n_draws=args.get("player_draws", 60))
            if draws_df is not None:
                _save(draws_df, out_dir, "prior_player_draws")
                _save(obs_df, out_dir, "prior_player_obs")
                print(f"[prior_check]   per-player prior draws: {draws_df['name'].nunique()} players "
                      f"x {draws_df['draw'].nunique()} draws -> prior_player_draws.parquet")
    except Exception as _e:
        print(f"[prior_check] per-player draws skipped ({type(_e).__name__}: {_e})")

    # Plots are produced by R (data_analysis/prior_check.r in the r-new container) from these parquet files.
    print(f"[prior_check] {out_dir}: ELPPD_total={elppd_total:.1f}  mean_cover90={cov['cover90'].mean():.3f}  "
          f"survival_elppd={per_metric.get('survival', float('nan')):.1f}")
    return {"elppd_total": elppd_total, "mean_cover90": float(cov["cover90"].mean()),
            **{f"elppd_{m}": v for m, v in per_metric.items()},
            **{k: v for k, v in surv_sum.items()},
            **{f"knob_{k}": v for k, v in knobs.items()}}


# Distribution-spec helpers for the OAT grid.
def _HN(s):      return {"dist": "HalfNormal", "args": [s]}
def _LN(mu, s):  return {"dist": "LogNormal", "args": [mu, s]}
def _Exp(r):     return {"dist": "Exponential", "args": [r]}
def _Uni(a, b):  return {"dist": "Uniform", "args": [a, b]}
def _IG(a, b):   return {"dist": "InverseGamma", "args": [a, b]}

# One-at-a-time knob grid for the convex-max family. Each value is a scalar (fixed constant) or a
# distribution spec. For the currently-FIXED scales we test fixed-low, fixed-high, AND an actual
# distributional prior (HalfNormal) — i.e. un-fixing the scale so it is sampled. Base = current priors.
OAT_GRID = {
    # currently-fixed scales: fixed brackets + as-a-prior
    "sigma_c":        [0.5, 3.0, _HN(1.0)],     # c_max (peak VALUE) spread
    "sigma_t":        [0.5, 3.0, _HN(1.0)],     # t_max (peak AGE) spread
    "sigma_t_offset": [0.5, 3.0, _HN(1.0)],     # per-player x metric peak-AGE RE
    "sigma_c_offset": [0.5, 3.0, _HN(1.0)],     # per-player x metric peak-VALUE RE
    "sigma_curve":    [0.2, 1.5, _HN(0.5)],     # per-player curvature amplitude
    "sigma_X":        [0.3, 3.0, _HN(1.0)],     # latent scale
    "sigma_W_proj":   [0.5, 2.0, _HN(1.0)],     # covariate->latent projection scale
    # other knobs
    "x_latent_df":    [2.0, 30.0],              # latent StudentT tail (2=heavy, 30~Normal)
    "alpha":          [0.5, 2.0, _HN(1.0)],     # GP amplitude (fixed=1 base); scalar broadcasts to (k,1)
    "lengthscale_deriv": [_LN(1.0986, 0.3), _LN(1.0986, 1.2)],  # curve smoothness (log(3) +- sigma)
    "sigma_negative_binomial": [_Exp(0.2), _Exp(5.0)],          # NB dispersion (fta/fg2a/fg3a)
    "sigma_beta":     [_Uni(0.0, 5.0), _HN(2.0)],               # beta precision (usg/pct_minutes)
    "sigma_beta_binomial": [_Exp(0.2), _Exp(5.0)],              # beta-binomial dispersion (games)
    "sigma":          [_IG(3.0, 6000.0), _IG(3.0, 500.0)],      # gaussian obs scale (obpm/dbpm)
}


def _val_label(v):
    if isinstance(v, dict) and "dist" in v:
        return f"{v['dist']}({','.join(map(str, v.get('args', [])))})"
    return str(v)


def quick_score(args, knobs):
    """Lightweight ELPPD + mean coverage for one knob config (no per-config CSVs/peaks/plots)."""
    inp = build_inference_inputs({**args, "prior_knobs": knobs})
    model, model_args = inp.model, inp.model_args
    S, s_chunk = args["num_prior_samples"], args["s_chunk"]
    elppd_total, per_metric, _degen, _long = compute_prior_elppd(
        model, model_args, inp.masks, inp.distribution_indices, inp.metrics, inp.age_min, S, s_chunk, args["seed"])
    cov, _ = compute_ppc(model, model_args, inp.masks, inp.distribution_indices, inp.metrics, inp.age_min, S, s_chunk, args["seed"])
    cov_by_metric = dict(zip(cov["metric"], cov["cover90"]))
    return {"elppd_total": elppd_total, "mean_cover90": float(cov["cover90"].mean()),
            **{f"elppd_{m}": v for m, v in per_metric.items()},
            **{f"cover90_{m}": cov_by_metric[m] for m in cov_by_metric}}


def run_oat(args, base_knobs, out_dir):
    """One-at-a-time scan: base + each (knob, value) with all other knobs held at base."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"swept_knob": "<base>", "value": "base", **quick_score(args, base_knobs)}]
    print(f"[oat] base: ELPPD={rows[0]['elppd_total']:.3e}  cover90={rows[0]['mean_cover90']:.3f}")
    for knob, vals in OAT_GRID.items():
        for v in vals:
            knobs = {**base_knobs, knob: v}
            try:
                r = {"swept_knob": knob, "value": _val_label(v), **quick_score(args, knobs)}
            except Exception as e:
                print(f"[oat] {knob}={_val_label(v)}: FAILED ({type(e).__name__}: {e})")
                continue
            rows.append(r)
            print(f"[oat] {knob}={_val_label(v)}: ELPPD={r['elppd_total']:.3e}  cover90={r['mean_cover90']:.3f}")
            _save(pd.DataFrame(rows), out_dir, "oat_summary")  # incremental
    print(f"[oat] wrote {os.path.join(out_dir, 'oat_summary.parquet')} ({len(rows)} configs)")


def main():
    a = parse_args()
    args = resolve_model_config(a.model_config, a.model_name, "map")
    args["model_name"] = a.model_name
    args.pop("fixed_params", None)
    args["num_prior_samples"] = a.num_prior_samples
    args["s_chunk"] = a.s_chunk
    args["seed"] = a.seed
    args["player_draws"] = a.player_draws
    if a.players:
        args["select_players"] = [p.strip() for p in a.players.split(",") if p.strip()]
    base_knobs = dict(args.get("prior_knobs", {}) or {})
    base_knobs.update(_knobs_from_overrides(a.overrides))
    # Outputs go to a dedicated "prior" inference dir (sibling of map/mcmc), e.g.
    # model_output/<base>/<scheme>/prior — mirroring how model_export.py lays out posterior dirs.
    _md = args.get("model_dir") or f"model_output/{a.model_name}/map"
    base_out = re.sub(r"/(map|mcmc|svi|prior)$", "/prior", _md)
    if not base_out.endswith("/prior"):
        base_out = os.path.join(_md, "prior")

    if a.oat:
        run_oat(args, base_knobs, os.path.join(base_out, "oat"))
    elif a.sweep:
        knob, _, vals = a.sweep.partition("=")
        knob = knob.strip()
        summary = []
        for v in vals.split(","):
            v = _parse_scalar(v.strip())
            knobs = {**base_knobs, knob: v}
            out_dir = os.path.join(base_out, f"{knob}={v}")
            summary.append(run_one_config(args, knobs, out_dir))
        os.makedirs(base_out, exist_ok=True)
        _save(pd.DataFrame(summary), base_out, "sweep_summary")
        print(f"[prior_check] sweep_summary written to {os.path.join(base_out, 'sweep_summary.parquet')}")
    else:
        run_one_config(args, base_knobs, base_out)


if __name__ == "__main__":
    main()
