import pandas as pd
import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import arviz as az
import jax
from jax import config, vmap
from numpyro.diagnostics import print_summary
from numpyro.infer.util import log_density as _log_density
config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, average_peak_differences, average_range_differences, create_surv_data, season_available_fraction
import numpyro
import jax.numpy as jnp
from model.model_utils import compute_residuals_map, compute_priors, make_survival_linear_injury_mcmc, apply_detrend_for_offsets, make_survival_linear_mcmc
from model.inference_utils import posterior_peaks_to_df, posterior_to_df, posterior_X_to_df, posterior_injury_to_df, posterior_injury_prior_mean_to_df, posterior_survival_to_df, posterior_player_scalar_to_df
from model.hsgp import vmap_make_convex_phi, eigenfunctions_multivariate, make_spectral_mixture_density, diag_spectral_density, sqrt_eigenvalues, make_convex_phi, make_convex_phi_prime, eigenfunctions
from visualization.visualization import make_diagnostic_heatmap, make_rhat_summary_barchart, plot_calendar_year_trends
from model.models import ConvexMaxARTVLinearLVM as _ARLinearLVM
from model.models import ConvexMaxTVLinearLVM, ConvexMaxInjuryTVLinearLVM, NaiveLinearLVM, TVLinearLVM, TVLinearLVM_AR
from model.inference_inputs import dispatch_model, apply_prior_knobs, _ATTRIBUTE_KNOBS, _build_knob_value
from model.inference_utils import create_metric_trajectory_all, create_metric_trajectory_map


def _c_max_re_from(res, c_max_var):
    """Scaled player x metric c_max random effect from a results dict, or None if the
    model did not fit it. Returns shape broadcastable to c_max: sigma_c_offset is put on
    each metric's natural scale via sqrt(c_max_var), then multiplied by the unit residual."""
    if "c_offset_re" not in res or "sigma_c_offset" not in res:
        return None
    sco = res["sigma_c_offset"]
    if c_max_var is not None:
        sco = sco * jnp.sqrt(jnp.asarray(c_max_var))
    return sco[..., None, :] * res["c_offset_re"]


if __name__ == "__main__":
    from config.config_utils import resolve_model_config, parse_metrics
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--inference_method", required=False, default="mcmc",
                        choices=["mcmc", "cut_mcmc"],
                        help="which sampler run to export (selects the regimes block / samples dir)")
    parser.add_argument("--concave_only", action="store_true",
                        help="write only the concave-loadings/canonical-curve parquets and exit — "
                             "skips log-posterior, latent-X, trajectory, injury, survival, and ELPPD "
                             "exports (cheap re-run feeding model_diagnostics.r's curvature plots)")
    parser.add_argument("--coverage_only", action="store_true",
                        help="write only the trajectory block feeding coverage.r (posterior_ar / "
                             "posterior_ar_conditional / peaks / mu / latent-ar / calendar-trend) and "
                             "exit — skips modal-latent, log-posterior, concave, injury, survival, "
                             "ELPPD and derivative exports (targeted re-run after sampler fixes)")
    parser.add_argument("--with_elppd", action="store_true",
                        help="modifier for --coverage_only: additionally run the per-sample "
                             "log-loss/ELPPD block (and the survival exports it needs) before "
                             "exiting — for ELPPD-checking pilot variants without a full export")
    parser.add_argument("--peaks_no_re", action="store_true",
                        help="zero the per-player curve random effects (c/t_offset_re, curve_re) "
                             "and write ONLY the resulting peaks to posterior_peaks_ar_shared / "
                             "posterior_peak_vals_ar_shared, then exit. These are the ARCHETYPAL "
                             "peaks -- the shared component f^k(X_p,t) with no idiosyncratic "
                             "offset -- and are what the peak-age / peak-value PCA figures must "
                             "use: the nugget is orthogonal to X by construction, so including it "
                             "injects an uninterpretable noise direction into axes whose whole "
                             "meaning is archetypal. Prediction/coverage keeps the RE-inclusive peaks.")
    numpyro.set_platform("cpu")
    _cli = vars(parser.parse_args())
    cfg = resolve_model_config(_cli["model_config"], _cli["model_name"], inference_method=_cli["inference_method"])

    model_name      = _cli["model_name"]
    _concave_only   = _cli["concave_only"]
    _coverage_only  = _cli["coverage_only"]
    _with_elppd     = _cli["with_elppd"]
    _peaks_no_re    = _cli["peaks_no_re"]
    _is_naive       = "naive" in model_name
    model_dir       = cfg.get("model_dir") or f"model_output/{model_name}/{_cli['inference_method']}"
    os.makedirs(model_dir, exist_ok=True)
    mcmc_path       = os.path.join(model_dir, "samples.pkl")
    svi_path        = cfg["init_path"]
    basis_dims      = cfg["basis_dims"]
    approx_x_dim    = cfg["approx_x_dim"]
    injury          = cfg["injury"]
    if _coverage_only and injury:
        raise SystemExit("--coverage_only does not support injury models: latent_val would omit "
                         "the injury effect (added inside the injury export block) and posterior_ar "
                         "would be wrong. Run a full export instead.")
    censor_survival_at_injury = cfg.get("censor_survival_at_injury", False)
    position_group  = cfg["position_group"]
    players         = cfg["player_names"]
    de_trend_metrics = cfg["de_trend_metrics"]
    validation_year = cfg["validation_year"]
    cohort_year     = cfg["cohort_year"]
    age_min         = cfg["age_min"]
    age_max         = cfg["age_max"]
    start_year      = cfg.get("start_year")
    end_year        = cfg.get("end_year")
    m_time          = cfg.get("m_time") or 15
    thin            = -1

    _year_filter = f"age <= {age_max} & name != 'Brandon Williams'"
    if start_year is not None:
        _year_filter += f" & year >= {start_year}"
    if end_year is not None:
        _year_filter += f" & year <= {end_year}"
    # Must mirror main.py's panel resolution. This was hardcoded to the base panel while main.py
    # read cfg["injury_data_csv"], so any run on an alternate panel (placebo, symmetric_v2, the
    # athleticism panel) was FIT on one dataframe and EXPORTED against a different one. Fatal for
    # panels with different metric columns; silently wrong for panels that only relabel injuries.
    _panel_csv = cfg.get("injury_data_csv") or "data/injury_player_cleaned.csv"
    print(f"[export] panel: {_panel_csv}")
    data = pd.read_csv(_panel_csv).query(_year_filter)
    # data = data.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year
    # data = data.groupby("id").filter(lambda x: len(x) >= 3) ### just test to keep guys who have played at least 3 years
    data["first_major_injury"] = (
        data["first_major_injury"]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace(
            {
                "Quad Tendon": "Quad/Patellar",
                "Patellar Tendon": "Quad/Patellar",
            },
        )
    )
    # Coarse mechanism grouping — MUST mirror main.py exactly. Without this the model is FIT on 4
    # mechanism groups while the export builds categories from the 8 ungrouped names, so
    # injury_type_labels is silently mismatched to the model's injury codes and every per-type
    # effect is attributed to the wrong injury (e.g. code 1 = "Axial", 19 players, was being
    # labelled "ACL"). Same class of defect as the panel path: the export must replicate main.py's
    # data prep, not a subset of it.
    if cfg.get("injury_type_grouping") == "mechanism4":
        data["first_major_injury"] = data["first_major_injury"].replace({
            "Achilles": "Tendon Rupture", "Quad/Patellar": "Tendon Rupture",
            "ACL": "Knee Structural", "Meniscus": "Knee Structural",
            "Foot Fracture": "Fracture", "Lower Body Fracture": "Fracture",
            "Hip": "Axial", "Back/Spine": "Axial",
        })
    data['first_major_injury'] = (
    data['first_major_injury']
            .astype('category')
            .cat.set_categories(
                ['None'] +
                [c for c in pd.unique(data['first_major_injury']) if c != 'None'],
                ordered=False
            ))
    data["injury_code"] = data["first_major_injury"].cat.codes
    injury_type_labels = [inj for inj in data["first_major_injury"].cat.categories if inj != "None"]
    injury_type_ids = np.arange(1, len(injury_type_labels) + 1)

    data["log_min"] = np.log(data["minutes"])
    data["usg"] /= 100
    data["usg"] += .01
    data["simple_exposure"] = 1
    # Games exposure must be rectified to AVAILABLE games exactly as main.py does, or GP% is FIT
    # against one denominator and EXPORTED against another. Measured on injury_player_cleaned_v2:
    # 590 rows (2.9%, all post-injury) differ by a mean of 28.5 games -- e.g. a rehab season fit
    # with denominator 51 was being exported as 82.
    _avail = season_available_fraction(data)
    _sched = data["total_games"] if _avail is None else np.round(data["total_games"] * _avail)
    data["games_exposure"] = np.maximum(_sched, data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games"]) / 48
    data["retirement"] = 1
    _fake_n = age_max - age_min + 1
    fake_data = pd.DataFrame({"age": range(age_min, age_max + 1), "id": 99999999, "year": range(2000, 2000 + _fake_n), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    data = pd.concat([data, fake_data], ignore_index=True)
    # computed AFTER the synthetic-player append, matching main.py (otherwise this list is one
    # short and any name->index lookup is shifted)
    names = data.groupby("id")["name"].first().values.tolist()
    _age_cols = range(age_min, age_max + 1)
    validation_mask = data[["year", "age", "id"]].pivot(columns="age", index="id", values=f"year").reindex(columns=_age_cols).apply(
                                                                        lambda r: r.dropna().iloc[0] + (np.array(list(_age_cols)) - r.dropna().index[0]) if r.notna().any() else r,
                                                                        axis=1,
                                                                        result_type="expand").to_numpy() > validation_year
    # validation_mask = data[["split","age", "id"]].pivot(columns="age", index="id", values="split").reindex(columns = range(age_min, age_max + 1)).to_numpy() == "test"
    metrics, metric_output, exposure_list = parse_metrics(cfg)

    scale_values = jnp.ones((len(metrics), 1))
    id_df = data[["id", "name", "position_group", "minutes"]].groupby("id").max().reset_index()
    id_df["id"] = id_df["id"].astype(str)

    holdout_indices_path = os.path.join(model_dir, "holdout_indices.csv")
    _stratum_mat = None
    _score_mask = None   # cells actually SCORED as holdout (== validation_mask unless a score_window flag narrows it, e.g. stratified_next_k)
    if os.path.exists(holdout_indices_path):
        holdout_df = pd.read_csv(holdout_indices_path)
        _age_cols_list = list(_age_cols)
        _age_to_idx = {age: idx for idx, age in enumerate(_age_cols_list)}
        _id_to_idx = {pid: idx for idx, pid in enumerate(id_df["id"].tolist())}
        _mask = np.zeros((len(id_df), len(_age_cols_list)), dtype=bool)
        _has_score_col = "score_window" in holdout_df.columns
        _score_mask = np.zeros((len(id_df), len(_age_cols_list)), dtype=bool)
        if "stratum" in holdout_df.columns:
            _stratum_mat = np.zeros((len(id_df), len(_age_cols_list)), dtype=int)
        for row in holdout_df.itertuples(index=False):
            pid = str(row.player)
            age = int(row.age)
            if pid in _id_to_idx and age in _age_to_idx:
                _r, _c = _id_to_idx[pid], _age_to_idx[age]
                _mask[_r, _c] = True
                if _has_score_col:
                    _sw = getattr(row, "score_window", 0)
                    if _sw is not None and not (isinstance(_sw, float) and np.isnan(_sw)) and int(_sw) == 1:
                        _score_mask[_r, _c] = True
                else:
                    _score_mask[_r, _c] = True   # no flag → every held-out cell is scored
                if _stratum_mat is not None:
                    _sv = getattr(row, "stratum", None)
                    if _sv is not None and not (isinstance(_sv, float) and np.isnan(_sv)):
                        _stratum_mat[_r, _c] = int(_sv)
        validation_mask = _mask
        print(f"Using holdout_indices.csv for validation_mask "
              f"({_mask.sum()} held-out cells; {_score_mask.sum()} scored).")
    # Cells scored as holdout (numpy bool). Defaults to the full held-out mask when no
    # score_window flag is present, so non-stratified schemes are unaffected.
    _score_mask_np = (np.asarray(_score_mask, dtype=bool)
                      if _score_mask is not None else np.asarray(validation_mask, dtype=bool))

    # NOTE ordering: main.py computes these BEFORE appending the synthetic player, this runs
    # AFTER, and the two do not agree -- the gaussian/beta branch is a merge on `year`, not a
    # transform, so the extra rows change the join. Measured: all 17 *_league_avg columns differ,
    # propagating to 446 cells of the de_trend tensor. Currently INERT because every config has
    # de_trend_metrics: [] and main.py zeroes de_trend under an empty mask, so no shipped number
    # is affected -- but it would bite silently the moment era de-trending is switched on.
    # Excluding the synthetic player here reproduces main.py's values exactly.
    _real = data["id"] != 99999999
    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data[_real].groupby(["year"]).apply(
            lambda g: (g[metric]*g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})

            data = data.merge(league_avg_broadcasted, on="year", how="left")
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli"]:
            _num = data[_real].groupby("year")[metric].sum()
            _den = data[_real].groupby("year")[exposure].sum()
            data[f"{metric}_league_avg"] = data["year"].map(_num / _den)
    
    if players:
        pattern = r"class-of-(\d{4})"
        player_indices = []
        for item in players:
            match = re.fullmatch(pattern, item)
            if item == "low-minutes":
                total_mins = data.groupby("id")["minutes"].sum().reset_index()
                for index, val in enumerate(total_mins["minutes"].values.tolist()):
                    if val <= np.percentile(total_mins["minutes"], 25) and index not in player_indices: 
                        player_indices.append(index) 
            elif match:
                year = int(match.group(1))
                subset =  data.groupby("id")["year"].min().reset_index()
                
                for index,val in enumerate(subset["year"].values.tolist()):
                    if index not in player_indices and val == year:
                        player_indices.append(index)
            else:
                if names.index(item) not in player_indices:
                    player_indices.append(names.index(item))
    elif position_group in ["G","F","C"]:
        all_indices = data.drop_duplicates(subset=["position_group","name","id"]).reset_index()
        player_indices = all_indices[all_indices["position_group"] == position_group].index.values.tolist()
    else:
        player_indices = []
    
    de_trend_indices = [True if metric in de_trend_metrics else False for metric in metrics]

    covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year, age_min=age_min, age_max=age_max)

    # Observed covariates: -log(draft_position_adj) and standardized height_inches
    # Player order matches create_fda_data groupby sort on id
    _player_obs = data.groupby("id")[["draft_position_adj", "height_inches", "position_group"]].first()
    _neg_log_draft = -np.log(_player_obs["draft_position_adj"].values.astype(float))
    _height_vals   = _player_obs["height_inches"].values.astype(float)
    _obs_numeric = np.stack([_neg_log_draft, _height_vals], axis=1)
    _obs_mean = np.nanmean(_obs_numeric, axis=0)
    _obs_std  = np.nanstd(_obs_numeric, axis=0) + 1e-8
    _obs_numeric_std = np.nan_to_num((_obs_numeric - _obs_mean) / _obs_std, nan=0.0)
    _pos_dummies = pd.get_dummies(_player_obs["position_group"], drop_first=True).astype(float).values  # (n, 2): F, G vs C baseline
    obs_covariates = jnp.array(np.concatenate([_obs_numeric_std, _pos_dummies], axis=1))  # (n, 4)

    surv_masks = None
    Y_surv = None
    distribution_families = set([data_entity["output"] for data_entity in data_set])
    distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
    injury_masks = jnp.stack([data_entity["injury_mask"] for data_entity in data_set])
    injury_types = jnp.stack([data_entity["injury_type"] for data_entity in data_set]).astype(jnp.int32)
    masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
    if "counterfactual" in model_name:
            masks = masks * (~injury_masks)
    exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
    Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
    de_trend = jnp.stack([data_entity["de_trend"] for data_entity in data_set])
    de_trend = jnp.where(jnp.array(de_trend_indices)[..., None, None], de_trend, 0.0)
    de_trend_adjusted = jnp.where(jnp.isnan(de_trend), 0.0, de_trend)

    # Calendar-year indices for TREND_AR reconstruction
    year_matrix = jnp.array(data_set[0]["year_matrix"])   # (n, j)
    min_year = int(jnp.nanmin(year_matrix))
    max_year = int(jnp.nanmax(year_matrix))
    num_years = max_year - min_year + 1
    year_indices = jnp.nan_to_num(year_matrix - min_year, nan=0).astype(int)   # (n, j)

    Y_for_offsets = apply_detrend_for_offsets(
        Y_obs=Y,
        exposures_obs=exposures,
        metric_families=metric_output,
        de_trend_values=de_trend,
        de_trend_mask=jnp.array(de_trend_indices),
    )
    offset_max, offset_max_var, offset_peak, offset_peak_var, offset_mean = compute_priors(Y_for_offsets, exposures, metric_output, exposure_list)
    offset_peak = offset_peak + age_min - basis.mean()

    offset_boundary_r = jnp.log(jnp.exp(2) - 1)
    offset_boundary_l = jnp.log(jnp.exp(2) - 1)
    data_dict = {}
    for family in distribution_families:
        family_dict = {}
        indices = distribution_indices[family]
        family_dict["Y"] = Y[indices]
        family_dict["exposure"] = exposures[indices]
        family_dict["mask"] = masks[indices]
        family_dict["indices"] = indices
        data_dict[family] = family_dict
    hsgp_params = {}
    if "convex" in model_name:
            x_time = basis - basis.mean()
  
            L_time = 2 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
            print(f"L_time: {L_time}, x_time: {x_time} ")
            M_time = m_time
            phi_time = vmap_make_convex_phi(jnp.squeeze(x_time), jnp.squeeze(L_time), M_time)
            hsgp_params["phi_x_time"] = phi_time
            hsgp_params["M_time"] = M_time
            hsgp_params["L_time"] = L_time
            hsgp_params["shifted_x_time"] = x_time + L_time
            hsgp_params["t_0"] = jnp.min(x_time)
            hsgp_params["t_r"] = jnp.max(x_time)
            hsgp_params["t_amplitude"] = float(jnp.squeeze(L_time)) / 2
            if "hsgp" in model_name:
                hsgp_params["eigenvalues_X"] = sqrt_eigenvalues(2 *  jnp.ones(basis_dims)[..., None] , approx_x_dim, basis_dims)


    
    player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                        "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                        "Chris Paul", "Shaquille O'Neal", "Trae Young"]
    predict_players = player_labels + ["Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                    "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                    "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                    "Devin Booker", "Paul Pierce", "Allen Iverson", 
                                    "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                    "Giannis Antetokounmpo", "Jrue Holiday", "No Name"]
    all_player_labels = id_df["name"].tolist()
    print("setup data")
    offset_dict = {"t_max": offset_peak, "c_max": offset_max, "c_mean": offset_mean, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_var, "c_max_var": offset_max_var}

    with open(svi_path, "rb") as f:
        results_map = pickle.load(f)
    f.close()
    results_map = {key.replace("__loc", ""): val for key,val in results_map.items()}
    with open(mcmc_path, "rb") as f:
        results_mcmc = pickle.load(f)
        if thin > 0:
            results_mcmc = {key: val[:, ::thin, ...] for key, val in results_mcmc.items()}
    f.close()
    _mcmc_sampled_keys = set(results_mcmc.keys())
    results_mcmc = {**results_map, **results_mcmc}

    # ── Legacy centered-X normalization ─────────────────────────────────────────────────────
    # The plain GP models (nba_tvlinearlvm / nba_tvlinearlvm_AR) were fit under the CENTERED X
    # parameterization: their stored "X" site is TOTAL X. The current classes rebuild
    # X = Z@W_proj + sigma_X * X_site from the substituted site, which double-applies the
    # covariate mean and shrinks by the MAP sigma_X — verified to shift e.g. USG predictions by
    # ~3 obs-SDs per player (coverage collapse to 2.8-16%). Pin the legacy semantics exactly:
    # X_used = X_stored (W_proj = 0, sigma_X = 1). Convex/RFF families are non-centered and
    # untouched.
    # Legacy fits predating the sampled sigma_X site were fit with the class default 1.0
    # (models.py "structured prior for X"); supply it so the standard non-centered
    # reconstruction X = Z @ W_proj + sigma_X * X_raw applies unchanged. W_proj is genuine and
    # is NOT touched — the large apparent bias that motivated an earlier pin here traced to the
    # random-effect fallthrough handled at model construction, not to the X reconstruction.
    if "sigma_X" not in results_mcmc:
        for _dct in (results_map, results_mcmc):
            _dct["sigma_X"] = jnp.asarray(1.0)
        print("[legacy] sigma_X absent -> pinned to the fit-time default 1.0 for", model_name)

    # Fixed MAP params (alpha, sigma_t, sigma_c, …) were not sampled by MCMC so
    # they have no chain/draw leading dims. Detect (chains, draws) from a
    # known MCMC-sampled param and broadcast any fixed param to match, so
    # vmap calls inside make_mu_*_mcmc don't fail with shape mismatches.
    _mcmc_leading = None
    for _k in sorted(_mcmc_sampled_keys):
        _v = results_mcmc.get(_k)
        if _v is not None and hasattr(_v, "ndim") and _v.ndim >= 2:
            _mcmc_leading = _v.shape[:2]  # (chains, draws)
            break
    if _mcmc_leading is not None:
        _fixed = {k for k, v in results_map.items()
                  if hasattr(v, "shape") and
                  results_mcmc.get(k, v).shape == v.shape}
        for _k in _fixed:
            if _k in results_mcmc:
                results_mcmc[_k] = jnp.broadcast_to(
                    results_mcmc[_k], _mcmc_leading + results_mcmc[_k].shape
                )

    if _is_naive:
        _c_off_posterior = jnp.transpose(results_mcmc["c_offset"].squeeze(-1), (0, 1, 3, 2))  # (chains, draws, n, k)
        df = posterior_X_to_df(_c_off_posterior, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
        df.to_parquet(os.path.join(model_dir, "posterior_latent_X.parquet"), index=False)
    # Non-naive X export is deferred until after the X_free assembly loop below.

    def _safe_sd(arr, axis=-1):
        sd = jnp.std(arr, axis=axis)
        return jnp.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale_X_samples(X_samples, scale_vec):
        return X_samples * scale_vec[..., None, :]

    def _latent_weight_sd(weight_arr, latent_dim):
        weight_latent = weight_arr[..., :latent_dim, :]
        return _safe_sd(weight_latent, axis=-1)

    def _latent_beta_sd(beta_arr, latent_dim):
        beta_latent = beta_arr[..., :latent_dim, :, :]
        return _safe_sd(beta_latent, axis=-1)

    # Capability gate (was `"linear" in model_name`): the structured-prior models (linear, cosine,
    # RFF) all sample W_proj, so key off its presence rather than the name.
    _is_rff = "rflvm" in model_name
    _supports_modal_exports = (not _is_naive) and ("W_proj" in results_mcmc) and (not _concave_only) and (not _coverage_only)
    if _supports_modal_exports and _is_rff:
        # The shared-kernel RFF model has a SINGLE kernel — there is no separate peak-age/peak-value/
        # curvature latent representation to decompose (those modalities differ only via per-metric
        # weights in the shared 2m-dim feature space, not via the latent geometry). To stay faithful to
        # the model, emit ONE latent rescaled by the per-dimension ARD relevance sqrt(lengthscale): in
        # the kernel the effective input is sqrt(l) ⊙ X (scaled_W = W·sqrt(l)), so sqrt(l_j) is dim j's
        # relevance. For the shared-kernel model the per-modality files are intentionally omitted;
        # latent_space.r skips the per-modality clusterings when they are absent. The SPLIT model
        # (per-modality bandwidths) re-emits them below.
        _ell_sqrt = jnp.sqrt(results_mcmc["lengthscale"])                 # (chains, draws, r)
        _X_rescaled = _scale_X_samples(results_mcmc["X"], _ell_sqrt)
        posterior_X_to_df(
            _X_rescaled, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], []
        ).to_parquet(os.path.join(model_dir, "posterior_latent_X_rescaled.parquet"), index=False)
        _ell_sqrt_map = np.sqrt(np.array(results_map["lengthscale"]))     # (r,)
        _phi_rescaled = np.array(results_map["X"]) * _ell_sqrt_map[None, :]
        _phi_rescaled_df = pd.concat(
            [pd.DataFrame(_phi_rescaled, columns=[f"Dim {i+1}" for i in range(_phi_rescaled.shape[1])]), id_df],
            axis=1,
        )
        _phi_rescaled_df.to_parquet(os.path.join(model_dir, "phi_X_rescaled.parquet"), index=False)

        # Split-RFF model (ConvexMaxSplitRFFTVLinearLVM): each modality has its OWN RFF kernel
        # (frequency draw + ARD bandwidth: lengthscale = curvature, lengthscale_t_max,
        # lengthscale_c_max), so a per-modality latent geometry exists again — sqrt(l_mod) ⊙ X is
        # the effective input of that modality's kernel (the frequency draw is kernel noise).
        # Emit the same per-modality files the linear model emits so latent_space.r's modality
        # archetype clusterings run for this model. Keyed off site presence, not the model name
        # (results_mcmc is back-filled from the MAP samples when the lengthscales are fixed at MCMC).
        if "lengthscale_t_max" in results_mcmc and "lengthscale_c_max" in results_mcmc:
            for _mod_tag, _ls_key in (("peak_age", "lengthscale_t_max"),
                                      ("peak_value", "lengthscale_c_max"),
                                      ("curvature", "lengthscale")):
                _ls_sqrt = jnp.sqrt(jnp.asarray(results_mcmc[_ls_key]))   # (chains, draws, r) or (r,)
                _X_mod = _scale_X_samples(results_mcmc["X"], _ls_sqrt)
                posterior_X_to_df(
                    _X_mod, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], []
                ).to_parquet(
                    os.path.join(model_dir, f"posterior_latent_X_{_mod_tag}.parquet"), index=False
                )
                _ls_sqrt_map = np.sqrt(np.array(results_map[_ls_key]))    # (r,)
                _phi_mod = np.array(results_map["X"]) * _ls_sqrt_map[None, :]
                pd.concat(
                    [pd.DataFrame(_phi_mod, columns=[f"Dim {i+1}" for i in range(_phi_mod.shape[1])]), id_df],
                    axis=1,
                ).to_parquet(os.path.join(model_dir, f"phi_X_{_mod_tag}.parquet"), index=False)
    elif _supports_modal_exports:
        latent_dim = results_map["X"].shape[1]
        if "t_max_raw" in results_mcmc and "c_max" in results_mcmc and "beta" in results_mcmc:
            # Peak age modality (t_max_raw)
            _t_sd = _latent_weight_sd(results_mcmc["t_max_raw"], latent_dim)
            _X_peak_age = _scale_X_samples(results_mcmc["X"], _t_sd)
            _df_peak_age = posterior_X_to_df(
                _X_peak_age, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], []
            )
            _df_peak_age.to_parquet(
                os.path.join(model_dir, "posterior_latent_X_peak_age.parquet"), index=False
            )
            _t_map_sd = np.std(np.array(results_map["t_max_raw"][:latent_dim, :]), axis=-1)
            _phi_peak_age = results_map["X"] * _t_map_sd[None, :]
            _phi_peak_age_df = pd.DataFrame(
                _phi_peak_age, columns=[f"Dim {i+1}" for i in range(_phi_peak_age.shape[1])]
            )
            _phi_peak_age_df = pd.concat([_phi_peak_age_df, id_df], axis=1)
            _phi_peak_age_df.to_parquet(
                os.path.join(model_dir, "phi_X_peak_age.parquet"), index=False
            )

            # Peak value modality (c_max)
            _c_sd = _latent_weight_sd(results_mcmc["c_max"], latent_dim)
            _X_peak_value = _scale_X_samples(results_mcmc["X"], _c_sd)
            _df_peak_value = posterior_X_to_df(
                _X_peak_value, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], []
            )
            _df_peak_value.to_parquet(
                os.path.join(model_dir, "posterior_latent_X_peak_value.parquet"), index=False
            )
            _c_map_sd = np.std(np.array(results_map["c_max"][:latent_dim, :]), axis=-1)
            _phi_peak_value = results_map["X"] * _c_map_sd[None, :]
            _phi_peak_value_df = pd.DataFrame(
                _phi_peak_value, columns=[f"Dim {i+1}" for i in range(_phi_peak_value.shape[1])]
            )
            _phi_peak_value_df = pd.concat([_phi_peak_value_df, id_df], axis=1)
            _phi_peak_value_df.to_parquet(
                os.path.join(model_dir, "phi_X_peak_value.parquet"), index=False
            )

            # Curvature modalities (beta basis functions)
            _beta_sd = _latent_beta_sd(results_mcmc["beta"], latent_dim)  # (chains, draws, latent_dim, M)
            _beta_map_sd = np.std(np.array(results_map["beta"][:latent_dim, :, :]), axis=-1)  # (latent_dim, M)
            _num_basis = _beta_sd.shape[-1]
            for _m in range(_num_basis):
                _m_tag = _m + 1
                _m_sd = _beta_sd[..., _m]
                _X_curv = _scale_X_samples(results_mcmc["X"], _m_sd)
                _df_curv = posterior_X_to_df(
                    _X_curv, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], []
                )
                _df_curv.to_parquet(
                    os.path.join(model_dir, f"posterior_latent_X_curvature_m{_m_tag}.parquet"),
                    index=False,
                )
                _phi_curv = results_map["X"] * _beta_map_sd[:, _m][None, :]
                _phi_curv_df = pd.DataFrame(
                    _phi_curv, columns=[f"Dim {i+1}" for i in range(_phi_curv.shape[1])]
                )
                _phi_curv_df = pd.concat([_phi_curv_df, id_df], axis=1)
                _phi_curv_df.to_parquet(
                    os.path.join(model_dir, f"phi_X_curvature_m{_m_tag}.parquet"), index=False
                )
    if not (_concave_only or _coverage_only):
        _summary_vars = ["sigma_beta", "sigma_beta_binomial", "sigma", "sigma_ar", "sigma_negative_binomial"]
        _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
        summary = az.summary(_summary_subset)
        print(summary)
        summary.to_parquet(os.path.join(model_dir, "posterior_variance_summary.parquet"), index=False)

    # Export per-sample dispersion parameters labelled by metric so model_diagnostics.r
    # can compute posterior log-loss intervals without needing to know index order.
    _disp_rows = []
    if _concave_only or _coverage_only:
        metrics_disp_iter = []
    else:
        metrics_disp_iter = list(zip(metrics, metric_output))
    _g_i = _beta_i = _nb_i = _bb_i = 0
    _disp_map = {
        "gaussian":       ("sigma",                  lambda i: _g_i),
        "beta":           ("sigma_beta",             lambda i: _beta_i),
        "negative-binomial": ("sigma_negative_binomial", lambda i: _nb_i),
        "beta-binomial":  ("sigma_beta_binomial",    lambda i: _bb_i),
    }
    for _mn, _fam in metrics_disp_iter:
        if _fam not in _disp_map:
            continue
        _param_key, _ = _disp_map[_fam]
        if _param_key not in results_mcmc:
            continue
        if _fam == "gaussian":
            _s = np.array(results_mcmc[_param_key])[..., _g_i];  _g_i  += 1
        elif _fam == "beta":
            _s = np.array(results_mcmc[_param_key])[..., _beta_i]; _beta_i += 1
        elif _fam == "negative-binomial":
            _s = np.array(results_mcmc[_param_key])[..., _nb_i];  _nb_i += 1
        elif _fam == "beta-binomial":
            _s = np.array(results_mcmc[_param_key])[..., _bb_i];  _bb_i += 1
        _nc, _nd = _s.shape[:2]
        _ci, _di = np.meshgrid(np.arange(_nc), np.arange(_nd), indexing="ij")
        _disp_rows.append(pd.DataFrame({
            "chain": _ci.ravel(), "draw": _di.ravel(),
            "metric": _mn, "family": _fam,
            "value": _s.ravel(),
        }))
    if _disp_rows:
        pd.concat(_disp_rows, ignore_index=True).to_parquet(
            os.path.join(model_dir, "posterior_dispersion.parquet"), index=False
        )
    # Injury hazard sites: the current models draw a per-type array (injury_exit_raw, scaled by
    # sigma_injury_exit around injury_exit_global_offset); legacy pkls carry the retired factor
    # parameterisation (injury_factor @ injury_exit_loading). Accept either.
    _surv_base_keys = {"gamma_global_log", "exit", "exit_rate", "injury_exit_global_offset"}
    _has_new_exit = {"injury_exit_raw", "sigma_injury_exit"} <= results_mcmc.keys()
    _has_legacy_exit = {"injury_factor", "injury_exit_loading"} <= results_mcmc.keys()
    has_survival_injury = (
        all(key in results_mcmc for key in _surv_base_keys)
        and (_has_new_exit or _has_legacy_exit)
    )

    _, surv_data_set, _ = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year, age_min=age_min, age_max=age_max)
    surv_masks = jnp.stack([data_entity["censored"] for data_entity in surv_data_set], -1)
    Y_surv = jnp.stack([data_entity["observations"] for data_entity in surv_data_set], -1)

    if censor_survival_at_injury:
        _onset = (
            data[data["injury_period"] != "pre-injury"]
            .groupby("id")["age"].min()
        )
        _player_ids = data.groupby("id").apply(lambda g: g["id"].iloc[0]).index.tolist()
        _Y_surv_np = np.array(Y_surv)
        _surv_masks_np = np.array(surv_masks)
        for _i, _pid in enumerate(_player_ids):
            if _pid in _onset.index:
                _onset_age = float(_onset[_pid])
                if _onset_age < _Y_surv_np[_i, 1]:
                    _Y_surv_np[_i, 1] = _onset_age
                    _surv_masks_np[_i, 1] = True
        Y_surv = jnp.array(_Y_surv_np)
        surv_masks = jnp.array(_surv_masks_np)

    # Preserve pre-holdout survival observations for evaluation; then censor
    # holdout players at their last in-sample age so the survival model does
    # not observe exit ages that fall inside the held-out window.
    Y_surv_eval     = Y_surv
    surv_masks_eval = surv_masks
    if os.path.exists(holdout_indices_path):
        _holdout_df_surv = pd.read_csv(holdout_indices_path)
        _Y_surv_h    = np.array(Y_surv)
        _smasks_h    = np.array(surv_masks)
        _id_map_surv = {str(pid): idx for idx, pid in enumerate(id_df["id"].tolist())}
        for _hpid, _hgrp in _holdout_df_surv.groupby("player"):
            _hpi = _id_map_surv.get(str(_hpid))
            if _hpi is None:
                continue
            # Censor at last in-sample age (one year before the first held-out season),
            # but never before the player's entrance age.
            _last_in = max(float(_hgrp["age"].min()) - 1.0, float(_Y_surv_h[_hpi, 0]))
            if _last_in < float(_Y_surv_h[_hpi, 1]):
                _Y_surv_h[_hpi, 1] = _last_in
                _smasks_h[_hpi, 1] = True
        Y_surv     = jnp.array(_Y_surv_h)
        surv_masks = jnp.array(_smasks_h)

    # ── Model instance + curve args: SINGLE source for log-posterior AND MAP/MCMC curves ─────
    # Built via the shared dispatch (same as build_inference_inputs / main.py) — fixes the prior
    # silent fallthrough that built plain-linear for lkj/cosine, and sets the RE flags. Curves are
    # reconstructed by running lp_model.compute_curves under numpyro.handlers.substitute (single
    # source of truth) instead of the retired make_mu_* duplicates.
    _n_players = covariate_X.shape[0]
    _output_shape = (_n_players, len(basis), len(metrics))
    _pk = cfg.get("prior_knobs") or {}
    lp_model = dispatch_model(
        model_name, latent_rank=basis_dims, output_shape=_output_shape, basis=basis,
        player_covariates=obs_covariates, injury=injury,
        num_injury_types=int(data["injury_code"].max()), prior_knobs=_pk,
        rff_dim=approx_x_dim)   # RFF leaves need rff_dim to size W / projected features
    for _ak in _ATTRIBUTE_KNOBS:
        if _ak in _pk:
            setattr(lp_model, _ak, _build_knob_value(_pk[_ak]))
    # Curve random effects must reflect HOW THE MODEL WAS FIT, not the dispatch default. The
    # non-injury default turns c/t-offset and curve REs on; if the fitted samples carry no such
    # site, compute_curves' _resolve_prior falls through to the PRIOR and draws a fresh
    # N(0,1) offset for every (player, metric) — silently injecting random per-player shifts into
    # every exported prediction. Older fits (e.g. the nba_tvlinearlvm* ablations) predate the REs
    # and hit exactly this path. Gate each RE on the presence of its site in the samples.
    for _re_flag, _re_site in (("use_c_offset_re", "c_offset_re"),
                               ("use_t_offset_re", "t_offset_re"),
                               ("use_curve_re", "curve_re")):
        if getattr(lp_model, _re_flag, False) and _re_site not in results_mcmc:
            setattr(lp_model, _re_flag, False)
            print(f"[export] {_re_site} absent from samples -> {_re_flag}=False (fit had no RE)")
    # --peaks_no_re: substitute ZEROS for the fitted random effects so compute_curves returns the
    # shared/archetypal curve f^k(X_p, t). The sites stay "present" (so the flags above keep the
    # RE code path live and the substitution actually bites) but contribute nothing.
    if _peaks_no_re:
        for _re_site in ("c_offset_re", "t_offset_re", "curve_re"):
            for _dct in (results_map, results_mcmc):
                if _re_site in _dct:
                    _dct[_re_site] = jnp.zeros_like(jnp.asarray(_dct[_re_site]))
        print("[export] --peaks_no_re: c/t_offset_re and curve_re zeroed -> archetypal peaks")
    lp_model.initialize_priors(scale_values=scale_values)
    apply_prior_knobs(lp_model, _pk, metrics=metrics)

    _first_obs_year = int(data.query("id != 99999999")["year"].min())
    _ref_year_idx = _first_obs_year - min_year
    _all_idx = jnp.arange(_n_players)
    _lp_offsets = {
        **offset_dict,
        "exit_times": Y_surv[:, 1] - age_min + 1e-6,
        "entrance_times": Y_surv[:, 0] - age_min + 1e-6,
        "right_censor": surv_masks[:, 1],
        "injury_indicator": injury_masks,
        "injury_type": injury_types,
    }
    # Positional args for compute_curves / _compute_mu (shared by MAP + MCMC reconstruction).
    _sample_free = jnp.array(player_indices)
    _sample_fixed = jnp.setdiff1d(_all_idx, _sample_free, assume_unique=True)
    _ar_metric_idx = jnp.where(jnp.array(de_trend_indices))[0]
    _curve_args = (hsgp_params, _lp_offsets, _sample_free, _sample_fixed,
                   _ar_metric_idx, year_indices, num_years, len(de_trend_metrics), _ref_year_idx)

    # ── Log posterior via numpyro.infer.util.log_density ─────────────────────
    _lp_path = os.path.join(model_dir, "log_posterior.parquet")
    try:
        _lp_model = lp_model
        _lp_kwargs = {
            "data_set": data_dict,
            "inference_method": "mcmc",
            "sample_free_indices": _sample_free,
            "sample_fixed_indices": _sample_fixed,
            "hsgp_params": hsgp_params,
            "offsets": _lp_offsets,
            "ar_metric_indices": _ar_metric_idx,
            "year_indices": year_indices,
            "num_years": num_years,
            "num_de_trend": len(de_trend_metrics),
            "ref_year_idx": _ref_year_idx,
        }
        if _concave_only or _coverage_only:
            print("log posterior skipped (--concave_only/--coverage_only)")
        elif _mcmc_leading is not None:
            _nc, _nd = _mcmc_leading
            # Identify the latent (non-observed) sample sites the model actually uses
            # so we don't pass MAP-derived or computed keys that collide with
            # observed/deterministic sites and corrupt the log density.
            _trace = numpyro.handlers.trace(
                numpyro.handlers.seed(_lp_model.model_fn, 0)
            ).get_trace(**_lp_kwargs)
            _sample_sites = {
                k for k, v in _trace.items()
                if v["type"] == "sample" and not v.get("is_observed", False)
            }
            _flat_s = {
                k: v.reshape(-1, *v.shape[2:]) if (hasattr(v, "ndim") and v.ndim >= 2) else v
                for k, v in results_mcmc.items()
                if k in _sample_sites
            }
            # results_map carries trained param-site values (W, lengthscale, …).
            # Merging here ensures log_density uses the actual trained values for
            # those sites rather than default init values, while MCMC samples in
            # p override for any overlapping sample-site keys.
            _lp_batch = 10
            _n_flat = next(iter(_flat_s.values())).shape[0]
            _lj_parts = []
            for _bi in range(0, _n_flat, _lp_batch):
                _bs = {k: v[_bi:_bi + _lp_batch] for k, v in _flat_s.items()}
                _lj_parts.append(jax.vmap(
                    lambda p: _log_density(_lp_model.model_fn, (), _lp_kwargs, {**results_map, **p})[0]
                )(_bs))
            _lj = jnp.concatenate(_lj_parts)
            _lp_arr = np.array(_lj).reshape(_nc, _nd)
            pd.DataFrame(
                [{"chain": c, "draw": d, "log_joint": float(_lp_arr[c, d])}
                 for c in range(_nc) for d in range(_nd)]
            ).to_parquet(_lp_path, index=False)
            print(f"Saved log_posterior.parquet ({_nc} chains × {_nd} draws)")
        else:
            print("Warning: log posterior skipped — no MCMC leading dims detected")
    except Exception as _lp_exc:
        import traceback
        print(f"Warning: log posterior computation failed: {_lp_exc}")
        traceback.print_exc()
    # ─────────────────────────────────────────────────────────────────────────

    for item in results_mcmc:
        print(item, getattr(results_mcmc[item], "shape", "(scalar)"))
        if item == "X":
            if "X_free" in results_mcmc:
                X_new = jnp.tile(results_mcmc["X"][None, None], (1, 50, 1, 1))
                X_new = X_new.at[..., jnp.array(player_indices), :].set(results_mcmc["X_free"])
                results_mcmc["X"] = X_new
            if "hsgp" in model_name:
                results_mcmc["X"] = jnp.tanh(results_mcmc["X"]) * 1.9

    # For linear models: reconstruct total X = Z @ W_proj + sigma_X * X_raw (non-centered).
    # X_loc (prior mean from covariates) is also exported for interpretability.
    # Naive model has no latent X; X_map_aug / X_mcmc_aug are left as None.
    # --concave_only: skipped — X_map_aug/X_mcmc_aug feed only the post-exit exports.
    if not _is_naive and not _concave_only:
        _x_was_sampled = "X" in _mcmc_sampled_keys or "X_free" in _mcmc_sampled_keys
        # Capability gate (was `"linear" in model_name and ...`): any structured-prior model with a
        # sampled W_proj (linear, cosine, RFF leaves) gets the total-X + X_loc reconstruction path.
        if "W_proj" in results_mcmc:
            _Z = obs_covariates                                      # (n, 2)
            # MAP total X
            _W_map   = results_map["W_proj"]                         # (2, r)
            # Legacy runs predate the sigma_X site; the model classes default sigma_X->1.0 when
            # absent (models.py "structured prior for X" note) — mirror that here.
            _sX_map  = results_map.get("sigma_X", jnp.asarray(1.0))  # scalar
            _X_loc_map = _Z @ _W_map                                 # (n, r)
            _X_raw_map = jnp.zeros((_Z.shape[0], basis_dims))
            _free_raw_map = results_map.get("X_free", results_map.get("X"))
            if _free_raw_map is not None:
                if len(player_indices) > 0:
                    _X_raw_map = _X_raw_map.at[jnp.array(player_indices, dtype=jnp.int32)].set(_free_raw_map)
                else:
                    _X_raw_map = _free_raw_map
            X_map_aug = _X_loc_map + _sX_map * _X_raw_map            # (n, r)

            # MCMC total X — results_mcmc["X"] now contains assembled X_raw (chains, draws, n, r)
            _W_mc  = results_mcmc["W_proj"]                          # (chains, draws, 2, r) or (2, r)
            _sX_mc = results_mcmc.get("sigma_X", jnp.asarray(1.0))   # (chains, draws) or scalar
            _X_loc_mc = jnp.einsum("...pr,np->...nr", _W_mc, _Z)    # (chains, draws, n, r)
            X_mcmc_aug = _X_loc_mc + _sX_mc[..., None, None] * results_mcmc["X"]  # (chains, draws, n, r)
            # NOTE: do NOT overwrite results_mcmc["X"] — the curve reconstruction (compute_curves under
            # substitute) reads the RAW X site and rebuilds total X = Z@W_proj + sigma_X*X internally.
            # X_mcmc_aug (total X) is kept as a local for the latent-X parquet + downstream exports.

            # Export total X to parquet (non-naive, linear path)
            df = posterior_X_to_df(X_mcmc_aug, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
            df.to_parquet(os.path.join(model_dir, "posterior_latent_X.parquet"), index=False)

            # Export X_loc (covariate prior mean) separately
            df_loc = posterior_X_to_df(_X_loc_mc, id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
            df_loc.to_parquet(os.path.join(model_dir, "posterior_X_loc.parquet"), index=False)
        else:
            # Non-linear (rflvm/hsgp) or model without W_proj: keep old augmentation
            X_map_aug = jnp.concatenate([results_map["X"], obs_covariates], axis=-1)
            _obs_bc = jnp.broadcast_to(
                obs_covariates[None, None],
                results_mcmc["X"].shape[:-1] + (obs_covariates.shape[-1],),
            )
            X_mcmc_aug = jnp.concatenate([results_mcmc["X"], _obs_bc], axis=-1)

            # Export raw latent X (non-naive, non-linear path)
            df = posterior_X_to_df(results_mcmc["X"], id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
            df.to_parquet(os.path.join(model_dir, "posterior_latent_X.parquet"), index=False)
    else:
        X_map_aug = None
        X_mcmc_aug = None

    # ── Survival latent representation (shared by the injury AND non-injury survival exports) ────
    # The survival forward must use the SAME latent representation the model's _survival_rates uses.
    # For linear/cosine the exit weights act on the r-dim latent directly. For RFF the exit weights
    # are sized to the 2m-dim projected feature map, so project X_mcmc_aug through the sampled
    # W / lengthscale to the norm-1 RFF features (matching _project_X) and tell the survival utils
    # the kernel self-cov is 1 (||phi||^2 = 1) rather than the feature width.
    # Hoisted here so the injury survival calls below use it too — passing the raw r-dim latent to
    # the RFF injury export is what produced the einsum "size of label 'r' (10) vs (100)" failure.
    _X_surv, _surv_kcov = X_mcmc_aug, None
    if X_mcmc_aug is not None and "rflvm" in model_name and "W" in results_mcmc:
        _Wm = results_mcmc["W"]                                   # (..., m, r)
        _lsm = jnp.asarray(results_mcmc["lengthscale"])           # (..., r)
        _wTx = jnp.einsum("...nr,...mr->...nm", X_mcmc_aug, _Wm * jnp.sqrt(_lsm)[..., None, :])
        _X_surv = jnp.concatenate([jnp.cos(_wTx), jnp.sin(_wTx)], axis=-1) / jnp.sqrt(_Wm.shape[-2])
        _surv_kcov = 1.0

    # ── Curve reconstruction via the model's own forward (single source) ─────────────────────
    # _curves_under_substitute runs lp_model.compute_curves under numpyro substitute so the exported
    # curve == the fitted model's forward exactly (incl. cosine/LKJ/curve_amp/1-over-sqrt-r); the AR
    # is added separately via _compute_player_ar (zero for non-AR).
    def _curves_under_substitute(params):
        def f():
            d = dict(lp_model.compute_curves(*_curve_args, include_derivs=not (_concave_only or _coverage_only), include_loadings=not _coverage_only))
            d["ar"] = lp_model._compute_player_ar()
            return d
        return numpyro.handlers.substitute(numpyro.handlers.seed(f, jax.random.PRNGKey(0)), data=params)()

    if _is_naive:
        _c_off_map = results_map["c_offset"]    # (k, n, 1)
        _s_map     = results_map["sigma_ar"]    # (k, 1)
        _r_map     = results_map["rho_ar"]      # (k, 1)
        _z_map     = results_map["beta_ar"]     # (j, k, n)
        _a0_map    = results_map["AR_0"] * (_s_map / jnp.sqrt(1 - _r_map ** 2))
        _ar_map    = _ARLinearLVM._compute_ar_process_from_parameters(_s_map, _r_map, _z_map, _a0_map)
        mu = jnp.repeat(_c_off_map, repeats=len(basis), axis=-1) + _ar_map  # (k, n, j)
    else:
        _d_map = _curves_under_substitute(results_map)
        mu = _d_map["mu"]   # (k, n, j) — calendar trend added below
    if "intercept" in results_map:
        mu += (results_map["intercept"] * results_map["sigma_intercept"])[..., None]

    # Reconstruct MAP calendar-year TREND_AR
    _year_ar_keys = ("beta_year_ar", "sigma_year_ar", "rho_year_ar", "AR_0_year")
    _has_year_ar = len(de_trend_metrics) > 0 and all(k in results_map for k in _year_ar_keys)
    _ar_global_indices = jnp.array([i for i, f in enumerate(de_trend_indices) if f])
    _num_ar = int(jnp.sum(jnp.array(de_trend_indices)))

    def _build_trend_ar_map(params):
        """Reconstruct (k, n, j) TREND_AR from MAP parameter dict."""
        _s = params["sigma_year_ar"]                          # (num_ar, 1)
        _r = params["rho_year_ar"]                            # (num_ar, 1)
        _z = params["beta_year_ar"]                           # (num_years, num_ar)
        _a = params["AR_0_year"] * _s[None, :, 0]            # (1, num_ar)
        trend = _ARLinearLVM._compute_ar1_calendar_process(_s, _r, _z, _a)  # (num_ar, num_years)
        trend_nj = trend[:, year_indices]                     # (num_ar, n, j)
        out = jnp.zeros((len(metrics),) + year_indices.shape)
        return out.at[_ar_global_indices].set(trend_nj)       # (k, n, j)

    if _has_year_ar:
        TREND_AR_map = _build_trend_ar_map(results_map)
    else:
        TREND_AR_map = de_trend_adjusted

    mu += TREND_AR_map
    if not (_concave_only or _coverage_only):
        obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)

        # `sigma` is absent when the panel has no gaussian head (see _disp below) -- default it
        # like the sigma_negative_binomial / sigma_beta arguments beside it.
        avg_sd, autocorr, lognormal_params, beta_params = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, metrics, results_map.get("sigma", 1.0), results_map.get("sigma_negative_binomial", 0),
                                                                    results_map.get("sigma_beta_binomial", 0), results_map.get("sigma_beta", 1))
    
    # avg_sd = jnp.ones((len(metrics))) * .01
    # autocorr = jnp.zeros_like(avg_sd)

    if _is_naive:
        # Naive has no convex/GPLVM curve — just per-player c_offset + AR(1), reconstructed inline.
        def _naive_one_draw(c_off, s, r, z, a0_raw):
            a0 = a0_raw * (s / jnp.sqrt(1 - r ** 2))
            ar = _ARLinearLVM._compute_ar_process_from_parameters(s, r, z, a0)   # (k, n, j)
            return jnp.repeat(c_off, repeats=z.shape[0], axis=-1), ar           # (k, n, j) each
        mu_mcmc, AR = vmap(vmap(_naive_one_draw))(
            results_mcmc["c_offset"], results_mcmc["sigma_ar"], results_mcmc["rho_ar"],
            results_mcmc["beta_ar"], results_mcmc["AR_0"])
        tmax_mcmc = None; cmax_mcmc = None
        second_deriv = None; third_deriv = None; first_deriv = jnp.zeros_like(mu_mcmc)
        TREND_AR_mcmc = de_trend_adjusted
    else:
        # SINGLE SOURCE: reconstruct every MCMC curve through the model's own forward. lax.map runs
        # one draw at a time (memory-safe — never materialises (chains,draws,k,n,t)); compute_curves
        # gives mu/peaks/derivs/calendar-trend and _compute_player_ar gives the per-player AR. This
        # replaces make_mu_linear* / make_mu_tvlinearlvm_mcmc and the per-family branches, so any
        # models.py change (cosine, 1/sqrt(r), curve_amp, …) propagates here automatically.
        _nc, _nd = _mcmc_leading
        _flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in results_mcmc.items() if k in _mcmc_sampled_keys}
        _keys = jax.random.split(jax.random.PRNGKey(0), _nc * _nd)
        def _one_draw(carry):
            draw, key = carry
            def f():
                d = dict(lp_model.compute_curves(*_curve_args, include_derivs=not (_concave_only or _coverage_only), include_loadings=not _coverage_only))
                if _concave_only:
                    # Return only what the concave block needs — XLA then dead-code-eliminates
                    # the mu/core-tensor einsums, so each draw's forward is nearly free.
                    return {k: d[k] for k in ("curve_loadings", "t_max") if k in d}
                d["ar"] = lp_model._compute_player_ar()
                return d
            return numpyro.handlers.substitute(numpyro.handlers.seed(f, key), data={**results_map, **draw})()
        _d_mc = jax.lax.map(_one_draw, (_flat, _keys))
        _d_mc = {k: v.reshape(_nc, _nd, *v.shape[1:]) for k, v in _d_mc.items()}
        if not _concave_only:
            mu_mcmc = _d_mc["mu"]; tmax_mcmc = _d_mc["t_max"]; cmax_mcmc = _d_mc["c_max"]
            AR = _d_mc["ar"]; TREND_AR_mcmc = _d_mc["trend_ar"]
            if _coverage_only:
                first_deriv = second_deriv = third_deriv = None
            else:
                first_deriv = _d_mc["first_deriv"]
                second_deriv = _d_mc["second_deriv"]; third_deriv = _d_mc["third_deriv"]

    if not _concave_only:
        latent_val = mu_mcmc + AR + TREND_AR_mcmc
        if _is_naive:
            _peak_idx = jnp.argmax(latent_val, axis=-1)                                     # (chains, draws, k, n)
            tmax_mcmc = jnp.swapaxes(jnp.array(basis)[_peak_idx] - basis.mean(), -1, -2)   # (chains, draws, n, k)
            cmax_mcmc = jnp.swapaxes(jnp.max(latent_val, axis=-1), -1, -2)                 # (chains, draws, n, k)

    # ── Concave loadings (curvature-root HSGP-basis loadings) ────────────────────────────────
    # gamma[n,k,l] = psi(x_n)^T (beta·sqrt(spd))[:,:,k] / sqrt(kernel_self_cov): the curve's second
    # derivative is -g(t)^2 with g(t) = sum_l gamma_l psi_l(t), so gamma is the loading of the
    # (player, metric) curvature root on orthonormal HSGP time basis l and sum_l gamma_l^2 is the
    # curvature energy. Only the convex-max family emits the key (capability gate). gamma is
    # identified up to a whole-vector sign flip per (metric, draw), so the identified curve-space
    # object is the Gram matrix S_k = mean_n gamma gamma^T: its diagonal gives the per-basis
    # loadings; its eigen-decomposition the canonical concave curves per metric. When use_curve_re=1,
    # curve_amp scales the quadratic form (not gamma): energy shares are invariant, RMS magnitudes
    # exclude it (the knob is off for all tvrflvm/tvlinearlvm configs).
    if not _is_naive and "curve_loadings" in _d_mc:
        _real = (id_df["id"] != "99999999").to_numpy()                  # drop the fake grid player
        _gam = np.asarray(_d_mc["curve_loadings"])[:, :, _real]         # (chains, draws, n_real, k, M)
        _nc_g, _nd_g, _n_real, _k_g, _M_g = _gam.shape
        _L_t = float(jnp.squeeze(hsgp_params["L_time"]))
        _phi_time_np = np.asarray(hsgp_params["phi_x_time"])            # (t, M, M)
        _x_time_np = np.asarray(jnp.squeeze(x_time))                    # (t,) centered ages
        _ages_g = np.arange(age_min, age_max + 1)
        _metric_arr = np.array(list(metrics))

        _S = np.einsum("cdnkl,cdnkz->cdklz", _gam, _gam) / _n_real      # (c, d, k, M, M) Gram
        _diag = np.einsum("cdkll->cdkl", _S)                            # mean_n gamma^2
        _share = _diag / (_diag.sum(axis=-1, keepdims=True) + 1e-12)
        _mean_g = _gam.mean(axis=2)                                     # (c, d, k, M)
        _sgn = np.sign(np.take_along_axis(_mean_g, np.argmax(np.abs(_mean_g), axis=-1)[..., None], axis=-1))
        _mean_signed = _mean_g * np.where(_sgn == 0, 1.0, _sgn)         # whole-vector flip per (c, d, k)

        _ci, _si, _ki, _li = np.meshgrid(np.arange(_nc_g), np.arange(_nd_g), np.arange(_k_g), np.arange(_M_g), indexing="ij")
        pd.DataFrame({
            "chain": _ci.ravel(), "sample": _si.ravel(),
            "metric": _metric_arr[_ki.ravel()], "basis": _li.ravel() + 1,
            "loading_rms": np.sqrt(_diag).ravel(),
            "energy_share": _share.ravel(),
            "loading_mean_signed": _mean_signed.ravel(),
        }).to_parquet(os.path.join(model_dir, "posterior_concave_loadings.parquet"), index=False)

        _ci, _si, _ki, _li, _zi = np.meshgrid(np.arange(_nc_g), np.arange(_nd_g), np.arange(_k_g), np.arange(_M_g), np.arange(_M_g), indexing="ij")
        pd.DataFrame({
            "chain": _ci.ravel(), "sample": _si.ravel(),
            "metric": _metric_arr[_ki.ravel()],
            "basis_l": _li.ravel() + 1, "basis_z": _zi.ravel() + 1,
            "value": _S.ravel(),
        }).to_parquet(os.path.join(model_dir, "posterior_concave_gram.parquet"), index=False)

        _gam_map = np.asarray(_d_map["curve_loadings"])[_real]          # (n_real, k, M)
        _diag_map = np.einsum("nkl,nkl->kl", _gam_map, _gam_map) / _n_real
        _mean_map = _gam_map.mean(axis=0)
        _sgn_map = np.sign(np.take_along_axis(_mean_map, np.argmax(np.abs(_mean_map), axis=-1)[..., None], axis=-1))
        _ki, _li = np.meshgrid(np.arange(_k_g), np.arange(_M_g), indexing="ij")
        pd.DataFrame({
            "metric": _metric_arr[_ki.ravel()], "basis": _li.ravel() + 1,
            "loading_rms": np.sqrt(_diag_map).ravel(),
            "energy_share": (_diag_map / (_diag_map.sum(axis=-1, keepdims=True) + 1e-12)).ravel(),
            "loading_mean_signed": (_mean_map * np.where(_sgn_map == 0, 1.0, _sgn_map)).ravel(),
        }).to_parquet(os.path.join(model_dir, "map_concave_loadings.parquet"), index=False)

        # Canonical eigen-curves per metric: S̄_k = V diag(λ) V^T on the posterior-mean Gram (the
        # sign gauge cancels in gamma gamma^T, so averaging draws/chains is valid). This gives
        # E_n[f''_k(t)] = -sum_j λ_j (v_j^T psi(t))^2 — an exact rank-M decomposition of the
        # population-mean curvature into canonical concave components with loadings λ_j.
        # Deliberately PER METRIC: the S̄_k do not commute (median normalized commutator ~0.19), so
        # no common eigenbasis exists, and "component j" names a different curve for every metric —
        # cross-metric comparisons go through the curve shapes, never the component index. (A shared
        # oblique basis via INDSCAL was tried and reverted: it fits, but the shapes that make each
        # metric's curvature interesting are exactly what sharing averages away.)
        _S_bar = _S.mean(axis=(0, 1))                                   # (k, M, M)
        _evecs = np.linalg.eigh(_S_bar)[1][:, :, ::-1]                  # columns v_j, descending λ
        _vsgn = np.sign(np.take_along_axis(_evecs, np.argmax(np.abs(_evecs), axis=1)[:, None, :], axis=1))
        _evecs = _evecs * np.where(_vsgn == 0, 1.0, _vsgn)

        # Per-draw loadings on the FIXED posterior-mean eigenvectors (>= 0: each draw's S_k is PSD)
        _lam = np.einsum("klj,cdklz,kzj->cdkj", _evecs, _S, _evecs)     # (c, d, k, M)

        # Per-metric GP variance alpha, exported alongside lambda because magnitude comparisons
        # ACROSS metrics require lambda_j / alpha_k. spd = sqrt(S(w)) and S(w) = alpha*sqrt(2pi)*l*
        # exp(-l^2 w^2/2), so gamma ~ sqrt(alpha) and every lambda_j scales linearly in alpha_k —
        # while the eigenvectors are untouched (a per-metric scalar cannot rotate S_k). alpha
        # absorbs the per-metric LINK scale (Poisson/Binomial/Normal), so raw lambda is not
        # comparable across metrics; alpha-normalized lambda is. Only the variance parameter is
        # divided out: the lengthscale part of the spd (the l prefactor and the exp(-l^2 w^2/2)
        # decay) stays in, since that carries the frequency structure rather than a nuisance scale.
        # alpha may be sampled (c,d,k,1) or fixed per-metric (k,)/scalar — normalize to (c,d,k).
        _alpha_src = results_mcmc.get("alpha")
        if _alpha_src is None:
            _alpha_src = lp_model.prior.get("alpha")
        _alpha_k = np.squeeze(np.asarray(_alpha_src, dtype=float))
        if _alpha_k.ndim < 3:                                           # fixed scalar or per-metric
            _alpha_k = np.broadcast_to(_alpha_k, (_nc_g, _nd_g, _k_g))
        _alpha_full = np.broadcast_to(_alpha_k[..., None], (_nc_g, _nd_g, _k_g, _M_g))

        _ci, _si, _ki, _ji = np.meshgrid(np.arange(_nc_g), np.arange(_nd_g), np.arange(_k_g), np.arange(_M_g), indexing="ij")
        pd.DataFrame({
            "chain": _ci.ravel(), "sample": _si.ravel(),
            "metric": _metric_arr[_ki.ravel()], "component": _ji.ravel() + 1,
            "loading": _lam.ravel(),
            "alpha": _alpha_full.ravel(),
            "loading_alpha_norm": (_lam / _alpha_full).ravel(),
            "energy_share": (_lam / (_lam.sum(axis=-1, keepdims=True) + 1e-12)).ravel(),
        }).to_parquet(os.path.join(model_dir, "posterior_concave_canonical_loadings.parquet"), index=False)

        # Canonical descent curves through the model's own max form, anchored at each metric's
        # posterior-mean peak age: v_j^T [Ψ(t̄*) − Ψ(t) + Ψ'(t̄*)(t − t̄*)] v_j (unit v_j — scale by
        # λ_j/α_k for the component's contribution), plus each component's curvature root
        # g_j(t) = v_j^T psi(t).
        _tbar = np.asarray(_d_mc["t_max"])[:, :, _real].mean(axis=(0, 1, 2))  # (k,) centered peak age
        _phi_tbar = np.asarray(jax.vmap(lambda t: make_convex_phi(t, _L_t, M_time))(jnp.asarray(_tbar)))
        _phi_p_tbar = np.asarray(jax.vmap(lambda t: make_convex_phi_prime(t, _L_t, M_time))(jnp.asarray(_tbar)))
        _core = (_phi_tbar[:, None] - _phi_time_np[None]
                 + _phi_p_tbar[:, None] * (_x_time_np[None, :] - _tbar[:, None])[..., None, None])  # (k, t, M, M)
        _curves = np.einsum("klj,ktlz,kzj->kjt", _evecs, _core, _evecs)  # (k, M, t) descent <= 0
        _psi_g = np.asarray(eigenfunctions(jnp.asarray(_x_time_np), _L_t, M_time))                  # (t, M)
        _g = np.einsum("klj,tl->kjt", _evecs, _psi_g)
        _ki, _ji, _ti = np.meshgrid(np.arange(_k_g), np.arange(_M_g), np.arange(len(_ages_g)), indexing="ij")
        pd.DataFrame({
            "metric": _metric_arr[_ki.ravel()], "component": _ji.ravel() + 1,
            "age": _ages_g[_ti.ravel()],
            "curve_value": _curves.ravel(), "g_value": _g.ravel(),
        }).to_parquet(os.path.join(model_dir, "concave_canonical_curves.parquet"), index=False)

        # Player canonical-weight profiles: the FULL outer product of the projections
        # c_pj = v_j^T gamma_p in each metric's fixed eigenbasis, posterior-averaged with the same
        # per-draw /alpha as the population loadings: P[n,k,l,z] = E_draws[c_l c_z / alpha_k].
        # The diagonal is the player's curvature energy on each canonical component
        # (mean_n over the diagonal = lambda_j exactly); the off-diagonals are the cross-term
        # (interference) weights, since gamma^T [.] gamma = sum_{l,z} c_l c_z v_l^T [.] v_z —
        # together the matrix is the player's complete curvature expansion in the canonical basis.
        # Cross terms average to ~zero over players (the eigenbasis diagonalizes the population
        # Gram), so a player's off-diagonal structure is exactly how they deviate from a
        # population-canonical mixture. Gauge: the per-(metric, draw) gamma sign flip hits c_l and
        # c_z together, so pairwise products are invariant; eigenvector signs are pinned above and
        # cross-term signs are relative to that convention.
        _c_play = np.einsum("klj,cdnkl->cdnkj", _evecs, _gam)           # (c, d, n_real, k, M)
        _P_play = np.einsum("cdnkl,cdnkz,cdk->nklz", _c_play, _c_play,
                            1.0 / _alpha_k) / (_nc_g * _nd_g)           # (n, k, M, M)
        _ids_real = id_df["id"].to_numpy()[_real]
        _names_real = id_df["name"].to_numpy()[_real]
        _ni, _ki, _li, _zi = np.meshgrid(np.arange(_n_real), np.arange(_k_g), np.arange(_M_g), np.arange(_M_g), indexing="ij")
        pd.DataFrame({
            "id": _ids_real[_ni.ravel()], "name": _names_real[_ni.ravel()],
            "metric": _metric_arr[_ki.ravel()],
            "comp_row": _li.ravel() + 1, "comp_col": _zi.ravel() + 1,
            "value": _P_play.ravel(),
        }).to_parquet(os.path.join(model_dir, "posterior_concave_player_profile.parquet"), index=False)

        # Fixed shared basis curves (concave curve generated by g = psi_l alone: −Ψ_ll(t)) and the
        # eigenfunction shapes themselves — so basis indices in the loadings are interpretable
        # (basis 1 = lowest frequency).
        _li, _ti = np.meshgrid(np.arange(_M_g), np.arange(len(_ages_g)), indexing="ij")
        pd.DataFrame({
            "basis": _li.ravel() + 1, "age": _ages_g[_ti.ravel()],
            "value": (-np.diagonal(_phi_time_np, axis1=1, axis2=2).T).ravel(),
        }).to_parquet(os.path.join(model_dir, "concave_basis_curves.parquet"), index=False)
        pd.DataFrame({
            "basis": _li.ravel() + 1, "age": _ages_g[_ti.ravel()],
            "value": _psi_g.T.ravel(),
        }).to_parquet(os.path.join(model_dir, "hsgp_time_basis.parquet"), index=False)
        print("exported concave loadings / canonical curves")

    if _concave_only:
        print("--concave_only: done, skipping all remaining exports")
        raise SystemExit(0)

    if injury:
        # Static (n, j) mask of injured player-seasons and its nonzero index — MUST match the
        # np.nonzero(row-major) order models.py uses to lay out injury_resid_raw's columns.
        _inj_nj = np.asarray(injury_types[0])            # (n, j) type codes, same across metrics
        _idx_n, _idx_j = np.nonzero(_inj_nj > 0)         # (S,) each — injured player-seasons
        _inj_type_s = _inj_nj[_idx_n, _idx_j].astype(int)  # (S,) type code per injured season

        _new_struct = "injury_raw" in results_mcmc
        if _new_struct:
            # ── Current structure: full (k, i) metric x type array (global mean + type deviation).
            # Mean includes the global offset (unlike the legacy export, which wrote the factor
            # product without it) so posterior_injury_prior_mean is the TOTAL type-level effect.
            _injury_global_offset = results_mcmc["injury_global_offset"]   # (c, d, k)
            _sigma_injury = results_mcmc["sigma_injury"]                   # (c, d, k)
            injury_mean_prior = (
                _injury_global_offset[..., None]
                + _sigma_injury[..., None] * results_mcmc["injury_raw"]
            )  # (c, d, k, i)
            injury_effect_raw = injury_mean_prior[:, :, :, None, None, :]  # (c, d, k, 1, 1, i)
            # The per-player-season residual was dropped (unidentified). Older pkls that still carry
            # it get it added back; current pkls have no such site, so the effect is purely the
            # (metric, type) mean, constant across a type's player-seasons.
            _injury_resid = (
                results_mcmc["sigma_injury_resid"][..., None] * results_mcmc["injury_resid_raw"]
                if "injury_resid_raw" in results_mcmc else None
            )  # (c, d, k, S) or None
            # Decline-acceleration variant: per-year (metric, type) slope in time-since-onset.
            _slope_mean = (
                results_mcmc["injury_slope_global_offset"][..., None]
                + results_mcmc["sigma_injury_slope"][..., None] * results_mcmc["injury_slope_raw"]
            ) if "injury_slope_raw" in results_mcmc else None  # (c, d, k, i) or None
        else:
            # ── Legacy factor pkls (incl. the decay model) ──
            injury_loading = results_mcmc["injury_loading"]
            injury_factor = results_mcmc["injury_factor"]
            _factor_mean = jnp.einsum("...ip, ...kp -> ...ki", injury_factor, injury_loading)
            _injury_global_offset = results_mcmc.get("injury_global_offset", jnp.zeros(_factor_mean.shape[:-1]))
            injury_mean_prior = _factor_mean + _injury_global_offset[..., None]  # (c, d, k, i)
            _sigma_injury = results_mcmc.get("sigma_injury")       # (c, d, k) or None
            _injury_time_raw = results_mcmc.get("injury_time_raw") # (c, d, j, i) or None
            if _sigma_injury is not None and _injury_time_raw is not None:
                injury_effect_raw = (
                    injury_mean_prior[:, :, :, None, None, :]
                    + _sigma_injury[:, :, :, None, None, None] * _injury_time_raw[:, :, None, None, :, :]
                )  # (c, d, k, 1, j, i)
            else:
                injury_effect_raw = injury_mean_prior[:, :, :, None, None, :]  # decay fallback
            _injury_resid = None
            _slope_mean = None

        injury_effect_padded = jnp.concatenate(
            [jnp.zeros(injury_effect_raw.shape[:-1] + (1,), dtype=injury_effect_raw.dtype),
             injury_effect_raw],
            axis=-1
        )  # (..., k, 1, T, i+1) — take_along_axis broadcasts over n
        injury_effect = jnp.take_along_axis(injury_effect_padded, injury_types[..., None][None, None], -1).squeeze(-1)
        if _injury_resid is not None:
            # scatter the residual onto its (n, j) cells — same index order as the model's forward
            injury_effect = injury_effect.at[:, :, :, _idx_n, _idx_j].add(_injury_resid)
        # Slope variant: add tau_slope[k, type] * (t - t0) on injured cells, matching the forward.
        _delta_nj = None
        if _slope_mean is not None:
            _inj_ind_nj = np.asarray(injury_masks[0]) if np.asarray(injury_masks).ndim == 3 else np.asarray(injury_masks)
            _t0_n = np.argmax(_inj_ind_nj.astype(float), axis=-1)                                   # (n,)
            _delta_nj = np.maximum(np.arange(_inj_nj.shape[1])[None, :] - _t0_n[:, None], 0.0)      # (n, j)
            _slope_padded = jnp.concatenate(
                [jnp.zeros(_slope_mean.shape[:-1] + (1,), dtype=_slope_mean.dtype), _slope_mean], axis=-1)
            _slope_knj = jnp.take_along_axis(
                _slope_padded[:, :, :, None, None, :], injury_types[..., None][None, None], -1
            ).squeeze(-1)  # (c, d, k, n, j)
            injury_effect = injury_effect + _slope_knj * jnp.asarray(_delta_nj)[None, None, None]
        latent_val = latent_val + injury_effect

        if _new_struct:
            # ── Current structure: write one row per (chain, draw, metric, injured player-season)
            # with REAL player ids — replacing the old type-indexed table whose broadcast player axis
            # stamped every row with the placeholder id and silently broke every player-keyed join.
            # value = that player's (metric, type) mean, plus the per-season residual if the pkl
            # still carries one (dropped in the current model). Rows for the same injury type share
            # the mean; keeping them per-player-season keeps the R ATT join simple and correct.
            _S = int(_idx_n.size)
            _nc, _nd, _nk = injury_mean_prior.shape[:3]
            _mean_sel = jnp.take(injury_mean_prior, _inj_type_s - 1, axis=-1)     # (c, d, k, S)
            if _injury_resid is not None:
                _mean_sel = _mean_sel + _injury_resid
            if _slope_mean is not None:
                # per-season realized effect includes the slope at that season's time-since-onset
                _delta_s = jnp.asarray(_delta_nj[_idx_n, _idx_j])                 # (S,)
                _mean_sel = _mean_sel + jnp.take(_slope_mean, _inj_type_s - 1, axis=-1) * _delta_s[None, None, None, :]
            _effect_s = np.asarray(_mean_sel)
            _ci, _di, _ki, _si = np.meshgrid(
                np.arange(_nc), np.arange(_nd), np.arange(_nk), np.arange(_S), indexing="ij")
            _player_ids_np = id_df["id"].to_numpy()
            _labels_np = np.array(injury_type_labels)
            injury_posterior_df = pd.DataFrame({
                "chain":  _ci.ravel(),
                "sample": _di.ravel(),
                "metric": np.array(list(metrics))[_ki.ravel()],
                "player": _player_ids_np[_idx_n[_si.ravel()]],
                "age":    (age_min + _idx_j[_si.ravel()]).astype(int),
                "id":     _inj_type_s[_si.ravel()],
                "injury_type": _labels_np[_inj_type_s[_si.ravel()] - 1],
                "value":  _effect_s.ravel(),
            })
        else:
            injury_posterior_df = posterior_injury_to_df(
                injury_effect_raw,
                id_df["id"].to_numpy(),
                metrics,
                list(range(age_min, age_max + 1)),
                injury_type_ids,
                injury_type_labels,
                injury_at_age=jnp.any(injury_masks, axis=0).astype(jnp.int32),
            )
        injury_posterior_df.to_parquet(os.path.join(model_dir, "posterior_injury_samples.parquet"), index=False)

        # Horizon estimand (slope variant): injury_effect_h2 = tau_level + 2*tau_slope, the total
        # (metric, type) effect two years after onset — the headline identified quantity when level
        # and slope trade off over short post-injury windows.
        if "injury_effect_h2" in results_mcmc:
            _h2 = np.asarray(results_mcmc["injury_effect_h2"])   # (c, d, k, i)
            _hc, _hd, _hk, _hi = _h2.shape
            _hci, _hdi, _hki, _hii = np.meshgrid(
                np.arange(_hc), np.arange(_hd), np.arange(_hk), np.arange(_hi), indexing="ij")
            pd.DataFrame({
                "chain":  _hci.ravel(),
                "sample": _hdi.ravel(),
                "metric": np.array(list(metrics))[_hki.ravel()],
                "injury_type": np.array(injury_type_labels)[_hii.ravel()],
                "value":  _h2.ravel(),
            }).to_parquet(os.path.join(model_dir, "posterior_injury_horizon.parquet"), index=False)

        # Export player-specific injury effect (already selected by injury_type) — shape (chains, draws, k, n, j)
        injury_effect_player_df = posterior_to_df(
            jnp.transpose(injury_effect, (0, 1, 3, 4, 2)),  # → (chains, draws, n, j, k)
            id_df["id"],
            metrics,
            range(age_min, age_max + 1),
        )
        injury_effect_player_df.to_parquet(os.path.join(model_dir, "posterior_injury_effect.parquet"), index=False)

        # Per-type exit-hazard effect draws (..., i) — the same array the survival exports use.
        # Current structure: offset + sigma * raw; legacy: factor product + offset.
        _injury_exit_effect = None
        if _has_new_exit:
            _injury_exit_effect = (
                results_mcmc["injury_exit_global_offset"][..., None]
                + results_mcmc["sigma_injury_exit"][..., None] * results_mcmc["injury_exit_raw"]
            )  # (c, d, i)
        elif _has_legacy_exit:
            _injury_exit_effect = jnp.einsum(
                "...ip, ...p -> ...i", results_mcmc["injury_factor"], results_mcmc["injury_exit_loading"])
            if "injury_exit_global_offset" in results_mcmc:
                _injury_exit_effect = _injury_exit_effect + results_mcmc["injury_exit_global_offset"][..., None]

        # No per-player hazard residual in the current model (dropped for the same identification
        # reason as the metric residual); the hazard effect is the per-injury-type shift alone.
        _injury_exit_resid_n = None

        injury_prior_mean_export = injury_mean_prior
        injury_prior_metrics = list(metrics)
        if _injury_exit_effect is not None:
            injury_prior_mean_export = jnp.concatenate(
                [injury_prior_mean_export, _injury_exit_effect[:, :, None, :]],
                axis=2,
            )
            injury_prior_metrics = injury_prior_metrics + ["exit_hazard"]

        injury_prior_df = posterior_injury_prior_mean_to_df(
            injury_prior_mean_export,
            injury_prior_metrics,
            injury_type_ids,
            injury_type_labels,
        )
        injury_prior_df.to_parquet(os.path.join(model_dir, "posterior_injury_prior_mean.parquet"), index=False)

        # Export the per-metric GLOBAL injury effect. Use the IDENTIFIED quantity — the mean of the
        # type-level effects, mean_i(injury_mean_prior[.,.,k,i]) — NOT the raw injury_global_offset
        # site. The raw offset is aliased with the type deviations by an additive constant (shift it
        # between them and the effect is unchanged), so across chains it is non-identified
        # (rhat ~10) and can differ from the true mean effect by an arbitrary amount — materially,
        # even flipping sign for some metrics. The mean over types is invariant to that alias and is
        # what "the global injury offset for metric k" actually means.
        _go = np.array(injury_mean_prior.mean(axis=-1))                # (chains, draws, k) — identified
        _n_chains, _n_draws, _k = _go.shape
        _ci, _si, _ki = np.meshgrid(np.arange(_n_chains), np.arange(_n_draws), np.arange(_k), indexing="ij")
        global_offset_df = pd.DataFrame({
            "chain":  _ci.ravel(),
            "sample": _si.ravel(),
            "metric": np.array(list(metrics))[_ki.ravel()],
            "value":  _go.ravel(),
        })
        _ci2, _si2 = np.meshgrid(np.arange(_n_chains), np.arange(_n_draws), indexing="ij")
        if _injury_exit_effect is not None:
            # identified mean over injury types on the hazard, same treatment
            _ego = np.array(np.asarray(_injury_exit_effect).mean(axis=-1))
            global_offset_df = pd.concat([global_offset_df, pd.DataFrame({
                "chain": _ci2.ravel(), "sample": _si2.ravel(),
                "metric": "exit_hazard", "value": _ego.ravel(),
            })], ignore_index=True)
        global_offset_df.to_parquet(os.path.join(model_dir, "posterior_injury_global_offset.parquet"), index=False)
    else:
        injury_effect = jnp.zeros_like(latent_val)

    surv_posterior = None
    if _coverage_only and (not _with_elppd) and os.path.exists(os.path.join(model_dir, "posterior_exit_age_sample.parquet")):
        # survival exports are unaffected by the trajectory-noise fix — skip for --coverage_only,
        # but ONLY when the artifact already exists (fresh dirs, e.g. pilots, still need it:
        # coverage.r reads posterior_exit_age_sample/posterior_exit_survival unconditionally).
        pass
    elif has_survival_injury and injury:
            surv_posterior = make_survival_linear_injury_mcmc(
                X=_X_surv,
                gamma_global_log=results_mcmc["gamma_global_log"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_exit_effect=_injury_exit_effect,
                injury_exit_player_resid=_injury_exit_resid_n,
                injury_indicator=injury_masks,
                injury_type=injury_types,
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc["sigma_exit_scale"],
                eta_global_log=results_mcmc.get("eta_global_log", jnp.log(0.04)),
                age_min=age_min,
                kernel_self_cov=_surv_kcov,
            )

            observed_surv_df = pd.DataFrame(
                {
                    "player": id_df["id"].to_numpy(),
                    "observed_entrance_age": np.asarray(Y_surv_eval[:, 0]),
                    "observed_exit_age": np.asarray(Y_surv_eval[:, 1]),
                    "exit_censored": np.asarray(surv_masks_eval[:, 1]).astype(np.int32),
                }
            )

            surv_posterior_counterfactual = make_survival_linear_injury_mcmc(
                X=_X_surv,
                gamma_global_log=results_mcmc["gamma_global_log"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_exit_effect=_injury_exit_effect,
                injury_exit_player_resid=_injury_exit_resid_n,
                injury_indicator=jnp.zeros_like(injury_masks),
                injury_type=jnp.zeros_like(injury_types),   # type=0 → true no-injury baseline
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc["sigma_exit_scale"],
                eta_global_log=results_mcmc.get("eta_global_log", jnp.log(0.04)),
                age_min=age_min,
                kernel_self_cov=_surv_kcov,
            )


            exit_survival_df_obs = posterior_survival_to_df(
                surv_posterior["exit_survival"],
                id_df["id"],
                list(range(age_min, age_max + 1)),
                "exit_survival",
            )
            exit_survival_df_obs = exit_survival_df_obs.merge(observed_surv_df, on="player", how="left")
            exit_survival_df_obs["scenario"] = "observed"
            exit_survival_df_cf = posterior_survival_to_df(
                surv_posterior_counterfactual["exit_survival"],
                id_df["id"],
                list(range(age_min, age_max + 1)),
                "exit_survival",
            )
            exit_survival_df_cf = exit_survival_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_survival_df_cf["scenario"] = "counterfactual"
            pd.concat([exit_survival_df_obs, exit_survival_df_cf], ignore_index=True).to_parquet(
                os.path.join(model_dir, "posterior_exit_survival.parquet"), index=False
            )

            exit_hazard_df_obs = posterior_survival_to_df(
                surv_posterior["exit_hazard"],
                id_df["id"],
                list(range(age_min, age_max + 1)),
                "exit_hazard",
            )
            exit_hazard_df_obs = exit_hazard_df_obs.merge(observed_surv_df, on="player", how="left")
            exit_hazard_df_obs["scenario"] = "observed"
            exit_hazard_df_cf = posterior_survival_to_df(
                surv_posterior_counterfactual["exit_hazard"],
                id_df["id"],
                list(range(age_min, age_max + 1)),
                "exit_hazard",
            )
            exit_hazard_df_cf = exit_hazard_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_hazard_df_cf["scenario"] = "counterfactual"
            pd.concat([exit_hazard_df_obs, exit_hazard_df_cf], ignore_index=True).to_parquet(
                os.path.join(model_dir, "posterior_exit_hazard.parquet"), index=False
            )

            _inj_ent_dur  = Y_surv[:, 0] - age_min + 1e-6
            _inj_tobs_dur = np.maximum(Y_surv[:, 1] - age_min, _inj_ent_dur)
            _surv_inj_tobs = make_survival_linear_injury_mcmc(
                X=_X_surv,
                gamma_global_log=results_mcmc["gamma_global_log"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_exit_effect=_injury_exit_effect,
                injury_exit_player_resid=_injury_exit_resid_n,
                injury_indicator=injury_masks,
                injury_type=injury_types,
                entrance_times=_inj_ent_dur,
                basis=basis,
                sigma_exit_scale=results_mcmc["sigma_exit_scale"],
                eta_global_log=results_mcmc.get("eta_global_log", jnp.log(0.04)),
                age_min=age_min,
                last_obs_times=_inj_tobs_dur,
                kernel_self_cov=_surv_kcov,
            )
            _exit_age_entrance_df = posterior_player_scalar_to_df(
                surv_posterior["exit_age_sample"], id_df["id"], "exit_age_sample"
            )
            _exit_age_entrance_df["conditioning_label"] = "entrance"
            _exit_age_entrance_df = _exit_age_entrance_df.merge(observed_surv_df, on="player", how="left")
            _exit_age_entrance_df["scenario"] = "observed"
            _exit_age_tobs_df = posterior_player_scalar_to_df(
                _surv_inj_tobs["exit_age_sample"], id_df["id"], "exit_age_sample"
            )
            _exit_age_tobs_df["conditioning_label"] = "last_observed"
            _exit_age_tobs_df = _exit_age_tobs_df.merge(observed_surv_df, on="player", how="left")
            _exit_age_tobs_df["scenario"] = "observed"
            exit_age_sample_df_cf = posterior_player_scalar_to_df(
                surv_posterior_counterfactual["exit_age_sample"],
                id_df["id"],
                "exit_age_sample",
            )
            exit_age_sample_df_cf = exit_age_sample_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_age_sample_df_cf["scenario"] = "counterfactual"
            exit_age_sample_df_cf["conditioning_label"] = "entrance"
            pd.concat(
                [_exit_age_entrance_df, _exit_age_tobs_df, exit_age_sample_df_cf], ignore_index=True
            ).to_parquet(os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False)
    elif _is_naive:
        # Naive survival: per-player Gompertz with no latent X.
        _naive_surv_key = jax.random.PRNGKey(42)
        def _naive_surv_one_draw(key, gamma_global_log, eta_global_log, entrance_times, last_obs_times):
            eta   = jnp.exp(eta_global_log.squeeze(-1))    # (n,) — baseline hazard
            gamma = jnp.exp(gamma_global_log.squeeze(-1))  # (n,) — aging rate
            tenure_grid = jnp.maximum(basis - age_min, 1e-6)  # (j,)
            # S(t | T > entrance) = exp(-(η/γ)*(exp(γ*t) - exp(γ*entrance)))
            def _surv(t, eta_i, gamma_i, ent):
                return jnp.exp(-(eta_i / gamma_i) * (jnp.exp(gamma_i * t) - jnp.exp(gamma_i * ent)))
            exit_survival = vmap(lambda e, g, ent: _surv(tenure_grid, e, g, ent))(eta, gamma, entrance_times)
            exit_hazard   = vmap(lambda e, g: e * jnp.exp(g * tenure_grid))(eta, gamma)
            # Gompertz inverse-CDF conditioned on T > last_obs_times:
            #   t = (1/γ) * log(exp(γ*last_obs) + target * γ/η)
            u = jnp.clip(jax.random.uniform(key, shape=eta.shape), 1e-6, 1.0 - 1e-6)
            target = -jnp.log(u)
            lam = gamma / eta
            last_obs_exp = jnp.exp(gamma * jnp.maximum(last_obs_times, 0.0))
            sampled_duration = jnp.log(last_obs_exp + target * lam) / gamma
            sampled_duration = jnp.clip(sampled_duration, last_obs_times, float(basis[-1] - age_min))
            exit_age_sample = float(age_min) + sampled_duration
            return {"exit_survival": exit_survival, "exit_hazard": exit_hazard, "exit_age_sample": exit_age_sample}

        _n_chains, _n_draws = results_mcmc["gamma_global_log"].shape[:2]
        _naive_surv_keys = jax.random.split(_naive_surv_key, _n_chains * _n_draws).reshape(_n_chains, _n_draws, 2)
        _entrance_dur = Y_surv[:, 0] - age_min + 1e-6
        _last_obs_dur = jnp.maximum(jnp.array(Y_surv[:, 1] - age_min), jnp.array(_entrance_dur))
        _naive_surv_vmap = vmap(vmap(lambda k, a, b: _naive_surv_one_draw(k, a, b, _entrance_dur, _entrance_dur)))
        surv_posterior = _naive_surv_vmap(
            _naive_surv_keys,
            results_mcmc["gamma_global_log"],  # (chains, draws, n, 1)
            results_mcmc["eta_global_log"],    # (chains, draws, n, 1)
        )

        observed_surv_df = pd.DataFrame(
            {
                "player": id_df["id"].to_numpy(),
                "observed_entrance_age": np.asarray(Y_surv_eval[:, 0]),
                "observed_exit_age": np.asarray(Y_surv_eval[:, 1]),
                "exit_censored": np.asarray(surv_masks_eval[:, 1]).astype(np.int32),
            }
        )

        exit_survival_df_obs = posterior_survival_to_df(
            surv_posterior["exit_survival"],
            id_df["id"],
            list(range(age_min, age_max + 1)),
            "exit_survival",
        )
        exit_survival_df_obs = exit_survival_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_survival_df_obs["scenario"] = "observed"
        exit_survival_df_obs.to_parquet(os.path.join(model_dir, "posterior_exit_survival.parquet"), index=False)

        exit_hazard_df_obs = posterior_survival_to_df(
            surv_posterior["exit_hazard"],
            id_df["id"],
            list(range(age_min, age_max + 1)),
            "exit_hazard",
        )
        exit_hazard_df_obs = exit_hazard_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_hazard_df_obs["scenario"] = "observed"
        exit_hazard_df_obs.to_parquet(os.path.join(model_dir, "posterior_exit_hazard.parquet"), index=False)

        # Exit age samples: entrance-conditioned and last_observed-conditioned
        def _sample_naive_cond(key, gamma_global_log, eta_global_log, last_obs_times):
            eta   = jnp.exp(eta_global_log.squeeze(-1))
            gamma = jnp.exp(gamma_global_log.squeeze(-1))
            u = jnp.clip(jax.random.uniform(key, shape=eta.shape), 1e-6, 1.0 - 1e-6)
            lam = gamma / eta
            t_exp = jnp.exp(gamma * jnp.maximum(last_obs_times, 0.0))
            sampled = jnp.log(t_exp + (-jnp.log(u)) * lam) / gamma
            return float(age_min) + jnp.clip(sampled, last_obs_times, float(basis[-1] - age_min))

        _cond_scenarios = [
            ("entrance",      _entrance_dur),
            ("last_observed", _last_obs_dur),
        ]
        _exit_age_dfs = []
        for _label, _cond_dur in _cond_scenarios:
            _cond_keys = jax.random.split(
                jax.random.PRNGKey(hash(_label) % (2 ** 31)), _n_chains * _n_draws
            ).reshape(_n_chains, _n_draws, 2)
            _exit_arr = vmap(vmap(lambda k, a, b: _sample_naive_cond(k, a, b, _cond_dur)))(
                _cond_keys, results_mcmc["gamma_global_log"], results_mcmc["eta_global_log"]
            )
            _df = posterior_player_scalar_to_df(_exit_arr, id_df["id"], "exit_age_sample")
            _df["conditioning_label"] = _label
            _df = _df.merge(observed_surv_df, on="player", how="left")
            _df["scenario"] = "observed"
            _exit_age_dfs.append(_df)
        pd.concat(_exit_age_dfs, ignore_index=True).to_parquet(
            os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False
        )
    else:
        # _X_surv / _surv_kcov are computed once near the X_mcmc_aug reconstruction above (hoisted
        # so the injury survival exports share the identical RFF projection).
        surv_posterior = make_survival_linear_mcmc(
                X=_X_surv,
                gamma_global_log=results_mcmc["gamma_global_log"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc.get("sigma_exit_scale", 1.0),
                eta_global_log=results_mcmc.get("eta_global_log", jnp.log(0.04)),
                age_min=age_min,
                kernel_self_cov=_surv_kcov,
            )

        observed_surv_df = pd.DataFrame(
            {
                "player": id_df["id"].to_numpy(),
                "observed_entrance_age": np.asarray(Y_surv_eval[:, 0]),
                "observed_exit_age": np.asarray(Y_surv_eval[:, 1]),
                "exit_censored": np.asarray(surv_masks_eval[:, 1]).astype(np.int32),
            }
        )

        exit_survival_df_obs = posterior_survival_to_df(
            surv_posterior["exit_survival"],
            id_df["id"],
            list(range(age_min, age_max + 1)),
            "exit_survival",
        )
        exit_survival_df_obs = exit_survival_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_survival_df_obs["scenario"] = "observed"

        exit_survival_df_obs.to_parquet(
            os.path.join(model_dir, "posterior_exit_survival.parquet"), index=False
        )

        exit_hazard_df_obs = posterior_survival_to_df(
            surv_posterior["exit_hazard"],
            id_df["id"],
            list(range(age_min, age_max + 1)),
            "exit_hazard",
        )
        exit_hazard_df_obs = exit_hazard_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_hazard_df_obs["scenario"] = "observed"

        exit_hazard_df_obs.to_parquet(
            os.path.join(model_dir, "posterior_exit_hazard.parquet"), index=False
        )

        _ent_dur = Y_surv[:, 0] - age_min + 1e-6
        _tobs_dur = np.maximum(Y_surv[:, 1] - age_min, _ent_dur)
        _surv_cond_tobs = make_survival_linear_mcmc(
            X=_X_surv,
            gamma_global_log=results_mcmc["gamma_global_log"],
            exit=results_mcmc["exit"],
            exit_rate=results_mcmc["exit_rate"],
            entrance_times=_ent_dur,
            basis=basis,
            sigma_exit_scale=results_mcmc.get("sigma_exit_scale", 1.0),
            eta_global_log=results_mcmc.get("eta_global_log", jnp.log(0.04)),
            age_min=age_min,
            last_obs_times=_tobs_dur,
            kernel_self_cov=_surv_kcov,
        )
        _exit_age_entrance_df = posterior_player_scalar_to_df(
            surv_posterior["exit_age_sample"], id_df["id"], "exit_age_sample"
        )
        _exit_age_entrance_df["conditioning_label"] = "entrance"
        _exit_age_entrance_df = _exit_age_entrance_df.merge(observed_surv_df, on="player", how="left")
        _exit_age_entrance_df["scenario"] = "observed"
        _exit_age_tobs_df = posterior_player_scalar_to_df(
            _surv_cond_tobs["exit_age_sample"], id_df["id"], "exit_age_sample"
        )
        _exit_age_tobs_df["conditioning_label"] = "last_observed"
        _exit_age_tobs_df = _exit_age_tobs_df.merge(observed_surv_df, on="player", how="left")
        _exit_age_tobs_df["scenario"] = "observed"
        pd.concat([_exit_age_entrance_df, _exit_age_tobs_df], ignore_index=True).to_parquet(
            os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False
        )


    players = id_df[id_df["name"].isin(predict_players)].index
    player_names = id_df[id_df["name"].isin(predict_players)]["name"].tolist()

    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)

    players_idx = jnp.array(id_df.index)
    ages = list(range(age_min, age_max + 1))
    n_players_sel = len(id_df)
    n_ages = len(ages)



    if not _coverage_only:
        make_diagnostic_heatmap(
            latent_val,
            n_players_sel,
            n_ages,
            ages,
            os.path.join(model_dir, "plots", "posterior_latent_ar.png"),
            player_labels=all_player_labels,
        )
        make_diagnostic_heatmap(
            mu_mcmc,
            n_players_sel,
            n_ages,
            ages,
            os.path.join(model_dir, "plots", "posterior_mu_ar.png"),
            player_labels=all_player_labels,
        )
        make_rhat_summary_barchart(
            latent_val,
            os.path.join(model_dir, "plots", "rhat_summary_latent_ar.png"),
            metric_labels=metrics,
        )
        make_rhat_summary_barchart(
            mu_mcmc,
            os.path.join(model_dir, "plots", "rhat_summary_mu_ar.png"),
            metric_labels=metrics,
        )




    # Summarise whatever dispersion sites this panel actually has. Was hardcoded to
    # [sigma_beta, sigma_beta_binomial], which yields an empty dict -- and an arviz
    # ValueError -- for a panel with no beta/beta-binomial head (e.g. the all-count/binomial
    # athleticism panel). Listing all four also makes the summary more informative for the
    # box-score panel, which has every family.
    _summary_vars = ["sigma", "sigma_beta", "sigma_beta_binomial", "sigma_negative_binomial"]
    _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
    if _summary_subset:
        summary = az.summary(_summary_subset)
        summary.to_parquet(os.path.join(model_dir, "posterior_variance_summary.parquet"), index=False)
    else:
        print("[export] no dispersion sites in this panel -- skipping posterior_variance_summary")

    peaks    = tmax_mcmc + basis.mean() if tmax_mcmc is not None else None
    peak_val = cmax_mcmc if cmax_mcmc is not None else None

    if _peaks_no_re:
        # Archetypal (RE-free) peaks for the peak-age / peak-value PCA figures.
        if peaks is not None:
            posterior_peaks_to_df(peaks, id_df["id"], metrics).to_parquet(
                os.path.join(model_dir, "posterior_peaks_ar_shared.parquet"), index=False)
        if peak_val is not None:
            posterior_peaks_to_df(peak_val, id_df["id"], metrics).to_parquet(
                os.path.join(model_dir, "posterior_peak_vals_ar_shared.parquet"), index=False)
        print("--peaks_no_re: wrote posterior_peaks_ar_shared / posterior_peak_vals_ar_shared; exiting")
        raise SystemExit(0)


    def _disp(_name):
        """Posterior dispersion samples for a family that may be absent from this panel.

        A metric panel only creates the dispersion site for families it actually contains:
        the athleticism panel is all count/binomial, so it has sigma_negative_binomial but no
        `sigma` (gaussian), `sigma_beta` (beta) or `sigma_beta_binomial` (beta-binomial).
        create_metric_trajectory_all coerces None -> 1 and never reads the value for a family
        the panel does not contain, so None is the correct thing to pass.
        """
        return jnp.transpose(results_mcmc[_name], (2, 0, 1)) if _name in results_mcmc else None

    _neg_bin_samples = _disp("sigma_negative_binomial")
    _, pos = create_metric_trajectory_all(latent_val, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            _disp("sigma"),
                                            _disp("sigma_beta"),
                                            posterior_kappa_samples=_disp("sigma_beta_binomial"),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    _, pos_mu = create_metric_trajectory_all(mu_mcmc + TREND_AR_mcmc, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            _disp("sigma"),
                                            _disp("sigma_beta"),
                                            posterior_kappa_samples=_disp("sigma_beta_binomial"),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    

    posterior_df = posterior_to_df(pos, id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_df.to_parquet(os.path.join(model_dir, "posterior_ar.parquet"), index=False)

    # Conditional coverage conditions holdout cells on observed games/pct_minutes exposures.
    # A panel without those heads has no exposure cascade to condition on (its exposures are
    # directly observed, e.g. per-possession), so the artifact is undefined -- skip it rather
    # than crash on metrics.index("pct_minutes").
    if ("pct_minutes" in metrics) and ("games" in metrics):
        # Conditional posterior: for holdout cells, condition on observed games and pct_minutes as
        # exposures so that only metric-rate uncertainty (FG2A/36, etc.) is propagated.  This enables
        # "conditional coverage" in model_diagnostics.r — coverage that removes the contribution of
        # minutes/games uncertainty and tests only the rate predictions.
        minutes_index = metrics.index("pct_minutes")
        games_index   = metrics.index("games")
        _pct_min_pivot = (
            data.pivot_table(index="id", columns="age", values="pct_minutes", aggfunc="first")
            .reindex(index=id_df["id"].tolist(), columns=range(age_min, age_max + 1))
        )
        _pct_min_obs = jnp.array(_pct_min_pivot.values.astype(np.float64))  # (n, j)
        _holdout_pct_obs = jnp.array(_score_mask_np) & ~jnp.isnan(_pct_min_obs)

        _games_pivot = (
            data.pivot_table(index="id", columns="age", values="games", aggfunc="first")
            .reindex(index=id_df["id"].tolist(), columns=range(age_min, age_max + 1))
        )
        _games_obs = jnp.array(_games_pivot.values.astype(np.float64))  # (n, j)
        _holdout_games_obs = jnp.array(_score_mask_np) & ~jnp.isnan(_games_obs)

        Y_conditional = (
            Y
            .at[minutes_index].set(jnp.where(_holdout_pct_obs, _pct_min_obs, Y[minutes_index]))
            .at[games_index].set(jnp.where(_holdout_games_obs, _games_obs, Y[games_index]))
        )
        _, pos_conditional = create_metric_trajectory_all(
            latent_val, Y_conditional, exposures,
            metric_output, metrics, exposure_list,
            _disp("sigma"),
            _disp("sigma_beta"),
            posterior_kappa_samples=_disp("sigma_beta_binomial"),
            posterior_neg_bin_samples=_neg_bin_samples,
            condition_on_observed=True,
        )
        posterior_conditional_df = posterior_to_df(pos_conditional, id_df["id"], metrics, range(age_min, age_max + 1))
        posterior_conditional_df.to_parquet(os.path.join(model_dir, "posterior_ar_conditional.parquet"), index=False)
    else:
        print("[export] no games/pct_minutes heads -- skipping posterior_ar_conditional")

    if peaks is not None:
        posterior_peaks = posterior_peaks_to_df(peaks, id_df["id"], metrics)
        posterior_peaks.to_parquet(os.path.join(model_dir, "posterior_peaks_ar.parquet"), index=False)

    if ("AR" in model_name) or injury or _is_naive:
        posterior_ar_df = posterior_to_df(jnp.transpose(AR, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(age_min, age_max + 1))
        posterior_ar_df.to_parquet(os.path.join(model_dir, "posterior_latent_ar.parquet"), index=False)

    if injury and ("counterfactual" not in model_name):
        _, pos_counterfactual = create_metric_trajectory_all(mu_mcmc + AR + TREND_AR_mcmc, Y, exposures,
                                        metric_output, metrics, exposure_list,
                                        _disp("sigma"),
                                        _disp("sigma_beta"),
                                        posterior_kappa_samples=_disp("sigma_beta_binomial"),
                                        posterior_neg_bin_samples=_neg_bin_samples,
                                        )
        posterior_counterfactual_df = posterior_to_df(pos_counterfactual, id_df["id"], metrics, range(age_min, age_max + 1))
        posterior_counterfactual_df.to_parquet(os.path.join(model_dir, "posterior_counterfactual_ar.parquet"), index=False)

    posterior_mu_df = posterior_to_df(jnp.transpose(mu_mcmc + TREND_AR_mcmc, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_mu_df.to_parquet(os.path.join(model_dir, "posterior_mu_ar.parquet"), index=False)

    # Export calendar-year AR(3) trend
    _has_year_ar_mcmc = len(de_trend_metrics) > 0 and all(k in results_mcmc for k in _year_ar_keys)
    if _has_year_ar_mcmc:
        _de_trend_metric_names = [m for m, f in zip(metrics, de_trend_indices) if f]
        _years_range = np.arange(min_year, min_year + num_years)

        # Recompute compact (chains*draws, num_ar, num_years) for plot / CSV
        _s = results_mcmc["sigma_year_ar"]
        _r = results_mcmc["rho_year_ar"]
        _z = results_mcmc["beta_year_ar"]
        _a = results_mcmc["AR_0_year"]
        def _ar3_compact(s, r, z, a0):
            a0_scaled = a0 * s[None, :, 0]
            return _ARLinearLVM._compute_ar1_calendar_process(s, r, z, a0_scaled)  # (num_ar, num_years)
        _trend_years_all = np.asarray(vmap(vmap(_ar3_compact))(_s, _r, _z, _a))  # (chains, draws, num_ar, num_years)
        chains_d, draws_d = _trend_years_all.shape[:2]
        _trend_flat = _trend_years_all.reshape(chains_d * draws_d, _num_ar, num_years)  # (S, num_ar, num_years)

        rows = []
        _mean = _trend_flat.mean(axis=0)   # (num_ar, num_years)
        _lo   = np.quantile(_trend_flat, 0.05, axis=0)
        _hi   = np.quantile(_trend_flat, 0.95, axis=0)
        for mi, mname in enumerate(_de_trend_metric_names):
            for yi, yr in enumerate(_years_range):
                rows.append({"metric": mname, "year": int(yr),
                             "mean": float(_mean[mi, yi]),
                             "q05":  float(_lo[mi, yi]),
                             "q95":  float(_hi[mi, yi])})
        pd.DataFrame(rows).to_parquet(os.path.join(model_dir, "posterior_calendar_year_trend.parquet"), index=False)

        os.makedirs(os.path.join(model_dir, "plots", "calendar_year_trends"), exist_ok=True)
        plot_calendar_year_trends(
            _trend_flat,
            _years_range,
            _de_trend_metric_names,
            os.path.join(model_dir, "plots", "calendar_year_trends", f"{model_name}_calendar_year_trends.png"),
        )

    if _coverage_only and not _with_elppd:
        print("--coverage_only: trajectory/coverage artifacts written, skipping log-loss/ELPPD and remaining exports")
        raise SystemExit(0)

    # ── Per-sample log-loss (posterior interval for predictive accuracy) ─────────
    # For each (chain, draw), compute avg NLL per metric on holdout and in-sample
    # splits using that sample's latent mean + dispersion parameters.
    # Produces a compact (chain, draw, split, metric, avg_log_loss) parquet that
    # model_diagnostics.r uses to build posterior log-loss intervals.
    try:
        from model.model_utils import summarize_metric_error_observed_substitutions as _sme
        from model.model_utils import summarize_pointwise_log_likelihoods as _spll
        _ll_rows = []
        _n_chains_ll, _n_draws_ll = latent_val.shape[:2]
        _val_mask_np  = np.asarray(validation_mask, dtype=bool)   # full held-out set (drives in-sample complement)
        _nval_mask_np = ~_val_mask_np                              # in-sample = never held out from training
        _hold_mask_np = _score_mask_np                            # SCORED holdout cells (next-k window for stratified)
        _Y_np  = np.asarray(Y)
        _E_np  = np.asarray(exposures)
        # Pre-compute per-sample survival log-likelihoods (shape C×D×n_players):
        #   log p(T_i | θ) = log S(T_i) + (1 − censored) · log h(T_i)
        _surv_ll_all = None
        if surv_posterior is not None:
            try:
                _exit_surv_np = np.asarray(surv_posterior["exit_survival"])   # (C,D,n_pl,n_ages)
                _exit_haz_np  = np.asarray(surv_posterior["exit_hazard"])
                _exit_age_idx = np.clip(
                    np.round(np.asarray(Y_surv_eval[:, 1]) - age_min).astype(int),
                    0, _exit_surv_np.shape[-1] - 1,
                )  # (n_pl,) — use eval (true) exit ages, not training-censored ones
                _is_censored  = np.asarray(surv_masks_eval[:, 1]).astype(bool)    # (n_pl,)
                _pl_idx       = np.arange(_exit_surv_np.shape[2])
                _log_s = np.log(np.clip(_exit_surv_np[:, :, _pl_idx, _exit_age_idx], 1e-300, 1.0))
                _log_h = np.log(np.clip(_exit_haz_np [:, :, _pl_idx, _exit_age_idx], 1e-300, None))
                _surv_ll_all  = _log_s + np.where(_is_censored[None, None, :], 0.0, _log_h)
            except Exception as _surv_pre_e:
                print(f"[warn] survival LL pre-computation skipped: {_surv_pre_e}")
        # ELPPD accumulators: log-sum-exp across samples, init at -inf
        _n_players_ll, _n_ages_ll = _val_mask_np.shape
        _elppd_acc = {
            _sp: {_m: np.full((_n_players_ll, _n_ages_ll), -np.inf) for _m in metrics}
            for _sp in ("holdout", "in_sample")
        }
        # Survival accumulator (one entry per player, not per player-age)
        _elppd_acc_surv      = {_sp: np.full((_n_players_ll,), -np.inf) for _sp in ("holdout", "in_sample")}
        _surv_holdout_pmask  = np.any(_val_mask_np, axis=1)   # players with any held-out season
        _surv_insample_pmask = ~_surv_holdout_pmask
        _n_samples_total = _n_chains_ll * _n_draws_ll
        for _c in range(_n_chains_ll):
            for _d in range(_n_draws_ll):
                _mu_cd = np.asarray(latent_val[_c, _d])   # (k, n, t)
                _sig_cd    = np.asarray(results_mcmc["sigma"][_c, _d])                    if "sigma"                    in results_mcmc else 1
                _sig_b_cd  = np.asarray(results_mcmc["sigma_beta"][_c, _d])               if "sigma_beta"               in results_mcmc else 1
                _sig_bb_cd = np.asarray(results_mcmc["sigma_beta_binomial"][_c, _d])      if "sigma_beta_binomial"      in results_mcmc else 1
                _sig_nb_cd = np.asarray(results_mcmc["sigma_negative_binomial"][_c, _d])  if "sigma_negative_binomial"  in results_mcmc else 1
                for _split, _mask in (("holdout", _hold_mask_np), ("in_sample", _nval_mask_np)):
                    _res = _sme(
                        posterior_mean_map=_mu_cd,
                        observations=_Y_np,
                        exposures=_E_np,
                        metric_outputs=metric_output,
                        metrics=metrics,
                        sigma_beta=_sig_b_cd,
                        sigma=_sig_cd,
                        sigma_beta_binomial=_sig_bb_cd,
                        sigma_negative_binomial=_sig_nb_cd,
                        evaluation_mask=_mask,
                    )
                    for _, row in _res.iterrows():
                        _ll_rows.append({
                            "chain": _c, "draw": _d, "split": _split,
                            "metric": row["metric"], "avg_log_loss": row["avg_log_loss"],
                        })
                    # ELPPD: accumulate log p(y_i|theta_s) via log-sum-exp
                    _pw = _spll(
                        posterior_mean_map=_mu_cd,
                        observations=_Y_np,
                        exposures=_E_np,
                        metric_outputs=metric_output,
                        metrics=metrics,
                        sigma_beta=_sig_b_cd,
                        sigma=_sig_cd,
                        sigma_beta_binomial=_sig_bb_cd,
                        sigma_negative_binomial=_sig_nb_cd,
                        evaluation_mask=_mask,
                    )
                    for _mn, _ll_arr in _pw.items():
                        _elppd_acc[_split][_mn] = np.logaddexp(_elppd_acc[_split][_mn], _ll_arr)
                    # Survival: accumulate per-player log-likelihood
                    if _surv_ll_all is not None:
                        _surv_cd = _surv_ll_all[_c, _d]   # (n_players,)
                        _elppd_acc_surv[_split] = np.logaddexp(_elppd_acc_surv[_split], _surv_cd)
                        _spm = _surv_holdout_pmask if _split == "holdout" else _surv_insample_pmask
                        _valid_sp = np.isfinite(_surv_cd) & _spm
                        if np.any(_valid_sp):
                            _ll_rows.append({
                                "chain": _c, "draw": _d, "split": _split,
                                "metric": "survival",
                                "avg_log_loss": float(-np.mean(_surv_cd[_valid_sp])),
                            })
        if _ll_rows:
            pd.DataFrame(_ll_rows).to_parquet(
                os.path.join(model_dir, "posterior_metric_log_loss.parquet"), index=False
            )
        # Compute ELPPD = log(mean_s p(y_i|theta_s)) summed over holdout obs,
        # plus SE via pointwise variance (Vehtari et al. 2017)
        _elppd_rows = []
        for _split in ("holdout", "in_sample"):
            _mask_s = _hold_mask_np if _split == "holdout" else _nval_mask_np
            for _mn in metrics:
                _elppd_i = _elppd_acc[_split][_mn] - np.log(_n_samples_total)
                _valid_e = np.isfinite(_elppd_i) & _mask_s
                _n_obs_e = int(np.sum(_valid_e))
                if _n_obs_e > 0:
                    _vals_e = _elppd_i[_valid_e]
                    _elppd_sum = float(np.sum(_vals_e))
                    _elppd_se = float(np.sqrt(_n_obs_e * np.var(_vals_e, ddof=1))) if _n_obs_e > 1 else float("nan")
                else:
                    _elppd_sum = _elppd_se = float("nan")
                _elppd_rows.append({
                    "split": _split, "metric": _mn,
                    "elppd": _elppd_sum,
                    "elppd_per_obs": _elppd_sum / _n_obs_e if _n_obs_e > 0 else float("nan"),
                    "elppd_se": _elppd_se,
                    "n_obs": _n_obs_e,
                })
        # Survival ELPPD (one observation per player)
        if _surv_ll_all is not None:
            for _sp in ("holdout", "in_sample"):
                _spm     = _surv_holdout_pmask if _sp == "holdout" else _surv_insample_pmask
                _surv_ei = _elppd_acc_surv[_sp] - np.log(_n_samples_total)
                _valid_s = np.isfinite(_surv_ei) & _spm
                _n_s     = int(np.sum(_valid_s))
                if _n_s > 0:
                    _vals_s = _surv_ei[_valid_s]
                    _e_s    = float(np.sum(_vals_s))
                    _se_s   = float(np.sqrt(_n_s * np.var(_vals_s, ddof=1))) if _n_s > 1 else float("nan")
                else:
                    _e_s = _se_s = float("nan")
                _elppd_rows.append({
                    "split": _sp, "metric": "survival",
                    "elppd": _e_s,
                    "elppd_per_obs": _e_s / _n_s if _n_s > 0 else float("nan"),
                    "elppd_se": _se_s,
                    "n_obs": _n_s,
                })
        # "all" row: pool pointwise elpd_i values across every metric including survival
        for _sp in ("holdout", "in_sample"):
            _mask_s = _hold_mask_np if _sp == "holdout" else _nval_mask_np
            _pw_all = []
            for _mn in metrics:
                _ei = _elppd_acc[_sp][_mn] - np.log(_n_samples_total)
                _valid = np.isfinite(_ei) & _mask_s
                if np.any(_valid):
                    _pw_all.append(_ei[_valid])
            if _surv_ll_all is not None:
                _surv_ei = _elppd_acc_surv[_sp] - np.log(_n_samples_total)
                _spm     = _surv_holdout_pmask if _sp == "holdout" else _surv_insample_pmask
                _valid_s = np.isfinite(_surv_ei) & _spm
                if np.any(_valid_s):
                    _pw_all.append(_surv_ei[_valid_s])
            if _pw_all:
                _pw_cat = np.concatenate(_pw_all)
                _n_all  = len(_pw_cat)
                _e_all  = float(np.sum(_pw_cat))
                _se_all = float(np.sqrt(_n_all * np.var(_pw_cat, ddof=1))) if _n_all > 1 else float("nan")
            else:
                _n_all = 0
                _e_all = _se_all = float("nan")
            _elppd_rows.append({
                "split": _sp, "metric": "all",
                "elppd": _e_all,
                "elppd_per_obs": _e_all / _n_all if _n_all > 0 else float("nan"),
                "elppd_se": _se_all,
                "n_obs": _n_all,
            })
        # Per-stratum ELPPD (stratified_next_k only)
        if _stratum_mat is not None:
            for _s in sorted(np.unique(_stratum_mat[_stratum_mat > 0]).tolist()):
                _s_mask = (_stratum_mat == _s) & _hold_mask_np
                if not np.any(_s_mask):
                    continue
                _pw_s_all = []
                for _mn in metrics:
                    _elppd_i = _elppd_acc["holdout"][_mn] - np.log(_n_samples_total)
                    _valid_e = np.isfinite(_elppd_i) & _s_mask
                    _n_obs_e = int(np.sum(_valid_e))
                    if _n_obs_e > 0:
                        _vals_e = _elppd_i[_valid_e]
                        _elppd_sum = float(np.sum(_vals_e))
                        _elppd_se = float(np.sqrt(_n_obs_e * np.var(_vals_e, ddof=1))) if _n_obs_e > 1 else float("nan")
                        _pw_s_all.append(_vals_e)
                    else:
                        _elppd_sum = _elppd_se = float("nan")
                    _elppd_rows.append({
                        "split": f"holdout_stratum_{_s}", "metric": _mn,
                        "elppd": _elppd_sum,
                        "elppd_per_obs": _elppd_sum / _n_obs_e if _n_obs_e > 0 else float("nan"),
                        "elppd_se": _elppd_se, "n_obs": _n_obs_e,
                    })
                if _pw_s_all:
                    _pw_cat_s = np.concatenate(_pw_s_all)
                    _n_s = len(_pw_cat_s)
                    _e_s = float(np.sum(_pw_cat_s))
                    _elppd_rows.append({
                        "split": f"holdout_stratum_{_s}", "metric": "all",
                        "elppd": _e_s, "elppd_per_obs": _e_s / _n_s,
                        "elppd_se": float(np.sqrt(_n_s * np.var(_pw_cat_s, ddof=1))) if _n_s > 1 else float("nan"),
                        "n_obs": _n_s,
                    })
            _stratum_elppd_rows = [r for r in _elppd_rows if r.get("split", "").startswith("holdout_stratum_")]
            if _stratum_elppd_rows:
                pd.DataFrame(_stratum_elppd_rows).to_csv(
                    os.path.join(model_dir, "stratum_elppd.csv"), index=False
                )
        if _elppd_rows:
            pd.DataFrame(_elppd_rows).to_parquet(
                os.path.join(model_dir, "posterior_elppd.parquet"), index=False
            )
    except Exception as _e:
        print(f"[warn] per-sample log-loss/ELPPD export skipped: {_e}")

    if _coverage_only and _with_elppd:
        print("--coverage_only --with_elppd: trajectory + log-loss/ELPPD artifacts written, skipping remaining exports")
        raise SystemExit(0)

    if third_deriv is not None:
        posterior_third_deriv = posterior_peaks_to_df(third_deriv, id_df["id"], metrics)
        posterior_third_deriv.to_parquet(os.path.join(model_dir, "posterior_third_deriv_ar.parquet"), index=False)

    posterior_first_deriv = posterior_to_df(jnp.transpose(first_deriv, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_first_deriv.to_parquet(os.path.join(model_dir, "posterior_first_deriv_ar.parquet"), index=False)

    if peak_val is not None:
        posterior_peak_vals = posterior_peaks_to_df(peak_val, id_df["id"], metrics)
        posterior_peak_vals.to_parquet(os.path.join(model_dir, "posterior_peak_vals_ar.parquet"), index=False)

    if _is_naive:
        _c_off_map = np.array(results_map["c_offset"]).squeeze(-1).T  # (k,n,1)→(k,n)→(n,k)
        latent_space_df = pd.DataFrame(_c_off_map, columns=[f"Dim {i+1}" for i in range(_c_off_map.shape[1])])
        latent_space_df = pd.concat([latent_space_df, id_df], axis=1)
        latent_space_df.to_parquet(os.path.join(model_dir, "latent_space.parquet"), index=False)
    else:
        latent_space_df = pd.DataFrame(results_map["X"], columns=[f"Dim {i+1}" for i in range(results_map["X"].shape[1])])
        latent_space_df = pd.concat([latent_space_df, id_df], axis=1)
        latent_space_df.to_parquet(os.path.join(model_dir, "latent_space.parquet"), index=False)

    if _is_naive:
        phi_x_latent = np.array(jnp.mean(results_mcmc["c_offset"], axis=(0, 1, -1)).T)  # (k,n)→(n,k)
    elif "hsgp" in model_name:
        phi_x = eigenfunctions_multivariate(results_map["X"],  2 * jnp.ones(basis_dims)[..., None] , approx_x_dim)
        if "spectral" in model_name:
            component_mu = results_map["mu"]
            component_scale = results_map["covariance"]
            mixture_weight = results_map["mixture_weight"]

            spd_X = make_spectral_mixture_density(hsgp_params["eigenvalues_X"], component_mu, component_scale, mixture_weight)
        else:
            lengthscale = results_map["lengthscale"]
            lengthscale_t_max = results_map["lengthscale_t_max"]
            lengthscale_c_max = results_map["lengthscale_c_max"]
            # alpha_c_max = results_map["sigma_c"]
            # alpha_t_max = results_map["sigma_t"]
            # alpha_X = results_map["alpha_X"]
            alpha_X = 1
            alpha_c_max = 1
            alpha_t_max = 1 ### lets just not care about the variance here but the correlation
            # spd_X = vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim)))(alpha_X)
            spd_X = jnp.sqrt(diag_spectral_density(basis_dims, alpha_X, lengthscale, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim))
            spd_c_max = jnp.sqrt(diag_spectral_density(basis_dims, alpha_c_max, lengthscale_c_max, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim))
            spd_t_max = jnp.sqrt(diag_spectral_density(basis_dims, alpha_t_max, lengthscale_c_max, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim))
            # spd_c_max = vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale_c_max, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim)))(alpha_c_max)
            # spd_t_max = vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale_t_max, 2 * jnp.ones(basis_dims)[..., None], approx_x_dim)))(alpha_t_max)
            phi_x_t = phi_x * spd_t_max 
            phi_x_c = phi_x * spd_c_max
        phi_x_latent = phi_x * spd_X
    elif "rflvm" in model_name:
        # The RFF map is only an internal computational approximation of the GP; the object we analyze
        # is the r-dim latent X itself (same as the linear model). So phi_X is the MAP latent X, NOT the
        # 2m-dim RFF feature map — this keeps it the r-dim Procrustes reference latent_space.r aligns
        # posterior_latent_X against, and the archetype clustering/NN run on the latent as for linear.
        phi_x_latent = results_map["X"]
    elif "linear" in model_name:
        phi_x_latent = results_map["X"]




    phi_X_df = pd.DataFrame(phi_x_latent, columns=[f"Dim {i+1}" for i in range(phi_x_latent.shape[1])])
    phi_X_df = pd.concat([phi_X_df, id_df], axis=1)
    phi_X_df.to_parquet(os.path.join(model_dir, "phi_X.parquet"), index=False)
    if not _is_naive and ("hsgplvm" in model_name) and "spectral" not in model_name:
        phi_X_df = pd.DataFrame(phi_x_t, columns=[f"Dim {i+1}" for i in range(phi_x_t.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis=1)
        phi_X_df.to_parquet(os.path.join(model_dir, "phi_X_peak_age.parquet"), index=False)
        phi_X_df = pd.DataFrame(phi_x_c, columns=[f"Dim {i+1}" for i in range(phi_x_c.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis=1)
        phi_X_df.to_parquet(os.path.join(model_dir, "phi_X_peak_value.parquet"), index=False)






