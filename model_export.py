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
from data.data_utils import create_fda_data, average_peak_differences, average_range_differences, create_surv_data
import numpyro
import jax.numpy as jnp
from model.model_utils import make_mu_rflvm, make_mu_hsgp, make_mu_linear, make_mu_rflvm_mcmc_AR, make_mu_hsgp_mcmc_AR, make_mu_linear_mcmc_AR, make_mu_linear_mcmc, compute_residuals_map, compute_priors, make_survival_linear_injury_mcmc, apply_detrend_for_offsets, make_survival_linear_mcmc
from model.inference_utils import posterior_peaks_to_df, posterior_to_df, posterior_X_to_df, posterior_injury_to_df, posterior_injury_prior_mean_to_df, posterior_survival_to_df, posterior_player_scalar_to_df
from model.hsgp import vmap_make_convex_phi, eigenfunctions_multivariate, make_spectral_mixture_density, diag_spectral_density, sqrt_eigenvalues
from visualization.visualization import make_diagnostic_heatmap, make_rhat_summary_barchart, plot_calendar_year_trends
from model.models import ConvexMaxARTVLinearLVM as _ARLinearLVM
from model.models import ConvexMaxTVLinearLVM, ConvexMaxInjuryTVLinearLVM, NaiveLinearLVM
from model.inference_utils import create_metric_trajectory_all, create_metric_trajectory_map


if __name__ == "__main__":
    from config.config_utils import resolve_model_config, parse_metrics
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_config", required=True)
    numpyro.set_platform("cpu")
    _cli = vars(parser.parse_args())
    cfg = resolve_model_config(_cli["model_config"], _cli["model_name"], inference_method="mcmc")

    model_name      = _cli["model_name"]
    _is_naive       = "naive" in model_name
    model_dir       = cfg.get("model_dir") or f"model_output/{model_name}/mcmc"
    os.makedirs(model_dir, exist_ok=True)
    mcmc_path       = os.path.join(model_dir, "samples.pkl")
    svi_path        = cfg["init_path"]
    basis_dims      = cfg["basis_dims"]
    approx_x_dim    = cfg["approx_x_dim"]
    injury          = cfg["injury"]
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
    data = pd.read_csv("data/injury_player_cleaned.csv").query(_year_filter)
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

    names = data.groupby("id")["name"].first().values.tolist()

    data["log_min"] = np.log(data["minutes"])
    data["usg"] /= 100
    data["usg"] += .01
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(data["total_games"], data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games"]) / 48
    data["retirement"] = 1
    _fake_n = age_max - age_min + 1
    fake_data = pd.DataFrame({"age": range(age_min, age_max + 1), "id": 99999999, "year": range(2000, 2000 + _fake_n), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    data = pd.concat([data, fake_data], ignore_index=True)
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

    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data.groupby(["year"]).apply(
            lambda g: (g[metric]*g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})
            
            data = data.merge(league_avg_broadcasted)
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli"]:
            data[f"{metric}_league_avg"] = data.groupby("year")[metric].transform("sum") / data.groupby("year")[exposure].transform("sum")
    
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
    _player_obs = data.groupby("id")[["draft_position_adj", "height_inches"]].first()
    _neg_log_draft = -np.log(_player_obs["draft_position_adj"].values.astype(float))
    _height_vals   = _player_obs["height_inches"].values.astype(float)
    _obs_raw = np.stack([_neg_log_draft, _height_vals], axis=1)
    _obs_mean = np.nanmean(_obs_raw, axis=0)
    _obs_std  = np.nanstd(_obs_raw, axis=0) + 1e-8
    obs_covariates = jnp.array(
        np.nan_to_num((_obs_raw - _obs_mean) / _obs_std, nan=0.0)
    )  # (n, 2), standardized; NaN (fake player, undrafted) → 0 = population mean

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
    offset_max, offset_max_var, offset_peak, offset_peak_var =  compute_priors(Y_for_offsets, exposures, metric_output, exposure_list)
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
    offset_dict = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_var, "c_max_var": offset_max_var}

    with open(svi_path, "rb") as f:
        results_map = pickle.load(f)
    f.close()
    results_map = {key.replace("__loc", ""): val for key,val in results_map.items()}
    with open(mcmc_path, "rb") as f:
        results_mcmc = pickle.load(f)
        if thin > 0:
            results_mcmc = {key: val[:, ::thin, ...] for key, val in results_mcmc.items()}
    f.close()
    results_mcmc = {**results_map, **results_mcmc}

    # Fixed MAP params (alpha, sigma_t, sigma_c, …) were not sampled by MCMC so
    # they have no chain/draw leading dims. Detect (chains, draws) from a
    # known MCMC-sampled param and broadcast any fixed param to match, so
    # vmap calls inside make_mu_*_mcmc don't fail with shape mismatches.
    _mcmc_leading = None
    for _k in ("beta", "c_max", "t_max_raw", "beta_ar", "X"):
        if _k in results_mcmc and hasattr(results_mcmc[_k], "ndim") and results_mcmc[_k].ndim >= 2:
            _mcmc_leading = results_mcmc[_k].shape[:2]  # (chains, draws)
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
    else:
        df = posterior_X_to_df(results_mcmc["X"], id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
        df.to_parquet(os.path.join(model_dir, "posterior_latent_X.parquet"), index=False)
    _summary_vars = ["sigma_beta", "sigma_beta_binomial", "sigma", "sigma_ar", "sigma_negative_binomial"]
    _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
    summary = az.summary(_summary_subset)
    print(summary)
    summary.to_parquet(os.path.join(model_dir, "posterior_variance_summary.parquet"), index=False)
    survival_injury_keys = {
        "exit_global_offset",
        "exit",
        "exit_rate",
        "injury_factor",
        "injury_exit_loading",
        "injury_exit_global_offset",
    }
    has_survival_injury = all(key in results_mcmc for key in survival_injury_keys)

    _, surv_data_set, _ = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year, age_min=age_min, age_max=age_max)
    surv_masks = jnp.stack([data_entity["censored"] for data_entity in surv_data_set], -1)
    Y_surv = jnp.stack([data_entity["observations"] for data_entity in surv_data_set], -1)


    # ── Log posterior via numpyro.infer.util.log_density ─────────────────────
    _lp_path = os.path.join(model_dir, "log_posterior.parquet")
    try:
        _n_players = covariate_X.shape[0]
        _output_shape = (_n_players, len(basis), len(metrics))
        if _is_naive:
            _lp_model = NaiveLinearLVM(latent_rank=basis_dims, output_shape=_output_shape, basis=basis)
        elif injury and "injury" in model_name:
            _lp_model = ConvexMaxInjuryTVLinearLVM(
                latent_rank=basis_dims, output_shape=_output_shape, basis=basis,
                injury_rank=5, num_injury_types=int(data["injury_code"].max()),
            )
        elif "AR" in model_name:
            _lp_model = _ARLinearLVM(latent_rank=basis_dims, output_shape=_output_shape, basis=basis)
        else:
            _lp_model = ConvexMaxTVLinearLVM(latent_rank=basis_dims, output_shape=_output_shape, basis=basis)
        _lp_model.initialize_priors(scale_values=scale_values)

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
        _lp_kwargs = {
            "data_set": data_dict,
            "inference_method": "mcmc",
            "sample_free_indices": jnp.array(player_indices),
            "sample_fixed_indices": jnp.setdiff1d(_all_idx, jnp.array(player_indices), assume_unique=True),
            "observed_covariates": obs_covariates,
            "hsgp_params": hsgp_params,
            "offsets": _lp_offsets,
            "ar_metric_indices": jnp.where(jnp.array(de_trend_indices))[0],
            "year_indices": year_indices,
            "num_years": num_years,
            "num_de_trend": len(de_trend_metrics),
            "ref_year_idx": _ref_year_idx,
        }
        if _mcmc_leading is not None:
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
            _lj = jax.vmap(
                lambda p: _log_density(_lp_model.model_fn, (), _lp_kwargs, {**results_map, **p})[0]
            )(_flat_s)
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
        print(item, results_mcmc[item].shape)
        if item == "X":
            if "X_free" in results_mcmc:
                X_new = jnp.tile(results_mcmc["X"][None, None], (1, 50, 1, 1))
                X_new = X_new.at[..., jnp.array(player_indices), :].set(results_mcmc["X_free"])
                results_mcmc["X"] = X_new
            if "hsgp" in model_name:
                results_mcmc["X"] = jnp.tanh(results_mcmc["X"]) * 1.9

    # Augment MAP and MCMC X with fixed observed covariates for trajectory/survival computations.
    # posterior_X_to_df above uses the raw latent X (already saved); augmentation only affects
    # the make_mu_* and survival utility calls below.
    # Naive model has no latent X; X_map_aug / X_mcmc_aug are left as None.
    if not _is_naive:
        X_map_aug = jnp.concatenate([results_map["X"], obs_covariates], axis=-1)
        _obs_bc = jnp.broadcast_to(
            obs_covariates[None, None],
            results_mcmc["X"].shape[:-1] + (obs_covariates.shape[-1],),
        )
        X_mcmc_aug = jnp.concatenate([results_mcmc["X"], _obs_bc], axis=-1)
    else:
        X_map_aug = None
        X_mcmc_aug = None

    if _is_naive:
        _c_off_map = results_map["c_offset"]    # (k, n, 1)
        _s_map     = results_map["sigma_ar"]    # (k, 1)
        _r_map     = results_map["rho_ar"]      # (k, 1)
        _z_map     = results_map["beta_ar"]     # (j, k, n)
        _a0_map    = results_map["AR_0"] * (_s_map / jnp.sqrt(1 - _r_map ** 2))
        _ar_map    = _ARLinearLVM._compute_ar_process_from_parameters(_s_map, _r_map, _z_map, _a0_map)
        mu = jnp.repeat(_c_off_map, repeats=len(basis), axis=-1) + _ar_map  # (k, n, j)
    elif "rflvm" in model_name:
        mu, *_ = make_mu_rflvm(results_map["X"], 3 + results_map["lengthscale_deriv"], results_map["alpha"], results_map["beta"],
                                                results_map["W"], results_map["W_t_max"], results_map["W_c_max"],  results_map["lengthscale"], results_map["lengthscale_t_max"], results_map["lengthscale_c_max"], results_map["c_max"], results_map["t_max_raw"], 
                                                results_map["sigma_t"],
                                                results_map["sigma_c"], 
                                                # offset_dict["t_max_var"],
                                                # offset_dict["c_max_var"],
                                                L_time, M_time, phi_time, x_time + L_time, offset_dict)
    elif "hsgplvm" in model_name: 
        mu, *_ = make_mu_hsgp(results_map["X"], 3 + results_map["lengthscale_deriv"], results_map["alpha"], results_map["alpha_X"], results_map["beta"], results_map["lengthscale"],
                              results_map["lengthscale_c_max"], results_map["lengthscale_t_max"],  
                              results_map["c_max"], results_map["t_max_raw"], 
                            #   results_map["sigma_t"],
                            #   results_map["sigma_c"], 
                            offset_dict["t_max_var"],
                            offset_dict["c_max_var"],
                              L_time, M_time, phi_time, x_time + L_time, offset_dict, basis_dims, 2 * jnp.ones(basis_dims)[..., None] ,approx_x_dim )
    elif "linear" in model_name:
        _sigma_c_eff = results_map["sigma_c"] * jnp.sqrt(jnp.asarray(offset_dict["c_max_var"]))
        _sigma_c_mcmc_eff = results_mcmc["sigma_c"] * jnp.sqrt(jnp.asarray(offset_dict["c_max_var"]))
        mu, *_ = make_mu_linear(X_map_aug, 3 + results_map["lengthscale_deriv"], results_map["alpha"], results_map["beta"], results_map["c_max"], results_map["t_max_raw"],
                                results_map["sigma_t"],
                                _sigma_c_eff,
                                  L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict)
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
    obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                        
    avg_sd, autocorr, lognormal_params, beta_params = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, metrics, results_map["sigma"], results_map.get("sigma_negative_binomial", 0),
                                                                results_map.get("sigma_beta_binomial", 0), results_map.get("sigma_beta", 1))
    
    # avg_sd = jnp.ones((len(metrics))) * .01
    # autocorr = jnp.zeros_like(avg_sd)

    if "rflvm" in model_name:
        wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_rflvm_mcmc_AR(results_mcmc["X"], 3 + results_mcmc["lengthscale_deriv"], results_mcmc["alpha"],
                            results_mcmc["beta"], results_mcmc["W"], results_mcmc["W_t_max"], results_mcmc["W_c_max"], results_mcmc["lengthscale"], results_mcmc["lengthscale_t_max"], results_mcmc["lengthscale_c_max"],  results_mcmc["c_max"],
                            results_mcmc["t_max_raw"], offset_dict["t_max_var"],
                              offset_dict["c_max_var"], L_time, M_time, x_time + L_time, offset_dict, approx_x_dim,
                            
                            sigma_ar = results_mcmc["sigma_ar"],
                            # sigma_ar = avg_sd[..., None][None, None],
                            beta_ar = results_mcmc["beta_ar"], 
                            rho_ar=results_mcmc["rho_ar"],
                            # rho_ar = autocorr[..., None][None, None],
                            AR_0_raw=results_mcmc["AR_0"],
                            # AR_0_raw = jnp.zeros((len(metrics), covariate_X.shape[0])),
                            phi_time=phi_time, orthogonalize=False)
    elif "hsgplvm" in model_name:
        wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_hsgp_mcmc_AR(results_mcmc["X"],  3 + results_mcmc["lengthscale_deriv"], 
                              results_mcmc["alpha"], results_mcmc["alpha_X"], results_mcmc["beta"], results_mcmc["lengthscale"],
                              results_mcmc["lengthscale_c_max"], results_mcmc["lengthscale_t_max"],  
                              results_mcmc["c_max"], results_mcmc["t_max_raw"], offset_dict["t_max_var"],
                              offset_dict["c_max_var"], L_time, M_time, phi_time, x_time + L_time, offset_dict,
                              basis_dims, 2 * jnp.ones(basis_dims)[..., None] ,approx_x_dim,
                            sigma_ar = results_mcmc["sigma_ar"],
                            # sigma_ar = avg_sd[..., None][None, None],
                            beta_ar = results_mcmc["beta_ar"], 
                            rho_ar=results_mcmc["rho_ar"],
                            # rho_ar = autocorr[..., None][None, None],
                            AR_0_raw=results_mcmc["AR_0"],
                            # AR_0_raw = jnp.zeros((len(metrics), covariate_X.shape[0])),
                             orthogonalize=False)
    elif "linear" in model_name:
        if ("AR" in model_name) or injury:
            wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_linear_mcmc_AR(X_mcmc_aug, 3 + results_mcmc["lengthscale_deriv"],
                                results_mcmc["alpha"], results_mcmc["beta"],
                                results_mcmc["c_max"], results_mcmc["t_max_raw"], results_mcmc["sigma_t"],
                                _sigma_c_mcmc_eff, L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict,
                                sigma_ar = results_mcmc["sigma_ar"],
                                # sigma_ar = avg_sd[..., None][None, None],
                                beta_ar = results_mcmc["beta_ar"], 
                                rho_ar=results_mcmc["rho_ar"],
                                # rho_ar = autocorr[..., None][None, None],
                                AR_0_raw=results_mcmc["AR_0"],
                                # AR_0_raw = jnp.zeros((len(metrics), covariate_X.shape[0])),
                                orthogonalize=False)
        else:
            wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_linear_mcmc(X_mcmc_aug, 3 + results_mcmc["lengthscale_deriv"],
                                results_mcmc["alpha"], results_mcmc["beta"],
                                results_mcmc["c_max"], results_mcmc["t_max_raw"], results_mcmc["sigma_t"],
                                _sigma_c_mcmc_eff, L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict)
    elif _is_naive:
        def _naive_one_draw(c_off, s, r, z, a0_raw):
            a0 = a0_raw * (s / jnp.sqrt(1 - r ** 2))
            ar = _ARLinearLVM._compute_ar_process_from_parameters(s, r, z, a0)   # (k, n, j)
            return jnp.repeat(c_off, repeats=z.shape[0], axis=-1), ar           # (k, n, j) each

        mu_mcmc, AR = vmap(vmap(_naive_one_draw))(
            results_mcmc["c_offset"],   # (chains, draws, k, n, 1)
            results_mcmc["sigma_ar"],   # (chains, draws, k, 1)
            results_mcmc["rho_ar"],     # (chains, draws, k, 1)
            results_mcmc["beta_ar"],    # (chains, draws, j, k, n)
            results_mcmc["AR_0"],       # (chains, draws, k, n)
        )
        tmax_mcmc   = None
        cmax_mcmc   = None
        third_deriv = None
        first_deriv = jnp.zeros_like(mu_mcmc)

    # Reconstruct MCMC calendar-year TREND_AR — shape (chains, draws, k, n, j)
    _has_year_ar_mcmc = _has_year_ar and all(k in results_mcmc for k in _year_ar_keys)
    if _has_year_ar_mcmc:
        def _ar3_one_draw(s, r, z, a0):
            a0_scaled = a0 * s[None, :, 0]                                      # (1, num_ar)
            traj = _ARLinearLVM._compute_ar1_calendar_process(s, r, z, a0_scaled)  # (num_ar, num_years)
            trend_nj = traj[:, year_indices]                                     # (num_ar, n, j)
            out = jnp.zeros((len(metrics),) + year_indices.shape)
            return out.at[_ar_global_indices].set(trend_nj)                      # (k, n, j)

        _ar3_mcmc = vmap(vmap(_ar3_one_draw))  # maps over (chains, draws)
        TREND_AR_mcmc = _ar3_mcmc(
            results_mcmc["sigma_year_ar"],   # (chains, draws, num_ar, 1)
            results_mcmc["rho_year_ar"],     # (chains, draws, num_ar, 1)
            results_mcmc["beta_year_ar"],    # (chains, draws, num_years, num_ar)
            results_mcmc["AR_0_year"],       # (chains, draws, 1, num_ar)
        )  # (chains, draws, k, n, j)
    else:
        TREND_AR_mcmc = de_trend_adjusted   # fallback: broadcast (k, n, j)

    latent_val = mu_mcmc + AR + TREND_AR_mcmc
    if _is_naive:
        _peak_idx = jnp.argmax(latent_val, axis=-1)                                     # (chains, draws, k, n)
        tmax_mcmc = jnp.swapaxes(jnp.array(basis)[_peak_idx] - basis.mean(), -1, -2)   # (chains, draws, n, k)
        cmax_mcmc = jnp.swapaxes(jnp.max(latent_val, axis=-1), -1, -2)                 # (chains, draws, n, k)
    if injury:
        injury_loading = results_mcmc["injury_loading"]
        injury_factor = results_mcmc["injury_factor"]
        injury_mean_prior = jnp.einsum("...ip, ...kp -> ...ki", injury_factor, injury_loading)
        # (chains, draws, k, i)
        _injury_global_offset = results_mcmc.get("injury_global_offset", jnp.zeros(injury_mean_prior.shape[-2]))
        _sigma_injury = results_mcmc.get("sigma_injury")       # (chains, draws, k) or None
        _injury_time_raw = results_mcmc.get("injury_time_raw") # (chains, draws, j, i) or None
        if _sigma_injury is not None and _injury_time_raw is not None:
            injury_effect_raw = (
                injury_mean_prior[:, :, :, None, None, :]                                              # (chains, draws, k, 1, 1, i)
                + _injury_global_offset[:, :, :, None, None, None]                                    # (chains, draws, k, 1, 1, 1)
                + _sigma_injury[:, :, :, None, None, None] * _injury_time_raw[:, :, None, None, :, :] # (chains, draws, k, 1, j, i)
            )  # (chains, draws, k, 1, j, i)
        else:
            injury_effect_raw = (
                injury_mean_prior[:, :, :, None, None, :]
                + _injury_global_offset[:, :, :, None, None, None]
            )  # (chains, draws, k, 1, 1, i) — decay model fallback
        injury_effect_padded = jnp.concatenate(
            [jnp.zeros(injury_effect_raw.shape[:-1] + (1,), dtype=injury_effect_raw.dtype),
             injury_effect_raw],
            axis=-1
        )  # (..., k, 1, T, i+1) — take_along_axis broadcasts over n
        injury_effect = jnp.take_along_axis(injury_effect_padded, injury_types[..., None][None, None], -1).squeeze(-1)
        latent_val = latent_val + injury_effect

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

        # Export player-specific injury effect (already selected by injury_type) — shape (chains, draws, k, n, j)
        injury_effect_player_df = posterior_to_df(
            jnp.transpose(injury_effect, (0, 1, 3, 4, 2)),  # → (chains, draws, n, j, k)
            id_df["id"],
            metrics,
            range(age_min, age_max + 1),
        )
        injury_effect_player_df.to_parquet(os.path.join(model_dir, "posterior_injury_effect.parquet"), index=False)

        injury_prior_mean_export = injury_mean_prior
        injury_prior_metrics = list(metrics)
        if "injury_exit_loading" in results_mcmc:
            injury_exit_prior_mean = jnp.einsum("...ip, ...p -> ...i", injury_factor, results_mcmc["injury_exit_loading"])
            if "injury_exit_global_offset" in results_mcmc:
                injury_exit_prior_mean = injury_exit_prior_mean + results_mcmc["injury_exit_global_offset"][..., None]
            injury_prior_mean_export = jnp.concatenate(
                [injury_prior_mean_export, injury_exit_prior_mean[:, :, None, :]],
                axis=2,
            )
            injury_prior_metrics = injury_prior_metrics + ["exit_hazard"]
        if "injury_scale_loading" in results_mcmc:
            injury_scale_prior_mean = jnp.einsum("...ip, ...p -> ...i", injury_factor, results_mcmc["injury_scale_loading"])
            if "injury_scale_global_offset" in results_mcmc:
                injury_scale_prior_mean = injury_scale_prior_mean + results_mcmc["injury_scale_global_offset"][..., None]
            injury_prior_mean_export = jnp.concatenate(
                [injury_prior_mean_export, injury_scale_prior_mean[:, :, None, :]],
                axis=2,
            )
            injury_prior_metrics = injury_prior_metrics + ["exit_scale"]

        injury_prior_df = posterior_injury_prior_mean_to_df(
            injury_prior_mean_export,
            injury_prior_metrics,
            injury_type_ids,
            injury_type_labels,
        )
        injury_prior_df.to_parquet(os.path.join(model_dir, "posterior_injury_prior_mean.parquet"), index=False)

        # Export global injury offsets (per metric + survival) separately
        _go = np.array(_injury_global_offset)                          # (chains, draws, k)
        _n_chains, _n_draws, _k = _go.shape
        _ci, _si, _ki = np.meshgrid(np.arange(_n_chains), np.arange(_n_draws), np.arange(_k), indexing="ij")
        global_offset_df = pd.DataFrame({
            "chain":  _ci.ravel(),
            "sample": _si.ravel(),
            "metric": np.array(list(metrics))[_ki.ravel()],
            "value":  _go.ravel(),
        })
        _ci2, _si2 = np.meshgrid(np.arange(_n_chains), np.arange(_n_draws), indexing="ij")
        if "injury_exit_global_offset" in results_mcmc:
            _ego = np.array(results_mcmc["injury_exit_global_offset"])
            global_offset_df = pd.concat([global_offset_df, pd.DataFrame({
                "chain": _ci2.ravel(), "sample": _si2.ravel(),
                "metric": "exit_hazard", "value": _ego.ravel(),
            })], ignore_index=True)
        if "injury_scale_global_offset" in results_mcmc:
            _sgo = np.array(results_mcmc["injury_scale_global_offset"])
            global_offset_df = pd.concat([global_offset_df, pd.DataFrame({
                "chain": _ci2.ravel(), "sample": _si2.ravel(),
                "metric": "exit_scale", "value": _sgo.ravel(),
            })], ignore_index=True)
        global_offset_df.to_parquet(os.path.join(model_dir, "posterior_injury_global_offset.parquet"), index=False)
    else:
        injury_effect = jnp.zeros_like(latent_val)

    if has_survival_injury and injury:
            surv_posterior = make_survival_linear_injury_mcmc(
                X=X_mcmc_aug,
                exit_global_offset=results_mcmc["exit_global_offset"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_factor=results_mcmc["injury_factor"],
                injury_exit_loading=results_mcmc["injury_exit_loading"],
                injury_exit_global_offset=results_mcmc["injury_exit_global_offset"],
                injury_scale_loading=results_mcmc["injury_scale_loading"],
                injury_scale_global_offset=results_mcmc["injury_scale_global_offset"],
                sigma_injury_scale=results_mcmc["sigma_injury_scale"],
                injury_scale_raw=results_mcmc["injury_scale_raw"],
                injury_indicator=injury_masks,
                injury_type=injury_types,
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc["sigma_exit_scale"],
                scale_global_log=results_mcmc.get("scale_global_log", jnp.log(11.5)),
                age_min=age_min,
            )

            observed_surv_df = pd.DataFrame(
                {
                    "player": id_df["id"].to_numpy(),
                    "observed_entrance_age": np.asarray(Y_surv[:, 0]),
                    "observed_exit_age": np.asarray(Y_surv[:, 1]),
                    "exit_censored": np.asarray(surv_masks[:, 1]).astype(np.int32),
                }
            )

            surv_posterior_counterfactual = make_survival_linear_injury_mcmc(
                X=X_mcmc_aug,
                exit_global_offset=results_mcmc["exit_global_offset"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_factor=results_mcmc["injury_factor"],
                injury_exit_loading=results_mcmc["injury_exit_loading"],
                injury_exit_global_offset=results_mcmc["injury_exit_global_offset"],
                injury_scale_loading=results_mcmc["injury_scale_loading"],
                injury_scale_global_offset=results_mcmc["injury_scale_global_offset"],
                sigma_injury_scale=results_mcmc["sigma_injury_scale"],
                injury_scale_raw=results_mcmc["injury_scale_raw"],
                injury_indicator=jnp.zeros_like(injury_masks),
                injury_type=jnp.zeros_like(injury_types),   # type=0 → true no-injury baseline
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc["sigma_exit_scale"],
                scale_global_log=results_mcmc.get("scale_global_log", jnp.log(11.5)),
                age_min=age_min,
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

            exit_age_sample_df_obs = posterior_player_scalar_to_df(
                surv_posterior["exit_age_sample"],
                id_df["id"],
                "exit_age_sample",
            )
            exit_age_sample_df_obs = exit_age_sample_df_obs.merge(observed_surv_df, on="player", how="left")
            exit_age_sample_df_obs["scenario"] = "observed"
            exit_age_sample_df_cf = posterior_player_scalar_to_df(
                surv_posterior_counterfactual["exit_age_sample"],
                id_df["id"],
                "exit_age_sample",
            )
            exit_age_sample_df_cf = exit_age_sample_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_age_sample_df_cf["scenario"] = "counterfactual"
            pd.concat([exit_age_sample_df_obs, exit_age_sample_df_cf], ignore_index=True).to_parquet(
                os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False
            )
    elif _is_naive:
        # Naive survival: per-player Weibull with no latent X.
        # concentration = 1 + 2*sigmoid(exit_global_offset), scale = exp(scale_global_log).
        _naive_surv_key = jax.random.PRNGKey(42)
        def _naive_surv_one_draw(key, exit_global_off, scale_global_log, entrance_times):
            concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_global_off.squeeze(-1))  # (n,)
            scale = jnp.exp(scale_global_log.squeeze(-1))                              # (n,)
            tenure_grid = jnp.maximum(basis - age_min, 1e-6)                          # (j,)
            # Conditional survival: S(t|entrance) = S(t)/S(entrance)
            def _surv(t, c, s): return jnp.exp(-jnp.power(jnp.maximum(t, 1e-6) / s, c))
            s_grid     = vmap(lambda c, s: _surv(tenure_grid, c, s))(concentration, scale)  # (n, j)
            s_entrance = vmap(lambda c, s, e: _surv(e, c, s))(concentration, scale, entrance_times)[:, None]  # (n, 1)
            exit_survival = s_grid / jnp.maximum(s_entrance, 1e-8)                    # (n, j)
            exit_hazard   = vmap(lambda c, s: (c / s) * jnp.power(jnp.maximum(tenure_grid / s, 1e-6), c - 1))(concentration, scale)
            # Weibull inverse-CDF conditioned on T > entrance: t = scale * (target + (ent/scale)^k)^(1/k)
            u = jnp.clip(jax.random.uniform(key, shape=concentration.shape), 1e-6, 1.0 - 1e-6)
            target = -jnp.log(u)
            entrance_term = jnp.power(jnp.maximum(entrance_times, 0.0) / scale, concentration)
            sampled_duration = scale * jnp.power(jnp.maximum(target + entrance_term, 1e-6), 1.0 / jnp.maximum(concentration, 1e-6))
            sampled_duration = jnp.clip(sampled_duration, entrance_times, float(basis[-1] - age_min))
            exit_age_sample = float(age_min) + sampled_duration
            return {"exit_survival": exit_survival, "exit_hazard": exit_hazard, "exit_age_sample": exit_age_sample}

        _n_chains, _n_draws = results_mcmc["exit_global_offset"].shape[:2]
        _naive_surv_keys = jax.random.split(_naive_surv_key, _n_chains * _n_draws).reshape(_n_chains, _n_draws, 2)
        _naive_surv_vmap = vmap(vmap(lambda k, a, b: _naive_surv_one_draw(k, a, b, Y_surv[:, 0] - age_min + 1e-6)))
        surv_posterior = _naive_surv_vmap(
            _naive_surv_keys,
            results_mcmc["exit_global_offset"],  # (chains, draws, n, 1)
            results_mcmc["scale_global_log"],    # (chains, draws, n, 1)
        )

        observed_surv_df = pd.DataFrame(
            {
                "player": id_df["id"].to_numpy(),
                "observed_entrance_age": np.asarray(Y_surv[:, 0]),
                "observed_exit_age": np.asarray(Y_surv[:, 1]),
                "exit_censored": np.asarray(surv_masks[:, 1]).astype(np.int32),
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

        exit_age_sample_df_obs = posterior_player_scalar_to_df(
            surv_posterior["exit_age_sample"],
            id_df["id"],
            "exit_age_sample",
        )
        exit_age_sample_df_obs = exit_age_sample_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_age_sample_df_obs["scenario"] = "observed"
        exit_age_sample_df_obs.to_parquet(os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False)
    else:
        surv_posterior = make_survival_linear_mcmc(
                X=X_mcmc_aug,
                exit_global_offset=results_mcmc["exit_global_offset"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                entrance_times=Y_surv[:, 0] - age_min + 1e-6,
                basis=basis,
                sigma_exit_scale=results_mcmc.get("sigma_exit_scale", 1.0),
                scale_global_log=results_mcmc.get("scale_global_log", jnp.log(11.5)),
                age_min=age_min,
            )

        observed_surv_df = pd.DataFrame(
            {
                "player": id_df["id"].to_numpy(),
                "observed_entrance_age": np.asarray(Y_surv[:, 0]),
                "observed_exit_age": np.asarray(Y_surv[:, 1]),
                "exit_censored": np.asarray(surv_masks[:, 1]).astype(np.int32),
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

        exit_age_sample_df_obs = posterior_player_scalar_to_df(
            surv_posterior["exit_age_sample"],
            id_df["id"],
            "exit_age_sample",
        )
        exit_age_sample_df_obs = exit_age_sample_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_age_sample_df_obs["scenario"] = "observed"

        exit_age_sample_df_obs.to_parquet(
            os.path.join(model_dir, "posterior_exit_age_sample.parquet"), index=False
        )


    players = id_df[id_df["name"].isin(predict_players)].index
    player_names = id_df[id_df["name"].isin(predict_players)]["name"].tolist()

    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)

    players_idx = jnp.array(id_df.index)
    ages = list(range(age_min, age_max + 1))
    n_players_sel = len(id_df)
    n_ages = len(ages)



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




    _summary_vars = ["sigma_beta", "sigma_beta_binomial"]
    _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
    summary = az.summary(_summary_subset)
    summary.to_parquet(os.path.join(model_dir, "posterior_variance_summary.parquet"), index=False)

    peaks    = tmax_mcmc + basis.mean() if tmax_mcmc is not None else None
    peak_val = cmax_mcmc if cmax_mcmc is not None else None
    
    _neg_bin_samples = jnp.transpose(results_mcmc["sigma_negative_binomial"], (2, 0, 1)) if "sigma_negative_binomial" in results_mcmc else None
    _, pos = create_metric_trajectory_all(latent_val, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    _, pos_mu = create_metric_trajectory_all(mu_mcmc + TREND_AR_mcmc, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    

    posterior_df = posterior_to_df(pos, id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_df.to_parquet(os.path.join(model_dir, "posterior_ar.parquet"), index=False)

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
    _holdout_pct_obs = jnp.array(validation_mask) & ~jnp.isnan(_pct_min_obs)

    _games_pivot = (
        data.pivot_table(index="id", columns="age", values="games", aggfunc="first")
        .reindex(index=id_df["id"].tolist(), columns=range(age_min, age_max + 1))
    )
    _games_obs = jnp.array(_games_pivot.values.astype(np.float64))  # (n, j)
    _holdout_games_obs = jnp.array(validation_mask) & ~jnp.isnan(_games_obs)

    Y_conditional = (
        Y
        .at[minutes_index].set(jnp.where(_holdout_pct_obs, _pct_min_obs, Y[minutes_index]))
        .at[games_index].set(jnp.where(_holdout_games_obs, _games_obs, Y[games_index]))
    )
    _, pos_conditional = create_metric_trajectory_all(
        latent_val, Y_conditional, exposures,
        metric_output, metrics, exposure_list,
        jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
        jnp.transpose(results_mcmc["sigma_beta"], (2, 0, 1)),
        posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
        posterior_neg_bin_samples=_neg_bin_samples,
        condition_on_observed=True,
    )
    posterior_conditional_df = posterior_to_df(pos_conditional, id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_conditional_df.to_parquet(os.path.join(model_dir, "posterior_ar_conditional.parquet"), index=False)

    if peaks is not None:
        posterior_peaks = posterior_peaks_to_df(peaks, id_df["id"], metrics)
        posterior_peaks.to_parquet(os.path.join(model_dir, "posterior_peaks_ar.parquet"), index=False)

    if ("AR" in model_name) or injury or _is_naive:
        posterior_ar_df = posterior_to_df(jnp.transpose(AR, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(age_min, age_max + 1))
        posterior_ar_df.to_parquet(os.path.join(model_dir, "posterior_latent_ar.parquet"), index=False)

    if injury and ("counterfactual" not in model_name):
        _, pos_counterfactual = create_metric_trajectory_all(mu_mcmc + AR + TREND_AR_mcmc, Y, exposures,
                                        metric_output, metrics, exposure_list,
                                        jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                        jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                        posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                        posterior_neg_bin_samples=_neg_bin_samples,
                                        )
        posterior_counterfactual_df = posterior_to_df(pos_counterfactual, id_df["id"], metrics, range(age_min, age_max + 1))
        posterior_counterfactual_df.to_parquet(os.path.join(model_dir, "posterior_counterfactual_ar.parquet"), index=False)

    posterior_mu_df = posterior_to_df(jnp.transpose(mu_mcmc + TREND_AR_mcmc, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(age_min, age_max + 1))
    posterior_mu_df.to_parquet(os.path.join(model_dir, "posterior_mu_ar.parquet"), index=False)

    # Export calendar-year AR(3) trend
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
        wTx = jnp.einsum("nr,mr -> nm", results_map["X"], results_map["W"]  * jnp.sqrt(results_map["lengthscale"]))
        phi_x_latent = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(approx_x_dim))  
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






