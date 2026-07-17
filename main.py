import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

_xla_flags = os.environ.get("XLA_FLAGS", "")
for _flag in ["--xla_force_host_platform_device_count=2", "--xla_cpu_multi_thread_eigen=false"]:
    if _flag not in _xla_flags:
        _xla_flags = f"{_xla_flags} {_flag}".strip()
# for _flag in [f"--xla_force_host_platform_device_count={os.cpu_count()}"]:
#     if _flag not in _xla_flags:
#         _xla_flags = f"{_xla_flags} {_flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

import yaml  # noqa: F401 — kept for any direct yaml use below
from config.config_utils import resolve_model_config, parse_metrics
import pandas as pd
import numpy as np
import jax
import flax.serialization as ser
import jax.numpy as jnp
import matplotlib.pyplot as plt
import re
import argparse
import pickle
import scipy
import numpyro
from matplotlib import cm 
from matplotlib.colors import Normalize
from functools import partial
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
from numpyro.diagnostics import print_summary
from numpyro.distributions import LogNormal, Weibull, Beta, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform
from model.hsgp import  diag_spectral_density, make_psi_gamma,  vmap_make_convex_phi, vmap_make_convex_phi_prime, sqrt_eigenvalues
jax.config.update("jax_enable_x64", True)
from model.inference_utils import get_latent_sites, create_metric_trajectory_map
from model.model_utils import compute_residuals_map, compute_priors, apply_detrend_for_offsets, compute_linear_predictor_mean_offsets, summarize_metric_error_observed_substitutions, summarize_metric_error_injury_splits, summarize_normalized_weighted_metric_residuals_by_age, write_coverage_tables
from data.data_utils import create_fda_data, create_surv_data, create_validation_mask
from model.inference_inputs import dispatch_model, fda_injury_flag


def _curves_under_substitute(model, model_args, params, include_derivs=False):
    """Run model.compute_curves under numpyro substitute with `params` — the single-source curve
    reconstruction (replaces the retired make_mu_* duplicates) for main.py's init/MAP diagnostics."""
    import numpyro
    _ca = (model_args["hsgp_params"], model_args["offsets"],
           model_args["sample_free_indices"], model_args["sample_fixed_indices"],
           model_args.get("ar_metric_indices", jnp.array([])), model_args.get("year_indices", jnp.array([])),
           model_args.get("num_years", 1), model_args.get("num_de_trend", 0), model_args.get("ref_year_idx", 0))
    return numpyro.handlers.substitute(
        numpyro.handlers.seed(lambda: model.compute_curves(*_ca, include_derivs=include_derivs), jax.random.PRNGKey(0)),
        data=params)()
from model.models import  ConvexMaxInjuryTVLinearLVM, ConvexMaxTVLinearLVM, ConvexMaxDecayInjuryTVLinearLVM, ConvexMaxTVRFLVM, NaiveLinearLVM, ConvexMaxARTVRFLVM, ConvexMaxARTVLinearLVM, ConvexMaxLinearTrendTVLinearLVM, ConvexMaxLKJTVLinearLVM, ConvexMaxARLKJTVLinearLVM, ConvexMaxTVCosineLVM, ConvexMaxARTVCosineLVM, TVLinearLVM, TVLinearLVM_AR, LKJTVLinearLVM, LKJTVLinearLVM_AR
from visualization.visualization import plot_posterior_predictive_career_trajectory_map, plot_prior_predictive_career_trajectory, plot_prior_mean_trajectory, plot_calendar_year_trends


def make_pca_pipeline(n_components=2, whiten=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=whiten))
    ])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA aging curve model')
    parser.add_argument('--model_name', required=True, help='model entry key in model_config.yaml')
    parser.add_argument('--model_config', required=True, help='path to model_config.yaml')
    parser.add_argument('--inference_method', required=False, default=None,
        choices=["mcmc", "svi", "map", "cut_mcmc"],
        help='inference regime; selects the matching regimes block in model_config.yaml. '
             'cut_mcmc (injury models only): two-block cut-posterior Gibbs — counterfactual latents '
             'NUTS-updated on the injury-free likelihood, injury params on the full likelihood')
    parser.add_argument('--eval_only', action='store_true', default=False,
        help='skip training; load saved samples.pkl and re-run coverage calculations only')
    parser.add_argument('--mcmc_init', required=False, default=None,
        choices=["map", "median", "median_nofix"],
        help='override mcmc_init from config: map (seed from MAP) | median (init_to_median, fixed plug-ins from MAP) | median_nofix (fully MAP-free)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
        help='override a prior knob, e.g. --set sigma_X=0.3 / --set alpha=HalfNormal:1.0 / '
             '--set use_c_offset_re=0 / --set sigma_c@usg=0.2 (repeatable)')
    _cli = vars(parser.parse_args())

    args = resolve_model_config(_cli["model_config"], _cli["model_name"], _cli.get("inference_method"))
    args["model_name"] = _cli["model_name"]
    # CLI knob overrides layer on top of any config prior_knobs (used for prior-tuned fits).
    from model.inference_inputs import knobs_from_overrides as _knobs_from_overrides
    _cli_knobs = _knobs_from_overrides(_cli.get("overrides", []))
    # non-prior run params can also come via --set (e.g. --set map_num_steps=50000)
    for _ra in ("map_num_steps",):
        if _ra in _cli_knobs:
            args[_ra] = int(_cli_knobs.pop(_ra))
    if _cli_knobs:
        args["prior_knobs"] = {**(args.get("prior_knobs") or {}), **_cli_knobs}
    _config_fixed_params = args.pop("fixed_params", [])

    eval_only = _cli.get("eval_only", False)
    inference_method = args["inference_method"]
    map_inference = (inference_method == "map")
    svi_inference = (inference_method == "svi")
    # cut_mcmc is a variant of mcmc (same init/fixed-param loading and downstream sample handling);
    # it only swaps the sampler for the two-block cut-posterior Gibbs on the injury models.
    cut_mcmc_inference = (inference_method == "cut_mcmc")
    mcmc_inference = (inference_method in ("mcmc", "cut_mcmc"))
    prior_predictive = False  # retired: prior-predictive draws + checks now live in prior_check.py
    num_warmup, num_samples, num_chains = args["num_warmup"], args["num_samples"], args["num_chains"]
    mcmc_init_strategy = _cli.get("mcmc_init") or args.get("mcmc_init", "map")   # CLI overrides config
    thinning = int(num_samples / (args.get("thinning") or 250))
    initial_params_path = args["init_path"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    approx_x_dim = args["approx_x_dim"]
    injury = args["injury"]
    censor_survival_at_injury = args.get("censor_survival_at_injury", False)
    age_min = args["age_min"]
    age_max = args["age_max"]
    start_year = args.get("start_year")
    end_year = args.get("end_year")
    m_time = args.get("m_time") or 15
    basis_dims_2 = args["basis_dims_2"]
    param_path = args["fixed_param_path"]
    vectorized = args["vectorized"]
    # 'median_nofix': fully MAP-free MCMC — sample every param (no fixed plug-ins) and init_to_median.
    # Clear the fixed-param set and both MAP file paths so nothing is loaded from a MAP fit.
    if mcmc_init_strategy == "median_nofix":
        _config_fixed_params = []
        param_path = None
        initial_params_path = None
    model_dir = args.get("model_dir") or f"model_output/{model_name}/{inference_method}"
    os.makedirs(model_dir, exist_ok=True)
    _plots_dir = os.path.join(model_dir, "plots")
    players = args["player_names"]
    de_trend_metrics = args["de_trend_metrics"]
    debug_nan = args["debug_nan"]
    validation_year = args["validation_year"]
    cohort_year = args["cohort_year"]
    validation_scheme = args["validation_scheme"]
    if validation_scheme == "none":
        validation_scheme = None
    holdout_fraction = args["holdout_fraction"]
    holdout_k = args["holdout_k"]
    holdout_seed = args["holdout_seed"]
    position_group = args["position_group"]
    _year_filter = f"age <= {age_max} & name != 'Brandon Williams'"
    if start_year is not None:
        _year_filter += f" & year >= {start_year}"
    if end_year is not None:
        _year_filter += f" & year <= {end_year}"
    data_all = pd.read_csv("data/injury_player_cleaned.csv").query(_year_filter)
    data_all["split"] = np.random.choice(["train", "test"], size=len(data_all), p=[0.8, 0.2])
    # data_all = data_all.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year
    data_all["first_major_injury"] = (
        data_all["first_major_injury"]
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
    data_all['first_major_injury'] = (
    data_all['first_major_injury']
            .astype('category')
            .cat.set_categories(
                ['None'] +
                [c for c in pd.unique(data_all['first_major_injury']) if c != 'None'],
                ordered=False
            ))
    data_all["injury_code"] = data_all["first_major_injury"].cat.codes
    data_all["log_min"] = np.log(data_all["minutes"])
    data_all["usg"] /= 100
    data_all["usg"] += .01
    data_all["simple_exposure"] = 1
    data_all["games_exposure"] = np.maximum(data_all["total_games"], data_all["games"]) ### 82 or whatever
    data_all["pct_minutes"] = (data_all["minutes"] / data_all["games"]) / 48
    data_all["retirement"] = 1

    metrics, metric_output, exposure_list = parse_metrics(args)

    scale_values = jnp.ones((len(metrics), 1))

    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data_all.groupby(["year"]).apply(
            lambda g: (g[metric]*g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})

            data_all = data_all.merge(league_avg_broadcasted)
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli"]:
            data_all[f"{metric}_league_avg"] = data_all.groupby("year")[metric].transform("sum") / data_all.groupby("year")[exposure].transform("sum")



    data = data_all 

   

    _fake_n = age_max - age_min + 1
    fake_data = pd.DataFrame({"age": range(age_min, age_max + 1), "id": 99999999, "year": range(2000, 2000 + _fake_n), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    fake_data["draft_position_adj"] = 39
    fake_data["height_inches"] = 79
    data = pd.concat([data, fake_data], ignore_index=True)
    names = data.groupby("id")["name"].first().values.tolist()
    validation_mask = data[["year", "age", "id"]].pivot(columns="age", index="id", values=f"year").reindex(columns = range(age_min,age_max + 1)).apply(
                                                                        lambda r: r.dropna().iloc[0] + (np.array(range(age_min,age_max + 1)) - r.dropna().index[0]) if r.notna().any() else r,
                                                                        axis=1,
                                                                        result_type="expand").to_numpy() > validation_year
    # validation_mask = data[["split","age", "id"]].pivot(columns="age", index="id", values="split").reindex(columns = range(age_min,age_max + 1)).to_numpy() == "test"
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

    _, surv_data_set, basis = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year, age_min=age_min, age_max=age_max)
    surv_masks = jnp.stack([data_entity["censored"] for data_entity in surv_data_set], -1)
    censor = jnp.stack([data_entity["censor_type"] for data_entity in surv_data_set], -1)
    Y_surv = jnp.stack([data_entity["observations"] for data_entity in surv_data_set], -1)

    if censor_survival_at_injury:
        # Censor the survival time at the first injury onset for each player.
        # Players without injury are unaffected; the fake player (id=99999999) has no
        # injury_period != "pre-injury" rows, so it is also unaffected.
        _onset = (
            data_all[data_all["injury_period"] != "pre-injury"]
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
                    _surv_masks_np[_i, 1] = True   # right-censor at injury onset
        Y_surv = jnp.array(_Y_surv_np)
        surv_masks = jnp.array(_surv_masks_np)

    surv_data_dict = {}
    surv_data_dict["observations"] = Y_surv - age_min
    surv_data_dict["censored"] = surv_masks
    surv_data_dict["censor_type"] = censor

    covariate_X, data_set, basis = create_fda_data(
        data, basis_dims, metric_output, metrics, exposure_list, [],
        injury=fda_injury_flag(model_name, injury), validation_year=validation_year, age_min=age_min, age_max=age_max)
    model = dispatch_model(
        model_name, latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)),
        basis=basis, injury=injury, num_injury_types=int(data["injury_code"].max()),
        rff_dim=approx_x_dim, prior_knobs=(args.get("prior_knobs") or {}))


    # Attribute knobs (e.g. x_latent_df) must be set before initialize_priors; dict knobs after.
    from model.inference_inputs import apply_prior_knobs as _apply_prior_knobs, _ATTRIBUTE_KNOBS as _ATTR_KNOBS, _build_knob_value as _bkv
    _prior_knobs = args.get("prior_knobs", {}) or {}
    for _k in _ATTR_KNOBS:
        if _k in _prior_knobs:
            setattr(model, _k, _bkv(_prior_knobs[_k]))
    model.initialize_priors(scale_values = scale_values)
    _apply_prior_knobs(model, _prior_knobs, metrics=metrics)
    initial_params = {}
    initial_map_init = None
    map_fixed_params_loc = {}
    if "lvm" in model_name or "naive" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            results_param = {key.replace("__loc", ""): val for key, val in results_param.items()}
            if (mcmc_inference or svi_inference or map_inference):
                _to_fix = set(_config_fixed_params)
                for param_name, value in results_param.items():
                    if param_name in _to_fix:
                        prior_dict[param_name] = numpyro.deterministic(param_name, value)
                        map_fixed_params_loc[f"{param_name}__loc"] = value
        if initial_params_path:
            if (mcmc_inference or svi_inference):
                with open(initial_params_path, "rb") as f_init:
                    initial_params = pickle.load(f_init)
                f_init.close()
                ### we're usually initializing from autodelta in the numpyro sense so the loc suffix is present. need for all guide models but not for mcmc since 
                ### the sample site is used and the name needs to match
                initial_params = {key.replace("__loc",""):val for key,val in initial_params.items()}
                if (len(player_indices) > 0):
                    initial_params["X_free"] = initial_params["X"][jnp.array(player_indices)]
            elif map_inference:
                with open(initial_params_path, "rb") as f_init:
                    initial_blob = f_init.read()
                f_init.close()
                try:
                    loaded_init = pickle.loads(initial_blob)
                except Exception:
                    loaded_init = initial_blob

                if isinstance(loaded_init, dict) and (("state" in loaded_init) or ("samples" in loaded_init)):
                    if loaded_init.get("state") is not None:
                        initial_map_init = loaded_init["state"]
                    else:
                        initial_map_init = loaded_init.get("samples")
                else:
                    initial_map_init = loaded_init

        model.prior.update(prior_dict)
        distribution_families = set([data_entity["output"] for data_entity in data_set])
        distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
        masks = jnp.stack([data_entity["mask"] for data_entity in data_set]) * (~validation_mask[None])
        _ref = np.array(data_set[0]["output_data"])
        _train_scheme = None if validation_scheme == "all" else validation_scheme
        holdout_mask = jnp.array(create_validation_mask(_ref, _train_scheme, holdout_fraction, holdout_k, holdout_seed, age_min=age_min))
        masks = masks * (~holdout_mask[None])
        if _train_scheme is not None:
            _hm = np.asarray(holdout_mask)
            _pids = data[["year", "age", "id"]].pivot(columns="age", index="id", values="year").index.values
            _ages = list(range(age_min, age_max + 1))
            _hi_rows, _hi_cols = np.where(_hm)
            _ho_ages_arr = np.array(_ages)[_hi_cols]
            _ho_df_dict: dict = {"player": _pids[_hi_rows], "age": _ho_ages_arr}
            if _train_scheme == "stratified_next_k":
                # The full career tail is held out of training, but only the first
                # holdout_k seasons of each player's tail are scored. score_window=1 marks
                # those scored cells; the remaining tail cells are held out of training yet
                # excluded from both scoring and the in-sample set. Stratum is the player's
                # cohort (derived from the holdout-start age), constant across the tail.
                _score_mask = np.zeros_like(_hm)
                _ho_start_age = np.full(_hm.shape[0], -1, dtype=int)
                for _r in range(_hm.shape[0]):
                    _cols = np.where(_hm[_r])[0]
                    if len(_cols):
                        _start = int(_cols[0])
                        _score_mask[_r, _start:_start + holdout_k] = _hm[_r, _start:_start + holdout_k]
                        _ho_start_age[_r] = _ages[_start]
                _strata_ends = [24, 26, 28, 30, 32, 34]
                def _stratum_of(_a):
                    return next((s for s, end in enumerate(_strata_ends, 1) if int(_a) <= end), None)
                _ho_df_dict["stratum"] = [_stratum_of(_ho_start_age[_r]) for _r in _hi_rows]
                _ho_df_dict["score_window"] = _score_mask[_hi_rows, _hi_cols].astype(int)
            pd.DataFrame(_ho_df_dict).to_csv(
                os.path.join(model_dir, "holdout_indices.csv"), index=False
            )
            # Censor survival at last in-sample age for each holdout player so the
            # survival model does not see exit ages that fall inside the held-out window.
            _Y_surv_h = np.array(Y_surv)
            _smasks_h = np.array(surv_masks)
            _ages_arr = np.array(_ages)
            for _pi in range(_hm.shape[0]):
                _ho_ages = _ages_arr[_hm[_pi]]
                if len(_ho_ages) == 0:
                    continue
                _last_in = max(float(_ho_ages.min()) - 1.0, float(_Y_surv_h[_pi, 0]))
                if _last_in < float(_Y_surv_h[_pi, 1]):
                    _Y_surv_h[_pi, 1] = _last_in
                    _smasks_h[_pi, 1] = True
            Y_surv     = jnp.array(_Y_surv_h)
            surv_masks = jnp.array(_smasks_h)
        injury_masks = jnp.stack([data_entity["injury_mask"] for data_entity in data_set])
        using_injury_model = isinstance(model, (ConvexMaxInjuryTVLinearLVM))
        should_mask_injury_values = ((injury and not using_injury_model) or ("counterfactual" in model_name))
        if should_mask_injury_values:
            masks = masks * (~injury_masks)
        injury_types = jnp.stack([data_entity["injury_type"] for data_entity in data_set]).astype(jnp.int32)
        exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
        Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
        # yao_index = names.index("Yao Ming")
        # print(masks[0,yao_index], Y[0, yao_index], injury_masks[3, yao_index], injury_types[3, yao_index], yao_index)
        # raise ValueError
        # Y_linearized = []
        # exp_linearized = []
        # for data_entity in data_set:
        #     family = data_entity["output"]
        #     Y_obs = data_entity["output_data"]
        #     exposure_obs = data_entity["exposure_data"]
        #     if family == "gaussian":
        #         Y_linearized.append(Y_obs)
        #         exp_linearized.append(jnp.square(exposure_obs))
        #     elif family in ["negative-binomial", "poisson"]:
        #         Y_linearized.append(jnp.log(Y_obs + 1))
        #         exp_linearized.append(jnp.exp(exposure_obs))
        #     elif family in ["binomial", "beta-binomial", "bernoulli"]:
        #         p = (Y_obs + .5) / (exposure_obs + 1)
        #         Y_linearized.append(jnp.log(p)/jnp.log(1 - p))
        #         exp_linearized.append(exposure_obs)
        #     elif family in ["beta"]:
        #         Y_linearized.append(jnp.log(Y_obs) / jnp.log(1 - Y_obs))
        #         exp_linearized.append(jnp.square(exposure_obs) - 1)
        # Y_linearized = jnp.stack(Y_linearized)
        # exp_linearized = jnp.stack(exp_linearized)
        # W_eff = exp_linearized * masks
        # Y_safe = jnp.nan_to_num(Y_linearized, nan=0.0, posinf=0.0, neginf=0.0)
        # Y_safe = Y_safe.reshape(Y_safe.shape[1], - 1)
        # W_eff = jnp.nan_to_num(1 / W_eff.reshape(W_eff.shape[1], -1), nan=0.0, posinf=0.0, neginf=0.0)
        # mu = (Y_safe * W_eff).sum(axis=0) / (W_eff.sum(axis=0))

        # Z = (Y_safe - mu) * W_eff
        de_trend = jnp.stack([data_entity["de_trend"] for data_entity in data_set])
        de_trend = jnp.where(jnp.array(de_trend_indices)[..., None, None], de_trend, 0.0)
        # Calendar-year indices for learned AR(3) trend (same across all metrics)
        year_matrix = jnp.array(data_set[0]["year_matrix"])   # (n, j)
        min_year = int(jnp.nanmin(year_matrix))
        max_year = int(jnp.nanmax(year_matrix))
        num_years = max_year - min_year + 1
        year_indices = jnp.nan_to_num(year_matrix - min_year, nan=0).astype(int)  # (n, j)
        # Reference year: pin TREND_AR = 0 at the first year with actual observed data.
        # year_matrix is extrapolated across all ages, so min_year can be significantly
        # earlier than the first real data year (e.g. min_year=1962, first data=1980).
        # Using an unobserved extrapolated year as reference is meaningless — use the
        # first year that actually appears in the raw data instead.
        first_observed_year = int(data_all["year"].min())
        ref_year_idx = first_observed_year - min_year  # e.g. 1980 - 1962 = 18
        print(f"Calendar AR reference year: {first_observed_year} (index {ref_year_idx} of {num_years})")
        # offset_list = []
        # offset_max_list = []
        # offset_peak_list = []
        # offset_boundary_l = []
        # offset_boundary_r = []
        # for index, family in enumerate(metric_output):  
        #     weight = exposures[index]
        #     individual_weight = jnp.nansum(weight, -1)
        #     if family == "gaussian":
        #         p = jnp.nansum(Y[index] * weight) / jnp.nansum(weight)
        #         offset_list.append(p)
        #         p_max = jnp.nansum(jnp.nanmax(Y[index], -1) * individual_weight) / jnp.nansum(individual_weight)
        #         p_max_var = jnp.nansum(jnp.square(jnp.nanmax(Y[index], -1) - p_max) * individual_weight) / jnp.nansum(individual_weight)
        #         offset_max_list.append( p_max)
        #         peak = jnp.nansum(jnp.nanargmax(Y[index], -1) * individual_weight) / jnp.nansum(individual_weight)
                
        #         boundary_l, boundary_r = average_peak_differences(Y[index])
        #         offset_boundary_l.append(boundary_l)
        #         offset_boundary_r.append(boundary_r)
        #     else:
        #         if family in ["poisson", "negative-binomial"]:
        #             individual_weight = jnp.exp(individual_weight)
        #             p = jnp.nansum(Y[index]) / jnp.nansum(jnp.exp(exposures[index]))
        #             p_max = jnp.nansum(jnp.nanmax(Y[index] / jnp.exp(exposures[index]), -1) * individual_weight) / jnp.nansum(individual_weight)
        #             p_max_var = jnp.nansum(jnp.square(jnp.nanmax(Y[index]/ jnp.exp(exposures[index]), -1) - p_max) * individual_weight) / jnp.nansum(individual_weight)
        #             peak = jnp.nansum(jnp.nanargmax(Y[index] / jnp.exp(exposures[index]), -1) * individual_weight) / jnp.nansum(individual_weight)
        #             offset_list.append(jnp.log(p))
        #             offset_max_list.append(jnp.log(p_max))                 
        #         elif family in ["beta-binomial", "binomial", "bernoulli"]:
        #             p = jnp.nansum(Y[index]) / jnp.nansum(exposures[index])
        #             p_max = jnp.nansum(jnp.nanmax(Y[index] / exposures[index], -1) * individual_weight) / jnp.nansum(individual_weight) if exposure_list[index] != "simple_exposure" else .5
        #             p_max_var = jnp.nansum(individual_weight * jnp.square(jnp.nanmax(Y[index] / exposures[index], -1)- p_max)) / jnp.nansum(individual_weight)
        #             offset_list.append(jnp.log(p/ (1-p)))
        #             offset_max_list.append(jnp.log(p_max/(1-p_max)))
        #             peak = jnp.nansum(jnp.nanargmax(Y[index] / exposures[index], -1) * individual_weight) / jnp.nansum(individual_weight) if exposure_list[index] != "simple_exposure" else jnp.argmax(jnp.nanmean(Y[index] / exposures[index], 0))
        #             p_star = Y[index] / exposures[index]     
        #         elif family == "beta":
        #             weight  = jnp.square(weight)
        #             individual_weight = jnp.nansum(weight, -1)
        #             p = jnp.nansum(Y[index] * weight) / jnp.nansum(weight)
        #             p_max = jnp.nansum(jnp.nanmax(Y[index], -1) * individual_weight) / jnp.nansum(individual_weight)
        #             p_max_var = jnp.nansum(jnp.square(jnp.nanmax(Y[index], -1) - p_max) * individual_weight) / jnp.nansum(individual_weight)
        #             peak = jnp.nansum(jnp.nanargmax(Y[index], -1) * individual_weight) / jnp.nansum(individual_weight)
        #             offset_list.append(jnp.log(p / (1 - p)))
        #             offset_max_list.append(jnp.log(p_max/(1-p_max)))
         

        #     print(peak + 18, p_max, jnp.sqrt(p_max_var), metrics[index])
            # offset_peak_list.append(peak + 18 - basis.mean())
        # raise ValueError
    
        Y_for_offsets = apply_detrend_for_offsets(
            Y_obs=Y,
            exposures_obs=exposures,
            metric_families=metric_output,
            de_trend_values=de_trend,
            de_trend_mask=jnp.array(de_trend_indices),
        )

        # Keep offset preprocessing aligned with the global masking logic above:
        # when --injury is disabled we keep injury rows; when enabled (or in
        # counterfactual mode) they are already removed from `masks`.
        offset_mask = masks

        family_requires_positive_exposure = jnp.array(
            [
                family in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli", "beta", "gaussian"]
                for family in metric_output
            ],
            dtype=bool,
        )[:, None, None]
        finite_mask = jnp.isfinite(Y_for_offsets) & jnp.isfinite(exposures)
        positive_exposure_mask = (~family_requires_positive_exposure) | (exposures > 0)
        offset_valid_mask = offset_mask & finite_mask & positive_exposure_mask

        Y_for_offsets_masked = jnp.where(offset_valid_mask, Y_for_offsets, jnp.nan)
        exposures_for_offsets = jnp.where(offset_valid_mask, exposures, jnp.nan)
        offset_linear_predictor_mean = compute_linear_predictor_mean_offsets(
            Y_for_offsets_masked,
            exposures_for_offsets,
            metric_output,
        )

        offset_max, offset_max_var, offset_peak_absolute, offset_peak_absolute_var, offset_mean = compute_priors(
            Y_for_offsets_masked,
            exposures_for_offsets,
            metric_output,
            exposure_list,
        )

        offset_peak_absolute = offset_peak_absolute + age_min - basis.mean() + 2.0  # +2 forward-shift to debias right-truncation
        print("offset_max:", offset_max)
        print("offset_max_var:", offset_max_var)
        print("offset_peak_absolute:", offset_peak_absolute)
        print("offset_peak_absolute_var:", offset_peak_absolute_var)
        print("offset_linear_predictor_mean:", offset_linear_predictor_mean)
        # offset_peak = offset_peak / 20
        # offset_peak -= .5
        # offset_peak *= 2
        # offset_peak_var /= 400
        # offsets = jnp.array(offset_list)[None]
        # offset_max = jnp.array(offset_max_list)[None]
        # offset_peak = jnp.array(offset_peak_list)[None]
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
            family_dict["de_trend"] = de_trend[indices]  # kept for RFLVM model compatibility
            data_dict[family] = family_dict
        hsgp_params = {}
        if "convex" in model_name:
                
                x_time = basis - basis.mean()
                # x_time = (basis - jnp.min(basis)) 
                # x_time /= jnp.max(x_time)
                # x_time -= jnp.mean(x_time)
                # x_time *= 2
                L_time = 2 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
                t_amplitude = float(jnp.squeeze(L_time)) / 2
                print(f"L_time: {L_time}, x_time: {x_time}, t_amplitude: {t_amplitude}")
                M_time = m_time
                phi_time = vmap_make_convex_phi(jnp.squeeze(x_time), jnp.squeeze(L_time), M_time)
                hsgp_params["phi_x_time"] = phi_time
                hsgp_params["M_time"] = M_time
                hsgp_params["L_time"] = L_time
                hsgp_params["t_amplitude"] = t_amplitude
                hsgp_params["shifted_x_time"] = x_time + L_time
                hsgp_params["t_0"] = jnp.min(x_time)
                hsgp_params["t_r"] = jnp.max(x_time)
                if "hsgp" in model_name:
                    hsgp_params["eigenvalues_X"] = sqrt_eigenvalues(2 *  jnp.ones(basis_dims)[..., None] , approx_x_dim, basis_dims)

        # Observed covariates: -log(draft_position_adj), height_inches, position one-hot
        # Player order matches create_basis groupby sort on id
        player_obs = data.groupby("id")[["draft_position_adj", "height_inches", "position_group"]].first()
        neg_log_draft = -np.log(player_obs["draft_position_adj"].values.astype(float))
        height_vals   = player_obs["height_inches"].values.astype(float)
        obs_numeric = np.stack([neg_log_draft, height_vals], axis=1)  # (n, 2)
        obs_mean = np.nanmean(obs_numeric, axis=0)
        obs_std  = np.nanstd(obs_numeric, axis=0) + 1e-8
        obs_numeric_std = (obs_numeric - obs_mean) / obs_std          # (n, 2), standardized
        pos_dummies = pd.get_dummies(player_obs["position_group"], drop_first=True).astype(float).values  # (n, 2): F, G vs C baseline
        obs_covariates = jnp.array(np.concatenate([obs_numeric_std, pos_dummies], axis=1))  # (n, 4)
        model.player_covariates = obs_covariates

        model_args = {"data_set": data_dict,  "inference_method": inference_method, "sample_free_indices": jnp.array(player_indices),
                      "sample_fixed_indices": jnp.setdiff1d(jnp.arange(covariate_X.shape[0]), jnp.array(player_indices), assume_unique=True)}
        # Calendar-year trend params: all LinearLVM models accept these
        last_observed_year = int(data_all["year"].max())
        year_max_idx = last_observed_year - min_year
        # Capability gate (was `"linear" in model_name`): pass the calendar-year AR-trend kwargs to
        # any model whose forward routes through compute_curves — the ConvexMaxTVLinearLVM family,
        # now incl. the RFF leaves (names contain "tvrflvm", not "linear"). Naive ignores the extra
        # kwargs harmlessly; the non-convex linear models already received them under the old gate.
        if hasattr(model, "compute_curves"):
            model_args["ar_metric_indices"] = jnp.where(jnp.array(de_trend_indices))[0]
            model_args["year_indices"] = year_indices
            model_args["num_years"] = num_years
            model_args["num_de_trend"] = len(de_trend_metrics)
            model_args["ref_year_idx"] = ref_year_idx
        if "linear_trend" in model_name:
            model_args["year_max_idx"] = year_max_idx
        

       



        model_args["offsets"] = {}
        # Y_surv[:, 0] = draft_age (always observed); Y_surv[:, 1] = last observed age
        entrance_times = Y_surv[:, 0]
        exit_times = Y_surv[:, 1]
        right_censor = surv_masks[:, 1]
        model_args["offsets"].update({"exit_times": exit_times - age_min + 1e-6, "entrance_times": entrance_times - age_min + 1e-6, "right_censor": right_censor})
        model_args["offsets"]["injury_indicator"] = injury_masks
        model_args["offsets"]["injury_type"] = injury_types
        model_args.update({"hsgp_params": hsgp_params})
        if "tvlinearlvm" in model_name and "convex" not in model_name:
            model_args["offsets"].update({
                "c_mean": offset_mean,
                "c_max_var": offset_max_var,
            })
        if "convex" in model_name:
            if "max" in model_name:
                model_args["offsets"].update({"t_max": offset_peak_absolute, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_absolute_var, "c_max_var": offset_max_var})
                if "AR" in model_name and (len(initial_params) > 0) & (inference_method == "mcmc"):
                    # Reconstruct mu from the MAP init via the model's OWN forward (single source).
                    _d_init = _curves_under_substitute(model, model_args, initial_params)
                    mu = _d_init["mu"] + _d_init["trend_ar"]
                    obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                    
                    avg_sd, autocorr, lognormal_params, beta_params = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, metrics, initial_params["sigma"], initial_params.get("sigma_negative_binomial",1),
                                                            initial_params.get("sigma_beta_binomial", 0), initial_params.get("sigma_beta",1),
                                                            ar_metric_indices=jnp.array(de_trend_indices))
                    print(avg_sd, autocorr)
                        
        if not eval_only:
            if map_inference:
                samples, state = model.run_map_inference(num_steps = args.get("map_num_steps", 50000), model_args=model_args, initial_state=initial_map_init)
            elif prior_predictive:
                print("sampling from prior")
                samples = model.predict({}, model_args, num_samples = num_samples)
            elif mcmc_inference:
                if os.environ.get("DEBUG_INIT"):
                    import sys as _sys
                    from numpyro.handlers import substitute as _subst, trace as _trace, seed as _seed
                    _tr = _trace(_subst(_seed(model.model_fn, jax.random.PRNGKey(0)), data=initial_params)).get_trace(**model_args)
                    print(f"[DEBUG_INIT] tracing {len(_tr)} sites at MAP init point")
                    for _nm, _site in _tr.items():
                        _v = _site.get("value")
                        try: _vfin = bool(jnp.all(jnp.isfinite(jnp.asarray(_v))))
                        except Exception: _vfin = True
                        _lpfin = True
                        if _site["type"] == "sample":
                            try: _lpfin = bool(jnp.all(jnp.isfinite(_site["fn"].log_prob(_v))))
                            except Exception as _e: _lpfin = False; print(f"  LPERR {_nm}: {_e}")
                        if (not _vfin) or (not _lpfin):
                            _a = jnp.asarray(_v)
                            print(f"  BAD {_site['type']:13s} {_nm:22s} shape={_a.shape} val_finite={_vfin} lp_finite={_lpfin} nan={int(jnp.isnan(_a).sum())} inf={int(jnp.isinf(_a).sum())}")
                    for _nm,_s in _tr.items():
                        if _s.get("is_observed") and "neg" in _nm.lower():
                            _fn=_s["fn"]
                            for _attr in ["mean","rate","concentration"]:
                                try:
                                    _av=np.asarray(getattr(_fn,_attr))
                                    print(f"  [REAL {_nm}.{_attr}] min={np.nanmin(_av):.3e} max={np.nanmax(_av):.3e} nan={int(np.isnan(_av).sum())} n={_av.size}")
                                except Exception as _e: print(f"   {_nm}.{_attr} err {_e}")
                    print("[DEBUG_INIT] forward done; checking gradients")
                    from numpyro.infer.util import unconstrain_fn as _uncon, potential_energy as _pe
                    _ssites = {n for n,s in _tr.items() if s["type"]=="sample" and not s.get("is_observed")}
                    _initc = {k:v for k,v in initial_params.items() if k in _ssites}
                    print("[DEBUG_INIT] sample sites:", sorted(_ssites))
                    print("[DEBUG_INIT] init covers:", sorted(_initc), " MISSING:", sorted(_ssites - set(_initc)))
                    _u = _uncon(model.model_fn, (), model_args, _initc)
                    _val, _g = jax.value_and_grad(lambda p: _pe(model.model_fn, (), model_args, p))(_u)
                    print(f"[DEBUG_INIT] potential_energy={_val}")
                    for _k,_gv in _g.items():
                        _ga=jnp.asarray(_gv)
                        if not bool(jnp.all(jnp.isfinite(_ga))):
                            print(f"  NAN-GRAD {_k:22s} shape={_ga.shape} nan={int(jnp.isnan(_ga).sum())} inf={int(jnp.isinf(_ga).sum())}")
                    _cre_g = jnp.asarray(_g.get("c_offset_re"))
                    _idx = np.asarray(jnp.argwhere(jnp.isnan(_cre_g)))
                    print(f"[DEBUG_INIT] c_offset_re NaN-grad cells: {_idx[:8].tolist()}")
                    _mu_full = np.asarray(_curves_under_substitute(model, model_args, initial_params)["mu"])
                    print(f"[DEBUG_INIT] reconstructed mu shape={_mu_full.shape}")
                    # Quantify negbin/poisson mean (exp(log_rate)) over OBSERVED cells — underflow check
                    for _fam,_ks in [("negbin",[11,12,13]),("poisson",[5,6,7,8,9,10])]:
                        _lrs=[]
                        for _k in _ks:
                            _Yk=np.asarray(Y[_k]); _ek=np.asarray(exposures[_k]); _mk=_mu_full[_k]
                            _ok=np.isfinite(_Yk)
                            _lrs.append((_mk+_ek)[_ok])
                        _lr=np.concatenate(_lrs)
                        _mean=np.exp(_lr)
                        print(f"  [{_fam}] obs cells={_lr.size} log_rate[min,max]=[{_lr.min():.1f},{_lr.max():.1f}] "
                              f"mean[min,max]=[{_mean.min():.2e},{_mean.max():.2e}] "
                              f"#underflow(mean=0)={int((_mean==0).sum())} #overflow(mean=inf)={int(np.isinf(_mean).sum())} "
                              f"#log_rate<-50={int((_lr<-50).sum())} #log_rate>50={int((_lr>50).sum())}")
                    for _row in _idx[:5]:
                        _n,_k = int(_row[0]), int(_row[1])
                        try:
                            _Yr = np.asarray(Y[_k,_n]); _er = np.asarray(exposures[_k,_n])
                            _obs_j = np.where(np.isfinite(_Yr))[0]
                            print(f"  cell n={_n} k={_k} metric={metrics[_k]} family={metric_output[_k]} "
                                  f"cre={float(initial_params['c_offset_re'][_n,_k]):.3f} obs_ages_idx={_obs_j.tolist()} Yobs={_Yr[_obs_j].tolist()}")
                            if _mu_full is not None:
                                _mu = _mu_full[_k,_n,:]
                                _lr = _mu + _er
                                print(f"    mu[min,max]=[{np.nanmin(_mu):.2f},{np.nanmax(_mu):.2f}]  "
                                      f"log_rate[min,max]=[{np.nanmin(_lr):.2f},{np.nanmax(_lr):.2f}]  "
                                      f"mean=exp(lr)[min,max]=[{np.exp(np.nanmin(_lr)):.3e},{np.exp(np.nanmax(_lr)):.3e}]")
                                for _j in _obs_j:
                                    print(f"      obs age_idx={_j}: mu={_mu[_j]:.3f} exp={_er[_j]:.3f} log_rate={_lr[_j]:.3f} mean={np.exp(_lr[_j]):.3e}")
                        except Exception as _e:
                            print("   inspect err", _e)
                    # Isolate: does the NaN grad come from the convex construction (make_mu_linear) alone?
                    def _mu_cell(tmr, bet, cm, Xp):
                        _p = {**initial_params, "t_max_raw": tmr, "beta": bet, "c_max": cm, "X": Xp}
                        return _curves_under_substitute(model, model_args, _p)["mu"][13, 591, :].sum()
                    _gt = jax.grad(_mu_cell, argnums=(0,1,2,3))(initial_params["t_max_raw"], initial_params["beta"], initial_params["c_max"], initial_params["X"])
                    for _nm,_gg in zip(["t_max_raw","beta","c_max","X"], _gt):
                        _ga=jnp.asarray(_gg)
                        print(f"  [construction grad] {_nm:10s} finite={bool(jnp.all(jnp.isfinite(_ga)))} nan={int(jnp.isnan(_ga).sum())}")
                    print("[DEBUG_INIT] enabling jax_debug_nans to locate the offending op...")
                    import traceback as _tb
                    jax.config.update("jax_debug_nans", True)
                    try:
                        _ = jax.grad(lambda p: _pe(model.model_fn, (), model_args, p))(_u)
                        print("  no NaN raised under jax_debug_nans (??)")
                    except FloatingPointError as _fe:
                        _tb.print_exc()
                    jax.config.update("jax_debug_nans", False)
                    print("[DEBUG_INIT] done"); _sys.exit(0)
                if cut_mcmc_inference:
                    if not isinstance(model, ConvexMaxInjuryTVLinearLVM):
                        raise ValueError("inference_method=cut_mcmc requires an injury model (ConvexMaxInjuryTVLinearLVM family)")
                    print("[MCMC] cut posterior: counterfactual block on the injury-free likelihood, "
                          "injury block on the full likelihood (init_to_median; MAP seeding not supported, "
                          "fixed plug-in params still applied)")
                    samples, mcmc_model = model.run_cut_gibbs_inference(
                        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                        model_args=model_args, thinning=thinning)
                else:
                    _mcmc_init = {} if mcmc_init_strategy in ("median", "median_nofix") else initial_params
                    print({
                        "map": "[MCMC] init: seed NUTS from MAP samples",
                        "median": "[MCMC] init: init_to_median; no MAP seeding (fixed plug-in params still from MAP)",
                        "median_nofix": "[MCMC] init: init_to_median, fully MAP-free (all params sampled, no fixed plug-ins)",
                    }.get(mcmc_init_strategy, f"[MCMC] init: {mcmc_init_strategy}"))
                    samples, mcmc_model = model.run_inference(num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, vectorized=vectorized,
                    model_args=model_args, initial_values=_mcmc_init, thinning=thinning)

            elif svi_inference:
                samples = model.run_svi_inference(num_steps=30000, guide_kwargs={}, model_args=model_args, initial_values=initial_params,
                                                            sample_shape = (num_chains, num_samples), debug_nan=debug_nan)
        else:
            samples_path = os.path.join(model_dir, "samples.pkl")
            print(f"eval_only: loading samples from {samples_path}")
            with open(samples_path, "rb") as f:
                samples = pickle.load(f)
            map_inference = True
            mcmc_inference = False
            svi_inference = False
            prior_predictive = False
    # if mcmc_inference:
        # mcmc_model.print_summary()


    if map_inference and len(map_fixed_params_loc) > 0:
        samples = dict(samples)
        for param_name, param_value in map_fixed_params_loc.items():
            if param_name not in samples:
                samples[param_name] = param_value

    # Parameters fixed to 1 in the model are not sampled sites, so they're absent from `samples`.
    # Inject them so reconstruction here (and in model_export via samples.pkl) doesn't KeyError.
    # setdefault is idempotent: a run where these were free keeps its fitted values.
    if map_inference:
        samples = dict(samples)
        if "lengthscale_deriv__loc" in samples:
            samples.setdefault("alpha__loc", jnp.ones_like(jnp.asarray(samples["lengthscale_deriv__loc"])))
        for _k in ("sigma_c__loc", "sigma_t__loc", "sigma_exit_scale__loc", "sigma_c_offset__loc"):
            samples.setdefault(_k, 1.0)
        samples.setdefault("sigma_t_offset__loc", 1.0)   # must match prior["sigma_t_offset"] in models.py
        samples.setdefault("sigma_curve__loc", 0.5)      # must match prior["sigma_curve"] in models.py

    if not eval_only:
        if not prior_predictive:
            with open(os.path.join(model_dir, "samples.pkl"), "wb") as f:
                pickle.dump(samples, f)
            f.close()
            print("saved samples")
            if map_inference:
                with open(os.path.join(model_dir, "state.pkl"), "wb") as f:
                    f.write(ser.to_bytes(state))
                f.close()
                print("saved state")


    if map_inference:
        if "max" in model_name:
            print("sigma", samples["sigma__loc"])
            alpha_time = samples["alpha__loc"]
            print("alpha_time", alpha_time)
            # Reconstruct via the model's OWN forward (single source): compute_curves rebuilds X via
            # _resolve_latent_X_structured, projects with the model's _project_X (linear X OR cosine
            # sqrt(r)*X/||X||), and applies the SAME /sqrt(r) (level/peak) and /r (curvature)
            # normalization + curve_amp as the fit. Replaces the hand-written forward that had desynced
            # from models.py (missing /sqrt(r) on t_max; /sqrt(r) instead of /r on curvature; psi_x=X
            # for cosine; sigma_X defaulting to 1.0). Mirrors the non-convex tvlinearlvm branch below.
            _map_data = {k[:-5]: v for k, v in samples.items() if k.endswith("__loc")}
            _d_map = _curves_under_substitute(model, model_args, _map_data, include_derivs=False)
            mu = _d_map["mu"]                                   # (k, n, t)
            X = _d_map["X"]                                     # raw latent (n, r)
            psi_x = _d_map["psi_x"]                             # projected features (identity=r for linear,
            #                                                    cosine=r, RFF=2*m); used by the survival +
            #                                                    injury reconstructions below (single source)
            X_center = X - jnp.mean(X, keepdims=True, axis=0)
            t_max = np.asarray(_d_map["t_max"])                 # (n, k) centered
            c_max = np.asarray(_d_map["c_max"])                 # (n, k)
        
            if injury:
                if ("counterfactual" not in model_name) and ("injury_loading__loc" in samples):
                    injury_loading = samples["injury_loading__loc"]
                    injury_factor = samples["injury_factor__loc"]
                    injury_mean_prior = jnp.einsum("ip, kp -> ki", injury_factor, injury_loading) ### remove the no-injury effect
                    injury_global_offset = samples.get("injury_global_offset__loc", jnp.zeros((injury_mean_prior.shape[0],)))
                    injury_player_x = samples.get("injury_player_x__loc", jnp.zeros((X.shape[-1], len(metrics), int(data["injury_code"].max()) )))
                    injury_indicator = model_args["offsets"]["injury_indicator"]
                    injury_type = model_args["offsets"]["injury_type"]
                    is_decay_model = "decay" in model_name
                    if is_decay_model:
                        # beta_0 is (k, i) — no per-player noise
                        injury_effect_component = (
                            injury_global_offset[:, None] + injury_mean_prior
                        )  # (k, i)
                    else:
                        injury_time_raw_map = samples.get("injury_time_raw__loc", jnp.zeros((len(basis), int(data["injury_code"].max()))))  # (j, i)
                        sigma_injury_map = samples.get("sigma_injury__loc", jnp.zeros(len(metrics)))  # (k,)
                        injury_effect_component = (
                            injury_global_offset[:, None, None, None]                                        # (k, 1, 1, 1)
                            + injury_mean_prior[:, None, None, :]                                            # (k, 1, 1, i)
                            + sigma_injury_map[:, None, None, None] * injury_time_raw_map[None, None, :, :]  # (k, 1, j, i)
                        )  # (k, 1, j, i) — uniform over players
                        injury_effect_padded = jnp.concatenate(
                            [jnp.zeros(injury_effect_component.shape[:-1] + (1,), dtype=injury_effect_component.dtype),
                             injury_effect_component],
                            axis=-1
                        )  # (k, 1, j, i+1) — take_along_axis broadcasts over n
                        full_mask = (injury_indicator * masks)[..., None]
                        avg_injury_effect = (injury_effect_padded * full_mask).sum(axis=(1, 2)) / full_mask.sum(axis=(1, 2))
                        injury_effect = jnp.take_along_axis(injury_effect_padded, injury_type[..., None], -1).squeeze(-1) * injury_indicator
                    injuries = data["first_major_injury"].cat.categories[1:]
                    injury_effect_data = pd.DataFrame(injury_mean_prior + injury_global_offset[:, None], columns = injuries, index = metrics)
                    plot_metric_names = list(metrics)
                    plot_metric_types = list(metric_output)
                    injury_exit_mean_prior = None
                    if "injury_exit_loading__loc" in samples:
                        injury_player_exit = samples.get("injury_player_exit__loc", jnp.zeros((psi_x.shape[0], len(metrics))))
                        injury_exit_loading = samples["injury_exit_loading__loc"]
                        injury_exit_global_offset = samples.get("injury_exit_global_offset__loc", 0.0)
                        injury_exit_mean_prior = injury_exit_global_offset + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)  # (i,) prior mean only
                    # ax = injury_effect_data.plot(kind = "bar", title = "Injury Effect by Metric")
                    # fig = ax.get_figure()
                    # Save to file

                    has_hazard_effect_panel = (injury_exit_mean_prior is not None)
                    num_metrics = len(plot_metric_names)
                    total_panels = num_metrics + int(has_hazard_effect_panel)
                    ncols = 4
                    nrows = (total_panels + ncols - 1) // ncols
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
                    axes = axes.flatten()

                    for i, metric in enumerate(plot_metric_names):
                        ax = axes[i]
                        
                        # Get link function for this metric
                        label = ""
                        metric_type = plot_metric_types[i]
                        if metric_type == "gaussian":
                            link_fn = lambda x: x 
                            label = "Change in Outcome"
                        elif metric_type in ["poisson", "negative-binomial"]:
                            link_fn = lambda x: np.exp(x) - 1
                            label = "% Change in Rate"
                        elif metric_type in ["beta", "beta-binomial", "binomial"]:
                            link_fn = lambda x: np.exp(x) - 1
                            label = "% Change in Odds"
                        
                        # Apply link function to the row (injuries)
                        transformed_values = link_fn(injury_effect_data.loc[metric])
                        
                        # Plot
                        transformed_values.plot(kind="bar", ax=ax, title=metric)
                        ax.set_ylabel(f"{label}")
                        ax.set_xlabel("Injury")
                        ax.set_xticklabels(injuries, rotation=90, ha='right')
                        
                        # Optional: scale y-axis nicely
                        # vals = transformed_values
                        # buffer = (vals.max() - vals.min()) * 0.1 if vals.max() != vals.min() else 0.1
                        # ax.set_ylim(vals.min() - buffer, vals.max() + buffer)

                    panel_idx = num_metrics
                    if has_hazard_effect_panel:
                        ax = axes[panel_idx]
                        exit = samples.get("exit__loc")
                        exit_rate = samples.get("exit_rate__loc")
                        exit_global_offset = samples.get("exit_global_offset__loc", 0.0)
                        # Full player-specific injury shift (n, i): prior mean + player loading
                        injury_shift = (
                            jnp.asarray(injury_exit_mean_prior)[None, :]
                            + jnp.einsum("nr,ri->ni", psi_x, injury_player_exit)
                        )  # (n, i)
                        # Injury effect on scale (n, i): prior mean + player loading
                        _injury_scale_loading = samples.get("injury_scale_loading__loc")
                        _injury_scale_global_offset = samples.get("injury_scale_global_offset__loc", 0.0)
                        _injury_player_scale = samples.get("injury_player_scale__loc")
                        if _injury_scale_loading is not None and _injury_player_scale is not None:
                            _injury_scale_mean_prior = jnp.einsum("ip,p->i", jnp.asarray(samples["injury_factor__loc"]), _injury_scale_loading)
                            injury_scale_shift = (
                                _injury_scale_global_offset
                                + _injury_scale_mean_prior[None, :]
                                + jnp.einsum("nr,ri->ni", psi_x, _injury_player_scale)
                            )  # (n, i)
                        else:
                            injury_scale_shift = jnp.zeros((psi_x.shape[0], injury_shift.shape[-1]))
                        if (exit is not None) and (exit_rate is not None):
                            sigma_exit_scale = samples.get("sigma_exit_scale__loc", 1.0)
                            scale_global_log = samples.get("scale_global_log__loc", jnp.log(11.5))
                            exit_raw = make_psi_gamma(psi_x, exit) / jnp.sqrt(model._kernel_self_cov(psi_x)) * sigma_exit_scale
                            scale_player = jnp.exp(scale_global_log + exit_raw)  # (n,) baseline
                            # Per-player per-injury-type scale: (n, i)
                            scale_injury = jnp.exp(
                                scale_global_log + exit_raw[:, None] + injury_scale_shift
                            )  # (n, i)

                            # Model: exit_rate_raw = make_psi_gamma(X, exit_rate) + exit_global_offset + injury_effect
                            baseline_exit_rate = make_psi_gamma(psi_x, exit_rate) + exit_global_offset  # (n,)
                            baseline_concentration = 1.0 + 2.0 * jax.nn.sigmoid(baseline_exit_rate)  # (n,)
                            # injury_shift is (n, i); baseline_exit_rate[:, None] is (n, 1) → broadcasts to (n, i)
                            injury_concentration = 1.0 + 2.0 * jax.nn.sigmoid(
                                baseline_exit_rate[:, None] + injury_shift
                            )  # (n, i)

                            tenure_grid = jnp.maximum(basis - age_min, 1e-3)
                            hazard_baseline = (
                                baseline_concentration[:, None] / scale_player[:, None]
                            ) * jnp.power(
                                tenure_grid[None, :] / scale_player[:, None],
                                baseline_concentration[:, None] - 1.0,
                            )  # (n, T)

                            injury_exit_raw_loc = samples.get("injury_exit_raw__loc")
                            sigma_injury_exit_loc = samples.get("sigma_injury_exit__loc")
                            if (injury_exit_raw_loc is not None) and (sigma_injury_exit_loc is not None):
                                # injury_shift is (n, i) → (n, 1, i) to combine with (n, t, i)
                                injury_exit_total = injury_shift[:, None, :] + injury_exit_raw_loc * sigma_injury_exit_loc  # (n, t, i)
                                injury_concentration_time = 1.0 + 2.0 * jax.nn.sigmoid(
                                    baseline_exit_rate[:, None, None] + injury_exit_total
                                )  # (n, t, i)
                                interval_idx = jnp.clip(
                                    jnp.floor(tenure_grid).astype(jnp.int32),
                                    0,
                                    injury_concentration_time.shape[1] - 1,
                                )
                                # transpose (n, t, i) → (n, i, t) then index along t-axis
                                injury_concentration_grid = jnp.take_along_axis(
                                    jnp.transpose(injury_concentration_time, (0, 2, 1)),  # (n, i, t)
                                    interval_idx[None, None, :],  # (1, 1, T)
                                    axis=-1,
                                )  # (n, i, T)
                                hazard_injury = (
                                    injury_concentration_grid / scale_injury[:, :, None]
                                ) * jnp.power(
                                    tenure_grid[None, None, :] / scale_injury[:, :, None],
                                    injury_concentration_grid - 1.0,
                                )  # (n, i, T)
                            else:
                                # injury_concentration: (n, i)
                                hazard_injury = (
                                    injury_concentration[:, :, None] / scale_injury[:, :, None]
                                ) * jnp.power(
                                    tenure_grid[None, None, :] / scale_injury[:, :, None],
                                    injury_concentration[:, :, None] - 1.0,
                                )  # (n, i, T)
                            hazard_ratio = np.asarray(
                                (hazard_injury + 1e-8) / (hazard_baseline[:, None, :] + 1e-8)
                            )  # (n, i, T)
                            median_hazard_ratio = np.median(hazard_ratio, axis=(0, 2))  # (i,)
                        else:
                            baseline_concentration = 1.0 + 2.0 * jax.nn.sigmoid(jnp.asarray(exit_global_offset))
                            injury_concentration = 1.0 + 2.0 * jax.nn.sigmoid(
                                jnp.asarray(exit_global_offset) + injury_shift
                            )  # (n, i)
                            median_hazard_ratio = np.asarray(
                                np.median((injury_concentration + 1e-8) / (baseline_concentration + 1e-8), axis=0)
                            )  # (i,)
                        pd.Series(median_hazard_ratio, index=injuries).plot(kind="bar", ax=ax)
                        ax.set_title("exit_hazard_median_ratio")
                        ax.set_xlabel("Injury")
                        ax.set_ylabel("Median Hazard Ratio")
                        ax.set_xticklabels(injuries, rotation=90, ha='right')
                        panel_idx += 1

                    # Remove any extra axes
                    for j in range(panel_idx, len(axes)):
                        fig.delaxes(axes[j])

                    fig.tight_layout()
                    fig.suptitle("Injury Effects Across Metrics and Hazard", fontsize=16)
                    os.makedirs(os.path.join(_plots_dir, "injury"), exist_ok=True)
                    fig.savefig(os.path.join(_plots_dir, "injury", f"{model_name}_metrics_injuries.png"), dpi=300, bbox_inches='tight')


                    injury_pca_input = injury_mean_prior
                    loading_labels = list(metrics)
                    if injury_exit_mean_prior is not None:
                        injury_pca_input = jnp.concatenate([injury_pca_input, injury_exit_mean_prior[None, :]], axis=0)
                        loading_labels.append("exit_hazard")

                    injury_pca = make_pca_pipeline().fit(injury_pca_input.T)
                    injury_loadings_df = pd.DataFrame(injury_pca.named_steps["pca"].components_.T, columns = ["PC1", "PC2"])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(injury_loadings_df["PC1"], injury_loadings_df["PC2"], alpha=0.01)
                    injury_loadings_df["metrics"] = loading_labels

                    for row in injury_loadings_df.itertuples():
                        ax.text(row.PC1, row.PC2, row.metrics, fontsize=8)

                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("PCA Visualization of Injury Loadings")
                    fig.savefig(os.path.join(_plots_dir, "injury", f"{model_name}_injury_loadings.png"), format="png")
                    plt.close()                

                    
                    injury_pca_df = pd.DataFrame(injury_pca.transform(injury_pca_input.T), columns = ["PC1", "PC2"])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(injury_pca_df["PC1"], injury_pca_df["PC2"], alpha=0.01)
                    
                    injury_pca_df["injury"] = injuries
                    for row in injury_pca_df.itertuples():
                        ax.text(row.PC1, row.PC2, row.injury, fontsize=8)

                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("PCA Visualization of Injury")
                    fig.savefig(os.path.join(_plots_dir, "injury", f"{model_name}_injury_pca.png"), format="png")
                    plt.close()

                    # ---- Decay-model-specific plots ---- #
                    if "decay" in model_name and "lambda_global_offset__loc" in samples:
                        lambda_global_offset_decay = samples["lambda_global_offset__loc"]   # (k, i)
                        avg_lambda = np.asarray(jax.nn.softplus(lambda_global_offset_decay))  # (k, i)

                        avg_beta0 = np.asarray(injury_effect_component)                      # (k, i)

                        ncols_d = 4
                        nrows_d = (num_metrics + ncols_d - 1) // ncols_d

                        # Plot 1: avg decay rate (lambda) by injury type
                        fig, axes = plt.subplots(nrows=nrows_d, ncols=ncols_d, figsize=(5 * ncols_d, 4 * nrows_d))
                        axes = axes.flatten()
                        for idx, metric in enumerate(plot_metric_names):
                            ax = axes[idx]
                            pd.Series(avg_lambda[idx], index=injuries).plot(kind="bar", ax=ax, title=metric)
                            ax.set_ylabel("Avg Decay Rate (λ)")
                            ax.set_xlabel("Injury")
                            ax.set_xticklabels(injuries, rotation=90, ha="right")
                        for j in range(num_metrics, len(axes)):
                            fig.delaxes(axes[j])
                        fig.tight_layout()
                        fig.suptitle("Average Injury Decay Rate (λ) by Injury Type", fontsize=16)
                        fig.savefig(os.path.join(_plots_dir, "injury", f"{model_name}_decay_rate.png"), dpi=300, bbox_inches="tight")
                        plt.close()

                        # Plot 2: avg initial shock (beta0) by injury type
                        fig, axes = plt.subplots(nrows=nrows_d, ncols=ncols_d, figsize=(5 * ncols_d, 4 * nrows_d))
                        axes = axes.flatten()
                        for idx, metric in enumerate(plot_metric_names):
                            ax = axes[idx]
                            metric_type = plot_metric_types[idx]
                            if metric_type == "gaussian":
                                shock_link = lambda x: x
                                shock_label = "Change in Outcome"
                            elif metric_type in ["poisson", "negative-binomial"]:
                                shock_link = lambda x: np.exp(x) - 1
                                shock_label = "% Change in Rate"
                            else:
                                shock_link = lambda x: np.exp(x) - 1
                                shock_label = "% Change in Odds"
                            pd.Series(shock_link(avg_beta0[idx]), index=injuries).plot(kind="bar", ax=ax, title=metric)
                            ax.set_ylabel(f"Avg Initial Shock ({shock_label})")
                            ax.set_xlabel("Injury")
                            ax.set_xticklabels(injuries, rotation=90, ha="right")
                        for j in range(num_metrics, len(axes)):
                            fig.delaxes(axes[j])
                        fig.tight_layout()
                        fig.suptitle("Average Injury Initial Shock (β₀) by Injury Type", fontsize=16)
                        fig.savefig(os.path.join(_plots_dir, "injury", f"{model_name}_initial_shock.png"), dpi=300, bbox_inches="tight")
                        plt.close()


        elif "naive" in model_name:
            _mean = samples.get("c_offset__loc", jnp.zeros((len(metrics), covariate_X.shape[0], 1)))
            mu = jnp.repeat(_mean, repeats=len(basis), axis=-1)  # (k, n, t) — AR added by has_ar_params block below

        elif "tvlinearlvm" in model_name and "convex" not in model_name:
            # Reconstruct via the model's OWN forward (single source): compute_curves rebuilds X,
            # applies the LKJ metric correlation, and derives peaks — replacing make_mu_tvlinearlvm.
            _map_data = {k[:-5]: v for k, v in samples.items() if k.endswith("__loc")}
            _d_map = _curves_under_substitute(model, model_args, _map_data, include_derivs=False)
            mu = _d_map["mu"]                                   # (k, n, t)
            X = _d_map["X"]; psi_x = X
            X_center = X - jnp.mean(X, keepdims=True, axis=0)
            t_max = np.asarray(_d_map["t_max"])                 # (n, k) centered
            c_max = np.asarray(_d_map["c_max"])                 # (n, k)

        has_survival_samples = ("exit__loc" in samples)
        if has_survival_samples:
            eps_h = 1e-6
            min_duration_for_plot = 0.25
            exit_hazard = None
            exit_survival = None
            plot_exit_time_tenure = jnp.maximum(Y_surv[:, 1] - age_min + eps_h, eps_h)
            plot_exit_censor = surv_masks[:, 1]
            plot_entrance_time_tenure = jnp.maximum(Y_surv[:, 0] - age_min + eps_h, eps_h)
            if "offsets" in model_args:
                offset_entrance_times = model_args["offsets"].get("entrance_times")
                offset_exit_times = model_args["offsets"].get("exit_times")
                offset_right_censor = model_args["offsets"].get("right_censor")
                if offset_entrance_times is not None:
                    plot_entrance_time_tenure = jnp.maximum(offset_entrance_times, eps_h)
                if offset_exit_times is not None:
                    plot_exit_time_tenure = jnp.maximum(offset_exit_times, eps_h)
                if offset_right_censor is not None:
                    plot_exit_censor = offset_right_censor
            base_hazard_grid = basis - age_min
            hazard_grid = jnp.maximum(base_hazard_grid, min_duration_for_plot)
            # Entrance is always observed — use directly
            entrance_latent = jnp.maximum(plot_entrance_time_tenure, eps_h)
            entrance_duration = plot_entrance_time_tenure


            exit = samples.get("exit__loc")
            exit_rate = samples.get("exit_rate__loc")
            exit_global_offset = samples.get("exit_global_offset__loc", 0.0)
            # scale = exp(scale_global_log + exit_raw + injury_scale_effect)  [log-normal, no sigmoid saturation]
            sigma_exit_scale = samples.get("sigma_exit_scale__loc", 1.0)
            scale_global_log = samples.get("scale_global_log__loc", jnp.log(11.5))
            exit_scale_raw = make_psi_gamma(psi_x, exit) / jnp.sqrt(model._kernel_self_cov(psi_x)) * sigma_exit_scale if exit is not None else 0.0
            # Per-player injury scale effect (additive in log-space)
            _surv_scale_loading = samples.get("injury_scale_loading__loc")
            _surv_scale_global_offset = samples.get("injury_scale_global_offset__loc", 0.0)
            _surv_player_scale = samples.get("injury_player_scale__loc")
            _surv_scale_raw_loc = samples.get("injury_scale_raw__loc")
            _surv_sigma_scale = samples.get("sigma_injury_scale__loc", 0.0)
            _surv_scale_effect = 0.0
            if (_surv_scale_loading is not None and "offsets" in model_args
                    and model_args["offsets"].get("injury_type") is not None):
                _surv_injury_type = jnp.asarray(model_args["offsets"]["injury_type"])
                if _surv_injury_type.ndim == 3:
                    _surv_injury_type = _surv_injury_type[0]
                _surv_type_scalar = _surv_injury_type[:, -1]  # (n,)
                _surv_injury_factor = samples["injury_factor__loc"]  # (i, p)
                _surv_scale_mean = jnp.einsum("ip,p->i", _surv_injury_factor, _surv_scale_loading)  # (i,)
                _surv_player_eff = jnp.einsum("nr,ri->ni", psi_x, _surv_player_scale) if _surv_player_scale is not None else 0.0
                _surv_scale_total = (
                    _surv_scale_global_offset
                    + _surv_scale_mean[None, :]
                    + _surv_player_eff
                    + (jnp.asarray(_surv_scale_raw_loc) * _surv_sigma_scale if _surv_scale_raw_loc is not None else 0.0)
                )  # (n, i)
                _surv_scale_effect = jnp.take_along_axis(
                    jnp.concatenate([jnp.zeros((_surv_scale_total.shape[0], 1)), _surv_scale_total], axis=-1),
                    _surv_type_scalar[:, None].astype(jnp.int32),
                    axis=-1,
                ).squeeze(-1)  # (n,)
            scale_exit = jnp.exp(scale_global_log + exit_scale_raw + _surv_scale_effect)

            # Model: exit_rate_raw = make_psi_gamma(X, exit_rate) + exit_global_offset + injury_effect_exit
            exit_rate_base = make_psi_gamma(psi_x, exit_rate) + exit_global_offset if exit_rate is not None else exit_global_offset  # (n,)
            num_exit_intervals = len(base_hazard_grid)
            interval_starts = jnp.arange(num_exit_intervals, dtype=exit_rate_base.dtype)[None, :]
            interval_ends = interval_starts + 1.0

            # Incorporate time-varying injury effect on concentration (ConvexMaxInjuryTVLinearLVM formulation):
            # injury_effect_exit[n, t] is selected from injury_exit_raw[n, t, i] via injury_type[n, t]
            if "injury_exit_loading__loc" in samples and "offsets" in model_args:
                _injury_type_raw = model_args["offsets"].get("injury_type")
                if _injury_type_raw is not None:
                    _injury_type_2d = jnp.asarray(_injury_type_raw)
                    if _injury_type_2d.ndim == 3:
                        _injury_type_2d = _injury_type_2d[0]  # (n, t_model)
                    _injury_factor = samples["injury_factor__loc"]          # (i, p)
                    _injury_exit_loading = samples["injury_exit_loading__loc"]  # (p,)
                    _injury_exit_global_offset = samples.get("injury_exit_global_offset__loc", 0.0)
                    _injury_player_exit = samples.get(
                        "injury_player_exit__loc",
                        jnp.zeros((psi_x.shape[1], _injury_factor.shape[0])),
                    )  # (r, i)
                    # Mean injury effect per player and injury type: (n, i)
                    _injury_shift = (
                        _injury_exit_global_offset
                        + jnp.einsum("ip,p->i", _injury_factor, _injury_exit_loading)
                        + jnp.einsum("nr,ri->ni", psi_x, _injury_player_exit)
                    )
                    _injury_exit_raw_loc = samples.get("injury_exit_raw__loc")  # (n, t_model, i) or None
                    _sigma = samples.get("sigma_injury_exit__loc", 0.0)
                    if _injury_exit_raw_loc is not None:
                        # Full time-varying effect: mean + raw * sigma
                        _injury_exit_full = _injury_shift[:, None, :] + jnp.asarray(_injury_exit_raw_loc) * _sigma  # (n, t_model, i)
                    else:
                        t_model = _injury_type_2d.shape[1]
                        _injury_exit_full = jnp.broadcast_to(
                            _injury_shift[:, None, :],
                            (_injury_shift.shape[0], t_model, _injury_shift.shape[1]),
                        )  # (n, t_model, i) using mean only
                    # Select effect at each (player, time) based on injury_type; 0 = no injury
                    _injury_effect_exit = jnp.take_along_axis(
                        jnp.concatenate(
                            [jnp.zeros((*_injury_exit_full.shape[:2], 1), dtype=_injury_exit_full.dtype),
                             _injury_exit_full],
                            axis=-1,
                        ),
                        _injury_type_2d[..., None].astype(jnp.int32),
                        axis=-1,
                    ).squeeze(-1)  # (n, t_model)
                    exit_rate_base = exit_rate_base[:, None] + _injury_effect_exit  # (n, t_model) — now 2D

            if exit_rate_base.ndim == 1:
                exit_rate_grid = jnp.repeat(exit_rate_base[:, None], num_exit_intervals, axis=1)
            else:
                interval_index_grid = jnp.clip(
                    jnp.floor(base_hazard_grid).astype(jnp.int32),
                    0,
                    exit_rate_base.shape[-1] - 1,
                )
                exit_rate_grid = jnp.take_along_axis(exit_rate_base, interval_index_grid[None, :], axis=1)
            concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_grid)

            tenure_grid = jnp.maximum(base_hazard_grid, 0.0)
            stop_grid = entrance_duration[:, None] + tenure_grid[None, :]
            entry = entrance_duration[:, None]

            seg_start = jnp.maximum(interval_starts[:, :, None], entry[:, None, :])
            seg_end = jnp.minimum(interval_ends[:, :, None], stop_grid[:, None, :])
            valid_seg = seg_end > seg_start

            seg_start_safe = jnp.maximum(seg_start, eps_h)
            seg_end_safe = jnp.maximum(seg_end, eps_h)
            log_scale = jnp.log(scale_exit.squeeze())[..., None, None]
            concentration_expanded = concentration[:, :, None]
            seg_start_exp = jnp.clip(
                concentration_expanded * (jnp.log(seg_start_safe) - log_scale),
                a_min=-40.0,
                a_max=40.0,
            )
            seg_end_exp = jnp.clip(
                concentration_expanded * (jnp.log(seg_end_safe) - log_scale),
                a_min=-40.0,
                a_max=40.0,
            )

            delta_H = jnp.where(valid_seg, jnp.exp(seg_end_exp) - jnp.exp(seg_start_exp), 0.0)
            cumulative_H_grid = delta_H.sum(axis=1)
            exit_survival = jnp.exp(-cumulative_H_grid)
            exit_survival = jnp.where(tenure_grid[None, :] >= 0.0, exit_survival, jnp.nan)

            interval_index = jnp.clip(jnp.floor(stop_grid).astype(jnp.int32), 0, num_exit_intervals - 1)
            concentration_at_grid = jnp.take_along_axis(concentration, interval_index, axis=1)
            conditioned_duration_grid = jnp.maximum(stop_grid, eps_h)
            exit_hazard = (concentration_at_grid / scale_exit[:, None]) * jnp.power(
                conditioned_duration_grid / scale_exit[:, None],
                concentration_at_grid - 1,
            )







    elif prior_predictive:
        mu = samples["mu"]
        mu_18 = mu[..., 0]
        boundary_l = samples["boundary_l_"]
        boundary_r = samples["boundary_r_"]
        mu_38 = mu[..., -1]
        tmax = samples["t_max"]
        cmax = samples["c_max_"]
        posterior_variance_samples = samples["sigma"]
        posterior_dispersion_samples = samples["sigma_beta"]
        alpha = samples["alpha"]
        X = samples["X"].mean(0)

    if map_inference or prior_predictive:
        file_pre = inference_method
        player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Yao Ming",
                            "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                            "Chris Paul", "Shaquille O'Neal", "Trae Young"]
        predict_players = player_labels + ["Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                        "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                        "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                        "Devin Booker", "Paul Pierce", "Allen Iverson", 
                                        "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                        "Giannis Antetokounmpo", "Jrue Holiday", "No Name"]
        categories = data["position_group"].unique()
        cmap = cm.get_cmap("tab10", len(categories))  # 'tab10' has 10 distinct colors
        category_to_color = {cat: cmap(i) for i, cat in enumerate(categories)}
        data["color"] = data["position_group"].map(category_to_color)
        id_df = data[["position_group","name","id", "minutes", "color"]].groupby("id").max().reset_index()
        id_df["id"] = id_df["id"].astype(str)

        if "max" in model_name or ("tvlinearlvm" in model_name and "convex" not in model_name):


            tsne = TSNE(n_components=2)
            X_tsne_df = pd.DataFrame(tsne.fit_transform(X_center), columns = ["Dim. 1", "Dim. 2"])
            X_tsne_df = pd.concat([X_tsne_df, id_df], axis = 1)
            X_tsne_df["name"] = X_tsne_df["name"].apply(lambda x: x if x in predict_players else "")
            X_tsne_df["minutes"] /= np.max(X_tsne_df["minutes"])
            X_tsne_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
            ax = X_tsne_df.plot.scatter(x = "Dim. 1", y = "Dim. 2", c = "color", s = "minutes", title="T-SNE Visualization of Latent Player Embedding")
            # for _, row in X_tsne_df.iterrows():
            #     ax.text(row["Dim. 1"], row["Dim. 2"], row["name"], ha='right')
            fig = ax.get_figure()
            os.makedirs(os.path.join(_plots_dir, "latent_space", file_pre), exist_ok=True)
            fig.savefig(os.path.join(_plots_dir, "latent_space", file_pre, f"{model_name}.png"), format="png")
            plt.close()


        

            X_pca_df = pd.DataFrame(make_pca_pipeline().fit_transform(X), columns = ["Dim. 1", "Dim. 2"])
            
            X_pca_df = pd.concat([X_pca_df, id_df], axis = 1)
            X_pca_df["name"] = X_pca_df["name"].apply(lambda x: x if x in predict_players else "")
            X_pca_df["minutes"] /= np.max(X_pca_df["minutes"])
            X_pca_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
            ax = X_pca_df.plot.scatter(x = "Dim. 1", y = "Dim. 2", c = "color", s = "minutes", title="PCA Visualization of Latent Player Embedding", )
            # for _, row in X_pca_df.iterrows():
            #     ax.text(row["Dim. 1"], row["Dim. 2"], row["name"], ha='right', )
            fig = ax.get_figure()
            os.makedirs(os.path.join(_plots_dir, "latent_space", "map"), exist_ok=True)
            fig.savefig(os.path.join(_plots_dir, "latent_space", "map", f"{model_name}_pca.png"), format="png")
            plt.close()

            # --- Singular value spectrum of the MAP latent X ---
            # Uses only the sampled latent dimensions (r columns), not the fixed
            # observed-covariate columns, so we use samples["X__loc"] directly.
            _X_raw = np.array(samples["X__loc"])
            _X_centered = _X_raw - _X_raw.mean(axis=0, keepdims=True)
            _sv = np.linalg.svd(_X_centered, compute_uv=False)        # (r,) descending
            _cumvar = np.cumsum(_sv ** 2) / (_sv ** 2).sum()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(range(1, len(_sv) + 1), _sv)
            axes[0].set_xlabel("Latent dimension")
            axes[0].set_ylabel("Singular value")
            axes[0].set_title("Singular value spectrum of MAP latent X")

            axes[1].plot(range(1, len(_sv) + 1), _cumvar, marker="o", markersize=3)
            axes[1].axhline(0.9, color="r", linestyle="--", linewidth=0.8, label="90 % variance")
            axes[1].axhline(0.95, color="orange", linestyle="--", linewidth=0.8, label="95 % variance")
            axes[1].set_xlabel("Number of dimensions")
            axes[1].set_ylabel("Cumulative variance explained")
            axes[1].set_title("Cumulative variance explained")
            axes[1].legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(os.path.join(_plots_dir, "latent_space", "map", f"{model_name}_singular_values.png"), format="png")
            plt.close()

            # --- Per-dimension loading of downstream parameters onto X ---
            # All four weight tensors are shaped (effective_r, ...) so the first r
            # entries are learned latent dims; the last 2 are -log(draft_pos) and height.
            if "linear" in model_name and "tvlinearlvm" not in model_name:
                _beta    = np.array(samples["beta__loc"])          # (effective_r, M_time, k)
                _c_max   = np.array(samples["c_max__loc"])         # (effective_r, k)
                _t_max_r = np.array(samples["t_max_raw__loc"])     # (effective_r, k)
                _exit    = np.array(samples.get("exit__loc", np.zeros(_beta.shape[0])))  # (effective_r,)

                _effective_r = _beta.shape[0]
                _dims = np.arange(1, _effective_r + 1)

                # L2 norm across all non-leading axes → one scalar per dimension
                _beta_load  = np.linalg.norm(_beta.reshape(_effective_r, -1),  axis=-1)
                _cmax_load  = np.linalg.norm(_c_max,                            axis=-1)
                _tmax_load  = np.linalg.norm(_t_max_r,                          axis=-1)
                _exit_load  = np.abs(_exit)

                # Normalise each series to [0, 1] so they're comparable on one axis
                def _norm01(v):
                    rng = v.max() - v.min()
                    return (v - v.min()) / rng if rng > 0 else v

                fig, ax = plt.subplots(figsize=(max(10, _effective_r // 2), 4))
                ax.plot(_dims, _norm01(_beta_load),  marker="o", markersize=3, label="beta (curvature)")
                ax.plot(_dims, _norm01(_cmax_load),  marker="s", markersize=3, label="c_max (peak value)")
                ax.plot(_dims, _norm01(_tmax_load),  marker="^", markersize=3, label="t_max (peak age)")
                ax.plot(_dims, _norm01(_exit_load),  marker="D", markersize=3, label="exit (survival)")

                # Shade the observed-covariate columns so they stand out
                _r_sampled = _X_raw.shape[1]
                if _effective_r > _r_sampled:
                    ax.axvspan(_r_sampled + 0.5, _effective_r + 0.5, alpha=0.12, color="grey",
                               label="observed covariates (-log draft, height)")

                ax.set_xlabel("Latent dimension")
                ax.set_ylabel("Normalised loading (L2 norm)")
                ax.set_title("Per-dimension weight loading on latent X")
                ax.legend(fontsize=8, loc="upper right")
                ax.set_xticks(_dims)
                fig.tight_layout()
                fig.savefig(os.path.join(_plots_dir, "latent_space", "map", f"{model_name}_dim_loadings.png"), format="png")
                plt.close()

                # --- Clustered loading heatmap (per-metric) ---
                # beta:   norm across M_time axis → (effective_r, k)
                # c_max:  (effective_r, k)  — already per-metric
                # t_max:  (effective_r, k)  — already per-metric
                # exit:   (effective_r,)    — single column
                _k = _c_max.shape[1]
                _beta_per_metric  = np.linalg.norm(_beta, axis=1)    # (effective_r, k)
                _cmax_per_metric  = np.abs(_c_max)                    # (effective_r, k)
                _tmax_per_metric  = np.abs(_t_max_r)                  # (effective_r, k)
                _exit_col         = np.abs(_exit)[:, None]            # (effective_r, 1)

                # Full loading matrix: (effective_r, 3k+1)
                _load_mat = np.concatenate(
                    [_beta_per_metric, _cmax_per_metric, _tmax_per_metric, _exit_col], axis=1
                )

                # Normalise each column by its max so every metric contributes equally
                _col_max = _load_mat.max(axis=0, keepdims=True)
                _col_max = np.where(_col_max == 0, 1.0, _col_max)
                _load_norm = _load_mat / _col_max

                # Column labels: one per metric for each output type, then exit
                _short = [m[:6] for m in metrics]  # truncate long metric names
                _load_labels = (
                    [f"β {m}" for m in _short] +
                    [f"c★ {m}" for m in _short] +
                    [f"t★ {m}" for m in _short] +
                    ["exit"]
                )

                # Row labels; mark observed-covariate dims
                _r_sampled = _X_raw.shape[1]
                _dim_labels = [f"d{i}" for i in _dims]
                for _di in range(_effective_r):
                    if _di >= _r_sampled:
                        _dim_labels[_di] += "*"

                from scipy.cluster import hierarchy as _sch
                _linkage = _sch.linkage(_load_norm, method="ward", metric="euclidean")
                _order   = _sch.leaves_list(_linkage)
                _load_reordered = _load_norm[_order]
                _row_labels     = [_dim_labels[i] for i in _order]

                _n_cols  = _load_norm.shape[1]
                _fig_h   = max(6, _effective_r * 0.28)
                _fig_w   = max(10, _n_cols * 0.55 + 3)
                fig2, (ax_dend, ax_heat) = plt.subplots(
                    1, 2, figsize=(_fig_w, _fig_h),
                    gridspec_kw={"width_ratios": [1, _n_cols]}
                )

                # Dendrogram on left
                _sch.dendrogram(_linkage, labels=_dim_labels, orientation="left",
                                ax=ax_dend, leaf_font_size=7,
                                link_color_func=lambda _k: "steelblue")
                ax_dend.set_title("Cluster", fontsize=9)
                ax_dend.axis("off")

                # Heatmap on right (rows ordered by dendrogram)
                _im = ax_heat.imshow(_load_reordered, aspect="auto", cmap="viridis",
                                     vmin=0, vmax=1, interpolation="nearest")
                ax_heat.set_xticks(range(_n_cols))
                ax_heat.set_xticklabels(_load_labels, fontsize=6, rotation=45, ha="right")
                ax_heat.set_yticks(range(_effective_r))
                ax_heat.set_yticklabels(_row_labels, fontsize=7)
                ax_heat.set_title("Normalised loading per metric (col max = 1)", fontsize=9)
                plt.colorbar(_im, ax=ax_heat, fraction=0.02, pad=0.02)

                # Draw vertical separators between output-type groups
                for _sep in [_k, 2 * _k, 3 * _k]:
                    ax_heat.axvline(_sep - 0.5, color="white", linewidth=1.5, linestyle="--")

                # Annotate cells (skip if too many columns to keep readable)
                if _n_cols <= 30:
                    for _ri in range(_effective_r):
                        for _ci in range(_n_cols):
                            _val = _load_reordered[_ri, _ci]
                            ax_heat.text(_ci, _ri, f"{_val:.2f}", ha="center", va="center",
                                         fontsize=4, color="white" if _val < 0.6 else "black")

                if _effective_r > _r_sampled:
                    fig2.text(0.99, 0.01, "* = observed covariate", fontsize=6,
                              ha="right", va="bottom", style="italic")

                fig2.tight_layout()
                fig2.savefig(os.path.join(_plots_dir, "latent_space", "map", f"{model_name}_dim_loadings_clustered.png"),
                             format="png", dpi=150)
                plt.close()

            tmax_pca = make_pca_pipeline().fit(t_max)
            tmax_loadings_df = pd.DataFrame(tmax_pca.named_steps["pca"].components_.T, columns = ["PC1", "PC2"])
            tmax_loadings_df["metric"] = metrics 
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(tmax_loadings_df["PC1"], tmax_loadings_df["PC2"], alpha=0.01)
            for row in tmax_loadings_df.itertuples():
                ax.text(row.PC1, row.PC2, row.metric, fontsize=8)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Visualization of Peak Age Loadings")
            os.makedirs(os.path.join(_plots_dir, "peaks", "map"), exist_ok=True)
            fig.savefig(os.path.join(_plots_dir, "peaks", "map", f"{model_name}_peak_age_loadings.png"), format="png")
            plt.close()



            tmax_pca_df = pd.DataFrame(tmax_pca.transform(t_max), columns = ["PC1", "PC2"])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(tmax_pca_df["PC1"], tmax_pca_df["PC2"], alpha=0.01)
            tmax_pca_df = pd.concat([tmax_pca_df, id_df], axis = 1)
            tmax_pca_df["name"] = tmax_pca_df["name"].apply(lambda x: x if x in predict_players else "")
            tmax_pca_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
            for row in tmax_pca_df.itertuples():
                ax.text(row.PC1, row.PC2, row.name, fontsize=8, color = category_to_color[row.Position])

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Visualization of Peak Age")
            fig.savefig(os.path.join(_plots_dir, "peaks", "map", f"{model_name}_peak_age_pca.png"), format="png")
            plt.close()

            fmax_pca = make_pca_pipeline().fit(c_max)
            fmax_loadings_df = pd.DataFrame(fmax_pca.named_steps["pca"].components_.T, columns = ["PC1", "PC2"])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(fmax_loadings_df["PC1"], fmax_loadings_df["PC2"], alpha=0.01)
            fmax_loadings_df["metric"] = metrics 

            for row in fmax_loadings_df.itertuples():
                ax.text(row.PC1, row.PC2, row.metric, fontsize=8)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Visualization of Peak Value Loadings")
            fig.savefig(os.path.join(_plots_dir, "peaks", "map", f"{model_name}_peak_value_loadings.png"), format="png")
            plt.close()


            fmax_pca_df = pd.DataFrame(fmax_pca.transform(c_max), columns = ["PC1", "PC2"])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(fmax_pca_df["PC1"], fmax_pca_df["PC2"], alpha=0.01)
            fmax_pca_df = pd.concat([fmax_pca_df, id_df], axis = 1)
            fmax_pca_df["name"] = fmax_pca_df["name"].apply(lambda x: x if x in predict_players else "")
            fmax_pca_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
            for row in fmax_pca_df.itertuples():
                ax.text(row.PC1, row.PC2, row.name, fontsize=8, color = category_to_color[row.Position])

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Visualization of Peak Value")
            fig.savefig(os.path.join(_plots_dir, "peaks", "map", f"{model_name}_peak_value_pca.png"), format="png")
            plt.close()

            # ── MAP CSV exports ──────────────────────────────────────────────
            _ls_cols = [f"Dim {i+1}" for i in range(np.array(X).shape[1])]
            _ls_df = pd.DataFrame(np.array(X), columns=_ls_cols)
            pd.concat([_ls_df, id_df.reset_index(drop=True)], axis=1).to_parquet(
                os.path.join(model_dir, "latent_space.parquet"), index=False
            )
            _ids = id_df["id"].to_numpy()
            _peaks_rows = []
            for _pi, _pid in enumerate(_ids):
                for _ki, _m in enumerate(metrics):
                    _peaks_rows.append({"id": _pid, "metric": _m, "peak_age": float(np.array(t_max)[_pi, _ki])})
            pd.DataFrame(_peaks_rows).to_parquet(os.path.join(model_dir, "map_peaks.parquet"), index=False)
            _pvals_rows = []
            for _pi, _pid in enumerate(_ids):
                for _ki, _m in enumerate(metrics):
                    _pvals_rows.append({"id": _pid, "metric": _m, "peak_value": float(np.array(c_max)[_pi, _ki])})
            pd.DataFrame(_pvals_rows).to_parquet(os.path.join(model_dir, "map_peak_vals.parquet"), index=False)
            print(f"MAP Parquet files written to {model_dir}")

        players_df = id_df[id_df["name"].isin(predict_players)]
        Y_plot = Y.copy()

        mu_plus_ar_map = mu
        ar_param_names = ("beta_ar__loc", "sigma_ar__loc", "rho_ar__loc", "AR_0__loc")
        has_ar_params = map_inference and all(param_name in samples for param_name in ar_param_names)
        if has_ar_params:
            beta_ar = samples["beta_ar__loc"]
            sigma_ar = samples["sigma_ar__loc"]
            rho_ar = samples["rho_ar__loc"]
            ar_0 = samples["AR_0__loc"] * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))    
            if mu.ndim == 3:
                def transition_fn(prev, z_t):
                    next_value = prev * rho_ar + z_t * sigma_ar
                    return next_value, next_value

                _, ar_values = jax.lax.scan(f=transition_fn, init=ar_0, xs=beta_ar)
                ar_values = jnp.transpose(ar_values, (1, 2, 0))
                if ar_values.shape[-1] > 0:
                    if using_injury_model and "injury_indicator" in model_args.get("offsets", {}):
                        injury_ind = model_args["offsets"]["injury_indicator"]
                        injury_ind_2d = injury_ind[0] if injury_ind.ndim == 3 else injury_ind  # (n, t)
                        healthy = ~injury_ind_2d                                                # (n, t)
                        healthy_sum = (ar_values * healthy[None]).sum(axis=-1, keepdims=True)
                        healthy_count = jnp.maximum(healthy.sum(axis=-1, keepdims=True)[None], 1)
                        ar_values = ar_values - healthy_sum / healthy_count
                    else:
                        ar_values = ar_values - jnp.mean(ar_values, axis=-1, keepdims=True)
                if ar_values.shape == mu.shape:
                    mu_plus_ar_map = mu + ar_values

        # --- Calendar-year trend: reconstruct MAP TREND_AR (AR or linear depending on model) ---
        _trend_years = None
        mu_plus_ar_trend_map = None
        if map_inference and len(de_trend_metrics) > 0:
            year_ar_param_names = ("beta_year_ar__loc", "sigma_year_ar__loc", "rho_year_ar__loc", "AR_0_year__loc")
            if all(p in samples for p in year_ar_param_names):
                _sigma_y = jnp.array(samples["sigma_year_ar__loc"])   # (num_ar, 1)
                _rho_y   = jnp.array(samples["rho_year_ar__loc"])     # (num_ar, 1)
                _z_y     = jnp.array(samples["beta_year_ar__loc"])    # (num_years, num_ar)
                _ar0_y   = jnp.array(samples["AR_0_year__loc"]) * _sigma_y[None, :, 0]
                _trend_years = ConvexMaxARTVLinearLVM._compute_ar1_calendar_process(
                    _sigma_y, _rho_y, _z_y, _ar0_y, ref_year_idx=ref_year_idx
                )  # (num_ar, num_years)
            elif "beta_slope_year__loc" in samples:
                _beta_slope = jnp.array(samples["beta_slope_year__loc"])   # (num_ar, 1)
                _trend_years = ConvexMaxLinearTrendTVLinearLVM._compute_linear_clamped_trend(
                    _beta_slope, num_years, ref_year_idx, year_max_idx
                )  # (num_ar, num_years)

        if _trend_years is not None:
            # Build full (k, n, j) TREND_AR tensor — shared by both trend types
            _ar_global_indices = jnp.array([i for i, f in enumerate(de_trend_indices) if f])
            _trend_nj = _trend_years[:, year_indices]           # (num_ar, n, j)
            TREND_AR_map = jnp.zeros_like(mu).at[_ar_global_indices].set(_trend_nj)  # (k, n, j)
            mu_plus_ar_trend_map = mu_plus_ar_map + TREND_AR_map

            # Standalone calendar-year trend plot
            _years_range = np.arange(min_year, min_year + num_years)
            _de_trend_metric_names = [m for m, flag in zip(metrics, de_trend_indices) if flag]
            os.makedirs(os.path.join(_plots_dir, "calendar_year_trends"), exist_ok=True)
            plot_calendar_year_trends(
                np.asarray(_trend_years),
                _years_range,
                _de_trend_metric_names,
                os.path.join(_plots_dir, "calendar_year_trends", f"{model_name}_calendar_year_trends.png"),
            )

        for index, row in players_df.iterrows():
            player_index = index
            name = row["name"]
            if map_inference:
                fig, axes = plot_posterior_predictive_career_trajectory_map(
                    player_index,
                    metrics,
                    metric_output,
                    mu[:, jnp.array(player_index), :].squeeze().copy(),
                    mu_plus_ar_map[:, jnp.array(player_index), :].squeeze().copy(),
                    Y_plot,
                    exposures,
                    validation_mask=validation_mask,
                    posterior_map_mu_ar_trend=(
                        mu_plus_ar_trend_map[:, jnp.array(player_index), :].squeeze().copy()
                        if mu_plus_ar_trend_map is not None else None
                    ),
                    basis=basis,
                )
                axes = axes.flatten()
                panel_idx = len(metrics)


                if has_survival_samples:
                    exit_obs = float(plot_exit_time_tenure[player_index])
                    exit_censored = bool(plot_exit_censor[player_index])
                    player_concentration_curve = np.asarray(concentration_at_grid[player_index])
                    player_scale_value = float(scale_exit[player_index].squeeze())
                    player_scale_curve = np.full_like(player_concentration_curve, player_scale_value)

                    if panel_idx < len(axes):
                        exit_curve = np.clip(np.array(exit_survival[player_index]), float(eps_h), 1.0)
                        player_tenure_grid = np.asarray(tenure_grid)
                        exit_tenure_obs = float(np.maximum(exit_obs - float(entrance_duration[player_index]), 0.0))

                        axes[panel_idx].plot(player_tenure_grid, exit_curve, color="tab:red")
                        axes[panel_idx].axvline(exit_tenure_obs, color="black", linestyle="--", linewidth=1)
                        axes[panel_idx].text(exit_tenure_obs, axes[panel_idx].get_ylim()[1] * 0.9, "censored" if exit_censored else "observed", rotation=90, va="top", ha="right", fontsize=8)
                        axes[panel_idx].set_title("Exit Survival | Entry")
                        axes[panel_idx].set_xlabel("Tenure since entry")
                        axes[panel_idx].set_ylabel("Survival")
                        axes[panel_idx].set_ylim(0, 1.05)
                        panel_idx += 1

                    if panel_idx < len(axes):
                        axes[panel_idx].plot(np.asarray(tenure_grid), np.maximum(player_concentration_curve, float(eps_h)), color="tab:purple", label="Concentration")
                        axes[panel_idx].plot(np.asarray(tenure_grid), np.maximum(player_scale_curve, float(eps_h)), color="tab:green", linestyle="--", label="Scale")
                        axes[panel_idx].set_title("Exit Params")
                        axes[panel_idx].set_xlabel("Tenure since entry")
                        axes[panel_idx].set_ylabel("Value (log scale)")
                        axes[panel_idx].set_yscale("log")
                        axes[panel_idx].legend(fontsize=8)
                        panel_idx += 1

                for unused_idx in range(panel_idx, len(axes)):
                    axes[unused_idx].axis("off")

                # fig.update_layout(title = dict(text=name))
                # fig.write_image(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_{name.replace(' ', '_')}.png", format = "png")
                os.makedirs(os.path.join(_plots_dir, "player_plots", file_pre), exist_ok=True)
                fig.savefig(os.path.join(_plots_dir, "player_plots", file_pre, f"{model_name}_{name.replace(' ', '_')}.png"), format="png")
                plt.close()

        if map_inference:
            ids_order = id_df["id"].to_numpy()
            age_grid = np.asarray(basis).astype(int)
            id_to_idx = {str(player_id): idx for idx, player_id in enumerate(ids_order)}
            age_to_idx = {int(age): idx for idx, age in enumerate(age_grid)}

            observations_eval = np.asarray(Y, dtype=float).copy()
            exposures_eval = np.asarray(exposures, dtype=float).copy()
            de_trend_eval = np.asarray(de_trend, dtype=float).copy()

            # for row in data_holdout_future.itertuples(index=False):
            #     player_id = str(row.id)
            #     age_value = int(row.age)
            #     if (player_id not in id_to_idx) or (age_value not in age_to_idx):
            #         continue
            #     player_index = id_to_idx[player_id]
            #     age_index = age_to_idx[age_value]

            #     for metric_index, (metric_name, metric_family, exposure_name) in enumerate(zip(metrics, metric_output, exposure_list)):
            #         obs_value = getattr(row, metric_name)
            #         exposure_raw = getattr(row, exposure_name)
            #         if pd.isna(obs_value):
            #             continue

            #         if metric_family in ["poisson", "negative-binomial"]:
            #             if pd.isna(exposure_raw) or (exposure_raw <= 0):
            #                 continue
            #             exposure_encoded = np.log(exposure_raw)
            #         elif metric_family == "beta":
            #             if pd.isna(exposure_raw) or (exposure_raw < 0):
            #                 continue
            #             exposure_encoded = np.sqrt(exposure_raw + 1.0)
            #         elif metric_family in ["binomial", "beta-binomial", "bernoulli"]:
            #             if pd.isna(exposure_raw) or (exposure_raw <= 0):
            #                 continue
            #             exposure_encoded = exposure_raw
            #         elif metric_family == "gaussian":
            #             exposure_encoded = np.sqrt(1.0 + (0.0 if pd.isna(exposure_raw) else exposure_raw))
            #         else:
            #             exposure_encoded = exposure_raw if not pd.isna(exposure_raw) else np.nan

            #         league_avg_col = f"{metric_name}_league_avg"
            #         league_avg = getattr(row, league_avg_col) if hasattr(row, league_avg_col) else np.nan
            #         if pd.isna(league_avg):
            #             detr_encoded = np.nan
            #         elif metric_family in ["poisson", "negative-binomial"]:
            #             detr_encoded = np.log(league_avg)
            #         elif metric_family in ["beta", "binomial", "beta-binomial", "bernoulli", "gaussian"]:
            #             league_avg_clipped = np.clip(league_avg, 1e-6, 1 - 1e-6)
            #             detr_encoded = np.log(league_avg_clipped / (1 - league_avg_clipped))
            #         else:
            #             detr_encoded = np.nan

            #         observations_eval[metric_index, player_index, age_index] = obs_value
            #         exposures_eval[metric_index, player_index, age_index] = exposure_encoded
            #         de_trend_eval[metric_index, player_index, age_index] = detr_encoded


            # raise ValueError("Debugging: Check the naive mean map values.")




            _ALL_SCHEMES = ["random_interior", "holdout_last_k", "holdout_first_k", "holdout_peak"]
            # Schemes where OR-ing with the year mask is semantically coherent
            # (holdout extends naturally into future career).
            _EXTEND_WITH_YEAR = {"holdout_last_k"}
            _year_mask = np.asarray(validation_mask)  # True = year > validation_year
            _year_matrix = np.asarray(data_set[0]["year_matrix"])
            # Reference array with future-year entries NaN'd so career schemes only
            # draw holdout observations from the training window.
            _ref_in_sample = np.where(_year_mask, np.nan, np.asarray(Y[0]))
            # observations with future years zeroed out — used for schemes that
            # should not evaluate on out-of-sample years at all.
            _obs_in_sample = np.where(_year_mask[None], np.nan, observations_eval)
            _exp_in_sample = np.where(_year_mask[None], np.nan, exposures_eval)
            _de_in_sample  = np.where(_year_mask[None], np.nan, de_trend_eval)

            def _call_write(scheme, scheme_mask):
                if scheme in _EXTEND_WITH_YEAR:
                    # future years are a natural continuation of the holdout window
                    write_coverage_tables(
                        mu_plus_ar_map, observations_eval, exposures_eval, de_trend_eval,
                        scheme_mask | _year_mask, metrics, metric_output, model_name, samples,
                        scheme, holdout_fraction, holdout_k, holdout_seed,
                    )
                else:
                    # future years are unrelated to the holdout intent — exclude them
                    # from both splits so they don't contaminate either metric
                    write_coverage_tables(
                        mu_plus_ar_map, _obs_in_sample, _exp_in_sample, _de_in_sample,
                        scheme_mask, metrics, metric_output, model_name, samples,
                        scheme, holdout_fraction, holdout_k, holdout_seed,
                    )

            if validation_scheme == "all":
                for _scheme in _ALL_SCHEMES:
                    _scheme_mask = create_validation_mask(
                        _ref_in_sample, _scheme, holdout_fraction, holdout_k, holdout_seed
                    )
                    _call_write(_scheme, _scheme_mask)
            elif validation_scheme == "year":
                _eval_mask = create_validation_mask(
                    np.asarray(Y[0]), "year",
                    year_matrix=_year_matrix, validation_year=validation_year,
                )
                write_coverage_tables(
                    mu_plus_ar_map, observations_eval, exposures_eval, de_trend_eval,
                    _eval_mask, metrics, metric_output, model_name, samples,
                    "year", holdout_fraction, holdout_k, holdout_seed,
                )
            elif validation_scheme is not None:
                _scheme_mask = create_validation_mask(
                    _ref_in_sample, validation_scheme, holdout_fraction, holdout_k, holdout_seed
                )
                _call_write(validation_scheme, _scheme_mask)
            else:
                write_coverage_tables(
                    mu_plus_ar_map, observations_eval, exposures_eval, de_trend_eval,
                    _year_mask, metrics, metric_output, model_name, samples,
                    None, holdout_fraction, holdout_k, holdout_seed,
                )

        if prior_predictive:
            fig = plot_prior_predictive_career_trajectory(metrics, metric_output, exposure_list, mu[:, :, jnp.array(0), :].squeeze(), prior_variance_samples=jnp.transpose(posterior_variance_samples), prior_dispersion_samples = posterior_dispersion_samples)
            fig.update_layout(title = "Prior Predictive Curves")
            fig.write_image(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}.png", format = "png")
            
            mu_filtered = mu[((cmax - boundary_r)[:, 0, :] >=2) & ((cmax - boundary_l)[:, 0, :] >=2), 0, :]
            fig = plot_prior_mean_trajectory(np.array(mu_filtered))
            fig.update_layout(title = "Prior Mean Curves")
            fig.write_image(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_mean_curve.png", format = "png")

            df_peak_age = pd.DataFrame(tmax[:, 0, :] + basis.mean(), columns= metrics)
            df_peak_val = pd.DataFrame(cmax[:, 0, :], columns= metrics)
            df_f18 = pd.DataFrame(mu_18[..., 0], columns= metrics)
            df_f38 = pd.DataFrame(mu_38[..., 0], columns= metrics)
            df_bl = pd.DataFrame(boundary_l[..., 0, :], columns= metrics)
            df_br = pd.DataFrame(boundary_r[..., 0, :], columns= metrics)
            df_alpha = pd.DataFrame(alpha, columns= metrics)
            new_cols = ["tmax", "cmax", "f18", "f38", "alpha", "b_l", "b_r"]

            df_concatenated = pd.DataFrame(np.vstack([df_peak_age["obpm"].to_numpy(), 
                                            df_peak_val["obpm"].to_numpy(),
                                            df_f18["obpm"].to_numpy(),
                                            df_f38["obpm"].to_numpy(),
                                            df_alpha["obpm"].to_numpy(),
                                            df_bl["obpm"].to_numpy(),
                                            df_br["obpm"].to_numpy()]).T, columns=new_cols)
            # Step 2: Plot faceted scatter
            fig = px.scatter_matrix(
                df_concatenated,
                opacity=0.75,
                title="Prior Correlation Plot of Function Metrics"
            )
            # Loop over subplot axis numbers (starts from 1)
            for i in range(1, len(new_cols) * len(new_cols) + 1):
                fig.update_layout({
                    f"xaxis{i}": dict(showticklabels=False, ticks="", showgrid=False, zeroline=False),
                    f"yaxis{i}": dict(showticklabels=False, ticks="", showgrid=False, zeroline=False),
                })
            fig.write_image(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_peak_age.png", format = "png")
