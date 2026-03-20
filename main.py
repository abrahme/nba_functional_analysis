import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

_xla_flags = os.environ.get("XLA_FLAGS", "")
for _flag in ["--xla_force_host_platform_device_count=1", "--xla_cpu_multi_thread_eigen=false"]:
    if _flag not in _xla_flags:
        _xla_flags = f"{_xla_flags} {_flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

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
from model.model_utils import make_mu_rflvm, make_mu_linear, make_mu_hsgp, compute_residuals_map, compute_priors, apply_detrend_for_offsets, compute_linear_predictor_mean_offsets, summarize_metric_error_observed_substitutions, summarize_metric_error_injury_splits, summarize_normalized_weighted_metric_residuals_by_age
from data.data_utils import create_fda_data, create_surv_data
from model.models import  ConvexMaxInjuryTVLinearLVM, ConvexMaxTVLinearLVM, ConvexMaxTVRFLVM, NaiveLinearLVM, ConvexMaxBoundaryTVRFLVM, ConvexMaxBoundaryARTVRFLVM, ConvexMaxARTVRFLVM, ConvexMaxARTVLinearLVM
from visualization.visualization import plot_posterior_predictive_career_trajectory_map, plot_prior_predictive_career_trajectory, plot_prior_mean_trajectory


def make_pca_pipeline(n_components=2, whiten=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=whiten))
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--approx_x_dim", help="size of the approx dim to X", required=True, type=int)
    parser.add_argument("--basis_dims_2", help="size of the basis", required=False, type=int)
    parser.add_argument("--num_chains", help = "number of chains to run mcmc (or first dimension of samples from approximate inference)"
                        , required=False, type = int, default = 4)
    parser.add_argument("--num_samples", help = "number of samples per chain", required=False, type = int, default = 2000)
    parser.add_argument("--num_warmup", help = "number of warmup samples per chain", required = False, type = int, default = 1000)
    parser.add_argument("--fixed_param_path",help="where to read in the fixed params from", required=False, default="")
    parser.add_argument("--output_path", help="where to store generated files", required = False, default="")
    parser.add_argument("--vectorized", help="whether to vectorize some chains so all gpus will be used", action="store_true")
    parser.add_argument("--injury", help="whether to mask out injury years", action="store_true")
    parser.add_argument("--inference_method", help = "which inference method to run the model for", required=True, choices=["mcmc", "svi", "map", "prior"], default = "mcmc" )
    parser.add_argument("--init_path", help = "where to initialize inference from", required=False, default="")
    parser.add_argument("--player_names", help = "which players to run the model for", required=False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--position_group", help = "which position group to run the model for", required = True, choices=["G", "F", "C", "all"])
    parser.add_argument("--validation_year", help = "year format of {yyyy} indicating for which dates prior and including will be included in training set", required=True, 
                        default = 2021, type = int)
    parser.add_argument("--cohort_year", help = "year format of {yyyy} indicating for which cohort dates prior and including will be included in training set", required=True, 
                        default = 2021, type = int)
    parser.add_argument("--de_trend_metrics", help = "csv list of which metrics to de trend", required = False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--debug_nan", help="enable step-wise SVI NaN diagnostics", action="store_true")
    numpyro.set_platform("gpu")
    # numpyro.set_host_device_count(4)
    args = vars(parser.parse_args())
    inference_method = args["inference_method"]
    map_inference = (inference_method == "map")
    svi_inference = (inference_method == "svi")
    mcmc_inference = (inference_method == "mcmc")
    prior_predictive = (inference_method == "prior")
    num_warmup, num_samples, num_chains = args["num_warmup"], args["num_samples"], args["num_chains"]
    initial_params_path = args["init_path"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    approx_x_dim = args["approx_x_dim"]
    injury = args["injury"]

    basis_dims_2 = args["basis_dims_2"]
    param_path = args["fixed_param_path"]
    vectorized = args["vectorized"]
    output_path = args["output_path"] if args["output_path"] else f"model_output/{model_name}.pkl"
    players = args["player_names"]
    de_trend_metrics = args["de_trend_metrics"]
    debug_nan = args["debug_nan"]
    validation_year = args["validation_year"]
    cohort_year = args["cohort_year"]
    position_group = args["position_group"]
    data_all = pd.read_csv("data/injury_player_cleaned.csv").query("age <= 38 & name != 'Brandon Williams'")
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

    metric_output = ["beta-binomial", "beta", "beta"] + (["gaussian"] * 2) + (["poisson"] * 6) + (["negative-binomial"] * 3) + (["binomial"] * 3)
    metrics = ["games","usg", "pct_minutes",  "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a", "fg3a", "ftm","fg2m", "fg3m"]
    exposure_list = ([ "games_exposure", "minutes", "games"]) + (["minutes"] * 11) + ["fta","fg2a", "fg3a"]


    scale_values = jnp.ones((len(metrics), 1))

    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data_all.groupby(["year"]).apply(
            lambda g: (g[metric]*g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})

            data_all = data_all.merge(league_avg_broadcasted)
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli"]:
            data_all[f"{metric}_league_avg"] = data_all.groupby("year")[metric].transform("sum") / data_all.groupby("year")[exposure].transform("sum")



    data = data_all 

   

    fake_data = pd.DataFrame({"age": range(18,39), "id": 99999999, "year": range(2000, 2021), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    data = pd.concat([data, fake_data], ignore_index=True)
    names = data.groupby("id")["name"].first().values.tolist()
    validation_mask = data[["year", "age", "id"]].pivot(columns="age", index="id", values=f"year").reindex(columns = range(18,39)).apply(
                                                                        lambda r: r.dropna().iloc[0] + (np.array(range(18,39)) - r.dropna().index[0]) if r.notna().any() else r,
                                                                        axis=1,
                                                                        result_type="expand").to_numpy() > validation_year
    # validation_mask = data[["split","age", "id"]].pivot(columns="age", index="id", values="split").reindex(columns = range(18,39)).to_numpy() == "test"
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

    _, surv_data_set, basis = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year)
    surv_masks = jnp.stack([data_entity["censored"] for data_entity in surv_data_set], -1)
    censor = jnp.stack([data_entity["censor_type"] for data_entity in surv_data_set], -1)
    Y_surv = jnp.stack([data_entity["observations"] for data_entity in surv_data_set], -1) 
    surv_data_dict = {}
    surv_data_dict["observations"] = Y_surv - 18
    surv_data_dict["censored"] = surv_masks
    surv_data_dict["censor_type"] = censor

    if "tvrflvm" in model_name:
        if "convex" in model_name:
            covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
            if "max" in model_name:
                model = ConvexMaxTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
                if "boundary" in model_name:
                    model = ConvexMaxBoundaryTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
                    if "AR" in model_name:
                        model = ConvexMaxBoundaryARTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
                else:
                    if "AR" in model_name:
                        model = ConvexMaxARTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
    elif "linear" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        model = ConvexMaxTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis)
        if injury and ("counterfactual" not in model_name) and ("injury" in model_name):
            model = ConvexMaxInjuryTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis, injury_rank=5, num_injury_types=int(data["injury_code"].max()))
        elif "AR" in model_name:
            model = ConvexMaxARTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis)
        elif "naive" in model_name:
            model = NaiveLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis)

    else:
        raise ValueError("Model not implemented")
    

    model.initialize_priors(scale_values = scale_values)
    initial_params = {}
    initial_map_init = None
    map_fixed_params_loc = {}
    if "lvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            results_param = {key.replace("__loc", ""): val for key, val in results_param.items()}
            if (mcmc_inference or svi_inference or map_inference):
                ### we're usually initializing from autodelta in the numpyro sense so the loc suffix is present. need for all guide models but not for mcmc since 
                ### the sample site is used and the name needs to match
                for param_name in results_param: ### only fixing in mcmc case
                    value = results_param[param_name]
                    response = input(f"Fix parameter {param_name} ?" + " [Y/N]: ")
                    if response == "Y":
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

        offset_max, offset_max_var, offset_peak_absolute, offset_peak_absolute_var = compute_priors(
            Y_for_offsets_masked,
            exposures_for_offsets,
            metric_output,
            exposure_list,
        )

        offset_peak_absolute = offset_peak_absolute + 18 - basis.mean() + 2.0  # +2 forward-shift to debias right-truncation
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
            family_dict["de_trend"] = de_trend[indices]
            data_dict[family] = family_dict
        hsgp_params = {}
        if "convex" in model_name:
                
                x_time = basis - basis.mean()
                # x_time = (basis - jnp.min(basis)) 
                # x_time /= jnp.max(x_time)
                # x_time -= jnp.mean(x_time)
                # x_time *= 2
                L_time = 2 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
                print(f"L_time: {L_time}, x_time: {x_time} ")
                M_time = 5
                phi_time = vmap_make_convex_phi(jnp.squeeze(x_time), jnp.squeeze(L_time), M_time)
                hsgp_params["phi_x_time"] = phi_time
                hsgp_params["M_time"] = M_time
                hsgp_params["L_time"] = L_time
                hsgp_params["shifted_x_time"] = x_time + L_time
                hsgp_params["t_0"] = jnp.min(x_time)
                hsgp_params["t_r"] = jnp.max(x_time)
                if "hsgp" in model_name:
                    hsgp_params["eigenvalues_X"] = sqrt_eigenvalues(2 *  jnp.ones(basis_dims)[..., None] , approx_x_dim, basis_dims)

        model_args = {"data_set": data_dict,  "inference_method": inference_method, "sample_free_indices": jnp.array(player_indices), 
                      "sample_fixed_indices": jnp.setdiff1d(jnp.arange(covariate_X.shape[0]), jnp.array(player_indices), assume_unique=True)}
        

       



        model_args["offsets"] = {}
        entrance_times = Y_surv[:, 0]
        exit_times = Y_surv[:, 1]
        right_censor = surv_masks[:, 1]
        left_censor = surv_masks[:, 0]
        apply_injury_leakage_censor = ("linear" in model_name) and (not using_injury_model)
        if apply_injury_leakage_censor:
            player_injury_mask = injury_masks[0]
            has_injury = jnp.any(player_injury_mask, axis=1)
            first_injury_index = jnp.argmax(player_injury_mask, axis=1)
            first_injury_age = first_injury_index.astype(exit_times.dtype) + 18.0
            first_injury_age = jnp.where(has_injury, first_injury_age, jnp.inf)
            censor_at_injury = has_injury & (first_injury_age > entrance_times) & (first_injury_age < exit_times)
            exit_times = jnp.where(censor_at_injury, first_injury_age, exit_times)
            right_censor = jnp.where(censor_at_injury, jnp.ones_like(right_censor, dtype=bool), right_censor)
        model_args["offsets"].update({"exit_times": exit_times - 18 + 1e-6, "entrance_times": entrance_times - 18 + 1e-6, "left_censor": left_censor, "right_censor": right_censor})
        model_args["offsets"]["injury_indicator"] = injury_masks
        model_args["offsets"]["injury_type"] = injury_types
        model_args.update({"hsgp_params": hsgp_params})
        if "convex" in model_name:
            if "max" in model_name:
                model_args["offsets"].update({"t_max": offset_peak_absolute, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_absolute_var, "c_max_var": offset_max_var})
                if "AR" in model_name and (len(initial_params) > 0) & (inference_method == "mcmc"):
                    if "rflvm" in model_name:
                        mu, *_ = make_mu_rflvm(initial_params["X"], initial_params["lengthscale_deriv"], initial_params["alpha"], initial_params["beta"],
                                            initial_params["W"], initial_params["W_t_max"], initial_params["W_c_max"],  initial_params["lengthscale"], initial_params["lengthscale_t_max"], initial_params["lengthscale_c_max"], initial_params["c_max"], initial_params["t_max_raw"], initial_params["sigma_t"],
                                            initial_params["sigma_c"], L_time, M_time, phi_time, x_time + L_time, model_args["offsets"])
                    elif "linear" in model_name:
                        mu, *_ = make_mu_linear(initial_params["X"], initial_params["lengthscale_deriv"], initial_params["alpha"], initial_params["beta"], initial_params["c_max"], initial_params["t_max_raw"], 
                            initial_params["sigma_t"],
                            initial_params["sigma_c"], L_time, M_time, phi_time, x_time + L_time, basis_dims, model_args["offsets"])
                    elif "hsgp" in model_name:
                        mu, *_ = make_mu_hsgp(initial_params["X"], initial_params["lengthscale_deriv"], initial_params["alpha"], initial_params["alpha_X"], initial_params["beta"], initial_params["lengthscale"],
                            initial_params["lengthscale_c_max"], initial_params["lengthscale_t_max"],  
                            initial_params["c_max"], initial_params["t_max_raw"], 
                            initial_params["sigma_t"],
                            initial_params["sigma_c"],
                            L_time, M_time, phi_time, x_time + L_time, model_args["offsets"], basis_dims, 2 * jnp.ones(basis_dims)[..., None] ,approx_x_dim
                            )
                    obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                    
                    avg_sd, autocorr, lognormal_params, beta_params = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, metrics, initial_params["sigma"], initial_params.get("sigma_negative_binomial",1),
                                                            initial_params.get("sigma_beta_binomial", 0), initial_params.get("sigma_beta",1))
                    print(avg_sd, autocorr)
                        
        if map_inference:
            samples, state = model.run_map_inference(num_steps = 50000, model_args=model_args, initial_state=initial_map_init)
        elif prior_predictive:
            print("sampling from prior")
            samples = model.predict({}, model_args, num_samples = num_samples)
        elif mcmc_inference:
            samples, mcmc_model = model.run_inference(num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, vectorized=vectorized, 
            model_args=model_args, initial_values=initial_params, thinning = int(num_samples / 500))

        elif svi_inference:
            samples = model.run_svi_inference(num_steps=30000, guide_kwargs={}, model_args=model_args, initial_values=initial_params, 
                                                        sample_shape = (num_chains, num_samples), debug_nan=debug_nan)
    if mcmc_inference:
        mcmc_model.print_summary()

    if map_inference and len(map_fixed_params_loc) > 0:
        samples = dict(samples)
        for param_name, param_value in map_fixed_params_loc.items():
            if param_name not in samples:
                samples[param_name] = param_value
   
    if not prior_predictive:
        with open(f"{output_path}_samples.pkl", "wb") as f:
            pickle.dump(samples, f)
        f.close()
        print("saved samples")
        if map_inference:
            with open(f"{output_path}_state.pkl", "wb") as f:
                f.write(ser.to_bytes(state))
            f.close()
            print("saved state")
        

    if map_inference:
        if "max" in model_name:
            print("sigma", samples["sigma__loc"])
            alpha_time = samples["alpha__loc"]
            print("alpha_time", alpha_time)
            shifted_x_time = hsgp_params["shifted_x_time"]
            ls_deriv = 3 + samples["lengthscale_deriv__loc"]
            weights = samples["beta__loc"]
            # print(samples["sigma_beta__loc"], samples["sigma_beta_binomial__loc"])
            if "back_constrained" not in model_name:
                X = samples["X__loc"]
                if "hsgp" in model_name:
                    X = jnp.tanh(X) * 1.9
            
            # X -= jnp.mean(X, keepdims = True, axis = 0)
            # X /= jnp.std(X, keepdims = True, axis = 0)
            spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))

            if "tvrflvm" in model_name:
                lengthscale =  samples["lengthscale__loc"]
                lengthscale_t_max = samples["lengthscale_t_max__loc"]
                lengthscale_c_max = samples["lengthscale_c_max__loc"]
                W = samples["W__loc"]
                W_c_max = samples["W_c_max__loc"]
                W_t_max = samples["W_t_max__loc"]
                wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(lengthscale))    
                psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(approx_x_dim))
                wTx_t_max = jnp.einsum("nr, mr -> nm", X, W_t_max * jnp.sqrt(lengthscale_t_max))
                psi_x_t_max = jnp.concatenate([jnp.cos(wTx_t_max), jnp.sin(wTx_t_max)], axis = -1) * (1/ jnp.sqrt(approx_x_dim))   
                wTx_c_max = jnp.einsum("nr, mr -> nm", X, W_c_max * jnp.sqrt(lengthscale_c_max))
                psi_x_c_max = jnp.concatenate([jnp.cos(wTx_c_max), jnp.sin(wTx_c_max)], axis = -1) * (1/ jnp.sqrt(approx_x_dim))

                spd = spd.T[None]
            
            elif "linear" in model_name:
                psi_x = X
                spd = spd.T[None]

            weights *= spd 
            gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
            X_center = X - jnp.mean(X, keepdims = True, axis = 0)
            sigma_c_max = samples["sigma_c__loc"]
            sigma_t_max = samples["sigma_t__loc"] 
            # sigma_c_max = model_args["offsets"]["c_max_var"]
            # sigma_t_max = model_args["offsets"]["t_max_var"]
            t_max_raw = samples["t_max_raw__loc"] 
            t_max_offset_absolute = samples.get("t_offset__loc", model_args["offsets"]["t_max"])
            c_max_offset_absolute = samples.get("c_offset__loc", model_args["offsets"]["c_max"])
            weight_offset = samples.get("weight_offset__loc", 0.0)
            eps_t = 1e-6
            t_max_offset_scaled = jnp.clip(t_max_offset_absolute / 10.0, -1.0 + eps_t, 1.0 - eps_t)


            if "linear" in model_name:
                t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T) + jnp.arctanh(t_max_offset_scaled)) * 10
                c_max = make_psi_gamma(psi_x, samples["c_max__loc"] * sigma_c_max.T ) + c_max_offset_absolute
            elif "rflvm" in model_name:

                t_max = jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max) + jnp.arctanh(t_max_offset_scaled)) * 10
                c_max = make_psi_gamma(psi_x_c_max, samples["c_max__loc"]) * sigma_c_max + c_max_offset_absolute

            phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
            phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
            intercept = jnp.transpose(c_max)[..., None]
            projected_weights = jnp.einsum("nm, mdk -> ndk", psi_x, weights) + weight_offset
            gamma_phi_gamma_x = jnp.einsum("ndk, nktdz, nzk -> knt", projected_weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), projected_weights)
            mu = intercept + gamma_phi_gamma_x
        
            if injury:
                if ("counterfactual" not in model_name) and ("injury_loading__loc" in samples):
                    injury_loading = samples["injury_loading__loc"]
                    injury_factor = samples["injury_factor__loc"]
                    injury_mean_prior = jnp.einsum("ip, kp -> ki", injury_factor, injury_loading) ### remove the no-injury effect
                    injury_global_offset = samples.get("injury_global_offset__loc")
                    if injury_global_offset is None:
                        injury_global_offset = jnp.zeros((injury_mean_prior.shape[0],))
                    injury_raw = samples["injury_raw__loc"]
                    sigma_injury = samples["sigma_injury__loc"]
                    injury_x_loading = samples.get("injury_x_loading__loc")
                    injury_indicator = model_args["offsets"]["injury_indicator"]
                    injury_type = model_args["offsets"]["injury_type"] 
                    if injury_x_loading is not None:
                        injury_x_effect = jnp.einsum("nr,kir->kni", X, injury_x_loading)
                    else:
                        injury_x_effect = jnp.zeros((len(metrics), X.shape[0], injury_mean_prior.shape[-1]))
                    injury_effect_component = injury_global_offset[:, None, None, None] + injury_mean_prior[:, None, None, :] + injury_x_effect[:, :, None, :] + injury_raw * sigma_injury[:, None, None, None]
                    injury_effect_raw = jnp.concatenate([jnp.zeros_like(injury_indicator)[..., None], injury_effect_component], -1)
                    full_mask = (injury_indicator * masks)[..., None]
                    avg_injury_effect =  (injury_effect_raw * full_mask).sum(axis = (1,2)) / full_mask.sum(axis = (1,2))
                    print(avg_injury_effect.shape)
                    injury_effect = jnp.take_along_axis(injury_effect_raw, injury_type[..., None], -1).squeeze(-1) * injury_indicator
                    print(injury_effect.shape)
                    injuries = data["first_major_injury"].cat.categories[1:]
                    injury_effect_data = pd.DataFrame(injury_mean_prior + injury_global_offset[:, None], columns = injuries, index = metrics)
                    plot_metric_names = list(metrics)
                    plot_metric_types = list(metric_output)
                    injury_exit_mean_prior = None
                    if "injury_exit_loading__loc" in samples:
                        injury_exit_loading = samples["injury_exit_loading__loc"]
                        injury_exit_global_offset = samples.get("injury_exit_global_offset__loc", 0.0)
                        injury_exit_mean_prior = injury_exit_global_offset + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)
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
                        injury_shift = jnp.asarray(injury_exit_mean_prior)
                        if (exit is not None) and (exit_rate is not None):
                            exit_raw = make_psi_gamma(X, exit)
                            scale_min = 8.0
                            scale_max = 15.0
                            scale_player = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)

                            baseline_exit_rate = make_psi_gamma(X, exit_rate)
                            baseline_concentration = 1.0 + 2.0 * jax.nn.sigmoid(baseline_exit_rate)
                            injury_concentration = 1.0 + 2.0 * jax.nn.sigmoid(
                                baseline_exit_rate[:, None] + injury_shift[None, :]
                            )

                            tenure_grid = jnp.maximum(basis - 18, 1e-3)
                            hazard_baseline = (
                                baseline_concentration[:, None] / scale_player[:, None]
                            ) * jnp.power(
                                tenure_grid[None, :] / scale_player[:, None],
                                baseline_concentration[:, None] - 1.0,
                            )

                            injury_exit_raw_loc = samples.get("injury_exit_raw__loc")
                            sigma_injury_exit_loc = samples.get("sigma_injury_exit__loc")
                            if (injury_exit_raw_loc is not None) and (sigma_injury_exit_loc is not None):
                                injury_exit_total = injury_shift[None, None, :] + injury_exit_raw_loc * sigma_injury_exit_loc
                                injury_concentration_time = 1.0 + 2.0 * jax.nn.sigmoid(
                                    baseline_exit_rate[:, None, None] + injury_exit_total
                                )
                                interval_idx = jnp.clip(
                                    jnp.floor(tenure_grid).astype(jnp.int32),
                                    0,
                                    injury_concentration_time.shape[1] - 1,
                                )
                                injury_concentration_grid = jnp.take_along_axis(
                                    jnp.transpose(injury_concentration_time, (0, 2, 1)),
                                    interval_idx[None, None, :],
                                    axis=-1,
                                )
                                hazard_injury = (
                                    injury_concentration_grid / scale_player[:, None, None]
                                ) * jnp.power(
                                    tenure_grid[None, None, :] / scale_player[:, None, None],
                                    injury_concentration_grid - 1.0,
                                )
                            else:
                                hazard_injury = (
                                    injury_concentration[:, :, None] / scale_player[:, None, None]
                                ) * jnp.power(
                                    tenure_grid[None, None, :] / scale_player[:, None, None],
                                    injury_concentration[:, :, None] - 1.0,
                                )
                            hazard_ratio = np.asarray(
                                (hazard_injury + 1e-8) / (hazard_baseline[:, None, :] + 1e-8)
                            )
                            median_hazard_ratio = np.median(hazard_ratio, axis=(0, 2))
                        else:
                            baseline_concentration = 1.0 + 2.0 * jax.nn.sigmoid(0.0)
                            injury_concentration = 1.0 + 2.0 * jax.nn.sigmoid(injury_shift)
                            median_hazard_ratio = np.asarray(
                                (injury_concentration + 1e-8) / (baseline_concentration + 1e-8)
                            )
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
                    fig.savefig(f"model_output/model_plots/injury/{model_name}_metrics_injuries.png", dpi=300, bbox_inches='tight')


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
                    fig.savefig(f"model_output/model_plots/injury/{model_name}_injury_loadings.png", format = "png")
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
                    fig.savefig(f"model_output/model_plots/injury/{model_name}_injury_pca.png", format = "png")
                    plt.close()
        elif "naive" in model_name:
            mu = jnp.repeat(samples.get("c_offset__loc", 0), repeats = len(basis) , axis = - 1)
            print(mu.shape)

        has_survival_samples = ("entrance_latent__loc" in samples)
        if has_survival_samples:
            eps_h = 1e-6
            min_duration_for_plot = 0.25
            entrance_hazard = None
            exit_hazard = None
            entrance_survival = None
            exit_survival = None
            plot_exit_time_tenure = jnp.maximum(Y_surv[:, 1] - 18 + eps_h, eps_h)
            plot_exit_censor = surv_masks[:, 1]
            plot_entrance_time_tenure = jnp.maximum(Y_surv[:, 0] - 18 + eps_h, eps_h)
            plot_left_censor = surv_masks[:, 0]
            if "offsets" in model_args:
                offset_entrance_times = model_args["offsets"].get("entrance_times")
                offset_exit_times = model_args["offsets"].get("exit_times")
                offset_left_censor = model_args["offsets"].get("left_censor")
                offset_right_censor = model_args["offsets"].get("right_censor")
                if offset_entrance_times is not None:
                    plot_entrance_time_tenure = jnp.maximum(offset_entrance_times, eps_h)
                if offset_exit_times is not None:
                    plot_exit_time_tenure = jnp.maximum(offset_exit_times, eps_h)
                if offset_left_censor is not None:
                    plot_left_censor = offset_left_censor
                if offset_right_censor is not None:
                    plot_exit_censor = offset_right_censor
            base_hazard_grid = basis - 18
            hazard_grid = jnp.maximum(base_hazard_grid, min_duration_for_plot)
            entrance_latent_raw = samples["entrance_latent__loc"]
            entrance_latent = jnp.where(
                plot_left_censor,
                jnp.maximum(entrance_latent_raw, eps_h) ,
                jnp.maximum(plot_entrance_time_tenure, eps_h),
            )
            entrance_duration = jnp.where(
                plot_left_censor,
                jnp.maximum(entrance_latent_raw, eps_h),
                plot_entrance_time_tenure,
            )
            entrance = samples.get("entrance__loc")
            sigma_entrance = samples.get("sigma_entrance__loc")
            if (sigma_entrance is not None):
                observed_mask = 1.0 - plot_left_censor.astype(plot_entrance_time_tenure.dtype)
                observed_count = jnp.maximum(observed_mask.sum(), 1.0)
                empirical_log_mean = jnp.sum(jnp.log(plot_entrance_time_tenure) * observed_mask) / observed_count
                entrance_global_offset = samples.get("entrance_global_offset__loc", 0.0)
                entrance_loc = entrance_global_offset + make_psi_gamma(X, entrance) + empirical_log_mean if entrance is not None else empirical_log_mean + entrance_global_offset
                entrance_scale = .35 + sigma_entrance
                hazard_eval_grid = hazard_grid


                entrance_dist = LogNormal(entrance_loc, entrance_scale)
                entrance_hazard = jax.vmap(
                    lambda t: jnp.exp(entrance_dist.log_prob(t)) / (1 - entrance_dist.cdf(t) + eps_h)
                )(hazard_eval_grid).T
                entrance_survival = jax.vmap(
                    lambda t: 1 - entrance_dist.cdf(t) + eps_h
                )(hazard_eval_grid).T


            exit = samples.get("exit__loc")
            exit_rate = samples.get("exit_rate__loc")
            exit_global_offset = samples.get("exit_global_offset__loc", 0.0)
            exit_scale_global_offset = samples.get("exit_scale_global_offset__loc", 0.0)
            exit_scale_raw = make_psi_gamma(X, exit) + exit_scale_global_offset if exit is not None else exit_scale_global_offset
            scale_min = 8.0
            scale_max = 15.0
            scale_exit = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_scale_raw)

            exit_rate_base = make_psi_gamma(X, exit_rate) + exit_global_offset if exit_rate is not None else exit_global_offset
            num_exit_intervals = len(base_hazard_grid)
            interval_starts = jnp.arange(num_exit_intervals, dtype=exit_rate_base.dtype)[None, :]
            interval_ends = interval_starts + 1.0

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

        if "max" in model_name:

            
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
            fig.savefig(f"model_output/model_plots/latent_space/{file_pre}/{model_name}.png", format = "png")
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
            fig.savefig(f"model_output/model_plots/latent_space/map/{model_name}_pca.png", format = "png")
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
            fig.savefig(f"model_output/model_plots/peaks/map/{model_name}_peak_age_loadings.png", format = "png")
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
            fig.savefig(f"model_output/model_plots/peaks/map/{model_name}_peak_age_pca.png", format = "png")
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
            fig.savefig(f"model_output/model_plots/peaks/map/{model_name}_peak_value_loadings.png", format = "png")
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
            fig.savefig(f"model_output/model_plots/peaks/map/{model_name}_peak_value_pca.png", format = "png")
            plt.close()


        players_df = id_df[id_df["name"].isin(predict_players)]
        Y_plot = Y.copy()
        if map_inference and len(de_trend_metrics) > 0:
            Y_plot = apply_detrend_for_offsets(
                Y_obs=Y,
                exposures_obs=exposures,
                metric_families=metric_output,
                de_trend_values=de_trend,
                de_trend_mask=jnp.array(de_trend_indices),
                eps=1e-6,
            )

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
                    ar_values = ar_values - jnp.mean(ar_values, axis=-1, keepdims=True)
                if ar_values.shape == mu.shape:
                    mu_plus_ar_map = mu + ar_values

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
                    validation_mask = validation_mask
                )
                axes = axes.flatten()
                panel_idx = len(metrics)


                if has_survival_samples:
                    entrance_obs = float(Y_surv[player_index, 0] - 18 + 1e-6)
                    entrance_obs_age = float(Y_surv[player_index, 0])
                    exit_obs = float(plot_exit_time_tenure[player_index])
                    entrance_censored = bool(surv_masks[player_index, 0])
                    exit_censored = bool(plot_exit_censor[player_index])
                    player_concentration_curve = np.asarray(concentration_at_grid[player_index])
                    player_scale_value = float(scale_exit[player_index].squeeze())
                    player_scale_curve = np.full_like(player_concentration_curve, player_scale_value)
                    if panel_idx < len(axes):
                        entrance_curve = np.maximum(np.array(entrance_hazard[player_index]), float(eps_h))
                        axes[panel_idx].plot(np.array(basis), entrance_curve, color="tab:blue")
                        axes[panel_idx].axvline(entrance_obs_age, color="black", linestyle="--", linewidth=1)
                        axes[panel_idx].text(entrance_obs_age, axes[panel_idx].get_ylim()[1] * 0.9, "censored" if entrance_censored else "observed", rotation=90, va="top", ha="right", fontsize=8)
                        axes[panel_idx].set_title("Entry Hazard")
                        axes[panel_idx].set_xlabel("Age")
                        axes[panel_idx].set_ylabel("Hazard")
                        panel_idx += 1

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
                fig.savefig(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_{name.replace(' ', '_')}.png", format = "png")
                plt.close()

        if map_inference:
            ids_order = id_df["id"].to_numpy()
            age_grid = np.asarray(basis).astype(int)
            id_to_idx = {str(player_id): idx for idx, player_id in enumerate(ids_order)}
            age_to_idx = {int(age): idx for idx, age in enumerate(age_grid)}

            observations_eval = np.asarray(Y, dtype=float).copy()
            exposures_eval = np.asarray(exposures, dtype=float).copy()
            de_trend_eval = np.asarray(de_trend, dtype=float).copy()
            holdout_mask_nt = validation_mask.copy() 

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




            summary_all = summarize_metric_error_observed_substitutions(
                posterior_mean_map=mu_plus_ar_map,
                observations=observations_eval,
                exposures=exposures_eval,
                metric_outputs=metric_output,
                metrics=metrics,
                de_trend_values=de_trend_eval,
                evaluation_mask=np.ones_like(holdout_mask_nt, dtype=bool),
                sigma_beta = samples.get("sigma_beta__loc", 1),
                sigma = samples.get("sigma__loc", 1),
                sigma_beta_binomial = samples.get("sigma_beta_binomial__loc", 1),
                sigma_negative_binomial = samples.get("sigma_negative_binomial__loc", 1),
            )
            summary_all["split"] = "all"


            summary_holdout = summarize_metric_error_observed_substitutions(
                posterior_mean_map=mu_plus_ar_map,
                observations=observations_eval,
                exposures=exposures_eval,
                metric_outputs=metric_output,
                metrics=metrics,
                de_trend_values=de_trend_eval,
                evaluation_mask=holdout_mask_nt,
                sigma_beta = samples.get("sigma_beta__loc", 1),
                sigma = samples.get("sigma__loc", 1),
                sigma_beta_binomial = samples.get("sigma_beta_binomial__loc", 1),
                sigma_negative_binomial = samples.get("sigma_negative_binomial__loc", 1),
            )
            summary_holdout["split"] = "holdout"



            summary_non_holdout = summarize_metric_error_observed_substitutions(
                posterior_mean_map=mu_plus_ar_map,
                observations=observations_eval,
                exposures=exposures_eval,
                metric_outputs=metric_output,
                metrics=metrics,
                de_trend_values=de_trend_eval,
                evaluation_mask=~holdout_mask_nt,
                sigma_beta = samples.get("sigma_beta__loc", 1),
                sigma = samples.get("sigma__loc", 1),
                sigma_beta_binomial = samples.get("sigma_beta_binomial__loc", 1),
                sigma_negative_binomial = samples.get("sigma_negative_binomial__loc", 1),
            )
            summary_non_holdout["split"] = "non_holdout"

            residual_non_holdout = summarize_normalized_weighted_metric_residuals_by_age(
                posterior_mean_map=mu_plus_ar_map,
                observations=observations_eval,
                exposures=exposures_eval,
                metric_outputs=metric_output,
                metrics=metrics,
                de_trend_values=de_trend_eval,
                evaluation_mask=~holdout_mask_nt,
  

            )

            residual_holdout = summarize_normalized_weighted_metric_residuals_by_age(
                posterior_mean_map=mu_plus_ar_map,
                observations=observations_eval,
                exposures=exposures_eval,
                metric_outputs=metric_output,
                metrics=metrics,
                de_trend_values=de_trend_eval,
                evaluation_mask=holdout_mask_nt,
        
            )



            metric_error_summaries = [summary_all, summary_holdout, summary_non_holdout]
            include_injury_splits = ("injury" in model_name.lower())
            # if include_injury_splits:
            #     summary_holdout_injury = summarize_metric_error_injury_splits(
            #         posterior_mean_map=mu_plus_ar_map,
            #         observations=observations_eval,
            #         exposures=exposures_eval,
            #         metric_outputs=metric_output,
            #         metrics=metrics,
            #         injury_time_mask=np.asarray(injury_masks),
            #         de_trend_values=de_trend_eval,
            #         evaluation_mask=holdout_mask_nt,
            #         naive_mean_reference_mask=~holdout_mask_nt,
            #         include_all=False,
            #     )
            #     summary_holdout_injury["split"] = summary_holdout_injury["split"].map(
            #         {
            #             "injured": "holdout_injured",
            #             "non_injured": "holdout_non_injured",
            #         }
            #     )
            #     metric_error_summaries.append(summary_holdout_injury)

            #     summary_non_holdout_injury = summarize_metric_error_injury_splits(
            #         posterior_mean_map=mu_plus_ar_map,
            #         observations=observations_eval,
            #         exposures=exposures_eval,
            #         metric_outputs=metric_output,
            #         metrics=metrics,
            #         injury_time_mask=np.asarray(injury_masks),
            #         de_trend_values=de_trend_eval,
            #         evaluation_mask=~holdout_mask_nt,
            #         naive_mean_reference_mask=~holdout_mask_nt,
            #         include_all=False,
            #     )
            #     summary_non_holdout_injury["split"] = summary_non_holdout_injury["split"].map(
            #         {
            #             "injured": "non_holdout_injured",
            #             "non_injured": "non_holdout_non_injured",
            #         }
            #     )
            #     metric_error_summaries.append(summary_non_holdout_injury)

            metric_error_df = pd.concat(metric_error_summaries, ignore_index=True)
            metric_error_table = metric_error_df.copy()
            metric_error_table["bias"] = metric_error_table["bias"].round(4)
            metric_error_table["rmse"] = metric_error_table["rmse"].round(4)
            metric_error_table["avg_log_loss"] = metric_error_table["avg_log_loss"].round(4)
            metric_error_table = metric_error_table[
                ["split", "metric", "bias", "rmse", "avg_log_loss", "n_obs"
            ]]
            coverage_dir = "model_output/model_plots/coverage"
            os.makedirs(coverage_dir, exist_ok=True)
            csv_path = f"{coverage_dir}/{model_name}.csv"
            latex_path = f"{coverage_dir}/{model_name}.tex"
            text_path = f"{coverage_dir}/{model_name}.txt"
            latex_caption = f"Bias and RMSE by metric for {model_name}"
            latex_label = f"tab:{model_name}_bias_rmse"

            metric_error_table.to_csv(csv_path, index=False)

           
            split_order = ["all", "holdout", "non_holdout"]
            if include_injury_splits:
                split_order = [
                    "all",
                    "holdout",
                    "holdout_injured",
                    "holdout_non_injured",
                    "non_holdout",
                    "non_holdout_injured",
                    "non_holdout_non_injured",
                ]
            metric_names = metric_error_table["metric"].dropna().unique().tolist()
            n_metrics = len(metric_names)
            ncols = min(4, max(1, n_metrics))
            nrows = int(np.ceil(n_metrics / ncols))


            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=False)
            axes = axes.flatten()
            for ax, metric in zip(axes, metrics):
                sub = residual_holdout[residual_holdout["metric"] == metric]

                ages = sorted(sub["age"].unique())
                grouped = [sub[sub["age"] == age]["normalized_weighted_residual"].values for age in ages]

                ax.violinplot(grouped, positions=range(len(ages)))
                ax.set_xticks(range(len(ages)))
                ax.set_xticklabels(ages)

                ax.set_title(metric)
                ax.set_xlabel("Age")
                ax.set_ylabel("Residual Value")

            # hide unused subplots
            for ax in axes[len(metrics):]:
                ax.axis("off")

            plt.suptitle("Residual Value Distribution (Holdout) by Age and Metric")
            plt.tight_layout()

            plt.savefig(f"{coverage_dir}/{model_name}_residual_by_age_holdout.png", dpi=300)

            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=False)
            axes = axes.flatten()
            for ax, metric in zip(axes, metrics):
                sub = residual_holdout[residual_holdout["metric"] == metric]



                ax.violinplot(sub["normalized_weighted_residual"].values, positions=[0])

                ax.set_title(metric)
                ax.set_ylabel("Residual Value")

            # hide unused subplots
            for ax in axes[len(metrics):]:
                ax.axis("off")

            plt.suptitle("Residual Value Distribution by Metric (Holdout)")
            plt.tight_layout()

            plt.savefig(f"{coverage_dir}/{model_name}_metric_residual_holdout.png", dpi=300)

            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=False)
            axes = axes.flatten()
            for ax, metric in zip(axes, metrics):
                sub = residual_non_holdout[residual_non_holdout["metric"] == metric]

                ages = sorted(sub["age"].unique())
                grouped = [sub[sub["age"] == age]["normalized_weighted_residual"].values for age in ages]

                ax.violinplot(grouped, positions=range(len(ages)))
                ax.set_xticks(range(len(ages)))
                ax.set_xticklabels(ages)

                ax.set_title(metric)
                ax.set_xlabel("Age")
                ax.set_ylabel("Residual Value")

            # hide unused subplots
            for ax in axes[len(metrics):]:
                ax.axis("off")

            plt.suptitle("Residual Value Distribution by Age and Metric")
            plt.tight_layout()

            plt.savefig(f"{coverage_dir}/{model_name}_residual_by_age.png", dpi=300)

            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=False)
            axes = axes.flatten()
            for ax, metric in zip(axes, metrics):
                sub = residual_non_holdout[residual_non_holdout["metric"] == metric]



                ax.violinplot(sub["normalized_weighted_residual"].values, positions=[0])

                ax.set_title(metric)
                ax.set_ylabel("Residual Value")

            # hide unused subplots
            for ax in axes[len(metrics):]:
                ax.axis("off")

            plt.suptitle("Residual Value Distribution by Metric")
            plt.tight_layout()

            plt.savefig(f"{coverage_dir}/{model_name}_metric_residual.png", dpi=300)



            with open(latex_path, "w") as f_latex:
                f_latex.write(
                    metric_error_table.to_latex(
                        index=False,
                        float_format="%.4f",
                        caption=latex_caption,
                        label=latex_label,
                        escape=True,
                    )
                )
            


            print(f"saved bias/rmse csv table to {csv_path}")

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
