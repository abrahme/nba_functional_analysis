import pandas as pd
import numpy as np
import jax
import flax.serialization as ser
import jax.numpy as jnp
import matplotlib.pyplot as plt
import re
import argparse
import pickle
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
from model.hsgp import  diag_spectral_density, make_spectral_mixture_density, make_psi_gamma,  vmap_make_convex_phi, vmap_make_convex_phi_prime, eigenfunctions_multivariate, sqrt_eigenvalues
jax.config.update("jax_enable_x64", True)
from model.inference_utils import get_latent_sites, create_metric_trajectory_map
from model.model_utils import make_mu_rflvm, make_mu_linear, make_mu_hsgp, compute_residuals_map, compute_priors
from data.data_utils import create_fda_data, average_peak_differences, create_surv_data
from model.models import ConvexMaxARTVHSGPLVM, ConvexMaxInjuryTVLinearLVM, ConvexMaxInjuryTVHSGPLVM, ConvexMaxSpectralMixtureTVHSGPLVM, ConvexMaxARBackConstrainedTVHSGPLVM, ConvexMaxTVHSGPLVM, ConvexMaxTVLinearLVM, ConvexMaxTVRFLVM, GibbsConvexMaxBoundaryTVRFLVM, GibbsConvexMaxBoundaryARTVRFLVM, ConvexMaxBackConstrainedTVHSGPLVM, ConvexMaxBoundaryTVRFLVM, ConvexMaxBoundaryARTVRFLVM, ConvexMaxARTVRFLVM, ConvexMaxARTVLinearLVM
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
    numpyro.set_platform("gpu")
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
    validation_year = args["validation_year"]
    cohort_year = args["cohort_year"]
    position_group = args["position_group"]
    data = pd.read_csv("data/injury_player_cleaned.csv").query(f"age <= 38 & name != 'Brandon Williams' & year <= {validation_year}") ### filter out years that happen after this year
    data = data.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year
    # data = data.groupby("id").filter(lambda x: len(x) >= 3) ### just test to keep guys who have played at least 3 years
    data["first_major_injury"] = data["first_major_injury"].fillna("None")
    data['first_major_injury'] = (
    data['first_major_injury']
            .astype('category')
            .cat.set_categories(
                ['None'] + 
                [c for c in pd.unique(data['first_major_injury']) if c != 'None'],
                ordered=False
            ))
    data["injury_code"] = data["first_major_injury"].cat.codes
    data["log_min"] = np.log(data["minutes"])
    data["usg"] /= 100
    data["usg"] += .01
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(data["total_games"], data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games"]) / 48
    data["retirement"] = 1
    fake_data = pd.DataFrame({"age": range(18,39), "id": 99999999, "year": range(2000, 2021), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    data = pd.concat([data, fake_data], ignore_index=True)
    names = data.groupby("id")["name"].first().values.tolist()

    metric_output = ["beta-binomial", "binomial", "beta", "beta"] + (["gaussian"] * 2) + (["negative-binomial"] * 9) + ["binomial"] + (["beta-binomial"] * 2)
    metrics = ["games", "retirement", "pct_minutes", "usg", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a", "fg3a", "ftm","fg2m", "fg3m"]
    exposure_list = ([ "games_exposure", "simple_exposure", "games_exposure", "minutes"]) + (["minutes"] * 11) + ["fta","fg2a", "fg3a"]

    # metric_output = ["beta-binomial", "gaussian", "beta"]
    # metrics = ["games", "obpm", "usg"]
    # exposure_list = ["games_exposure", "minutes", "minutes"]
    # metric_output = ["binomial", "gaussian", "gaussian", "beta", "beta-binomial"] + ["poisson"] * 7 + ["beta", "binomial"]
    # metrics = ["retirement", "obpm", "dbpm", "usg", "games", "fta", "fg2a", "ast", "blk", "stl", "oreb", "dreb", "pct_minutes" ,"fg2m"]
    # exposure_list = ["simple_exposure"] + ["minutes"] * 3 + ["games_exposure"] + ["minutes"] * 7 + ["games_exposure", "fg2a"]
    scale_values = jnp.ones((len(metrics), 1))

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

    
    if model_name == "gibbs_nba_convex_tvrflvm_max_boundary":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        model = GibbsConvexMaxBoundaryTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
    elif model_name == "gibbs_nba_convex_tvrflvm_max_boundary_AR":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        model = GibbsConvexMaxBoundaryARTVRFLVM(latent_rank=basis_dims, rff_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis=basis)
    elif "tvrflvm" in model_name:
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
    elif "hsgp" in model_name:
        if "convex" in model_name:
            covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
            # _, data_set_surv, _ = create_surv_data(data, basis_dims, censor_type, metrics_surv, [], validation_year=validation_year)

            if "max" in model_name:
                model = ConvexMaxTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), L_X = 2*jnp.ones(basis_dims)[..., None],  basis=basis)
                if injury:
                    model = ConvexMaxInjuryTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis, L_X = 2*jnp.ones(basis_dims)[..., None], injury_rank=3, num_injury_types=int(data["injury_code"].max() + 1))
                if "spectral" in model_name:
                    model = ConvexMaxSpectralMixtureTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), L_X = 2*jnp.ones(basis_dims)[..., None],  basis=basis, mixture_dim=10)
                if "AR" in model_name:
                    model = ConvexMaxARTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), L_X = 2*jnp.ones(basis_dims)[..., None], basis=basis)
                if "back_constrained" in model_name:
                    model = ConvexMaxBackConstrainedTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), L_X = 2*jnp.ones(basis_dims)[..., None],  basis=basis)
                    if "AR" in model_name:
                        model = ConvexMaxARBackConstrainedTVHSGPLVM(latent_rank=basis_dims, hsgp_dim=approx_x_dim, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), L_X = 2*jnp.ones(basis_dims)[..., None],  basis=basis)
    elif "linear" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        model = ConvexMaxTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis)
        if injury:
            model = ConvexMaxInjuryTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis, injury_rank=3, num_injury_types=int(data["injury_code"].max() + 1))
        if "AR" in model_name:
            model = ConvexMaxARTVLinearLVM(latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)), basis = basis)


    else:
        raise ValueError("Model not implemented")
    

    model.initialize_priors(scale_values = scale_values)
    initial_params = {}
    if "lvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            if (mcmc_inference or svi_inference):
                ### we're usually initializing from autodelta in the numpyro sense so the loc suffix is present. need for all guide models but not for mcmc since 
                ### the sample site is used and the name needs to match
                results_param = {key.replace("__loc",""):val for key,val in results_param.items()}
                for param_name in results_param: ### only fixing in mcmc case
                    value = results_param[param_name]
                    response = input(f"Fix parameter {param_name} ?" + " [Y/N]: ")
                    if response == "Y":
                        prior_dict[param_name] = numpyro.deterministic(param_name, value)
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
                    initial_params = f_init.read()
                f_init.close()

        model.prior.update(prior_dict)
        distribution_families = set([data_entity["output"] for data_entity in data_set])
        distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
        masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
        injury_masks = jnp.stack([data_entity["injury_mask"] for data_entity in data_set])
        injury_types = jnp.stack([data_entity["injury_type"] for data_entity in data_set]).astype(jnp.int32)
        exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
        Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
        # yao_index = names.index("Yao Ming")
        # print(masks[0,yao_index], Y[0, yao_index], injury_masks[3, yao_index], injury_types[3, yao_index], yao_index)
        # raise ValueError
        Y_linearized = []
        exp_linearized = []
        for data_entity in data_set:
            family = data_entity["output"]
            Y_obs = data_entity["output_data"]
            exposure_obs = data_entity["exposure_data"]
            if family == "gaussian":
                Y_linearized.append(Y_obs)
                exp_linearized.append(jnp.square(exposure_obs))
            elif family in ["negative-binomial", "poisson"]:
                Y_linearized.append(jnp.log(Y_obs + 1))
                exp_linearized.append(jnp.exp(exposure_obs))
            elif family in ["binomial", "beta-binomial", "bernoulli"]:
                p = (Y_obs + .5) / (exposure_obs + 1)
                Y_linearized.append(jnp.log(p)/jnp.log(1 - p))
                exp_linearized.append(exposure_obs)
            elif family in ["beta"]:
                Y_linearized.append(jnp.log(Y_obs) / jnp.log(1 - Y_obs))
                exp_linearized.append(jnp.square(exposure_obs) - 1)
        Y_linearized = jnp.stack(Y_linearized)
        exp_linearized = jnp.stack(exp_linearized)
        W_eff = exp_linearized * masks
        Y_safe = jnp.nan_to_num(Y_linearized, nan=0.0, posinf=0.0, neginf=0.0)
        Y_safe = Y_safe.reshape(Y_safe.shape[1], - 1)
        W_eff = jnp.nan_to_num(1 / W_eff.reshape(W_eff.shape[1], -1), nan=0.0, posinf=0.0, neginf=0.0)
        mu = (Y_safe * W_eff).sum(axis=0) / (W_eff.sum(axis=0))

        Z = (Y_safe - mu) * W_eff
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
    
        offset_max, offset_max_var, offset_peak, offset_peak_var =  compute_priors(Y, exposures, metric_output, exposure_list)
        offset_peak = offset_peak + 18 - basis.mean()
        print(offset_max, offset_max_var, offset_peak, offset_peak_var)
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
        if "convex" in model_name:
                hsgp_params = {}
                x_time = basis - basis.mean()
                # x_time = (basis - jnp.min(basis)) 
                # x_time /= jnp.max(x_time)
                # x_time -= jnp.mean(x_time)
                # x_time *= 2
                L_time = 2 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
                print(f"L_time: {L_time}, x_time: {x_time} ")
                M_time = 20
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
        

       


        if "gibbs" in model_name:
            if "convex" in model_name:
                model_args.update({"hsgp_params": hsgp_params})
                if "max" in model_name:
                    model_args["offsets"] = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l}
            gibbs_site_1 = []
            gibbs_site_2 = []
            for param_name in get_latent_sites(model.model_fn, model_args):
                response = int(input(f"Which gibbs site for  {param_name} ?" + " [1/2]: "))
                if response == 1:
                    gibbs_site_1.append(param_name)
                else:
                    gibbs_site_2.append(param_name)
            samples = model.run_inference(num_chains=num_chains, num_samples=num_samples, vectorized=vectorized, num_warmup=num_warmup, model_args=model_args, gibbs_sites=[gibbs_site_1, gibbs_site_2])
        else:
            if "convex" in model_name:
                model_args.update({"hsgp_params": hsgp_params})
                if "max" in model_name:
                    model_args["offsets"] = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_var, "c_max_var": offset_max_var}
                    model_args["offsets"]["W_eff"] = W_eff
                    model_args["offsets"]["Y_linearized"] = Z
                    model_args["offsets"]["injury_indicator"] = injury_masks
                    model_args["offsets"]["injury_type"] = injury_types
                    # model_args["offsets"]["surv_offsets"] = {"t_max": jnp.log(jnp.sum((Y_surv - 17) * (1 - masks_surv), axis = 0, keepdims = True) / jnp.sum(1 - masks_surv, axis = 0, keepdims = True))}
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
                        if "intercept" in initial_params:
                            mu += (initial_params["intercept"] * initial_params["sigma_intercept"])[..., None]
                        obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                        
                        avg_sd, autocorr = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, metrics, initial_params["sigma"], initial_params.get("sigma_negative_binomial",1),  
                                                                initial_params.get("sigma_beta_binomial", 0), initial_params.get("sigma_beta",1))
                        # avg_sd = jnp.ones((len(metrics))) * .01
                        # autocorr = jnp.zeros_like(avg_sd)
                        model_args["offsets"].update({"avg_sd": avg_sd, "rho":autocorr})
                        print(avg_sd, autocorr)
                        # raise ValueError
                        # model_args["offsets"].update({"avg_sd" : jnp.ones(16)})
            if map_inference:
                samples, state = model.run_map_inference(num_steps=20000, model_args=model_args, initial_state=initial_params if initial_params_path else None)
            elif prior_predictive:
                print("sampling from prior")
                samples = model.predict({}, model_args, num_samples = num_samples)
            elif mcmc_inference:
                samples = model.run_inference(num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, vectorized=vectorized, 
                model_args=model_args, initial_values=initial_params, thinning = int(num_samples / 50))

            elif svi_inference:
                samples = model.run_svi_inference(num_steps=30000, guide_kwargs={}, model_args=model_args, initial_values=initial_params, 
                                                         sample_shape = (num_chains, num_samples))
    if mcmc_inference:
        print_summary(samples)
   
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
        print("sigma", samples["sigma__loc"])
        alpha_time = samples["alpha__loc"]
        print("alpha_time", alpha_time)
        shifted_x_time = hsgp_params["shifted_x_time"]
        ls_deriv =  3 + samples["lengthscale_deriv__loc"]
        weights = samples["beta__loc"]
        intercept_raw = samples["intercept__loc"]
        sigma_intercept = samples["sigma_intercept__loc"]
        player_intercept = (intercept_raw * sigma_intercept)[..., None]
        # print(samples["sigma_beta__loc"], samples["sigma_beta_binomial__loc"])
        if "back_constrained" not in model_name:
            X = samples["X__loc"]
            if "hsgp" in model_name:
                X = jnp.tanh(X) * 1.9
        else:
            W = samples["W__loc"]
            X = jnp.tanh(jnp.dot(Z, W) / W_eff.sum(axis=1, keepdims=True))
        
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
        elif "tvhsgplvm" in model_name:
            
            # alpha_X = jnp.ones_like(alpha_time)
            # X = jnp.tanh(X)
            if "spectral" in model_name:
                
                spd_X = make_spectral_mixture_density(hsgp_params["eigenvalues_X"], samples["mu__loc"], samples["covariance__loc"], samples["mixture_weight__loc"])
            else:
                lengthscale =  samples["lengthscale__loc"]
                # lengthscale = jnp.ones(basis_dims)
                print("lengthscale x", lengthscale)

                alpha_X = samples["alpha_X__loc"]
                # alpha_X = jnp.ones((len(metrics),))
                print("alpha x", alpha_X)
                spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale, 2 *  jnp.ones(basis_dims)[..., None], approx_x_dim)))(alpha_X)
            spd = jnp.einsum("kt, km -> mtk", spd, spd_X)

            psi_x = eigenfunctions_multivariate(X, 2 *  jnp.ones(basis_dims)[..., None], approx_x_dim)
        elif "linear" in model_name:
            psi_x = X
            spd = spd.T[None]

        weights *= spd 
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
        X_center = X - jnp.mean(X, keepdims = True, axis = 0)
        if "max" in model_name:
            sigma_c_max = samples["sigma_c__loc"]
            sigma_t_max = samples["sigma_t__loc"] 
            # sigma_c_max = model_args["offsets"]["c_max_var"]
            # sigma_t_max = model_args["offsets"]["t_max_var"]
            t_max_raw = samples["t_max_raw__loc"] 
            if "hsgp" in model_name:

                lengthscale_t_max = samples["lengthscale_t_max__loc"]
                lengthscale_c_max = samples["lengthscale_c_max__loc"]

                spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale_c_max, 2*jnp.ones(basis_dims)[..., None],  approx_x_dim)))(sigma_c_max)
                spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(basis_dims, alpha, lengthscale_t_max, 2*jnp.ones(basis_dims)[..., None],  approx_x_dim)))(sigma_t_max)
                t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(model_args["offsets"]["t_max"]/10)) * 10 
                c_max = make_psi_gamma(psi_x, samples["c_max__loc"]* spd_c_max.T)  + model_args["offsets"]["c_max"]
            else:
                if "linear" in model_name:
                    t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)  + jnp.arctanh(model_args["offsets"]["t_max"]/10)) * 10
                    c_max = make_psi_gamma(psi_x, samples["c_max__loc"] * sigma_c_max.T ) + model_args["offsets"]["c_max"]
                elif "rflvm" in model_name:

                    t_max = jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)   + jnp.arctanh(model_args["offsets"]["t_max"]/10)) * 10  
                    c_max = make_psi_gamma(psi_x_c_max, samples["c_max__loc"]) * sigma_c_max + model_args["offsets"]["c_max"]

            phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
            phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
            intercept = jnp.transpose(c_max)[..., None]
            gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
            mu = intercept + gamma_phi_gamma_x + player_intercept
        
            if injury:
                injury_loading = samples["injury_loading__loc"]
                injury_factor = samples["injury_factor__loc"]
                injury_mean_prior = jnp.einsum("ip, kp -> ki", injury_factor, injury_loading) ### remove the no-injury effect
                injury_raw = samples["injury_raw__loc"]
                sigma_injury = samples["sigma_injury__loc"]
                injury_indicator = model_args["offsets"]["injury_indicator"]
                injury_type = model_args["offsets"]["injury_type"] 
                injury_effect_raw = injury_mean_prior[:, None, None, :] + injury_raw * sigma_injury[:, None, None, :]
                full_mask = (injury_indicator * masks)[..., None]
                avg_injury_effect =  (injury_effect_raw * full_mask).sum(axis = (1,2)) / full_mask.sum(axis = (1,2))
                injury_effect = jnp.take_along_axis(injury_effect_raw, injury_type[..., None], -1).squeeze(-1) * injury_indicator
                injuries = data["first_major_injury"].cat.categories[1:]
                injury_effect_data = pd.DataFrame(avg_injury_effect[1:], columns = injuries, index = metrics)
                ax = injury_effect_data.plot(kind = "bar", title = "Injury Effect by Metric")
                fig = ax.get_figure()
                # Save to file
                fig.savefig(f"model_output/model_plots/injury/{model_name}_metrics_injuries.png", dpi=300, bbox_inches='tight')


                injury_pca = make_pca_pipeline().fit(injury_mean_prior.T)
                injury_loadings_df = pd.DataFrame(injury_pca.named_steps["pca"].components_.T, columns = ["PC1", "PC2"])
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(injury_loadings_df["PC1"], injury_loadings_df["PC2"], alpha=0.01)
                injury_loadings_df["metrics"] = metrics

                for row in injury_loadings_df.itertuples():
                    ax.text(row.PC1, row.PC2, row.metrics, fontsize=8)

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("PCA Visualization of Injury Loadings")
                fig.savefig(f"model_output/model_plots/injury/{model_name}_injury_loadings.png", format = "png")
                plt.close()                

                
                injury_pca_df = pd.DataFrame(injury_pca.transform(injury_mean_prior.T), columns = ["PC1", "PC2"])
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
        tsne = TSNE(n_components=2)
        X_tsne_df = pd.DataFrame(tsne.fit_transform(X_center), columns = ["Dim. 1", "Dim. 2"])
        id_df = data[["position_group","name","id", "minutes", "color"]].groupby("id").max().reset_index()
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
        for index, row in players_df.iterrows():
            player_index = index
            name = row["name"]
            if map_inference:
                fig = plot_posterior_predictive_career_trajectory_map(player_index, metrics, metric_output, mu[:, jnp.array(player_index), :].squeeze(), Y, exposures)
                # fig.update_layout(title = dict(text=name))
                # fig.write_image(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_{name.replace(' ', '_')}.png", format = "png")
                fig.savefig(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_{name.replace(' ', '_')}.png", format = "png")
                plt.close()
        # for index in samples["boundary_hit"]:
        #     if index == 1:
        #         if map_inference:
        #             name = names[index]
        #             fig = plot_posterior_predictive_career_trajectory_map(player_index, metrics, metric_output, mu[:, jnp.array(index), :].squeeze(), Y, exposures)
        #             fig.savefig(f"model_output/model_plots/player_plots/predictions/{file_pre}/{model_name}_{name.replace(' ', '_')}.png", format = "png")
        #             plt.close()
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
