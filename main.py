import pandas as pd
import numpy as np
import jax
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
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
from numpyro.diagnostics import print_summary
from model.hsgp import  diag_spectral_density, make_convex_f, make_psi_gamma,  vmap_make_convex_phi, vmap_make_convex_phi_prime
jax.config.update("jax_enable_x64", True)
from model.inference_utils import get_latent_sites, create_metric_trajectory_map
from model.model_utils import make_mu, compute_residuals_map
from data.data_utils import create_fda_data, average_peak_differences
from model.models import  ConvexTVRFLVM, ConvexMaxTVRFLVM, GibbsConvexMaxBoundaryTVRFLVM, GibbsConvexMaxBoundaryARTVRFLVM, ConvexMaxBoundaryTVRFLVM, ConvexMaxBoundaryARTVRFLVM, ConvexMaxARTVRFLVM
from visualization.visualization import plot_posterior_predictive_career_trajectory_map, plot_prior_predictive_career_trajectory, plot_prior_mean_trajectory



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--rff_dim", help="size of the rff approx", required=True, type=int)
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
    rff_dim = args["rff_dim"]
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
    data["first_major_injury"] = data["first_major_injury"].fillna("None")
    names = data.groupby("id")["name"].first().values.tolist()
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(data["total_games"], data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games"]) / 48
    data["retirement"] = 1
    metric_output = ["beta-binomial", "beta"] + (["gaussian"] * 2) + (["negative-binomial"] * 9) + (["binomial"] * 3)
    metrics = ["games", "pct_minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["games_exposure", "games_exposure"]) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data.groupby(["year"]).apply(
            lambda g: (g[metric]*g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})
            
            data = data.merge(league_avg_broadcasted)
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial"]:
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
        model = GibbsConvexMaxBoundaryTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif model_name == "gibbs_nba_convex_tvrflvm_max_boundary_AR":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        model = GibbsConvexMaxBoundaryARTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif "convex" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
        
        model = ConvexTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        if "max" in model_name:
            model = ConvexMaxTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
            if "boundary" in model_name:
                model = ConvexMaxBoundaryTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)

                if "AR" in model_name:
                    model = ConvexMaxBoundaryARTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
            else:
                if "AR" in model_name:
                    model = ConvexMaxARTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)

   
    else:
        raise ValueError("Model not implemented")
    

    model.initialize_priors()
    initial_params = {}
    if "rflvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            if (mcmc_inference or svi_inference):
                ### we're usually initializing from autodelta in the numpyro sense so the loc suffix is present. need for all guide models but not for mcmc since 
                ### the sample site is used and the name needs to match
                results_param = {key.replace("__loc",""):val for key,val in results_param.items()}
            for param_name in results_param:
                value = results_param[param_name]
                response = input(f"Fix parameter {param_name} ?" + " [Y/N]: ")
                if response == "Y":
                    prior_dict[param_name] = numpyro.deterministic(param_name, value)
        if initial_params_path:
            with open(initial_params_path, "rb") as f_init:
                initial_params = pickle.load(f_init)
            f_init.close()
            if (mcmc_inference or svi_inference):
                ### we're usually initializing from autodelta in the numpyro sense so the loc suffix is present. need for all guide models but not for mcmc since 
                ### the sample site is used and the name needs to match
                initial_params = {key.replace("__loc",""):val for key,val in initial_params.items()}
                if (len(player_indices) > 0):
                    initial_params["X_free"] = initial_params["X"][jnp.array(player_indices)]

        model.prior.update(prior_dict)
        distribution_families = set([data_entity["output"] for data_entity in data_set])
        distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
        masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
        exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
        Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
        de_trend = jnp.stack([data_entity["de_trend"] for data_entity in data_set]) 
        de_trend = jnp.where(jnp.array(de_trend_indices)[..., None, None], de_trend, 0.0)
        offset_list = []
        offset_max_list = []
        offset_peak_list = []
        offset_boundary_l = []
        offset_boundary_r = []
        for index, family in enumerate(metric_output):  
            if family == "gaussian":
                offset_list.append(jnp.nanmean(Y[index]))
                offset_max_list.append(jnp.nanmean(jnp.nanmax(Y[index], -1)))
                peak = jnp.nanmean(jnp.nanargmax(Y[index], -1))
                boundary_l, boundary_r = average_peak_differences(Y[index])
                offset_boundary_l.append(boundary_l)
                offset_boundary_r.append(boundary_r)
            else:
                if family in ["poisson", "negative-binomial"]:
                    p = jnp.nansum(Y[index]) / jnp.nansum(jnp.exp(exposures[index]))
                    p_max = jnp.nanmean(jnp.nanmax(Y[index] / jnp.exp(exposures[index]), -1))
                    peak = jnp.nanmean(jnp.nanargmax(Y[index] / jnp.exp(exposures[index]), -1))
                    offset_list.append(jnp.log(p))
                    offset_max_list.append(jnp.log(p_max))                 
                elif family in ["beta-binomial", "binomial"]:
                    p = jnp.nansum(Y[index]) / jnp.nansum(exposures[index])
                    p_max = jnp.nanmean(jnp.nanmax(Y[index] / exposures[index], -1))
                    offset_list.append(jnp.log(p/ (1-p)))
                    offset_max_list.append(jnp.log(p_max/(1-p_max)))
                    peak = jnp.nanmean(jnp.nanargmax(Y[index] / exposures[index], -1))
                    p_star = Y[index] / exposures[index]     
                elif family == "beta":
                    p = jnp.nanmean(Y[index])
                    p_max = jnp.nanmean(jnp.nanmax(Y[index], -1))
                    peak = jnp.nanmean(jnp.nanargmax(Y[index], -1))
                    offset_list.append(jnp.log(p / (1 - p)))
                    offset_max_list.append(jnp.log(p_max/(1-p_max)))

            offset_peak_list.append(peak + 18 - basis.mean())

        offsets = jnp.array(offset_list)[None]
        offset_max = jnp.array(offset_max_list)[None]
        offset_peak = jnp.array(offset_peak_list)[None]
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
                L_time = 2.0 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
                M_time = 20
                phi_time = vmap_make_convex_phi(jnp.squeeze(x_time), jnp.squeeze(L_time), M_time)
                hsgp_params["phi_x_time"] = phi_time
                hsgp_params["M_time"] = M_time
                hsgp_params["L_time"] = L_time
                hsgp_params["shifted_x_time"] = x_time + L_time
                hsgp_params["t_0"] = jnp.min(x_time)
                hsgp_params["t_r"] = jnp.max(x_time)

        model_args = {"data_set": data_dict, "offsets": offsets, "inference_method": inference_method, "sample_free_indices": jnp.array(player_indices), 
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
                model_args.update({ "hsgp_params": hsgp_params})
                if "max" in model_name:
                    model_args["offsets"] = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l}
                    if "AR" in model_name and (len(initial_params) > 0) & (inference_method == "mcmc"):
                        mu, *_ = make_mu(initial_params["X"], initial_params["lengthscale_deriv"], initial_params["alpha"], initial_params["beta"],
                                            initial_params["W"], initial_params["lengthscale"], initial_params["c_max"], initial_params["t_max_raw"], initial_params["sigma_t"],
                                            initial_params["sigma_c"], L_time, M_time, phi_time, x_time + L_time, model_args["offsets"])
                        obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                        
                        avg_sd, autocorr = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, initial_params["sigma"], initial_params["sigma_negative_binomial"], 
                                                                initial_params["sigma_beta_binomial"], initial_params["sigma_beta"])
                        # avg_sd = jnp.ones((len(metrics))) * .01
                        # autocorr = jnp.zeros_like(avg_sd)
                        model_args["offsets"].update({"avg_sd": avg_sd, "rho":autocorr})
                        print(avg_sd, autocorr)
                        # raise ValueError
                        # model_args["offsets"].update({"avg_sd" : jnp.ones(16)})
  
            if map_inference:
                samples = model.run_map_inference(num_steps=30000, model_args=model_args, initial_values=initial_params)
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
        
        # log_prior, log_likelihood = extra["log_prior"], extra["log_likelihood"]
        # num_chains, num_samples = log_prior.shape

        # fig = make_subplots(
        #     rows=1, cols=2,
        #     subplot_titles=("Log Prior", "Log Likelihood"),
        #     shared_xaxes=True
        # )

        # x = jnp.arange(num_samples)

        # # Left subplot: log prior
        # for c in range(num_chains):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=x,
        #             y=log_prior[c],
        #             mode="lines",
        #             name=f"chain {c}",
        #         ),
        #         row=1, col=1
        #     )

        # # Right subplot: log likelihood
        # for c in range(num_chains):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=x,
        #             y=log_likelihood[c],
        #             mode="lines",
        #             name=f"chain {c}",
        #             showlegend=(c == 0),  # only show legend once
        #         ),
        #         row=1, col=2
        #     )

        # fig.update_xaxes(title_text="Sample", row=1, col=1)
        # fig.update_xaxes(title_text="Sample", row=1, col=2)
        # fig.update_yaxes(title_text="Value", row=1, col=1)
        # fig.update_yaxes(title_text="Value", row=1, col=2)

        # fig.update_layout(
        #     title="Log Prior and Log Likelihood by Chain",
        #     template="plotly_white",
        #     legend_title="Chain"
        # )

        # fig.write_image("model_outputs/model_plots/log_terms.png", width=1200, height=400)
    if not prior_predictive:
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)
        f.close()
        print("saved samples")

    if map_inference:
        print(samples["sigma__loc"])
        alpha_time = samples["alpha__loc"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        ls_deriv = samples["lengthscale_deriv__loc"]
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = samples["beta__loc"]
        weights = weights * spd * .0001
        lengthscale = samples["lengthscale__loc"]
        print(1 / lengthscale)
        print(samples["sigma_beta__loc"], samples["sigma_beta_binomial__loc"])
        W = samples["W__loc"]
        X = samples["X__loc"]
        X_center = X - jnp.mean(X, keepdims = True, axis = 0)
        # X -= jnp.mean(X, keepdims = True, axis = 0)
        # X /= jnp.std(X, keepdims = True, axis = 0)

        wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(lengthscale))    
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(rff_dim))
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
     

        if "max" in model_name:
            sigma_c_max = samples["sigma_c__loc"]
            sigma_t_max = samples["sigma_t__loc"] 
            t_max_raw = samples["t_max_raw__loc"] 
            if "kron" in model_name:
                t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5 + model_args["offsets"]["t_max"]  
                c_max = make_psi_gamma(psi_x, samples["c_max__loc"]) * sigma_c_max + model_args["offsets"]["c_max"]
            else:
                t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5  + model_args["offsets"]["t_max"]  
                c_max = make_psi_gamma(psi_x, samples["c_max__loc"]) * sigma_c_max + model_args["offsets"]["c_max"]

            phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
            phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
            intercept = jnp.transpose(c_max)[..., None]
            weights /= .0001
            gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
            mu = intercept + gamma_phi_gamma_x

        else:
            intercept_sigma = 1
            intercept = make_psi_gamma(psi_x, samples["intercept__loc"])   
            slope =  make_psi_gamma(psi_x, samples["slope__loc"])    
            mu = make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None]) 

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
        player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                            "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                            "Chris Paul", "Shaquille O'Neal"]
        predict_players = player_labels + ["Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                        "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                        "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                        "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", "LaMelo Ball",
                                        "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                        "Giannis Antetokounmpo", "Jrue Holiday"]
        
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


        pca_latent_space = PCA(n_components=2)

        X_pca_df = pd.DataFrame(pca_latent_space.fit_transform(X_center), columns = ["Dim. 1", "Dim. 2"])
        
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
