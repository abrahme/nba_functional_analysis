import pandas as pd
import pickle
import re
import numpy as np
import argparse
import arviz as az
from jax import config, vmap
from numpyro.diagnostics import print_summary
config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, average_peak_differences, average_range_differences
import numpyro
import jax.numpy as jnp
from model.model_utils import make_mu_rflvm, make_mu_hsgp, make_mu_linear, make_mu_rflvm_mcmc_AR, make_mu_hsgp_mcmc_AR, make_mu_linear_mcmc_AR, make_mu_linear_mcmc, compute_residuals_map, compute_priors
from model.inference_utils import posterior_peaks_to_df, posterior_to_df, posterior_X_to_df
from model.hsgp import vmap_make_convex_phi, eigenfunctions_multivariate, make_spectral_mixture_density, diag_spectral_density, sqrt_eigenvalues

from model.inference_utils import create_metric_trajectory_all, create_metric_trajectory_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--injury", help="whether to mask out injury years", action="store_true")
    parser.add_argument("--approx_x_dim", help="size of the X approx", required=True, type=int)
    parser.add_argument("--mcmc_path", help="where to get mcmc from", required = False, default="")
    parser.add_argument("--svi_path", help = "where to get svi from", required=False, default="")
    parser.add_argument("--position_group", help = "which position group to run the model for", required = True, choices=["G", "F", "C", "all"])
    parser.add_argument("--player_names", help = "which players to run the model for", required=False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--thin", help = "keep every thin sample per chain", required=False, default=100, type = int)
    parser.add_argument("--de_trend_metrics", help = "csv list of which metrics to de trend", required = False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--cohort_year", help = "year format of {yyyy} indicating for which cohort dates prior and including will be included in training set", required=True, 
                        default = 2021, type = int)
    parser.add_argument("--validation_year", help = "year format of {yyyy} indicating for which dates prior and including will be included in training set", required=True, 
                        default = 2021, type = int)
    numpyro.set_platform("cpu")
    args = vars(parser.parse_args())
    mcmc_path = args["mcmc_path"]
    model_name = args["model_name"]
    if "hsgp" in model_name:
        model_suffix = "_hsgp"
    elif "linear" in model_name:
        model_suffix = "_linear"
    elif "rflvm" in model_name:
        model_suffix = "_rff"
    else:
        model_suffix = "_"
    basis_dims = args["basis_dims"]
    de_trend_metrics = args["de_trend_metrics"]
    injury = args["injury"]
    approx_x_dim = args["approx_x_dim"]
    svi_path = args["svi_path"]
    position_group = args["position_group"]
    players = args["player_names"]
    thin = args["thin"]
    validation_year = args["validation_year"]
    cohort_year = args["cohort_year"]

    
    data = pd.read_csv("data/injury_player_cleaned.csv").query(f"age <= 38 & name != 'Brandon Williams' & year <= {validation_year}") ### filter out years that happen after this year
    data = data.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year
    # data = data.groupby("id").filter(lambda x: len(x) >= 3) ### just test to keep guys who have played at least 3 years
    data["first_major_injury"] = data["first_major_injury"].fillna("None")
    names = data.groupby("id")["name"].first().values.tolist()
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
    metric_output = ["beta-binomial", "binomial", "beta", "beta"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["games", "retirement", "pct_minutes", "usg", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a", "fg3a", "ftm","fg2m", "fg3m"]
    exposure_list = ([ "games_exposure", "simple_exposure", "games_exposure", "minutes"]) + (["minutes"] * 11) + ["fta","fg2a", "fg3a"]
    # metric_output = ["gaussian", "gaussian", "negative-binomial"]
    # metrics = ["obpm", "dbpm", "minutes"]
    # exposure_list = ["minutes", "minutes", "games_exposure"]
    scale_values = jnp.ones((len(metrics), 1))
    id_df = data[["id", "name", "position_group", "minutes"]].groupby("id").max().reset_index()

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
    covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], injury=injury, validation_year=validation_year)
    distribution_families = set([data_entity["output"] for data_entity in data_set])
    distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
    masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
    exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
    Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
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


    
    player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                        "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                        "Chris Paul", "Shaquille O'Neal", "Trae Young"]
    predict_players = player_labels + ["Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                    "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                    "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                    "Devin Booker", "Paul Pierce", "Allen Iverson", 
                                    "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                    "Giannis Antetokounmpo", "Jrue Holiday", "No Name"]
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
    for item in results_mcmc:
        print(item, results_mcmc[item].shape)
        if item == "X":
            if "X_free" in results_mcmc:
                X_new = jnp.tile(results_mcmc["X"][None, None], (1, 50, 1, 1))
                X_new = X_new.at[..., jnp.array(player_indices), :].set(results_mcmc["X_free"])
                results_mcmc["X"] = X_new
            if "hsgp" in model_name:
                results_mcmc["X"] = jnp.tanh(results_mcmc["X"]) * 1.9
    if "rflvm" in model_name:
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
        mu, *_ = make_mu_linear(results_map["X"], 3 + results_map["lengthscale_deriv"], results_map["alpha"], results_map["beta"], results_map["c_max"], results_map["t_max_raw"], 
                                # offset_dict["t_max_var"],
                                # offset_dict["c_max_var"],
                                results_map["sigma_t"],
                                results_map["sigma_c"], 
                                  L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict)
    if "intercept" in results_map:
        mu += (results_map["intercept"] * results_map["sigma_intercept"])[..., None]
    obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                        
    # avg_sd, autocorr = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, results_map["sigma"], results_map["sigma_negative_binomial"], 
    #                                                             results_map["sigma_beta_binomial"], results_map["sigma_beta"])
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
        if "AR" in model_name:
            wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_linear_mcmc_AR(results_mcmc["X"], 3 + results_mcmc["lengthscale_deriv"], 
                                results_mcmc["alpha"], results_mcmc["beta"],
                                results_mcmc["c_max"], results_mcmc["t_max_raw"], results_mcmc["sigma_t"],
                                results_mcmc["sigma_c"], L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict,
                                sigma_ar = results_mcmc["sigma_ar"],
                                # sigma_ar = avg_sd[..., None][None, None],
                                beta_ar = results_mcmc["beta_ar"], 
                                rho_ar=results_mcmc["rho_ar"],
                                # rho_ar = autocorr[..., None][None, None],
                                AR_0_raw=results_mcmc["AR_0"],
                                # AR_0_raw = jnp.zeros((len(metrics), covariate_X.shape[0])),
                                orthogonalize=False)
        else:
            wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_linear_mcmc(results_mcmc["X"], 3 + results_mcmc["lengthscale_deriv"], 
                                results_mcmc["alpha"], results_mcmc["beta"],
                                results_mcmc["c_max"], results_mcmc["t_max_raw"], results_mcmc["sigma_t"],
                                results_mcmc["sigma_c"], L_time, M_time, phi_time, x_time + L_time, basis_dims, offset_dict)

    if "intercept" in results_mcmc:
        print("added offset")
        mu_mcmc += (results_mcmc["intercept"] * results_mcmc["sigma_intercept"][None, None])[..., None]
    latent_val = mu_mcmc + AR
    players = id_df[id_df["name"].isin(predict_players)].index
    player_names = id_df[id_df["name"].isin(predict_players)]["name"].tolist()
    idata = az.from_dict(
    posterior={
        "theta": latent_val[:, :, :, jnp.array(players)]
    },
    coords={
        "players": predict_players,
        "age": range(18,39),
        "observable": metrics,
    },
    dims={
        "theta": ["chain", "draw", "observable", "players", "age"]
    })
    summary = az.summary(idata)
    summary.to_csv(f"model_output/posterior_latent_ar{model_suffix}_summary.csv")
    # print(jnp.mean(results_mcmc["sigma_beta"], (0, 1)), jnp.mean(results_mcmc["sigma_negative_binomial"], (0, 1)), jnp.mean(results_mcmc["tau"], (0, 1)))
    # print_summary({k:results_mcmc[k] for k in results_mcmc if k in ["sigma_beta", "tau", "sigma_negative_binomial"]})
    # raise ValueError 

    summary = az.summary(results_mcmc, var_names = ["sigma_beta", "sigma_negative_binomial", "tau"])
    summary.to_csv(f"model_output/posterior_variance{model_suffix}_summary.csv")

    peaks = tmax_mcmc + basis.mean()
    peak_val = cmax_mcmc 
    
    _, pos = create_metric_trajectory_all(mu_mcmc + AR, Y, exposures, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"]),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            # jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=jnp.transpose(results_mcmc["sigma_negative_binomial"], (2, 0, 1)),
                                            posterior_tau_samples=jnp.transpose(results_mcmc["tau"], (2,0,1))
                                            ) 
    _, pos_no_ar = create_metric_trajectory_all(mu_mcmc, Y, exposures, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"]),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            # jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=jnp.transpose(results_mcmc["sigma_negative_binomial"], (2, 0, 1)),
                                            posterior_tau_samples=jnp.transpose(results_mcmc["tau"], (2,0,1))
                                            ) 
    

    posterior_df = posterior_to_df(pos, id_df["id"], metrics, range(18,39))
    
    
    posterior_df.to_csv(f"posterior_ar{model_suffix}.csv", index = False)
    
    posterior_peaks = posterior_peaks_to_df(peaks, id_df["id"], metrics)
    posterior_peaks.to_csv(f"posterior_peaks_ar{model_suffix}.csv", index = False)
    if not injury:

        posterior_df_no_ar = posterior_to_df(pos_no_ar, id_df["id"], metrics, range(18,39))
        posterior_df_no_ar.to_csv(f"posterior_no_ar{model_suffix}.csv", index = False)
        

    
        posterior_mu_df = posterior_to_df(jnp.transpose(mu_mcmc, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(18,39))
        posterior_mu_df.to_csv(f"posterior_mu_ar{model_suffix}.csv", index = False)

        posterior_third_deriv = posterior_peaks_to_df(third_deriv, id_df["id"], metrics)
        posterior_third_deriv.to_csv(f"posterior_third_deriv_ar{model_suffix}.csv", index = False)

        posterior_first_deriv = posterior_to_df(jnp.transpose(first_deriv, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(18,39))
        posterior_first_deriv.to_csv(f"posterior_first_deriv_ar{model_suffix}.csv", index = False)

        posterior_peak_vals = posterior_peaks_to_df(peak_val, id_df["id"], metrics)
        posterior_peak_vals.to_csv(f"posterior_peak_vals_ar{model_suffix}.csv", index = False)


    latent_space_df = pd.DataFrame(results_map["X"], columns = [f"Dim {i+1}" for i in range(results_map["X"].shape[1])])
    latent_space_df = pd.concat([latent_space_df, id_df], axis = 1)
    latent_space_df.to_csv(f"latent_space{model_suffix}.csv", index = False)

    df = posterior_X_to_df(results_mcmc["X"], id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], player_indices)
    df.to_csv(f"latent_X{model_suffix}.csv", index = False)

    if "hsgp" in model_name:
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
        phi_x_latent = results_map["X"][:, basis_dims // 3 : ]
        phi_x_t = results_map["X"][:, 0: basis_dims // 3]
        phi_x_c = results_map["X"][:, basis_dims // 3 : 2 * basis_dims // 3]



    phi_X_df = pd.DataFrame(phi_x_latent, columns = [f"Dim {i+1}" for i in range(phi_x_latent.shape[1])])
    phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
    phi_X_df.to_csv(f"phi_X{model_suffix}.csv", index = False)
    if ("hsgplvm" in model_name or "linear" in model_name) and "spectral" not in model_name:
        phi_X_df = pd.DataFrame(phi_x_t, columns = [f"Dim {i+1}" for i in range(phi_x_t.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
        phi_X_df.to_csv(f"phi_X_peak_age{model_suffix}.csv", index = False)
        phi_X_df = pd.DataFrame(phi_x_c, columns = [f"Dim {i+1}" for i in range(phi_x_c.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
        phi_X_df.to_csv(f"phi_X_peak_value{model_suffix}.csv", index = False)






