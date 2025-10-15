import pandas as pd
import pickle
import re
import numpy as np
import argparse
import arviz as az
from jax import config
config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, average_peak_differences, average_range_differences
import numpyro
import jax.numpy as jnp
from model.model_utils import make_mu, make_mu_mcmc_AR, make_mu_mcmc, make_mu_mcmc_AR_fixed_X, transform_mu, compute_residuals_map, make_mu_mcmc_fixed_X
from model.inference_utils import posterior_peaks_to_df, posterior_to_df, loadings_to_df, time_factors_to_df
from model.hsgp import vmap_make_convex_phi

from model.inference_utils import create_metric_trajectory_all, create_metric_trajectory_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--injury", help="whether to mask out injury years", action="store_true")
    parser.add_argument("--rff_dim", help="size of the rff approx", required=True, type=int)
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
    basis_dims = args["basis_dims"]
    de_trend_metrics = args["de_trend_metrics"]
    injury = args["injury"]
    rff_dim = args["rff_dim"]
    svi_path = args["svi_path"]
    position_group = args["position_group"]
    players = args["player_names"]
    thin = args["thin"]
    validation_year = args["validation_year"]
    cohort_year = args["cohort_year"]

    data = pd.read_csv("data/injury_player_cleaned.csv").query(f"age <= 38 & name != 'Brandon Williams' & year <= {validation_year}")
    data = data.groupby("id").filter(lambda x: x["year"].min() <= cohort_year) ### filter out players who entered the league after this cohort year

    names = data.groupby("id")["name"].first().values.tolist()
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(data["total_games"], data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games"]) / 48
    data["retirement"] = 1

    id_df = data[["position_group","name","id", "minutes"]].groupby("id").max().reset_index()
    agg_dict = {"obpm":"mean", "dbpm":"mean",  
            "position_group": "max",
        "minutes":"sum", "dreb": "sum", "fta":"sum", "ftm":"sum", "oreb":"sum",
        "ast":"sum", "tov":"sum", "fg2m":"sum", "fg3m":"sum", "fg3a":"sum", "fg2a":"sum", "blk":"sum", "stl":"sum"}
    agged_data = data.groupby("id").agg(agg_dict).reset_index()
    agged_data["ft_pct"] = agged_data["ftm"] / agged_data["fta"]
    agged_data["fg2_pct"] = agged_data["fg2m"] / agged_data["fg2a"]
    agged_data["fg3_pct"] = agged_data["fg3m"] / agged_data["fg3a"]
    agged_data["dreb_rate"] = 36.0 * agged_data["dreb"] / agged_data["minutes"]
    agged_data["oreb_rate"] = 36.0 * agged_data["oreb"] / agged_data["minutes"]
    agged_data["ast_rate"] = 36.0 * agged_data["ast"] / agged_data["minutes"]
    agged_data["tov_rate"] = 36.0 * agged_data["tov"] / agged_data["minutes"]
    agged_data["blk_rate"] = 36.0 * agged_data["blk"] / agged_data["minutes"]
    agged_data["stl_rate"] = 36.0 * agged_data["stl"] / agged_data["minutes"]
    agged_data["ft_rate"] = 36.0 * agged_data["fta"] / agged_data["minutes"]
    agged_data["fg2_rate"] = 36.0 * agged_data["fg2a"] / agged_data["minutes"]
    agged_data["fg3_rate"] = 36.0 * agged_data["fg3a"] / agged_data["minutes"]
    agged_data.fillna(0, inplace=True)


    metric_output = ["beta-binomial", "beta"] + (["gaussian"] * 2) + (["negative-binomial"] * 9) + (["binomial"] * 3)
    metrics = ["games", "pct_minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["games_exposure", "games_exposure"]) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    print("setup data")
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
    positions = ["G", "F", "C"]
    covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, [], validation_year=validation_year, injury=injury)
    hsgp_params = {}
    x_time = basis - basis.mean()
    L_time = 2.0 * np.max(np.abs(x_time), 0, keepdims=True)
    M_time = 20
    phi_time = vmap_make_convex_phi(np.squeeze(x_time), np.squeeze(L_time), M_time)
    hsgp_params["phi_x_time"] = phi_time
    hsgp_params["M_time"] = M_time
    hsgp_params["L_time"] = L_time
    hsgp_params["shifted_x_time"] = x_time + L_time

    masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
    exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
    Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
    injury_masks = jnp.stack([data_entity["injury_mask"] for data_entity in data_set])

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
            if family in ["negative-binomial","poisson"]:
                p = jnp.nansum(Y[index]) / jnp.nansum(np.exp(exposures[index]))
                p_max = jnp.nanmean(jnp.nanmax(Y[index] / np.exp(exposures[index]), -1))
                peak = jnp.nanmean(jnp.nanargmax(Y[index] / np.exp(exposures[index]), -1))
                offset_list.append(np.log(p))
                offset_max_list.append(np.log(p_max))   

            elif family in ["beta-binomial", "binomial"]:
                p = jnp.nansum(Y[index]) / jnp.nansum(exposures[index])
                p_max = jnp.nanmean(jnp.nanmax(Y[index] / exposures[index], -1))
                offset_list.append(np.log(p/ (1-p)))
                offset_max_list.append(np.log(p_max/(1-p_max)))
                peak = jnp.nanmean(jnp.nanargmax(Y[index] / exposures[index], -1))
                p_star = Y[index] / exposures[index] 

            elif family == "beta":
                p = jnp.nanmean(Y[index])
                p_max = jnp.nanmedian(jnp.nanmax(Y[index], -1))
                peak = jnp.nanmean(jnp.nanargmax(Y[index], -1))
                offset_list.append(np.log(p / (1 - p)))
                offset_max_list.append(np.log(p_max/(1-p_max)))

    
        offset_peak_list.append(peak + 18 - basis.mean())
    offsets = np.array(offset_list)[None]
    offset_max = np.array(offset_max_list)[None]
    offset_peak = np.array(offset_peak_list)[None]
    offset_boundary_r = np.log(np.exp(2) - 1)
    offset_boundary_l = np.log(np.exp(2) - 1)
    offset_dict = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l}
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
        if item == "X":
            X_new = jnp.tile(results_mcmc["X"][None, None], (1, 50, 1, 1))
            X_new = X_new.at[..., jnp.array(player_indices), :].set(results_mcmc["X_free"])
            results_mcmc["X"] = X_new
    mu, *_ = make_mu(results_map["X"], results_map["lengthscale_deriv"], results_map["alpha"], results_map["beta"],
                                            results_map["W"], results_map["lengthscale"], results_map["c_max"], results_map["t_max_raw"], results_map["sigma_t"],
                                            results_map["sigma_c"], L_time, M_time, phi_time, x_time + L_time, offset_dict)
    obs, preds = create_metric_trajectory_map(mu, [], Y, exposures, metric_output, metrics)
                        
    avg_sd, autocorr = compute_residuals_map(preds["y"], obs["y"], exposures, metric_output, results_map["sigma"], results_map["sigma_negative_binomial"], 
                                                                results_map["sigma_beta_binomial"], results_map["sigma_beta"])
    # avg_sd = jnp.ones((len(metrics))) * .01
    # autocorr = jnp.zeros_like(avg_sd)
    wTx, mu_mcmc, tmax_mcmc, cmax_mcmc, AR, second_deriv, third_deriv, first_deriv = make_mu_mcmc_AR(results_mcmc["X"], results_mcmc["lengthscale_deriv"], results_mcmc["alpha"],
                        results_mcmc["beta"], results_mcmc["W"], results_mcmc["lengthscale"], results_mcmc["c_max"],
                        results_mcmc["t_max_raw"], results_mcmc["sigma_t"], results_mcmc["sigma_c"], L_time, M_time, x_time + L_time, offset_dict,
                        sigma_ar = results_mcmc["sigma_ar"],
                        # sigma_ar = avg_sd[..., None][None, None],
                        beta_ar = results_mcmc["beta_ar"], 
                        rho_ar=results_mcmc["rho_ar"],
                        # rho_ar = autocorr[..., None][None, None],
                        AR_0_raw=results_mcmc["AR_0"],
                        # AR_0_raw = jnp.zeros((len(metrics), covariate_X.shape[0])),
                          phi_time=phi_time, orthogonalize=False)

    peaks = tmax_mcmc + basis.mean()
    peak_val = cmax_mcmc 
    
    _, pos = create_metric_trajectory_all(mu_mcmc + AR, Y, exposures, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"][None, None], (2,0,1)),
                                            results_mcmc["sigma_beta"][..., None, None],
                                            results_mcmc["sigma_beta_binomial"][..., None, None],
                                            posterior_neg_bin_samples=results_mcmc["sigma_negative_binomial"],
                                            ) 
    _, pos_no_ar = create_metric_trajectory_all(mu_mcmc, Y, exposures, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"][None, None], (2,0,1)),
                                            results_mcmc["sigma_beta"][..., None, None],
                                            results_mcmc["sigma_beta_binomial"][..., None, None],
                                            posterior_neg_bin_samples=results_mcmc["sigma_negative_binomial"],
                                            ) 
    

    posterior_df = posterior_to_df(pos, id_df["id"], metrics, range(18,39))
    
    posterior_df.to_csv("posterior_ar_2025.csv", index = False)
    
    posterior_peaks = posterior_peaks_to_df(peaks, id_df["id"], metrics)
    posterior_peaks.to_csv("posterior_peaks_ar_2025.csv", index = False)
    if not injury:

        posterior_df_no_ar = posterior_to_df(pos_no_ar, id_df["id"], metrics, range(18,39))
        posterior_df_no_ar.to_csv("posterior_no_ar_2025.csv", index = False)
        

    
        posterior_mu_df = posterior_to_df(jnp.transpose(mu_mcmc, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(18,39))
        posterior_mu_df.to_csv("posterior_mu_ar_2025.csv", index = False)

        posterior_third_deriv = posterior_peaks_to_df(third_deriv, id_df["id"], metrics)
        posterior_third_deriv.to_csv("posterior_third_deriv_ar_2025.csv", index = False)

        posterior_first_deriv = posterior_to_df(jnp.transpose(first_deriv, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(18,39))
        posterior_first_deriv.to_csv("posterior_first_deriv_ar_2025.csv", index = False)

        posterior_peak_vals = posterior_peaks_to_df(peak_val, id_df["id"], metrics)
        posterior_peak_vals.to_csv("posterior_peak_vals_ar_2025.csv", index = False)


    latent_space_df = pd.DataFrame(results_map["X"], columns = [f"Dim {i+1}" for i in range(results_map["X"].shape[1])])
    latent_space_df = pd.concat([latent_space_df, id_df], axis = 1)
    latent_space_df.to_csv("latent_space_2025.csv", index = False)

    wTx = jnp.einsum("nr,mr -> nm", results_map["X"], results_map["W"]  * jnp.sqrt(results_map["lengthscale"]))
    phi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(rff_dim))   
    phi_X_df = pd.DataFrame(phi_x, columns = [f"Dim {i+1}" for i in range(phi_x.shape[1])])
    phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
    phi_X_df.to_csv("phi_X_2025.csv", index = False)


