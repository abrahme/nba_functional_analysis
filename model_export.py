import pandas as pd
import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import arviz as az
from jax import config, vmap
from numpyro.diagnostics import print_summary
config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, average_peak_differences, average_range_differences, create_surv_data
import numpyro
import jax.numpy as jnp
from model.model_utils import make_mu_rflvm, make_mu_hsgp, make_mu_linear, make_mu_rflvm_mcmc_AR, make_mu_hsgp_mcmc_AR, make_mu_linear_mcmc_AR, make_mu_linear_mcmc, compute_residuals_map, compute_priors, make_survival_linear_injury_mcmc, apply_detrend_for_offsets, make_survival_linear_mcmc
from model.inference_utils import posterior_peaks_to_df, posterior_to_df, posterior_X_to_df, posterior_injury_to_df, posterior_injury_prior_mean_to_df, posterior_survival_to_df, posterior_player_scalar_to_df
from model.hsgp import vmap_make_convex_phi, eigenfunctions_multivariate, make_spectral_mixture_density, diag_spectral_density, sqrt_eigenvalues
from visualization.visualization import make_diagnostic_heatmap, make_rhat_summary_barchart
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
    injury = args["injury"]
    if "hsgp" in model_name:
        model_suffix = "_hsgp"
    elif "linear" in model_name:
        model_suffix = "_linear"
    elif "rflvm" in model_name:
        model_suffix = "_rff"
    else:
        model_suffix = "_"
    if injury:
        model_suffix += "_injury"
    if "counterfactual" in model_name:
        model_suffix += "_counterfactual"
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

    
    data = pd.read_csv("data/injury_player_cleaned.csv").query(f"age <= 38 & name != 'Brandon Williams'") ### filter out years that happen after this year
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
    fake_data = pd.DataFrame({"age": range(18,39), "id": 99999999, "year": range(2000, 2021), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    data = pd.concat([data, fake_data], ignore_index=True)
    validation_mask = data[["year", "age", "id"]].pivot(columns="age", index="id", values=f"year").reindex(columns = range(18,39)).apply(
                                                                        lambda r: r.dropna().iloc[0] + (np.array(range(18,39)) - r.dropna().index[0]) if r.notna().any() else r,
                                                                        axis=1,
                                                                        result_type="expand").to_numpy() > validation_year
    # validation_mask = data[["split","age", "id"]].pivot(columns="age", index="id", values="split").reindex(columns = range(18,39)).to_numpy() == "test"
    metric_output = ["beta-binomial", "beta", "beta"] + (["gaussian"] * 2) + (["poisson"] * 6) + (["negative-binomial"] * 3) + (["binomial"] * 3)
    metrics = ["games","usg", "pct_minutes",  "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a", "fg3a", "ftm","fg2m", "fg3m"]
    exposure_list = ([ "games_exposure", "minutes", "games"]) + (["minutes"] * 11) + ["fta","fg2a", "fg3a"]

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
    Y_for_offsets = apply_detrend_for_offsets(
        Y_obs=Y,
        exposures_obs=exposures,
        metric_families=metric_output,
        de_trend_values=de_trend,
        de_trend_mask=jnp.array(de_trend_indices),
    )
    offset_max, offset_max_var, offset_peak, offset_peak_var =  compute_priors(Y_for_offsets, exposures, metric_output, exposure_list)
    offset_peak = offset_peak + 18 - basis.mean()

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
    
    df = posterior_X_to_df(results_mcmc["X"], id_df["id"], id_df["name"], id_df["minutes"], id_df["position_group"], [])
    df.to_csv(f"posterior_latent_X{model_suffix}.csv", index = False)
    _summary_vars = ["sigma_beta", "sigma_beta_binomial", "sigma", "sigma_ar", "sigma_negative_binomial"]
    _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
    summary = az.summary(_summary_subset)
    print(summary)
    summary.to_csv(f"model_output/posterior_variance{model_suffix}_summary.csv")
    survival_injury_keys = {
        "entrance",
        "entrance_global_offset",
        "exit_global_offset",
        "sigma_entrance",
        "entrance_latent",
        "exit",
        "exit_rate",
        "injury_factor",
        "injury_exit_loading",
        "injury_exit_global_offset",
        "sigma_injury_exit",
        "injury_exit_raw",
    }
    has_survival_injury = all(key in results_mcmc for key in survival_injury_keys)

    _, surv_data_set, _ = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year)
    surv_masks = jnp.stack([data_entity["censored"] for data_entity in surv_data_set], -1)
    Y_surv = jnp.stack([data_entity["observations"] for data_entity in surv_data_set], -1)
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

    mu += de_trend_adjusted
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

    latent_val = mu_mcmc + AR + de_trend_adjusted
    if injury:
        injury_loading = results_mcmc["injury_loading"]
        injury_factor = results_mcmc["injury_factor"]
        sigma_injury = results_mcmc["sigma_injury"]
        injury_mean_prior = jnp.einsum("...ip, ...kp -> ...ki", injury_factor, injury_loading )
        injury_raw = results_mcmc["injury_raw"]
        injury_effect_raw = injury_mean_prior[:,:,:,None, None, :] + injury_raw * sigma_injury[..., None, None, None]
        injury_effect = jnp.take_along_axis(jnp.concatenate([jnp.zeros_like(AR)[..., None], injury_effect_raw ], -1), injury_types[..., None][None, None], -1).squeeze(-1) 
        latent_val = latent_val + injury_effect

        injury_posterior_df = posterior_injury_to_df(
            injury_effect_raw,
            id_df["id"].to_numpy(),
            metrics,
            range(18, 39),
            injury_type_ids,
            injury_type_labels,
            injury_at_age=jnp.any(injury_masks, axis=0).astype(jnp.int32),
        )
        injury_posterior_df.to_csv(f"posterior_injury_samples{model_suffix}.csv", index=False)

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

        injury_prior_df = posterior_injury_prior_mean_to_df(
            injury_prior_mean_export,
            injury_prior_metrics,
            injury_type_ids,
            injury_type_labels,
        )
        injury_prior_df.to_csv(f"posterior_injury_prior_mean{model_suffix}.csv", index=False)
    else:
        injury_effect = jnp.zeros_like(latent_val)

    if has_survival_injury and injury:
            surv_posterior = make_survival_linear_injury_mcmc(
                X=results_mcmc["X"],
                entrance=results_mcmc["entrance"],
                entrance_global_offset=results_mcmc["entrance_global_offset"],
                exit_global_offset=results_mcmc["exit_global_offset"],
                sigma_entrance=results_mcmc["sigma_entrance"],
                entrance_latent_raw=results_mcmc["entrance_latent"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_factor=results_mcmc["injury_factor"],
                injury_exit_loading=results_mcmc["injury_exit_loading"],
                injury_exit_global_offset=results_mcmc["injury_exit_global_offset"],
                sigma_injury_exit=results_mcmc["sigma_injury_exit"],
                injury_exit_raw=results_mcmc["injury_exit_raw"],
                injury_indicator=injury_masks,
                injury_type=injury_types,
                entrance_times=Y_surv[:, 0] - 18 + 1e-6,
                left_censor=surv_masks[:, 0],
                basis=basis,
                entry_shift=0.0,
            )

            observed_surv_df = pd.DataFrame(
                {
                    "player": id_df["id"].to_numpy(),
                    "observed_entrance_age": np.asarray(Y_surv[:, 0]),
                    "observed_exit_age": np.asarray(Y_surv[:, 1]),
                    "entrance_censored": np.asarray(surv_masks[:, 0]).astype(np.int32),
                    "exit_censored": np.asarray(surv_masks[:, 1]).astype(np.int32),
                }
            )

            entrance_survival_df = posterior_survival_to_df(
                surv_posterior["entrance_survival"],
                id_df["id"],
                range(18, 39),
                "entrance_survival",
            )
            entrance_survival_df = entrance_survival_df.merge(observed_surv_df, on="player", how="left")
            entrance_survival_df["scenario"] = "observed"
            entrance_survival_df.to_csv(f"posterior_entrance_survival{model_suffix}.csv", index=False)

            entrance_hazard_df = posterior_survival_to_df(
                surv_posterior["entrance_hazard"],
                id_df["id"],
                range(18, 39),
                "entrance_hazard",
            )
            entrance_hazard_df = entrance_hazard_df.merge(observed_surv_df, on="player", how="left")
            entrance_hazard_df["scenario"] = "observed"
            entrance_hazard_df.to_csv(f"posterior_entrance_hazard{model_suffix}.csv", index=False)

            entrance_latent_df = posterior_player_scalar_to_df(
                surv_posterior["entrance_latent"],
                id_df["id"],
                "entrance_latent_duration",
            )
            entrance_latent_df = entrance_latent_df.merge(observed_surv_df, on="player", how="left")
            entrance_latent_df["scenario"] = "observed"
            entrance_latent_df.to_csv(f"posterior_entrance_latent{model_suffix}.csv", index=False)
    
            surv_posterior_counterfactual = make_survival_linear_injury_mcmc(
                X=results_mcmc["X"],
                entrance=results_mcmc["entrance"],
                entrance_global_offset=results_mcmc["entrance_global_offset"],
                exit_global_offset=results_mcmc["exit_global_offset"],
                sigma_entrance=results_mcmc["sigma_entrance"],
                entrance_latent_raw=results_mcmc["entrance_latent"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                injury_factor=results_mcmc["injury_factor"],
                injury_exit_loading=results_mcmc["injury_exit_loading"],
                injury_exit_global_offset=results_mcmc["injury_exit_global_offset"],
                sigma_injury_exit=results_mcmc["sigma_injury_exit"],
                injury_exit_raw=results_mcmc["injury_exit_raw"],
                injury_indicator=jnp.zeros_like(injury_masks),
                injury_type=jnp.zeros_like(injury_types),
                entrance_times=Y_surv[:, 0] - 18 + 1e-6,
                left_censor=surv_masks[:, 0],
                basis=basis,
            )


            exit_survival_df_obs = posterior_survival_to_df(
                surv_posterior["exit_survival"],
                id_df["id"],
                range(18, 39),
                "exit_survival",
            )
            exit_survival_df_obs = exit_survival_df_obs.merge(observed_surv_df, on="player", how="left")
            exit_survival_df_obs["scenario"] = "observed"
            exit_survival_df_cf = posterior_survival_to_df(
                surv_posterior_counterfactual["exit_survival"],
                id_df["id"],
                range(18, 39),
                "exit_survival",
            )
            exit_survival_df_cf = exit_survival_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_survival_df_cf["scenario"] = "counterfactual"
            pd.concat([exit_survival_df_obs, exit_survival_df_cf], ignore_index=True).to_csv(
                f"posterior_exit_survival{model_suffix}.csv", index=False
            )

            exit_hazard_df_obs = posterior_survival_to_df(
                surv_posterior["exit_hazard"],
                id_df["id"],
                range(18, 39),
                "exit_hazard",
            )
            exit_hazard_df_obs = exit_hazard_df_obs.merge(observed_surv_df, on="player", how="left")
            exit_hazard_df_obs["scenario"] = "observed"
            exit_hazard_df_cf = posterior_survival_to_df(
                surv_posterior_counterfactual["exit_hazard"],
                id_df["id"],
                range(18, 39),
                "exit_hazard",
            )
            exit_hazard_df_cf = exit_hazard_df_cf.merge(observed_surv_df, on="player", how="left")
            exit_hazard_df_cf["scenario"] = "counterfactual"
            pd.concat([exit_hazard_df_obs, exit_hazard_df_cf], ignore_index=True).to_csv(
                f"posterior_exit_hazard{model_suffix}.csv", index=False
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
            pd.concat([exit_age_sample_df_obs, exit_age_sample_df_cf], ignore_index=True).to_csv(
                f"posterior_exit_age_sample{model_suffix}.csv", index=False
            )
    else:
        surv_posterior = make_survival_linear_mcmc(
                X=results_mcmc["X"],
                entrance=results_mcmc["entrance"],
                entrance_global_offset=results_mcmc["entrance_global_offset"],
                exit_global_offset=results_mcmc["exit_global_offset"],
                sigma_entrance=results_mcmc["sigma_entrance"],
                entrance_latent_raw=results_mcmc["entrance_latent"],
                exit=results_mcmc["exit"],
                exit_rate=results_mcmc["exit_rate"],
                entrance_times=Y_surv[:, 0] - 18 + 1e-6,
                left_censor=surv_masks[:, 0],
                basis=basis,
                entry_shift=0.0,
            )

        observed_surv_df = pd.DataFrame(
            {
                "player": id_df["id"].to_numpy(),
                "observed_entrance_age": np.asarray(Y_surv[:, 0]),
                "observed_exit_age": np.asarray(Y_surv[:, 1]),
                "entrance_censored": np.asarray(surv_masks[:, 0]).astype(np.int32),
                "exit_censored": np.asarray(surv_masks[:, 1]).astype(np.int32),
            }
        )

        entrance_survival_df = posterior_survival_to_df(
            surv_posterior["entrance_survival"],
            id_df["id"],
            range(18, 39),
            "entrance_survival",
        )
        entrance_survival_df = entrance_survival_df.merge(observed_surv_df, on="player", how="left")
        entrance_survival_df["scenario"] = "observed"
        entrance_survival_df.to_csv(f"posterior_entrance_survival{model_suffix}.csv", index=False)

        entrance_hazard_df = posterior_survival_to_df(
            surv_posterior["entrance_hazard"],
            id_df["id"],
            range(18, 39),
            "entrance_hazard",
        )
        entrance_hazard_df = entrance_hazard_df.merge(observed_surv_df, on="player", how="left")
        entrance_hazard_df["scenario"] = "observed"
        entrance_hazard_df.to_csv(f"posterior_entrance_hazard{model_suffix}.csv", index=False)

        entrance_latent_df = posterior_player_scalar_to_df(
            surv_posterior["entrance_latent"],
            id_df["id"],
            "entrance_latent_duration",
        )
        entrance_latent_df = entrance_latent_df.merge(observed_surv_df, on="player", how="left")
        entrance_latent_df["scenario"] = "observed"
        entrance_latent_df.to_csv(f"posterior_entrance_latent{model_suffix}.csv", index=False)

        


        exit_survival_df_obs = posterior_survival_to_df(
            surv_posterior["exit_survival"],
            id_df["id"],
            range(18, 39),
            "exit_survival",
        )
        exit_survival_df_obs = exit_survival_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_survival_df_obs["scenario"] = "observed"
        
        exit_survival_df_obs.to_csv(
            f"posterior_exit_survival{model_suffix}.csv", index=False
        )

        exit_hazard_df_obs = posterior_survival_to_df(
            surv_posterior["exit_hazard"],
            id_df["id"],
            range(18, 39),
            "exit_hazard",
        )
        exit_hazard_df_obs = exit_hazard_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_hazard_df_obs["scenario"] = "observed"
        
        exit_hazard_df_obs.to_csv(
            f"posterior_exit_hazard{model_suffix}.csv", index=False
        )

        exit_age_sample_df_obs = posterior_player_scalar_to_df(
            surv_posterior["exit_age_sample"],
            id_df["id"],
            "exit_age_sample",
        )
        exit_age_sample_df_obs = exit_age_sample_df_obs.merge(observed_surv_df, on="player", how="left")
        exit_age_sample_df_obs["scenario"] = "observed"
        
        exit_age_sample_df_obs.to_csv(
            f"posterior_exit_age_sample{model_suffix}.csv", index=False
        )
        

    players = id_df[id_df["name"].isin(predict_players)].index
    player_names = id_df[id_df["name"].isin(predict_players)]["name"].tolist()

    os.makedirs("model_output/model_plots/coverage", exist_ok=True)

    players_idx = jnp.array(id_df.index)
    ages = list(range(18, 39))
    n_players_sel = len(id_df)
    n_ages = len(ages)



    make_diagnostic_heatmap(
        latent_val,
        n_players_sel,
        n_ages,
        ages,
        f"model_output/model_plots/coverage/posterior_latent_ar{model_suffix}.png",
        player_labels=all_player_labels,
    )
    make_diagnostic_heatmap(
        mu_mcmc,
        n_players_sel,
        n_ages,
        ages,
        f"model_output/model_plots/coverage/posterior_mu_ar{model_suffix}.png",
        player_labels=all_player_labels,
    )
    make_rhat_summary_barchart(
        latent_val,
        f"model_output/model_plots/coverage/rhat_summary_latent_ar{model_suffix}.png",
        metric_labels=metrics,
    )
    make_rhat_summary_barchart(
        mu_mcmc,
        f"model_output/model_plots/coverage/rhat_summary_mu_ar{model_suffix}.png",
        metric_labels=metrics,
    )




    _summary_vars = ["sigma_beta", "sigma_beta_binomial"]
    _summary_subset = {k: results_mcmc[k] for k in _summary_vars if k in results_mcmc}
    summary = az.summary(_summary_subset)
    summary.to_csv(f"model_output/posterior_variance{model_suffix}_summary.csv")

    peaks = tmax_mcmc + basis.mean()
    peak_val = cmax_mcmc 
    
    _neg_bin_samples = jnp.transpose(results_mcmc["sigma_negative_binomial"], (2, 0, 1)) if "sigma_negative_binomial" in results_mcmc else None
    _, pos = create_metric_trajectory_all(latent_val, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    _, pos_mu = create_metric_trajectory_all(mu_mcmc + de_trend_adjusted, Y, exposures,
                                            metric_output, metrics, exposure_list,
                                            jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                            jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                            posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                            posterior_neg_bin_samples=_neg_bin_samples,
                                            )
    

    posterior_df = posterior_to_df(pos, id_df["id"], metrics, range(18,39))
    posterior_df.to_csv(f"posterior_ar{model_suffix}.csv", index = False)
    
    posterior_peaks = posterior_peaks_to_df(peaks, id_df["id"], metrics)
    posterior_peaks.to_csv(f"posterior_peaks_ar{model_suffix}.csv", index = False)
    
    if injury and ("counterfactual" not in model_name):
        _, pos_counterfactual = create_metric_trajectory_all(mu_mcmc + AR + de_trend_adjusted, Y, exposures,
                                        metric_output, metrics, exposure_list,
                                        jnp.transpose(results_mcmc["sigma"], (2, 0, 1)),
                                        jnp.transpose(results_mcmc["sigma_beta"],(2, 0, 1)),
                                        posterior_kappa_samples=jnp.transpose(results_mcmc["sigma_beta_binomial"], (2, 0, 1)),
                                        posterior_neg_bin_samples=_neg_bin_samples,
                                        )
        posterior_counterfactual_df = posterior_to_df(pos_counterfactual, id_df["id"], metrics, range(18,39))
        posterior_counterfactual_df.to_csv(f"posterior_counterfactual_ar{model_suffix}.csv", index = False)
        

    posterior_mu_df = posterior_to_df(jnp.transpose(mu_mcmc + de_trend_adjusted, (0, 1, 3, 4, 2)), id_df["id"], metrics, range(18,39))
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
        phi_x_latent = results_map["X"]




    phi_X_df = pd.DataFrame(phi_x_latent, columns = [f"Dim {i+1}" for i in range(phi_x_latent.shape[1])])
    phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
    phi_X_df.to_csv(f"phi_X{model_suffix}.csv", index = False)
    if ("hsgplvm" in model_name) and "spectral" not in model_name:
        phi_X_df = pd.DataFrame(phi_x_t, columns = [f"Dim {i+1}" for i in range(phi_x_t.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
        phi_X_df.to_csv(f"phi_X_peak_age{model_suffix}.csv", index = False)
        phi_X_df = pd.DataFrame(phi_x_c, columns = [f"Dim {i+1}" for i in range(phi_x_c.shape[1])])
        phi_X_df = pd.concat([phi_X_df, id_df], axis = 1)
        phi_X_df.to_csv(f"phi_X_peak_value{model_suffix}.csv", index = False)






