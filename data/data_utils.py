import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
import jax.scipy.special as jsci
from numpyro.distributions import Normal, Exponential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def generalized_beta_density(x, alpha, beta_, min_val, max_val):
    y = (x - min_val)/(max_val - min_val)
    return (jnp.power(y, alpha) * jnp.power(1 - y, beta_))/((max_val - min_val) *  jsci.beta(alpha, beta_))

def create_convex_data(num_samples, noise_level = .01, exposure = 1, data_range = [0, 1], alpha = 2, beta = 5, key = random.PRNGKey(0)):
    intercept = Normal().sample(key)
    print(f"intercept: {intercept}")
    multiplier = Exponential().sample(key)
    print(f"multiplier : {multiplier}")
    noise = Normal().sample(key, sample_shape=(num_samples,)) * (noise_level / exposure)
    print(f"noise level: {noise_level}")
    samples = jnp.linspace(data_range[0], data_range[1], num_samples)
    y_vals = generalized_beta_density(samples, alpha, beta, data_range[0], data_range[1])

    return samples, intercept, multiplier, noise, y_vals

def process_data(df, output_metric, exposure, model, input_metrics, player_indices:list = [], normalize = False, validation_year = 2021, injury = False):

    agg_dict = {input_metric:"max" for input_metric in input_metrics}
    df = df.sort_values(by=["id","year"])
    df["ft_pct"] = df["ftm"] / df["fta"]
    df["three_pct"] = df["fg3m"] / df["fg3a"]
    df["two_pct"] = df["fg2m"] / df["fg2a"]
    if input_metrics:
        X = df[input_metrics + ["id"]].groupby("id").agg(agg_dict).reset_index()[input_metrics]
    else:
        X = None
    de_trend_array = df[[f"{output_metric}_league_avg", "id", "age"]].pivot(columns="age", index="id", values=f"{output_metric}_league_avg").reindex(columns = range(18,39)).to_numpy()
    metric_df = df[[output_metric, "id", "age"]]
    exposure_df = df[["id", "age", exposure]]
    metric_df  = metric_df.pivot(columns="age",values=output_metric,index="id").reindex(columns=range(18,39))
    exposure_array = exposure_df.pivot(columns="age", index="id", values=exposure).reindex(columns = range(18,39)).to_numpy()
    if model == "poisson":
        metric_array = metric_df.to_numpy()
        if normalize:
            metric_array_obs = metric_array / exposure_array
            metric_array = (metric_array_obs - np.nanmean(metric_array_obs))/np.nanstd(metric_array_obs)
        adj_exp_array = jnp.log(exposure_array) if not normalize else jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array)
    elif model == "negative-binomial":
        metric_array = metric_df.to_numpy()
        if normalize:
            metric_array_obs = metric_array / exposure_array
            metric_array = (metric_array_obs - np.nanmean(metric_array_obs))/np.nanstd(metric_array_obs)
        adj_exp_array = jnp.log(exposure_array) if not normalize else jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array)
    elif model == "beta":
        metric_array = metric_df.to_numpy()
        if normalize:
            metric_array_obs = metric_array
            metric_array = (metric_array_obs - np.nanmean(metric_array_obs))/np.nanstd(metric_array_obs)
        adj_exp_array = jnp.sqrt(exposure_array + 1) if not normalize else jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array / (1 - de_trend_array))
    elif model == "beta-binomial":
        metric_array = metric_df.to_numpy()
        metric_array[~np.isnan(metric_array)] = np.int32(metric_array[~np.isnan(metric_array)])
        exposure_array[~np.isnan(metric_array)] = np.int32(exposure_array[~np.isnan(exposure_array)])
        if normalize:
            metric_array_obs = metric_array / exposure_array
            metric_array = (metric_array_obs - np.nanmean(metric_array_obs)) / np.nanstd(metric_array_obs)
        adj_exp_array = jnp.array(exposure_array) if not normalize else jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array / (1 - de_trend_array))
    elif model in ["binomial"]:
        metric_array = metric_df.to_numpy()
        if exposure == "simple_exposure":
            season_array = df[["id", "age", "year"]].pivot(columns="age",values="year",index="id").reindex(columns = range(18,39)).to_numpy()
            retirement_array = np.where((season_array == validation_year).sum(axis = 1) == 0)[0] 
            entrance_array = np.where((season_array == 1997).sum(axis = 1) == 0)[0] 
            max_season_array = (21 - np.argmax(np.flip(~np.isnan(metric_array), axis = 1), axis = 1))
            min_season_array = np.argmax(~np.isnan(metric_array), axis = 1)
            for player_index, ret_index in zip(retirement_array, max_season_array[retirement_array]):
                metric_array[player_index, ret_index:] = 0
                exposure_array[player_index, ret_index:] = 1
            for player_index, ent_index in zip (entrance_array, min_season_array[entrance_array]):
                exposure_array[player_index, 0:ent_index] = 1 
                metric_array[player_index, 0:ent_index] = 0
        elif exposure == "games_exposure":
            season_array = df[["id", "age", "year"]].pivot(columns="age",values="year",index="id").to_numpy()
            retirement_array = np.where((season_array == validation_year).sum(axis = 1) == 0)[0] 
            entrance_array = np.where((season_array == 1997).sum(axis = 1) == 0)[0] 
            max_season_array = (21 - np.argmax(np.flip(~np.isnan(metric_array), axis = 1), axis = 1))
            min_season_array = np.argmax(~np.isnan(metric_array), axis = 1)
            for player_index, ret_index in zip(retirement_array, max_season_array[retirement_array]):
                metric_array[player_index, ret_index:] = 0
                exposure_array[player_index, ret_index:] = 82

            for player_index, ent_index in zip (entrance_array, min_season_array[entrance_array]):
                exposure_array[player_index, 0:ent_index] = 82 
                metric_array[player_index, 0:ent_index] = 0

        metric_array[~np.isnan(metric_array)] = np.int32(metric_array[~np.isnan(metric_array)])
        exposure_array[~np.isnan(metric_array)] = np.int32(exposure_array[~np.isnan(exposure_array)])
        if normalize:
            metric_array_obs = metric_array / exposure_array
            metric_array = (metric_array_obs - np.nanmean(metric_array_obs)) / np.nanstd(metric_array_obs)
        adj_exp_array = jnp.array(exposure_array) if not normalize else jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array / (1 - de_trend_array))
    elif model == "gaussian":
        metric_array = metric_df.to_numpy()
        if normalize:
            metric_array = (metric_array - np.nanmean(metric_array)) / (np.nanstd(metric_array))
        adj_exp_array = jnp.sqrt(exposure_array)
        de_trend_array = jnp.log(de_trend_array / (1 - de_trend_array))
    if injury:  
        injury_array = df[["injury_period", "id", "age"]].pivot(columns = "age", values = "injury_period", index = "id").reindex(columns=range(18,39))
        injury_array = injury_array.ffill(axis = 1).fillna("pre-injury").to_numpy()
        injury_mask = (injury_array != "pre-injury")
        injury_type_array = df[["injury_code", "id", "age"]].pivot(columns = "age", values = "injury_code", index = "id").reindex(columns=range(18,39)) ### gives me per player, if they have an injury
        injury_type_array = injury_type_array.ffill(axis = 1).fillna(0).to_numpy()
        mask = injury_mask
        injury_type_array = jnp.where(injury_mask, injury_type_array, 0)
    else:
        mask = jnp.ones_like(metric_array, dtype= bool)
    return adj_exp_array, jnp.array(metric_array), X, mask, de_trend_array, injury_type_array if injury else jnp.zeros_like(metric_array)

def process_surv_data(df, output_metric, censor_type, input_metrics, validation_year = 2021):
    agg_dict = {input_metric:"max" for input_metric in input_metrics}
    df = df.sort_values(by=["id","year"])
    df["ft_pct"] = df["ftm"] / df["fta"]
    df["three_pct"] = df["fg3m"] / df["fg3a"]
    df["two_pct"] = df["fg2m"] / df["fg2a"]
    if input_metrics:
        X = df[input_metrics + ["id"]].groupby("id").agg(agg_dict).reset_index()[input_metrics]
    else:
        X = None
  
    metric_df = df[[output_metric, "id", "age"]]

    metric_df  = metric_df.pivot(columns="age",values=output_metric,index="id").reindex(columns=range(18,39))

    

    metric_array = metric_df.to_numpy()

    season_array = df[["id", "age", "year"]].pivot(columns="age",values="year",index="id").reindex(columns = range(18,39)).to_numpy()
    if censor_type == "right":
        cens_array = (season_array == validation_year).sum(axis = 1) != 0
        obs_array = (21 - np.argmax(np.flip(~np.isnan(metric_array), axis = 1), axis = 1)) + 18
    elif censor_type == "left":
        cens_array = (season_array == 1997).sum(axis = 1) != 0
        obs_array = np.argmax(~np.isnan(metric_array), axis = 1) + 18 ### here if cens_array == 1, we have censored the observation, if 0 it is observed
    return cens_array, obs_array




def create_basis(data, dims):
    agg_dict = {"obpm":"mean", "dbpm":"mean", 
            "minutes":"sum", "dreb": "sum", "fta":"sum", "ftm":"sum", "oreb":"sum",
            "ast":"sum", "tov":"sum", "fg2m":"sum", "fg3m":"sum", "fg3a":"sum", "fg2a":"sum", "blk":"sum", "stl":"sum"}
    agged_data = data.groupby("id").agg(agg_dict).reset_index()
    agged_data["ft_pct"] = agged_data["ftm"] / agged_data["fta"]
    agged_data["fg2_pct"] = agged_data["fg2m"] / agged_data["fg2a"]
    agged_data["fg3_pct"] = agged_data["fg3m"] / agged_data["fg3a"]
    agged_data["dreb_rate"] = agged_data["dreb"] / agged_data["minutes"]
    agged_data["oreb_rate"] = agged_data["oreb"] / agged_data["minutes"]
    agged_data["ast_rate"] = agged_data["ast"] / agged_data["minutes"]
    agged_data["tov_rate"] = agged_data["tov"] / agged_data["minutes"]
    agged_data["blk_rate"] = agged_data["blk"] / agged_data["minutes"]
    agged_data["stl_rate"] = agged_data["stl"] / agged_data["minutes"]
    agged_data.fillna(0, inplace=True)

    latent_metrics = ["obpm","dbpm","minutes","ft_pct","fg2_pct","fg3_pct","dreb_rate","oreb_rate","ast_rate","tov_rate","blk_rate","stl_rate"]
    X = StandardScaler().fit_transform(agged_data[latent_metrics])
    pca_x = PCA(whiten=True).fit(X)
    X_pca = pca_x.transform(X)
    covariate_X = jnp.array(MinMaxScaler().fit_transform(X_pca[:, 0:dims]))
    return covariate_X


def create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_index: list[int] = [], validation_year:int = 2021, injury: bool = False):
    covariate_X = create_basis(data, basis_dims)
    if player_index:
        covariate_X = covariate_X[jnp.array(player_index)]
    data_set = []
    for output,metric,exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _, injury_mask, de_trend, injury_type = process_data(data, metric, exposure_val, output, ["position_group"], validation_year=validation_year, injury=injury)
        if player_index:
            exposure = exposure[jnp.array(player_index)]
            Y = Y[jnp.array(player_index)]
            injury_mask = injury_mask[jnp.array(player_index)]
            de_trend = de_trend[jnp.array(player_index)]
            injury_type = injury_type[jnp.array(injury_type)]

        data_dict = {"metric":metric, "output": output, "exposure_data": exposure, "output_data": Y, "mask": jnp.isfinite(Y), "injury_mask": injury_mask,
                    "de_trend": de_trend, "injury_type": injury_type}
        data_set.append(data_dict)
    basis = jnp.arange(18,39)

    return covariate_X, data_set, basis


def create_surv_data(data, basis_dims, censor_type, metrics, player_index: list[int] = [], validation_year:int = 2021):
    covariate_X = create_basis(data, basis_dims)
    if player_index:
        covariate_X = covariate_X[jnp.array(player_index)]
    data_set = []
    for censor, metric in zip(censor_type, metrics):
        censor_mask, Y = process_surv_data(data, metric, censor, ["position_group"], validation_year)
        if player_index:
            censor_mask = censor_mask[jnp.array(player_index)]
            Y = Y[jnp.array(player_index)]
        data_dict = {"observations": Y, "censor_type":jnp.ones_like(Y) if censor == "right" else jnp.zeros_like(Y), "censored":censor_mask}
        data_set.append(data_dict)
    basis = jnp.arange(18,39)
    return covariate_X, data_set, basis

def average_peak_differences(x):
    mask = ~jnp.isnan(x)

    first_idx = jnp.argmax(mask, axis=1)
    last_idx = x.shape[1] - 1 - jnp.argmax(jnp.flip(mask, axis=1), axis=1)

    peak_vals = jnp.nanmax(x, axis=1)
    left_vals = x[jnp.arange(x.shape[0]), first_idx]
    right_vals = x[jnp.arange(x.shape[0]), last_idx]
    left_diffs = peak_vals - left_vals
    right_diffs = peak_vals - right_vals

    avg_left = jnp.nanmean(left_diffs)
    avg_right = jnp.nanmean(right_diffs)

    return avg_left, avg_right


def average_range_differences(x):
    peak_vals = jnp.nanmax(x, axis=1)
    min_vals = jnp.nanmin(x, axis = 1)
    pct_change = (peak_vals - min_vals)/peak_vals
    mask = jnp.isfinite(pct_change)
    return jnp.sum(jnp.where(mask, pct_change, 0.0)) / jnp.sum(mask)