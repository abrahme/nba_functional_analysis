import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def process_data(df, output_metric, exposure, model, input_metrics):

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
    exposure_df = df[["id", "age", exposure]]
    metric_df  = metric_df.pivot(columns="age",values=output_metric,index="id")
    if model == "poisson":
        offset = jnp.log(exposure_df.pivot(columns="age", values=exposure,index="id").to_numpy())
        return offset, jnp.array(metric_df.to_numpy()), X
    elif model == "binomial":
        trials = jnp.array(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return trials, jnp.array(metric_df.to_numpy()), X
    elif model == "gaussian":
        variance_scale = jnp.sqrt(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return variance_scale, jnp.array(metric_df.to_numpy()), X
    return ValueError

def process_data_time(df, output_metric, exposure, model):
    df = df.sort_values(by=["id","year"])
    df["ft_pct"] = df["ftm"] / df["fta"]
    df["three_pct"] = df["fg3m"] / df["fg3a"]
    df["two_pct"] = df["fg2m"] / df["fg2a"]

    df["year"] = pd.Categorical(df.year).codes.astype(int)

    metric_df = df[[output_metric, "id", "age"]]
    exposure_df = df[["id", "age", exposure]]
    year_df = df[["id", "age", "year"]].pivot(columns = "age", values = "year", index = "id")
    metric_df  = metric_df.pivot(columns="age",values=output_metric,index="id")
    if model == "poisson":
        offset = jnp.log(exposure_df.pivot(columns="age", values=exposure,index="id").to_numpy())
        return offset, jnp.array(metric_df.to_numpy()), jnp.array(year_df.to_numpy(dtype="int"))
    elif model == "binomial":
        trials = jnp.array(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return trials, jnp.array(metric_df.to_numpy()), jnp.array(year_df.to_numpy(dtype="int"))
    elif model == "gaussian":
        variance_scale = jnp.sqrt(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return variance_scale, jnp.array(metric_df.to_numpy()), jnp.array(year_df.to_numpy(dtype="int"))
    return ValueError


def process_data_multi_way(df, output_metric, exposure, model):
    n_players = len(df["id"].unique())
    n_years = len(df["year"].unique())
    n_age = len(df["age"].unique())

    interaction_tensor = np.full((n_players, n_years, n_age), np.nan)
    exposure_tensor = np.full((n_players, n_years, n_age), np.nan)
    df = df.sort_values(by=["id","year"])
    df["ft_pct"] = df["ftm"] / df["fta"]
    df["three_pct"] = df["fg3m"] / df["fg3a"]
    df["two_pct"] = df["fg2m"] / df["fg2a"]

    df["id"] = pd.Categorical(df.id).codes.astype(int)
    df["year"] = pd.Categorical(df.year).codes.astype(int)
    df["age"] = pd.Categorical(df.age).codes.astype(int) 

    metric_df = df[[output_metric, "id", "age", "year"]]
    exposure_df = df[["id", "age", "year", exposure]]

    for _, row in metric_df.iterrows():

        interaction_tensor[int(row.id), int(row.year), int(row.age)] = row[output_metric]
    for _, row in exposure_df.iterrows():
        exposure_tensor[int(row.id), int(row.year), int(row.age)] = row[exposure]
    if model == "poisson":
        offset = jnp.log(exposure_tensor)
        return offset, jnp.array(interaction_tensor)
    elif model == "binomial":
        trials = jnp.array(exposure_tensor)
        return trials, jnp.array(interaction_tensor)
    elif model == "gaussian":
        variance_scale = jnp.sqrt(exposure_tensor)
        return variance_scale, jnp.array(interaction_tensor)
    return ValueError


def create_pca_data(df, metric_output, exposure_list, metrics):
    """
    metric_output: list of [poisson, gaussian, binomial]
    exposure_list: list indicating which column to use as an exposure
    metrics: list of metric names 
    """
    exposures = []
    masks = []
    data = []
    outputs = []
    for output, metric, exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _ = process_data(df, metric, exposure_val, output, [])
        data.append(Y)
        masks.append(jnp.isfinite(exposure))
        exposures.append(exposure)
        if output == "gaussian":
            outputs.append(jnp.ones_like(Y, dtype=int))
        elif output == "poisson":
            outputs.append(2 * jnp.ones_like(Y, dtype=int))
        elif output == "binomial":
            outputs.append(3 * jnp.ones_like(Y, dtype=int))

    return jnp.hstack(exposures), jnp.hstack(masks), jnp.hstack(data), jnp.hstack(outputs)

def create_cp_data(df, metric_output, exposure_list, metrics):
    """
    metric_output: list of [poisson, gaussian, binomial]
    exposure_list: list indicating which column to use as an exposure
    metrics: list of metric names 
    """
    exposures = []
    masks = []
    data = []
    outputs = []
    for output, metric, exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _ = process_data(df, metric, exposure_val, output, [])
        data.append(Y)
        masks.append(jnp.isfinite(exposure))
        exposures.append(exposure)
        if output == "gaussian":
            outputs.append(jnp.ones_like(Y, dtype=int))
        elif output == "poisson":
            outputs.append(2 * jnp.ones_like(Y, dtype=int))
        elif output == "binomial":
            outputs.append(3 * jnp.ones_like(Y, dtype=int))

    return jnp.stack(exposures, axis = -1), jnp.stack(masks, axis = -1), jnp.stack(data, axis = -1), jnp.stack(outputs, axis = -1)

def create_cp_data_time(df, metric_output, exposure_list, metrics):
    """
    metric_output: list of [poisson, gaussian, binomial]
    exposure_list: list indicating which column to use as an exposure
    metrics: list of metric names 
    """
    exposures = []
    times = []
    masks = []
    data = []
    outputs = []
    for output, metric, exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, time = process_data_time(df, metric, exposure_val, output)
        data.append(Y)
        times.append(time)
        masks.append(jnp.isfinite(exposure))
        exposures.append(exposure)
        if output == "gaussian":
            outputs.append(jnp.ones_like(Y, dtype=int))
        elif output == "poisson":
            outputs.append(2 * jnp.ones_like(Y, dtype=int))
        elif output == "binomial":
            outputs.append(3 * jnp.ones_like(Y, dtype=int))

    return jnp.stack(exposures, axis = -1), jnp.stack(masks, axis = -1), jnp.stack(data, axis = -1), jnp.stack(outputs, axis = -1), jnp.stack(times, axis = -1)

def create_cp_data_multi_way(df, metric_output, exposure_list, metrics):
    """
    metric_output: list of [poisson, gaussian, binomial]
    exposure_list: list indicating which column to use as an exposure
    metrics: list of metric names 
    """
    exposures = []
    masks = []
    data = []
    outputs = []
    for output, metric, exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y  = process_data_multi_way(df, metric, exposure_val, output)
        data.append(Y)
        masks.append(jnp.isfinite(exposure))
        exposures.append(exposure)
        if output == "gaussian":
            outputs.append(jnp.ones_like(Y, dtype=int))
        elif output == "poisson":
            outputs.append(2 * jnp.ones_like(Y, dtype=int))
        elif output == "binomial":
            outputs.append(3 * jnp.ones_like(Y, dtype=int))
    return jnp.stack(exposures, axis = -1), jnp.stack(masks, axis = -1), jnp.stack(data, axis = -1), jnp.stack(outputs, axis = -1)


def create_basis(data, dims):
    agg_dict = {"obpm":"mean", "dbpm":"mean", "bpm":"mean", 
            "minutes":"sum", "dreb": "sum", "fta":"sum", "ftm":"sum", "oreb":"sum",
            "ast":"sum", "tov":"sum", "fg2m":"sum", "fg3m":"sum", "fg3a":"sum", "fg2a":"sum", "blk":"sum", "stl":"sum"}
    data["total_minutes"] = data["median_minutes_per_game"] * data["games"] 
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


def create_fda_data(data, basis_dims, metric_output, metrics, exposure_list):
    covariate_X = create_basis(data, basis_dims)
    data_set = []
    for output,metric,exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _ = process_data(data, metric, exposure_val, output, ["position_group"])
        data_dict = {"metric":metric, "output": output, "exposure_data": exposure, "output_data": Y, "mask": jnp.isfinite(exposure)}
        data_set.append(data_dict)

    basis = jnp.arange(18,39)

    return covariate_X, data_set, basis

def create_fda_data_time(data, basis_dims, metric_output, metrics, exposure_list):
    covariate_X = create_basis(data, basis_dims)
    data_set = []
    time_basis = jnp.arange(data["year"].min(), data["year"].max() + 1)
    for output,metric,exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, time = process_data_time(data, metric, exposure_val, output)
        data_dict = {"metric":metric, "output": output, "exposure_data": exposure, "output_data": Y, "mask": jnp.isfinite(exposure), "time": time}
        data_set.append(data_dict)

    basis = jnp.arange(18,39)
    return covariate_X, data_set, basis, time_basis
