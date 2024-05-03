import jax.numpy as jnp
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

def create_pca_matrix(df, metric_output, exposure_list, metrics):
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
            outputs.append(jnp.ones_like(Y))
        elif output == "poisson":
            outputs.append(2 * jnp.ones_like(Y))
        elif output == "binomial":
            outputs.append(3 * jnp.ones_like(Y))

    return jnp.hstack(exposures), jnp.hstack(masks), jnp.hstack(data), jnp.hstack(outputs)

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


