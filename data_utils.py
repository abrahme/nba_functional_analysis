import jax.numpy as jnp

def process_data(df, output_metric, exposure, model, input_metrics):

    agg_dict = {input_metric:"max" for input_metric in input_metrics}
    df = df.sort_values(by=["id","year"])
    df["ft_pct"] = df["ftm"] / df["fta"]
    df["three_pct"] = df["fg3m"] / df["fg3a"]
    df["two_pct"] = df["fg2m"] / df["fg2a"]
    X = df[input_metrics + ["id"]].groupby("id").agg(agg_dict).reset_index()[input_metrics]
    metric_df = df[[output_metric, "id", "age"]]
    exposure_df = df[["id", "age", exposure]]
    games_df = df[["id", "age", "games"]]
    metric_df  = metric_df.pivot(columns="age",values=output_metric,index="id")
    if model == "poisson":
        offset = jnp.log(exposure_df.pivot(columns="age", values=exposure,index="id").to_numpy()) + jnp.log(games_df.pivot(columns="age", values = "games", index = "id").to_numpy())
        return offset, jnp.array(metric_df.to_numpy()), X
    elif model == "binomial":
        trials = jnp.array(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return trials, jnp.array(metric_df.to_numpy()), X
    elif model == "gaussian":
        variance_scale = jnp.sqrt(exposure_df.pivot(columns="age", index="id", values=exposure).to_numpy())
        return variance_scale, jnp.array(metric_df.to_numpy()), X
    return ValueError



