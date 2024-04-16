import pandas as pd
import jax.numpy as jnp
import jax
import numpyro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_utils import process_data
from model.models import NBAFDAModel



if __name__ == "__main__":
    numpyro.set_platform("METAL")
    print(jax.local_device_count())
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")

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
    covariate_X = jnp.array(MinMaxScaler().fit_transform(X_pca[:, 0:3]))
    metric_output = (["gaussian"] * 2) + (["poisson"] * 6) + (["binomial"] * 3)
    metrics = ["obpm","dbpm","blk","stl","ast","dreb","oreb","tov","ftm","fg2m","fg3m"]
    exposure_list = (["median_minutes_per_game"] * 8) + ["fta","fg2a","fg3a"]
    covariate_size = covariate_X.shape[1]
    data_set = []

    for output,metric,exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _ = process_data(data, metric, exposure_val, output, ["position_group"])
        data_dict = {"metric":metric, "output": output, "exposure_data": exposure, "output_data": Y, "mask": jnp.isfinite(exposure)}
        data_set.append(data_dict)

    basis = jnp.arange(18,39)

    model = NBAFDAModel(basis, output_size=len(metric_output), M=10)
    model.initialize_priors()
    mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"covariate_X": covariate_X, "data_set": data_set})


    