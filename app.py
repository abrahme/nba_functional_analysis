import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
from shiny import reactive, render, req, App
from shinywidgets import render_plotly
from shiny.express import input, ui
from visualization.visualization import plot_correlation_dendrogram, plot_mcmc_diagnostics, plot_posterior_predictive_career_trajectory, plot_scatter
from data.data_utils import create_fda_data


#### LOAD EVERYTHING 
data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
names = data.groupby("id")["name"].first().values

metric_output = (["gaussian"] * 3) + (["poisson"] * 9) + (["binomial"] * 3)

metrics = ["log_min", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
exposure_list = ["simple_exposure"] + (["minutes"] * 11) + ["fta","fg2a","fg3a"]


data["log_min"] = np.log(data["minutes"])
data["simple_exposure"] = 1 
_ , outputs, _ = create_fda_data(data, basis_dims=3, metric_output=metric_output, 
                                     metrics = metrics
, exposure_list =  exposure_list)

with open("model_output/fixed_nba_rflvm.pkl", "rb") as f:
    results = pickle.load(f)
f.close()

W = results["W"]
with open("model_output/exponential_cp.pkl", "rb") as f:
    results_embedding = pickle.load(f)
f.close()

X = results_embedding["U_auto_loc"]
U, _, _ = np.linalg.svd(X, full_matrices=False)
L       = np.linalg.cholesky(np.cov(U.T) + 1e-6 * np.eye(3)).T
aligned_X  = np.linalg.solve(L, U.T).T
X_rflvm_aligned = aligned_X / np.std(X, axis=0)




parameters = results.keys()

agg_dict = {"obpm":"mean", "dbpm":"mean", "bpm":"mean", 
        "minutes":"sum", "dreb": "sum", "fta":"sum", "ftm":"sum", "oreb":"sum",
        "ast":"sum", "tov":"sum", "fg2m":"sum", "fg3m":"sum", "fg3a":"sum", "fg2a":"sum", "blk":"sum", "stl":"sum"}
data["total_minutes"] = data["median_minutes_per_game"] * data["games"] 
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
df = pd.concat([pd.DataFrame(X_rflvm_aligned, columns=[f"dim{i+1}" for i in range(3)]), agged_data], axis = 1)
df["player_name"] = names




agged_metrics = ["obpm","minutes","dbpm"] + [col for col in df.columns if ("rate" in col) or ("pct" in col)]

ui.page_opts(title="NBA Career Trajectories")

with ui.nav_panel("Player Embeddings & Trajectories"):
    with ui.layout_column_wrap():
    
        ui.input_select(id="metric", label = "Select a metric", choices = {index : name for index, name in enumerate(agged_metrics)})
        ui.input_select(id="player", label = "Select a player", choices = {index : name for index, name in enumerate(names)})
    with ui.layout_column_wrap():
        @render_plotly
        def plot_latent_space():
            return plot_scatter(df, agged_metrics[int(input.metric())], "Latent Embedding", int(input.player()) )

        
        @render_plotly
        def player_trajectory():
            player_index = int(input.player())
            player_name = names[player_index]
            wTx = np.einsum("r,ijmr -> ijm", X_rflvm_aligned[player_index], W)
            phi = np.concatenate([np.cos(wTx), np.sin(wTx)], -1) * (1/ np.sqrt(100))
            mu = np.einsum("ijk,ijmkt -> ijmt", phi, results["beta"])
            return plot_posterior_predictive_career_trajectory(player_name=player_name,
                                                            player_index=player_index,
                                                            metrics=metrics, 
                                                            metric_outputs=metric_output,
                                                            posterior_mean_samples=mu,
                                                            observations=jnp.stack([output["output_data"] for output in outputs], axis = 1), 
                                                            exposures = jnp.stack([outputs[i]["exposure_data"] for i in range(len(metrics))],axis=0),
                                                            posterior_variance_samples=jnp.stack([results["sigma_log_min"], results["sigma_obpm"], results["sigma_dbpm"]], axis = 0))




# with ui.nav_panel("Player Correlations"):
#     @render_plotly
#     def plot_latent_space_dendrogram():
#         return plot_correlation_dendrogram(X_rflvm_aligned, labels = names, title = "Embedding Correlation")

