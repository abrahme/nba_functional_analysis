import pickle
import jax.numpy as jnp
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import arviz as az
import jax
from shinywidgets import render_plotly
from shiny.express import input, ui, render
from visualization.visualization import plot_correlation_dendrogram, plot_mcmc_diagnostics, plot_posterior_predictive_career_trajectory, plot_scatter
from data.data_utils import create_fda_data
from model.hsgp import make_convex_phi,  diag_spectral_density
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")



#### LOAD EVERYTHING 
data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
names = data.groupby("id")["name"].first().values

metric_output = ["binomial", "exponential"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
basis = np.arange(18,39)
x = basis - basis.mean()
L = 1.3 * np.max(np.abs(x))


metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]

data["retirement"] = 1
data["log_min"] = np.log(data["minutes"])
data["simple_exposure"] = 1 
_ , outputs, _ = create_fda_data(data, basis_dims=3, metric_output=metric_output, 
                                     metrics = metrics
, exposure_list =  exposure_list)

with open("model_output/latent_variable.pkl", "rb") as f:
    results_latent = pickle.load(f)
f.close()

with open("model_output/fixed_latent_convex_nba_tvrflvm_svi.pkl", "rb") as f:
    results = pickle.load(f)
f.close()

inf_data = az.from_dict(results)


x_time = basis - basis.mean()
L_time = 1.5 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
shifted_x_time = x_time + L_time
M_time = 5
M = 50


X = results_latent["X"]
X = (X - X.mean(0))/(X.std(0))

W = results["W"][0:1]
weights = results["beta"][0:1]
alpha_time = results["alpha"][0:1]

length_time = results["lengthscale_deriv"][0:1][..., None]

intercept_samples = results["intercept"][0:1]
slope_samples = results["slope"][0:1]
spd = jnp.reshape(jnp.sqrt(diag_spectral_density(1, alpha_time, length_time, L_time, M_time)), (1, 2000, 1, M_time, len(metrics)))
weights = weights * spd * .0001
phi_time = make_convex_phi(x_time, L_time, M_time)
left_result = jnp.einsum("tmz, ijlmk -> ijtklz", phi_time, weights) ### (t x M_time x M_time) , (M * 2, M _time, k) -> (t, k, M * 2, M_time)
gamma_phi_gamma_time = jnp.einsum("iftkjz,iflzk -> iftklj", left_result, weights) ## -> (t, k , M *2, M*2)


# X_rflvm_aligned_mean = X[-1, -10, ...] ### TODO: align 
X_rflvm_aligned = X

X_tsne = TSNE(n_components=3).fit_transform(X_rflvm_aligned)
knn = NearestNeighbors(n_neighbors=6).fit(X_tsne)


parameters = list(results.keys()) + ["mu"]

agg_dict = {"obpm":"mean", "dbpm":"mean", "bpm":"mean", 
            "position_group": "max",
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
df = pd.concat([pd.DataFrame(X_tsne, columns=[f"dim{i+1}" for i in range(3)]), agged_data], axis = 1)
df["player_name"] = names


ui.page_opts(title="NBA Career Trajectories")

with ui.nav_panel("Player Embeddings & Trajectories"):
    with ui.layout_column_wrap():
        ui.input_select(id="player", label = "Select a player", choices = {index : name for index, name in enumerate(names)})
        @render.data_frame
        def produce_neighbors():
            distances, neighbors = knn.kneighbors(X_tsne[int(input.player())][None,:], return_distance=True)
            name_df = df.iloc[neighbors[0][1:]][["player_name"]]
            name_df["distances"] = distances[0,1:]
            return name_df
        
    with ui.layout_column_wrap():
        @render_plotly
        def plot_latent_space():
            return plot_scatter(df,  "Latent Embedding", int(input.player()) )


        
        @render_plotly
        def player_trajectory():
            player_index = int(input.player())
            wTx = jnp.einsum("r,...mr -> ...m", X[player_index], W)
            psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)],-1) * (1/ jnp.sqrt(M))
            phi_x = jnp.einsum("...m,...k -> ...mk", psi_x, psi_x)
            slope = jnp.einsum("...m,...mk -> ...k",psi_x, slope_samples)
            intercept = jnp.einsum("...m, ...mki -> ...ki", psi_x, intercept_samples)
            gamma_phi_gamma_x = jnp.einsum("ij...,ijtk... -> ijkt", phi_x, gamma_phi_gamma_time)
            mu = intercept + jnp.einsum("...k, t -> ...kt", slope, shifted_x_time) - gamma_phi_gamma_x
            return plot_posterior_predictive_career_trajectory(
                                                            player_index=player_index,
                                                            metrics=metrics, 
                                                            metric_outputs=metric_output,
                                                            posterior_mean_samples=mu,
                                                            observations=jnp.stack([output["output_data"] for output in outputs], axis = 1), 
                                                            exposures = jnp.stack([outputs[i]["exposure_data"] for i in range(len(metrics))],axis=0),
                                                            posterior_variance_samples=jnp.moveaxis(results["sigma"][0:1],-1, 0))



# with ui.nav_panel("MCMC Diagnostics"):
#     ui.input_select(id="model", label = "Select a model parameter", choices={index : name for index, name in enumerate(parameters) if ("sigma" in name) or (name == "lengthscale") or ("f" in name)})

#     with ui.layout_column_wrap():
#         @render.plot
#         def plot_trace():
#             param_name = parameters[int(input.model())]
#             return plot_mcmc_diagnostics(inf_data, param_name, plot="trace" )
#         @render.table
#         def plot_summary():
#             param_name = parameters[int(input.model())]
#             return plot_mcmc_diagnostics(inf_data, param_name, plot="summary" )
    
#     ui.input_select(id="player_model", label = "Select a player", choices={index : name for index, name in enumerate(names)})
 
#     with ui.layout_column_wrap():
#         ui.input_select(id="metric_mcmc", label = "Select a metric", choices = {index : name for index, name in enumerate(metrics)})
#         ui.input_select(id="age", label = "Select an age", choices = {index : name for index, name in enumerate(range(18,39))})
#     with ui.layout_column_wrap():
#         @render.plot
#         def plot_player_trace_mu():
#             player_index = int(input.player_model())
#             wTx = np.einsum("r,...mr -> ...m", X_rflvm_aligned[:,:,player_index,:], W)
#             phi = np.concatenate([np.cos(wTx), np.sin(wTx)], -1) * (1/ np.sqrt(100))
#             mu = np.einsum("k,ijk -> ij", phi, convex_beta[:, :, int(input.metric_mcmc()), :, int(input.age())])
#             results["mu"] = mu
#             inf_data = az.from_dict(results)
#             return plot_mcmc_diagnostics(inf_data, "mu", plot = "trace")
        
#         @render.table
#         def plot_player_summary_mu():
#             player_index = int(input.player_model())
#             wTx = np.einsum("r,...mr -> ...m", X_rflvm_aligned[:,:,player_index,:], W)
#             phi = np.concatenate([np.cos(wTx), np.sin(wTx)], -1) * (1/ np.sqrt(100))
#             mu = np.einsum("k,ijk -> ij", phi, convex_beta[:, :, int(input.metric_mcmc()), :, int(input.age())])
#             results["mu"] = mu
#             inf_data = az.from_dict(results)
#             return plot_mcmc_diagnostics(inf_data, "mu", plot = "summary")

# with ui.nav_panel("Metric Correlations"):
#     ui.input_select(id="chain", label = "Select a chain", choices = {index : name for index, name in enumerate(range(1,3))})
#     @render_plotly
#     def plot_latent_space_dendrogram():
#         return plot_correlation_dendrogram(convex_beta[int(input.chain())-1, :, :, :, :].mean(axis = (0,3)), labels = metrics, title = "Metric Correlation")

