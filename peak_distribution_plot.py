import pandas as pd
import ridgeplot as rp
import plotly.express as px
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorly.decomposition import tucker
import jax 
import jax.numpy as jnp
from jax import vmap
from scipy.special import expit
from model.hsgp import make_convex_f, diag_spectral_density, make_convex_phi, make_psi_gamma
from model.inference_utils import create_metric_trajectory_all



def transform_mu(mu, metric_outputs):
    for index, output_type in enumerate(metric_outputs):
        if output_type == "gaussian":
            transform_function = lambda x: x 
        elif output_type == "poisson":
            transform_function = lambda x: np.exp(x)
        elif output_type == "binomial":
            transform_function = lambda x: expit(x)
        mu = mu.at[index].set(transform_function(mu[index]))
    return mu

def make_mu_mcmc(X, ls_deriv, alpha_time, weights, W, slope_weights, intercept_weights, L_time, M_time, phi_time, shifted_x_time):
    spd = jax.vmap(jax.vmap(lambda a, ls: jnp.sqrt(diag_spectral_density(1, a, ls, L_time, M_time))))(alpha_time, ls_deriv)
    weights = weights * spd[:, :, None, ...] * .0001
    wTx = jnp.einsum("nr,...mr -> ...nm", X, W)
    psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], -1) * (1/ jnp.sqrt(50))
    slope = jnp.einsum("ijnm, ijm... -> ijn...", psi_x,slope_weights)
    intercept = jnp.einsum("ijnm, ijm... -> ijn...", psi_x, intercept_weights)
    gamma_phi_gamma_x = jnp.einsum("...nm, ...mdk, tdz, ...jzk, ...nj -> ...nkt", psi_x, weights, phi_time, weights, psi_x)
    mu = intercept + jnp.einsum("...nk, t -> ...nkt", slope, shifted_x_time) - gamma_phi_gamma_x
    return jnp.swapaxes(mu, 2,3)

def make_mu(X, ls_deriv, alpha_time, shifted_x_time, weights, W, slope_weights, intercept_weights, L_time, M_time, phi_time):
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    weights = weights * spd * .0001
    wTx = jnp.einsum("nr,mr -> nm", X, W)
    psi_x = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(50))
    slope = make_psi_gamma(psi_x, slope_weights)
    intercept = make_psi_gamma(psi_x, intercept_weights)
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
    mu = make_convex_f(gamma_phi_gamma_x, x_time + L_time, slope, intercept)
    return mu





### set up 
from data.data_utils import create_fda_data
data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
names = data.groupby("id")["name"].first().values
names_df = pd.DataFrame(names, columns = ["Name"])
names_df["Player"] = range(len(names))
metric_output = ["binomial", "poisson"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
metric_df = pd.DataFrame(metrics, columns=["Statistic"])
metric_df["Metric"] = range(len(metrics))
basis = np.arange(18, 39)
age_df = pd.DataFrame(range(18,39), columns = ["Age"])
age_df["Time"] = age_df["Age"] - 18
exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
data["retirement"] = 1
print(f"mean age of retirement: {data.groupby('id')['age'].max().median()}")
data["log_min"] = np.log(data["minutes"])
data["simple_exposure"] = 1 
_ , outputs, _ = create_fda_data(data, basis_dims=3, metric_output=metric_output, 
                                     metrics = metrics
, exposure_list =  exposure_list)
observations = np.stack([output["output_data"] for output in outputs], axis = 1)
exposures = np.stack([output["exposure_data"] for output in outputs], axis = 0)
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

print("setup data")



with open("model_output/latent_variable.pkl", "rb") as f:
    results = pickle.load(f)
f.close()

X = results["X"]


with open("model_output/fixed_latent_convex_nba_tvrflvm_mcmc_poisson_minutes.pkl", "rb") as f:
    results_tvrflvm = pickle.load(f)
f.close()


with open("model_output/fixed_latent_convex_nba_tvrflvm_svi_poisson_minutes.pkl", "rb") as f:
    results_tvrflvm_svi = pickle.load(f)
f.close()

print("loaded all samples")

hsgp_params = {}
x_time = basis - basis.mean()
L_time = 1.5 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
M_time = 15
phi_time = make_convex_phi(x_time, L_time, M_time)
hsgp_params["phi_x_time"] = phi_time
hsgp_params["M_time"] = M_time
hsgp_params["L_time"] = L_time
hsgp_params["shifted_x_time"] = x_time + L_time

### thin for every 8th sample (so 4 chains  x 250 samples)
# mu_mcmc = make_mu_mcmc(X, results_tvrflvm["lengthscale_deriv"][:, ::8, ...], 
# results_tvrflvm["alpha"][:, ::8, ...],
# results_tvrflvm["beta"][:, ::8, ...], results_tvrflvm["W"][:, ::8, ...], 
# results_tvrflvm["slope"][:, ::8, ...], results_tvrflvm["intercept"][:, ::8, ...], 
# L_time, M_time, phi_time, hsgp_params["shifted_x_time"])

mu_mcmc = make_mu_mcmc(X, results_tvrflvm["lengthscale_deriv"], 
results_tvrflvm["alpha"],
results_tvrflvm["beta"], results_tvrflvm["W"], 
results_tvrflvm["slope"], results_tvrflvm["intercept"], 
L_time, M_time, phi_time, hsgp_params["shifted_x_time"])

# posterior_mcmc = create_metric_trajectory_all(mu_mcmc, observations, exposures, metric_output, metrics, exposure_list, jnp.transpose(results_tvrflvm["sigma"], (2, 0, 1)))
# print(posterior_mcmc.shape)
# mu_svi = make_mu(X, results_tvrflvm_svi["lengthscale_deriv__loc"], 
# results_tvrflvm_svi["alpha__loc"], hsgp_params["shifted_x_time"],
# results_tvrflvm_svi["beta__loc"], results_tvrflvm_svi["W__loc"], 
# results_tvrflvm_svi["slope__loc"], results_tvrflvm_svi["intercept__loc"], 
# L_time, M_time, phi_time)

print("produced mu samples")


peaks = jnp.argmax(mu_mcmc, -1) + 18
decay =  (jnp.take_along_axis(mu_mcmc, jnp.minimum(peaks - 18 + 3,20)[..., None], axis = -1).squeeze() - jnp.max(mu_mcmc, -1)) / jnp.minimum(3, (39 - peaks))
# peaks = jnp.argmax(transform_mu(mu_svi,metric_output), -1) + 18
print(f"calculated the peak of samples resulting in size {peaks.shape}")
positions = ["G", "F", "C"]
# positions = ["G"]
pos_indices = data.drop_duplicates(subset=["id","position_group","name"]).reset_index()
position_samples_list = []
position_samples_decay_list = []
for pos in positions:
    player_indices = pos_indices[pos_indices["position_group"] == pos].index.values
    player_samples = jnp.vstack(peaks[..., player_indices].mean(-1))
    player_samples_decay = jnp.vstack(decay[..., player_indices].mean(-1))
    print(f"indexed position: {pos} and reshaped to size {player_samples.shape}")
    pos_samples_decay_df = pd.DataFrame(player_samples_decay, columns= metrics).melt(value_name = "decay", var_name="metric")
    pos_samples_decay_df["position"] = pos
    pos_samples_df = pd.DataFrame(player_samples, columns=metrics).melt(value_name="peak", var_name="metric")
    pos_samples_df["position"] = pos
    position_samples_list.append(pos_samples_df)
    position_samples_decay_list.append(pos_samples_decay_df)


position_samples_df = pd.concat(position_samples_list)
position_samples_decay_df = pd.concat(position_samples_decay_list)



samples_ridgeplot = [
    [
        position_samples_df[(position_samples_df["position"] == pos) & (position_samples_df["metric"] == metric)]["peak"].to_numpy()
        for pos in positions   
    ]
    for metric in metrics
]

samples_ridgeplot_decay = [
    [
        position_samples_decay_df[(position_samples_decay_df["position"] == pos) & (position_samples_decay_df["metric"] == metric)]["decay"].to_numpy()
        for pos in positions   
    ]
    for metric in metrics
]

print("setup samples for plotting")
   

fig = rp.ridgeplot(
    samples=samples_ridgeplot,
    labels=metrics,
    colorscale= ["deepskyblue", "orangered", "green"],
    colormode="trace-index-row-wise",
    spacing=.5,
    norm = "probability",
    
    )

fig.update_layout(
title="Distribution of Peak Performance by Position",
height=650,
width=950,
font_size=14,
plot_bgcolor="rgb(245, 245, 245)",
xaxis_gridcolor="white",
yaxis_gridcolor="white",
xaxis_gridwidth=2,
yaxis_title="Metric",
xaxis_title="Peak Age",
showlegend=False,
    )


fig.write_image("model_output/model_plots/debug_peak_position_full_poisson_minutes.png", format = "png")



fig = rp.ridgeplot(
    samples=samples_ridgeplot_decay,
    labels=metrics,
    colorscale= ["deepskyblue", "orangered", "green"],
    colormode="trace-index-row-wise",
    spacing=.5,
    norm = "probability",
    
    )

fig.update_layout(
title="Distribution of Decay after Peak Performance by Position",
height=650,
width=950,
font_size=14,
plot_bgcolor="rgb(245, 245, 245)",
xaxis_gridcolor="white",
yaxis_gridcolor="white",
xaxis_gridwidth=2,
yaxis_title="Metric",
xaxis_title="Decay Rate",
showlegend=False,
    )


fig.write_image("model_output/model_plots/debug_decay_position_full_poisson_minutes.png", format = "png")
print("finished plotting the samples")


