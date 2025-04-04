import pandas as pd
import ridgeplot as rp
import plotly.express as px
import pickle
import plotly.graph_objects as go
import numpy as np
import arviz as az
import jax 
import xarray as xr
import jax.numpy as jnp
import jax.scipy.special as jsci
from model.hsgp import make_convex_f, diag_spectral_density, make_convex_phi, make_psi_gamma
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from visualization.visualization import plot_posterior_predictive_career_trajectory
from model.inference_utils import create_metric_trajectory_all


def transform_mu(mu, metric_outputs):
    transformed_mu = np.zeros_like(mu)
    for index, output_type in enumerate(metric_outputs):
        if output_type == "gaussian":
            transform_function = lambda x: x 
        elif output_type == "poisson":
            transform_function = lambda x: jnp.exp(x) 
        elif output_type == "binomial":
            transform_function = lambda x: jsci.expit(x)
        transformed_mu[:,:,index,...] = transform_function(mu[:,:,index,...])

    return transformed_mu

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

def make_mu(X, ls_deriv, alpha_time, weights, W, slope_weights, intercept_weights, L_time, M_time, phi_time):
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

positions = ["G", "F", "C"]


hsgp_params = {}
x_time = basis - basis.mean()
L_time = 1.5 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
M_time = 15
phi_time = make_convex_phi(x_time, L_time, M_time)
hsgp_params["phi_x_time"] = phi_time
hsgp_params["M_time"] = M_time
hsgp_params["L_time"] = L_time
hsgp_params["shifted_x_time"] = x_time + L_time


with open(f"model_output/latent_variable.pkl", "rb") as f:
    results = pickle.load(f)
f.close()

with open(f"model_output/fixed_latent_convex_nba_tvrflvm_mcmc_poisson_minutes.pkl", "rb") as f:
    results_tvrflvm = pickle.load(f)
f.close()

mu_mcmc = make_mu_mcmc(results["X"], results_tvrflvm["lengthscale_deriv"], 
    results_tvrflvm["alpha"],
    results_tvrflvm["beta"], results_tvrflvm["W"], 
    results_tvrflvm["slope"], results_tvrflvm["intercept"], 
    L_time, M_time, phi_time, hsgp_params["shifted_x_time"])

transformed_mu_mcmc = transform_mu(mu_mcmc, metric_output)
peaks = jnp.argmax(transformed_mu_mcmc, -1) + 18
peak_val = jnp.take_along_axis(transformed_mu_mcmc, (peaks - 18)[..., None], axis = -1).squeeze()
decay = (jnp.take_along_axis(transformed_mu_mcmc, jnp.minimum(peaks - 18 + 3,20)[..., None], axis = -1).squeeze() - peak_val) / peak_val

decay = decay.at[jnp.isinf(decay)].set(jnp.nan)



pos_indices = data.drop_duplicates(subset=["id","position_group","name"]).reset_index()


player_indices = pos_indices.index.values
player_samples = jnp.vstack(peaks[..., player_indices].mean(-1))
player_samples_decay = jnp.vstack(jnp.nanmean(decay[..., player_indices],-1))
player_samples_decay_kde = np.zeros((100, len(metrics)))
player_samples_decay_x = np.zeros((100, len(metrics)))
metric_kde_list = []
position_samples_list =  []
position_samples_decay_list = []
position_samples_decay_kde_list = []
for index, metric in enumerate(metrics):
    metric_kde = gaussian_kde(player_samples_decay[..., index])
    player_samples_decay_kde[..., index] = metric_kde.evaluate(jnp.linspace(
    player_samples_decay[..., index].min(), player_samples_decay[..., index].max(), 100))
    player_samples_decay_x[..., index] = jnp.linspace(
    player_samples_decay[..., index].min(), player_samples_decay[..., index].max(), 100)
    player_samples_decay_kde[..., index] /= player_samples_decay_kde[..., index].sum()
    metric_kde_df = pd.DataFrame(jnp.stack([player_samples_decay_x[..., index],player_samples_decay_kde[..., index]]).T, columns = ["x","density"])
    metric_kde_df["metric"] = metric
    metric_kde_list.append(metric_kde_df)
    pos_samples_decay_df = pd.DataFrame(player_samples_decay, columns= metrics).melt(value_name = "decay", var_name="metric")
    pos_samples_decay_kde_df = pd.concat(metric_kde_list)

    pos_samples_df = pd.DataFrame(player_samples, columns=metrics).melt(value_name="peak", var_name="metric")
    position_samples_list.append(pos_samples_df)
    position_samples_decay_list.append(pos_samples_decay_df)
    position_samples_decay_kde_list.append(pos_samples_decay_kde_df)

position_samples_df = pd.concat(position_samples_list)
position_samples_decay_df = pd.concat(position_samples_decay_list)
position_samples_decay_kde_df = pd.concat(position_samples_decay_kde_list)

labels_sorted = position_samples_df.groupby("metric")["peak"].mean().reset_index().sort_values(by = "peak")["metric"]

samples_ridgeplot = [
        position_samples_df[(position_samples_df["metric"] == metric)]["peak"].to_numpy()
    for metric in labels_sorted
]

samples_ridgeplot_decay = [
        position_samples_decay_df[ (position_samples_decay_df["metric"] == metric)]["decay"].to_numpy()
    for metric in labels_sorted
]

print("setup samples for plotting")
   

fig = rp.ridgeplot(
    samples=samples_ridgeplot,
    labels=labels_sorted,
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



fig.write_image("model_output/model_plots/debug_peak_full_poisson_minutes.png", format = "png")



fig = make_subplots(rows = 4, cols=4,  subplot_titles=metrics)

for index, metric in enumerate(labels_sorted):
    row = int(np.floor(index / 4)) + 1 
    col = (index % 4) + 1
    metric_type = metric_output[index]
    pos_metric_df = position_samples_decay_kde_df[(position_samples_decay_kde_df["metric"] == metric)]
    x = pos_metric_df["x"].to_numpy()
    y = pos_metric_df["density"].to_numpy()
    fig.add_trace(go.Scatter(x = np.append(x, None), y = np.append(y, None), mode = "lines", showlegend=False), row = row, col=col)
        
fig.update_xaxes(tickangle=90)
fig.update_layout({'width':650, 'height': 650,
                            'showlegend':False, 'hovermode': 'closest',
                            })
fig.write_image("model_output/model_plots/debug_decay_full_poisson_minutes.png", format = "png")
print("finished plotting the samples")



player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                         "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                         "Chris Paul", "Shaquille O'Neal"]


X = results["X"]
tsne = TSNE(n_components=2)
X_tsne_df = pd.DataFrame(tsne.fit_transform(X), columns = ["Dim. 1", "Dim. 2"])
id_df = data[["position_group","name","id", "minutes"]].groupby("id").max().reset_index()
X_tsne_df = pd.concat([X_tsne_df, id_df], axis = 1)
X_tsne_df["name"] = X_tsne_df["name"].apply(lambda x: x if x in player_labels else "")
X_tsne_df["minutes"] /= np.max(X_tsne_df["minutes"])
X_tsne_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
fig = px.scatter(X_tsne_df, x = "Dim. 1", y = "Dim. 2", color = "Position", text="name", size = "minutes",
                    opacity = .1, title="T-SNE Visualization of Latent Player Embedding", )
fig.write_image(f"model_output/model_plots/exponential_cp_latent_space.png", format = "png")




for player in player_labels:
    index_player = id_df[id_df["name"] == player].index.values[0]
    fig = plot_posterior_predictive_career_trajectory(index_player, metrics, metric_output , posterior_mean_samples=mu_mcmc[..., index_player, :], observations=observations,
                                                      exposures= exposures, 
                                                      posterior_variance_samples=jnp.transpose(results_tvrflvm["sigma"], (2,0,1)),
                                                      exposure_names= exposure_list)
    fig.write_image(f"model_output/model_plots/player_plots/original_{player}.png", format = "png")

obs, pos = create_metric_trajectory_all(mu_mcmc, observations, exposures, 
                                        metric_output, metrics, exposure_list, 
                                        jnp.transpose(results_tvrflvm["sigma"], (2,0,1)))

hdi = az.hdi(np.array(pos), hdi_prob = .95)

hdi_low = hdi[..., 0]
hdi_high = hdi[..., 1]
avg_coverages = ((obs <= hdi_high) & (obs >= hdi_low)).sum((0,1)) / (~np.isnan(obs)).sum((0, 1))
coverage_df = pd.DataFrame(avg_coverages, columns=["coverage"])
coverage_df["metric"] = metrics
fig = px.bar(coverage_df.sort_values(by = "coverage"), x='metric', y='coverage')
fig.write_image(f"model_output/model_plots/original_coverage.png")