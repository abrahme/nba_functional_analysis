import pandas as pd
import ridgeplot as rp
import plotly.express as px
import pickle
import plotly.graph_objects as go
import numpy as np
import argparse
import arviz as az
from jax import config
config.update("jax_enable_x64", True)
import jax 
from data.data_utils import create_fda_data, average_peak_differences
import numpyro
import jax.numpy as jnp
import jax.scipy.special as jsci
from model.hsgp import  diag_spectral_density, make_psi_gamma, make_convex_phi, make_convex_phi_prime, vmap_make_convex_phi, vmap_make_convex_phi_prime
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from visualization.visualization import plot_posterior_predictive_career_trajectory
from model.inference_utils import create_metric_trajectory_all

@jax.jit
def contract_single_nkt_lazy(psi_n, psi_k, weights_mdk, weights_jzk,
                              ptm, pptm, phi_time_t, shifted, t_m):
    delta = (shifted - t_m)  # (d, 1)
    intermediate = ptm[None] - phi_time_t + jnp.einsum("dz ,t -> tdz",pptm , delta)  # (t, d, z)
    return jnp.einsum("m,md,tdz,jz,j->t", psi_n, weights_mdk, intermediate, weights_jzk, psi_k)

@jax.jit
def compute_nk_lazy(psi, w, ptm, pptm, phi_time, shifted, t_mx, n, k):
    psi_n = psi[n]                  # (M,)
                 # (J,) assumed same
    weights_mdk = w[:, :, k]        # (M, D)
        # (J, Z)

    ptm_nkt = ptm[n, k]             # (D, Z)
    pptm_nkt = pptm[n, k]           # (D, Z)
    t_m = t_mx[n, k]                # scalar

    return contract_single_nkt_lazy(
        psi_n, psi_n, weights_mdk, weights_mdk,
        ptm_nkt, pptm_nkt, phi_time, shifted, t_m
    )

@jax.jit
def compute_gamma_lazy_batched(psi_x, weights, phi_t_max, phi_prime_t_max,
                                phi_time, shifted_x_time, L_time, t_max):
    S, C, N, K = t_max.shape[:4]  # Sample, Chain, N, K
    shifted = shifted_x_time - L_time

    def process_single_sc(s, c):
        psi = psi_x[s, c]              # (N, M)
        w = weights[s, c]              # (M, D, K)
        ptm = phi_t_max[s, c]          # (N, K, D, Z)
        pptm = phi_prime_t_max[s, c]   # (N, K, D, Z)
        t_mx = t_max[s, c]             # (N, K)

        compute = lambda n, k: compute_nk_lazy(psi, w, ptm, pptm, phi_time, shifted, t_mx, n, k)
        return jax.vmap(jax.vmap(compute, in_axes=(None, 0)), in_axes=(0, None))(jnp.arange(N), jnp.arange(K))

    return jax.vmap(jax.vmap(process_single_sc, in_axes=(0, None)), in_axes=(None, 0))(jnp.arange(S), jnp.arange(C))









def transform_mu(mu, metric_outputs):
    transformed_mu = np.zeros_like(mu)
    for index, output_type in enumerate(metric_outputs):
        if output_type == "gaussian":
            transform_function = lambda x: x 
        elif output_type == "poisson":
            transform_function = lambda x: jnp.exp(x) 
        elif output_type == "binomial":
            transform_function = lambda x: jsci.expit(x)
        elif output_type == "beta":
            transform_function = lambda x: jsci.expit(x)
        transformed_mu[:,:,index,...] = transform_function(mu[:,:,index,...])

    return transformed_mu


def make_mu_mcmc(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict):
    # spd = jax.vmap(jax.vmap(lambda a, l: jnp.sqrt(diag_spectral_density(1, a, l, L_time, M_time))))(alpha_time, ls_deriv)
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    # weights = weights * spd[..., None, :, :]
    weights *= spd
    wTx = jnp.einsum("...nr, mr -> ...nm", X, W * jnp.sqrt(ls))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("...nm, mk -> ...nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 2  + offset_dict["t_max"]  
    c_max = (jnp.einsum("...nm, mk -> ...nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    # gamma_phi_gamma_x =  compute_gamma_lazy_batched(psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max)
    # mu = intercept + jnp.transpose(gamma_phi_gamma_x, (1,0, 3, 2, 4))
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = gamma_phi_gamma_x + intercept
    return wTx, mu, t_max, c_max

def make_mu(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, offset_dict):
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    weights = weights * spd 
    wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(ls))   
    psi_x = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offset_dict["t_max"]  
    c_max = make_psi_gamma(psi_x, c_max) * sigma_c_max + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    intercept = jnp.transpose(c_max)[..., None]
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    return mu, t_max, c_max





### set up 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--rff_dim", help="size of the rff approx", required=True, type=int)
    parser.add_argument("--mcmc_path", help="where to get mcmc from", required = False, default="")
    parser.add_argument("--svi_path", help = "where to get svi from", required=False, default="")
    parser.add_argument("--position_group", help = "which position group to run the model for", required = True, choices=["G", "F", "C", "all"])
    parser.add_argument("--player_names", help = "which players to run the model for", required=False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--thin", help = "keep every thin sample per chain", required=False, default=100, type = int)
    numpyro.set_platform("cpu")
    args = vars(parser.parse_args())
    mcmc_path = args["mcmc_path"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    rff_dim = args["rff_dim"]
    svi_path = args["svi_path"]
    position_group = args["position_group"]
    players = args["player_names"]
    thin = args["thin"]

    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    names = data.groupby("id")["name"].first().values.tolist()
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(82, data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games_exposure"]) / 48
    data["retirement"] = 1

 
    data["total_minutes"] = data["median_minutes_per_game"] * data["games"] 


    validation_data = pd.read_csv("data/validation_player_data.csv").query(" age <= 38 ")
    validation_data["log_min"] = np.log(validation_data["minutes"])
    validation_data["simple_exposure"] = 1
    validation_data["games_exposure"] = np.maximum(82, validation_data["games"]) ### 82 or whatever
    validation_data["pct_minutes"] = (validation_data["minutes"] / validation_data["games_exposure"]) / 48
    validation_data["retirement"] = 1
    # validation_data["total_minutes"] = validation_data["median_minutes_per_game"] * validation_data["games"] 
    validation_data["season"] = validation_data["year"].apply(lambda x: str(x - 1) + "-" + str(x)[-2:])
    validation_data["age"] = pd.Categorical(validation_data["age"])

    id_df = data[["position_group","name","id", "minutes"]].groupby("id").max().reset_index()
    validation_data = pd.merge(id_df[["position_group", "id", "name"]], validation_data, on = ["name", "position_group"])
    season_mask = season_array = validation_data[["id", "age", "season"]].pivot(columns="age",values="season",index="id").reindex(columns = range(18,39)).to_numpy()

    agg_dict = {"obpm":"mean", "dbpm":"mean", "bpm":"mean", 
            "position_group": "max",
        "minutes":"sum", "dreb": "sum", "fta":"sum", "ftm":"sum", "oreb":"sum",
        "ast":"sum", "tov":"sum", "fg2m":"sum", "fg3m":"sum", "fg3a":"sum", "fg2a":"sum", "blk":"sum", "stl":"sum"}
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


    metric_output = ["binomial", "beta"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["games", "pct_minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["games_exposure", "games_exposure"]) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    print("setup data")
    if players:
        player_indices = [names.index(item) for item in players]
    elif position_group in ["G","F","C"]:
        all_indices = data.drop_duplicates(subset=["position_group","name","id"]).reset_index()
        player_indices = all_indices[all_indices["position_group"] == position_group].index.values.tolist()
    else:
        player_indices = []
    positions = ["G", "F", "C"]
    covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
    _, validation_data_set, _ = create_fda_data(validation_data, basis_dims, metric_output, metrics, exposure_list, player_indices, validation_data=True)

    hsgp_params = {}
    x_time = basis - basis.mean()
    L_time = 2.0 * np.max(np.abs(x_time), 0, keepdims=True)
    M_time = 15 
    phi_time = vmap_make_convex_phi(np.squeeze(x_time), np.squeeze(L_time), M_time)
    hsgp_params["phi_x_time"] = phi_time
    hsgp_params["M_time"] = M_time
    hsgp_params["L_time"] = L_time
    hsgp_params["shifted_x_time"] = x_time + L_time

    masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
    exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
    Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])

    masks_val = jnp.stack([data_entity["mask"] for data_entity in validation_data_set])
    exposures_val = jnp.stack([data_entity["exposure_data"] for data_entity in validation_data_set])
    Y_val = jnp.stack([data_entity["output_data"] for data_entity in validation_data_set])



    offset_list = []
    offset_max_list = []
    offset_peak_list = []
    offset_boundary_l = []
    offset_boundary_r = []
    for index, family in enumerate(metric_output):  
        if family == "gaussian":
            offset_list.append(jnp.nanmean(Y[index]))
            offset_max_list.append(jnp.nanmean(jnp.nanmax(Y[index], -1)))
            peak = jnp.nanmean(jnp.nanargmax(Y[index], -1))
            boundary_l, boundary_r = average_peak_differences(Y[index])
            offset_boundary_l.append(boundary_l)
            offset_boundary_r.append(boundary_r)
        else:
            if family == "poisson":
                p = jnp.nansum(Y[index]) / jnp.nansum(np.exp(exposures[index]))
                p_max = jnp.nanmean(jnp.nanmax(Y[index] / np.exp(exposures[index]), -1))
                peak = jnp.nanmean(jnp.nanargmax(Y[index] / np.exp(exposures[index]), -1))
                offset_list.append(np.log(p))
                offset_max_list.append(np.log(p_max))                 
            elif family == "binomial":
                p = jnp.nansum(Y[index]) / jnp.nansum(exposures[index])
                p_max = jnp.nanmean(jnp.nanmax(Y[index] / exposures[index], -1))
                offset_list.append(np.log(p/ (1-p)))
                offset_max_list.append(np.log(p_max/(1-p_max)))
                peak = jnp.nanmean(jnp.nanargmax(Y[index] / exposures[index], -1))
                p_star = Y[index] / exposures[index]     
            elif family == "beta":
                p = jnp.nanmean(Y[index])
                p_max = jnp.nanmean(jnp.nanmax(Y[index], -1))
                peak = jnp.nanmean(jnp.nanargmax(Y[index], -1))
                offset_list.append(np.log(p / (1 - p)))
                offset_max_list.append(np.log(p_max/(1-p_max)))

        offset_peak_list.append(peak + 18 - basis.mean())

    offsets = np.array(offset_list)[None]
    offset_max = np.array(offset_max_list)[None]
    offset_peak = np.array(offset_peak_list)[None]
    offset_boundary_r = np.log(np.exp(2) - 1)
    offset_boundary_l = np.log(np.exp(2) - 1)
    offset_dict = {"t_max": offset_peak, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l}
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
    wTx, mu_mcmc, tmax_mcmc, cmax_mcmc = make_mu_mcmc(results_mcmc["X"], results_mcmc["lengthscale_deriv"], results_mcmc["alpha"],
                        results_mcmc["beta"], results_mcmc["W"], results_mcmc["lengthscale"], results_mcmc["c_max"],
                        results_mcmc["t_max_raw"], results_mcmc["sigma_t"], results_mcmc["sigma_c"], L_time, M_time, x_time + L_time, offset_dict)

    peaks = tmax_mcmc + basis.mean()
    peak_val = cmax_mcmc


    

    print("setup samples for plotting")
    flat_posterior = np.array(peaks.mean(-2))  # shape: (samples, metrics)

    # Posterior mean and 95% CI
    means = flat_posterior.mean(axis=(0,1))
    hdi = az.hdi(flat_posterior, hdi_prob=0.95)  # shape: (metrics, 2)

    # Sort by mean
    sorted_idx = np.argsort(means)
    sorted_means = means[sorted_idx]
    sorted_hdi = hdi[sorted_idx]
    sorted_names = [metrics[i] for i in sorted_idx]
    group_color = ["purple", "purple", "purple", "purple", "red", "red", "purple", "purple", "red", "blue", "red","red", "blue", "blue","blue","blue",]
    sorted_colors = [group_color[i] for i in sorted_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_names,
        y=sorted_means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=sorted_hdi[:, 1] - sorted_means,
            arrayminus=sorted_means - sorted_hdi[:, 0],
            thickness=1.5,
            
            width=0
        ),
        mode='markers',
        marker=dict(color = sorted_colors, size=6),
        showlegend=False,
        name='Posterior Mean'
        ))
    
    unique_colors = ["red", "blue", "purple"]
    group_names = ["Athleticism", "Skill", "Both"]
    for color, color_name in zip(unique_colors, group_names):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, size=8),
            name=f'Group: {color_name}',  # replace with actual group name if available
            showlegend=True
        ))

    fig.update_layout(
    title='95% CI of Peak Age by Metric',
    yaxis_title='Age of Peak',
    xaxis_title='Metric',
    height=25 * len(metrics),  # autoscale height for readability
    template='simple_white'
    )



    fig.write_image(f"model_output/model_plots/peaks/mcmc/{model_name}.png", format = "png")



    player_labels =  ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                            "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                            "Chris Paul", "Shaquille O'Neal"]
    predict_players = player_labels + ["Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                        "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                        "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                        "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", "LaMelo Ball",
                                        "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                        "Giannis Antetokounmpo", "Jrue Holiday"]
    
    predict_indices = [data.groupby("id")["name"].first().values.tolist().index(player) for player in predict_players]
    X = results_map["X"][jnp.array(predict_indices)]
    W = results_map["W"]
    wTx = np.einsum("nr,mr -> nm", X, W * np.sqrt(results_map["lengthscale"]))
    psi_x = np.hstack([np.cos(wTx), np.sin(wTx)]) * (1/ rff_dim)
    K_X = np.einsum("nk,mk -> nm", psi_x, psi_x)
    Dinv = np.diag(1 / np.sqrt(np.diag(K_X))) 
    corr = Dinv @ K_X @ Dinv
    dist_matrix = 1 - np.abs(corr)
    linkage_matrix = linkage(squareform(dist_matrix,checks = False), method="centroid")
    row_order = leaves_list(linkage_matrix)
    
    distances = linkage_matrix[:, 2]
    reordered_labels = np.array(predict_players)[row_order]


    # # Step 4: Reorder the distance matrix
    clustered_matrix = dist_matrix[row_order][:, row_order]

    # Step 5: Plot the heatmap


    fig = go.Figure(data=go.Heatmap(
        z=clustered_matrix,
        x = reordered_labels,
        y = reordered_labels,
        colorscale='Viridis',
        colorbar=dict(title='Distance',
        ),
        reversescale=True
    ))
    fig.update_xaxes(tickangle=90)
    fig.update_yaxes(autorange='reversed') 
    fig.update_yaxes(tickangle=90)

    fig.update_layout(
        title='Clustered K_X',
        xaxis=dict(showticklabels=True, tickfont=dict(size=5) ),
        yaxis=dict(showticklabels=False),
    )

    fig.write_image(f"model_output/model_plots/latent_space/{model_name}_K_X.png", format = "png")
    



    



    pca_latent_space = PCA(n_components=2)

    X_pca_df = pd.DataFrame(pca_latent_space.fit_transform(results_map["X"]), columns = ["Dim. 1", "Dim. 2"])
    
    X_pca_df = pd.concat([X_pca_df, id_df], axis = 1)
    X_pca_df["name"] = X_pca_df["name"].apply(lambda x: x if x in predict_players else "")
    X_pca_df["minutes"] /= np.max(X_pca_df["minutes"])
    X_pca_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
    fig = px.scatter(X_pca_df, x = "Dim. 1", y = "Dim. 2", color = "Position", text="name", size = "minutes",
                    opacity = .1, title="PCA Visualization of Latent Player Embedding", )
    fig.update_traces(textfont = dict(size = 7))
    fig.write_image(f"model_output/model_plots/latent_space/map/{model_name}_pca.png", format = "png")
    # clusters = GaussianMixture(n_components=3).fit_predict(X)
    # X_tsne_df["cluster"] = clusters
    # X_tsne_df["name"] = X_tsne_df["name"].apply(lambda x: x if x in player_labels else "")
    # X_tsne_df["minutes"] /= np.max(X_tsne_df["minutes"])
    # X_tsne_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
    # for index , metric in enumerate(metrics):
    #     X_tsne_df["peak_age"] = basis[np.argmax(transformed_mu_mcmc_mean[index], -1)]
    #     X_tsne_df["peak_val"] = np.max(transformed_mu_mcmc_mean[index], -1)
    #     cluster_avg = []
    #     subplot_titles = [f"Cluster {i}"  for i in range(0, X_tsne_df["cluster"].max() + 1)]
    #     fig = make_subplots(rows=2, cols=int(clusters.max()) + 1, subplot_titles=subplot_titles, shared_xaxes=True,
    #     shared_yaxes=True,specs=[[{"type": "table"}] * 3,
    #                             [{"type": "xy"}] * 3],
    #                             vertical_spacing=0.2)
    #     for cluster_idx in range(0, X_tsne_df["cluster"].max() + 1):
    #         cluster_indices = X_tsne_df[X_tsne_df["cluster"] == cluster_idx].index.values
    #         cluster_avg_curve = np.mean(transformed_mu_mcmc_curves[index, cluster_indices], 0)
    #         cluster_df = pd.DataFrame(cluster_avg_curve, columns = ["x","value"])
    #         cluster_df["cluster"] = f"Cluster {cluster_idx}" 
    #         cluster_avg.append(cluster_df)
    #         row = 1
    #         col = cluster_idx + 1
    #         for i in cluster_indices:
    #             inter_cluster_curve = transformed_mu_mcmc_curves[index, i]
    #             fig.add_trace(go.Scatter(
    #                 x=inter_cluster_curve[:,0], y=inter_cluster_curve[:,1],
    #                 mode='lines',
    #                 line=dict(color='gray', width=1),
    #                 opacity=0.05,
    #                 showlegend=False), 
    #                 row=row+1, col=col)
    #                 ## example players in cluster
    #         example_players = X_tsne_df[X_tsne_df["cluster"] == cluster_idx].sort_values(by = "minutes", ascending = False)[["id"]].merge(id_df[["id", "name"]], on = ["id"])[["name"]]
    #         fig.add_trace(
    #                 go.Table(
    #                     header=dict(values=list(example_players.columns)),
    #                     cells=dict(values=[example_players[col].tolist() for col in example_players.columns])
    #                 ),
    #                 row=1, col=col
    #             )
    #         # Plot one red curve
    #         fig.add_trace(go.Scatter(
    #             x=cluster_df["x"], y=cluster_df["value"],
    #             mode='lines',
    #             line=dict(color='red', width=2),
    #             showlegend=False
    #         ), row=row+1, col=col)
            

    #     fig.update_layout(
    #         title_text=f"Per Cluster Curves for {metric}",
    #         title_x=0.5  )
    #     fig.write_image(f"model_output/model_plots/{metric}_latent_space_cluster_spaghetti.png", format = "png")



    #     metric_cluster_df = pd.concat(cluster_avg)

    #     fig = px.line(metric_cluster_df, x="x", y = "value", color = "cluster")
    #     fig.write_image(f"model_output/model_plots/{metric}_latent_space_cluster_curves.png", format = "png")
    #     print(f"finished plotting curve analysis for {metric}")


    for index_player, player in zip(predict_indices, predict_players):

        fig = plot_posterior_predictive_career_trajectory(index_player, metrics, metric_output , posterior_mean_samples=mu_mcmc[..., index_player, :], observations=Y,
                                                        exposures= exposures, 
                                                        posterior_variance_samples=jnp.transpose(results_mcmc["sigma"][None,None], (2,0,1)),
                                                        posterior_dispersion_samples=results_mcmc["sigma_beta"][None,None],
                                                        exposure_names= exposure_list)
        fig.update_layout(
            title=dict(text=player))
        fig.write_image(f"model_output/model_plots/player_plots/predictions/mcmc/{model_name}_{player}.png", format = "png")





    obs, pos = create_metric_trajectory_all(mu_mcmc, Y, exposures, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"][None, None], (2,0,1)),
                                            results_mcmc["sigma_beta"][None, None]) 

    hdi = az.hdi(np.array(pos), hdi_prob = .95)

    hdi_low = hdi[..., 0]
    hdi_high = hdi[..., 1]
    avg_coverages = ((obs <= hdi_high) & (obs >= hdi_low)).sum((0,1)) / (~np.isnan(obs)).sum((0, 1))
    coverage_df = pd.DataFrame(avg_coverages, columns=["coverage"])
    coverage_df["metric"] = metrics
    fig = px.bar(coverage_df.sort_values(by = "coverage"), x='metric', y='coverage', title = "In Sample Coverage")
    fig.write_image(f"model_output/model_plots/coverage/{model_name}.png")

    val_indices = [data.groupby("id")["name"].first().values.tolist().index(player) for player in validation_data.groupby("id")["name"].first().values.tolist()]

    obs_val, pos_val = create_metric_trajectory_all(mu_mcmc[..., val_indices, :], Y_val, exposures_val, 
                                            metric_output, metrics, exposure_list, 
                                            jnp.transpose(results_mcmc["sigma"][None, None], (2,0,1)),
                                            results_mcmc["sigma_beta"][None, None]) 

    hdi_val = az.hdi(np.array(pos_val), hdi_prob = .95)

    hdi_low = hdi_val[..., 0]
    hdi_high = hdi_val[..., 1]
    avg_coverages = ((obs_val <= hdi_high) & (obs_val >= hdi_low)).sum((0,1)) / (~np.isnan(obs_val)).sum((0, 1))
    coverage_df = pd.DataFrame(avg_coverages, columns=["coverage"])
    coverage_df["metric"] = metrics
    fig = px.bar(coverage_df.sort_values(by = "coverage"), x='metric', y='coverage', title="Out of Sample Coverage")
    fig.write_image(f"model_output/model_plots/coverage/{model_name}_validation.png")

    player_avg_coverages = ((obs_val <= hdi_high) & (obs_val >= hdi_low)).sum((1)) / (~np.isnan(obs_val)).sum((1))
    player_coverage_df = pd.DataFrame(player_avg_coverages, columns=metrics)
    player_coverage_df["num_seasons"] = data.groupby("id").size().reset_index(name='count').merge(validation_data.groupby("id")["name"].first().reset_index())["count"]
    player_coverage_df = player_coverage_df.reset_index().melt(id_vars = ["index", "num_seasons"],  
                                                     var_name="metric",
                                                     value_name="coverage")
    player_coverage_df = player_coverage_df.groupby(["num_seasons", "metric"])["coverage"].mean().reset_index()
    fig = px.scatter(
        player_coverage_df,
        x="num_seasons",
        y="coverage",
        facet_col="metric",
        facet_col_wrap=4,
        title="Out of Sample Coverage vs Minutes Played (Faceted by Metric)",
        opacity=0.7  # optional, makes overlapping points easier to see
        )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.write_image(f"model_output/model_plots/coverage/{model_name}_validation_minutes.png")

    season_coverage_dict = {}
    for season in set(validation_data["season"]):
        season_mask_val = (season_mask == season)
        avg_coverages = ((obs_val[season_mask_val, :] <= hdi_high[season_mask_val, :]) & (obs_val[season_mask_val, :] >= hdi_low[season_mask_val, :])).mean((0)) 
        season_coverage_dict[season] = avg_coverages
    season_coverage_df = pd.DataFrame.from_dict(season_coverage_dict, orient = "index", columns = metrics)
    season_coverage_df.index.name = "season"
    season_coverage_df = season_coverage_df.reset_index().melt(id_vars = ["season"],  
                                                     var_name="metric",
                                                     value_name="coverage")
    fig = px.bar(season_coverage_df.sort_values(by = "season"), x='season', y='coverage', title="Out of Sample Coverage", facet_col="metric", facet_col_wrap = 4)
    fig.write_image(f"model_output/model_plots/coverage/{model_name}_validation_yearly.png")

    
    normalized_wTx = jnp.mod(wTx, jnp.pi * 2)
    rhat_normalized = az.rhat(np.array(normalized_wTx))
    rhat = az.rhat(np.array(wTx))
    print(f"max rhat across dimensions: {jnp.max(rhat)}")
    print(f"max rhat normalized across dimensions: {jnp.max(rhat_normalized)}")