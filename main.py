import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsci
import argparse
import pickle
import numpyro
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform, pdist
from numpyro.distributions import MatrixNormal, Normal
from numpyro.diagnostics import print_summary
from model.hsgp import make_convex_phi, diag_spectral_density, make_convex_f, make_psi_gamma
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, create_cp_data
from model.models import  NBAMixedOutputProbabilisticCPDecomposition, NBANormalApproxProbabilisticCPDecomposition, RFLVM, TVRFLVM, IFTVRFLVM, ConvexTVRFLVM, GibbsRFLVM, GibbsTVRFLVM, GibbsIFTVRFLVM
from visualization.visualization import plot_posterior_predictive_career_trajectory_map



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--rff_dim", help="size of the rff approx", required=True, type=int)
    parser.add_argument("--fixed_param_path",help="where to read in the fixed params from", required=False, default="")
    parser.add_argument("--prior_x_path", help="if there is a prior on x, where to get the required params", required=False, default="")
    parser.add_argument("--output_path", help="where to store generated files", required = False, default="")
    parser.add_argument("--vectorized", help="whether to vectorize some chains so all gpus will be used", action="store_true")
    parser.add_argument("--run_neutra", help = "whether or not to run neural reparametrization", action="store_true")
    parser.add_argument("--run_svi", help = "whether or not to run variational inference", action="store_true")
    parser.add_argument("--init_path", help = "where to initialize mcmc from", required=False, default="")
    parser.add_argument("--player_names", help = "which players to run the model for", required=False, default = [], type = lambda x: x.split(","))
    parser.add_argument("--position_group", help = "which position group to run the model for", required = True, choices=["G", "F", "C", "all"])
    numpyro.set_platform("cuda")
    args = vars(parser.parse_args())
    neural_parametrization = args["run_neutra"]
    svi_inference = args["run_svi"]
    initial_params_path = args["init_path"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    rff_dim = args["rff_dim"]
    param_path = args["fixed_param_path"]
    vectorized = args["vectorized"]
    prior_x_path = args["prior_x_path"]
    output_path = args["output_path"] if args["output_path"] else f"model_output/{model_name}.pkl"
    players = args["player_names"]
    position_group = args["position_group"]
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    names = data.groupby("id")["name"].first().values.tolist()
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["games_exposure"] = np.maximum(82, data["games"]) ### 82 or whatever
    data["pct_minutes"] = (data["minutes"] / data["games_exposure"]) / 48
    data["retirement"] = 1
    metric_output = ["binomial", "beta"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["games", "pct_minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["games_exposure", "games_exposure"]) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]

    if players:
        player_indices = [names.index(item) for item in players]
    elif position_group in ["G","F","C"]:
        all_indices = data.drop_duplicates(subset=["position_group","name","id"]).reset_index()
        player_indices = all_indices[all_indices["position_group"] == position_group].index.values.tolist()
    else:
        player_indices = []
        
    if model_name == "gibbs_nba_rflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)))
    elif model_name == "gibbs_nba_tvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif model_name == "gibbs_nba_iftvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsIFTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)    
    elif model_name == "exponential_cp":
        exposures, masks, X, outputs = create_cp_data(data, metric_output, exposure_list, metrics, player_indices)
        model = NBAMixedOutputProbabilisticCPDecomposition(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
    elif model_name == "exponential_cp_normalize":
        exposures, masks, X, outputs = create_cp_data(data, metric_output, exposure_list, metrics, player_indices, normalize=True)
        model = NBANormalApproxProbabilisticCPDecomposition(X, basis_dims, masks, exposures)
    elif "rflvm" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = RFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)))
        if "convex" in model_name:
            model = ConvexTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "iftvrflvm" in model_name:
            model = IFTVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "tvrflvm" in model_name:
            model = TVRFLVM(latent_rank=basis_dims, rff_dim=rff_dim, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
   
    else:
        raise ValueError("Model not implemented")

    model.initialize_priors()
    initial_params = {}
    if "cp" in model_name:
        if initial_params_path:
            with open(initial_params_path, "rb") as f_init:
                initial_params = pickle.load(f_init)
            f_init.close()
        svi_run = model.run_inference(num_steps=10000000, initial_values=initial_params)
        samples = svi_run.params
    elif "rflvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            for param_name in results_param:
                value = results_param[param_name]
                if (param_name == "X") and (player_indices):
                    value = results_param[param_name][jnp.array(player_indices)]
                prior_dict[param_name] = numpyro.deterministic(param_name, value)
        if prior_x_path:
            with open(prior_x_path, "rb") as f_prior:
                results_prior = pickle.load(f_prior)
            f_prior.close()
            X_rflvm = results_prior["X"]
            U, _, _ = jnp.linalg.svd(X_rflvm, full_matrices=False)
            L       = jnp.linalg.cholesky(jnp.cov(U.T) + 1e-6 * jnp.eye(basis_dims)).T
            aligned_X  = np.linalg.solve(L, U.T).T
            X = aligned_X / jnp.std(X_rflvm, axis=0) 
            n, r = X.shape
            cov_rows = jnp.cov(X) + jnp.eye(n) * (1e-6) 
            cholesky_rows = jnp.linalg.cholesky(cov_rows) + jnp.eye(n) * (1e-6)
            cov_cols = jnp.cov(X.T) + jnp.eye(r) * (1e-6)
            cholesky_cols = jnp.linalg.cholesky(cov_cols) + jnp.eye(r) * (1e-6)
            prior_dict["X"] = MatrixNormal(loc = X, scale_tril_column=cholesky_cols, scale_tril_row=cholesky_rows)
        if initial_params_path:
            with open(initial_params_path, "rb") as f_init:
                initial_params = pickle.load(f_init)
            f_init.close()
            if not svi_inference:
                initial_params = {key.replace("__loc",""):val for key,val in initial_params.items()}
        model.prior.update(prior_dict)
        distribution_families = set([data_entity["output"] for data_entity in data_set])
        distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
        masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
        exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
        Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
        offset_list = []
        for index, family in enumerate(metric_output):  
            if family == "gaussian":
                offset_list.append(jnp.nanmean(Y[index], keepdims = True))
            else:
                if family == "poisson":
                    p = jnp.nansum(Y[index], keepdims = True) / jnp.nansum(jnp.exp(exposures[index]), keepdims = True)
                    offset_list.append(jnp.log(p))
                elif family == "binomial":
                    p = jnp.nansum(Y[index], keepdims = True) / jnp.nansum(exposures[index], keepdims = True)
                    offset_list.append(jnp.log(p/ (1-p)))
                elif family == "beta":
                    p = jnp.nanmean(Y[index], keepdims = True)
                    offset_list.append(jnp.log(p / (1 - p)))

        offsets = jnp.stack(offset_list)
        num_gaussians = 0 if "gaussian" not in distribution_indices else len(distribution_indices.get('gaussian'))
        data_dict = {}
        for family in distribution_families:
            family_dict = {}
            indices = distribution_indices[family]
            family_dict["Y"] = Y[indices]
            family_dict["offset"] = offsets[indices]
            family_dict["exposure"] = exposures[indices]
            family_dict["mask"] = masks[indices]
            family_dict["indices"] = indices
            data_dict[family] = family_dict
        if "convex" in model_name:
                hsgp_params = {}
                x_time = basis - basis.mean()
                L_time = 1.5 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
                M_time = 15
                phi_time = make_convex_phi(x_time, L_time, M_time)
                hsgp_params["phi_x_time"] = phi_time
                hsgp_params["M_time"] = M_time
                hsgp_params["L_time"] = L_time
                hsgp_params["shifted_x_time"] = x_time + L_time
        if "gibbs" in model_name:
            if "convex" in model_name:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict, "hsgp_params": hsgp_params}, gibbs_sites=[["X", "lengthscale", "alpha"], ["W", "beta_time","beta","sigma", "slope", "intercept"]])
            elif "tvrflvm" in model_name:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict}, gibbs_sites=[["X", "lengthscale"], ["W","beta","sigma"]])  
            else:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict}, gibbs_sites=[["X"], ["W","beta","sigma"]])  

        else:
            model_args = {"data_set": data_dict, "offsets": offsets}
            if "convex" in model_name:
                model_args.update({ "hsgp_params": hsgp_params})
            if svi_inference:
                samples = model.run_svi_inference(num_steps=500000, model_args=model_args, initial_values=initial_params)
            elif not neural_parametrization:
                samples, extra_fields = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, vectorized=vectorized, model_args=model_args, initial_values=initial_params)
            else:
                mcmc_run, neutra = model.run_neutra_inference(num_chains=4, num_samples=2000, num_warmup=1000, num_steps=1000000, guide_kwargs={}, model_args=model_args)
                samples = mcmc_run.get_samples(group_by_chain=True)
        if neural_parametrization:
            samples = neutra.transform_sample(samples)
            print_summary(samples)
        elif svi_inference:
            pass
        else:
            print_summary(samples)

    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    f.close()


    if svi_inference:
        
        print(samples["sigma__loc"])
        slope_sigma = samples["slope_sigma__loc"][None, ...]
        intercept_sigma = samples["intercept_sigma__loc"][None, ...]
        slope_offset = samples["slope_offset__loc"][None, ...]
        curvature_offset = samples["curvature_offset__loc"]
        ls_deriv = samples["lengthscale_deriv__loc"]
        alpha_time = samples["alpha__loc"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = samples["beta__loc"]
        weights = weights * spd * .0001
        W = samples["W__loc"]
        lengthscale = samples["lengthscale__loc"][None]


        # C = np.corrcoef(lengthscale)  # N X N correlation matrix
        # labels = metrics

        # # ---- Convert correlation matrix to distance matrix ----
        # D = 1 - C  # correlation distance (0 = perfect corr, 2 = perfect anti-corr)

        # # ---- Hierarchical clustering ----
        # linkage_mat = linkage(squareform(D, checks=False), method="ward")
        # order = leaves_list(linkage_mat)

        # # ---- Reorder correlation matrix ----
        # C_reordered = C[np.ix_(order, order)]
        # labels_ordered = [labels[i] for i in order]

        # # ---- Create dendrograms ----
        # dendro_rows = ff.create_dendrogram(C, orientation='left', linkagefun=lambda _: linkage_mat, labels=labels_ordered)
        # dendro_cols = ff.create_dendrogram(C, orientation='top', linkagefun=lambda _: linkage_mat, labels=labels_ordered)

        # # ---- Create heatmap ----
        # heatmap = go.Heatmap(
        #     z=C_reordered,
        #     x=labels_ordered,
        #     y=labels_ordered,
        #     colorscale='RdBu',
        #     zmin=0,
        #     zmax=1,
        #     xaxis='x2',
        #     yaxis='y2',
        #     reversescale=True
        # )

        # # ---- Combine into single figure ----
        # fig = go.Figure()

        # for trace in dendro_cols['data']:
        #     fig.add_trace(trace)
        # for trace in dendro_rows['data']:
        #     fig.add_trace(trace)
        # fig.add_trace(heatmap)

        # fig.update_layout(
        #     width=900,
        #     height=900,
        #     showlegend=False,
        #     hovermode='closest',
        #     xaxis=dict(domain=[0.15, 1.0], zeroline=False, showticklabels=False, ticks=''),
        #     yaxis=dict(domain=[0.15, 1.0], zeroline=False, showticklabels=False, ticks=''),
        #     xaxis2=dict(domain=[0.15, 1.0], anchor='y2'),
        #     yaxis2=dict(domain=[0.15, 1.0], anchor='x2'),
        #     margin=dict(t=50, l=50)
        # )

        # # Plot!

        # fig.write_image(f"model_output/model_plots/loading_correlation/svi/ard_model.png", format = "png")
                

        # fig = px.imshow(lengthscale, zmin=0, labels = dict(x = "Dimension",
        #                                              y = "Metric"),
        #                                              x = [f"Dimension {i+1}" for i in range(basis_dims)],
        #                                              y = metrics,

        #                                              )
        # fig.write_image("model_output/model_plots/loading/svi/ard_model.png", format = "png")

        print(lengthscale)

        X = samples["X__loc"]
        X -= jnp.mean(X, keepdims = True, axis = 0)
        X /= jnp.std(X, keepdims = True, axis = 0)
        wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(100))
        slope = make_psi_gamma(psi_x, samples["slope__loc"]) * slope_sigma + slope_offset
        intercept = make_psi_gamma(psi_x, samples["intercept__loc"]) * intercept_sigma
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
        mu = (make_convex_f(gamma_phi_gamma_x, x_time + L_time, slope, intercept[..., None])) + offsets - jnp.einsum("k,t -> kt", curvature_offset * .001, jnp.square(shifted_x_time)/2)[:, None, :]
        peaks = pd.DataFrame(jnp.mean(jnp.argmax(mu, axis = -1) + 18, axis = -1), columns=["Peak Age"])
        peaks["Metric"] = metrics
        fig = px.bar(peaks.sort_values(by = "Peak Age"), x = "Metric", y = "Peak Age")
        fig.write_image("model_output/model_plots/peaks/svi/ard_model.png", format = "png")

        



        player_labels = ["Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", 
                         "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                         "Chris Paul", "Shaquille O'Neal"]
        predict_players = player_labels + ["Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                                           "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd",
                                           "Marcus Camby", "Rudy Gobert", "Tim Duncan", "Manu Ginobili", "James Harden", "Russell Westbrook",
                                           "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", "LaMelo Ball",
                                           "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", 
                                           "Giannis Antetokounmpo", "Jrue Holiday"]
        tsne = TSNE(n_components=2)
        X_tsne_df = pd.DataFrame(tsne.fit_transform(X), columns = ["Dim. 1", "Dim. 2"])
        id_df = data[["position_group","name","id", "minutes"]].groupby("id").max().reset_index()
        X_tsne_df = pd.concat([X_tsne_df, id_df], axis = 1)
        X_tsne_df["name"] = X_tsne_df["name"].apply(lambda x: x if x in predict_players else "")
        X_tsne_df["minutes"] /= np.max(X_tsne_df["minutes"])
        X_tsne_df.rename(mapper = {"position_group": "Position"}, inplace=True, axis=1)
        fig = px.scatter(X_tsne_df, x = "Dim. 1", y = "Dim. 2", color = "Position", text="name", size = "minutes",
                         opacity = .1, title="T-SNE Visualization of Latent Player Embedding", )
        fig.update_traces(textfont = dict(size = 7))
        fig.write_image(f"model_output/model_plots/latent_space/svi/ard_model.png", format = "png")

        
        
        players_df = id_df[id_df["name"].isin(predict_players)]
        for index, row in players_df.iterrows():
            player_index = index
            name = row["name"]
            fig = plot_posterior_predictive_career_trajectory_map(player_index, metrics, metric_output, mu[:, jnp.array(player_index), :].squeeze(), Y, exposures)
            fig.update_layout(title = dict(text=name))
            fig.write_image(f"model_output/model_plots/player_plots/predictions/svi/ard_games_poisson_minutes_{name.replace(' ', '_')}.png", format = "png")
        