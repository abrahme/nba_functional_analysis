import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsci
import argparse
import pickle
import numpyro
import matplotlib.pyplot as plt
from numpyro.distributions import MatrixNormal
from numpyro.diagnostics import print_summary
from model.hsgp import make_convex_phi, diag_spectral_density, make_convex_f, make_psi_gamma
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, create_cp_data
from model.models import  NBAMixedOutputProbabilisticCPDecomposition, RFLVM, TVRFLVM, IFTVRFLVM, ConvexTVRFLVM, GibbsRFLVM, GibbsTVRFLVM, GibbsIFTVRFLVM
from visualization.visualization import plot_posterior_predictive_career_trajectory_map



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
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
    data["retirement"] = 1
    metric_output = ["binomial", "poisson"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]

    if players:
        player_indices = [names.index(item) for item in players]
    elif position_group in ["G","F","C"]:
        all_indices = data.drop_duplicates(subset=["position_group","name","id"]).reset_index()
        player_indices = all_indices[all_indices["position_group"] == position_group].index.values.tolist()
    else:
        player_indices = []
        
    if model_name == "gibbs_nba_rflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)))
    elif model_name == "gibbs_nba_tvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif model_name == "gibbs_nba_iftvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = GibbsIFTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)    
    elif model_name == "exponential_cp":
        exposures, masks, X, outputs = create_cp_data(data, metric_output, exposure_list, metrics, player_indices)
        model = NBAMixedOutputProbabilisticCPDecomposition(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
    elif "rflvm" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list, player_indices)
        model = RFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)))
        if "convex" in model_name:
            model = ConvexTVRFLVM(latent_rank=basis_dims, rff_dim=50, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "iftvrflvm" in model_name:
            model = IFTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "tvrflvm" in model_name:
            model = TVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
   
    else:
        raise ValueError("Model not implemented")

    model.initialize_priors()
    initial_params = {}
    if "cp" in model_name:
        if initial_params_path:
            with open(initial_params_path, "rb") as f_init:
                initial_params = pickle.load(f_init)
            f_init.close()
        svi_run = model.run_inference(num_steps=1000000, initial_values=initial_params)
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
        num_gaussians = 0 if "gaussian" not in distribution_indices else len(distribution_indices.get('gaussian'))
        data_dict = {}
        for family in distribution_families:
            family_dict = {}
            indices = distribution_indices[family]
            family_dict["Y"] = Y[indices]
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
            model_args = {"data_set": data_dict}
            if "convex" in model_name:
                model_args.update({ "hsgp_params": hsgp_params})
            if svi_inference:
                samples = model.run_svi_inference(num_steps=1000000, model_args=model_args, initial_values=initial_params)
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
        player_indices = [1323] ### curry
        print(samples["sigma__loc"])
        ls_deriv = samples["lengthscale_deriv__loc"]
        alpha_time = samples["alpha__loc"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = samples["beta__loc"]
        weights = weights * spd * .0001
        W = samples["W__loc"] 
        X = results_param["X"]
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        psi_x = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(50))
        slope = make_psi_gamma(psi_x, samples["slope__loc"])
        intercept = make_psi_gamma(psi_x, samples["intercept__loc"])
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
        mu = (make_convex_f(gamma_phi_gamma_x, x_time + L_time, slope, intercept))[:, jnp.array(player_indices), :].squeeze()
        fig = plot_posterior_predictive_career_trajectory_map(player_indices[0], metrics, metric_output, mu, Y, exposures)
        fig.write_image("model_output/model_plots/debug_predictions_svi_full_poisson_minutes.png", format = "png")

    

   

    
        
        