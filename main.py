import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import pickle
import numpyro
from numpyro.distributions import MatrixNormal
from numpyro.diagnostics import print_summary
from model.hsgp import eigenfunctions, make_convex_phi, sqrt_eigenvalues
# jax. config. update("jax_debug_infs", True)
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, create_cp_data
from model.models import  NBAMixedOutputProbabilisticCPDecomposition, RFLVM, TVRFLVM, IFTVRFLVM, ConvexTVRFLVM, GibbsRFLVM, GibbsTVRFLVM, GibbsIFTVRFLVM, GibbsConvexVRFLVM



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--fixed_param_path",help="where to read in the fixed params from", required=False, default="")
    parser.add_argument("--prior_x_path", help="if there is a prior on x, where to get the required params", required=False, default="")
    parser.add_argument("--output_path", help="where to store generated files", required = False, default="")
    parser.add_argument("--vectorized", help="whether to vectorize some chains so all gpus will be used", action="store_true")
    parser.add_argument("--run_neutra", help = "whether or not to run neural reparametrization", action="store_true")
    numpyro.set_platform("cuda")
    # numpyro.set_host_device_count(4)
    args = vars(parser.parse_args())
    neural_parametrization = args["run_neutra"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    param_path = args["fixed_param_path"]
    vectorized = args["vectorized"]
    prior_x_path = args["prior_x_path"]
    output_path = args["output_path"] if args["output_path"] else f"model_output/{model_name}.pkl"
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["retirement"] = 1
    metric_output = ["binomial", "exponential"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    

    if model_name == "gibbs_nba_rflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = GibbsRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)))
    elif model_name == "gibbs_nba_tvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = GibbsTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif model_name == "gibbs_nba_iftvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = GibbsIFTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)    
    elif model_name == "exponential_cp":
        exposures, masks, X, outputs = create_cp_data(data, metric_output, exposure_list, metrics)
        model = NBAMixedOutputProbabilisticCPDecomposition(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
    elif "rflvm" in model_name:
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = RFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)))
        if "convex" in model_name:
            model = ConvexTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "iftvrflvm" in model_name:
            model = IFTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
        elif "tvrflvm" in model_name:
            model = TVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
   
    else:
        raise ValueError("Model not implemented")

    model.initialize_priors()
    if "cp" in model_name:
        svi_run = model.run_inference(num_steps=1000000)
        samples = svi_run.params
    elif "rflvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            for param_name in results_param:
                value = results_param[param_name]
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
        model.prior.update(prior_dict)
        distribution_families = set([data_entity["output"] for data_entity in data_set])
        distribution_indices = {family: jnp.array([index for index, data_entity in enumerate(data_set) if family == data_entity["output"]]) for family in distribution_families}
        masks = jnp.stack([data_entity["mask"] for data_entity in data_set])
        exposures = jnp.stack([data_entity["exposure_data"] for data_entity in data_set])
        Y = jnp.stack([data_entity["output_data"] for data_entity in data_set])
        num_gaussians = len(distribution_indices['gaussian'])
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
                M_time = 5
                phi_time = make_convex_phi(x_time, L_time, M_time)
                psi_time = eigenfunctions(x_time, L_time, M_time)
                eig_val_time = jnp.square(sqrt_eigenvalues(M_time, L_time))
                psi_x_time_cross = (x_time + L_time)[..., None]  - psi_time / eig_val_time.T
                hsgp_params["psi_x_time_cross"] = psi_x_time_cross
                hsgp_params["eig_val_time"] = eig_val_time
                hsgp_params['psi_x_time'] = psi_time
                hsgp_params["phi_x_time"] = phi_time
                hsgp_params["shifted_x_time"] = x_time + L_time
                hsgp_params["M_time"] = M_time
                hsgp_params["L_time"] = L_time
                hsgp_params["M"] = 35

        if "gibbs" in model_name:
            if "convex" in model_name:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict, "hsgp_params": hsgp_params}, gibbs_sites=[["X", "lengthscale", "alpha"], ["beta_time","beta","sigma", "slope", "intercept"]])
            elif "tvrflvm" in model_name:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict}, gibbs_sites=[["X", "lengthscale"], ["W","beta","sigma"]])  
            else:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_dict}, gibbs_sites=[["X"], ["W","beta","sigma"]])  
        else:
            model_args = {"data_set": data_dict}
            if "convex" in model_name:
                model_args.update({ "hsgp_params": hsgp_params})
            if not neural_parametrization:
                samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, vectorized=vectorized, model_args=model_args)
            else:
                mcmc_run, neutra = model.run_neutra_inference(num_chains=4, num_samples=2000, num_warmup=1000, num_steps=1000000, guide_kwargs={}, model_args=model_args)
                samples = mcmc_run.get_samples(group_by_chain=True)
        if neural_parametrization:
            samples = neutra.transform_sample(samples)
            print_summary(samples)
        else:
            print_summary(samples)

    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    f.close()
    
        
        