import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import pickle
import numpyro
from numpyro.distributions import MatrixNormal
from numpyro.diagnostics import print_summary
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data, create_pca_data, create_cp_data, create_cp_data_multi_way, create_fda_data_time
from model.models import NBAFDAModel, NBAFDAREModel, NBAFDALatentModel, NBAMixedOutputProbabilisticPCA, NBAMixedOutputProbabilisticCPDecomposition, NBAMixedOutputProbabilisticCPDecompositionMultiWay, RFLVM, TVRFLVM, DriftRFLVM, DriftTVRFLVM, GibbsRFLVM, GibbsTVRFLVM



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    parser.add_argument("--fixed_param_path",help="where to read in the fixed params from", required=False, default="")
    parser.add_argument("--prior_x_path", help="if there is a prior on x, where to get the required params", required=False, default="")
    parser.add_argument("--output_path", help="where to store generated files", required = False, default="")
    parser.add_argument("--run_neutra", help = "whether or not to run neural reparametrization", action="store_true")
    numpyro.set_platform("cpu")
    # numpyro.set_host_device_count(4)
    args = vars(parser.parse_args())
    neural_parametrization = args["run_neutra"]
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    param_path = args["fixed_param_path"]
    prior_x_path = args["prior_x_path"]
    output_path = args["output_path"] if args["output_path"] else f"model_output/{model_name}.pkl"
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["retirement"] = 1
    metric_output = ["binomial", "exponential"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    

    if model_name == "nba_fda_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_re_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAREModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_latent_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDALatentModel(basis, output_size=len(metric_output), M = 10, latent_dim1=covariate_X.shape[0], latent_dim2=basis_dims)
    elif model_name == "gibbs_nba_rflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = GibbsRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)))
    elif model_name == "gibbs_nba_tvrflvm":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = GibbsTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    elif model_name == "exponential_pca":
        exposures, masks, X, outputs = create_pca_data(data, metric_output, exposure_list, metrics)
        model = NBAMixedOutputProbabilisticPCA(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
    elif model_name.startswith("exponential_cp"):
        if "multi" in model_name:
            exposures, masks, X, outputs  = create_cp_data_multi_way(data, metric_output, exposure_list, metrics)
            model = NBAMixedOutputProbabilisticCPDecompositionMultiWay(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
        else:
            exposures, masks, X, outputs = create_cp_data(data, metric_output, exposure_list, metrics)
            model = NBAMixedOutputProbabilisticCPDecomposition(X, basis_dims, masks, exposures, outputs, metric_output, metrics)

    elif "nba_rflvm" in model_name:
        if "drift" in model_name:
            covariate_X, data_set, basis, time_basis = create_fda_data_time(data, basis_dims, metric_output, metrics, exposure_list)
            model = DriftRFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(covariate_X.shape[0], len(basis)), drift_basis=time_basis)
        else:
            covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
            model = RFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(covariate_X.shape[0], len(basis)))
    elif "nba_tvrflvm" in model_name:
        if "drift" in model_name:
            covariate_X, data_set, basis, time_basis = create_fda_data_time(data, basis_dims, metric_output, metrics, exposure_list)
            model = DriftTVRFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(covariate_X.shape[0], len(basis)), basis=basis, drift_basis=time_basis)
        else:
            covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
            model = TVRFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(covariate_X.shape[0], len(basis)), basis=basis)
    else:
        raise ValueError("Wrong model name")
    
    model.initialize_priors()
    if model_name in ["nba_fda_model", "nba_fda_re_model"]:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"covariate_X": covariate_X, "data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)
    elif "pca" in model_name:
        svi_run = model.run_inference(num_steps=1000000)
        samples = svi_run.params
    elif "cp" in model_name:
        svi_run = model.run_inference(num_steps=1000000)
        samples = svi_run.params
    elif "rflvm" in model_name:
        prior_dict = {}
        if param_path:
            with open(param_path, "rb") as f_param:
                results_param = pickle.load(f_param)
            f_param.close()
            for param_name in results_param:
                # value = results_param[param_name].mean((0,1))
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
        if "gibbs" in model_name:
            if "tvrflvm" in model_name:
                mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set}, gibbs_sites=[["X", "lengthscale"], ["W","beta","sigma"]])  
            else:
                mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set}, gibbs_sites=[["X"], ["W","beta","sigma"]])  


        else:
            if not neural_parametrization:
                mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set})
            else:
                mcmc_run, neutra = model.run_neutra_inference(num_chains=4, num_samples=2000, num_warmup=1000, num_steps=10000, guide_kwargs={}, model_args={"data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)
        if neural_parametrization:
            samples = neutra.transform_sample(samples)
            print_summary(samples)

    else:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)

    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    f.close()
    