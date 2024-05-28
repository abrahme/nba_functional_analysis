import pandas as pd
import numpy as np
import jax.numpy as jnp
import argparse
import pickle
import numpyro

from data.data_utils import create_fda_data, create_pca_data, create_cp_data, create_cp_data_multi_way, create_fda_data_time
from model.models import NBAFDAModel, NBAFDAREModel, NBAFDALatentModel, NBAMixedOutputProbabilisticPCA, NBAMixedOutputProbabilisticCPDecomposition, NBAMixedOutputProbabilisticCPDecompositionMultiWay, RFLVM, TVRFLVM, DriftRFLVM, DriftTVRFLVM, FixedRFLVM, FixedTVRFLVM



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)
    args = vars(parser.parse_args())
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    metric_output = (["gaussian"] * 3) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["log_min", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = ["simple_exposure"] + (["minutes"] * 11) + ["fta","fg2a","fg3a"]
    

    if model_name == "nba_fda_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_re_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAREModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_latent_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDALatentModel(basis, output_size=len(metric_output), M = 10, latent_dim1=covariate_X.shape[0], latent_dim2=basis_dims)
    elif model_name == "fixed_nba_rflvm":
        _, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        with open("model_output/nba_rflvm.pkl", "rb") as f:
            results = pickle.load(f)
        f.close()
        X_rflvm = results["X_raw_auto_loc"]
        W = results["W_auto_loc"]
        U, _, _ = jnp.linalg.svd(X_rflvm, full_matrices=False)
        L       = jnp.linalg.cholesky(np.cov(U.T) + 1e-6 * np.eye(basis_dims)).T
        aligned_X  = np.linalg.solve(L, U.T).T
        X_tvrflvm_aligned = aligned_X / jnp.std(X_rflvm, axis=0)
        model = FixedRFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(X_rflvm.shape[0], len(basis)))
    elif model_name == "fixed_nba_tvrflvm":
        _, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        with open("model_output/nba_tvrflvm.pkl", "rb") as f:
            results = pickle.load(f)
        f.close()
        X_rflvm = results["X_raw_auto_loc"]
        W = results["W_auto_loc"]
        U, _, _ = jnp.linalg.svd(X_rflvm, full_matrices=False)
        L       = jnp.linalg.cholesky(np.cov(U.T) + 1e-6 * np.eye(basis_dims)).T
        aligned_X  = np.linalg.solve(L, U.T).T
        X_tvrflvm_aligned = aligned_X / jnp.std(X_rflvm, axis=0)
        model = FixedTVRFLVM(latent_rank=basis_dims, rff_dim=10, output_shape=(X_rflvm.shape[0], len(basis)), basis=basis)
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
        if "fixed" in model_name:
            mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set, "X": X_tvrflvm_aligned})
            mcmc_run.print_summary()
            samples = mcmc_run.get_samples(group_by_chain=True)
        else:
            svi_run = model.run_inference(num_steps=100000, model_args={"data_set": data_set})
            samples = svi_run.params
    else:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)

    with open(f"model_output/{model_name}.pkl", "wb") as f:
        pickle.dump(samples, f)
    f.close()
    