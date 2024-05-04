import pandas as pd
import numpy as np
import argparse
import pickle
import numpyro

from data.data_utils import create_fda_data, create_pca_data
from model.models import NBAFDAModel, NBAFDAREModel, NBAFDALatentModel, NBAMixedOutputProbabilisticPCA



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    args = vars(parser.parse_args())
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    metric_output = (["gaussian"] * 3) + (["poisson"] * 6) + (["binomial"] * 3)
    metrics = ["log_min", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","ftm","fg2m","fg3m"]
    exposure_list = ["simple_exposure"] + (["minutes"] * 8) + ["fta","fg2a","fg3a"]
    

    if model_name == "nba_fda_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_re_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDAREModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_latent_model":
        covariate_X, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
        model = NBAFDALatentModel(basis, output_size=len(metric_output), M = 10, latent_dim1=covariate_X.shape[0], latent_dim2=basis_dims)
    elif model_name == "exponential_pca":
        exposures, masks, X, outputs = create_pca_data(data, metric_output, exposure_list, metrics)
        model = NBAMixedOutputProbabilisticPCA(X, basis_dims, masks, exposures, outputs, metric_output, metrics)
    else:
        raise ValueError("Wrong model name")
    
    model.initialize_priors()
    if "latent" not in model_name:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"covariate_X": covariate_X, "data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)
    elif "pca" in model_name:
        svi_run = model.run_inference(num_steps=100000)
        samples = svi_run.state
    else:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set})
        mcmc_run.print_summary()
        samples = mcmc_run.get_samples(group_by_chain=True)

    with open(f"model_output/{model_name}.pkl", "wb") as f:
        pickle.dump(samples, f)
    f.close()
    