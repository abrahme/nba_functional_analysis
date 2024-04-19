import pandas as pd
import argparse
import pickle
import jax.numpy as jnp
import numpyro

from data.data_utils import process_data, create_basis
from model.models import NBAFDAModel, NBAFDAREModel, NBAFDALatentModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--basis_dims", help="size of the basis", required=True)
    args = vars(parser.parse_args())
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")

    metric_output = (["gaussian"] * 2) + (["poisson"] * 6) + (["binomial"] * 3)
    metrics = ["obpm","dbpm","blk","stl","ast","dreb","oreb","tov","ftm","fg2m","fg3m"]
    exposure_list = (["median_minutes_per_game"] * 8) + ["fta","fg2a","fg3a"]
    covariate_X = create_basis(data, basis_dims)
    data_set = []
    for output,metric,exposure_val in zip(metric_output, metrics, exposure_list):
        exposure, Y, _ = process_data(data, metric, exposure_val, output, ["position_group"])
        data_dict = {"metric":metric, "output": output, "exposure_data": exposure, "output_data": Y, "mask": jnp.isfinite(exposure)}
        data_set.append(data_dict)

    basis = jnp.arange(18,39)

    if model_name == "nba_fda_model":
        model = NBAFDAModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_re_model":
        model = NBAFDAREModel(basis, output_size=len(metric_output), M=10)
    elif model_name == "nba_fda_latent_model":
        model = NBAFDALatentModel(basis, output_size=len(metric_output), M = 10, latent_dim1=covariate_X.shape[0], latent_dim2=basis_dims)
    else:
        raise ValueError("Wrong model name")
    
    model.initialize_priors()
    if "latent" not in model_name:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"covariate_X": covariate_X, "data_set": data_set})
    else:
        mcmc_run = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set})
    mcmc_run.print_summary()
    samples = mcmc_run.get_samples(group_by_chain=True)

    with open(f"model_output/{model_name}.pkl", "wb") as f:
        pickle.dump(samples, f)
    f.close()
    