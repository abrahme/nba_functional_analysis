import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import pickle
import numpyro
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_fda_data
from model.models import FixedTVRFLVM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--basis_dims", help="size of the basis", required=True, type=int)
    numpyro.set_platform("cuda")
    numpyro.set_host_device_count(2)
    args = vars(parser.parse_args())
    basis_dims = args["basis_dims"]
    data = pd.read_csv("data/player_data.csv").query(" age <= 38 ")
    data["log_min"] = np.log(data["minutes"])
    data["simple_exposure"] = 1
    data["retirement"] = 1
    metric_output = ["binomial", "exponential"] + (["gaussian"] * 2) + (["poisson"] * 9) + (["binomial"] * 3)
    metrics = ["retirement", "minutes", "obpm","dbpm","blk","stl","ast","dreb","oreb","tov","fta","fg2a","fg3a","ftm","fg2m","fg3m"]
    exposure_list = (["simple_exposure"] * 2) + (["minutes"] * 11) + ["fta","fg2a","fg3a"]


    _, data_set, basis = create_fda_data(data, basis_dims, metric_output, metrics, exposure_list)
    with open("model_output/exponential_cp_test.pkl", "rb") as f:
        results = pickle.load(f)
    f.close()
    X_rflvm = results["U_auto_loc"]
    U, _, _ = jnp.linalg.svd(X_rflvm, full_matrices=False)
    L       = jnp.linalg.cholesky(np.cov(U.T) + 1e-6 * np.eye(basis_dims)).T
    aligned_X  = np.linalg.solve(L, U.T).T
    X_tvrflvm_aligned = aligned_X / jnp.std(X_rflvm, axis=0)
    model = FixedTVRFLVM(latent_rank=basis_dims, rff_dim=100, output_shape=(X_rflvm.shape[0], len(basis)), basis=basis)
    model.initialize_priors()
    def do_mcmc(dummy_arg):
        mcmc_run = model.run_inference(num_chains=2, num_samples=2000, num_warmup=1000, model_args={"data_set": data_set, "X": X_tvrflvm_aligned})
        samples = mcmc_run.get_samples(group_by_chain=True)
        return {**samples}
    
n_parallel = jax.local_device_count()

traces = jax.pmap(do_mcmc)(jnp.arange(2))
# concatenate traces along pmap'ed axis
trace = {k: np.concatenate(v) for k, v in traces.items()}