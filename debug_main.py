import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import pickle
import numpyro
import matplotlib.pyplot as plt
from numpyro.diagnostics import print_summary
from model.hsgp import make_convex_phi, diag_spectral_density
jax.config.update("jax_enable_x64", True)
from data.data_utils import create_convex_data
from model.models import ConvexGP



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model_name', help='which model to fit', required=True)
    parser.add_argument("--output_path", help="where to store generated files", required = False, default="")
    parser.add_argument("--vectorized", help="whether to vectorize some chains so all gpus will be used", action="store_true")
    parser.add_argument("--run_neutra", help = "whether or not to run neural reparametrization", action="store_true")
    parser.add_argument("--run_svi", help = "whether or not to run variational inference", action="store_true")
    parser.add_argument("--init_path", help = "where to initialize mcmc from", required=False, default="")
    numpyro.set_platform("cuda")
    args = vars(parser.parse_args())
    neural_parametrization = args["run_neutra"]
    svi_inference = args["run_svi"]
    initial_params_path = args["init_path"]
    model_name = args["model_name"]
    vectorized = args["vectorized"]
    output_path = args["output_path"] if args["output_path"] else f"model_output/{model_name}.pkl"
    
    
    
    samples, intercept, multiplier, noise, y_vals = create_convex_data(num_samples=10,  data_range=[-10.5, 10.5], noise_level=.001)

    basis = jnp.array(samples)
    obs = (y_vals) * multiplier + noise + intercept
    model = ConvexGP(basis)
    model.initialize_priors()
    initial_params = {}


    if initial_params_path:
        with open(initial_params_path, "rb") as f_init:
            initial_params = pickle.load(f_init)
        f_init.close()
        
    exposures = jnp.ones_like(basis)
    Y = jnp.array(obs)
    data_dict = {}

    data_dict["Y"] = Y
    data_dict["exposure"] = exposures

    hsgp_params = {}
    x_time = basis - basis.mean()
    L_time = 1.5 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
    M_time = 15
    phi_time = make_convex_phi(x_time, L_time, M_time)
    hsgp_params["phi_x_time"] = phi_time
    hsgp_params["M_time"] = M_time
    hsgp_params["L_time"] = L_time
    hsgp_params["shifted_x_time"] = x_time + L_time

    model_args = {"data_set": data_dict}
    model_args.update({ "hsgp_params": hsgp_params})
    if svi_inference:
        samples = model.run_svi_inference(num_steps=1000000, model_args=model_args, initial_values=initial_params)
    elif not neural_parametrization:
        samples = model.run_inference(num_chains=4, num_samples=2000, num_warmup=1000, vectorized=vectorized, model_args=model_args, initial_values=initial_params)
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
        slope = samples["slope__loc"]
        ls_deriv = samples["lengthscale_deriv__loc"]
        intercept = samples["intercept__loc"]
        alpha_time = samples["alpha__loc"]
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = samples["beta__loc"]
        weights = weights * spd 
        gamma_phi_gamma_time = jnp.einsum("tmz, m, z -> t", phi_time, weights, weights) 
        mu = intercept + slope * (x_time + L_time) - gamma_phi_gamma_time

        plt.plot(basis, obs)
        plt.plot(basis, mu)
        plt.savefig("model_output/model_plots/debug_predictions.png")

    
    
        