import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def fda_model_re(covariate_X, data_set, basis):
    covariate_size = covariate_X.shape[1]
    with pm.Model() as fda_model_intercept:
        fixed_effects = pm.MutableData("X", covariate_X) ### X data (latent space)
        for data_entity in data_set:
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure = pm.MutableData(f"exposure_{metric}", exposure_data[mask].flatten())
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            gp = pm.gp.HSGP(m=[10], c=4.0, cov_func=cov_func)
            basis_weights =  pm.math.stack([gp.prior(f"basis_weights_{metric}_{i}", X = basis[:, None]) for i in range(covariate_size)], axis = 0)
            mu =  pm.math.dot(fixed_effects, basis_weights) ### get mu
            intercept = pm.Normal(f"intercept_{metric}", 0, 5, shape = (1,))
            random_intercept = pm.Normal(f"random_player_intercept_{metric}", 0, 1, shape = output_data.shape)
            sd_re = pm.Exponential(f"sd_re_{metric}", 1)

            if output == "gaussian":
                sd = pm.Exponential(f"sigma_{metric}", 1.0)
                ## likelihood
                y = pm.Normal(f"likelihood_{metric}", mu = sd_re*pm.math.flatten(random_intercept[mask]) + intercept + pm.math.flatten(mu[mask]), sigma = sd / exposure, observed=output_data[mask].flatten())
            
            elif output == "poisson":
                y = pm.Poisson(f"likelihood_{metric}", mu = pm.math.exp(pm.math.flatten(mu[mask]) + intercept + exposure + sd_re*pm.math.flatten(random_intercept[mask])), observed = output_data[mask].flatten())
            
            elif output == "binomial":
                y = pm.Binomial(f"likelihod_{metric}", logit_p = sd_re*pm.math.flatten(random_intercept[mask]) + pm.math.flatten(mu[mask]) + intercept, n = exposure, observed=output_data[mask].flatten())
        
        # gv = pm.model_graph.model_to_graphviz()
        # gv.format = 'png'
        # gv.render(filename='model_graph')

    with fda_model_intercept:
        print("fitting model")
        trace = pm.sample()
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    az.to_netcdf(trace, "data/uncorrelated_metrics_pca_re.ncdf")