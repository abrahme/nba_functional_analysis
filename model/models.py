import pandas as pd
from abc import abstractmethod, ABC
import arviz as az
import jax 
from numpyro import sample , plate
from numpyro.distributions import HalfNormal, InverseGamma, Normal, Exponential, Poisson, Binomial
from numpyro.infer import MCMC, NUTS, init_to_median
import jax.numpy as jnp
from .hsgp import approx_se_ncp



class HilbertSpaceFunctionalRegression(ABC):
    def __init__(self, basis, output_size: int = 1, M: int = 10) -> None:
        self.basis = basis 
        self.output_size = output_size
        self.M = M
        self.prior = {}
        self.L = jnp.max(jnp.abs(basis), axis=0)*1.5
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def model_fn(self, *args, **kwargs) -> None:
        raise NotImplementedError 

    @abstractmethod
    def sample_basis(self, *args, **kwargs):
        alpha = sample("alpha", self.prior["alpha"])
        length = sample("length", self.prior["length"])
        return approx_se_ncp(self.basis, alpha=alpha, length=length, M = self.M, output_size=self.output_size,
                             L =  self.L)
    @abstractmethod
    def run_inference(self, *args, **kwargs):
        raise NotImplementedError

class NBAFDAModel(HilbertSpaceFunctionalRegression):
    def __init__(self, basis, output_size: int = 1, M: int = 10) -> None:
        super().__init__(basis, output_size, M)
    
    def initialize_priors(self) -> None:
        self.prior["alpha"] = HalfNormal()
        self.prior["length"] = InverseGamma(10.0, 2.0)
    
    def sample_basis(self, dim):
        alpha = sample("alpha", self.prior["alpha"], sample_shape=(dim,))
        length = sample("length", self.prior["length"], sample_shape=(dim,))
        return jnp.stack([approx_se_ncp(self.basis, alpha=alpha[i], length=length[i], M = self.M, output_size=self.output_size,
                             L =  self.L) for i in range(dim)] )
    
    def model_fn(self, covariate_X, data_set) -> None:
        covariate_dim = covariate_X.shape[1]
        num_outputs = len(data_set)
        intercept = sample("intercept", Normal(0, 5), sample_shape =  (self.basis.shape[0], num_outputs))

        basis = self.sample_basis(covariate_dim)
        
        mu = intercept + jnp.matmul(covariate_X, basis) 
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure =  exposure_data[mask].flatten()
            if output == "gaussian":
                sd = Exponential(f"sigma_{metric}", 1.0)
                ## likelihood
                y = sample(f"likelihood_{metric}", Normal( loc = mu[:,:,index][mask].flatten(), scale = sd / exposure),  obs=output_data[mask].flatten())
            
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson(jnp.exp(mu[:,:,index][mask].flatten() + exposure)) , obs = output_data[mask].flatten())
            
            elif output == "binomial":
                y = sample(f"likelihod_{metric}", Binomial(logits =  mu[:,:,index][mask].flatten + intercept, total_count = exposure), obs=output_data[mask].flatten())
        
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        mcmc = MCMC(
        NUTS(self.model_fn, init_strategy=init_to_median),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
        mcmc.run(jax.random.PRNGKey(0), **model_args)
        return mcmc
