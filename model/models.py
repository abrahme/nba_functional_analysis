from abc import abstractmethod, ABC
import jax 
from numpyro import sample 
from numpyro import deterministic
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
                             L =  self.L, name = "beta")
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
        return deterministic("basis", jnp.stack([approx_se_ncp(self.basis, alpha=alpha[i], length=length[i], M = self.M, output_size=self.output_size,
                             L =  self.L, name= f"beta_{i}") for i in range(dim)] ))
    
    def model_fn(self, covariate_X, data_set) -> None:
        covariate_dim = covariate_X.shape[1]
        num_outputs = len(data_set)
        intercept = sample("intercept", Normal(0, 5), sample_shape =  (1, num_outputs))

        basis = self.sample_basis(covariate_dim)
        mu = intercept + jnp.einsum("...i,ijk -> ...jk",covariate_X, basis)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure =  exposure_data[mask].flatten()
            if output == "gaussian":
                sd = sample(f"sigma_{metric}", Exponential(1.0))
                ## likelihood
                y = sample(f"likelihood_{metric}", Normal( loc = mu[:,:,index][mask].flatten(), scale = sd / exposure),  obs=output_data[mask].flatten())
            
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson(jnp.exp(mu[:,:,index][mask].flatten() + exposure)) , obs = output_data[mask].flatten())
            
            elif output == "binomial":
                y = sample(f"likelihod_{metric}", Binomial(logits =  mu[:,:,index][mask].flatten(), total_count = exposure), obs=output_data[mask].flatten())
        
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        mcmc = MCMC(
        NUTS(self.model_fn, init_strategy=init_to_median),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        chain_method="parallel"
    )
        mcmc.run(jax.random.PRNGKey(0), **model_args)
        return mcmc


class NBAFDAREModel(NBAFDAModel):
    def __init__(self, basis, output_size: int = 1, M: int = 10) -> None:
        super().__init__(basis, output_size, M)
    
    def initialize_priors(self) -> None:
        super().initialize_priors()
        self.prior["ranef_sigma"] = InverseGamma(10.0, 2.0)
    def sample_basis(self, dim):
        return super().sample_basis(dim)
    
    def model_fn(self, covariate_X, data_set) -> None:
        num_players, covariate_dim = covariate_X.shape
        num_outputs = len(data_set)
        intercept = sample("intercept", Normal(0, 5), sample_shape =  (1, num_outputs))
        ranef_sigma = sample("ranef_sigma", self.prior["ranef_sigma"], sample_shape=(1, num_outputs))
        ranef_intercept = sample("ranef_intercept_raw", Normal(0, 1), sample_shape=(num_players, self.basis.shape[0] , num_outputs))
        basis = self.sample_basis(covariate_dim)
        mu = intercept + jnp.einsum("...i,ijk -> ...jk",covariate_X, basis) + ranef_intercept * ranef_sigma
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure =  exposure_data[mask].flatten()
            if output == "gaussian":
                sd = sample(f"sigma_{metric}", Exponential(1.0))
                ## likelihood
                y = sample(f"likelihood_{metric}", Normal( loc = mu[:,:,index][mask].flatten(), scale = sd / exposure),  obs=output_data[mask].flatten())
            
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson(jnp.exp(mu[:,:,index][mask].flatten() + exposure)) , obs = output_data[mask].flatten())
            
            elif output == "binomial":
                y = sample(f"likelihod_{metric}", Binomial(logits =  mu[:,:,index][mask].flatten(), total_count = exposure), obs=output_data[mask].flatten())

    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)
    
class NBAFDALatentModel(NBAFDAModel):
    def __init__(self, basis, output_size: int = 1, M: int = 10, latent_dim1: int = 100, latent_dim2: int = 2) -> None:
        super().__init__(basis, output_size, M)
        self.latent_dim = (latent_dim1, latent_dim2)

    def initialize_priors(self) -> None:
        return super().initialize_priors()

    def sample_basis(self, dim):
        return super().sample_basis(dim)
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)
    
    def sample_latent(self):
        ### samples  from uniform distribution on steifel 
        X = sample("latent_raw", Normal(0, 1), sample_shape=self.latent_dim)
        XtX = jnp.matmul(X.T, X)
        U, L, V = jnp.linalg.svd(XtX, full_matrices = False)
        return jnp.matmul(X, jnp.matmul(U * jnp.power(L, -.5), V))

    def model_fn(self, data_set) -> None:
        covariate_dim = self.latent_dim[1]
        num_outputs = len(data_set)
        intercept = sample("intercept", Normal(0, 5), sample_shape =  (1, num_outputs))
        covariate_X = self.sample_latent()
        basis = self.sample_basis(covariate_dim)
        mu = intercept + jnp.einsum("...i,ijk -> ...jk",covariate_X, basis)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure =  exposure_data[mask].flatten()
            if output == "gaussian":
                sd = sample(f"sigma_{metric}", Exponential(1.0))
                ## likelihood
                y = sample(f"likelihood_{metric}", Normal( loc = mu[:,:,index][mask].flatten(), scale = sd / exposure),  obs=output_data[mask].flatten())
            
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson(jnp.exp(mu[:,:,index][mask].flatten() + exposure)) , obs = output_data[mask].flatten())
            
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits =  mu[:,:,index][mask].flatten(), total_count = exposure), obs=output_data[mask].flatten())



