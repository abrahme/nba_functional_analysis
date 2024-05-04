from abc import abstractmethod, ABC
import jax 
import numpy as np
from numpyro import sample 
from numpyro import deterministic
from numpyro.distributions import HalfNormal, InverseGamma, Normal, Exponential, Poisson, Binomial
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, initialization
from numpyro.infer.autoguide import AutoDelta
from optax import linear_onecycle_schedule, adam
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



class ProbabilisticPCA(ABC):
    """
    model for probabilistic pca with normal output 
    """
    def __init__(self, X, rank: int, *args, **kwargs) -> None:
        self.X = X 
        self.r = rank 
        self.n, self.p = self.X.shape
        self.prior = {}
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def model_fn(self, *args, **kwargs) -> None:
        raise NotImplementedError 

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        raise NotImplementedError


class NBAMixedOutputProbabilisticPCA(ProbabilisticPCA):
    """
    model for probabilistic pca with mixed output + missing data
    """

    def __init__(self, X, rank: int, M, E, O, output_name: list, feature_name: list) -> None:
        super().__init__(X, rank)
        assert (self.X.shape == E.shape) & (self.X.shape == M.shape) & (self.X.shape == O.shape) ### have to have same shape
        self.exposure = E ### exposure for each element in X
        self.output = O ### type of output for each element in X (1 gaussian, 2 poisson, 3 binomial)
        self.missing = M ### matrix of 1 / 0 indicating missing or not
        self.T = int(self.X.shape[1] // len(feature_name)) ### time steps
        self.outputs = output_name ### ex: [gaussian, gaussian, poisson, ...]
        self.features = feature_name ## ex: [obpm, minutes, stl, ...]
        self.num_gaussian = 0
        self.gaussian_variance_indices = np.zeros_like(self.X, dtype=int)
        for i, output_type in enumerate(self.outputs):
            if output_type == "gaussian":
                self.num_gaussian += 1
                self.gaussian_variance_indices[:, i*self.T: (i+1)*self.T] = self.num_gaussian - 1
        self.gaussian_variance_indices = jnp.array(self.gaussian_variance_indices)
        self.gaussian_indices = (self.output == 1) & self.missing
        self.poisson_indices = (self.output == 2) & self.missing
        self.binomial_indices = (self.output == 3) & self.missing
    
    def initialize_priors(self) -> None:
        ### initialize sigma
        self.prior["sigma"] = InverseGamma(10.0, 2.0)
        ### initialize alpha
        self.prior["alpha"] = Normal()

        ### initialize Beta
        self.prior["beta"] = Normal()

        ### initialize W
        self.prior["W"] = Normal()
    
    def model_fn(self) -> None:
        """
        gaussian_index: array of 1, 2, ... d of length p indicating which indices go for gaussian 
        gaussian_scale: n x p array of 1 / value indicating how much to divide gaussian obs by. mssing val gets 1
        """
        alpha = sample("alpha", self.prior["alpha"], sample_shape=(self.p,))
        beta = sample("beta", self.prior["beta"], sample_shape=(self.r, self.p))
        W = sample("W", self.prior["W"], sample_shape=(self.n, self.r))
        sigma = sample("sigma", self.prior["sigma"], sample_shape=(self.num_gaussian,))
        y = jnp.matmul(W, beta) + jnp.outer(jnp.ones((self.n,)), alpha)
        y_normal = sample("likelihood_normal", Normal(loc = y[self.gaussian_indices].flatten(), 
                                                      scale=sigma[self.gaussian_variance_indices[self.gaussian_indices].flatten()]/self.exposure[self.gaussian_indices].flatten()), 
                          obs = self.X[self.gaussian_indices].flatten())
        y_poisson = sample("likelihood_poisson", Poisson(rate = jnp.exp(y[self.poisson_indices].flatten() + self.exposure[self.poisson_indices].flatten())), 
                           obs=self.X[self.poisson_indices].flatten())
        y_binomial = sample("likelihood_binomial", Binomial(total_count=self.exposure[self.binomial_indices].flatten(), 
                                                            logits = y[self.binomial_indices].flatten()), 
                                                            obs=self.X[self.binomial_indices].flatten())

    def run_inference(self, num_warmup, num_samples, num_chains):
        mcmc = MCMC(
        NUTS(self.model_fn, init_strategy=init_to_median),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        chain_method="parallel"
    )
        mcmc.run(jax.random.PRNGKey(0))
        return mcmc
        





        