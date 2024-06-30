from abc import abstractmethod, ABC
import jax 
import numpy as np
from numpyro import sample 
from numpyro import deterministic
from numpyro.distributions import HalfNormal, InverseGamma, Normal, Exponential, Poisson, Binomial, Dirichlet, MultivariateNormal, ZeroInflatedPoisson, Categorical, MixtureGeneral, Delta
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation
from optax import linear_onecycle_schedule, adam
import jax.numpy as jnp
import jax.scipy as jsc
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
    
    @abstractmethod
    def predict(self, *args, **kwargs):
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

    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]


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
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)
    
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
    
    def _stabilize_x(self, X):
        """Fix the rotation according to the SVD.
        """
        U, _, _ = jnp.linalg.svd(X, full_matrices=False)
        L       = jnp.linalg.cholesky(jnp.cov(U.T) + 1e-6 * jnp.eye(self.latent_dim[1])).T
        aligned_X  = jnp.linalg.solve(L, U.T).T
        return aligned_X / jnp.std(X, axis=0)

    def sample_latent(self):
        ### samples  from distribution on steifel 
        X = sample("latent_raw", Normal(), sample_shape=self.latent_dim)
        return self._stabilize_x(X)

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

    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)

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

    @abstractmethod
    def predict(self, *args, **kwargs):
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

    def run_inference(self, num_steps):
        svi = SVI(self.model_fn, AutoDelta(self.model_fn), optim=adam(learning_rate=linear_onecycle_schedule(100, .5)), loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps)
        return result
    
    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]

class ProbabilisticCPDecomposition(ABC):
    """
    model for probabilistic cp decomposition with normal output 
    """
    def __init__(self, X, rank: int, *args, **kwargs) -> None:
        self.X = X 
        self.r = rank 
        self.n, self.t, self.k = self.X.shape
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
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

class NBAMixedOutputProbabilisticCPDecomposition(ProbabilisticCPDecomposition):

    def __init__(self, X, rank: int, M, E, O, output_name: list, feature_name: list, time_trend = None) -> None:
        super().__init__(X, rank)
        assert (self.X.shape == E.shape) & (self.X.shape == M.shape) & (self.X.shape == O.shape) ### have to have same shape
        self.exposure = E ### exposure for each element in X
        self.output = O ### type of output for each element in X (1 gaussian, 2 poisson, 3 binomial)
        self.missing = M ### matrix of 1 / 0 indicating missing or not
        self.outputs = output_name ### ex: [gaussian, gaussian, poisson, ...]
        self.features = feature_name ## ex: [obpm, minutes, stl, ...]
        self.num_gaussian = 0
        self.gaussian_variance_indices = np.zeros_like(self.X, dtype=int)
        for i, output_type in enumerate(self.outputs):
            if output_type == "gaussian":
                self.num_gaussian += 1
                self.gaussian_variance_indices[..., i] = self.num_gaussian - 1
        self.gaussian_variance_indices = jnp.array(self.gaussian_variance_indices)
        self.gaussian_indices = (self.output == 1) & self.missing
        self.poisson_indices = (self.output == 2) & self.missing
        self.binomial_indices = (self.output == 3) & self.missing
        self.exponential_indices = (self.output == 4) & self.missing
        self.time_trend = jnp.zeros_like(X) if time_trend is None else time_trend

    
    def initialize_priors(self) -> None:
        ### initialize sigma
        self.prior["sigma"] = InverseGamma(10.0, 2.0)
        ### initialize U
        self.prior["U"] = Normal()
        ### initialize V
        self.prior["V"] = Normal()
        ### initialize W
        self.prior["W"] = Normal()
        ### initialize lambda
        self.prior["lambda"] = Dirichlet(concentration=jnp.ones(shape=(self.r,)) / self.r)
        ### initialize alpha
        self.prior["alpha"] = Normal()

    def model_fn(self) -> None:
        alpha = sample("alpha", self.prior["alpha"], sample_shape=(self.t, self.k)) ### mean across the time points and the metrics
        V = sample("V", self.prior["V"], sample_shape=(self.t, self.r))
        U = sample("U", self.prior["U"], sample_shape=(self.n, self.r ))
        W = sample("W", self.prior["W"], sample_shape=(self.k, self.r))
        sigma = sample("sigma", self.prior["sigma"], sample_shape=(self.num_gaussian,))
        weights = sample("lambda", self.prior["lambda"])
        core = jnp.einsum("ntr, kr->ntkr", jnp.einsum("nr, tr -> ntr", U, V), W)
        y = jnp.einsum("ntkr, r -> ntk", core, weights) + alpha


        y_normal = sample("likelihood_normal", Normal(loc = y[self.gaussian_indices].flatten() + self.time_trend[self.gaussian_indices].flatten(), 
                                                      scale=sigma[self.gaussian_variance_indices[self.gaussian_indices].flatten()]/self.exposure[self.gaussian_indices].flatten()), 
                          obs = self.X[self.gaussian_indices].flatten())
        y_poisson = sample("likelihood_poisson", Poisson(rate = jnp.exp(y[self.poisson_indices].flatten() + self.exposure[self.poisson_indices].flatten()  + self.time_trend[self.poisson_indices].flatten())), 
                           obs=self.X[self.poisson_indices].flatten())
        y_binomial = sample("likelihood_binomial", Binomial(total_count=self.exposure[self.binomial_indices].flatten(), 
                                                            logits = y[self.binomial_indices].flatten()  + self.time_trend[self.binomial_indices].flatten() ), 
                                                            obs=self.X[self.binomial_indices].flatten())
        y_exponential = sample("likelihood_exponential",Exponential(rate = jnp.exp(y[self.exponential_indices].flatten()  + self.time_trend[self.exponential_indices].flatten())), obs = self.X[self.exponential_indices].flatten())

    def run_inference(self, num_steps):
        svi = SVI(self.model_fn, AutoDelta(self.model_fn), optim=adam(learning_rate=linear_onecycle_schedule(100, .5)), loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps)
        return result

    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]
    


class NBAMixedOutputProbabilisticCPDecompositionMultiWay(ProbabilisticCPDecomposition):

    def __init__(self, X, rank: int, M, E, O, output_name: list, feature_name: list) -> None:
        self.X = X 
        self.r = rank 
        self.n, self.y, self.t, self.k = self.X.shape
        self.prior = {}
        assert (self.X.shape == E.shape) & (self.X.shape == M.shape) & (self.X.shape == O.shape) ### have to have same shape
        self.exposure = E ### exposure for each element in X
        self.output = O ### type of output for each element in X (1 gaussian, 2 poisson, 3 binomial)
        self.missing = M ### matrix of 1 / 0 indicating missing or not
        self.outputs = output_name ### ex: [gaussian, gaussian, poisson, ...]
        self.features = feature_name ## ex: [obpm, minutes, stl, ...]
        self.num_gaussian = 0
        self.gaussian_variance_indices = np.zeros_like(self.X, dtype=int)
        for i, output_type in enumerate(self.outputs):
            if output_type == "gaussian":
                self.num_gaussian += 1
                self.gaussian_variance_indices[..., i] = self.num_gaussian - 1
        self.gaussian_variance_indices = jnp.array(self.gaussian_variance_indices)
        self.gaussian_indices = (self.output == 1) & self.missing
        self.poisson_indices = (self.output == 2) & self.missing
        self.binomial_indices = (self.output == 3) & self.missing

    
    def initialize_priors(self) -> None:
        ### initialize sigma
        self.prior["sigma"] = InverseGamma(10.0, 2.0)
        ### initialize U (latent players)
        self.prior["U"] = Normal()
        ### initialize V (latent age trend )
        self.prior["V"] = Normal()
        ### initialize W  (latent metric trend)
        self.prior["W"] = Normal()
        ### initialize Z (latent year trend)
        self.prior["Z"] = Normal()
        ### initialize lambda
        self.prior["lambda"] = Dirichlet(concentration=jnp.ones(shape=(self.r,)) / self.r)

    def model_fn(self) -> None:
        Z = sample("Z", self.prior["Z"], sample_shape=(self.y, self.r))
        V = sample("V", self.prior["V"], sample_shape=(self.t, self.r))
        U = sample("U", self.prior["U"], sample_shape=(self.n, self.r ))
        W = sample("W", self.prior["W"], sample_shape=(self.k, self.r))
        sigma = sample("sigma", self.prior["sigma"], sample_shape=(self.num_gaussian,))
        weights = sample("lambda", self.prior["lambda"])
        core = jnp.einsum("yr,ntkr ->nytkr", Z, jnp.einsum("ntr, kr->ntkr", jnp.einsum("nr, tr -> ntr", U, V), W))
        y = jnp.einsum("nytkr, r -> nytk", core, weights) 
        y_normal = sample("likelihood_normal", Normal(loc = y[self.gaussian_indices].flatten(), 
                                                      scale=sigma[self.gaussian_variance_indices[self.gaussian_indices].flatten()]/self.exposure[self.gaussian_indices].flatten()), 
                          obs = self.X[self.gaussian_indices].flatten())
        y_poisson = sample("likelihood_poisson", Poisson(rate = jnp.exp(y[self.poisson_indices].flatten() + self.exposure[self.poisson_indices].flatten())), 
                           obs=self.X[self.poisson_indices].flatten())
        y_binomial = sample("likelihood_binomial", Binomial(total_count=self.exposure[self.binomial_indices].flatten(), 
                                                            logits = y[self.binomial_indices].flatten()), 
                                                            obs=self.X[self.binomial_indices].flatten())

    def run_inference(self, num_steps):
        svi = SVI(self.model_fn, AutoDelta(self.model_fn), optim=adam(learning_rate=linear_onecycle_schedule(100, .5)), loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps)
        return result

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)


class RFLVMBase(ABC):
    """ 
    HMC implementation of 
    https://arxiv.org/pdf/2006.11145
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        self.r = latent_rank 
        self.m = rff_dim
        self.n, self.j = output_shape
        self.prior = {}
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["W"] = Normal()
        self.prior["beta"] = Normal()
        self.prior["X"] = Normal()
        self.prior["sigma"] = InverseGamma(1, 1)
    
    def _stabilize_x(self, X):
        """Fix the rotation according to the SVD.
        """
        U, _, _ = jnp.linalg.svd(X, full_matrices=False)
        L       = jnp.linalg.cholesky(jnp.cov(U.T) + 1e-6 * jnp.eye(self.r)).T
        aligned_X  = jnp.linalg.solve(L, U.T).T
        return aligned_X / jnp.std(X, axis=0)

    @abstractmethod
    def model_fn(self, data_set) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X_raw = sample("X_raw", self.prior["X"], sample_shape=(self.n, self.r))
        X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()
            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() , sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson( jnp.exp(mu[index,:,:][mask].flatten() + exposure_) ), obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten(), total_count=exposure_), obs = Y_)
            
    @abstractmethod
    def run_inference(self, num_steps, model_args):
        svi = SVI(self.model_fn, AutoLaplaceApproximation(self.model_fn), optim=adam(learning_rate=linear_onecycle_schedule(100, .5)), loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps, **model_args)
        return result

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]



class RFLVM(RFLVMBase):

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    def run_inference(self, num_steps, model_args):
        return super().run_inference(num_steps, model_args)
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)



    
class TVRFLVM(RFLVM):
    """
    model for time varying functional 
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis ### basis for time dimension
    

    def make_kernel(self, jitter = 1e-6):
        deltaXsq = jnp.power((self.basis[:, None] - self.basis), 2.0)
        k = jnp.exp(-0.5 * deltaXsq) + jitter * jnp.eye(self.basis.shape[0])
        return k

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        kernel = self.make_kernel()
        self.prior["beta"] = MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel) ### basic gp prior on the time 


    def model_fn(self, data_set) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X_raw = sample("X_raw", self.prior["X"], sample_shape=(self.n, self.r))
        X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m)) ### don't need extra dimension
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()

            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() , sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson( jnp.exp(mu[index,:,:][mask].flatten() + exposure_) ), obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten(), total_count=exposure_), obs = Y_)
    
    def run_inference(self, num_steps, model_args):
        return super().run_inference(num_steps, model_args)

    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)


class DriftRFLVM(RFLVMBase):

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, drift_basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.drift_basis = drift_basis 
        self.L =  jnp.max(jnp.abs(drift_basis), axis=0)*1.5
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)

    def sample_basis(self, dim):
        return deterministic("drift_basis", jnp.vstack([approx_se_ncp(self.drift_basis, alpha=1.0, length=1.0, M = self.m, output_size=1,
                             L =  self.L, name= f"drift_{i}") for i in range(dim)] ))

    def model_fn(self, data_set) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X_raw = sample("X_raw", self.prior["X"], sample_shape=(self.n, self.r))
        X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m, self.j))
        drift = self.sample_basis(len(data_set))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            time = data_entity["time"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()
            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() + drift[..., index][time[mask].flatten()], sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson( jnp.exp(mu[index,:,:][mask].flatten() + exposure_ + drift[..., index][time[mask].flatten()])) , obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten() + drift[..., index][time[mask].flatten()], total_count=exposure_), obs = Y_)
            
    def run_inference(self, num_steps, model_args):
        return super().run_inference(num_steps, model_args)
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)


class DriftTVRFLVM(RFLVMBase):


    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis, drift_basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis ### basis for time dimension
        self.drift_basis = drift_basis 
        self.L =  jnp.max(jnp.abs(drift_basis), axis=0)*1.5
    

    def make_kernel(self, jitter = 1e-6):
        deltaXsq = jnp.power((self.basis[:, None] - self.basis), 2.0)
        k = jnp.exp(-0.5 * deltaXsq) + jitter * jnp.eye(self.basis.shape[0])
        return k

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        kernel = self.make_kernel()
        self.prior["beta"] = MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel) ### basic gp prior on the time


    def sample_basis(self, dim):
        return deterministic("drift_basis", jnp.vstack([approx_se_ncp(self.drift_basis, alpha=1.0, length=1.0, M = self.m, output_size=1,
                             L =  self.L, name= f"drift_{i}") for i in range(dim)] ))

    def model_fn(self, data_set) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X_raw = sample("X_raw", self.prior["X"], sample_shape=(self.n, self.r))
        X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m))
        drift = self.sample_basis(len(data_set))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            time = data_entity["time"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()
            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() + drift[..., index][time[mask].flatten()], sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson( jnp.exp(mu[index,:,:][mask].flatten() + exposure_ + drift[..., index][time[mask].flatten()])) , obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten() + drift[..., index][time[mask].flatten()], total_count=exposure_), obs = Y_)
            
    def run_inference(self, num_steps, model_args):
        return super().run_inference(num_steps, model_args)
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)


class FixedRFLVMBase(ABC):
    """ 
    HMC implementation of 
    https://arxiv.org/pdf/2006.11145
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        self.r = latent_rank 
        self.m = rff_dim
        self.n, self.j = output_shape
        self.prior = {}
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["beta"] = Normal()
        self.prior["sigma"] = InverseGamma(1.0, 1.0)
        self.prior["W"] = Normal()

    @abstractmethod
    def model_fn(self, data_set, X) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()
            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() , sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson(rate = jnp.exp(mu[index,:,:][mask].flatten() + exposure_) ), obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten(), total_count=exposure_), obs = Y_)
            elif output == "exponential":
                y = sample(f"likelihood_{metric}", Exponential(rate = jnp.exp(mu[index,:,:][mask].flatten())), obs = Y_)

            
    @abstractmethod
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
    
    @abstractmethod
    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]

class FixedRFLVM(FixedRFLVMBase):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    def model_fn(self, data_set, X) -> None:
        return super().model_fn(data_set, X)
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)

class FixedTVRFLVM(FixedRFLVM):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis
    
    def make_kernel(self, jitter = 1e-6):
        deltaXsq = jnp.power((self.basis[:, None] - self.basis), 2.0)
        k = jnp.exp(-0.5 * deltaXsq) + jitter * jnp.eye(self.basis.shape[0])
        return k
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        kernel = self.make_kernel()
        self.prior["beta"] = MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel) ### basic gp prior on the time 
    
    def model_fn(self, data_set, X) -> None:
        W = sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = sample(f"beta", self.prior["beta"], sample_shape=(len(data_set), 2 * self.m)) ### don't need extra dimension
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        for index, data_entity in enumerate(data_set):
            output = data_entity["output"]
            metric = data_entity["metric"]
            mask = data_entity["mask"]
            exposure_data = data_entity["exposure_data"]
            output_data = data_entity["output_data"]
            exposure_ =  exposure_data[mask].flatten()

            Y_ = output_data[mask].flatten() ### non missing values
            if output == "gaussian":
                sigma = sample(f"sigma_{metric}", self.prior["sigma"])
                y = sample(f"likelihood_{metric}", Normal( mu[index,:,:][mask].flatten() , sigma/exposure_), obs=Y_)
            elif output == "poisson":
                y = sample(f"likelihood_{metric}", Poisson( jnp.exp(mu[index,:,:][mask].flatten() + exposure_) ), obs=Y_)
            elif output == "binomial":
                y = sample(f"likelihood_{metric}", Binomial(logits=mu[index,:,:][mask].flatten(), total_count=exposure_), obs = Y_)
            elif output == "exponential":
                y = sample(f"likelihood_{metric}", Exponential(rate = jnp.exp(mu[index,:,:][mask].flatten())), obs = Y_)
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)