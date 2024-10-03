from abc import abstractmethod, ABC
import jax 
import numpy as np
from numpyro import sample 
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, Binomial, Dirichlet, MultivariateNormal, Distribution
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta
from optax import linear_onecycle_schedule, adam
import jax.numpy as jnp
from .MultiHMCGibbs import MultiHMCGibbs

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

    def __init__(self, X, rank: int, M, E, O, output_name: list, feature_name: list) -> None:
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
        self.gate_index = feature_name.index("retirement")
        self.gaussian_variance_indices = jnp.array(self.gaussian_variance_indices)
        self.gaussian_indices = (self.output == 1) & self.missing
        self.poisson_indices = (self.output == 2) & self.missing
        self.binomial_indices = (self.output == 3) & self.missing
        self.exponential_indices = (self.output == 4) & self.missing

    
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
        y_normal = sample("likelihood_normal", Normal(loc = y[self.gaussian_indices].flatten(), 
                                                      scale=sigma[self.gaussian_variance_indices[self.gaussian_indices].flatten()]/self.exposure[self.gaussian_indices].flatten()), 
                          obs = self.X[self.gaussian_indices].flatten())
        y_poisson = sample("likelihood_poisson", Poisson(rate = jnp.exp(y[self.poisson_indices].flatten() + self.exposure[self.poisson_indices].flatten())), 
                           obs=self.X[self.poisson_indices].flatten())
        y_binomial = sample("likelihood_binomial", Binomial(total_count=self.exposure[self.binomial_indices].flatten(), 
                                                            logits = y[self.binomial_indices].flatten()), 
                                                            obs=self.X[self.binomial_indices].flatten())
        y_exponential = sample("likelihood_exponential",Exponential(rate = jnp.exp(y[self.exponential_indices].flatten())), obs = self.X[self.exponential_indices].flatten())

    def run_inference(self, num_steps):
        svi = SVI(self.model_fn, AutoDelta(self.model_fn), optim=adam(learning_rate=linear_onecycle_schedule(100, .5)), loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps)
        return result

    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]
    

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
        self.prior["X"] = Normal(loc=jnp.zeros((self.n, self.r)))
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
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        # X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(num_metrics, 2 * self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
        expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        for family in data_set:
            linear_predictor = mu[data_set[family]["indices"]]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                dist = Normal(linear_predictor[mask], expanded_sigmas[mask]/exposure[mask])
            elif family == "poisson":
                dist = Poisson(jnp.exp(linear_predictor[mask] + exposure[mask])) 
            elif family == "binomial":
                dist = Binomial(logits = linear_predictor[mask], total_count=exposure[mask])
            elif family == "exponential":
                dist = Exponential(jnp.exp(linear_predictor[mask] + exposure[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask])
    @abstractmethod
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        n_parallel = jax.local_device_count()
        n_vectorized = num_chains // n_parallel
        def do_mcmc(rng_key):
            mcmc = MCMC(
            NUTS(self.model_fn, init_strategy=init_to_median),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=n_vectorized,
            progress_bar=False,
            chain_method="vectorized")
            mcmc.run(rng_key, **model_args)
            return {**mcmc.get_samples()}
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_parallel)
        traces = jax.pmap(do_mcmc)(rng_keys)
        return {k: jnp.concatenate(v) for k, v in traces.items()}
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
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)

    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)



    
class TVRFLVM(RFLVM):
    """
    model for time varying functional 
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis ### basis for time dimension
    

    def make_kernel(self, lengthscale, jitter = 1e-6):
        deltaXsq = jnp.power((self.basis[:, None] - self.basis), 2.0)
        k = jnp.exp(-0.5 * deltaXsq / lengthscale) + jitter * jnp.eye(self.basis.shape[0])
        return k

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(1.0, 1.0)


    def model_fn(self, data_set) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m, self.r))

        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        # X = self._stabilize_x(X_raw)
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))

        ls = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        kernel = self.make_kernel(ls)
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta",  MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel), sample_shape=(num_metrics, 2 * self.m)) ### don't need extra dimension
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
        expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        for family in data_set:
            linear_predictor = mu[data_set[family]["indices"]]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                dist = Normal(linear_predictor[mask], expanded_sigmas[mask] /exposure[mask])
            elif family == "poisson":
                dist = Poisson(jnp.exp(linear_predictor[mask] + exposure[mask])) 
            elif family == "binomial":
                dist = Binomial(logits = linear_predictor[mask], total_count=exposure[mask])
            elif family == "exponential":
                dist = Exponential(jnp.exp(linear_predictor[mask] + exposure[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask])
    def run_inference(self, num_warmup, num_samples, num_chains, model_args):
        return super().run_inference(num_warmup, num_samples, num_chains, model_args)

    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)

class GibbsRFLVM(RFLVM):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = []):
        n_parallel = jax.local_device_count()
        n_vectorized = num_chains // n_parallel
        inner_kernels = [NUTS(self.model_fn) for _ in range(len(gibbs_sites))]
        outer_kernel = MultiHMCGibbs(inner_kernels, gibbs_sites_list=gibbs_sites)
        def do_mcmc(rng_key):
            mcmc = MCMC(
            outer_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=n_vectorized,
            progress_bar=False,
            chain_method="vectorized")
            mcmc.run(rng_key, **model_args)
            return {**mcmc.get_samples()}
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_parallel)
        traces = jax.pmap(do_mcmc)(rng_keys)
        return {k: jnp.concatenate(v) for k, v in traces.items()}
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)
    

class GibbsTVRFLVM(TVRFLVM):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = []):
        n_parallel = jax.local_device_count()
        n_vectorized = num_chains // n_parallel
        inner_kernels = [NUTS(self.model_fn) for _ in range(len(gibbs_sites))]
        outer_kernel = MultiHMCGibbs(inner_kernels, gibbs_sites_list=gibbs_sites)
        def do_mcmc(rng_key):
            mcmc = MCMC(
            outer_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=n_vectorized,
            progress_bar=False,
            chain_method="vectorized")
            mcmc.run(rng_key, **model_args)
            return {**mcmc.get_samples()}
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_parallel)
        traces = jax.pmap(do_mcmc)(rng_keys)
        return {k: jnp.concatenate(v) for k, v in traces.items()}
        
    
    def predict(self, posterior_samples: dict, model_args):
        return super().predict(posterior_samples, model_args)
    

        