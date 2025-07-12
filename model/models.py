from abc import abstractmethod, ABC
import jax 
import numpyro
import numpy as np
from jax.flatten_util import ravel_pytree
from jax import hessian
from scipy.linalg import cho_solve, cho_factor, cholesky
from numpyro.infer.util import log_density
from numpyro import sample 
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, Binomial, Dirichlet, MultivariateNormal, LKJ, BetaProportion, Distribution, Uniform, BetaBinomial, Gamma, BinomialLogits
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, TraceEnum_ELBO, Predictive, init_to_value, init_to_feasible, init_to_uniform, init_to_mean
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoNormal, AutoLowRankMultivariateNormal, AutoLaplaceApproximation
import optax
from optax import linear_onecycle_schedule, adam
from numpyro.optim import optax_to_numpyro
from jaxopt import LBFGS
from .hsgp import make_convex_f, make_psi_gamma, diag_spectral_density, make_convex_phi,  vmap_make_convex_phi, vmap_make_convex_phi_prime
import jax.numpy as jnp
import jax.scipy as jsci
from .MultiHMCGibbs import MultiHMCGibbs
from .model_utils import Type2Gumbel


def step_decay_schedule(init_lr, drop_every=10000, drop_factor=10, total_steps=100000):
    num_drops = total_steps // drop_every
    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(init_lr / (drop_factor ** i))
            for i in range(num_drops + 1)
        ],
        boundaries=[drop_every * i for i in range(1, num_drops + 1)]
    )


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
                dist = Binomial(logits = linear_predictor[mask], total_count=exposure[mask].astype(int))
            elif family == "exponential":
                dist = Exponential(jnp.exp(linear_predictor[mask] + exposure[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask])
    @abstractmethod
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized:bool, model_args, initial_values = {}):
        kernel = NUTS(self.model_fn, init_strategy=init_to_value(values=initial_values))
        key = jax.random.PRNGKey(0)
        if vectorized:
            n_parallel = jax.local_device_count()
            n_vectorized = num_chains // n_parallel
            def do_mcmc(rng_key):
                mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=n_vectorized,
                progress_bar=False,
                chain_method="vectorized")
                mcmc.run(rng_key, **model_args)
                return {**mcmc.get_samples()}
            rng_keys = jax.random.split(key, n_parallel)
            traces = jax.pmap(do_mcmc)(rng_keys)
            return {k: jnp.concatenate(v) for k, v in traces.items()}
        else:
            mcmc = MCMC(kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains = num_chains,
                        chain_method="parallel")
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True)
    @abstractmethod
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}, sample_shape = (4, 2000)):
        guide = AutoNormal(self.model_fn, prefix="", **guide_kwargs, init_loc_fn=init_to_value(values=initial_values),
                                   init_scale= 1e-10)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True,  **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples
    
    @abstractmethod
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result.params
    
    

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args, num_samples = 1000):
        predictive = Predictive(self.model_fn, posterior_samples,  num_samples=num_samples)
        return predictive(jax.random.PRNGKey(0), **model_args)



class RFLVM(RFLVMBase):

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)

    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_values)

    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)



    
class TVRFLVM(RFLVM):
    """
    model for time varying functional 
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis ### basis for time dimension
        self.t = len(basis)
    

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
                dist = Binomial(logits = linear_predictor[mask], total_count=exposure[mask].astype(int))
            elif family == "exponential":
                dist = Exponential(jnp.exp(linear_predictor[mask] + exposure[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask])

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)

    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_values)



    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexTVRFLVM(TVRFLVM):
    """
    model for time varying functional enforcing convexity in the shape parameters
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale_deriv"] = InverseGamma(2.0, 1.0)
        self.prior["lengthscale"] = InverseGamma(.3, .7)
        self.prior["sigma_beta"] = InverseGamma(2.0, 1.0)
        self.prior["sigma"] = InverseGamma(299.0, 6000.0)
        self.prior["alpha"] = InverseGamma(1.0, 1.0)
        self.prior["intercept"] = Normal()
        self.prior["slope"] = Normal()


    def model_fn(self, data_set, hsgp_params, offsets = 0, prior = False) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        X -= jnp.mean(X, keepdims = True, axis = 0)
        X /= jnp.std(X, keepdims = True, axis = 0)
        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale)[None])
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))

     
        slope =  make_psi_gamma(psi_x, self.prior["slope"] if not isinstance(self.prior["slope"], Distribution) else sample("slope", self.prior["slope"], sample_shape=(self.m*2, num_metrics))) 
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(num_metrics,))
        intercept = make_psi_gamma(psi_x, self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"] , sample_shape=(2 * self.m, num_metrics))) 
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights = weights * spd * .0001
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
        mu = make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None]) if not prior else numpyro.deterministic("mu", make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None]))
        if num_gaussians > 0 :
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])

        for family in data_set:
            linear_predictor = mu[data_set[family]["indices"]] 
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                dist = Normal(linear_predictor[mask], expanded_sigmas[mask] /exposure[mask])
            elif family == "poisson":
                rate = jnp.exp(linear_predictor[mask] + exposure[mask])
                dist = Poisson(rate) 
            elif family == "binomial":
                dist = BinomialLogits(logits = linear_predictor[mask], total_count=exposure[mask].astype(int))
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor[mask])
                dist = BetaProportion(rate, exposure[mask] * sigma_beta)
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn = init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                          num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)



    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)

class ConvexMaxTVRFLVM(ConvexTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_beta"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_c"] = InverseGamma(2.0, 3)
        self.prior["sigma_t"] = InverseGamma(2.0, .3)
        self.prior["alpha"] = InverseGamma(100.0, .0003)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()

    def model_fn(self, data_set, hsgp_params, offsets = {}, prior = False) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights =  weights * spd 
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        # X -= jnp.mean(X, keepdims = True, axis = 0)
        # X /= jnp.std(X, keepdims = True, axis = 0)
        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, num_metrics))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, num_metrics))
        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offsets["t_max"])
        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, num_metrics))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(num_metrics,))
        c_max = make_psi_gamma(psi_x, self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, num_metrics))) * sigma_c_max + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        for family in data_set:
            linear_predictor = mu[data_set[family]["indices"]] 
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]

            if family == "gaussian":
                rate = linear_predictor[mask]
                dist = Normal(rate, expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                rate = jnp.exp(linear_predictor[mask] + exposure[mask])
                dist = Poisson(rate) 
            elif family == "binomial":
                rate = linear_predictor[mask]
                dist = BinomialLogits(logits = rate, total_count=exposure[mask].astype(int))
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor[mask])
                dist = BetaProportion(rate, exposure[mask] * sigma_beta)
            
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn = init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params=initial_values, stable_update=True, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexMaxBoundaryTVRFLVM(ConvexTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        and boundary conditions
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_beta"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_beta_binomial"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_c"] = InverseGamma(2.0, .3)
        self.prior["sigma_boundary_r"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_boundary_l"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_t"] = InverseGamma(2.0, .3)
        self.prior["alpha"] = InverseGamma(100.0, .0003)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["boundary_l"] = Normal()
        self.prior["boundary_r"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior") -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(num_metrics,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        

        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, num_metrics))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, num_metrics))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, num_metrics))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, num_metrics))
        
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(1, num_metrics))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(1, num_metrics))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(2 * self.m, num_metrics))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(2 * self.m, num_metrics))
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        

        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5  + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw) * sigma_c_max + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw ) * sigma_boundary_r + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x,boundary_l_raw) * sigma_boundary_l + offsets["boundary_l"])
        
        if prior:
            boundary_l = numpyro.deterministic("boundary_l_", boundary_l)
            boundary_r = numpyro.deterministic("boundary_r_", boundary_r)

        boundary_diff = boundary_l - boundary_r
        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        if prior:
            G = (make_convex_phi(t_r, L_time, M_time) - make_convex_phi(t_0, L_time, M_time))[None, None] + (t_0 - t_r) * phi_prime_t_max
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, num_metrics) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0))
                dist = Poisson(rate).mask(mask)
            elif family == "binomial":
                rate = linear_predictor
                dist = BinomialLogits(logits = rate, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)

        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, 0.1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())              


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params = initial_values, stable_update=True, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs,init_loc_fn=init_to_value(values=initial_values))
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True,  **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    
    @staticmethod
    def solve_for_weights(A, rho, phi, X, b, spd):
        def loss_fn(A, rho, phi, X, b, s):
            A = A * s
            def residual(n, k):
                phi_nk = phi[n][None]  # shape (M,)
                A_k = A[..., k]          # (D, M)
                X_nk = X[n, k]      # (D, D)
                v = jnp.dot(phi_nk, A_k)    # (D,)

                return jnp.einsum("jd, dd, kd - > jk", v, X_nk, v) - b[n, k]

            # Vectorize residual over n and k
            r = jax.vmap(lambda n: jax.vmap(lambda k: residual(n, k))(jnp.arange(b.shape[-1])))(jnp.arange(b.shape[0]))
            penalty = jnp.sum(jnp.square(r))
            reg = jnp.sum(jnp.square(A))
            return reg + rho * penalty

        solver = LBFGS(fun=loss_fn, maxiter=500, implicit_diff=True, verbose=False)

        result = solver.run(A, rho, phi, X, b, spd)
        A_opt = result.params
        return A_opt

class ConvexMaxBoundaryARTVRFLVM(ConvexMaxBoundaryTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        and boundary conditions
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = InverseGamma(2.0, 3.0)
        self.prior["beta_ar"] = Normal()
    
    def make_kernel(self, sigma, rho, time_delta):
        return (1 / (1 - jnp.square(rho[..., None]))) * (rho[..., None] ** jnp.abs(time_delta[None])) * sigma[..., None] 
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior") -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(num_metrics,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        

        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, num_metrics))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, num_metrics))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, num_metrics))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, num_metrics))
        
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(1, num_metrics))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(1, num_metrics))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(2 * self.m, num_metrics))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(2 * self.m, num_metrics))
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        

        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 2  + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw) * sigma_c_max + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw ) * sigma_boundary_r + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x,boundary_l_raw) * sigma_boundary_l + offsets["boundary_l"])
        
        if prior:
            boundary_l = numpyro.deterministic("boundary_l_", boundary_l)
            boundary_r = numpyro.deterministic("boundary_r_", boundary_r)

        boundary_diff = boundary_l - boundary_r
        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        if prior:
            G = (make_convex_phi(t_r, L_time, M_time) - make_convex_phi(t_0, L_time, M_time))[None, None] + (t_0 - t_r) * phi_prime_t_max
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, num_metrics) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights *= spd

        time_delta = self.basis[None] - self.basis[..., None]
        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(num_metrics,1))

        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(num_metrics,1))

            
        kernel = self.make_kernel(sigma_ar, rho_ar, time_delta)
        L = jnp.linalg.cholesky(kernel)  # (K, T, T)

        # Standard normals: shape (K, N, T)
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(num_metrics, self.n, self.j))

        # Apply Cholesky: result (K, N, T)
        AR = jnp.einsum('ktd,knd->knt', L, z)

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x) 
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices] + AR[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0))
                dist = Poisson(rate).mask(mask)
            elif family == "binomial":
                rate = linear_predictor
                dist = BinomialLogits(logits = rate, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)


        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, 0.1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())   


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params = initial_values, stable_update=True, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):

        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs,init_loc_fn=init_to_value(values=initial_values))
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True,  **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    
    @staticmethod
    def solve_for_weights(A, rho, phi, X, b, spd):
        def loss_fn(A, rho, phi, X, b, s):
            A = A * s
            def residual(n, k):
                phi_nk = phi[n][None]  # shape (M,)
                A_k = A[..., k]          # (D, M)
                X_nk = X[n, k]      # (D, D)
                v = jnp.dot(phi_nk, A_k)    # (D,)

                return jnp.einsum("jd, dd, kd - > jk", v, X_nk, v) - b[n, k]

            # Vectorize residual over n and k
            r = jax.vmap(lambda n: jax.vmap(lambda k: residual(n, k))(jnp.arange(b.shape[-1])))(jnp.arange(b.shape[0]))
            penalty = jnp.sum(jnp.square(r))
            reg = jnp.sum(jnp.square(A))
            return reg + rho * penalty

        solver = LBFGS(fun=loss_fn, maxiter=500, implicit_diff=True, verbose=False)

        result = solver.run(A, rho, phi, X, b, spd)
        A_opt = result.params
        return A_opt


class ConvexMaxBoundaryKronTVRFLVM(ConvexMaxBoundaryTVRFLVM):
    def __init__(self, latent_rank_1: int, rff_dim: int, latent_rank_2: int,  num_metrics:int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank_1, rff_dim, output_shape, basis)
        self.r2 = latent_rank_2
        self.k = num_metrics
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["LKJ"] = LKJ(dimension = self.k, concentration=1)
    
    def make_kernel(self, corr_mat,  sigma):
        return jnp.einsum("nk, nj -> njk", sigma, sigma) * corr_mat
    
    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior") -> None:

        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_metrics = sum(len(data_set[family]["indices"]) for family in data_set)
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(num_metrics,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        
        LKJ_metric = self.prior["LKJ"] if not isinstance(self.prior["LKJ"], Distribution) else sample("LKJ", self.prior["LKJ"], sample_shape=(4, ))

        kernel = LKJ_metric
        cov_t_max, cov_c_max, cov_boundary_l, cov_boundary_r = kernel[0], kernel[1], kernel[2], kernel[3]


        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, num_metrics))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, num_metrics))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw",  MultivariateNormal(loc=jnp.zeros(self.k), covariance_matrix=cov_t_max), sample_shape=(2 * self.m, ))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max",  MultivariateNormal(loc=jnp.zeros(self.k), covariance_matrix=cov_c_max) , sample_shape=(2 * self.m, ))
        
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(1, num_metrics))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(1, num_metrics))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r",  MultivariateNormal(loc=jnp.zeros(self.k), covariance_matrix=cov_boundary_r) , sample_shape=(2 * self.m, ))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l",  MultivariateNormal(loc=jnp.zeros(self.k), covariance_matrix=cov_boundary_l) , sample_shape=(2 * self.m, ))
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        
        


        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5  + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5  + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw) * sigma_c_max + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw ) * sigma_boundary_r + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x,boundary_l_raw) * sigma_boundary_l + offsets["boundary_l"])
        
        if prior:
            boundary_l = numpyro.deterministic("boundary_l_", boundary_l)
            boundary_r = numpyro.deterministic("boundary_r_", boundary_r)

        boundary_diff = boundary_l - boundary_r
        
        
        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)




        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0))
                dist = Poisson(rate).mask(mask)
            elif family == "binomial":
                rate = linear_predictor
                dist = BinomialLogits(logits = rate, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor)
                dist = BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)

        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, 0.1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())  

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params = initial_values, stable_update=True, **model_args)
        return result.params
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)

class ConvexGP(TVRFLVM):
    def __init__(self, basis) -> None:
        self.basis = basis
        self.prior = {}
        self.j = basis.shape[-1]
    
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["beta"] = Normal()
        self.prior["sigma"] = InverseGamma(100.0, 1.0)
        self.prior["lengthscale_deriv"] = InverseGamma(1.0, 1.0)
        self.prior["alpha"] = InverseGamma(1.0, 1.0)
        self.prior["intercept"] = Normal()
        self.prior["slope"] = Normal()


    def model_fn(self, data_set, hsgp_params) -> None:

        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]

        slope = self.prior["slope"] if not isinstance(self.prior["slope"], Distribution) else sample("slope", self.prior["slope"])
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(1,))
        intercept = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"]) 
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"])
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(M_time,))
        weights = weights * spd 
        gamma_phi_gamma_time = jnp.einsum("tmz, m, z -> t", phi_time, weights, weights) 
        mu = intercept + slope * shifted_x_time - gamma_phi_gamma_time
        sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"])
        expanded_sigmas = sigmas * jnp.ones((self.j,)) 
            
        exposure = data_set["exposure"]
        obs = data_set["Y"]
        dist = Normal(mu, expanded_sigmas / exposure)
        y = sample("gaussian", dist, obs)    
    

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.000003), loss=Trace_ELBO(num_particles=10),
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)



    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


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
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    

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
        
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)

class GibbsConvexTVRFLVM(ConvexTVRFLVM):
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
        
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    
class GibbsConvexMaxBoundaryTVRFLVM(ConvexMaxBoundaryTVRFLVM):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior") -> None:
        return super().model_fn(data_set, hsgp_params, offsets, inference_method)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        raise NotImplementedError
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        raise NotImplementedError
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, gibbs_sites: list = []):
        key = jax.random.PRNGKey(0)
        inner_kernels = [NUTS(self.model_fn, init_strategy=init_to_value(values = {k: initial_values[k] for k in gibbs_sites[i] if k in initial_values})) for i in range(len(gibbs_sites))]
        outer_kernel = MultiHMCGibbs(inner_kernels, gibbs_sites_list=gibbs_sites)
        if vectorized:
            n_parallel = jax.local_device_count()
            n_vectorized = num_chains // n_parallel
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
            rng_keys = jax.random.split(key, n_parallel)
            traces = jax.pmap(do_mcmc)(rng_keys)
            return {k: jnp.concatenate(v) for k, v in traces.items()}
        else:
            mcmc = MCMC(outer_kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains = num_chains,
                        chain_method="parallel")
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True), None