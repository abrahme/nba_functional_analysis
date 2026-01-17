from abc import abstractmethod, ABC
import jax 
import flax.serialization as ser
import numpyro
import numpy as np
from numpyro import sample 
from numpyro.infer.util import log_density
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, StudentT, Independent, HalfCauchy, LogNormal, Binomial, HalfNormal, Categorical, MultivariateNormal, BetaProportion, Distribution, Uniform, BetaBinomial, Gamma, BinomialLogits, NegativeBinomial2, Dirichlet, MixtureSameFamily
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoNormal,  AutoLaplaceApproximation
from numpyro.handlers import substitute, seed, trace
import optax
from optax import linear_onecycle_schedule, adam
from jaxopt import LBFGS
from .hsgp import make_convex_f, make_psi_gamma, make_spectral_mixture_density, diag_spectral_density, make_convex_phi,  vmap_make_convex_phi, vmap_make_convex_phi_prime, eigenfunctions_multivariate, vmap_make_convex_phi_double_prime
import jax.numpy as jnp
import jax.scipy as jsci
from .MultiHMCGibbs import MultiHMCGibbs
from .HMCMetrics import NUTSWithMetrics
from model.model_utils import Type2Gumbel
from scipy.special import roots_legendre






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
        self.n, self.j, self.k = output_shape
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
        
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m, self.r))
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.k, 2 * self.m, self.j))
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
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized:bool, model_args, initial_values = {}, thinning = 1):
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
                chain_method="vectorized",
                thinning=thinning)
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
                        chain_method="parallel",
                        thinning=thinning)
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
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
       
        state = svi.init(jax.random.PRNGKey(0),**model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        return result.params, result.state
    
    

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
    
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning=thinning)

    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)

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
        
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m, self.r))

        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        wTx = jnp.einsum("nr,mr -> nm", X, W)
        phi = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(self.m))

        ls = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        kernel = self.make_kernel(ls)
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta",  MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel), sample_shape=(self.k, 2 * self.m)) ### don't need extra dimension
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

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values,thinning=thinning)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)

    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)



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
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["alpha"] = InverseGamma(1.0, 1.0)
        self.prior["intercept"] = Normal()
        self.prior["slope"] = Normal()


    def model_fn(self, data_set, hsgp_params, offsets = 0, prior = False) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        
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

     
        slope =  make_psi_gamma(psi_x, self.prior["slope"] if not isinstance(self.prior["slope"], Distribution) else sample("slope", self.prior["slope"], sample_shape=(self.m*2, self.k))) 
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,))
        intercept = make_psi_gamma(psi_x, self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"] , sample_shape=(2 * self.m, self.k))) 
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.k))
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

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values,thinning=thinning)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
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
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = HalfNormal()
        self.prior["intercept"] = Normal()
        self.prior["sigma_intercept"] = HalfNormal(.001)
        self.prior["sigma_beta"] = Exponential()
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["sigma_negative_binomial"] = InverseGamma(3, 2)
        self.prior["lengthscale_deriv"] = HalfNormal()
        self.prior["sigma_boundary_l"] = HalfNormal(.1)
        self.prior["sigma_boundary_r"] = HalfNormal(.1)
        self.prior["sigma_t"] = InverseGamma(2, 1)
        self.prior["sigma_c"] = InverseGamma(2, 1)
        self.prior["alpha"] = HalfNormal()
        self.prior["sigma_c"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_t"] = InverseGamma(2.0, 1.0)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["W_t_max"] = Normal()
        self.prior["W_c_max"] = Normal()
        self.prior["lengthscale_t_max"] = HalfNormal()
        self.prior["lengthscale_c_max"] = HalfNormal()
        self.prior["X_free"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r, ))
        lengthscale_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r, ))
        lengthscale_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r, ))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, 1))
        ls_deriv =  self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else  sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k, 1))
        spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        W_t_max = self.prior["W_t_max"] if not isinstance(self.prior["W_t_max"], Distribution) else sample("W_t_max", self.prior["W_t_max"], sample_shape=(self.m,self.r))
        W_c_max = self.prior["W_c_max"] if not isinstance(self.prior["W_c_max"], Distribution) else sample("W_c_max", self.prior["W_c_max"], sample_shape=(self.m,self.r))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, self.k))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, self.k))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, self.k))
        
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
        #     sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
        #     expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 5 + sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])


        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        wTx_t_max = jnp.einsum("nr, mr -> nm", X, W_t_max * jnp.sqrt(lengthscale_t_max))
        psi_x_t_max = jnp.concatenate([jnp.cos(wTx_t_max), jnp.sin(wTx_t_max)], axis = -1) * (1/ jnp.sqrt(self.m))   
        wTx_c_max = jnp.einsum("nr, mr -> nm", X, W_c_max * jnp.sqrt(lengthscale_c_max))
        psi_x_c_max = jnp.concatenate([jnp.cos(wTx_c_max), jnp.sin(wTx_c_max)], axis = -1) * (1/ jnp.sqrt(self.m))
        t_max = jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)   + jnp.arctanh(offsets["t_max"]/10)) * 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)   + jnp.arctanh(offsets["t_max"]/10)) * 10 )
        c_max = make_psi_gamma(psi_x_c_max, c_max_raw * sigma_c_max)  + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd.T[None]

        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor  
                dist = Poisson(jnp.exp(log_rate[mask]) *  jnp.exp(exposure[mask] +  de_trend[mask]))
            elif family == "negative-binomial":
                log_rate = linear_predictor 
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, log_rate.shape)
                # jax.debug.print(" {family}: {highest_val}, {highest_obs}", highest_val = log_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                # jax.debug.print("concentration neg bin: {lowest_val}", lowest_val = jnp.min(expanded_sigma_neg_bin[mask]))

                dist = NegativeBinomial2(mean = jax.nn.softplus(log_rate[mask]) * jnp.exp(exposure[mask] +  de_trend[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :] )
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count= exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(logit_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, logit_rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = logit_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
                # jax.debug.print("concentration beta: {lowest_val}", lowest_val = jnp.min(jnp.square(exposure[mask]) *  expanded_sigma_beta[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)   


              


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
 
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
 
  
        state = svi.init(jax.random.PRNGKey(0),**model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)


        return result.params, result.state

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












class ConvexMaxBoundaryTVRFLVM(ConvexTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        and boundary conditions
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(2, 1)
        self.prior["sigma_beta"] = Gamma(5000, 100.0)
        self.prior["sigma_beta_binomial"] = Gamma(5000, 100.0)
        self.prior["sigma_negative_binomial"] =  Gamma(5000, 100)
        self.prior["sigma_c"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_boundary_r"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_boundary_l"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_t"] = InverseGamma(2.0, 1.0)
        self.prior["alpha"] = InverseGamma(2.0, .1)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["boundary_l"] = Normal()
        self.prior["boundary_r"] = Normal()
        self.prior["X_free"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        

        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, self.k))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, self.k))
        
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(1, self.k))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(1, self.k))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(2 * self.m, self.k))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(2 * self.m, self.k))
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        if num_neg_bins > 0:
            sigma_negative_binomial = self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if (len(sample_free_indices > 0)):
            
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
        else:
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
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, self.k) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] 
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)) 
                dist = Poisson(rate).mask(mask)
            elif family == "negative-binomial":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
                dist = NegativeBinomial2(mean = rate, concentration = expanded_sigma_neg_bin).mask(mask)
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate , total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                dist = BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)

        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())   

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
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
        self.prior["sigma_ar"] = InverseGamma(2, 1)
        # self.prior["sigma_ar"] = Type2Gumbel(alpha=.05, scale=.001)
        self.prior["beta_ar"] = Normal()
        self.prior["AR_0"] = Normal()
        self.prior["X_free"] = Normal()
    
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"])
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        

        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, self.k))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, self.k))
        
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(1, self.k))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(1, self.k))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(2 * self.m, self.k))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(2 * self.m, self.k))
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        if (len(sample_free_indices > 0)):
            
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])


        if num_neg_bins > 0:
            sigma_negative_binomial = self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))

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
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, self.k) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd

        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(self.k,1))
        
        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(self.k,1))
        # sigma_ar = offsets["avg_sd"][..., None]
        # rho_ar = offsets["rho"][..., None]
        # Standard normals: shape (K, N, T)
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(self.j, self.k, self.n))
        AR_0_raw = self.prior["AR_0"] if not isinstance(self.prior["AR_0"], Distribution) else sample("AR_0", self.prior["AR_0"], sample_shape=(self.k, self.n))
        AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
        # AR_0 = jnp.zeros((self.k, self.n))

        def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
        _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = z)
        AR = jnp.transpose(AR, (1,2,0))
        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x) 
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices] + AR[k_indices] + data_set[family]["de_trend"]
            exposure = data_set[family]["exposure"]
            de_trend = data_set[family]["de_trend"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
                dist = Poisson(rate).mask(mask)
            elif family == "negative-binomial":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
                dist = NegativeBinomial2(mean = rate, concentration = expanded_sigma_neg_bin).mask(mask)
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                concentration = jnp.where(mask, exposure, 1.0) * sigma_beta
                dist = BetaProportion(rate, concentration).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)


        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, 0.1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())   

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)

    
        
    
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






class ConvexMaxARTVRFLVM(ConvexMaxTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
    
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        # self.prior["sigma_ar"] = InverseGamma(2, 1)
        self.prior["sigma_ar"] = Type2Gumbel(alpha=.05, scale=.001)
        self.prior["beta_ar"] = Normal()
        self.prior["AR_0"] = Normal()

    
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r, ))
        lengthscale_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r, ))
        lengthscale_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r, ))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, 1))
        ls_deriv =  self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else  sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k, 1))
        spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        W_t_max = self.prior["W_t_max"] if not isinstance(self.prior["W_t_max"], Distribution) else sample("W_t_max", self.prior["W_t_max"], sample_shape=(self.m,self.r))
        W_c_max = self.prior["W_c_max"] if not isinstance(self.prior["W_c_max"], Distribution) else sample("W_c_max", self.prior["W_c_max"], sample_shape=(self.m,self.r))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(1, self.k))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(1, self.k))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(2 * self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(2 * self.m, self.k))
        
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
        #     sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
        #     expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 5 + sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])

        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]

        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale))
        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))   
        wTx_t_max = jnp.einsum("nr, mr -> nm", X, W_t_max * jnp.sqrt(lengthscale_t_max))
        psi_x_t_max = jnp.concatenate([jnp.cos(wTx_t_max), jnp.sin(wTx_t_max)], axis = -1) * (1/ jnp.sqrt(self.m))   
        wTx_c_max = jnp.einsum("nr, mr -> nm", X, W_c_max * jnp.sqrt(lengthscale_c_max))
        psi_x_c_max = jnp.concatenate([jnp.cos(wTx_c_max), jnp.sin(wTx_c_max)], axis = -1) * (1/ jnp.sqrt(self.m))
        t_max = jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)   + jnp.arctanh(offsets["t_max"]/10)) * 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)   + jnp.arctanh(offsets["t_max"]/10)) * 10 )
        c_max = make_psi_gamma(psi_x_c_max, c_max_raw * sigma_c_max)  + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd.T[None]

        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(self.k,1))
        
        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(self.k,1))
        # sigma_ar = offsets["avg_sd"][..., None]
        # rho_ar = offsets["rho"][..., None]
        # Standard normals: shape (K, N, T)
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(self.j, self.k, self.n))
        AR_0_raw = self.prior["AR_0"] if not isinstance(self.prior["AR_0"], Distribution) else sample("AR_0", self.prior["AR_0"], sample_shape=(self.k, self.n))
        AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
        # AR_0 = jnp.zeros((self.k, self.n))

        def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
        _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = z)
        AR = jnp.transpose(AR, (1,2,0))
        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x) 
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices] + AR[k_indices] + data_set[family]["de_trend"] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            de_trend = data_set[family]["de_trend"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor  
                dist = Poisson(jnp.exp(log_rate[mask]) *  jnp.exp(exposure[mask] +  de_trend[mask]))
            elif family == "negative-binomial":
                log_rate = linear_predictor 
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, log_rate.shape)
                # jax.debug.print(" {family}: {highest_val}, {highest_obs}", highest_val = log_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                # jax.debug.print("concentration neg bin: {lowest_val}", lowest_val = jnp.min(expanded_sigma_neg_bin[mask]))

                dist = NegativeBinomial2(mean = jax.nn.softplus(log_rate[mask]) * jnp.exp(exposure[mask] +  de_trend[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :] )
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count= exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(logit_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, logit_rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = logit_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
                # jax.debug.print("concentration beta: {lowest_val}", lowest_val = jnp.min(jnp.square(exposure[mask]) *  expanded_sigma_beta[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)   


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)

    
        
    
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
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = [], thinning = 1):
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
            chain_method="vectorized",
            thinning = thinning)
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
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = [], thinning = 1):
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
            chain_method="vectorized",
            thinning = thinning)
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
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = [], thinning = 1):
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
            chain_method="vectorized",
            thinning = thinning)
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
    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        return super().model_fn(data_set, hsgp_params, offsets, inference_method, sample_free_indices, sample_fixed_indices)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        raise NotImplementedError
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        raise NotImplementedError
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, gibbs_sites: list = [], thinning = 1):
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
                chain_method="vectorized",
                thinning = thinning)
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
                        chain_method="parallel",
                        thinning=thinning)
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True), None


class GibbsConvexMaxBoundaryARTVRFLVM(ConvexMaxBoundaryARTVRFLVM):

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        return super().model_fn(data_set, hsgp_params, offsets, inference_method, sample_free_indices, sample_fixed_indices)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000)):
        raise NotImplementedError
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        raise NotImplementedError
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, gibbs_sites: list = [], thinning = 1):
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
                chain_method="vectorized",
                thinning = thinning)
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
                        chain_method="parallel",
                        thinning=thinning)
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True), None
        






class HSGPLVMBase(ABC):
    """ 
    Latent Variable model with Hilbert Space GP Approximation
    """
    def __init__(self, latent_rank: int, hsgp_dim: list[int] | int, output_shape: tuple, L_X: jnp.array, basis: jnp.array = None) -> None:
        self.r = latent_rank 
        self.m = int(np.prod(hsgp_dim * np.ones(self.r))) 
        self.M_X = hsgp_dim
        self.n, self.j, self.k = output_shape
        self.prior = {}
        self.L_X = L_X ## per dimension of X 
        self.basis = basis
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["beta"] = Normal()
        self.prior["X"] = Normal(loc=jnp.zeros((self.n, self.r)))
        self.prior["sigma"] = InverseGamma(1, 1)
        self.prior["lengthscale"] = HalfNormal()
        self.prior["alpha_X"] = HalfNormal(.1)
        self.prior["intercept"] = Normal()
        self.prior["sigma_intercept"] = HalfNormal(.001)
    
    def _stabilize_x(self, X):
        """Make sure X is within [-1, 1]^D
        """
        
        return jnp.tanh(X)

    @abstractmethod
    def model_fn(self, data_set) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        
        X_raw = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        X = self._stabilize_x(X_raw)
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))

        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))

        spd = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, lengthscale, self.M_X)))(alpha_X)    
        phi = eigenfunctions_multivariate(X, self.L_X, self.M_X) ### need function for computing the 
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.k, self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, spd * beta)
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
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized:bool, model_args, initial_values = {}, thinning = 1):
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
                chain_method="vectorized",
                thinning=thinning)
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
                        chain_method="parallel",
                        thinning=thinning)
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True)
    @abstractmethod
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state:dict = {}, sample_shape = (4, 2000)):
        guide = AutoNormal(self.model_fn, prefix="", **guide_kwargs, init_state = initial_state,
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
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_state=initial_state, **model_args)
        return result.params, result.state
    
    

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args, num_samples = 1000):
        predictive = Predictive(self.model_fn, posterior_samples,  num_samples=num_samples)
        return predictive(jax.random.PRNGKey(0), **model_args)


class SurvTVHSGPLVM(HSGPLVMBase):
    def __init__(self, latent_rank: int, hsgp_dim: list[int] | int, output_shape: tuple, L_X: jnp.array, basis: jnp.array = None) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale_deriv"] = HalfNormal()
        self.prior["alpha"] = HalfNormal(.1)
        self.prior["X_free"] = Normal()
        self.prior["sigma"] = HalfNormal()
        self.prior["t_max_raw"] = Normal()

        

    def model_fn(self, data_set, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]),) -> None:
        obs = data_set["observations"] ### time to event
        censor = data_set["censored"] ### is it censored
        censor_type = data_set["censor_type"] ### left or right
        self.k = censor.shape[-1]
        # sigma = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(self.k,))
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            # X = self._stabilize_x(X)
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            # X = self._stabilize_x(X)

        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        t_max = make_psi_gamma(psi_x, t_max_raw * spd_X.T)

        distribution = LogNormal(loc = t_max, scale = 1 * jnp.ones_like(t_max))
        cdf = distribution.cdf(obs)
        log_density = distribution.log_prob(obs)
        likelihood = (1 - censor) * log_density + censor*(censor_type * jnp.log1p(-1*cdf) + (1 - censor_type)* jnp.log(cdf))
        numpyro.factor("log_lik", likelihood.sum())



        

        
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)
    
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning)
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    




class ConditionalSurvTVHSGPLVM(HSGPLVMBase):
    def __init__(self, latent_rank: int, hsgp_dim: list[int] | int, output_shape: tuple, L_X: jnp.array, basis: jnp.array = None) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale_deriv"] = HalfNormal()
        self.prior["alpha"] = HalfNormal(.1)
        self.prior["X_free"] = Normal()
        self.prior["sigma"] = HalfNormal()
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()


        

    def model_fn(self, data_set, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]),) -> None:
        obs = data_set["observations"] ### time to event
        censor = data_set["censored"] ### is it censored
        self.k = censor.shape[-1]
        # sigma = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(self.k,))
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        alpha = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(1, ))
        spd = jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            # X = self._stabilize_x(X)
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            # X = self._stabilize_x(X)

        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"], sample_shape=(self.m, 1))
        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        t_max = make_psi_gamma(psi_x, t_max_raw * spd_X.T)
        c_max = make_psi_gamma(psi_x, c_max * spd)
        distribution_entry = LogNormal(loc = t_max[:,0], scale = 1 * jnp.ones_like(t_max[:,0]))
        log_density_entry = distribution_entry.log_prob(obs[:,0])
        kappa = 1.01
        time_since_entry = obs[:,1] - obs[:,0]
        log_base_hazard_exit = jnp.log(kappa) + kappa * t_max[:,1] + (kappa - 1) * (time_since_entry) + c_max * obs[:, 0]
        log_surv_exit = - 1 * jnp.exp(c_max * obs[:, 0]) * jnp.power(jax.nn.softplus(t_max[:,1]) * time_since_entry, kappa)
        log_likelihood = (1 - censor[:,0])*(log_density_entry + (log_surv_exit) + (1 - censor[:,1])* log_base_hazard_exit) + censor[:,0]*(self.marginalize_entry(kappa=kappa, t=obs[:,1], entry_dist=distribution_entry, t_max=t_max[:,1], c_max=c_max, censor_right=(1 - censor[:,1]), entry_upper_bound=obs[:,0], num_gridpoints=50))
        numpyro.factor("log_lik", log_likelihood.sum())

    @staticmethod
    def compute_joint_log_density(kappa, entry, t, entry_dist, t_max, c_max, censor_right):
        time_since_entry = t - entry
        log_density_entry = jax.vmap(lambda e: entry_dist.log_prob(e))(entry.T).T
        log_base_hazard_exit = jnp.log(kappa) + kappa * t_max + (kappa - 1) * (time_since_entry) + c_max * entry
        log_surv_exit = - 1 * jnp.exp(c_max * entry) * jnp.power(jax.nn.softplus(t_max) * time_since_entry, kappa)
        log_density_exit = censor_right[..., None]*log_base_hazard_exit + log_surv_exit
        return log_density_exit + log_density_entry

 
    def marginalize_entry(self, kappa, t, entry_dist, t_max, c_max, censor_right, entry_upper_bound, num_gridpoints,e_min=1):
        x, w = roots_legendre(num_gridpoints)
        # map x ∈ [-1,1] to e ∈ [e_min, L]
        entry = (entry_upper_bound - e_min)[..., None] * (x[None] + 1)/2 + e_min
        weights = w[None] * (entry_upper_bound - e_min)[..., None] / 2
        integrand = self.compute_joint_log_density(kappa, entry, t[..., None], entry_dist, t_max[..., None], c_max, censor_right )
        return jsci.special.logsumexp(integrand + jnp.log(weights))        

        
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)
    
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning)
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    

class ConvexMaxTVHSGPLVM(HSGPLVMBase):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        max value using hsgp for the bas
        
    """
    def __init__(self, latent_rank: int, hsgp_dim: int, output_shape: tuple, L_X, basis) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["sigma_beta"] = Exponential()
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["sigma_negative_binomial"] = InverseGamma(3, 2)
        self.prior["lengthscale_deriv"] = HalfNormal()
        self.prior["lengthscale_t_max"] = HalfNormal(.1)
        self.prior["lengthscale_c_max"] = HalfNormal(.1)
        self.prior["sigma_boundary_l"] = HalfNormal(.1)
        self.prior["sigma_boundary_r"] = HalfNormal(.1)
        self.prior["sigma_t"] = InverseGamma(2, 1)
        self.prior["sigma_c"] = InverseGamma(2, 1)
        self.prior["alpha"] = HalfNormal()
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["X_free"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        # num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale =  self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else  sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, 1))
        ls_deriv =  self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else  sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,1)) 
        # jax.debug.print("ls_deriv: {highest_val}", highest_val = jnp.max(ls_deriv))
        # jax.debug.print("ls: {highest_val}", highest_val = jnp.max(lengthscale))
        # jax.debug.print("alpha: {highest_val}", highest_val = jnp.max(alpha_time))



        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        # alpha_X = jnp.ones((self.k,))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        spd = jnp.einsum("kt, km -> mtk", spd_time, spd_X)
        # jax.debug.print("alpha x: {highest_val}", highest_val = jnp.max(alpha_X))
        # jax.debug.print("spd_time: {highest_val}", highest_val = jnp.max(spd_time))
        # jax.debug.print("spd_X: {highest_val}", highest_val = jnp.max(spd_X))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        # sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(self.k,))
        # sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(self.k,))
        # boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(self.m, self.k))
        # boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(self.m, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.m, self.k))
        ls_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r,))
        ls_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r,))
        # ls_boundary_l = self.prior["lengthscale_boundary_l"] if not isinstance(self.prior["lengthscale_boundary_l"], Distribution) else sample("lengthscale_boundary_l", self.prior["lengthscale_boundary_l"], sample_shape=(self.r,))
        # ls_boundary_r = self.prior["lengthscale_boundary_r"] if not isinstance(self.prior["lengthscale_boundary_r"], Distribution) else sample("lengthscale_boundary_r", self.prior["lengthscale_boundary_r"], sample_shape=(self.r,))

        spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_t_max, self.L_X, self.M_X)))(sigma_c_max)
        spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_c_max, self.L_X, self.M_X)))(sigma_t_max)
        # spd_boundary_r = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_r, self.L_X, self.M_X)))(sigma_boundary_l)
        # spd_boundary_l = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_l, self.L_X, self.M_X)))(sigma_boundary_r)


        

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
            # sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
            # expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 5 + sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            X = self._stabilize_x(X) * 1.9
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            X = self._stabilize_x(X) * 1.9

        # jax.debug.print("biggest X val :{x_val}", x_val = jnp.max(jnp.abs(X)))

        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offsets["t_max"]/10)) * 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offsets["t_max"]/10)) * 10 )
        c_max = make_psi_gamma(psi_x, c_max_raw * spd_c_max.T)  + offsets["c_max"]

        # boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw  * spd_boundary_r.T) + offsets["boundary_r"])
        # boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_l_raw * spd_boundary_l.T) + offsets["boundary_l"])
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        


        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
        # rank = 3
        # U = self.prior["U"] if not isinstance(self.prior["U"], Distribution) else numpyro.sample("U", self.prior["U"], sample_shape=(self.m, rank))
        # V = self.prior["V"] if not isinstance(self.prior["V"], Distribution) else numpyro.sample("V", self.prior["V"], sample_shape=(M_time, rank))
        # A = self.prior["A"] if not isinstance(self.prior["A"], Distribution) else numpyro.sample("A", self.prior["A"], sample_shape=(self.k, rank))  
        # weights = jnp.einsum('ir, tr, jr -> itj', U, V, A) / jnp.sqrt(rank)
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m , M_time, self.k))
        weights *= spd


        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor  
                dist = Poisson(jnp.exp(log_rate[mask]) *  jnp.exp(exposure[mask] +  de_trend[mask]))
            elif family == "negative-binomial":
                log_rate = linear_predictor 
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, log_rate.shape)
                # jax.debug.print(" {family}: {highest_val}, {highest_obs}", highest_val = log_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                # jax.debug.print("concentration neg bin: {lowest_val}", lowest_val = jnp.min(expanded_sigma_neg_bin[mask]))

                dist = NegativeBinomial2(mean = jax.nn.softplus(log_rate[mask]) * jnp.exp(exposure[mask] +  de_trend[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :] )
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count= exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(logit_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, logit_rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = logit_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
                # jax.debug.print("concentration beta: {lowest_val}", lowest_val = jnp.min(jnp.square(exposure[mask]) *  expanded_sigma_beta[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)      

        # value = (boundary_r - jnp.transpose(mu[..., -1]))
        # logp = Normal(0, .1).log_prob(value)
        # numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        # value = (boundary_l - jnp.transpose(mu[..., 0]))
        # logp = Normal(0, .1).log_prob(value)
        # numpyro.factor(f"boundary_conditions_l", logp.sum()) 
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        # surgical_init = {
        # "beta": jnp.zeros((self.m, 17, self.k)),    # Reset the 2.2M weights to 0
        # "X": jnp.zeros((self.n, self.r)), 
        #         "c_max": jnp.zeros((self.m, self.k)),
        #         "t_max_raw": jnp.zeros((self.m, self.k)),        # Reset latent space to center
        # "alpha": jnp.ones(self.k) * 0.1,           # Force a small but active signal
        # "lengthscale": jnp.ones(5) * 0.5   # Force a smooth start
        #     }
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state

  
        state = svi.init(jax.random.PRNGKey(0),**model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)




        # try:
        #     with seed(substitute(self.model_fn, result.params), jax.random.PRNGKey(0)):
        #         # Just run the model once to see if it finishes
        #         tr = trace(self.model_fn).get_trace(**model_args)
                
        #     for name, site in tr.items():
        #         if site['type'] == 'sample':
        #             val = site['value']
        #             print(f"Site: {name:15} | Shape: {str(val.shape):15} | Max: {jnp.max(val)}")
        # except Exception as e:
        #     print(f"Manual trace failed: {e}")
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, initial_state = initial_state, **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)





class ConvexMaxBackConstrainedTVHSGPLVM(HSGPLVMBase):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        max value using hsgp for the basis. back constrained
        
    """
    def __init__(self, latent_rank: int, hsgp_dim: int, output_shape: tuple, L_X, basis) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["sigma_beta"] = Uniform()
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["sigma_negative_binomial"] = Exponential()
        self.prior["lengthscale_deriv"] = HalfNormal(.1)
        self.prior["lengthscale_t_max"] = HalfNormal(.1)
        self.prior["lengthscale_c_max"] = HalfNormal(.1)
        self.prior["lengthscale_boundary_l"] = HalfNormal(.1)
        self.prior["lengthscale_boundary_r"] = HalfNormal(.1)
        self.prior["sigma_boundary_l"] = HalfNormal(.1)
        self.prior["sigma_boundary_r"] = HalfNormal(.1)
        self.prior["sigma_t"] = InverseGamma(2, .5)
        self.prior["alpha"] = HalfNormal(.0001)
        self.prior["t_max_raw"] = Normal()
        self.prior["boundary_r"] = Normal()
        self.prior["boundary_l"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["W"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale =  self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else  sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        ls_deriv = 2 + self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,)) + 2 
        spd_time = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        spd = jnp.einsum("tk, km -> mtk", spd_time, spd_X)
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(self.k,))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(self.k,))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(self.m, self.k))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(self.m, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.m, self.k))
        ls_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r,))
        ls_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r,))
        ls_boundary_l = self.prior["lengthscale_boundary_l"] if not isinstance(self.prior["lengthscale_boundary_l"], Distribution) else sample("lengthscale_boundary_l", self.prior["lengthscale_boundary_l"], sample_shape=(self.r,))
        ls_boundary_r = self.prior["lengthscale_boundary_r"] if not isinstance(self.prior["lengthscale_boundary_r"], Distribution) else sample("lengthscale_boundary_r", self.prior["lengthscale_boundary_r"], sample_shape=(self.r,))

        spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_t_max, self.L_X, self.M_X)))(sigma_c_max)
        spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_c_max, self.L_X, self.M_X)))(sigma_t_max)
        spd_boundary_r = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_r, self.L_X, self.M_X)))(sigma_boundary_l)
        spd_boundary_l = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_l, self.L_X, self.M_X)))(sigma_boundary_r)

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        if num_beta > 0:
            sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 1 / self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 1 / sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        

        ### encoder
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.k * self.j, self.r))
        W_eff, Y_linearized = offsets["W_eff"], offsets["Y_linearized"]
        X = jnp.tanh(jnp.dot(Y_linearized, W) / W_eff.sum(axis=1, keepdims=True))
        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        t_max = make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw * spd_c_max.T)  + offsets["c_max"]

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw  * spd_boundary_r.T) + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_l_raw * spd_boundary_l.T) + offsets["boundary_l"])
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m , M_time, self.k))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] 
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                dist = Poisson(jnp.exp(log_rate)).mask(mask)
            elif family == "negative-binomial":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                dist = NegativeBinomial2(mean = jnp.exp(log_rate), concentration = expanded_sigma_neg_bin).mask(mask)
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate , total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate)), concentration1= jsci.special.expit(logit_rate), total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaProportion(jsci.special.expit(logit_rate), jnp.where(mask, exposure, 1.0) * expanded_sigma_beta).mask(mask)
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)      

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning)
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state
        
        state = svi.init(jax.random.PRNGKey(0), **model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        return result.params, result.state
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)



class ConvexMaxSpectralMixtureTVHSGPLVM(ConvexMaxTVHSGPLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        max value using hsgp for the basis, and a gmm prior on the spectral density 
        
    """
    def __init__(self, latent_rank: int, hsgp_dim: int, mixture_dim: int, output_shape: tuple, L_X, basis) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)
        self.mixture_dim = mixture_dim
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        # self.prior["sigma_beta"] = Gamma(5000, 100.0)
        # self.prior["sigma_beta_binomial"] = Gamma(5000, 100.0)
        self.prior["mu"] = Normal()
        self.prior["covariance"] = InverseGamma(2,1)
        self.prior["mixture_weight"] = Dirichlet(.03 * jnp.ones(self.mixture_dim))
    



    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,))
        spd_time = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        mixture_weight = self.prior["mixture_weight"] if not isinstance(self.prior["mixture_weight"], Distribution) else sample("mixture_weight", self.prior["mixture_weight"], sample_shape=(self.k, ))
        component_mu = self.prior["mu"] if not isinstance(self.prior["mu"], Distribution) else sample("mu", self.prior["mu"], sample_shape=(self.k,  self.mixture_dim, self.r))
        component_scale = self.prior["covariance"] if not isinstance(self.prior["covariance"], Distribution) else sample("covariance", self.prior["covariance"], sample_shape=(self.k, self.mixture_dim, self.r))
        spd_X = make_spectral_mixture_density(hsgp_params["eigenvalues_X"], component_mu, component_scale, mixture_weight)
        spd = jnp.einsum("tk, km -> mtk", spd_time, spd_X)
        sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.m, self.k))
        ls_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r,))
        ls_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r,))
        spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_t_max, self.L_X, self.M_X)))(sigma_c_max)
        spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_c_max, self.L_X, self.M_X)))(sigma_t_max)
        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"])
        sigma_beta_binomial = self.prior["sigma_beta_binomial"] if not isinstance(self.prior["sigma_beta_binomial"], Distribution) else sample("sigma_beta_binomial", self.prior["sigma_beta_binomial"])
        if num_neg_bins > 0:
            sigma_negative_binomial = self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            X = self._stabilize_x(X)
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            X = self._stabilize_x(X)



        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T) ) * 5  + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T) ) * 5  + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw * spd_c_max.T)  + offsets["c_max"]
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m , M_time, self.k))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] 
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)) 
                dist = Poisson(rate).mask(mask)
            elif family == "negative-binomial":
                rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
                dist = NegativeBinomial2(mean = rate, concentration = expanded_sigma_neg_bin).mask(mask)
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate , total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                dist = BetaBinomial(concentration0=(1-rate) * sigma_beta_binomial, concentration1= rate * sigma_beta_binomial, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
                dist = BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask)

              
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)           


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)




class ConvexMaxARTVHSGPLVM(ConvexMaxTVHSGPLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        max value using hsgp for the bas
        
    """
    def __init__(self, latent_rank: int, hsgp_dim: int, output_shape: tuple, L_X, basis) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = InverseGamma(2, 1)
        self.prior["beta_ar"] = Normal()
        # self.prior["beta_ar"] = StudentT(df = 3)
        self.prior["AR_0"] = Normal()
        # self.prior["AR_0"] = StudentT(df  = 3)
        self.prior["X_free"] = Normal()
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        

        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(self.k,1))
        
        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(self.k,1))
        # sigma_ar = offsets["avg_sd"][..., None]
        # rho_ar = offsets["rho"][..., None]
        # Standard normals: shape (K, N, T)
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        # num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale =  self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else  sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, 1))
        ls_deriv =  self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else  sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,1)) 
        # jax.debug.print("ls_deriv: {highest_val}", highest_val = jnp.max(ls_deriv))
        # jax.debug.print("ls: {highest_val}", highest_val = jnp.max(lengthscale))
        # jax.debug.print("alpha: {highest_val}", highest_val = jnp.max(alpha_time))



        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        # alpha_X = jnp.ones((self.k,))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        spd = jnp.einsum("kt, km -> mtk", spd_time, spd_X)
        # jax.debug.print("alpha x: {highest_val}", highest_val = jnp.max(alpha_X))
        # jax.debug.print("spd_time: {highest_val}", highest_val = jnp.max(spd_time))
        # jax.debug.print("spd_X: {highest_val}", highest_val = jnp.max(spd_X))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        # sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(self.k,))
        # sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(self.k,))
        # boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(self.m, self.k))
        # boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(self.m, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.m, self.k))
        ls_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r,))
        ls_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r,))
        # ls_boundary_l = self.prior["lengthscale_boundary_l"] if not isinstance(self.prior["lengthscale_boundary_l"], Distribution) else sample("lengthscale_boundary_l", self.prior["lengthscale_boundary_l"], sample_shape=(self.r,))
        # ls_boundary_r = self.prior["lengthscale_boundary_r"] if not isinstance(self.prior["lengthscale_boundary_r"], Distribution) else sample("lengthscale_boundary_r", self.prior["lengthscale_boundary_r"], sample_shape=(self.r,))

        spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_t_max, self.L_X, self.M_X)))(sigma_c_max)
        spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_c_max, self.L_X, self.M_X)))(sigma_t_max)
        # spd_boundary_r = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_r, self.L_X, self.M_X)))(sigma_boundary_l)
        # spd_boundary_l = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_l, self.L_X, self.M_X)))(sigma_boundary_r)


        

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
            # sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
            # expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 5 + sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            X = self._stabilize_x(X) * 1.9
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            X = self._stabilize_x(X) * 1.9

        # jax.debug.print("biggest X val :{x_val}", x_val = jnp.max(jnp.abs(X)))

        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        

        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offsets["t_max"]/10)) * 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offsets["t_max"]/10)) * 10 )
        c_max = make_psi_gamma(psi_x, c_max_raw * spd_c_max.T)  + offsets["c_max"]

        # boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw  * spd_boundary_r.T) + offsets["boundary_r"])
        # boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_l_raw * spd_boundary_l.T) + offsets["boundary_l"])
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        


        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
        # rank = 3
        # U = self.prior["U"] if not isinstance(self.prior["U"], Distribution) else numpyro.sample("U", self.prior["U"], sample_shape=(self.m, rank))
        # V = self.prior["V"] if not isinstance(self.prior["V"], Distribution) else numpyro.sample("V", self.prior["V"], sample_shape=(M_time, rank))
        # A = self.prior["A"] if not isinstance(self.prior["A"], Distribution) else numpyro.sample("A", self.prior["A"], sample_shape=(self.k, rank))  
        # weights = jnp.einsum('ir, tr, jr -> itj', U, V, A) / jnp.sqrt(rank)
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m , M_time, self.k))
        weights *= spd


        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(self.j, self.k, self.n))
        AR_0_raw = self.prior["AR_0"] if not isinstance(self.prior["AR_0"], Distribution) else sample("AR_0", self.prior["AR_0"], sample_shape=(self.k, self.n))
        AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
        def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
        _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = z)
        AR = jnp.transpose(AR, (1,2,0))
        
        

        

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        
        for family in data_set:
            k_indices = data_set[family]["indices"]
            linear_predictor = mu[k_indices] + AR[k_indices] + data_set[family]["de_trend"] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            de_trend = data_set[family]["de_trend"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor  
                dist = Poisson(jnp.exp(log_rate[mask]) *  jnp.exp(exposure[mask] +  de_trend[mask]))
            elif family == "negative-binomial":
                log_rate = linear_predictor 
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, log_rate.shape)
                # jax.debug.print(" {family}: {highest_val}, {highest_obs}", highest_val = log_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                # jax.debug.print("concentration neg bin: {lowest_val}", lowest_val = jnp.min(expanded_sigma_neg_bin[mask]))

                dist = NegativeBinomial2(mean = jax.nn.softplus(log_rate[mask]) * jnp.exp(exposure[mask] +  de_trend[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :] )
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count= exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(jnp.abs(logit_rate))
                # problematic_player_index1, problematic_player_index2, _ = jnp.unravel_index(highest_val, logit_rate.shape)
                # jax.debug.print("{family}: {highest_val}, {highest_obs} ", highest_val = logit_rate[problematic_player_index1, problematic_player_index2, :], family = family,
                #                 highest_obs = obs[problematic_player_index1, problematic_player_index2, :]  )
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
                # jax.debug.print("concentration beta: {lowest_val}", lowest_val = jnp.min(jnp.square(exposure[mask]) *  expanded_sigma_beta[mask]))
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)      


              


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)



class ConvexMaxTVLinearLVM(ConvexMaxTVRFLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        self.r = latent_rank 
        self.n, self.j, self.k = output_shape
        self.basis = basis ### basis for time dimension
        self.t = len(basis)
        self.prior = {}
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["sigma_beta"] = Uniform()
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["sigma_negative_binomial"] = Exponential()
        self.prior["lengthscale_deriv"] = HalfNormal(.1)
        self.prior["alpha"] = HalfNormal(kwargs.get("scale_values", 1e-1))
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["X_free"] = Normal()
        # self.prior["sigma_c"] = HalfNormal(.1)
        # self.prior["sigma_t"] = HalfNormal(.1)
    

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"])
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.r , self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.r , self.k))
        

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
        #     sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
        #     expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial =  self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else  sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            # X = self._stabilize_x(X) * 1.9
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            # X = self._stabilize_x(X) * 1.9


        psi_x = X
        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + jnp.arctanh(offsets["t_max"]/10))* 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + jnp.arctanh(offsets["t_max"]/10))* 10)
        c_max = make_psi_gamma(psi_x, c_max_raw * sigma_c_max.T)  + offsets["c_max"]


        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        


        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.r , M_time, self.k))
        weights *= spd_time.T[None]


        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor[mask] + exposure[mask] + de_trend[mask]
                dist = Poisson(jnp.exp(log_rate))
            elif family == "negative-binomial":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index = jnp.unravel_index(highest_val, log_rate.shape)[1]
                # jax.debug.print("{highest_val}, {problematic_player}, {problematic_player_exposure}", highest_val = log_rate[:, problematic_player_index, :], problematic_player = obs[:, problematic_player_index, :],
                #                 problematic_player_exposure = exposure[:, problematic_player_index, :] )
                dist = NegativeBinomial2(mean = jnp.exp(log_rate[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count=exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)      

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state
        
        state = svi.init(jax.random.PRNGKey(0), **model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, initial_state = initial_state, **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)






class ConvexMaxARTVLinearLVM(ConvexMaxTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = InverseGamma(2, 1)
        self.prior["beta_ar"] = Normal()
        # self.prior["beta_ar"] = StudentT(df = 3)
        self.prior["AR_0"] = Normal()
        # self.prior["AR_0"] = StudentT(df  = 3)
    

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"])
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.r  , self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.r  , self.k))
        

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        # if num_beta > 0:
        #     sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
        #     expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial =  self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else  sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        if (len(sample_free_indices > 0)):
            X = jnp.zeros((self.n, self.r))
            X_free = sample("X_free", self.prior["X_free"], sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            # X = self._stabilize_x(X) * 1.9
        else:
            X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
            # X = self._stabilize_x(X) * 1.9


        
        psi_x = X
        t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + jnp.arctanh(offsets["t_max"]/10))* 10  if not prior else numpyro.deterministic("t_max", jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + jnp.arctanh(offsets["t_max"]/10))* 10)
        c_max = make_psi_gamma(psi_x, c_max_raw * sigma_c_max.T)  + offsets["c_max"]


        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        


        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.r , M_time, self.k))
        weights *= spd_time.T[None]




        intercept_raw = self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"], sample_shape=(self.k, self.n))
        sigma_intercept = self.prior["sigma_intercept"] if not isinstance(self.prior["sigma_intercept"], Distribution) else sample("sigma_intercept", self.prior["sigma_intercept"], sample_shape=(self.k, 1 ))

        player_intercept = (intercept_raw * sigma_intercept)[..., None]

        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(self.k,1))
        
        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(self.k,1))
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(self.j, self.k, self.n))
        AR_0_raw = self.prior["AR_0"] if not isinstance(self.prior["AR_0"], Distribution) else sample("AR_0", self.prior["AR_0"], sample_shape=(self.k, self.n))
        AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
        def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
        _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = z)
        AR = jnp.transpose(AR, (1,2,0))

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] + AR[k_indices] + player_intercept[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            elif family == "poisson":
                log_rate = linear_predictor[mask] + exposure[mask] + de_trend[mask]
                dist = Poisson(jnp.exp(log_rate))
            elif family == "negative-binomial":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                # highest_val = jnp.argmax(-1*(log_rate))
                # problematic_player_index = jnp.unravel_index(highest_val, log_rate.shape)[1]
                # jax.debug.print("{highest_val}, {problematic_player}, {problematic_player_exposure}", highest_val = log_rate[:, problematic_player_index, :], problematic_player = obs[:, problematic_player_index, :],
                #                 problematic_player_exposure = exposure[:, problematic_player_index, :] )
                dist = NegativeBinomial2(mean = jnp.exp(log_rate[mask]), concentration = expanded_sigma_neg_bin[mask] * jnp.exp(exposure[mask]))
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate[mask] , total_count=exposure[mask].astype(int))
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate[mask])), concentration1= jsci.special.expit(logit_rate[mask]), total_count=exposure[mask].astype(int))
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask]) )
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs=None)      

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state
        
        state = svi.init(jax.random.PRNGKey(0), **model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, initial_state = initial_state, **model_args)
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexMaxARBackConstrainedTVHSGPLVM(ConvexMaxBackConstrainedTVHSGPLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        max value using hsgp for the basis. back constrained w AR
        
    """
    def __init__(self, latent_rank: int, hsgp_dim: int, output_shape: tuple, L_X, basis) -> None:
        super().__init__(latent_rank, hsgp_dim, output_shape, L_X, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = InverseGamma(2, 1)
        self.prior["beta_ar"] = Normal()
        # self.prior["beta_ar"] = StudentT(df = 3)
        self.prior["AR_0"] = Normal()
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale =  self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else  sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.k, ))
        ls_deriv = 2 + self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.k,)) + 2 
        spd_time = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        alpha_X = self.prior["alpha_X"] if not isinstance(self.prior["alpha_X"], Distribution) else sample("alpha_X", self.prior["alpha_X"], sample_shape=(self.k, ))
        spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, self.L_X, self.M_X)))(alpha_X)
        spd = jnp.einsum("tk, km -> mtk", spd_time, spd_X)
        # sigma_c_max = self.prior["sigma_c"] if not isinstance(self.prior["sigma_c"], Distribution) else sample("sigma_c", self.prior["sigma_c"], sample_shape=(self.k,))
        # sigma_t_max = self.prior["sigma_t"] if not isinstance(self.prior["sigma_t"], Distribution) else sample("sigma_t", self.prior["sigma_t"], sample_shape=(self.k,))
        sigma_c_max = offsets["c_max_var"]
        sigma_t_max = offsets["t_max_var"]
        sigma_boundary_r = self.prior["sigma_boundary_r"] if not isinstance(self.prior["sigma_boundary_r"], Distribution) else sample("sigma_boundary_r", self.prior["sigma_boundary_r"], sample_shape=(self.k,))
        sigma_boundary_l = self.prior["sigma_boundary_l"] if not isinstance(self.prior["sigma_boundary_l"], Distribution) else sample("sigma_boundary_l", self.prior["sigma_boundary_l"], sample_shape=(self.k,))
        boundary_r_raw = self.prior["boundary_r"] if not isinstance(self.prior["boundary_r"], Distribution) else sample("boundary_r", self.prior["boundary_r"] , sample_shape=(self.m, self.k))
        boundary_l_raw = self.prior["boundary_l"] if not isinstance(self.prior["boundary_l"], Distribution) else sample("boundary_l", self.prior["boundary_l"] , sample_shape=(self.m, self.k))
        t_max_raw = self.prior["t_max_raw"] if not isinstance(self.prior["t_max_raw"], Distribution) else sample("t_max_raw", self.prior["t_max_raw"], sample_shape=(self.m, self.k))
        c_max_raw = self.prior["c_max"] if not isinstance(self.prior["c_max"], Distribution) else sample("c_max", self.prior["c_max"] , sample_shape=(self.m, self.k))
        ls_t_max = self.prior["lengthscale_t_max"] if not isinstance(self.prior["lengthscale_t_max"], Distribution) else sample("lengthscale_t_max", self.prior["lengthscale_t_max"], sample_shape=(self.r,))
        ls_c_max = self.prior["lengthscale_c_max"] if not isinstance(self.prior["lengthscale_c_max"], Distribution) else sample("lengthscale_c_max", self.prior["lengthscale_c_max"], sample_shape=(self.r,))
        ls_boundary_l = self.prior["lengthscale_boundary_l"] if not isinstance(self.prior["lengthscale_boundary_l"], Distribution) else sample("lengthscale_boundary_l", self.prior["lengthscale_boundary_l"], sample_shape=(self.r,))
        ls_boundary_r = self.prior["lengthscale_boundary_r"] if not isinstance(self.prior["lengthscale_boundary_r"], Distribution) else sample("lengthscale_boundary_r", self.prior["lengthscale_boundary_r"], sample_shape=(self.r,))

        spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_t_max, self.L_X, self.M_X)))(sigma_c_max)
        spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_c_max, self.L_X, self.M_X)))(sigma_t_max)
        spd_boundary_r = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_r, self.L_X, self.M_X)))(sigma_boundary_l)
        spd_boundary_l = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, ls_boundary_l, self.L_X, self.M_X)))(sigma_boundary_r)

        if num_gaussians > 0:
            sigmas = self.prior["sigma"] if not isinstance(self.prior["sigma"], Distribution) else sample("sigma", self.prior["sigma"], sample_shape=(num_gaussians,))
            expanded_sigmas = jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
        if num_beta > 0:
            sigma_beta = self.prior["sigma_beta"] if not isinstance(self.prior["sigma_beta"], Distribution) else sample("sigma_beta", self.prior["sigma_beta"], sample_shape = (num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = 1 / self.prior["sigma_negative_binomial"] if not isinstance(self.prior["sigma_negative_binomial"], Distribution) else 1 / sample("sigma_negative_binomial", self.prior["sigma_negative_binomial"], sample_shape=(num_neg_bins, ))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        

        sigma_ar = self.prior["sigma_ar"] if not isinstance(self.prior["sigma_ar"], Distribution) else sample("sigma_ar", self.prior["sigma_ar"], sample_shape=(self.k,1))
        
        rho_ar = self.prior["rho_ar"] if not isinstance(self.prior["rho_ar"], Distribution) else sample("rho_ar", self.prior["rho_ar"], sample_shape=(self.k,1))
        # sigma_ar = offsets["avg_sd"][..., None]
        # rho_ar = offsets["rho"][..., None]
        # Standard normals: shape (K, N, T)
        z = self.prior["beta_ar"] if not isinstance(self.prior["beta_ar"], Distribution) else sample("beta_ar", self.prior["beta_ar"], sample_shape=(self.j, self.k, self.n))
        AR_0_raw = self.prior["AR_0"] if not isinstance(self.prior["AR_0"], Distribution) else sample("AR_0", self.prior["AR_0"], sample_shape=(self.k, self.n))
        AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
        def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
        _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = z)
        AR = jnp.transpose(AR, (1,2,0))
        ### encoder
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.k * self.j, self.r))
        W_eff, Y_linearized = offsets["W_eff"], offsets["Y_linearized"]
        X = jnp.tanh(jnp.dot(Y_linearized, W) / W_eff.sum(axis=1, keepdims=True))
        psi_x = eigenfunctions_multivariate(X, self.L_X, self.M_X)
        t_max = make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + offsets["t_max"]  if not prior else numpyro.deterministic("t_max", make_psi_gamma(psi_x, t_max_raw * sigma_t_max.T)   + offsets["t_max"])
        c_max = make_psi_gamma(psi_x, c_max_raw * spd_c_max.T)  + offsets["c_max"]

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw  * spd_boundary_r.T) + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_l_raw * spd_boundary_l.T) + offsets["boundary_l"])
        if prior:
            c_max = numpyro.deterministic("c_max_", c_max)        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m , M_time, self.k))
        weights *= spd

        intercept = jnp.transpose(c_max)[..., None]
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
        mu = intercept + gamma_phi_gamma_x  if not prior else numpyro.deterministic("mu", intercept + gamma_phi_gamma_x)
        for family in data_set:
            k_indices = data_set[family]["indices"]
            de_trend = data_set[family]["de_trend"]
            linear_predictor = mu[k_indices] + AR[k_indices]
            exposure = data_set[family]["exposure"]
            obs = data_set[family]["Y"]
            mask = data_set[family]["mask"]
            if family == "gaussian":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask)
            elif family == "poisson":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                dist = Poisson(jnp.exp(log_rate)).mask(mask)
            elif family == "negative-binomial":
                log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
                dist = NegativeBinomial2(mean = jnp.exp(log_rate), concentration = expanded_sigma_neg_bin).mask(mask)
            elif family == "binomial":
                rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BinomialLogits(logits = rate , total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta-binomial":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaBinomial(concentration0=(1-jsci.special.expit(logit_rate)), concentration1= jsci.special.expit(logit_rate), total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask)
            elif family == "beta":
                logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
                dist = BetaProportion(jsci.special.expit(logit_rate), jnp.where(mask, exposure, 1.0) * expanded_sigma_beta).mask(mask)
            y = sample(f"likelihood_{family}", dist, obs) if not prior else sample(f"likelihood_{family}", dist, obs=None)      

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning)
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state
        
        state = svi.init(jax.random.PRNGKey(0), **model_args)
        if initial_state is not None:
            # your unpickled SVIState object
            state = ser.from_bytes(state, initial_state)

        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        return result.params, result.state
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000)):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape)