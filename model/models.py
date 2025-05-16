from abc import abstractmethod, ABC
import jax 
import numpyro
import numpy as np
from numpyro import sample 
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, Binomial, Dirichlet, MultivariateNormal, BetaProportion, Distribution, Uniform, Beta, Gamma, BinomialLogits
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive, init_to_value, init_to_feasible, init_to_uniform, init_to_mean
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.autoguide import AutoDelta, AutoBNAFNormal
import optax
from optax import linear_onecycle_schedule, adam
from numpyro.optim import optax_to_numpyro
from .hsgp import make_convex_f, make_psi_gamma, diag_spectral_density, make_convex_phi, make_convex_phi_prime, vmap_make_convex_phi, vmap_make_convex_phi_prime, make_psi_gamma_kron
import jax.numpy as jnp
import jax.scipy as jsci
from .MultiHMCGibbs import MultiHMCGibbs


def step_decay_schedule(init_lr, drop_every=10000, drop_factor=10, total_steps=100000):
    num_drops = total_steps // drop_every
    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(init_lr / (drop_factor ** i))
            for i in range(num_drops + 1)
        ],
        boundaries=[drop_every * i for i in range(1, num_drops + 1)]
    )



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
        self.gaussian_variance_indices = jnp.array(self.gaussian_variance_indices)
        self.gaussian_indices = (self.output == 1) & self.missing
        self.poisson_indices = (self.output == 2) & self.missing
        self.binomial_indices = (self.output == 3) & self.missing

    
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
        U = sample("U", self.prior["U"], sample_shape=(self.n, self.r))
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

    def run_inference(self, num_steps, initial_values:dict = {}):
        optimizer = adam(learning_rate = linear_onecycle_schedule(100, .5))
        svi = SVI(self.model_fn, AutoDelta(self.model_fn, prefix=""),
                   optim=optimizer,
                     loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps, init_params=initial_values)
        return result

    def predict(self, posterior_samples: dict, model_args):
        predictive = Predictive(self.model_fn, posterior_samples, **model_args)
        return predictive(jax.random.PRNGKey(0))["obs"]


class NBANormalApproxProbabilisticCPDecomposition(ProbabilisticCPDecomposition):

    def __init__(self, X, rank: int, M, E) -> None:
        super().__init__(X, rank)
        assert (self.X.shape == E.shape) & (self.X.shape == M.shape) ### have to have same shape
        self.exposure = E ### exposure for each element in X
        self.missing = M ### matrix of 1 / 0 indicating missing or not
        
    
    def initialize_priors(self) -> None:
        ### initialize U
        self.prior["U"] = Normal()
        ### initialize V
        self.prior["V"] = Normal()
        ### initialize W
        self.prior["W"] = Normal()
        ### initialize lambda
        self.prior["lambda"] = Dirichlet(concentration=jnp.ones(shape=(self.r,)) / self.r)

    def model_fn(self) -> None:
        V = sample("V", self.prior["V"], sample_shape=(self.t, self.r)) 
        U = sample("U", self.prior["U"], sample_shape=(self.n, self.r )) 
        W = sample("W", self.prior["W"], sample_shape=(self.k, self.r)) 
        weights = sample("lambda", self.prior["lambda"])
        core = jnp.einsum("ntr, kr->ntkr", jnp.einsum("nr, tr -> ntr", U, V), W)
        y = jnp.einsum("ntkr, r -> ntk", core, weights)
        y_normal = sample("likelihood_normal", Normal(loc = y[self.missing].flatten(), 
                                                      scale=1/self.exposure[self.missing].flatten()), 
                          obs = self.X[self.missing].flatten())
        

    def run_inference(self, num_steps, initial_values:dict = {}):
        optimizer = adam(learning_rate = linear_onecycle_schedule(100, .5))
        svi = SVI(self.model_fn, AutoDelta(self.model_fn, prefix=""),
                   optim=optimizer,
                     loss=Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps, init_params=initial_values)
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
            mcmc.run(key, **model_args, extra_fields=("potential_energy",))
            return mcmc.get_samples(group_by_chain=True), mcmc.get_extra_fields(group_by_chain=True)
    @abstractmethod
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoBNAFNormal(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.003), loss=Trace_ELBO(), 
                  )
        print("Setup SVI")

        
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result, guide
    
    @abstractmethod
    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args:dict = {}):
        svi_result, guide = self.run_svi_inference(num_steps, guide_kwargs, model_args)
        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(self.model_fn)
        mcmc = MCMC(
        NUTS(neutra_model, init_strategy=init_to_median),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        chain_method="parallel"
    )
        mcmc.run(jax.random.PRNGKey(0), **model_args)
        return mcmc, neutra
    

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

    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values)

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
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values)

    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)

class IFTVRFLVM(TVRFLVM):
    """
    model for time varying functional 
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)

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
        beta = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta",  Normal(), sample_shape=(num_metrics, 2 * self.m, 10, 2)) * ls 
        ifft_beta = jnp.fft.irfftn(beta[..., 0] + 1j * beta[..., -1], axes = [-1], norm = "ortho", s = [self.j])
        mu = jnp.einsum("nm,kmj -> knj", phi, ifft_beta)
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

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values)

    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

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
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn = init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0),
                          num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result.params


    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)

class ConvexKronTVRFLVM(ConvexTVRFLVM):
    def __init__(self, latent_rank_1: int, rff_dim_1: int, latent_rank_2: int,  num_metrics:int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank_1, rff_dim_1, output_shape, basis)
        self.r2 = latent_rank_2
        self.k = num_metrics
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["metric_factor"] = Dirichlet(concentration=jnp.ones(shape=(self.r2,)) / self.r2)
        self.prior["metric_scale"] = Gamma(concentration=2, rate=3)
    def model_fn(self, data_set, hsgp_params, offsets = 0, prior = False) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self.prior["lengthscale"] if not isinstance(self.prior["lengthscale"], Distribution) else sample("lengthscale", self.prior["lengthscale"], sample_shape=(self.r,))
        
        W = self.prior["W"] if not isinstance(self.prior["W"], Distribution) else sample("W", self.prior["W"], sample_shape=(self.m,self.r))
        metric_factor = self.prior["metric_factor"] if not isinstance(self.prior["metric_factor"], Distribution) else sample("metric_factor", self.prior["metric_factor"], sample_shape=(self.k,))
        metric_scale = self.prior["metric_scale"] if not isinstance(self.prior["metric_scale"], Distribution) else sample("metric_scale", self.prior["metric_scale"], sample_shape=(self.k,))
        X = self.prior["X"] if not isinstance(self.prior["X"], Distribution) else sample("X", self.prior["X"])
        X -= jnp.mean(X, keepdims = True, axis = 0)
        X /= jnp.std(X, keepdims = True, axis = 0)


        wTx = jnp.einsum("nr,mr -> nm", X, W  * jnp.sqrt(lengthscale)[None])
 

        psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis = -1) * (1/ jnp.sqrt(self.m))
       


       
        slope = make_psi_gamma(psi_x, self.prior["slope"] if not isinstance(self.prior["slope"], Distribution) else sample("slope", self.prior["slope"], sample_shape=(self.m*2, self.r2))) 
        ls_deriv = self.prior["lengthscale_deriv"] if not isinstance(self.prior["lengthscale_deriv"], Distribution) else sample("lengthscale_deriv", self.prior["lengthscale_deriv"], sample_shape=(self.r2,))
        intercept = make_psi_gamma(psi_x, self.prior["intercept"] if not isinstance(self.prior["intercept"], Distribution) else sample("intercept", self.prior["intercept"] , sample_shape=(2 * self.m, self.r2))) 
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(self.r2, ))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, self.r2))
        weights = weights * spd * .0001
        gamma_phi_gamma_x = jnp.einsum("nm, mad , taz, lzd, nl -> ndt", psi_x, weights, phi_time, weights, psi_x)
        mu_core = make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept)[..., None])  
        mu = jnp.einsum("dnt, kd -> knt", mu_core, metric_factor * metric_scale[..., None]) + jnp.transpose(offsets)[..., None] if not prior else numpyro.deterministic("mu",jnp.einsum("dnt, dk -> knt", mu_core, metric_factor * metric_scale[..., None]) + jnp.transpose(offsets)[..., None])
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
            y = sample(f"likelihood_{family}", dist, obs[mask]) if not prior else sample(f"likelihood_{family}", dist, obs = None)
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values)
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values)
    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

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
        self.prior["sigma_c"] = InverseGamma(2.0, .3)
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
        alpha_time = self.prior["alpha"] if not isinstance(self.prior["alpha"], Distribution) else sample("alpha", self.prior["alpha"], sample_shape=(num_metrics, ))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self.prior["beta"] if not isinstance(self.prior["beta"], Distribution) else sample("beta", self.prior["beta"], sample_shape=(self.m * 2, M_time, num_metrics))
        weights =  weights * spd 
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
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn = init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params=initial_values, stable_update=True, **model_args)
        return result.params


    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

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
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.000003), loss=Trace_ELBO(num_particles=10),
                  )
        print("Setup SVI")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_params=initial_values, **model_args)
        return result.params


    def run_neutra_inference(self, num_warmup, num_samples, num_chains, num_steps, guide_kwargs: dict = {}, model_args: dict = {}):
        return super().run_neutra_inference(num_warmup, num_samples, num_chains, num_steps, guide_kwargs, model_args)

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

class GibbsConvexVRFLVM(ConvexTVRFLVM):
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
    

class GibbsIFTVRFLVM(IFTVRFLVM):
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
    