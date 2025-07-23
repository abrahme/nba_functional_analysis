import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsci
import numpyro.distributions as dist
from numpyro.distributions import constraints
from jax import random
import jax
from .hsgp import  diag_spectral_density, make_psi_gamma, make_convex_phi, make_convex_phi_prime, vmap_make_convex_phi, vmap_make_convex_phi_prime, vmap_make_convex_phi_double_prime, vmap_make_convex_phi_triple_prime


class Type2Gumbel(dist.Distribution):
    support = constraints.positive
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    reparametrized_params = ['alpha', 'scale']
    
    def __init__(self, alpha, scale=1.0, validate_args=None):
        self.alpha = alpha
        self.scale = scale
        batch_shape = jnp.shape(jnp.broadcast_arrays(alpha, scale)[0])
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)
    
    def sample(self, key, sample_shape=()):
        # Inverse CDF method:
        u = random.uniform(key, shape=sample_shape + self.batch_shape)
        return self.scale / (-jnp.log(u))**(1.0 / self.alpha)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.alpha
        s = self.scale
        x = value
        log_unnormalized = jnp.log(alpha) + alpha * jnp.log(s) - (alpha + 1) * jnp.log(x)
        log_normalizer = (s / x) ** alpha
        return log_unnormalized - log_normalizer


@jax.jit
def contract_single_nkt_lazy(psi_n, psi_k, weights_mdk, weights_jzk,
                              ptm, pptm, phi_time_t, shifted, t_m):
    delta = (shifted - t_m)  # (d, 1)
    intermediate = ptm[None] - phi_time_t + jnp.einsum("dz ,t -> tdz",pptm , delta)  # (t, d, z)
    return jnp.einsum("m,md,tdz,jz,j->t", psi_n, weights_mdk, intermediate, weights_jzk, psi_k)

@jax.jit
def compute_nk_lazy(psi, w, ptm, pptm, phi_time, shifted, t_mx, n, k):
    psi_n = psi[n]                  # (M,)
                 # (J,) assumed same
    weights_mdk = w[:, :, k]        # (M, D)
        # (J, Z)

    ptm_nkt = ptm[n, k]             # (D, Z)
    pptm_nkt = pptm[n, k]           # (D, Z)
    t_m = t_mx[n, k]                # scalar

    return contract_single_nkt_lazy(
        psi_n, psi_n, weights_mdk, weights_mdk,
        ptm_nkt, pptm_nkt, phi_time, shifted, t_m
    )

@jax.jit
def compute_gamma_lazy_batched(psi_x, weights, phi_t_max, phi_prime_t_max,
                                phi_time, shifted_x_time, L_time, t_max):
    S, C, N, K = t_max.shape[:4]  # Sample, Chain, N, K
    shifted = shifted_x_time - L_time

    def process_single_sc(s, c):
        psi = psi_x[s, c]              # (N, M)
        w = weights[s, c]              # (M, D, K)
        ptm = phi_t_max[s, c]          # (N, K, D, Z)
        pptm = phi_prime_t_max[s, c]   # (N, K, D, Z)
        t_mx = t_max[s, c]             # (N, K)

        compute = lambda n, k: compute_nk_lazy(psi, w, ptm, pptm, phi_time, shifted, t_mx, n, k)
        return jax.vmap(jax.vmap(compute, in_axes=(None, 0)), in_axes=(0, None))(jnp.arange(N), jnp.arange(K))

    return jax.vmap(jax.vmap(process_single_sc, in_axes=(0, None)), in_axes=(None, 0))(jnp.arange(S), jnp.arange(C))









def transform_mu(mu, metric_outputs):
    transformed_mu = np.zeros_like(mu)
    for index, output_type in enumerate(metric_outputs):
        if output_type == "gaussian":
            transform_function = lambda x: x 
        elif output_type == "poisson":
            transform_function = lambda x: jnp.exp(x) 
        elif output_type == "binomial":
            transform_function = lambda x: jsci.expit(x)
        elif output_type == "beta":
            transform_function = lambda x: jsci.expit(x)
        transformed_mu[:,:,index,...] = transform_function(mu[:,:,index,...])

    return transformed_mu


def make_mu_mcmc(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict, phi_time):
    # spd = jax.vmap(jax.vmap(lambda a, l: jnp.sqrt(diag_spectral_density(1, a, l, L_time, M_time))))(alpha_time, ls_deriv)
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    # weights = weights * spd[..., None, :, :]
    weights *= spd
    wTx = jnp.einsum("...nr, mr -> ...nm", X, W * jnp.sqrt(ls))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("...nm, mk -> ...nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = (jnp.einsum("...nm, mk -> ...nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    # gamma_phi_gamma_x =  compute_gamma_lazy_batched(psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max)
    # mu = intercept + jnp.transpose(gamma_phi_gamma_x, (1,0, 3, 2, 4))
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = gamma_phi_gamma_x + intercept
    return wTx, mu, t_max, c_max

def make_mu_mcmc_AR(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict, beta_ar, sigma_ar, rho_ar, phi_time):
    # spd = jax.vmap(jax.vmap(lambda a, l: jnp.sqrt(diag_spectral_density(1, a, l, L_time, M_time))))(alpha_time, ls_deriv)
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    # weights = weights * spd[..., None, :, :]
    weights *= spd
    wTx = jnp.einsum("...nr, mr -> ...nm", X, W * jnp.sqrt(ls))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("...nm, mk -> ...nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = (jnp.einsum("...nm, mk -> ...nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    # intercept = jnp.transpose(c_max)[..., None]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    time_delta = (shifted_x_time[None] - shifted_x_time[..., None])[None, None]
    kernel = (1 / (1 - jnp.square(rho_ar[..., None]))) * (rho_ar[..., None] ** jnp.abs(time_delta[None])) * sigma_ar[..., None] 
    L = jnp.linalg.cholesky(kernel)  # (K, T, T)
    # Apply Cholesky: result (K, N, T)
    AR = jnp.einsum('...ktd,...knd->...knt', L, beta_ar)
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = (gamma_phi_gamma_x + intercept) + AR
    return wTx, mu, t_max, c_max, AR

def make_mu(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, offset_dict):
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    weights = weights * spd 
    wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(ls))   
    psi_x = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = make_psi_gamma(psi_x, c_max) * sigma_c_max + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    intercept = jnp.transpose(c_max)[..., None]
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    return mu, t_max, c_max, second_deriv, third_deriv
