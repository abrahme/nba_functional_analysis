import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsci
import numpyro.distributions as dist
from numpyro.infer.hmc import HMC
from numpyro.distributions import constraints
from jax import random
import jax
from .hsgp import  diag_spectral_density, make_psi_gamma, make_convex_phi, make_convex_phi_prime, vmap_make_convex_phi, vmap_make_convex_phi_prime, vmap_make_convex_phi_double_prime, vmap_make_convex_phi_triple_prime, eigenfunctions, eigenfunctions_multivariate


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





def orthogonalize_ar(ar_raw, Q):
    QtQ = jnp.einsum("...td, ...tj -> ...dj", Q, Q)
    alpha = jnp.linalg.solve(QtQ, jnp.einsum("...td, ...t -> ...d", Q, ar_raw))
    return ar_raw - jnp.einsum("...tm, ...m -> ...t", Q, alpha)



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
    wTx = jnp.einsum("...nr, ...mr -> ...nm", X, W * jnp.sqrt(ls[..., None, None]))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("...nm, mk -> ...nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = (jnp.einsum("...nm, mk -> ...nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)

    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    second_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = gamma_phi_gamma_x + intercept
    return wTx, mu, t_max, c_max, 0, second_deriv, third_deriv, first_deriv


def make_mu_mcmc_fixed_X(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict,beta_ar, sigma_ar, rho_ar, AR_0_raw, phi_time):
    spd = jax.vmap(jax.vmap(lambda a, l: jnp.sqrt(diag_spectral_density(1, a, l, L_time, M_time))))(alpha_time, ls_deriv)
    # spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    weights = weights * spd[..., None, :, :]
    # weights *= spd
    wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(ls))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("nm, mk -> nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = (jnp.einsum("nm, mk -> nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    intercept = jnp.transpose(c_max)[..., None]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)

    second_deriv = -1 * jnp.einsum("nm, ...mdk, nkdz, ...jzk, nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, ...mdk, nkdz, ...jzk, nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("nm, ...mdk, nktdz, ...jzk, nj -> ...knt", psi_x, weights, phi_prime_t_max[:,:, None, ...] - phi_prime_t[ None, None], weights, psi_x)
    gamma_phi_gamma_x = jnp.einsum("nm, ...mdk, nktdz, ...jzk, nj -> ...knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = gamma_phi_gamma_x + intercept

    AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
    def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
    _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = jnp.transpose(beta_ar, (2, 0, 1, 3, 4)))
    AR = jnp.transpose(AR, (1,2,3,4,0))

    return wTx, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def make_mu_rflvm_mcmc_AR(X, ls_deriv, alpha_time, weights, W, W_t_max, W_c_max, ls, ls_t_max, ls_c_max, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict, rank, beta_ar, sigma_ar, rho_ar, AR_0_raw, phi_time, orthogonalize = False):
    spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    # weights = weights * spd[..., None, :, :]
    weights *= spd.T[None]
    wTx = jnp.einsum("...nr, mr -> ...nm", X, W * jnp.sqrt(ls))   
    psi_x = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], -1) * (1/ jnp.sqrt(rank ) )
    wTx_t_max = jnp.einsum("...nr, mr -> ...nm", X, W_t_max * jnp.sqrt(ls_t_max))
    psi_x_t_max = jnp.concatenate([jnp.cos(wTx_t_max), jnp.sin(wTx_t_max)], axis = -1) * (1/ jnp.sqrt(rank)) 
    wTx_c_max = jnp.einsum("...nr, mr -> ...nm", X, W_c_max * jnp.sqrt(ls_c_max))
    psi_x_c_max = jnp.concatenate([jnp.cos(wTx_c_max), jnp.sin(wTx_c_max)], axis = -1) * (1/ jnp.sqrt(rank))
    t_max =  jnp.tanh(jnp.einsum("ijnm, m... -> ijn...", psi_x_t_max, t_max_raw * sigma_t_max)  + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = jnp.einsum("ijnm, m... -> ijn...",psi_x_c_max, c_max * sigma_c_max)  + offset_dict["c_max"]
    # intercept = jnp.transpose(c_max)[..., None]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
    def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
    _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = jnp.transpose(beta_ar, (2, 0, 1, 3, 4)))
    AR = jnp.transpose(AR, (1,2,3,4,0))
    second_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = (gamma_phi_gamma_x + intercept) 
    if orthogonalize:
        Q = jnp.stack([mu, jnp.ones_like(mu)], axis = -1)
        AR = orthogonalize_ar(AR, Q)
    return wTx, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def make_mu_mcmc_AR_fixed_X(X, ls_deriv, alpha_time, weights, W, ls, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, shifted_x_time, offset_dict, beta_ar, sigma_ar, rho_ar, phi_time):
    # spd = jax.vmap(jax.vmap(lambda a, l: jnp.sqrt(diag_spectral_density(1, a, l, L_time, M_time))))(alpha_time, ls_deriv)
    spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
    # weights = weights * spd[..., None, :, :]
    weights *= spd
    wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(ls))  
    psi_x = jnp.concatenate([np.cos(wTx), np.sin(wTx)],-1) * (1/ jnp.sqrt(W.shape[0]))
    t_max = jnp.tanh(jnp.einsum("nm, mk -> nk", psi_x, t_max_raw, optimize = True) * sigma_t_max) * 5  + offset_dict["t_max"]  
    c_max = (jnp.einsum("nm, mk -> nk", psi_x, c_max, optimize = True)) * sigma_c_max + offset_dict["c_max"]
    intercept = jnp.transpose(c_max)[..., None]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)

    # intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    time_delta = (shifted_x_time[None] - shifted_x_time[..., None])[None, None]
    kernel = (1 / (1 - jnp.square(rho_ar[..., None]))) * (rho_ar[..., None] ** jnp.abs(time_delta[None])) * sigma_ar[..., None] 
    L = jnp.linalg.cholesky(kernel)  # (K, T, T)
    # Apply Cholesky: result (K, N, T)
    second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...nk", psi_x, weights, phi_prime_t_max[:,:, None, ...] - phi_prime_t[ None, None], weights, psi_x)
    AR = jnp.einsum('...ktd,...knd->...knt', L, beta_ar)
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = (gamma_phi_gamma_x + intercept)[None, None] 
    return wTx, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def make_mu_rflvm(X, ls_deriv, alpha_time, weights, W, W_t_max, W_c_max, ls,ls_t_max, ls_c_max, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, offset_dict):
    spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    weights = weights * spd.T[None] 
    wTx = jnp.einsum("nr, mr -> nm", X, W * jnp.sqrt(ls))   
    psi_x = jnp.hstack([jnp.cos(wTx), jnp.sin(wTx)]) * (1/ jnp.sqrt(W.shape[0])) 
    wTx_t_max = jnp.einsum("nr, mr -> nm", X, W_t_max * jnp.sqrt(ls_t_max))
    psi_x_t_max = jnp.concatenate([jnp.cos(wTx_t_max), jnp.sin(wTx_t_max)], axis = -1) * (1/ jnp.sqrt(W_t_max.shape[0]))   
    wTx_c_max = jnp.einsum("nr, mr -> nm", X, W_c_max * jnp.sqrt(ls_c_max))
    psi_x_c_max = jnp.concatenate([jnp.cos(wTx_c_max), jnp.sin(wTx_c_max)], axis = -1) * (1/ W_c_max.shape[0])
    t_max =  jnp.tanh(make_psi_gamma(psi_x_t_max, t_max_raw * sigma_t_max)  + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = make_psi_gamma(psi_x_c_max, c_max * sigma_c_max)  + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time), np.squeeze(L_time), M_time)
    intercept = jnp.transpose(c_max)[..., None]
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("nm, mdk, nktdz, jzk, nj -> nk", psi_x, weights, phi_prime_t_max[:,:, None, ...] - phi_prime_t[ None, None], weights, psi_x)
    
    return mu, t_max, c_max, 0, second_deriv, third_deriv, first_deriv

def make_mu_hsgp(X, ls_deriv, alpha_time, alpha_X, weights, ls, ls_c_max, ls_t_max, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, offset_dict, rank, L_X, M_X ):
    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls, L_X, M_X)))(alpha_X)
    spd = jnp.einsum("tk, km -> mtk", spd_time, spd_X)
    spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls_c_max, L_X, M_X)))(sigma_c_max)
    spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls_t_max, L_X, M_X)))(sigma_t_max)
    weights = weights * spd  
    psi_x = eigenfunctions_multivariate(X, L_X, M_X)
    t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = make_psi_gamma(psi_x, c_max * spd_c_max.T)  + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)
    intercept = jnp.transpose(c_max)[..., None]
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("nm, mdk, nktdz, jzk, nj -> nk", psi_x, weights, phi_prime_t_max[:,:, None, ...] - phi_prime_t[ None, None], weights, psi_x)
    
    return mu, t_max, c_max, 0, second_deriv, third_deriv, first_deriv

def make_mu_linear(X, ls_deriv, alpha_time, weights, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, rank, offset_dict):
    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    weights = weights * spd_time.T[None] 
    psi_x = X
    t_max = jnp.tanh(make_psi_gamma(psi_x, t_max_raw * sigma_t_max)   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = make_psi_gamma(psi_x, c_max * sigma_c_max)  + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
    phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
    phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
    phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time), jnp.squeeze(L_time), M_time)
    intercept = jnp.transpose(c_max)[..., None]
    gamma_phi_gamma_x = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights, phi_t_max[:,:,None,...] - phi_time[None, None] +  phi_prime_t_max[:, :, None, ...] * (((shifted_x_time - L_time)[None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> kn", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("nm, mdk, nktdz, jzk, nj -> nk", psi_x, weights, phi_prime_t_max[:,:, None, ...] - phi_prime_t[ None, None], weights, psi_x)
    return mu, t_max, c_max, 0, second_deriv, third_deriv, first_deriv


def make_mu_linear_mcmc_AR(X, ls_deriv, alpha_time, weights, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time,rank,  offset_dict, beta_ar, sigma_ar, rho_ar, AR_0_raw, orthogonalize = False):
    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    weights = weights * spd_time.T[None] 
    psi_x = X
    t_max = jnp.tanh(jnp.einsum("ijnm, m... -> ijn...", X, t_max_raw * sigma_t_max)   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = jnp.einsum("ijnm, m... -> ijn...", X, c_max * sigma_c_max)  + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time), jnp.squeeze(L_time), M_time)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
    def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
    _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = jnp.transpose(beta_ar, (2, 0, 1, 3, 4)))
    AR = jnp.transpose(AR, (1,2,3,4,0))


    return None, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def make_mu_hsgp_mcmc_AR(X, ls_deriv, alpha_time, alpha_X, weights, ls, ls_c_max, ls_t_max, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time, offset_dict, rank, L_X, M_X , beta_ar, sigma_ar, rho_ar, AR_0_raw, orthogonalize = False):

    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    spd_X = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls, L_X, M_X)))(alpha_X)
    spd = jnp.einsum("tk, km -> mtk", spd_time, spd_X)
    spd_c_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls_c_max, L_X, M_X)))(sigma_c_max)
    spd_t_max = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(rank, alpha, ls_t_max, L_X, M_X)))(sigma_t_max)
    weights *= spd
    psi_x = jax.vmap(jax.vmap(lambda z: eigenfunctions_multivariate(z, L_X, M_X)))(X)

    t_max = jnp.tanh(jnp.einsum("ijnm, m... -> ijn...", psi_x, t_max_raw * spd_t_max.T)   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = jnp.einsum("ijnm, m... -> ijn...",psi_x, c_max * spd_c_max.T)  + offset_dict["c_max"]
    # intercept = jnp.transpose(c_max)[..., None]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(np.squeeze(shifted_x_time), np.squeeze(L_time), M_time)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
    def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
    _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = jnp.transpose(beta_ar, (2, 0, 1, 3, 4)))
    AR = jnp.transpose(AR, (1,2,3,4,0))
    second_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = (gamma_phi_gamma_x + intercept) 
    if orthogonalize:
        Q = jnp.stack([mu, jnp.ones_like(mu)], axis = -1)
        AR = orthogonalize_ar(AR, Q)
    return None, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def compute_residuals_map(map_values, obs_values, exposures, metric_output, gaussian_var = None, nb_variance = None, beta_binom_variance = None, beta_variance = None):
    residuals = []
    rho = []
    gaussian_index = 0
    negative_binom_index = 0
    for index, metric_type in enumerate(metric_output):
        if metric_type == "gaussian":
            resid = map_values[..., index] - obs_values[..., index]
            weight = 1/jnp.square(gaussian_var[gaussian_index]/exposures[index])
            gaussian_index += 1
        elif metric_type in ["poisson", "negative-binomial"]:
            scaled_obs_val = jnp.log(obs_values[..., index] / 36)
            multiplier = 0
            if metric_type == "negative-binomial":
                multiplier =  nb_variance[negative_binom_index]
            resid = (scaled_obs_val - jnp.log(map_values[..., index] / 36))
            weight = 1 / (multiplier / (jnp.exp(exposures[index])))
            if metric_type == "negative-binomial":
                negative_binom_index += 1
        elif metric_type in ["binomial", "beta-binomial"]:
            scaled_obs_val = jsci.logit(obs_values[..., index])
            multiplier = 1
            if metric_type == "beta-binomial":
                multiplier = (exposures[index] + beta_binom_variance)/(1 + beta_binom_variance)
            resid = (scaled_obs_val - jsci.logit(map_values[..., index]))
            weight =  (exposures[index]) / multiplier
        elif metric_type in ["beta"]:
            scaled_obs_val = jsci.logit(obs_values[..., index] /48)
            multiplier = exposures[index] * beta_variance
            resid = (scaled_obs_val - jsci.logit(map_values[..., index]/48))
            weight = 1 / multiplier
        
        residual_weighted = jnp.square(resid) * weight
        mask = jnp.isfinite(residual_weighted)
        avg_residuals = jnp.nanmean(jnp.sum(mask * residual_weighted, axis = -1) / jnp.sum(mask * weight, axis = -1))
        residuals.append(avg_residuals)
        autocorr_resid_weighted = jnp.sqrt(weight[..., 1:] * weight[..., 0:-1]) * resid[..., 1:]*resid[..., 0:-1]
        mask_autocorr = jnp.isfinite(autocorr_resid_weighted)
        autocorr =  jnp.nanmean(jnp.sum(mask_autocorr * autocorr_resid_weighted, axis = -1) / jnp.sum(mask * residual_weighted, axis = -1))
        rho.append(autocorr)
    
    return jnp.sqrt(jnp.array(residuals) * (1 - jnp.square(jnp.array(rho)))), jnp.array(rho)

def compute_priors(Y, exposures, metric_output, exposure_list=None):
    """
    Compute priors on max value and peak age, returning separate arrays for means and variances.

    Args:
        Y: list or array [num_metrics, ...] of observed values
        exposures: list or array of exposure weights, same shape as Y
        metric_output: list of families per metric ("gaussian", "poisson", "negative-binomial", "binomial", "beta", ...)
        exposure_list: optional list for special handling ("simple_exposure")
    
    Returns:
        prior_max_mean: list of max value means on natural parameter scale
        prior_max_var: list of max value variances
        prior_peak_mean: list of peak indices
        prior_peak_var: list of peak variances
    """
    prior_max_mean = []
    prior_max_var = []
    prior_peak_mean = []
    prior_peak_var = []
    
    for idx, family in enumerate(metric_output):
        weight = exposures[idx] if exposure_list[idx] != "simple_exposure" else exposures[exposure_list.index("minutes")]
        individual_weight = jnp.nansum(weight, axis=-1)

        # weighted mean of raw values (not needed for priors, optional)
        weighted_sum = jnp.nansum(Y[idx] * weight)
        total_weight = jnp.nansum(weight)
        p_mean = weighted_sum / total_weight

        # scale observations depending on family
        Y_scaled = Y[idx]
        if family in ["poisson", "negative-binomial"]:
            Y_scaled = Y[idx] / jnp.exp(exposures[idx])
            individual_weight = jnp.nansum(jnp.exp(exposures[idx]), axis = -1)
        elif family in ["binomial", "beta-binomial", "bernoulli", "beta"]:
            Y_scaled = Y[idx] / exposures[idx]

        max_per_obs = jnp.nanmax(Y_scaled, axis=-1)
        peak_idx = jnp.nanargmax(Y_scaled, axis=-1)

        # weighted max and peak
        p_max = jnp.nansum(max_per_obs * individual_weight) / jnp.nansum(individual_weight)
        peak = jnp.nansum(peak_idx * individual_weight) / jnp.nansum(individual_weight)

        # weighted variance of max and peak
        p_max_var = jnp.nansum(individual_weight * (max_per_obs - p_max)**2) / jnp.nansum(individual_weight)
        peak_var = jnp.nansum(individual_weight * (peak_idx - peak)**2) / jnp.nansum(individual_weight)

        # convert p_max to natural parameter
        if family == "gaussian":
            mu_eta = p_max
            tau2 = p_max_var 
        elif family in ["poisson", "negative-binomial"]:
            tau2 = jnp.log(1 + p_max_var / (p_max**2 + 1e-12))
            mu_eta = jnp.log(p_max + 1e-12) - 0.5 * tau2
        elif family in ["binomial", "bernoulli", "beta", "beta-binomial"]:
            p_max_clip = jnp.clip(p_max, 1e-6, 1 - 1e-6)
            mu_eta = jnp.log(p_max_clip / (1 - p_max_clip)) if exposure_list[idx] != "simple_exposure" else 0
            tau2 = p_max_var / (p_max_clip**2 * (1 - p_max_clip)**2 + 1e-12) if exposure_list[idx] != "simple_exposure" else .1
        else:
            raise ValueError(f"Unknown family: {family}")

        prior_max_mean.append(mu_eta)
        prior_max_var.append(tau2)
        prior_peak_mean.append(peak)
        prior_peak_var.append(peak_var)
    return jnp.array(prior_max_mean), jnp.array(prior_max_var), jnp.array(prior_peak_mean), jnp.array(prior_peak_var)
