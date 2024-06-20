import jax.numpy as jnp 
from scipy.spatial.distance import cdist
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.distributions import Normal, Poisson, Binomial
jax.config.update('jax_platform_name', 'cpu')

def varimax(Phi, gamma = 1, q = 20):
    p,k = Phi.shape
    R = jnp.eye(k)
    for _ in range(q):
        Lambda = jnp.dot(Phi, R)
        u,s,vh = jnp.linalg.svd(jnp.dot(Phi.T,(Lambda)**3 - (gamma/p) * jnp.dot(Lambda, jnp.diag(jnp.diag(jnp.dot(Lambda.T,Lambda))))))
        R = jnp.dot(u,vh)
    
    return jnp.dot(Phi, R)

def select_pivot(Phi) -> int:
    """
    chooses a pivot from a set of matrices (phi should have shape n x k x m)
    """
    S = jnp.stack([jnp.linalg.svd(x, full_matrices = False)[1] for x in Phi], axis = 0)
    S_max = jnp.max(S, axis = 1)
    S_min = jnp.min(S, axis = 1)
    condition = S_max / S_min
    pivot = jnp.where(condition == jnp.quantile(condition, .5, method="nearest"))
    return pivot[0][0]

def match_permutation(Phi, pivot):
    """
    this matches the permutations and sign of the pivot to the varimax rotated phi
    Phi has shape D x k, pivot has same shape as Phi
    k is number of columns
    """

    max_norm_indices = (np.linalg.norm(Phi,axis=0)).argsort().argsort()
    phi_dist_pos = cdist(Phi.T, pivot.T)
    phi_dist_neg = cdist(Phi.T, -1*pivot.T)
    phi_min_dist = np.minimum(phi_dist_pos, phi_dist_neg)
    phi_min_arg = 2*(phi_dist_pos <= phi_dist_neg) - 1
    new_Phi = np.zeros_like(Phi)

    for index in max_norm_indices:
        opt_index = np.argmin(phi_min_dist[index])
        new_Phi[:, index] = Phi[:, opt_index] * phi_min_arg[index, opt_index]
        phi_min_dist[index, opt_index] = np.inf
    
    return new_Phi

def match_align(Phi):
    """
    stacked sequence of T elements
    """
    Phi_tilde = jnp.stack([varimax(x) for x in Phi], axis = 0)
    Phi_pivot = Phi[select_pivot(Phi)]
    Phi_star = jnp.stack([match_permutation(x, Phi_pivot) for x in Phi_tilde])

    return Phi_star


def create_metric_trajectory(posterior_mean_samples, player_index, observations, exposures, metric_outputs: list[str], posterior_variance_samples = None):
    key = jax.random.key(0)
    gaussian_index = 0
    posteriors = []
    obs_normalized = []

    for metric_index, metric_output in enumerate(metric_outputs):
        exposure = exposures[metric_index, ...]
        obs = observations[player_index, metric_index, :]
        post = posterior_mean_samples[..., metric_index, :]
        inds = jnp.where(jnp.isnan(exposure))
        exposure = exposure.at[inds].set(np.take(np.nanmean(exposure, axis = 0), inds[1]))

        if metric_output == "gaussian":
            scale = jnp.einsum("cs,t -> cst", posterior_variance_samples[gaussian_index],  1.0 / exposure[player_index, :])
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            obs_normal = obs
            gaussian_index += 1
        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post + exposure[player_index, :]))
            posterior_predictions = 36.0 * (dist.sample(key = key) / jnp.exp(exposure[player_index, :])) ### per 36 min statistics
            obs_normal = 36.0 * (obs / jnp.exp(exposure[player_index, :]))
        elif metric_output == "binomial":
            dist = Binomial(total_count = jnp.round(exposure[player_index, :], decimals=0).astype(int), logits = post)
            posterior_predictions = dist.sample(key = key) / jnp.round(exposure[player_index, :], decimals=0) ### per shot
            obs_normal = obs / jnp.round(exposure[player_index, :], decimals=0) ### per shot
    
        posteriors.append(posterior_predictions)
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1)}  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack(posteriors, axis = -1)}  ### has shape (chains, draws, time, metrics)

    return obs_data, posterior_predictive












