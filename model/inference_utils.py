import jax.numpy as jnp 
from scipy.spatial.distance import cdist
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from numpyro.distributions import Normal, Poisson, Exponential, Bernoulli
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


def create_metric_trajectory(posterior_mean_samples, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str], posterior_variance_samples = None):
    key = jax.random.key(0)
    gaussian_index = 0
    minutes_index = metrics.index("minutes")
    retirement_index = metrics.index("retirement")  ### 1 -> playing, 0 --> retired

    ### first sample retirement
    post_retirement = posterior_mean_samples[..., retirement_index, :]
    posterior_predictions_retirement = Bernoulli(logits=post_retirement).sample(key = key) 
    obs_retirement = observations[player_index, retirement_index, :]

    #### then sample minutes 
    
    post_min = posterior_mean_samples[..., minutes_index, :]
    posterior_predictions_min = Exponential(rate = jnp.exp(post_min)).sample(key = key)
    posterior_predictions_min = posterior_predictions_min.at[jnp.where(posterior_predictions_retirement == 0)].set(0) ## fix the minutes to sample from zero inflated process
    obs_min = observations[player_index, minutes_index, :]
    posteriors = [jsc.special.expit(post_retirement), posterior_predictions_min]
    obs_normalized = [obs_retirement, obs_min]
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        if (metric_index in [ minutes_index, retirement_index]) :
            continue 
        exposure  = exposures[metric_index, player_index, ...]
        obs = observations[player_index, metric_index, :]
        post = posterior_mean_samples[..., metric_index, :]
        if metric_output == "gaussian":
            scale = posterior_variance_samples[gaussian_index][..., None] / jnp.sqrt(posterior_predictions_min)
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_retirement == 0)].set(-2.0)
            obs_normal = obs
            gaussian_index += 1
        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post + jnp.log(Exponential(rate = jnp.exp(post_min)).sample(key = key))))
            posterior_predictions = 36.0 * (dist.sample(key = key) / posterior_predictions_min) ### per 36 min statistics
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_retirement == 0)].set(0) ### set to 0 wherever 
            obs_normal = 36.0 * (obs / jnp.exp(exposure))
            obs_normal = obs_normal.at[jnp.where(obs_retirement == 0)].set(0)
        elif metric_output == "binomial":
            posterior_predictions = jsc.special.expit(post)
            obs_normal = obs / exposure ### per shot
    
        posteriors.append(posterior_predictions)
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1)}  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack(posteriors, axis = -1)}  ### has shape (chains, draws, time, metrics)

    return obs_data, posterior_predictive












