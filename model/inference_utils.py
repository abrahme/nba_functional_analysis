import jax.numpy as jnp 
from scipy.spatial.distance import cdist
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from numpyro.distributions import Normal, Poisson, BinomialLogits, BetaProportion
# jax.config.update('jax_platform_name', 'cuda')

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

def create_metric_trajectory_all(posterior_mean_samples, observations, exposures, metric_outputs: list[str], metrics: list[str], exposure_names: list[str], posterior_variance_samples = None, posterior_dispersion_samples = None):
    key = jax.random.key(0)
    gaussian_index = 0
    obs_exposure_map = {m: e for m, e in zip(metrics, exposure_names)}
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample games
    post_games = posterior_mean_samples[..., games_index, :, :]
    exposure_games = jnp.astype(exposures[games_index], jnp.int32)
    posterior_predictions_games = BinomialLogits(logits=post_games,total_count=exposure_games[None, None, ...]).sample(key = key) 
    obs_games = observations[..., games_index, :]
    #### then sample minutes 
    post_min = posterior_mean_samples[..., minutes_index, :, :]
    posterior_predictions_min = BetaProportion(jsc.special.expit(post_min), posterior_dispersion_samples ).sample(key = key) * (48 * posterior_predictions_games)
    obs_min = observations[:, minutes_index, :]
    posterior_predictions_min_exposure = jnp.where(~jnp.isnan(obs_min)[None, None, ...], obs_min[None,None,...], posterior_predictions_min)
    posteriors = {"games": posterior_predictions_games, "minutes":posterior_predictions_min}
    obs_normalized = {"games": obs_games / exposure_games, "minutes": obs_min * 48 * exposure_games}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index]) :
            continue 
        exposure  = exposures[metric_index, ...]
        obs = observations[..., metric_index, :]
        post = posterior_mean_samples[..., metric_index, :, :]
        if metric_output == "gaussian":
            scale = posterior_variance_samples[gaussian_index][..., None, None] / jnp.sqrt(posterior_predictions_min_exposure)
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure == 0)].set(-2.0)
            gaussian_index += 1
            obs_normal = obs
        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post) * posterior_predictions_min_exposure)
            posterior_predictions = 36.0 * (dist.sample(key = key) / posterior_predictions_min_exposure) ### per 36 min statistics
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure == 0)].set(0) ### set to 0 wherever 
            obs_normal = 36.0 * (obs / jnp.exp(exposure))
        elif metric_output == "binomial":
            exp_name = obs_exposure_map[metric]
            exp_values = posteriors[exp_name] * (posterior_predictions_min_exposure / 36)
            counts = (jnp.where(~jnp.isnan(exposure)[None, None, ...], exposure[None,None,...], exp_values)) 
            dist = BinomialLogits(logits = post, total_count = jnp.astype(counts, jnp.int32)) 
            posterior_predictions = dist.sample(key = key) / counts
            posterior_predictions = posterior_predictions.at[jnp.where(counts == 0)].set(0)
            obs_normal = obs / exposure

        posteriors[metric] = posterior_predictions
        obs_normalized[metric] = obs_normal
        
    return jnp.stack([p for _, p in obs_normalized.items()], axis = -1), jnp.stack([p for _, p in posteriors.items()], axis = -1)

def create_metric_trajectory(posterior_mean_samples, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str], exposure_names: list[str], posterior_variance_samples = None, posterior_dispersion_samples = None):
    key = jax.random.key(0)
    gaussian_index = 0
    obs_exposure_map = {m: e for m, e in zip(metrics, exposure_names)}
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample games
    post_games = posterior_mean_samples[..., games_index, :]   
    exposure_games = exposures[games_index, player_index, :]
    exposure_games = exposure_games.at[jnp.isnan(exposure_games)].set(82)
    posterior_predictions_games = BinomialLogits(logits=post_games, total_count = jnp.astype(exposure_games, jnp.int64)[None,None,...]).sample(key = key) 
    obs_games = observations[games_index,player_index, :]
    posterior_predictions_games_exposure = jnp.where(~jnp.isnan(obs_games)[None, None, ...], obs_games[None,None,...], jnp.squeeze(posterior_predictions_games))

    #### then sample minutes 
    post_min = posterior_mean_samples[..., minutes_index, :]
    posterior_predictions_min = BetaProportion(jsc.special.expit(post_min), posterior_dispersion_samples[..., None] * jnp.log(posterior_predictions_games_exposure) ).sample(key = key) * (48 * posterior_predictions_games)
    obs_min = observations[minutes_index, player_index, :]
    posterior_predictions_min_exposure = jnp.where(~jnp.isnan(obs_min)[None, None, ...], obs_min[None,None,...], posterior_predictions_min)
    posteriors = {"games": posterior_predictions_games / exposure_games, "minutes":posterior_predictions_min}
    obs_normalized = {"games": obs_games / exposure_games, "minutes": obs_min * 48 * exposure_games}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index]) :
            continue 
        exposure  = exposures[metric_index, player_index, ...]
        obs = observations[metric_index, player_index, :]
        post = posterior_mean_samples[..., metric_index, :]
        if metric_output == "gaussian":
            scale = posterior_variance_samples[gaussian_index][..., None] / jnp.sqrt(posterior_predictions_min_exposure)
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure == 0)].set(-2.0)
            obs_normal = obs
            gaussian_index += 1

        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post) * posterior_predictions_min_exposure)
            posterior_predictions = 36.0 * (dist.sample(key = key) / posterior_predictions_min_exposure) ### per 36 min statistics
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure == 0)].set(0) ### set to 0 wherever 
            obs_normal = 36.0 * (obs / jnp.exp(exposure))

        elif metric_output == "binomial":
            exp_name = obs_exposure_map[metric]
            exp_values = posteriors[exp_name] * (posterior_predictions_min_exposure / 36)
            counts = (jnp.where(~jnp.isnan(exposure)[None, None, ...], exposure[None,None,...], exp_values)) 
            dist = BinomialLogits(logits = post, total_count = jnp.astype(counts, jnp.int64)) 
            posterior_predictions = dist.sample(key = key) / counts
            posterior_predictions = posterior_predictions.at[jnp.where(counts == 0)].set(0)
            obs_normal = obs / exposure ### per shot

        posteriors[metric] = posterior_predictions
        obs_normalized[metric] = obs_normal


    obs_data = {"y": jnp.stack([obs for _, obs in obs_normalized.items()], axis = -1)}  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack([p for _, p in posteriors.items()], axis = -1)}  ### has shape (chains, draws, time, metrics)

    return obs_data, posterior_predictive


def create_metric_trajectory_map(posterior_mean_map: jnp.ndarray, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str]):
    ### this is assuming that posterior_mean_map is shape (k,t)
    gaussian_index = 0
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample retirement
    post_games = posterior_mean_map[games_index]
    obs_games = observations[games_index, player_index, :]
    exposure_games = exposures[games_index,player_index,  :]
    exposure_min = exposures[minutes_index,player_index, :]

    #### then sample minutes 
    
    post_min = posterior_mean_map[minutes_index]
    posterior_predictions_min = jsc.special.expit(post_min) * 48
    obs_min = observations[ minutes_index, player_index, :]

    posteriors = [jsc.special.expit(post_games), posterior_predictions_min]
    obs_normalized = [obs_games / exposure_games, obs_min * 48]
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        if (metric_index in [ minutes_index, games_index]) :
            continue 
        exposure  = exposures[metric_index, player_index, ...]
        obs = observations[metric_index, player_index,:]
        post = posterior_mean_map[metric_index]
        if metric_output == "gaussian":
            posterior_predictions = post
            obs_normal = obs
            gaussian_index += 1
        elif metric_output == "poisson":
            posterior_predictions = 36.0 * jnp.exp(post) / 1
            obs_normal = 36.0 * (obs / (1 * jnp.exp(exposure)))
        elif metric_output == "binomial":
            posterior_predictions = jsc.special.expit(post)
            obs_normal = obs / exposure ### per shot
    
        posteriors.append(posterior_predictions)
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1)}  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack(posteriors, axis = -1)}  ### has shape (time, metrics)

    return obs_data, posterior_predictive

def create_metric_trajectory_prior(prior_mean_samples,  metric_outputs: list[str], metrics: list[str], exposure_names: list[str], prior_variance_samples = None, prior_dispersion_samples = None):
    key = jax.random.key(1)
    gaussian_index = 0
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample games
    post_games = prior_mean_samples[..., games_index, :]   
    prior_predictions_games = BinomialLogits(logits=post_games, total_count = 82).sample(key = key) 
    #### then sample minutes 
    post_min = prior_mean_samples[..., minutes_index, :]
    prior_predictions_min = BetaProportion(jsc.special.expit(post_min), prior_dispersion_samples[..., None] * jnp.log(3000) ).sample(key = key) * (48 * prior_predictions_games)
    priors = {"games": prior_predictions_games / 82, "minutes":prior_predictions_min}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index]) :
            continue 
        post = prior_mean_samples[..., metric_index, :]
        if metric_output == "gaussian":
            scale = prior_variance_samples[gaussian_index][..., None] / jnp.sqrt(3000)
            dist = Normal()
            prior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            gaussian_index += 1

        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post))
            prior_predictions = 36.0 * (dist.sample(key = key)) ### per 36 min statistics

        elif metric_output == "binomial":
            dist = BinomialLogits(logits = post, total_count = 1000) 
            prior_predictions = dist.sample(key = key) / 1000
        priors[metric] = prior_predictions

    prior_predictive = {"y": jnp.stack([p for _, p in priors.items()], axis = -1)}  ### has shape (chains, draws, time, metrics)

    return prior_predictive

def create_metric_trajectory_observations(player_index, observations, exposures, metric_outputs: list[str], metrics: list[str]):
    gaussian_index = 0
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ###  > 1 -> playing, 0 --> retired

    ### first get retirement
    obs_games = observations[player_index, games_index, :]
    exposure_games = exposures[games_index, player_index, :]
    #### then sample minutes 
    
    
    obs_min = observations[player_index, minutes_index, :]
    obs_normalized = [obs_games / exposure_games , obs_min]
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        if (metric_index in [ minutes_index, games_index]) :
            continue 
        exposure  = exposures[metric_index, player_index, ...]
        obs = observations[player_index, metric_index, :]
        if metric_output == "gaussian":
            obs_normal = obs
            gaussian_index += 1
        elif metric_output == "poisson":
            obs_normal = 36.0 * (obs / jnp.exp(exposure))
            obs_normal = obs_normal.at[jnp.where(obs_games == 0)].set(0)
        elif metric_output == "binomial":
            obs_normal = obs / exposure ### per shot
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1)}  ### has shape (time, metrics)

    return obs_data   







