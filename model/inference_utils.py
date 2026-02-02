import jax.numpy as jnp 
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from numpyro import handlers
from numpyro.handlers import trace, condition, seed
from numpyro.distributions import Normal, Poisson, BinomialLogits, BetaProportion, BetaBinomial, NegativeBinomial2
# jax.config.update('jax_platform_name', 'cuda')


import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def posterior_X_to_df(posterior_samples, ids, names, minutes, position_group, labels):
    """
    posterior_samples: array of shape (num_chain, num_samples, N, D)
    ids: array of length N giving arbitrary id numbers for each index in N
    labels: subset of ids -> marked as "sampled"; others as "fixed"
    """
    num_chain, num_samples, N, D = posterior_samples.shape
    # Create index grids
    chain_idx, sample_idx, n_idx = np.meshgrid(
        np.arange(num_chain),
        np.arange(num_samples),
        np.arange(N),
        indexing="ij"
    )

    # Flatten values
    flat_values = posterior_samples.reshape(-1, D)
    flat_chain = chain_idx.ravel()
    flat_sample = sample_idx.ravel()
    flat_n = n_idx.ravel()

    # Map index -> arbitrary id
    flat_id = np.array(ids)[flat_n]
    flat_names = np.array(names)[flat_n]
    flat_minutes = np.array(minutes)[flat_n]
    flat_position_group = np.array(position_group)[flat_n]
    # Label based on id membership
    flat_label = np.where(np.isin(flat_n, labels), "sampled", "fixed")

    # Build DataFrame
    df = pd.DataFrame(flat_values, columns=[f"Dim {str(i+1)}" for i in range(D)])
    df["chain"] = flat_chain
    df["sample"] = flat_sample
    df["id"] = flat_id
    df["name"] = flat_names
    df["minutes"] = flat_minutes
    df["position_group"] = flat_position_group
    df["label"] = flat_label

    return df



def posterior_to_df(posterior_samples, ids, metrics, ages):
    N_chains, N_samples, D, T, K = posterior_samples.shape
    # Create index grids
    chain_idx, sample_idx, d_idx, t_idx, k_idx = np.meshgrid(
        np.arange(N_chains),
        np.arange(N_samples),
        np.arange(D),
        np.arange(T),
        np.arange(K),
        indexing='ij'
    )

    # Flatten and convert indices to labels
    df = pd.DataFrame({
        'chain': chain_idx.ravel(),
        'sample': sample_idx.ravel(),
        'player': np.array(ids)[d_idx.ravel()],
        'metric': np.array(metrics)[k_idx.ravel()],
        'age': np.array(ages)[t_idx.ravel()],
        'value': posterior_samples.ravel()
    })
    return df

def loadings_to_df(loading_samples, ids, metrics, weights):
    K, D, T = loading_samples.shape
        # Create index grids
    d_idx, t_idx, k_idx = np.meshgrid(
        np.arange(D),
        np.arange(T),
        np.arange(K),
        indexing='ij'
    )

    # Flatten and convert indices to labels
    df = pd.DataFrame({
        'player': np.array(ids)[d_idx.ravel()],
        'metric': np.array(metrics)[k_idx.ravel()],
        'weights': np.array(weights)[t_idx.ravel()],
        'value': loading_samples.ravel()
    })
    return df

def time_factors_to_df(factor_samples, ages):
    D, T = factor_samples.shape
    d_idx, t_idx = np.meshgrid(
        np.arange(D),
        np.arange(T),
        indexing='ij'
    )

    # Flatten and convert indices to labels
    df = pd.DataFrame({
        'factor': np.arange(D)[d_idx.ravel()],
        'age': np.array(ages)[t_idx.ravel()],
        'value': factor_samples.ravel()
    })
    return df


def posterior_peaks_to_df(posterior_peak_samples, ids, metrics):
    N_chains, N_samples, D, K = posterior_peak_samples.shape
    chain_idx, sample_idx, d_idx, k_idx = np.meshgrid(
    np.arange(N_chains),
    np.arange(N_samples),
    np.arange(D),
    np.arange(K),
    indexing='ij'
    )

    # Flatten and convert indices to labels
    df = pd.DataFrame({
    'chain': chain_idx.ravel(),
    'sample': sample_idx.ravel(),
    'player': np.array(ids)[d_idx.ravel()],
    'metric': np.array(metrics)[k_idx.ravel()],
    'value': posterior_peak_samples.ravel()
    })
    return df

def single_log_terms(s, model, model_args):
    conditioned = condition(model, s)
    tr = trace(seed(conditioned, rng_seed=0)).get_trace(**model_args)

    log_prior, log_likelihood = 0.0, 0.0
    for site in tr.values():
        if site["type"] == "sample":
            logp = site["fn"].log_prob(site["value"]).sum()
            if site.get("is_observed", False):
                log_likelihood += logp
            else:
                log_prior += logp
    return log_prior, log_likelihood

def get_latent_sites(model, model_args):
    seeded_model = handlers.seed(model, rng_seed=0)
    trace = handlers.trace(seeded_model).get_trace(**model_args)
    latent_sites = [
        name for name, site in trace.items()
        if site['type'] == 'sample' and not site['is_observed']
    ]
    return latent_sites


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

def create_metric_trajectory_all(posterior_mean_samples, observations, exposures, metric_outputs: list[str], metrics: list[str], exposure_names: list[str], posterior_variance_samples = None, posterior_dispersion_samples = None, posterior_kappa_samples=None,
                                 posterior_neg_bin_samples = None, posterior_tau_samples = None):
    posterior_kappa_samples = 1 if posterior_kappa_samples is None else posterior_kappa_samples
    posterior_tau_samples = 0 if posterior_tau_samples is None else posterior_tau_samples
    posterior_dispersion_samples = 1 if posterior_dispersion_samples is None else posterior_dispersion_samples
    posterior_variance_samples = 1 if posterior_variance_samples is None else posterior_variance_samples
    posterior_neg_bin_samples = 1 if posterior_neg_bin_samples is None else posterior_neg_bin_samples
    key = jax.random.key(0)
    gaussian_index = 0
    neg_bin_index = 0
    beta_index = 0
    obs_exposure_map = {m: e for m, e in zip(metrics, exposure_names)}
    retirement_index = metrics.index("retirement") ### 1 -> playing, 0 --> retired
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  
    ### first sample retirement
    post_retirement = jsc.special.expit(posterior_mean_samples[..., retirement_index, :, :])
    exposure_retirement = exposures[retirement_index]
    exposure_retirement = exposure_retirement.at[jnp.isnan(exposure_retirement)].set(1)
    exposure_retirement = jnp.astype(exposure_retirement, jnp.int64)
    posterior_predictions_retirement = BetaBinomial(concentration0 = 1 - post_retirement, concentration1= post_retirement, total_count = exposure_retirement).sample(key = key)
    obs_retirement = observations[retirement_index]

    ### then sample games
    post_games = posterior_mean_samples[..., games_index, :, :]
    exposure_games = exposures[games_index]
    exposure_games = exposure_games.at[jnp.isnan(exposure_games)].set(82)
    exposure_games = jnp.astype(exposure_games, jnp.int64)

    posterior_predictions_games = BetaBinomial(concentration0= (1-jsc.special.expit(post_games)) * posterior_kappa_samples, 
                                               concentration1=jsc.special.expit(post_games) * posterior_kappa_samples,
                                               total_count=exposure_games[None, None, ...]).sample(key = key)
    obs_games = observations[games_index]
    # posterior_predictions_games_exposure = jnp.where(~jnp.isnan(obs_games)[None, None, ...], obs_games[None,None,...], jnp.squeeze(posterior_predictions_games))
    posterior_predictions_games_exposure = posterior_predictions_games * posterior_predictions_retirement
    #### then sample minutes 
    post_min = posterior_mean_samples[..., minutes_index, :, :]
    posterior_predictions_min = BetaProportion(jsc.special.expit(post_min), posterior_dispersion_samples[beta_index][..., None, None] * (posterior_predictions_games_exposure + 1)).sample(key = key) * (48 * posterior_predictions_games)
    beta_index += 1
    # posterior_predictions_min = posterior_predictions_min.at[posterior_predictions_games == 0].set(0)
    obs_min = observations[minutes_index]
    posterior_predictions_min_exposure = jnp.where(~jnp.isnan(obs_min)[None, None, ...], obs_min[None,None,...] * 48 * exposure_games, posterior_predictions_min)
    posteriors = {"games": posterior_predictions_games / exposure_games, "retirement": jsc.special.expit(post_retirement), "minutes":jnp.where(posterior_predictions_games == 0, 0, posterior_predictions_min / (posterior_predictions_games * 48))}
    obs_normalized = {"games": obs_games / exposure_games, "minutes": obs_min, "retirement": obs_retirement}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    print(posterior_variance_samples.shape)
    
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index, retirement_index]) :
            continue 
        exposure  = exposures[metric_index]
        obs = observations[metric_index]
        post = posterior_mean_samples[..., metric_index, :, :]
        if metric_output == "gaussian":
            scale = jnp.sqrt(jnp.square(posterior_variance_samples[gaussian_index]) / (posterior_predictions_min_exposure + 1) + jnp.square(posterior_tau_samples[gaussian_index][..., None, None]))
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            # posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure < 1)].set(-2.0)
            gaussian_index += 1
            obs_normal = obs
        elif metric_output in ["poisson", "negative-binomial"]:
            rate = jnp.exp(post) * posterior_predictions_min_exposure
            if metric_output == "poisson":
                dist = Poisson(rate = rate)
            elif metric_output == "negative-binomial":
                dist = NegativeBinomial2(mean = rate, concentration=posterior_neg_bin_samples[neg_bin_index][..., None, None])
                neg_bin_index += 1
            posterior_predictions = 36 * (dist.sample(key = key) / posterior_predictions_min_exposure)  ### per 36 min statistics
            # posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure == 0)].set(0) ### set to 0 wherever 
            obs_normal = 36.0 * (obs / jnp.exp(exposure))
        elif metric_output == "binomial":
            exp_name = obs_exposure_map[metric]
            if exp_name in posteriors:
                exp_values = posteriors[exp_name] *  ( posterior_predictions_min_exposure / 36) 
                counts = (jnp.where(~jnp.isnan(exposure)[None, None, ...], exposure[None,None,...], exp_values)) 
            else:
                counts = jnp.where(~jnp.isnan(exposure)[None, None, ...], exposure[None, None, ...], 0)
            dist = BinomialLogits(logits = post, total_count = jnp.astype(counts, jnp.int64)) 
            posterior_predictions = dist.sample(key = key) / counts
            # posterior_predictions = posterior_predictions.at[jnp.where(counts == 0)].set(0)
            obs_normal = obs / exposure
        elif metric_output == "beta":
            dist = BetaProportion(jsc.special.expit(post), posterior_dispersion_samples[beta_index][..., None, None] * (posterior_predictions_min_exposure + 1))
            posterior_predictions = dist.sample(key = key)
            obs_normal = obs
            beta_index += 1
        posteriors[metric] = posterior_predictions
        obs_normalized[metric] = obs_normal
        
    return jnp.stack([p for _, p in obs_normalized.items()], axis = -1), jnp.stack([p for _, p in posteriors.items()], axis = -1)

def create_metric_trajectory(posterior_mean_samples, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str], exposure_names: list[str], posterior_variance_samples = None, posterior_dispersion_samples = None, posterior_kappa_samples=None):
    key = jax.random.key(0)
    gaussian_index = 0
    obs_exposure_map = {m: e for m, e in zip(metrics, exposure_names)}
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample games
    post_games = posterior_mean_samples[..., games_index, :]   
    exposure_games = exposures[games_index, player_index, :]
    exposure_games = exposure_games.at[jnp.isnan(exposure_games)].set(82)
    exposure_games = jnp.astype(exposure_games, jnp.int64)
    posterior_predictions_games = BetaBinomial(concentration0= (1-jsc.special.expit(post_games)) * posterior_kappa_samples, 
                                               concentration1=jsc.special.expit(post_games) * posterior_kappa_samples,
                                               total_count=exposure_games[None, None, ...]).sample(key = key) 
    obs_games = observations[games_index,player_index, :]
    posterior_predictions_games_exposure = posterior_predictions_games
    
    #### then sample minutes 
    post_min = posterior_mean_samples[..., minutes_index, :]
    
    posterior_predictions_min = BetaProportion(jsc.special.expit(post_min), posterior_dispersion_samples[..., None] * jnp.sqrt(posterior_predictions_games_exposure + 1) ).sample(key = key) * (48 * posterior_predictions_games)
    posterior_predictions_min = posterior_predictions_min.at[posterior_predictions_games_exposure == 0].set(0)
    obs_min = observations[minutes_index, player_index, :]
    posterior_predictions_min_exposure = jnp.where(~jnp.isnan(obs_min)[None, None], obs_min[None,None] * 48 * exposure_games, posterior_predictions_min)
    posteriors = {"games": posterior_predictions_games / exposure_games, "minutes":posterior_predictions_min}
    obs_normalized = {"games": obs_games / exposure_games, "minutes": obs_min * 48 * exposure_games}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index]) :
            continue 
        exposure  = exposures[metric_index, player_index]
        obs = observations[metric_index, player_index]
        post = posterior_mean_samples[..., metric_index, :]
        if metric_output == "gaussian":
            scale = posterior_variance_samples[gaussian_index][..., None] / jnp.sqrt(posterior_predictions_min_exposure)
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            posterior_predictions = posterior_predictions.at[jnp.where(posterior_predictions_min_exposure < 1.0)].set(-2.0)
            obs_normal = obs
            gaussian_index += 1

        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post) * posterior_predictions_min_exposure)
            posterior_predictions = (dist.sample(key = key) / posterior_predictions_min_exposure) * 36 ### per 36 min statistics
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



def create_metric_trajectory_mu(posterior_mean_samples, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str]):

    retirement_index = metrics.index("retirement") ### 1 -> playing, 0 --> retired
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  

    exposure_retirement = exposures[retirement_index, player_index, :]
    exposure_retirement = exposure_retirement.at[jnp.isnan(exposure_retirement)].set(1)
    exposure_games = exposures[games_index, player_index, :]
    exposure_games = exposure_games.at[jnp.isnan(exposure_games)].set(82)
    obs_games = observations[games_index,player_index, :]
    obs_retirement = observations[retirement_index, player_index, :]
    ### first sample games
    post_games = posterior_mean_samples[..., games_index, :]   
    posterior_predictions_games = jsc.special.expit(post_games)

    post_retirement = posterior_mean_samples[..., retirement_index, :]   
    posterior_predictions_retirement = jsc.special.expit(post_retirement)

    #### then sample minutes 
    post_min = posterior_mean_samples[..., minutes_index, :]
    posterior_predictions_min = jsc.special.expit(post_min) 
    obs_min = observations[minutes_index, player_index, :]

    posteriors = {"games": posterior_predictions_games , "minutes":posterior_predictions_min, "retirement": posterior_predictions_retirement}
    obs_normalized = {"games": obs_games / exposure_games, "minutes": obs_min , "retirement": obs_retirement}
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        metric = metrics[metric_index]
        if (metric_index in [minutes_index, games_index, retirement_index]) :
            continue 
        exposure  = exposures[metric_index, player_index]
        obs = observations[metric_index, player_index]
        post = posterior_mean_samples[..., metric_index, :]
        if metric_output == "gaussian":
            posterior_predictions = post
            obs_normal = obs

        elif metric_output in ["poisson", "negative-binomial"]:
            posterior_predictions = 36 * jnp.exp(post) 
            obs_normal = 36.0 * (obs / jnp.exp(exposure))

        elif metric_output in ["beta-binomial", "beta", "binomial"]:
            posterior_predictions = jsc.special.expit(post)
            obs_normal = obs / exposure ### per shot

        posteriors[metric] = posterior_predictions
        obs_normalized[metric] = obs_normal


    obs_data = {"y": jnp.stack([obs for _, obs in obs_normalized.items()], axis = -1)}  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack([p for _, p in posteriors.items()], axis = -1)}  ### has shape (chains, draws, time, metrics)

    return obs_data, posterior_predictive


def create_hazard_trajectory_map(posterior_mean_map: jnp.ndarray, player_index, observations,  metrics: list[str]):
    if type(player_index) == int:
        pass
    elif len(player_index) == 0:
        player_index = jnp.arange(0, posterior_mean_map.shape[1])
    if posterior_mean_map.ndim == 1:
        posterior_mean_map = posterior_mean_map[None]

    obs_data = {"y": observations[player_index] * jnp.ones_like(posterior_mean_map) }  ### has shape (time, metrics)
    posterior_predictive = {"y": posterior_mean_map }  ### has shape (time, metrics)

    return obs_data, posterior_predictive

def create_metric_trajectory_map(posterior_mean_map: jnp.ndarray, player_index, observations, exposures, metric_outputs: list[str], metrics: list[str]):
    if type(player_index) == int:
        pass
    elif len(player_index) == 0:
        player_index = jnp.arange(0, posterior_mean_map.shape[1])
    if posterior_mean_map.ndim == 1:
        posterior_mean_map = posterior_mean_map[None]
    ### this is assuming that posterior_mean_map is shape (k,t)
    gaussian_index = 0
    # minutes_index = metrics.index("minutes")
    # games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    # ### first sample retirement
    # post_games = posterior_mean_map[games_index]
    # obs_games = observations[games_index, player_index, :]
    # exposure_games = exposures[games_index,player_index,  :]
    # exposure_min = exposures[minutes_index,player_index, :]

    #### then sample minutes 
    posteriors = []
    obs_normalized = []
    # posteriors = [jsc.special.expit(post_games), posterior_predictions_min]
    # obs_normalized = [obs_games / exposure_games, obs_min * 48]
    ### sample all the poisson metrics using posterior predictions log min as exposure, and sample obpm / dbpm using sqrt(minutes) as exposure
    for metric_index, metric_output in enumerate(metric_outputs):
        exposure  = exposures[metric_index, player_index, ...]
        obs = observations[metric_index, player_index,:]
        post = posterior_mean_map[metric_index]
        if metric_output == "gaussian":
            posterior_predictions = post
            obs_normal = obs
            gaussian_index += 1
        elif metric_output in ["poisson", "negative-binomial"]:
            if metrics[metric_index] != "minutes":
                posterior_predictions = 36.0 * jnp.exp(post) / 1
                obs_normal = 36.0 * (obs / (1 * jnp.exp(exposure)))
            else:
                posterior_predictions = jnp.exp(post)
                obs_normal = obs / jnp.exp(exposure)
        elif metric_output in ["binomial", "beta", "beta-binomial"]:
            posterior_predictions = jsc.special.expit(post)
            if metric_output != "beta":
                obs_normal = obs / exposure ### per shot
            else:
                obs_normal = obs
    
        posteriors.append(posterior_predictions)
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1) }  ### has shape (time, metrics)
    posterior_predictive = {"y": jnp.stack(posteriors, axis = -1) }  ### has shape (time, metrics)

    return obs_data, posterior_predictive

def create_metric_trajectory_prior(prior_mean_samples,  metric_outputs: list[str], metrics: list[str], exposure_names: list[str], prior_variance_samples = None, prior_dispersion_samples = None, posterior_kappa_samples = None):
    key = jax.random.key(1)
    gaussian_index = 0
    minutes_index = metrics.index("pct_minutes")
    games_index = metrics.index("games")  ### 1 -> playing, 0 --> retired
    ### first sample games
    post_games = prior_mean_samples[..., games_index, :]   
    prior_predictions_games = BetaBinomial(concentration0= (1-jsc.special.expit(post_games)) * posterior_kappa_samples, 
                                               concentration1=jsc.special.expit(post_games) * posterior_kappa_samples,
                                               total_count=82).sample(key = key) 
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







