import arviz as az 
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal, Poisson, Binomial


def plot_metric_trajectory(player_index, posterior_mean_samples, observations, exposures, metric_outputs: list[str], posterior_variance_samples = None):
    key = jax.random.key(0)
    gaussian_index = 0
    posteriors = []
    obs_normalized = []
    
    for metric_index, metric_output in enumerate(metric_outputs):
        exposure = exposures[metric_index, ...]
        obs = observations[player_index, metric_index, :]
        post = posterior_mean_samples[..., metric_index, player_index, :]
        inds = jnp.where(jnp.isnan(exposure))
        exposure = exposure.at[inds].set(np.take(np.nanmean(exposure, axis = 0), inds[1]))

        if metric_output == "gaussian":
            scale = jnp.einsum("cs,t -> cst", posterior_variance_samples[gaussian_index],  1.0 / exposure[player_index, :])
            dist = Normal()
            posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
            obs_normal = obs
            gaussian_index += 1
        elif metric_output == "poisson":
            dist = Poisson(rate = jnp.exp(post + exposures[player_index, :]))
            posterior_predictions = 36.0 * (dist.sample(key = key) / jnp.exp(exposure[player_index, :])) ### per 36 min statistics
            obs_normal = 36.0 * (obs / jnp.exp(exposure[player_index, :]))
        elif metric_output == "binomial":
            dist = Binomial(total_count = jnp.round(exposure[player_index, :], decimals=0), logits = post)
            posterior_predictions = dist.sample(key = key) / jnp.round(exposure[player_index, :], decimals=0) ### per shot
            obs_normal = obs / jnp.round(exposure[player_index, :], decimals=0) ### per shot
    
        posteriors.append(posterior_predictions)
        obs_normalized.append(obs_normal)


    obs_data = {"y": jnp.stack(obs_normalized, axis = -1)}  ### has shape (time, metrics)
    print(obs_data["y"].shape)
    posterior_predictive = {"y": jnp.stack(posteriors, axis = -1)}  ### has shape (chains, draws, time, metrics)
    print(posterior_predictive["y"].shape)
    idata = az.from_dict(observed_data=obs_data, posterior_predictive=posterior_predictive,
                         dims={"y": ["age", "metric"]},
                         coords={"age": range(21), "metric": range(len(metric_outputs))})
    az.plot_ts(idata, y = "y", plot_dim="age")