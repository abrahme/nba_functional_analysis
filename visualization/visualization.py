import arviz as az 
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal, Poisson, Binomial


def plot_metric_trajectory(metric_index, player_index, posterior_mean_samples, observations, exposures, metric_output: str, posterior_variance_samples = None):
    key = jax.random.key(0)
    obs = observations[player_index, metric_index, :]
    post = posterior_mean_samples[..., metric_index, player_index, :]
    inds = jnp.where(jnp.isnan(exposures))
    exposures = exposures.at[inds].set(np.take(np.nanmean(exposures, axis = 0), inds[1]))
    obs_data = {"y": obs}
    if metric_output == "gaussian":
        scale = jnp.einsum("cs,t -> cst", posterior_variance_samples,  1.0 / exposures[player_index, :])
        dist = Normal()
        posterior_predictions = (dist.sample(key = key, sample_shape=post.shape) * scale + post)
    elif metric_output == "poisson":
        dist = Poisson(rate = jnp.exp(post + exposures[player_index, :]))
        posterior_predictions = dist.sample(key = key)
    elif metric_output == "binomial":
        dist = Binomial(total_count = exposures[player_index, :], logits = post)
        posterior_predictions = dist.sample(key = key)
    posterior_predictive = {"y": posterior_predictions}
    idata = az.from_dict(observed_data=obs_data, posterior_predictive=posterior_predictive)
    az.plot_ts(idata, y = "y")