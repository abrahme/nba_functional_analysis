import arviz as az 

def plot_metric_trajectory(metric_index, player_index, posterior_mean_samples, observations):
    obs = observations[player_index, metric_index, :]
    post = posterior_mean_samples[..., metric_index, player_index, :]
    obs_data = {"y": obs}
    posterior_predictive = {"y": post}
    idata = az.from_dict(observed_data=obs_data, posterior_predictive=posterior_predictive)
    az.plot_ts(idata, y = "y")