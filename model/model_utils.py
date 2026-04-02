
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsci
import scipy.special as sps
import numpyro.distributions as dist
from numpyro.infer.hmc import HMC
from numpyro.distributions import constraints
import pandas as pd
from jax import random
import jax
from .hsgp import  diag_spectral_density, make_psi_gamma, make_convex_phi, make_convex_phi_prime, vmap_make_convex_phi, vmap_make_convex_phi_prime, vmap_make_convex_phi_double_prime, vmap_make_convex_phi_triple_prime, eigenfunctions, eigenfunctions_multivariate
from .inference_utils import create_metric_trajectory_map


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
    rhs = jnp.einsum("...td, ...t -> ...d", Q, ar_raw)
    alpha = jnp.linalg.solve(QtQ, rhs[..., None]).squeeze(-1)
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


def summarize_normalized_weighted_metric_residuals_by_age(
    posterior_mean_map,
    observations,
    exposures,
    metric_outputs,
    metrics,
    ages=None,
    de_trend_values=None,
    evaluation_mask=None,
):
    """
    Returns a DataFrame with columns: metric, normalized_weighted_residual, residual, normalized_weight, age
    for each valid (non-NaN) entry. The weight is normalized by the sum of all valid weights for that metric.
    """
    eps = 1e-8
    posterior_mean_for_eval = posterior_mean_map
    if de_trend_values is not None:
        posterior_mean_for_eval = posterior_mean_for_eval + de_trend_values

    obs_map, pred_map = create_metric_trajectory_map(
        posterior_mean_for_eval,
        [],
        observations,
        exposures,
        metric_outputs,
        metrics,
    )

    obs_vals = np.asarray(obs_map["y"])  # shape (T, K)
    pred_vals = np.asarray(pred_map["y"])  # shape (T, K)
    exposure_vals = np.asarray(exposures)
    if ages is None:
        ages = np.arange(obs_vals.shape[1])

    rows = []
    eval_mask_array = None
    if evaluation_mask is not None:
        eval_mask_array = np.asarray(evaluation_mask)

    for metric_index, metric_name in enumerate(metrics):
        family = metric_outputs[metric_index]
        obs_metric = obs_vals[..., metric_index]
        pred_metric = pred_vals[..., metric_index]
        diff = pred_metric - obs_metric
        raw_exposure = exposure_vals[metric_index]

        if family in ["poisson", "negative-binomial"]:
            exposure_weight = np.exp(raw_exposure)
        elif family in ["beta", "gaussian"]:
            exposure_weight = np.square(raw_exposure) - 1.0
        else:
            exposure_weight = raw_exposure
        exposure_weight = np.where(np.isfinite(exposure_weight) & (exposure_weight > 0), exposure_weight, 1.0)

        if family in ["binomial", "beta-binomial", "beta"]:
            n_eff = np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), raw_exposure, 0)
            weight = n_eff
        elif family in ["poisson", "negative-binomial"]:
            count_exposure = np.exp(raw_exposure)
            weight = np.where(np.isfinite(count_exposure), count_exposure, 0.0)
        elif family == "gaussian":
            gaussian_exposure = np.square(raw_exposure) - 1.0
            weight = np.where(
                np.isfinite(gaussian_exposure) & (gaussian_exposure > 0),
                gaussian_exposure,
                0,
            )
        else:
            weight = np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), raw_exposure, 1.0)

        valid = np.isfinite(diff) & np.isfinite(weight) & (weight > 0)
        if eval_mask_array is not None:
            if eval_mask_array.ndim == 2:
                valid = valid & eval_mask_array
            elif eval_mask_array.ndim == 3:
                valid = valid & eval_mask_array[..., metric_index]

        weight_valid = weight[valid]
        weight_sum = np.sum(weight_valid)
        metric_df = pd.DataFrame()
        metric_df["residual"] = diff.flatten()
        metric_df["normalized_weighted_residual"] = (diff * weight / weight_sum).flatten()
        metric_df["metric"] = metric_name
        metric_df["normalized_weight"] = weight.flatten() / weight_sum
        metric_df["age"] = np.tile(ages, obs_vals.shape[0])
        rows.append(metric_df)

    return pd.concat(rows).dropna().reset_index(drop=True)




    

        



   
def summarize_metric_error_observed_substitutions(
    posterior_mean_map,
    observations,
    exposures,
    metric_outputs,
    metrics,
    sigma_beta,
    sigma,
    sigma_beta_binomial,
    de_trend_values=None,
    evaluation_mask=None,
    sigma_negative_binomial=1,
):
    eps = 1e-8
    posterior_mean_for_eval = posterior_mean_map
    if de_trend_values is not None:
        posterior_mean_for_eval = posterior_mean_for_eval + de_trend_values



    sigma_index = 0
    sigma_beta_index = 0
    sigma_beta_binomial_index = 0
    sigma_neg_bin_index = 0

    obs_map, pred_map = create_metric_trajectory_map(
        posterior_mean_for_eval,
        [],
        observations,
        exposures,
        metric_outputs,
        metrics,
    )


    obs_vals = np.asarray(obs_map["y"])
    pred_vals = np.asarray(pred_map["y"])
    observations_vals = np.asarray(observations)
    exposure_vals = np.asarray(exposures)

    rows = []
    eval_mask_array = None
    if evaluation_mask is not None:
        eval_mask_array = np.asarray(evaluation_mask)

    for metric_index, metric_name in enumerate(metrics):

        family = metric_outputs[metric_index]
        obs_metric = obs_vals[..., metric_index]
        pred_metric = pred_vals[..., metric_index]
        obs_raw_metric = observations_vals[metric_index]
        diff = pred_metric - obs_metric
        square_diff = np.square(diff)
        
        raw_exposure = exposure_vals[metric_index]

        if family in ["poisson", "negative-binomial"]:
            exposure_weight = np.exp(raw_exposure) / (obs_metric + eps)
        elif family in ["beta", "gaussian"]:
            exposure_weight = np.square(raw_exposure) - 1.0
        elif family in ["beta-binomial", "binomial"]:
            p_hat = np.clip(obs_metric, eps, 1.0 - eps)
            exposure_weight = raw_exposure / (p_hat * (1.0 - p_hat))
        exposure_weight = np.where(np.isfinite(exposure_weight) & (exposure_weight > 0), exposure_weight, 0.0)
        # exposure_weight = jnp.ones_like(exposure_weight)

        # if family in ["binomial", "beta-binomial", "beta"]:
        #     # n_eff = np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), raw_exposure, 1.0) if family != "beta" else np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), np.square(raw_exposure) - 1, 1.0)
        #     # p_hat = np.clip(pred_metric, eps, 1.0 - eps)
        #     # variance = (p_hat * (1.0 - p_hat)) / n_eff
        #     # weight = 1.0 / np.maximum(variance, eps)
        #     pass
            
        # elif family in ["poisson", "negative-binomial"]:
        #     # count_exposure = np.exp(raw_exposure)
        #     # count_exposure = np.where(np.isfinite(count_exposure) & (count_exposure > 0), count_exposure, 1.0)
        #     # pred_rate = np.maximum(pred_metric, 0.0)
        #     # variance = np.maximum(36.0 * pred_rate / count_exposure, eps)
        #     # weight = 1.0 / variance
        #     pass 
        # elif family == "gaussian":
        #     # gaussian_exposure = np.square(raw_exposure) - 1.0
        #     # weight = np.where(
        #     #     np.isfinite(gaussian_exposure) & (gaussian_exposure > 0),
        #     #     gaussian_exposure,
        #     #     1.0,
        #     # )
        #     pass
        # else:
        #     # weight = np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), raw_exposure, 1.0)
        #     pass 
       
        valid = np.isfinite(exposure_weight) & evaluation_mask & np.isfinite(diff)

        if eval_mask_array is not None:
            if eval_mask_array.ndim == 2:
                valid = valid & eval_mask_array
            elif eval_mask_array.ndim == 3:
                valid = valid & eval_mask_array[..., metric_index]
        if np.any(valid):
            bias = np.sum(diff[valid] * exposure_weight[valid]) / np.sum(exposure_weight[valid]) 
            rmse = np.sqrt(np.sum(square_diff[valid] * exposure_weight[valid]) / np.sum(exposure_weight[valid]))
            def compute_pointwise_log_loss(predicted_metric, sigma_beta_binomial_index, sigma_index, sigma_beta_index, sigma_neg_bin_index):
                pred_prob = np.clip(predicted_metric, eps, 1.0 - eps)
                log_loss = np.full_like(predicted_metric, np.nan, dtype=float)

                if family in ["binomial", "beta-binomial"]:
                    n_trials = np.where(np.isfinite(raw_exposure) & (raw_exposure > 0), raw_exposure, np.nan)
                    y_count = obs_raw_metric
                    valid_count = (
                        valid
                        & np.isfinite(n_trials)
                        & np.isfinite(y_count)
                        & (n_trials >= 0)
                        & (y_count >= 0)
                        & (y_count <= n_trials)
                    )

                    log_comb = sps.gammaln(n_trials + 1.0) - sps.gammaln(y_count + 1.0) - sps.gammaln(n_trials  - y_count + 1.0)
                    ll = log_comb + y_count * np.log(pred_prob) + (n_trials - y_count) * np.log1p(-pred_prob)


                    if family == "binomial":
                        ll = (
                            log_comb
                            + y_count * np.log(pred_prob)
                            + (n_trials - y_count) * np.log1p(-pred_prob)
                        )

                    else:  # beta-binomial
                        mu = pred_prob
                        phi = sigma_beta_binomial[sigma_beta_binomial_index]   # concentration parameter

                        alpha = mu * phi
                        beta = (1 - mu) * phi

                        log_beta_ratio = (
                            sps.gammaln(y_count + alpha)
                            + sps.gammaln(n_trials - y_count + beta)
                            - sps.gammaln(n_trials + alpha + beta)
                            - sps.gammaln(alpha)
                            - sps.gammaln(beta)
                            + sps.gammaln(alpha + beta)
                        )

                        ll = log_comb + log_beta_ratio
                        sigma_beta_binomial_index += 1

                    log_loss = np.where(valid_count, -ll, np.nan)
                elif family == "beta":
                    y_beta = np.clip(obs_raw_metric, eps, 1.0 - eps)
                    concentration = exposure_weight
                    alpha = np.maximum(pred_prob * sigma_beta[sigma_beta_index] * concentration, eps)
                    beta = np.maximum((1.0 - pred_prob) * sigma_beta[sigma_beta_index] * concentration, eps)
                    ll = (
                        sps.gammaln(alpha + beta)
                        - sps.gammaln(alpha)
                        - sps.gammaln(beta)
                        + (alpha - 1.0) * np.log(y_beta)
                        + (beta - 1.0) * np.log1p(-y_beta)
                    )
                    valid_beta = valid & np.isfinite(obs_raw_metric)
                    log_loss = np.where(valid_beta, -ll, np.nan)
                    sigma_beta_index += 1
                elif family == "poisson":
                    count_exposure = np.exp(raw_exposure)
                    y_count = np.where(np.isfinite(obs_raw_metric), np.maximum(obs_raw_metric, 0.0), np.nan)
                    mu_count = np.maximum(predicted_metric * count_exposure / 36.0, eps)
                    ll = y_count * np.log(mu_count) - mu_count - sps.gammaln(y_count + 1.0)
                    valid_count = valid & np.isfinite(y_count) & np.isfinite(count_exposure) & (count_exposure > 0)
                    log_loss = np.where(valid_count, -ll, np.nan)
                elif family == "negative-binomial":
                    count_exposure = np.exp(raw_exposure)
                    y_count = np.where(np.isfinite(obs_raw_metric), np.maximum(obs_raw_metric, 0.0), np.nan)
                    mu_count = np.maximum(predicted_metric * count_exposure / 36.0, eps)
                    r = np.maximum(sigma_negative_binomial[sigma_neg_bin_index] if hasattr(sigma_negative_binomial, "__len__") else sigma_negative_binomial, eps)
                    ll = (
                        sps.gammaln(y_count + r)
                        - sps.gammaln(r)
                        - sps.gammaln(y_count + 1.0)
                        + r * np.log(r / (r + mu_count))
                        + y_count * np.log(mu_count / (r + mu_count))
                    )
                    valid_count = valid & np.isfinite(y_count) & np.isfinite(count_exposure) & (count_exposure > 0)
                    log_loss = np.where(valid_count, -ll, np.nan)
                    sigma_neg_bin_index += 1
                elif family == "gaussian":
                    obs_gaussian = obs_metric
                    variance = jnp.square(sigma[sigma_index]) / np.maximum(exposure_weight, eps)
                    ll = -0.5 * (np.log(2.0 * np.pi * variance) + np.square(obs_gaussian - predicted_metric) / variance)
                    log_loss = np.where(valid, -ll, np.nan)
                    sigma_index += 1
                else:
                    variance = 1.0 / np.maximum(exposure_weight, eps)
                    ll = -0.5 * (np.log(2.0 * np.pi * variance) + np.square(obs_metric - predicted_metric) / variance)
                    log_loss = np.where(valid, -ll, np.nan)

                return log_loss, sigma_beta_index, sigma_beta_binomial_index, sigma_index, sigma_neg_bin_index

            log_loss_point, sigma_beta_index, sigma_beta_binomial_index, sigma_index, sigma_neg_bin_index = compute_pointwise_log_loss(pred_metric, sigma_beta_binomial_index, sigma_index, sigma_beta_index, sigma_neg_bin_index)
            valid_log_loss = valid & np.isfinite(log_loss_point)
            if np.any(valid_log_loss):
                avg_log_loss = np.sum(np.where(valid_log_loss, log_loss_point, 0.0)) / np.sum(valid_log_loss)
            else:
                avg_log_loss = np.nan
            n_obs = np.sum(valid)
        else:
            
            bias = np.nan
            rmse = np.nan
            avg_log_loss = np.nan
            n_obs = 0

        rows.append(
            {
                "metric": metric_name,
                "bias": bias,
                "rmse": rmse,
                
                "avg_log_loss": avg_log_loss,
                "n_obs": n_obs,
            }
        )
    
    return pd.DataFrame(rows)


def summarize_metric_error_injury_splits(
    posterior_mean_map,
    observations,
    exposures,
    metric_outputs,
    metrics,
    injury_time_mask,
    de_trend_values=None,
    evaluation_mask=None,
    naive_mean_reference_mask=None,
    include_all=False,
):
    injury_mask_array = np.asarray(injury_time_mask).astype(bool)
    metric_count = len(metrics)

    if injury_mask_array.ndim == 3:
        if injury_mask_array.shape[-1] == metric_count:
            injury_eval_mask = injury_mask_array
        elif injury_mask_array.shape[0] == metric_count:
            injury_eval_mask = np.transpose(injury_mask_array, (1, 2, 0))
        else:
            raise ValueError(
                "injury_time_mask with 3 dimensions must be shaped (N, T, K) or (K, N, T)."
            )
    elif injury_mask_array.ndim == 2:
        injury_eval_mask = injury_mask_array
    else:
        raise ValueError("injury_time_mask must be a 2D (N, T) or 3D mask.")

    def _align_mask_dims(mask_a, mask_b):
        if mask_a.ndim == mask_b.ndim:
            return mask_a, mask_b
        if (mask_a.ndim == 2) and (mask_b.ndim == 3):
            return mask_a[..., None], mask_b
        if (mask_a.ndim == 3) and (mask_b.ndim == 2):
            return mask_a, mask_b[..., None]
        raise ValueError("Masks must be 2D or 3D with compatible dimensions.")

    base_eval_mask = None if evaluation_mask is None else np.asarray(evaluation_mask).astype(bool)

    if base_eval_mask is None:
        injured_mask = injury_eval_mask
        non_injured_mask = ~injury_eval_mask
    else:
        eval_aligned, injury_aligned = _align_mask_dims(base_eval_mask, injury_eval_mask)
        injured_mask = eval_aligned & injury_aligned
        non_injured_mask = eval_aligned & (~injury_aligned)

    if naive_mean_reference_mask is None:
        injured_naive_reference = None
        non_injured_naive_reference = None
    else:
        naive_reference = np.asarray(naive_mean_reference_mask).astype(bool)
        naive_aligned, injury_for_naive = _align_mask_dims(naive_reference, injury_eval_mask)
        injured_naive_reference = naive_aligned & injury_for_naive
        non_injured_naive_reference = naive_aligned & (~injury_for_naive)

    split_frames = []
    if include_all:
        summary_all = summarize_metric_error_observed_substitutions(
            posterior_mean_map=posterior_mean_map,
            observations=observations,
            exposures=exposures,
            metric_outputs=metric_outputs,
            metrics=metrics,
            de_trend_values=de_trend_values,
            evaluation_mask=base_eval_mask,
            naive_mean_reference_mask=naive_mean_reference_mask,
        )
        summary_all["split"] = "all"
        split_frames.append(summary_all)

    summary_injured = summarize_metric_error_observed_substitutions(
        posterior_mean_map=posterior_mean_map,
        observations=observations,
        exposures=exposures,
        metric_outputs=metric_outputs,
        metrics=metrics,
        de_trend_values=de_trend_values,
        evaluation_mask=injured_mask,
        naive_mean_reference_mask=injured_naive_reference,
    )
    summary_injured["split"] = "injured"
    split_frames.append(summary_injured)

    summary_non_injured = summarize_metric_error_observed_substitutions(
        posterior_mean_map=posterior_mean_map,
        observations=observations,
        exposures=exposures,
        metric_outputs=metric_outputs,
        metrics=metrics,
        de_trend_values=de_trend_values,
        evaluation_mask=non_injured_mask,
        naive_mean_reference_mask=non_injured_naive_reference,
    )
    summary_non_injured["split"] = "non_injured"
    split_frames.append(summary_non_injured)

    return pd.concat(split_frames, ignore_index=True)


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
    AR = AR - jnp.mean(AR, axis=-1, keepdims=True)
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


def make_mu_linear_mcmc(X, ls_deriv, alpha_time, weights, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time,rank,  offset_dict):
    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    # spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(jax.vmap(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))))(alpha_time, ls_deriv)))


    weights = weights * spd_time.T[None] 
    # weights = weights * jnp.transpose(spd_time[None, :, None, :], (0,1,2, 4,3))
    psi_x = X
    t_max = jnp.tanh(jnp.einsum("ijnm, ijm... -> ijn...", X, t_max_raw * sigma_t_max[None, None, None, :])   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = jnp.einsum("ijnm, ijm... -> ijn...", X, c_max * sigma_c_max[None, None, None, :])  + offset_dict["c_max"]

    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time), jnp.squeeze(L_time), M_time)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]

    gamma_phi_gamma_x = jnp.einsum("...nm, ...mdk, ...nktdz, ...jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("...nm, ...mdk, ...nkdz, ...jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, ...mdk, ...nkdz, ...jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, ...mdk, ...nktdz, ...jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)

    return None, mu, t_max, c_max, 0, second_deriv, third_deriv, first_deriv

def make_mu_linear_mcmc_AR(X, ls_deriv, alpha_time, weights, c_max, t_max_raw, sigma_t_max, sigma_c_max, L_time, M_time, phi_time, shifted_x_time,rank,  offset_dict, beta_ar, sigma_ar, rho_ar, AR_0_raw, orthogonalize = False):
    spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
    weights = weights * spd_time.T[None] 
    psi_x = X
    t_max = jnp.tanh(jnp.einsum("ijnm, ijm... -> ijn...", X, t_max_raw * sigma_t_max[..., None, :])   + jnp.arctanh(offset_dict["t_max"]/10)) * 10 
    c_max = jnp.einsum("ijnm, ijm... -> ijn...", X, c_max * sigma_c_max[..., None, :])  + offset_dict["c_max"]
    phi_prime_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))))(t_max)
    phi_double_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))))(t_max)
    phi_triple_prime_tmax = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))))(t_max)
    phi_t_max = jax.vmap(jax.vmap(jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))))(t_max)
    phi_prime_t =  vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time), jnp.squeeze(L_time), M_time)
    intercept = jnp.swapaxes(c_max, -2, -1)[..., None]
    gamma_phi_gamma_x = jnp.einsum("...nm, ...mdk, ...nktdz, ...jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = intercept + gamma_phi_gamma_x
    second_deriv = -1 * jnp.einsum("...nm, ...mdk, ...nkdz, ...jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, ...mdk, ...nkdz, ...jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, ...mdk, ...nktdz, ...jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    AR_0 = AR_0_raw * (sigma_ar / jnp.sqrt((1 - jnp.square(rho_ar))))
    def transition_fn(prev, z_t):
            next = prev * rho_ar + z_t * sigma_ar
            return next, next
        
    _, AR = jax.lax.scan(f = transition_fn, init = AR_0, xs = jnp.transpose(beta_ar, (2, 0, 1, 3, 4)))
    AR = jnp.transpose(AR, (1,2,3,4,0))
    AR = AR - jnp.mean(AR, axis=-1, keepdims=True)

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
    AR = AR - jnp.mean(AR, axis=-1, keepdims=True)
    second_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x)
    third_deriv = -1 * jnp.einsum("...nm, mdk, ...nkdz, jzk, ...nj -> ...nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x)
    first_deriv = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_prime_t_max[:, :,:,:, None, ...] - phi_prime_t[None, None, None, None], weights, psi_x)
    gamma_phi_gamma_x = jnp.einsum("...nm, mdk, ...nktdz, jzk, ...nj -> ...knt", psi_x, weights, phi_t_max[:,:,:,:,None,...] - phi_time[None, None, None, None] +  phi_prime_t_max[:, :,:,:, None, ...] * (((shifted_x_time - L_time)[None, None, None, None] - t_max[...,None])[..., None, None]), weights, psi_x)
    mu = (gamma_phi_gamma_x + intercept)
    if orthogonalize:
        Q = jnp.stack([mu, jnp.ones_like(mu)], axis = -1)
        AR = orthogonalize_ar(AR, Q)
    return None, mu, t_max, c_max, AR, second_deriv, third_deriv, first_deriv

def compute_residuals_map(map_values, obs_values, exposures, metric_output, metrics, gaussian_var = None, nb_variance = None, beta_binom_variance = None, beta_variance = None):
    residuals = []
    rho = []
    gaussian_index = 0
    negative_binom_index = 0
    beta_index = 0
    beta_binom_index = 0
    for index, metric_type in enumerate(metric_output):
        # Skip retirement — terminal event, not suitable for AR estimation
        if metrics[index] == "retirement":
            if metric_type == "beta-binomial":
                beta_binom_index += 1
            residuals.append(jnp.array(0.0))
            rho.append(jnp.array(0.0))
            continue

        if metric_type == "gaussian":
            # resid in original space; weight = 1/Var_obs = (exposure^2 - 1) / sigma^2
            resid = map_values[..., index] - obs_values[..., index]
            eff_exposure = jnp.square(exposures[index]) - 1.0
            weight = jnp.maximum(eff_exposure, 1e-6) / jnp.square(gaussian_var[gaussian_index])
            gaussian_index += 1
        elif metric_type in ["poisson", "negative-binomial"]:
            # resid in log-space; weight = precision of log(Y) = 1/Var(log Y)
            # obs=0 gives log(0)=-inf residual, correctly masked by isfinite check below
            mu = map_values[..., index] * jnp.exp(exposures[index]) / 36.0
            resid = jnp.log(obs_values[..., index] / 36) - jnp.log(map_values[..., index] / 36)
            if metric_type == "negative-binomial":
                r = nb_variance[negative_binom_index]
                # Var(log Y) = 1/mu + 1/r for NB; weight = harmonic precision
                weight = 1.0 / (1.0 / jnp.maximum(mu, 1e-6) + 1.0 / jnp.maximum(r, 1e-6))
                negative_binom_index += 1
            else:
                # Var(log Y) = 1/mu for Poisson
                weight = jnp.maximum(mu, 1e-6)
        elif metric_type in ["binomial", "beta-binomial"]:
            # resid in logit-space; obs=0 or obs=1 gives ±inf residual, masked below
            p_hat = jnp.clip(map_values[..., index], 1e-4, 1 - 1e-4)
            p_factor = p_hat * (1.0 - p_hat)
            multiplier = 1.0
            if metric_type == "beta-binomial":
                phi = beta_binom_variance[beta_binom_index]
                multiplier = (exposures[index] + phi) / (1.0 + phi)
                beta_binom_index += 1
            resid = jsci.logit(obs_values[..., index]) - jsci.logit(p_hat)
            # weight = precision of logit(Y/n) = n * p*(1-p) / multiplier
            weight = exposures[index] * p_factor / multiplier
        elif metric_type in ["beta"]:
            # resid in logit-space; Var(logit Y) ≈ 1/(phi * p*(1-p)), phi = exposure * beta_var
            p_hat = jnp.clip(map_values[..., index], 1e-4, 1 - 1e-4)
            p_factor = p_hat * (1.0 - p_hat)
            multiplier = exposures[index] * beta_variance[beta_index]
            resid = jsci.logit(obs_values[..., index]) - jsci.logit(p_hat)
            weight = p_factor / jnp.maximum(multiplier, 1e-6)
            beta_index += 1

        # mask handles NaN obs (missing seasons), obs=0 for counts, obs=0/1 for proportions
        residual_weighted = jnp.square(resid) * weight
        mask = jnp.isfinite(residual_weighted)
        valid_w_sum = jnp.sum(mask * weight, axis=-1)
        valid_n = jnp.sum(mask, axis=-1)
        avg_residuals = jnp.nanmean(jnp.sum(mask * residual_weighted, axis=-1) / valid_w_sum)
        # noise_floor = E[T_i / sum_t(w_it)] = mean observation variance per player
        # E[avg_residuals] = sigma_total^2 + noise_floor, so subtract to isolate AR signal
        noise_floor = jnp.nanmean(valid_n / valid_w_sum)
        sigma_total_sq_metric = jnp.maximum(avg_residuals - noise_floor, 0.0)
        residuals.append(sigma_total_sq_metric)
        # autocorr: NaN resid propagates to both adjacent pairs, so no cross-gap correlation
        autocorr_resid_weighted = jnp.sqrt(weight[..., 1:] * weight[..., 0:-1]) * resid[..., 1:] * resid[..., 0:-1]
        mask_autocorr = jnp.isfinite(autocorr_resid_weighted)
        autocorr = jnp.nanmean(jnp.sum(mask_autocorr * autocorr_resid_weighted, axis=-1) / jnp.sum(mask * residual_weighted, axis=-1))
        rho.append(autocorr)

    # residuals already has noise floor subtracted; clamp to 0 if model explains all variance
    rho_arr = jnp.array(rho)
    sigma_total_sq = jnp.maximum(jnp.array(residuals), 0.0)
    sigma_ar_per_metric = jnp.sqrt(sigma_total_sq * (1.0 - jnp.square(rho_arr)))

    # Boolean mask: True for non-retirement metrics
    non_ret_mask = jnp.array([m != "retirement" for m in metrics])

    # --- Per-metric lognormal prior on sigma_ar ---
    # mu_ln_i = log(sigma_ar_i); shared sigma_ln from cross-metric spread
    log_sigma_all = jnp.log(jnp.maximum(sigma_ar_per_metric, 1e-8))
    log_sigma_non_ret = jnp.where(non_ret_mask, log_sigma_all, jnp.nan)
    if jnp.any(non_ret_mask):
        pooled_mu_ln = jnp.nanmean(log_sigma_non_ret)
        sigma_ln = (
            jnp.sqrt(jnp.nanmean(jnp.square(log_sigma_non_ret - pooled_mu_ln)))
            if jnp.sum(non_ret_mask) >= 2
            else jnp.array(0.5)
        )
    else:
        pooled_mu_ln = jnp.array(-3.0)   # median sigma_ar ≈ 0.05 as fallback
        sigma_ln = jnp.array(1.0)
    # Retirement metrics get the pooled mean as a neutral placeholder
    per_metric_mu_ln = jnp.where(non_ret_mask, log_sigma_all, pooled_mu_ln)
    lognormal_params = (per_metric_mu_ln, sigma_ln)

    # --- Per-metric beta prior on rho ---
    # Transform rho in (-1,1) to p in (0,1): p_i = (rho_i + 1) / 2
    # Shared concentration from cross-metric variance; per-metric alpha/beta from each p_i
    p_all = jnp.clip((rho_arr + 1.0) / 2.0, 1e-4, 1 - 1e-4)
    p_non_ret = jnp.where(non_ret_mask, p_all, jnp.nan)
    if jnp.any(non_ret_mask):
        pooled_m = jnp.nanmean(p_non_ret)
        v = (
            jnp.nanmean(jnp.square(p_non_ret - pooled_m))
            if jnp.sum(non_ret_mask) >= 2
            else jnp.array(1.0 / 12.0)
        )
    else:
        pooled_m = jnp.array(0.5)
        v = jnp.array(1.0 / 12.0)
    concentration = jnp.maximum(pooled_m * (1.0 - pooled_m) / jnp.maximum(v, 1e-8) - 1.0, 1e-4)
    p_for_params = jnp.where(non_ret_mask, p_all, pooled_m)
    beta_params = (p_for_params * concentration, (1.0 - p_for_params) * concentration)

    return sigma_ar_per_metric, rho_arr, lognormal_params, beta_params

def apply_detrend_for_offsets(Y_obs, exposures_obs, metric_families, de_trend_values, de_trend_mask, eps=1e-6):
    Y_adj = Y_obs
    for idx, family in enumerate(metric_families):
        if not bool(de_trend_mask[idx]):
            continue

        trend_i = de_trend_values[idx]
        y_i = Y_adj[idx]
        exp_i = exposures_obs[idx]

        if family == "gaussian":
            y_new = y_i - trend_i
        elif family in ["poisson", "negative-binomial"]:
            y_new = y_i * jnp.exp(-trend_i)
        elif family == "beta":
            valid = jnp.isfinite(y_i)
            p_obs = jnp.clip(jnp.where(valid, y_i, 0.5), eps, 1 - eps)
            logit_p = jnp.log(p_obs / (1 - p_obs))
            p_det = jax.nn.sigmoid(logit_p - trend_i)
            y_new = jnp.where(valid, p_det, y_i)
        elif family in ["binomial", "beta-binomial", "bernoulli"]:
            valid = jnp.isfinite(y_i) & jnp.isfinite(exp_i) & (exp_i > 0)
            p_obs = jnp.clip(jnp.where(valid, y_i / exp_i, 0.5), eps, 1 - eps)
            logit_p = jnp.log(p_obs / (1 - p_obs))
            p_det = jax.nn.sigmoid(logit_p - trend_i)
            y_new = jnp.where(valid, p_det * exp_i, y_i)
        else:
            y_new = y_i

        Y_adj = Y_adj.at[idx].set(y_new)
    return Y_adj


def _linear_predictor_eta_weight_valid(y_i, exp_i, family, eps=1e-6):
    if family == "gaussian":
        eta_i = y_i
        weight_i = jnp.where(jnp.isfinite(exp_i), jnp.maximum(jnp.square(exp_i) - 1.0, 1.0), 1.0)
        valid_i = jnp.isfinite(eta_i) & jnp.isfinite(weight_i) & (weight_i > 0)
    elif family in ["poisson", "negative-binomial"]:
        count_exposure = jnp.exp(exp_i)
        rate_per36 = 36.0 * y_i / jnp.maximum(count_exposure, eps)
        eta_i = jnp.log(jnp.maximum(rate_per36, eps))
        weight_i = jnp.where(
            jnp.isfinite(count_exposure) & (count_exposure > 0),
            count_exposure,
            1.0,
        )
        valid_i = jnp.isfinite(y_i) & (y_i >= 0) & jnp.isfinite(eta_i) & jnp.isfinite(weight_i) & (weight_i > 0)
    elif family in ["binomial", "beta-binomial", "bernoulli"]:
        n_i = exp_i
        p_i = jnp.clip(y_i / jnp.maximum(n_i, eps), eps, 1.0 - eps)
        eta_i = jnp.log(p_i / (1.0 - p_i))
        weight_i = jnp.where(jnp.isfinite(n_i) & (n_i > 0), n_i, 1.0)
        valid_i = jnp.isfinite(y_i) & jnp.isfinite(n_i) & (n_i > 0) & (y_i >= 0) & (y_i <= n_i) & jnp.isfinite(eta_i)
    elif family == "beta":
        p_i = jnp.clip(y_i, eps, 1.0 - eps)
        eta_i = jnp.log(p_i / (1.0 - p_i))
        beta_weight = jnp.square(exp_i) - 1.0
        weight_i = jnp.where(jnp.isfinite(beta_weight) & (beta_weight > 0), beta_weight, 1.0)
        valid_i = jnp.isfinite(y_i) & jnp.isfinite(eta_i) & jnp.isfinite(weight_i) & (weight_i > 0)
    else:
        eta_i = y_i
        weight_i = jnp.where(jnp.isfinite(exp_i) & (exp_i > 0), exp_i, 1.0)
        valid_i = jnp.isfinite(eta_i) & jnp.isfinite(weight_i) & (weight_i > 0)

    return eta_i, weight_i, valid_i


def compute_linear_predictor_mean_offsets(Y, exposures, metric_output, eps=1e-6):
    """
    Compute a metric-level weighted mean offset on the linear predictor scale.

    Inputs are expected to be de-trended observations in natural space and the
    encoded exposure tensors used by the model.
    """
    offset_values = []
    for idx, family in enumerate(metric_output):
        y_i = jnp.asarray(Y[idx])
        exp_i = jnp.asarray(exposures[idx])
        eta_i, weight_i, valid_i = _linear_predictor_eta_weight_valid(y_i, exp_i, family, eps=eps)

        weighted_sum = jnp.nansum(jnp.where(valid_i, eta_i * weight_i, 0.0))
        weight_sum = jnp.nansum(jnp.where(valid_i, weight_i, 0.0))
        offset_i = jnp.where(weight_sum > 0, weighted_sum / jnp.maximum(weight_sum, eps), 0.0)
        offset_values.append(offset_i)

    return jnp.array(offset_values)


def compute_linear_predictor_delta_peak_offsets(Y, exposures, metric_output, intercept_offsets, eps=1e-6):
    """
    Compute metric-level peak-time offsets on the index scale where the linear
    predictor delta (eta - intercept_offset) is largest.
    """
    peak_means = []
    peak_vars = []

    for idx, family in enumerate(metric_output):
        y_i = jnp.asarray(Y[idx])
        exp_i = jnp.asarray(exposures[idx])
        eta_i, weight_i, valid_i = _linear_predictor_eta_weight_valid(y_i, exp_i, family, eps=eps)

        delta_i = eta_i - intercept_offsets[idx]
        delta_safe = jnp.where(valid_i, delta_i, -jnp.inf)
        row_has_valid = jnp.any(valid_i, axis=-1)
        peak_idx = jnp.argmax(delta_safe, axis=-1)
        peak_idx = jnp.where(row_has_valid, peak_idx, jnp.nan)

        row_weight = jnp.nansum(jnp.where(valid_i, weight_i, 0.0), axis=-1)
        valid_row = row_has_valid & (row_weight > 0)
        valid_weight = jnp.where(valid_row, row_weight, 0.0)
        weight_sum = jnp.sum(valid_weight)
        weight_sum_safe = jnp.maximum(weight_sum, eps)

        default_peak = (delta_i.shape[-1] - 1) / 2.0
        peak_mean = jnp.nansum(jnp.where(valid_row, peak_idx, 0.0) * valid_weight) / weight_sum_safe
        peak_mean = jnp.where(weight_sum > 0, peak_mean, default_peak)

        peak_var = jnp.nansum(valid_weight * jnp.where(valid_row, jnp.square(peak_idx - peak_mean), 0.0)) / weight_sum_safe
        peak_var = jnp.nan_to_num(peak_var, nan=1.0, posinf=1.0, neginf=1.0)
        peak_mean = jnp.nan_to_num(peak_mean, nan=default_peak, posinf=default_peak, neginf=default_peak)

        peak_means.append(peak_mean)
        peak_vars.append(peak_var)

    return jnp.array(peak_means), jnp.array(peak_vars)

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
    
    def _safe_nanmax_last_axis(x):
        valid = jnp.isfinite(x)
        x_safe = jnp.where(valid, x, -jnp.inf)
        max_v = jnp.max(x_safe, axis=-1)
        return jnp.where(jnp.any(valid, axis=-1), max_v, jnp.nan)

    def _safe_top_k_mean_last_axis(x, k=2):
        """Mean of top-k values per player — less noisy than single max."""
        valid = jnp.isfinite(x)
        x_safe = jnp.where(valid, x, -jnp.inf)
        top_k = jnp.sort(x_safe, axis=-1)[..., -k:]
        valid_top = jnp.isfinite(top_k)
        top_k_safe = jnp.where(valid_top, top_k, 0.0)
        count = jnp.maximum(jnp.sum(valid_top.astype(jnp.float32), axis=-1), 1.0)
        mean_top_k = jnp.sum(top_k_safe, axis=-1) / count
        return jnp.where(jnp.any(valid, axis=-1), mean_top_k, jnp.nan)

    def _safe_nanargmax_last_axis(x):
        valid = jnp.isfinite(x)
        x_safe = jnp.where(valid, x, -jnp.inf)
        argmax_v = jnp.argmax(x_safe, axis=-1)
        return jnp.where(jnp.any(valid, axis=-1), argmax_v, jnp.nan)

    for idx, family in enumerate(metric_output):
        weight = exposures[idx] if exposure_list[idx] != "simple_exposure" else exposures[exposure_list.index("minutes")]
        weight = jnp.where(jnp.isfinite(weight), weight, jnp.nan)
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

        valid_scaled = jnp.isfinite(Y_scaled)
        Y_scaled = jnp.where(valid_scaled, Y_scaled, jnp.nan)

        individual_weight = jnp.where(jnp.isfinite(individual_weight), individual_weight, 0.0)
        individual_weight = jnp.where(individual_weight > 0, individual_weight, 0.0)

        max_per_obs = _safe_top_k_mean_last_axis(Y_scaled, k=2)
        peak_idx = _safe_nanargmax_last_axis(Y_scaled)

        valid_obs = jnp.isfinite(max_per_obs) & jnp.isfinite(peak_idx) & (individual_weight > 0)
        valid_weight = jnp.where(valid_obs, individual_weight, 0.0)
        weight_sum = jnp.sum(valid_weight)
        weight_sum_safe = jnp.maximum(weight_sum, 1e-8)
        default_peak = (Y_scaled.shape[-1] - 1) / 2.0

        # weighted max and peak
        p_max = jnp.nansum(jnp.where(valid_obs, max_per_obs, 0.0) * valid_weight) / weight_sum_safe
        peak = jnp.nansum(jnp.where(valid_obs, peak_idx, 0.0) * valid_weight) / weight_sum_safe
        peak = jnp.where(weight_sum > 0, peak, default_peak)

        # weighted variance of max and peak
        p_max_var = jnp.nansum(valid_weight * jnp.where(valid_obs, (max_per_obs - p_max)**2, 0.0)) / weight_sum_safe
        peak_var = jnp.nansum(valid_weight * jnp.where(valid_obs, (peak_idx - peak)**2, 0.0)) / weight_sum_safe

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

        mu_eta = jnp.nan_to_num(mu_eta, nan=0.0, posinf=0.0, neginf=0.0)
        tau2 = jnp.nan_to_num(tau2, nan=0.1, posinf=0.1, neginf=0.1)
        peak = jnp.nan_to_num(peak, nan=default_peak, posinf=default_peak, neginf=default_peak)
        peak_var = jnp.nan_to_num(peak_var, nan=1.0, posinf=1.0, neginf=1.0)

        prior_max_mean.append(mu_eta)
        prior_max_var.append(tau2)
        prior_peak_mean.append(peak)
        prior_peak_var.append(peak_var)
    return jnp.array(prior_max_mean), jnp.array(prior_max_var), jnp.array(prior_peak_mean), jnp.array(prior_peak_var)


def make_survival_linear_injury_mcmc(
    X,
    entrance,
    entrance_global_offset,
    exit_global_offset,
    sigma_entrance,
    entrance_latent_raw,
    exit,
    exit_rate,
    injury_factor,
    injury_exit_loading,
    injury_exit_global_offset,
    sigma_injury_exit,
    injury_exit_raw,
    injury_indicator,
    injury_type,
    entrance_times,
    left_censor,
    basis,
    eps=1e-6,
    entry_shift=1.0,
    random_seed=0,
):
    entrance_times = jnp.maximum(jnp.nan_to_num(entrance_times, nan=eps, posinf=1e3, neginf=eps), eps)
    left_censor = left_censor.astype(bool)
    observed_mask = 1.0 - left_censor.astype(entrance_times.dtype)
    observed_count = jnp.maximum(observed_mask.sum(), 1.0)
    empirical_log1p_mean = jnp.sum(jnp.log(entrance_times + entry_shift) * observed_mask) / observed_count

    entrance_raw = jnp.einsum("...nr,...r->...n", X, entrance)
    entrance_loc = entrance_global_offset[..., None] + entrance_raw + empirical_log1p_mean
    entrance_dist = dist.LogNormal(entrance_loc, 0.35 + sigma_entrance[..., None])
    entrance_latent = jnp.where(
        left_censor[None, None],
        jnp.maximum(entrance_latent_raw - entry_shift, eps),
        entrance_times[None, None],
    )

    if injury_indicator.ndim == 3:
        injury_indicator = injury_indicator[0]
    if injury_type.ndim == 3:
        injury_type = injury_type[0]

    scale_min = 8
    scale_max = 15
    exit_raw = jnp.einsum("...nr,...r->...n", X, exit) 
    scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)[..., None]
    injury_exit_mean = jnp.einsum("...ip,...p->...i", injury_factor, injury_exit_loading)
    injury_exit_total = (
        injury_exit_global_offset[..., None, None, None]
        + injury_exit_mean[..., None, None, :]
        + injury_exit_raw * sigma_injury_exit[..., None, None, None]
    )
    injury_base_channel = jnp.zeros_like(injury_exit_total[..., :1])
    injury_effect_exit = jnp.take_along_axis(
        jnp.concatenate([injury_base_channel, injury_exit_total], axis=-1),
        injury_type[None, None, ..., None],
        axis=-1,
    ).squeeze(-1)
    exit_rate_base = jnp.einsum("...nr,...r->...n", X, exit_rate)[..., None]
    exit_rate_raw = exit_rate_base + injury_effect_exit + exit_global_offset
    concentration = 1.0 + 2*jax.nn.sigmoid(exit_rate_raw)
    concentration_is_time_varying = jnp.any(
        jnp.abs(concentration - concentration[..., :1]) > 1e-6,
        axis=-1,
    )

    basis_duration = jnp.maximum(jnp.asarray(basis) - 18.0, 0.0)
    num_intervals = concentration.shape[-1]
    interval_starts = jnp.arange(num_intervals, dtype=concentration.dtype)[None, None, None, :]
    interval_ends = interval_starts + 1.0
    entry = entrance_latent[..., None]

    def cumulative_h_at_stop(stop_t):
        stop = jnp.broadcast_to(stop_t, entrance_latent.shape)
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop[..., None])
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.maximum(seg_start, eps)
        seg_end_safe = jnp.maximum(seg_end, eps)
        log_scale = jnp.log(scale)
        seg_start_exp = jnp.clip(concentration * (jnp.log(seg_start_safe) - log_scale), a_min=-40.0, a_max=40.0)
        seg_end_exp = jnp.clip(concentration * (jnp.log(seg_end_safe) - log_scale), a_min=-40.0, a_max=40.0)
        delta_h = jnp.where(valid_seg, jnp.exp(seg_end_exp) - jnp.exp(seg_start_exp), 0.0)
        return delta_h.sum(axis=-1)

    cumulative_h_interval = jax.vmap(cumulative_h_at_stop, in_axes=0, out_axes=-1)(basis_duration)

    concentration_const = concentration[..., 0]

    def cumulative_h_closed_at_stop(stop_t):
        stop = jnp.broadcast_to(stop_t, entrance_latent.shape)
        entry_effective = jnp.maximum(entrance_latent, 0.0)
        stop_effective = jnp.maximum(stop, 0.0)
        has_window = stop_effective > entry_effective
        scale_base = scale.squeeze(-1)
        return jnp.where(
            has_window,
            jnp.power(stop_effective / scale_base, concentration_const)
            - jnp.power(entry_effective / scale_base, concentration_const),
            0.0,
        )

    cumulative_h_closed = jax.vmap(cumulative_h_closed_at_stop, in_axes=0, out_axes=-1)(basis_duration)
    cumulative_h = jnp.where(
        concentration_is_time_varying[..., None],
        cumulative_h_interval,
        cumulative_h_closed,
    )
    at_risk = basis_duration[None, None, None, :] >= entrance_latent[..., None]
    exit_survival = jnp.where(at_risk, jnp.exp(-cumulative_h), jnp.nan)

    stop_grid = basis_duration[None, None, None, :]
    interval_idx = jnp.clip(jnp.floor(stop_grid).astype(jnp.int32), 0, num_intervals - 1)
    concentration_grid_interval = jnp.take_along_axis(concentration, interval_idx, axis=-1)
    concentration_grid_closed = concentration_const[..., None]
    concentration_grid = jnp.where(
        concentration_is_time_varying[..., None],
        concentration_grid_interval,
        concentration_grid_closed,
    )
    duration_grid = jnp.maximum(stop_grid, eps)
    exit_hazard = (concentration_grid / scale) * jnp.power(duration_grid / scale, concentration_grid - 1.0)
    exit_hazard = jnp.where(at_risk, exit_hazard, jnp.nan)

    key = random.PRNGKey(random_seed)
    u = jnp.clip(random.uniform(key, shape=entrance_latent.shape), eps, 1.0 - eps)
    target = -jnp.log(u)

    scale_base = scale.squeeze(-1)

    def sample_step(interval_idx, state):
        sampled_so_far, cumulative_h_so_far, active_so_far = state
        k_i = concentration[..., interval_idx]
        start_i = jnp.maximum(jnp.asarray(interval_idx, dtype=entrance_latent.dtype), entrance_latent)
        end_i = jnp.asarray(interval_idx + 1.0, dtype=entrance_latent.dtype)
        valid_i = end_i > start_i
        delta_h_i = jnp.where(
            valid_i,
            jnp.power(end_i / scale_base, k_i) - jnp.power(start_i / scale_base, k_i),
            0.0,
        )

        hit_i = active_so_far & valid_i & (cumulative_h_so_far + delta_h_i >= target)
        rem_i = target - cumulative_h_so_far
        start_term_i = jnp.power(start_i / scale_base, k_i)
        sampled_i = scale_base * jnp.power(
            jnp.maximum(rem_i + start_term_i, eps),
            1.0 / jnp.maximum(k_i, eps),
        )
        sampled_i = jnp.clip(sampled_i, start_i, end_i)

        sampled_next = jnp.where(hit_i, sampled_i, sampled_so_far)
        cumulative_h_next = jnp.where(active_so_far & (~hit_i), cumulative_h_so_far + delta_h_i, cumulative_h_so_far)
        active_next = active_so_far & (~hit_i)
        return sampled_next, cumulative_h_next, active_next

    sampled_init = jnp.full_like(entrance_latent, jnp.nan)
    cumulative_h_init = jnp.zeros_like(entrance_latent)
    active_init = jnp.ones_like(entrance_latent, dtype=bool)
    sampled_exit_duration, _, active_final = jax.lax.fori_loop(
        0,
        num_intervals,
        sample_step,
        (sampled_init, cumulative_h_init, active_init),
    )
    sampled_exit_duration = jnp.where(active_final, jnp.asarray(float(num_intervals), dtype=sampled_exit_duration.dtype), sampled_exit_duration)
    sampled_exit_age = 18.0 + sampled_exit_duration

    entrance_survival = jax.vmap(lambda t: 1.0 - entrance_dist.cdf(t + entry_shift) + eps, in_axes=0, out_axes=-1)(basis_duration)
    entrance_hazard = jax.vmap(
        lambda t: jnp.exp(entrance_dist.log_prob(t + entry_shift)) / (1.0 - entrance_dist.cdf(t + entry_shift) + eps),
        in_axes=0,
        out_axes=-1,
    )(basis_duration)

    return {
        "entrance_latent": entrance_latent,
        "entrance_survival": entrance_survival,
        "entrance_hazard": entrance_hazard,
        "exit_survival": exit_survival,
        "exit_hazard": exit_hazard,
        "exit_concentration": concentration,
        "exit_scale": scale,
        "exit_duration_sample": sampled_exit_duration,
        "exit_age_sample": sampled_exit_age,
    }




def make_survival_linear_mcmc(
    X,
    entrance,
    entrance_global_offset,
    exit_global_offset,
    sigma_entrance,
    entrance_latent_raw,
    exit,
    exit_rate,
    entrance_times,
    left_censor,
    basis,
    eps=1e-6,
    entry_shift=0.0,
    random_seed=0,
):
    entrance_times = jnp.maximum(jnp.nan_to_num(entrance_times, nan=eps, posinf=1e3, neginf=eps), eps)
    left_censor = left_censor.astype(bool)
    observed_mask = 1.0 - left_censor.astype(entrance_times.dtype)
    observed_count = jnp.maximum(observed_mask.sum(), 1.0)
    empirical_log1p_mean = jnp.sum(jnp.log(entrance_times + entry_shift) * observed_mask) / observed_count

    entrance_raw = jnp.einsum("...nr,...r->...n", X, entrance)
    entrance_loc = entrance_global_offset[..., None] + entrance_raw + empirical_log1p_mean
    entrance_dist = dist.LogNormal(entrance_loc, 0.35 + sigma_entrance[..., None])
    entrance_latent = jnp.where(
        left_censor[None, None],
        jnp.maximum(entrance_latent_raw - entry_shift, eps),
        entrance_times[None, None],
    )



    exit_raw = jnp.einsum("...nr,...r->...n", X, exit)
    scale_min = 8
    scale_max = 15
    scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)[..., None]
    exit_rate_base = jnp.einsum("...nr,...r->...n", X, exit_rate)[..., None]
    exit_rate_raw = exit_rate_base  + exit_global_offset
    concentration = 1.0 + 2*jax.nn.sigmoid(exit_rate_raw)
    concentration_is_time_varying = jnp.any(
        jnp.abs(concentration - concentration[..., :1]) > 1e-6,
        axis=-1,
    )

    basis_duration = jnp.maximum(jnp.asarray(basis) - 18.0, 0.0)
    # concentration has shape (..., n_players, 1) — use basis length for the survival grid
    num_intervals = len(basis_duration)
    interval_starts = jnp.arange(num_intervals, dtype=concentration.dtype)[None, None, None, :]
    interval_ends = interval_starts + 1.0
    entry = entrance_latent[..., None]

    def cumulative_h_at_stop(stop_t):
        stop = jnp.broadcast_to(stop_t, entrance_latent.shape)
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop[..., None])
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.maximum(seg_start, eps)
        seg_end_safe = jnp.maximum(seg_end, eps)
        log_scale = jnp.log(scale)
        seg_start_exp = jnp.clip(concentration * (jnp.log(seg_start_safe) - log_scale), a_min=-40.0, a_max=40.0)
        seg_end_exp = jnp.clip(concentration * (jnp.log(seg_end_safe) - log_scale), a_min=-40.0, a_max=40.0)
        delta_h = jnp.where(valid_seg, jnp.exp(seg_end_exp) - jnp.exp(seg_start_exp), 0.0)
        return delta_h.sum(axis=-1)

    cumulative_h_interval = jax.vmap(cumulative_h_at_stop, in_axes=0, out_axes=-1)(basis_duration)

    concentration_const = concentration[..., 0]

    def cumulative_h_closed_at_stop(stop_t):
        stop = jnp.broadcast_to(stop_t, entrance_latent.shape)
        entry_effective = jnp.maximum(entrance_latent, 0.0)
        stop_effective = jnp.maximum(stop, 0.0)
        has_window = stop_effective > entry_effective
        scale_base = scale.squeeze(-1)
        return jnp.where(
            has_window,
            jnp.power(stop_effective / scale_base, concentration_const)
            - jnp.power(entry_effective / scale_base, concentration_const),
            0.0,
        )

    cumulative_h_closed = jax.vmap(cumulative_h_closed_at_stop, in_axes=0, out_axes=-1)(basis_duration)
    cumulative_h = jnp.where(
        concentration_is_time_varying[..., None],
        cumulative_h_interval,
        cumulative_h_closed,
    )
    at_risk = basis_duration[None, None, None, :] >= entrance_latent[..., None]
    exit_survival = jnp.where(at_risk, jnp.exp(-cumulative_h), jnp.nan)

    stop_grid = basis_duration[None, None, None, :]
    interval_idx = jnp.clip(jnp.floor(stop_grid).astype(jnp.int32), 0, num_intervals - 1)
    concentration_grid_interval = jnp.take_along_axis(concentration, interval_idx, axis=-1)
    concentration_grid_closed = concentration_const[..., None]
    concentration_grid = jnp.where(
        concentration_is_time_varying[..., None],
        concentration_grid_interval,
        concentration_grid_closed,
    )
    duration_grid = jnp.maximum(stop_grid, eps)
    exit_hazard = (concentration_grid / scale) * jnp.power(duration_grid / scale, concentration_grid - 1.0)
    exit_hazard = jnp.where(at_risk, exit_hazard, jnp.nan)

    key = random.PRNGKey(random_seed)
    u = jnp.clip(random.uniform(key, shape=entrance_latent.shape), eps, 1.0 - eps)
    target = -jnp.log(u)

    scale_base = scale.squeeze(-1)
    # concentration has shape (..., n_players, 1) — constant hazard across time.
    # Use Weibull inverse-CDF directly, conditioning on T > entrance_latent:
    #   H(t) = (t / scale)^k  =>  t = scale * (target + (entrance/scale)^k)^(1/k)
    k_const = concentration_const
    entrance_term = jnp.power(jnp.maximum(entrance_latent, 0.0) / scale_base, k_const)
    sampled_exit_duration = scale_base * jnp.power(
        jnp.maximum(target + entrance_term, eps),
        1.0 / jnp.maximum(k_const, eps),
    )
    max_duration = float(basis_duration[-1])
    sampled_exit_duration = jnp.clip(sampled_exit_duration, entrance_latent, max_duration)
    sampled_exit_age = 18.0 + sampled_exit_duration

    entrance_survival = jax.vmap(lambda t: 1.0 - entrance_dist.cdf(t + entry_shift) + eps, in_axes=0, out_axes=-1)(basis_duration)
    entrance_hazard = jax.vmap(
        lambda t: jnp.exp(entrance_dist.log_prob(t + entry_shift)) / (1.0 - entrance_dist.cdf(t + entry_shift) + eps),
        in_axes=0,
        out_axes=-1,
    )(basis_duration)

    return {
        "entrance_latent": entrance_latent,
        "entrance_survival": entrance_survival,
        "entrance_hazard": entrance_hazard,
        "exit_survival": exit_survival,
        "exit_hazard": exit_hazard,
        "exit_concentration": concentration,
        "exit_scale": scale,
        "exit_duration_sample": sampled_exit_duration,
        "exit_age_sample": sampled_exit_age,
    }


