from abc import abstractmethod, ABC
from types import SimpleNamespace
import jax 
import flax.serialization as ser
import numpyro
import numpy as np
from numpyro import sample 
from numpyro.infer.util import log_density
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, Weibull, StudentT, Independent, Beta, HalfCauchy, LogNormal, Binomial, HalfNormal, Categorical, MultivariateNormal, BetaProportion, Distribution, Uniform, BetaBinomial, Gamma, BinomialLogits, NegativeBinomial2, Dirichlet, MixtureSameFamily, LKJCholesky
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoNormal,  AutoLaplaceApproximation
from numpyro.handlers import substitute, seed, trace, mask, replay
import optax
from optax import linear_onecycle_schedule, adam
from jaxopt import LBFGS
from .hsgp import make_convex_f, make_psi_gamma, make_spectral_mixture_density, diag_spectral_density, make_convex_phi,  vmap_make_convex_phi, vmap_make_convex_phi_prime, eigenfunctions_multivariate, vmap_make_convex_phi_double_prime, vmap_make_convex_phi_triple_prime
import jax.numpy as jnp
import jax.scipy as jsci
from .MultiHMCGibbs import MultiHMCGibbs
from .HMCMetrics import NUTSWithMetrics
from model.model_utils import Type2Gumbel
from scipy.special import roots_legendre






def step_decay_schedule(init_lr, drop_every=10000, drop_factor=10, total_steps=100000):
    num_drops = total_steps // drop_every
    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(init_lr / (drop_factor ** i))
            for i in range(num_drops + 1)
        ],
        boundaries=[drop_every * i for i in range(1, num_drops + 1)]
    )


def _nonfinite_summary(x):
    try:
        arr = np.asarray(x)
    except Exception:
        return None, []
    bad = ~np.isfinite(arr)
    bad_count = int(np.sum(bad))
    if bad_count == 0:
        return 0, []
    bad_idx = np.argwhere(bad)
    preview = [tuple(int(i) for i in row) for row in bad_idx[:10]]
    return bad_count, preview


def _safe_take(x, idx):
    if x is None:
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.ndim == 0:
        return arr
    try:
        return arr[idx]
    except Exception:
        return None


def _fmt_value(x):
    if x is None:
        return "None"
    try:
        arr = np.asarray(x)
    except Exception:
        return str(x)
    if arr.size == 1:
        scalar = arr.item()
        if isinstance(scalar, (float, np.floating)):
            return f"{float(scalar):.8g}"
        return str(scalar)
    return f"array(shape={arr.shape})"


def _collect_nonfinite_tree_entries(tree, max_entries=20):
    entries = []

    def visit(node, path):
        if len(entries) >= max_entries:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                next_path = f"{path}.{key_text}" if path else key_text
                visit(value, next_path)
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                visit(value, next_path)
            return

        bad_count, bad_idx = _nonfinite_summary(node)
        if bad_count in (None, 0):
            return
        shape = None
        try:
            shape = np.asarray(node).shape
        except Exception:
            shape = None
        entries.append(
            {
                "path": path if path else "<root>",
                "shape": shape,
                "bad_count": bad_count,
                "bad_idx": bad_idx,
            }
        )

    visit(tree, "")
    return entries


def _distribution_snapshot(dist, bad_index):
    param_names = (
        "rate",
        "logits",
        "total_count",
        "concentration",
        "concentration0",
        "concentration1",
        "loc",
        "scale",
    )
    snapshots = {}
    for param_name in param_names:
        try:
            value = getattr(dist, param_name)
        except Exception:
            continue
        snapshots[param_name] = _safe_take(value, bad_index)
    return snapshots


def _summarize_trace_nonfinite_sites(trace_data, max_sites=25):
    issues = []
    for site_name, site in trace_data.items():
        if len(issues) >= max_sites:
            break
        if site.get("type") != "sample":
            continue
        value = site.get("value", None)
        dist = site.get("fn", None)
        if value is None or dist is None:
            continue

        value_bad_count, value_bad_idx = _nonfinite_summary(value)
        support_bad_count, support_bad_idx = (None, [])
        log_prob_bad_count, log_prob_bad_idx = (None, [])
        log_prob_error = None
        log_prob = None

        try:
            support_ok = dist.support.check(value)
            support_bad = ~np.asarray(support_ok)
            support_bad_count = int(np.sum(support_bad))
            if support_bad_count > 0:
                support_bad_idx = [tuple(int(i) for i in row) for row in np.argwhere(support_bad)[:10]]
        except Exception:
            support_bad_count = None

        try:
            log_prob = dist.log_prob(value)
            log_prob_bad_count, log_prob_bad_idx = _nonfinite_summary(log_prob)
        except Exception as error:
            log_prob_error = str(error)
            log_prob_bad_count = None

        has_issue = any(
            count not in (None, 0)
            for count in (value_bad_count, support_bad_count, log_prob_bad_count)
        ) or (log_prob_error is not None)
        if not has_issue:
            continue

        snapshots = {}
        if log_prob_bad_idx:
            for idx in log_prob_bad_idx[:5]:
                snapshots[idx] = _distribution_snapshot(dist, idx)

        issues.append(
            {
                "site": site_name,
                "observed": bool(site.get("is_observed", False)),
                "shape": getattr(value, "shape", None),
                "value_bad_count": value_bad_count,
                "value_bad_idx": value_bad_idx,
                "support_bad_count": support_bad_count,
                "support_bad_idx": support_bad_idx,
                "log_prob_bad_count": log_prob_bad_count,
                "log_prob_bad_idx": log_prob_bad_idx,
                "log_prob_error": log_prob_error,
                "dist_param_snapshots": snapshots,
            }
        )

    return issues


def _print_trace_issues(label, issues):
    if not issues:
        print(f"[{label}] No problematic sample sites found")
        return
    print(f"[{label}] Problematic sample sites:")
    for issue in issues:
        print(
            f"  - {issue['site']} (observed={issue['observed']}): "
            f"shape={issue['shape']}, value_nonfinite={issue['value_bad_count']}, "
            f"support_violations={issue['support_bad_count']}, "
            f"log_prob_nonfinite={issue['log_prob_bad_count']}"
        )
        if issue["value_bad_idx"]:
            print(f"      value bad idx (first 10): {issue['value_bad_idx']}")
        if issue["support_bad_idx"]:
            print(f"      support bad idx (first 10): {issue['support_bad_idx']}")
        if issue["log_prob_bad_idx"]:
            print(f"      log_prob bad idx (first 10): {issue['log_prob_bad_idx']}")
        if issue["log_prob_error"]:
            print(f"      log_prob error: {issue['log_prob_error']}")
        if issue["dist_param_snapshots"]:
            for idx, snapshot in issue["dist_param_snapshots"].items():
                text = ", ".join(f"{k}={_fmt_value(v)}" for k, v in snapshot.items())
                if text:
                    print(f"      idx={idx} dist params: {text}")


def _audit_svi_sample_sites(model_fn, guide, params, model_args, rng_key):
    result = {
        "guide_error": None,
        "model_error": None,
        "guide_issues": [],
        "model_issues": [],
    }
    try:
        guide_key, model_key = jax.random.split(rng_key)
        guide_trace = trace(seed(substitute(guide, data=params), guide_key)).get_trace(**model_args)
        result["guide_issues"] = _summarize_trace_nonfinite_sites(guide_trace)
    except Exception as error:
        result["guide_error"] = str(error)
        return result

    try:
        replayed_model = replay(model_fn, guide_trace)
        model_trace = trace(seed(replayed_model, model_key)).get_trace(**model_args)
        result["model_issues"] = _summarize_trace_nonfinite_sites(model_trace)
    except Exception as error:
        result["model_error"] = str(error)

    return result


def _sum_log_prob_terms(log_prob):
    try:
        return float(np.asarray(jnp.sum(log_prob)))
    except Exception:
        try:
            return float(np.sum(np.asarray(log_prob)))
        except Exception:
            return None


def _estimate_svi_loss_breakdown(
    model_fn,
    guide,
    params,
    model_args,
    rng_key,
    *,
    num_particles=8,
):
    num_particles = max(int(num_particles), 1)
    keys = jax.random.split(rng_key, num_particles)
    likelihood_terms = []
    prior_terms = []
    guide_terms = []
    errors = []

    for key in keys:
        try:
            guide_key, model_key = jax.random.split(key)
            guide_trace = trace(seed(substitute(guide, data=params), guide_key)).get_trace(**model_args)

            guide_log_prob = 0.0
            for _, site in guide_trace.items():
                if site.get("type") != "sample":
                    continue
                dist = site.get("fn", None)
                value = site.get("value", None)
                if dist is None or value is None:
                    continue
                log_prob_term = _sum_log_prob_terms(dist.log_prob(value))
                if log_prob_term is None:
                    raise ValueError("Failed to evaluate guide log_prob term")
                guide_log_prob += log_prob_term

            replayed_model = replay(model_fn, guide_trace)
            model_trace = trace(seed(replayed_model, model_key)).get_trace(**model_args)

            model_log_likelihood = 0.0
            model_log_prior = 0.0
            for site_name, site in model_trace.items():
                if site.get("type") != "sample":
                    continue
                dist = site.get("fn", None)
                value = site.get("value", None)
                if dist is None or value is None:
                    continue
                log_prob_term = _sum_log_prob_terms(dist.log_prob(value))
                if log_prob_term is None:
                    raise ValueError(f"Failed to evaluate model log_prob for site '{site_name}'")
                is_observed = bool(site.get("is_observed", False))
                is_likelihood_site = site_name.startswith("likelihood_") or site_name.startswith("log_lik_")
                if is_observed or is_likelihood_site:
                    model_log_likelihood += log_prob_term
                else:
                    model_log_prior += log_prob_term

            likelihood_terms.append(-model_log_likelihood)
            prior_terms.append(-model_log_prior)
            guide_terms.append(guide_log_prob)
        except Exception as error:
            errors.append(str(error))

    if not likelihood_terms:
        return {
            "ok": False,
            "error": errors[0] if errors else "Unknown error while estimating SVI loss breakdown",
        }

    likelihood_arr = np.asarray(likelihood_terms, dtype=float)
    prior_arr = np.asarray(prior_terms, dtype=float)
    guide_arr = np.asarray(guide_terms, dtype=float)
    total_arr = likelihood_arr + prior_arr + guide_arr
    return {
        "ok": True,
        "num_particles": int(likelihood_arr.shape[0]),
        "likelihood_loss_mean": float(np.mean(likelihood_arr)),
        "prior_loss_mean": float(np.mean(prior_arr)),
        "guide_loss_mean": float(np.mean(guide_arr)),
        "total_loss_mean": float(np.mean(total_arr)),
        "total_loss_std": float(np.std(total_arr)),
        "num_errors": len(errors),
    }


def _print_svi_loss_breakdown(
    model_fn,
    guide,
    params,
    model_args,
    rng_key,
    debug_label,
    *,
    num_particles=None,
):
    if num_particles is None:
        guide_name = type(guide).__name__
        num_particles = 1 if guide_name == "AutoDelta" else 8
    breakdown = _estimate_svi_loss_breakdown(
        model_fn=model_fn,
        guide=guide,
        params=params,
        model_args=model_args,
        rng_key=rng_key,
        num_particles=num_particles,
    )
    if not breakdown["ok"]:
        print(f"[{debug_label}] Could not compute final loss breakdown: {breakdown['error']}")
        return
    print(
        f"[{debug_label}] Final loss breakdown (MC mean over {breakdown['num_particles']} particles): "
        f"likelihood={breakdown['likelihood_loss_mean']:.6f}, "
        f"prior={breakdown['prior_loss_mean']:.6f}, "
        f"guide={breakdown['guide_loss_mean']:.6f}, "
        f"total={breakdown['total_loss_mean']:.6f} (std={breakdown['total_loss_std']:.6f})"
    )
    if breakdown["num_errors"] > 0:
        print(
            f"[{debug_label}] Loss breakdown had {breakdown['num_errors']} particle(s) skipped due to trace/log_prob errors"
        )


def _run_svi_with_nan_checks(
    svi,
    guide,
    model_fn,
    model_args,
    num_steps,
    *,
    rng_key=jax.random.PRNGKey(0),
    progress_bar=True,
    debug_nan=False,
    debug_label="SVI DEBUG",
    init_state=None,
):
    if not debug_nan:
        run_kwargs = {
            "num_steps": num_steps,
            "progress_bar": progress_bar,
            **model_args,
        }
        if init_state is not None:
            run_kwargs["init_state"] = init_state
        result = svi.run(rng_key, **run_kwargs)
        _print_svi_loss_breakdown(
            model_fn=model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.fold_in(rng_key, num_steps),
            debug_label=debug_label,
        )
        return result

    print(f"[{debug_label}] Step-wise NaN diagnostics enabled")
    state = init_state if init_state is not None else svi.init(rng_key, **model_args)
    losses = []
    for step in range(num_steps):
        state, loss = svi.update(state, **model_args)
        try:
            loss_value = float(np.asarray(loss))
        except Exception:
            loss_value = float("nan")
        losses.append(loss_value)

        params = svi.get_params(state)
        bad_params = _collect_nonfinite_tree_entries(params)
        loss_is_finite = bool(np.isfinite(loss_value))
        if loss_is_finite and not bad_params:
            continue

        print(
            f"[{debug_label}] Non-finite values detected at step {step + 1}/{num_steps}: "
            f"loss={loss_value}"
        )
        if bad_params:
            print(f"[{debug_label}] Non-finite parameter entries:")
            for entry in bad_params:
                print(
                    f"  - {entry['path']}: shape={entry['shape']}, "
                    f"nonfinite={entry['bad_count']}, bad idx (first 10)={entry['bad_idx']}"
                )

        audit = _audit_svi_sample_sites(
            model_fn=model_fn,
            guide=guide,
            params=params,
            model_args=model_args,
            rng_key=jax.random.fold_in(rng_key, step),
        )
        if audit["guide_error"] is not None:
            print(f"[{debug_label}] Could not trace guide sites: {audit['guide_error']}")
        else:
            _print_trace_issues(f"{debug_label} GUIDE", audit["guide_issues"])

        if audit["model_error"] is not None:
            print(f"[{debug_label}] Could not trace model sites: {audit['model_error']}")
        else:
            _print_trace_issues(f"{debug_label} MODEL", audit["model_issues"])

        raise FloatingPointError(
            f"Non-finite SVI state detected at step {step + 1}. "
            f"See {debug_label} logs for problematic parameters and sample sites."
        )

    result = SimpleNamespace(params=svi.get_params(state), state=state, losses=np.asarray(losses))
    _print_svi_loss_breakdown(
        model_fn=model_fn,
        guide=guide,
        params=result.params,
        model_args=model_args,
        rng_key=jax.random.fold_in(rng_key, num_steps),
        debug_label=debug_label,
    )
    return result


class LinearPredictorCompositionMixin:
    @staticmethod
    def _compose_additive(base_value, *effects):
        value = base_value
        for effect in effects:
            value = value + effect
        return value

    def _build_linear_predictor(self, mu, k_indices, *effects):
        base_linear_predictor = mu[k_indices]
        return self._compose_additive(base_linear_predictor, *effects)

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        return self._build_linear_predictor(mu, k_indices)

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            dist = Normal(linear_predictor[mask], expanded_sigmas[mask] / exposure[mask])
            return dist, obs[mask]
        if family == "poisson":
            dist = Poisson(jnp.exp(linear_predictor[mask] + exposure[mask]))
            return dist, obs[mask]
        if family == "binomial":
            dist = Binomial(logits=linear_predictor[mask], total_count=exposure[mask].astype(int))
            return dist, obs[mask]
        if family == "exponential":
            dist = Exponential(jnp.exp(linear_predictor[mask] + exposure[mask]))
            return dist, obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for default likelihood builder")

    def _sample_family_likelihoods(self, data_set, mu, *, prior: bool = False, **context):
        for family, family_data in data_set.items():
            linear_predictor = self._compute_family_linear_predictor(family, mu, family_data, **context)
            dist, obs = self._build_family_distribution(family, linear_predictor, family_data, **context)
            sample(f"likelihood_{family}", dist, obs if not prior else None)


class RFLVMBase(LinearPredictorCompositionMixin, ABC):
    """ 
    HMC implementation of 
    https://arxiv.org/pdf/2006.11145
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        self.r = latent_rank 
        self.m = rff_dim
        self.n, self.j, self.k = output_shape
        self.prior = {}
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["W"] = Normal()
        self.prior["beta"] = Normal()
        self.prior["X"] = Normal(loc=jnp.zeros((self.n, self.r)))
        self.prior["sigma"] = InverseGamma(1, 1)

    def _is_distribution_prior(self, key: str) -> bool:
        return isinstance(self.prior.get(key), Distribution)

    def _resolve_prior(self, key: str, sample_shape=None, site_name: str = None, dist_override=None):
        prior_value = self.prior.get(key)
        # When the numpyro site name differs from the internal prior key (e.g. "ar_0" vs "AR_0"),
        # fixed_params are stored under the site name — check it before falling through to sampling.
        if site_name and site_name != key:
            site_value = self.prior.get(site_name)
            if site_value is not None and not isinstance(site_value, Distribution):
                return site_value
        if isinstance(prior_value, Distribution):
            dist_to_sample = dist_override if dist_override is not None else prior_value
            sample_kwargs = {}
            # If the distribution is already batched to the requested shape (e.g. a per-metric
            # HalfNormal with scale of shape (k,1)), don't re-expand — its batch shape already
            # provides the per-metric dimensions. Scalar-param dists (batch_shape ()) expand as before.
            if sample_shape is not None and tuple(dist_to_sample.batch_shape) != tuple(sample_shape):
                sample_kwargs["sample_shape"] = sample_shape
            return sample(site_name or key, dist_to_sample, **sample_kwargs)
        return prior_value

    def _build_rff_features(self, X: jnp.ndarray, W: jnp.ndarray, lengthscale: jnp.ndarray = None):
        scaled_W = W if lengthscale is None else W * jnp.sqrt(lengthscale)
        wTx = jnp.einsum("nr,mr -> nm", X, scaled_W)
        phi = jnp.concatenate([jnp.cos(wTx), jnp.sin(wTx)], axis=-1) * (1 / jnp.sqrt(self.m))
        return wTx, phi

    def _project_X(self, X: jnp.ndarray, *, W: jnp.ndarray, lengthscale: jnp.ndarray = None):
        _, phi = self._build_rff_features(X, W, lengthscale)
        return phi

    def _resolve_latent_X(self, sample_free_indices=None, sample_fixed_indices=None):
        has_free = sample_free_indices is not None and np.asarray(sample_free_indices).size > 0
        has_fixed = sample_fixed_indices is not None and np.asarray(sample_fixed_indices).size > 0
        if has_free:
            X = jnp.zeros((self.n, self.r))
            X_free = self._resolve_prior("X_free", sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            if has_fixed:
                X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            return X
        return self._resolve_prior("X")

    def _sample_gaussian_sigmas(self, num_gaussians: int):
        sigmas = self._resolve_prior("sigma", sample_shape=(num_gaussians,))
        return jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
    
    def _stabilize_x(self, X):
        """Fix the rotation according to the SVD.
        """
        U, _, _ = jnp.linalg.svd(X, full_matrices=False)
        L       = jnp.linalg.cholesky(jnp.cov(U.T) + 1e-6 * jnp.eye(self.r)).T
        aligned_X  = jnp.linalg.solve(L, U.T).T
        return aligned_X / jnp.std(X, axis=0)

    @abstractmethod
    def model_fn(self, data_set) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        X = self._resolve_prior("X")
        phi = self._project_X(X, W=W)
        beta = self._resolve_prior("beta", sample_shape=(self.k, 2 * self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        self._sample_family_likelihoods(data_set, mu, expanded_sigmas=expanded_sigmas)
    @abstractmethod
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized:bool, model_args, initial_values = {}, thinning = 1):
        # Empty initial_values -> init each param at its prior median (init_to_median), the natural
        # "regularized to prior" start for data-poor players, instead of seeding from the (overfit) MAP.
        init_strategy = init_to_median() if not initial_values else init_to_value(values=initial_values)
        kernel = NUTS(self.model_fn, init_strategy=init_strategy)
        key = jax.random.PRNGKey(0)
        if vectorized:
            n_parallel = jax.local_device_count()
            n_vectorized = num_chains // n_parallel
            def do_mcmc(rng_key):
                mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=n_vectorized,
                progress_bar=False,
                chain_method="vectorized",
                thinning=thinning)
                mcmc.run(rng_key, **model_args)
                return {**mcmc.get_samples()}
            rng_keys = jax.random.split(key, n_parallel)
            traces = jax.pmap(do_mcmc)(rng_keys)
            return {k: jnp.concatenate(v) for k, v in traces.items()}, None
        else:
            mcmc = MCMC(kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains = num_chains,
                        chain_method="parallel",
                        thinning=thinning)
            mcmc.run(key, extra_fields=("potential_energy",), **model_args)
            return mcmc.get_samples(group_by_chain=True), mcmc
    @abstractmethod
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}, sample_shape = (4, 2000), debug_nan: bool = False):
        guide = AutoNormal(self.model_fn, prefix="", **guide_kwargs, init_loc_fn=init_to_value(values=initial_values),
                                   init_scale= 1e-10)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples
    
    @abstractmethod
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
       
        state = svi.init(jax.random.PRNGKey(0),**model_args)
        init_params = None
        init_state_bytes = None
        if isinstance(initial_state, dict) and (("state" in initial_state) or ("samples" in initial_state)):
            if initial_state.get("state") is not None:
                init_state_bytes = initial_state["state"]
            else:
                init_params = initial_state.get("samples")
        elif isinstance(initial_state, (bytes, bytearray)):
            init_state_bytes = initial_state
        elif initial_state is not None:
            init_params = initial_state

        if init_state_bytes is not None:
            state = ser.from_bytes(state, init_state_bytes)
            result = svi.run(jax.random.PRNGKey(0),
                            num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        else:
            if isinstance(init_params, dict) and (len(init_params) > 0) and all((isinstance(key, str) and (not key.endswith("__loc"))) for key in init_params.keys()):
                init_params = {f"{key}__loc": value for key, value in init_params.items()}
            result = svi.run(jax.random.PRNGKey(0),
                            num_steps = num_steps,progress_bar = True, init_state = state if init_params is None else None,
                            init_params=init_params, stable_update=True, **model_args)
        _print_svi_loss_breakdown(
            model_fn=self.model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.PRNGKey(1),
            debug_label=f"{type(self).__name__} MAP",
        )
        return result.params, result.state
    
    

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args, num_samples = 1000):
        predictive = Predictive(self.model_fn, posterior_samples,  num_samples=num_samples)
        return predictive(jax.random.PRNGKey(0), **model_args)


class RFLVMMaxBase(LinearPredictorCompositionMixin, ABC):
    """
    Abstract base class for Max-parameterized latent variable models.
    """
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def model_fn(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_svi_inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_map_inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        pass

    def _sample_max_raw_parameters(self, feature_dim: int):
        t_max_raw = self._resolve_prior("t_max_raw", sample_shape=(feature_dim, self.k))
        c_max_raw = self._resolve_prior("c_max", sample_shape=(feature_dim, self.k))
        return t_max_raw, c_max_raw

    def _build_t_max_curve(
        self,
        psi_x,
        t_max_raw,
        sigma_t_max,
        t_offset,
        prior: bool,
        *,
        amplitude: float = 10.0,
        offset_mode: str = "arctanh",
        scale_inside: bool = True,
        deterministic_amplitude: float | None = None,
    ):
        if scale_inside:
            t_base = make_psi_gamma(psi_x, t_max_raw * sigma_t_max)
        else:
            t_base = make_psi_gamma(psi_x, t_max_raw) * sigma_t_max


        if offset_mode == "arctanh":
            eps = 1e-6
            t_offset_scaled = jnp.clip(t_offset / amplitude, -1.0 + eps, 1.0 - eps)
            t_max_value = jnp.tanh(t_base + jnp.arctanh(t_offset_scaled)) * amplitude
            t_max_det_value = t_max_value
        elif offset_mode == "additive":
            t_max_value = jnp.tanh(t_base) * amplitude + t_offset
            det_amp = amplitude if deterministic_amplitude is None else deterministic_amplitude
            t_max_det_value = jnp.tanh(t_base) * det_amp + t_offset
        else:
            raise ValueError(f"Unknown offset_mode '{offset_mode}'")

        return t_max_value  # no deterministic recording; curves are produced via _compute_mu

    def _build_c_max_curve(self, psi_x, c_max_raw, sigma_c_max, c_offset, prior: bool, *, scale_inside: bool = True):
        c_offset_value = c_offset
        c_max_value = (
            make_psi_gamma(psi_x, c_max_raw * sigma_c_max)
            if scale_inside
            else make_psi_gamma(psi_x, c_max_raw) * sigma_c_max
        ) + c_offset_value
        return c_max_value  # no deterministic recording

    def _build_max_curves(self, psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max, t_offset, c_offset, prior: bool, *, amplitude: float = 10.0):
        t_max = self._build_t_max_curve(
            psi_x,
            t_max_raw,
            sigma_t_max,
            t_offset,
            prior,
            amplitude=amplitude,
            offset_mode="arctanh",
            scale_inside=True,
        )
        c_max = self._build_c_max_curve(
            psi_x,
            c_max_raw,
            sigma_c_max,
            c_offset,
            prior,
            scale_inside=True,
        )
        return t_max, c_max

    def _compute_phi_at_max(self, t_max, L_time, M_time):
        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)
        return phi_t_max, phi_prime_t_max

    def _compute_convex_mu(self, psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max, c_max, prior: bool, weight_offset = 0.0):
        intercept = jnp.transpose(c_max)[..., None]
        core_tensor = (
            phi_t_max[:, :, None, ...] - phi_time[None, None]
            + phi_prime_t_max[:, :, None, ...]
            * (((shifted_x_time - L_time)[None, None] - t_max[..., None])[..., None, None])
        )
        projected_weights = jnp.einsum("nm,mdk->nkd", psi_x, weights) + weight_offset
        gamma_phi_gamma_x = jnp.einsum("nkd,nktdz,nkz->knt", projected_weights, core_tensor, projected_weights)
        mu_value = intercept + gamma_phi_gamma_x
        return mu_value  # no deterministic recording

    def _build_mu_from_base(self, mu_base, prior: bool, *effects):
        mu_value = self._compose_additive(mu_base, *effects)
        return mu_value  # no deterministic recording

    def _orthogonalize_ar_to_mu(self, ar: jnp.ndarray, mu: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        mu_norm_sq = jnp.sum(jnp.square(mu), axis=-1, keepdims=True)
        ar_dot_mu = jnp.sum(ar * mu, axis=-1, keepdims=True)
        return ar - (ar_dot_mu / (mu_norm_sq + eps)) * mu

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return 0

    @staticmethod
    def _compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0):
        def transition_fn(prev, z_t):
            next_value = prev * rho_ar + z_t * sigma_ar
            return next_value, next_value

        _, ar = jax.lax.scan(f=transition_fn, init=ar_0, xs=z)
        ar = jnp.transpose(ar, (1, 2, 0))
        if ar.shape[-1] > 0:
            ar = ar - jnp.mean(ar, axis=-1, keepdims=True)
        return ar

    @staticmethod
    def _compute_frozen_ar_process(sigma_ar, rho_ar, z, ar_0, injury_mask):
        # injury_mask: (j, n) bool — True means this observation is an injury period.
        # During injury periods the AR state is held at its last healthy value;
        # propagation resumes from that frozen state on return to play.
        def transition_fn(carry, inputs):
            ar_prev, frozen_ar = carry
            z_t, is_injured_t = inputs          # z_t: (k, n), is_injured_t: (n,)
            propagated = ar_prev * rho_ar + z_t * sigma_ar
            ar_next = jnp.where(is_injured_t[None, :], frozen_ar, propagated)
            frozen_next = jnp.where(is_injured_t[None, :], frozen_ar, ar_next)
            return (ar_next, frozen_next), ar_next

        init = (ar_0, ar_0)
        _, ar = jax.lax.scan(f=transition_fn, init=init, xs=(z, injury_mask))
        ar = jnp.transpose(ar, (1, 2, 0))      # (j, k, n) -> (k, n, j)
        if ar.shape[-1] > 0:
            # mean-center using only healthy steps so injury periods don't bias the baseline
            healthy = ~jnp.transpose(injury_mask)           # (j, n) -> (n, j)
            healthy_sum = (ar * healthy[None]).sum(axis=-1, keepdims=True)       # (k, n, 1)
            healthy_count = jnp.maximum(healthy.sum(axis=-1, keepdims=True)[None], 1)  # (1, n, 1)
            ar = ar - healthy_sum / healthy_count
        return ar

    @staticmethod
    def _compute_ar1_calendar_process(sigma_ar, rho_ar, z, ar_0, ref_year_idx: int = 0):
        """AR(1) process over calendar years for era-level metric trends. No player dimension.

        rho_ar is used directly (the prior Uniform(0, 0.99) already enforces 0 < rho < 1,
        giving positive persistence without unit-root explosion).  High rho (e.g. 0.95-0.99)
        lets the process sustain monotonic 40-year era trends (e.g. fg3a: +1.6 log units,
        1980→2023).  The prior on sigma (HalfNormal(0.05)) means a z-shock of ~1 produces
        a ~0.05/year step — enough for the optimizer to track large league-wide shifts.

        Total variation is controlled jointly by sigma_ar (shock scale) and rho
        (persistence). The stationary std is sigma / sqrt(1 - rho^2); with
        sigma=0.05, rho=0.99 this is ~0.35 in log-rate space, sufficient to
        span the full fg3a era range.

        A reference-year anchor is applied: the value at ref_year_idx is subtracted
        from the entire trajectory so that TREND_AR[ref_year] = 0 by construction.
        This resolves the identifiability ambiguity between the per-player c_max
        level and the calendar-year trend level — without a reference year the
        optimizer can freely shift c_max up and TREND_AR down (or vice-versa),
        producing an inverted trend.  Setting ref_year_idx=0 (the earliest year)
        means all later years are expressed as deviations from that baseline.

        Args:
            sigma_ar:     (num_ar, 1) — shock std per metric (HalfNormal prior)
            rho_ar:       (num_ar, 1) — raw AR coefficient; mapped via tanh → (-1, 1)
            z:            (num_years, num_ar) — iid shocks per year per metric
            ar_0:         (1, num_ar) — initial state (lag-1), scaled by sigma
            ref_year_idx: int — year index (0-based) to pin to zero

        Returns:
            (num_ar, num_years) — reference-year-anchored AR(1) trajectory per metric
        """
        # Constrain to (-1, 1) — critical for stationarity and bounded extrapolation
        rho = rho_ar[..., 0]  # (num_ar,)

        def transition_fn(ar_t1, z_t):          # z_t: (num_ar,)
            ar_t = rho * ar_t1 + sigma_ar[..., 0] * z_t
            return ar_t, ar_t

        _, ar = jax.lax.scan(
            f=transition_fn,
            init=ar_0[0],
            xs=z,
        )
        # ar: (num_years, num_ar) → (num_ar, num_years)
        ar = jnp.transpose(ar)
        # Anchor: subtract reference year so TREND_AR[ref_year_idx] = 0 for all metrics
        ar = ar - ar[:, ref_year_idx : ref_year_idx + 1]
        return ar


class RFLVM(RFLVMBase):

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        return super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning=thinning)

    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)

    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)



    
class TVRFLVM(RFLVM):
    """
    model for time varying functional 
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
        self.basis = basis ### basis for time dimension
        self.t = len(basis)
    

    def make_kernel(self, lengthscale, jitter = 1e-6):
        deltaXsq = jnp.power((self.basis[:, None] - self.basis), 2.0)
        k = jnp.exp(-0.5 * deltaXsq / lengthscale) + jitter * jnp.eye(self.basis.shape[0])
        return k

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(1.0, 1.0)


    def model_fn(self, data_set) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        X = self._resolve_prior("X")
        phi = self._project_X(X, W=W)

        ls = self._resolve_prior("lengthscale")
        kernel = self.make_kernel(ls)
        beta = self._resolve_prior(
            "beta",
            site_name="beta",
            dist_override=MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel),
            sample_shape=(self.k, 2 * self.m),
        )
        mu = jnp.einsum("nm,kmj -> knj", phi, beta)
        expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        self._sample_family_likelihoods(data_set, mu, expanded_sigmas=expanded_sigmas)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values,thinning=thinning)
    
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape, debug_nan=debug_nan)

    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)



    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexTVRFLVM(TVRFLVM):
    """
    model for time varying functional enforcing convexity in the shape parameters
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale_deriv"] = InverseGamma(2.0, 1.0)
        self.prior["lengthscale"] = InverseGamma(.3, .7)
        self.prior["sigma_beta"] = InverseGamma(2.0, 1.0)
        self.prior["sigma"] = InverseGamma(300.0, 6000.0)
        self.prior["alpha"] = LogNormal(jnp.log(3.0), 0.35)
        self.prior["intercept"] = Normal()
        self.prior["slope"] = Normal()

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            dist = Normal(linear_predictor[mask], expanded_sigmas[mask] / exposure[mask])
            return dist, obs[mask]
        if family == "poisson":
            rate = jnp.exp(linear_predictor[mask] + exposure[mask])
            return Poisson(rate), obs[mask]
        if family == "binomial":
            dist = BinomialLogits(logits=linear_predictor[mask], total_count=exposure[mask].astype(int))
            return dist, obs[mask]
        if family == "beta":
            sigma_beta = context["sigma_beta"]
            rate = jsci.special.expit(linear_predictor[mask])
            return BetaProportion(rate, exposure[mask] * sigma_beta), obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexTVRFLVM likelihood builder")


    def model_fn(self, data_set, hsgp_params, offsets = 0, prior = False) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self._resolve_prior("lengthscale", sample_shape=(self.r,))
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        X = self._resolve_prior("X")
        X -= jnp.mean(X, keepdims = True, axis = 0)
        X /= jnp.std(X, keepdims = True, axis = 0)
        psi_x = self._project_X(X, W=W, lengthscale=lengthscale[None])

     
        slope = make_psi_gamma(psi_x, self._resolve_prior("slope", sample_shape=(self.m * 2, self.k)))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k,))
        intercept = make_psi_gamma(psi_x, self._resolve_prior("intercept", sample_shape=(2 * self.m, self.k)))
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        weights = self._resolve_prior("beta", sample_shape=(self.m * 2, M_time, self.k))
        weights = weights * spd * .0001
        gamma_phi_gamma_x = jnp.einsum("nm, mdk, tdz, jzk, nj -> nkt", psi_x, weights, phi_time, weights, psi_x)
        mu = make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None])  # no deterministic recording
        if num_gaussians > 0 :
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        sigma_beta = self._resolve_prior("sigma_beta")
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            sigma_beta=sigma_beta,
        )

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values,thinning=thinning)
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn = init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        state = svi.init(jax.random.PRNGKey(0), **model_args)
        init_params = None
        init_state_bytes = None
        if isinstance(initial_state, dict) and (("state" in initial_state) or ("samples" in initial_state)):
            if initial_state.get("state") is not None:
                init_state_bytes = initial_state["state"]
            else:
                init_params = initial_state.get("samples")
        elif isinstance(initial_state, (bytes, bytearray)):
            init_state_bytes = initial_state
        elif initial_state is not None:
            init_params = initial_state

        if init_state_bytes is not None:
            state = ser.from_bytes(state, init_state_bytes)
            result = svi.run(jax.random.PRNGKey(0),
                             num_steps = num_steps,progress_bar = True, init_state=state, stable_update=True, **model_args)
        else:
            if isinstance(init_params, dict) and (len(init_params) > 0) and all((isinstance(key, str) and (not key.endswith("__loc"))) for key in init_params.keys()):
                init_params = {f"{key}__loc": value for key, value in init_params.items()}
            result = svi.run(jax.random.PRNGKey(0),
                             num_steps = num_steps,progress_bar = True, init_params=init_params, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_values, sample_shape, debug_nan=debug_nan)



    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    


class ConvexMaxTVRFLVM(ConvexTVRFLVM, RFLVMMaxBase):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = HalfNormal()
        self.prior["sigma_beta"] = Exponential()
        self.prior["sigma"] = InverseGamma(1000.0, 14985.0)
        self.prior["sigma_negative_binomial"] = InverseGamma(3, 2)
        self.prior["lengthscale_deriv"] = HalfNormal()
        self.prior["sigma_boundary_l"] = HalfNormal(.1)
        self.prior["sigma_boundary_r"] = HalfNormal(.1)
        self.prior["sigma_t"] = InverseGamma(2, 1)
        self.prior["sigma_c"] = InverseGamma(2, 1)
        self.prior["alpha"] = HalfNormal(.01)
        self.prior["sigma_c"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_t"] = InverseGamma(2.0, 1.0)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["W_t_max"] = Normal()
        self.prior["W_c_max"] = Normal()
        self.prior["lengthscale_t_max"] = HalfNormal()
        self.prior["lengthscale_c_max"] = HalfNormal()
        self.prior["X_free"] = Normal()

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            return Normal(linear_predictor[mask], expanded_sigmas[mask] / exposure[mask]), obs[mask]
        if family == "poisson":
            log_rate = linear_predictor
            return Poisson(jnp.exp(log_rate[mask] + exposure[mask])), obs[mask]
        if family == "negative-binomial":
            expanded_sigma_neg_bin = context["expanded_sigma_neg_bin"]
            log_rate = linear_predictor
            dist = NegativeBinomial2(
                mean=jnp.exp(log_rate[mask] + exposure[mask]),
                concentration=expanded_sigma_neg_bin[mask],
            )
            return dist, obs[mask]
        if family == "binomial":
            return BinomialLogits(logits=linear_predictor[mask], total_count=exposure[mask].astype(int)), obs[mask]
        if family == "beta-binomial":
            logit_rate = linear_predictor
            dist = BetaBinomial(
                concentration0=(1 - jsci.special.expit(logit_rate[mask])),
                concentration1=jsci.special.expit(logit_rate[mask]),
                total_count=exposure[mask].astype(int),
            )
            return dist, obs[mask]
        if family == "beta":
            return BetaProportion(jsci.special.expit(linear_predictor[mask]), jnp.square(exposure[mask])), obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexMaxTVRFLVM likelihood builder")

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self._resolve_prior("lengthscale", sample_shape=(self.r,))
        lengthscale_t_max = self._resolve_prior("lengthscale_t_max", sample_shape=(self.r,))
        lengthscale_c_max = self._resolve_prior("lengthscale_c_max", sample_shape=(self.r,))
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        W_t_max = self._resolve_prior("W_t_max", sample_shape=(self.m, self.r))
        W_c_max = self._resolve_prior("W_c_max", sample_shape=(self.m, self.r))
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(1, self.k))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(1, self.k))
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(2 * self.m)
        
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        X = self._resolve_latent_X(sample_free_indices, sample_fixed_indices)


        psi_x = self._project_X(X, W=W, lengthscale=lengthscale[None])
        psi_x_t_max = self._project_X(X, W=W_t_max, lengthscale=lengthscale_t_max[None])
        psi_x_c_max = self._project_X(X, W=W_c_max, lengthscale=lengthscale_c_max[None])
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.k, 1)) 
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.k, 1))
        if c_offset is None:
            c_offset = offsets["c_max"]
        t_max, c_max = self._build_max_curves(
            psi_x_t_max,
            t_max_raw,
            c_max_raw,
            sigma_t_max,
            sigma_c_max,
            t_offset,
            c_offset,
            prior,
            amplitude=hsgp_params["t_amplitude"],
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd.T[None]


        mu = self._compute_convex_mu(
            psi_x,
            weights,
            phi_t_max,
            phi_prime_t_max,
            phi_time,
            shifted_x_time,
            L_time,
            t_max,
            c_max,
            prior,
        )
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
        )


              


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
 
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
 
  
        state = svi.init(jax.random.PRNGKey(0),**model_args)
        init_params = None
        init_state_bytes = None
        if isinstance(initial_state, dict) and (("state" in initial_state) or ("samples" in initial_state)):
            if initial_state.get("state") is not None:
                init_state_bytes = initial_state["state"]
            else:
                init_params = initial_state.get("samples")
        elif isinstance(initial_state, (bytes, bytearray)):
            init_state_bytes = initial_state
        elif initial_state is not None:
            init_params = initial_state

        if init_state_bytes is not None:
            state = ser.from_bytes(state, init_state_bytes)
            result = svi.run(jax.random.PRNGKey(0),
                            num_steps = num_steps,progress_bar = True, init_state = state, stable_update=True, **model_args)
        else:
            if isinstance(init_params, dict) and (len(init_params) > 0) and all((isinstance(key, str) and (not key.endswith("__loc"))) for key in init_params.keys()):
                init_params = {f"{key}__loc": value for key, value in init_params.items()}
            result = svi.run(jax.random.PRNGKey(0),
                            num_steps = num_steps,progress_bar = True, init_state = state if init_params is None else None,
                            init_params=init_params, stable_update=True, **model_args)

        _print_svi_loss_breakdown(
            model_fn=self.model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.PRNGKey(1),
            debug_label=f"{type(self).__name__} MAP",
        )
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs,init_loc_fn=init_to_value(values=initial_values))
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)












class ConvexMaxARTVRFLVM(ConvexMaxTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
    
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = Type2Gumbel(alpha=.05, scale=.001)
        self.prior["beta_ar"] = Normal()
        self.prior["ar_0"] = Normal()

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        ar = context["ar"]
        return self._build_linear_predictor(mu, k_indices, ar[k_indices])

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)

    
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self._resolve_prior("lengthscale", sample_shape=(self.r,))
        lengthscale_t_max = self._resolve_prior("lengthscale_t_max", sample_shape=(self.r,))
        lengthscale_c_max = self._resolve_prior("lengthscale_c_max", sample_shape=(self.r,))
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        W_t_max = self._resolve_prior("W_t_max", sample_shape=(self.m, self.r))
        W_c_max = self._resolve_prior("W_c_max", sample_shape=(self.m, self.r))
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(1, self.k))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(1, self.k))
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(2 * self.m)
        
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_neg_bins > 0:
            sigma_negative_binomial = 5 + self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        
        X = self._resolve_latent_X(sample_free_indices, sample_fixed_indices)

        psi_x = self._project_X(X, W=W, lengthscale=lengthscale[None])
        psi_x_t_max = self._project_X(X, W=W_t_max, lengthscale=lengthscale_t_max[None])
        psi_x_c_max = self._project_X(X, W=W_c_max, lengthscale=lengthscale_c_max[None])
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.k, 1))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.k, 1))
        if c_offset is None:
            c_offset = offsets["c_max"]
        t_max, c_max = self._build_max_curves(
            psi_x,
            t_max_raw,
            c_max_raw,
            sigma_t_max,
            sigma_c_max,
            t_offset,
            c_offset,
            prior,
            amplitude=hsgp_params["t_amplitude"],
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd.T[None]

        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * sigma_ar
        AR = self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)
        mu = self._compute_convex_mu(
            psi_x,
            weights,
            phi_t_max,
            phi_prime_t_max,
            phi_time,
            shifted_x_time,
            L_time,
            t_max,
            c_max,
            prior,
        )
        
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            ar=AR,
        )


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)

    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0),
                        num_steps = num_steps,progress_bar = True, init_params = initial_values, stable_update=True, **model_args)
        return result.params

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_values: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):

        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs,init_loc_fn=init_to_value(values=initial_values))
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)





class GibbsRFLVM(RFLVM):
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple) -> None:
        super().__init__(latent_rank, rff_dim, output_shape)
    
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
    
    def model_fn(self, data_set) -> None:
        return super().model_fn(data_set)
    
    def run_inference(self, num_warmup, num_samples, num_chains, model_args, gibbs_sites: list = [], thinning = 1):
        n_parallel = jax.local_device_count()
        n_vectorized = num_chains // n_parallel
        inner_kernels = [NUTS(self.model_fn) for _ in range(len(gibbs_sites))]
        outer_kernel = MultiHMCGibbs(inner_kernels, gibbs_sites_list=gibbs_sites)
        def do_mcmc(rng_key):
            mcmc = MCMC(
            outer_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=n_vectorized,
            progress_bar=False,
            chain_method="vectorized",
            thinning = thinning)
            mcmc.run(rng_key, **model_args)
            return {**mcmc.get_samples()}
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_parallel)
        traces = jax.pmap(do_mcmc)(rng_keys)
        return {k: jnp.concatenate(v) for k, v in traces.items()}
    
    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
    




        






class HSGPLVMBase(LinearPredictorCompositionMixin, ABC):
    """ 
    Latent Variable model with Hilbert Space GP Approximation
    """
    def __init__(self, latent_rank: int, hsgp_dim: list[int] | int, output_shape: tuple, L_X: jnp.array, basis: jnp.array = None) -> None:
        self.r = latent_rank 
        self.m = int(np.prod(hsgp_dim * np.ones(self.r))) 
        self.M_X = hsgp_dim
        self.n, self.j, self.k = output_shape
        self.t = len(basis)
        self.prior = {}
        self.L_X = L_X ## per dimension of X 
        self.basis = basis
    
    @abstractmethod
    def initialize_priors(self, *args, **kwargs) -> None:
        self.prior["beta"] = Normal()
        self.prior["X"] = Normal(loc=jnp.zeros((self.n, self.r)))
        self.prior["sigma"] = InverseGamma(1, 1)
        self.prior["lengthscale"] = HalfNormal()
        self.prior["alpha_X"] = HalfNormal(.1)
        self.prior["intercept"] = Normal()
        self.prior["sigma_intercept"] = HalfNormal(.001)

    def _resolve_prior(self, key: str, sample_shape=None, site_name: str = None, dist_override=None):
        prior_value = self.prior.get(key)
        # When the numpyro site name differs from the internal prior key (e.g. "ar_0" vs "AR_0"),
        # fixed_params are stored under the site name — check it before falling through to sampling.
        if site_name and site_name != key:
            site_value = self.prior.get(site_name)
            if site_value is not None and not isinstance(site_value, Distribution):
                return site_value
        if isinstance(prior_value, Distribution):
            dist_to_sample = dist_override if dist_override is not None else prior_value
            sample_kwargs = {}
            # If the distribution is already batched to the requested shape (e.g. a per-metric
            # HalfNormal with scale of shape (k,1)), don't re-expand — its batch shape already
            # provides the per-metric dimensions. Scalar-param dists (batch_shape ()) expand as before.
            if sample_shape is not None and tuple(dist_to_sample.batch_shape) != tuple(sample_shape):
                sample_kwargs["sample_shape"] = sample_shape
            return sample(site_name or key, dist_to_sample, **sample_kwargs)
        return prior_value

    def _resolve_latent_X(self, sample_free_indices=None, sample_fixed_indices=None):
        has_free = sample_free_indices is not None and np.asarray(sample_free_indices).size > 0
        has_fixed = sample_fixed_indices is not None and np.asarray(sample_fixed_indices).size > 0
        if has_free:
            X = jnp.zeros((self.n, self.r))
            X_free = self._resolve_prior("X_free", sample_shape=(len(sample_free_indices), self.r))
            X = X.at[sample_free_indices].set(X_free)
            if has_fixed:
                X = X.at[sample_fixed_indices].set(self.prior["X"].at[sample_fixed_indices].get())
            return X
        return self._resolve_prior("X")

    def _sample_gaussian_sigmas(self, num_gaussians: int):
        sigmas = self._resolve_prior("sigma", sample_shape=(num_gaussians,))
        return jnp.tile(sigmas[:, None, None], (1, self.n, self.j))
    
    def _stabilize_x(self, X):
        """Make sure X is within [-1, 1]^D
        """
        
        return jnp.tanh(X)

    def _project_X(self, X: jnp.ndarray):
        return eigenfunctions_multivariate(X, self.L_X, self.M_X)

    @abstractmethod
    def model_fn(self, data_set) -> None:
        num_gaussians = data_set["gaussian"]["Y"].shape[0]
        
        X_raw = self._resolve_prior("X")
        X = self._stabilize_x(X_raw)
        alpha_X = self._resolve_prior("alpha_X", sample_shape=(self.k,))

        lengthscale = self._resolve_prior("lengthscale", sample_shape=(self.r,))

        spd = jax.vmap(lambda alpha: jnp.sqrt(diag_spectral_density(self.r, alpha, lengthscale, lengthscale, self.M_X)))(alpha_X)    
        phi = self._project_X(X)
        beta = self._resolve_prior("beta", sample_shape=(self.k, self.m, self.j))
        mu = jnp.einsum("nm,kmj -> knj", phi, spd * beta)
        expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        self._sample_family_likelihoods(data_set, mu, expanded_sigmas=expanded_sigmas)
    @abstractmethod
    def run_inference(self, num_warmup, num_samples, num_chains, vectorized:bool, model_args, initial_values = {}, thinning = 1):
        # Empty initial_values -> init each param at its prior median (init_to_median), the natural
        # "regularized to prior" start for data-poor players, instead of seeding from the (overfit) MAP.
        init_strategy = init_to_median() if not initial_values else init_to_value(values=initial_values)
        kernel = NUTS(self.model_fn, init_strategy=init_strategy)
        key = jax.random.PRNGKey(0)
        if vectorized:
            n_parallel = jax.local_device_count()
            n_vectorized = num_chains // n_parallel
            def do_mcmc(rng_key):
                mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=n_vectorized,
                progress_bar=False,
                chain_method="vectorized",
                thinning=thinning)
                mcmc.run(rng_key, **model_args)
                return {**mcmc.get_samples()}
            rng_keys = jax.random.split(key, n_parallel)
            traces = jax.pmap(do_mcmc)(rng_keys)
            return {k: jnp.concatenate(v) for k, v in traces.items()}
        else:
            mcmc = MCMC(kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains = num_chains,
                        chain_method="parallel",
                        thinning=thinning)
            mcmc.run(key, **model_args)
            return mcmc.get_samples(group_by_chain=True)
    @abstractmethod
    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state:dict = {}, sample_shape = (4, 2000), debug_nan: bool = False):
        guide = AutoNormal(self.model_fn, prefix="", **guide_kwargs, init_state = initial_state,
                                   init_scale= 1e-10)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        effective_init_state = None if (initial_state is None or (isinstance(initial_state, dict) and len(initial_state) == 0)) else initial_state
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
            init_state=effective_init_state,
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples
    
    @abstractmethod
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state:dict = {}):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup MAP")
        result = svi.run(jax.random.PRNGKey(0), num_steps = num_steps,progress_bar = True, init_state=initial_state, **model_args)
        _print_svi_loss_breakdown(
            model_fn=self.model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.PRNGKey(1),
            debug_label=f"{type(self).__name__} MAP",
        )
        return result.params, result.state
    
    

    @abstractmethod
    def predict(self, posterior_samples: dict, model_args, num_samples = 1000):
        predictive = Predictive(self.model_fn, posterior_samples,  num_samples=num_samples)
        return predictive(jax.random.PRNGKey(0), **model_args)







class ConvexMaxTVLinearLVM(ConvexMaxTVRFLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis, player_covariates=None) -> None:
        self.r = latent_rank
        self.n, self.j, self.k = output_shape
        self.basis = basis ### basis for time dimension
        self.t = len(basis)
        self.prior = {}
        self.player_covariates = player_covariates  # (n, 2): standardized [neg_log_draft, height]

    def _project_X(self, X: jnp.ndarray, *, W: jnp.ndarray = None, lengthscale: jnp.ndarray = None):
        return X

    def _projected_feature_dim(self):
        """Width of psi_x = _project_X(X) — used to SIZE the level/peak/curvature/survival weights so
        they contract with psi_x. Linear/cosine: r (identity / scaled unit-norm). The RFF leaf
        overrides to 2*m. Distinct from _kernel_self_cov (the normalization scale)."""
        return self.r

    def _kernel_self_cov(self, psi_x):
        """Kernel self-covariance phi(x)^T phi(x) ~ K(x,x) used to NORMALIZE the projections so the
        curve/hazard amplitude tracks K(x,x) rather than the feature count (see _compute_convex_mu).
        Linear/cosine: ~ psi_x.shape[-1] = r (E||X||^2 = r / cosine ||phi||^2 = r). The RFF leaf
        overrides to 1.0 (a unit-norm SE-kernel approximation: ||phi||^2 = 1), which keeps phi norm-1
        — the standard RFF — while the curves stay correctly scaled."""
        return psi_x.shape[-1]

    def _build_c_max_curve(self, psi_x, c_max_raw, sigma_c_max, c_offset, prior: bool, *, scale_inside: bool = True):
        # Divide by sqrt(r) so that psi_x @ c_max_raw has unit variance regardless of r.
        # Without this, the effective std of c_max is sigma_c_max * sqrt(r) instead of sigma_c_max,
        # allowing the optimizer to push c_max deeply negative for sparse-metric players.
        n_features = self._kernel_self_cov(psi_x)
        c_max_value = (
            make_psi_gamma(psi_x, c_max_raw * sigma_c_max)
            if scale_inside
            else make_psi_gamma(psi_x, c_max_raw) * sigma_c_max
        ) / jnp.sqrt(n_features) + c_offset
        return c_max_value  # no deterministic recording

    def _build_t_max_curve(self, psi_x, t_max_raw, sigma_t_max, t_offset, prior: bool, *,
                           amplitude: float = 10.0, offset_mode: str = "arctanh",
                           scale_inside: bool = True, deterministic_amplitude: float | None = None):
        # Scaled-dot-product normalization (1/sqrt(r)) on the X @ t_max_raw projection, matching
        # _build_c_max_curve and _compute_convex_mu. The base method omits it; since psi_x = X here
        # (||X|| unbounded), the un-normalized t_base ~ sqrt(r)*sigma_t AND scales with ||X||, which
        # inflated the peak-age spread by ~sqrt(r) and pushed extreme-||X|| players to the tanh rails.
        n_features = self._kernel_self_cov(psi_x)
        if scale_inside:
            t_base = make_psi_gamma(psi_x, t_max_raw * sigma_t_max) / jnp.sqrt(n_features)
        else:
            t_base = make_psi_gamma(psi_x, t_max_raw) / jnp.sqrt(n_features) * sigma_t_max
        if offset_mode == "arctanh":
            eps = 1e-6
            t_offset_scaled = jnp.clip(t_offset / amplitude, -1.0 + eps, 1.0 - eps)
            return jnp.tanh(t_base + jnp.arctanh(t_offset_scaled)) * amplitude
        elif offset_mode == "additive":
            return jnp.tanh(t_base) * amplitude + t_offset
        raise ValueError(f"Unknown offset_mode '{offset_mode}'")

    def _compute_convex_mu(self, psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max, c_max, prior: bool, weight_offset=0.0):
        # Scaled-dot-product normalization: divide the projection gamma = X @ weights by sqrt(r),
        # so the quadratic descent gamma^T [.] gamma carries a 1/r factor and is invariant to the
        # latent dimension r (consistent with c_max/t_max, which get 1/sqrt(r) on their LINEAR
        # projections). This matches the paper's requirement that phi(x)^T phi(x) ~ K(x,x): the
        # descent amplitude should track the kernel self-covariance, not grow with r. NOTE: this is
        # still only a 1/r (dimension) normalization; because psi_x = X is a *linear* kernel, the
        # per-player ||X||^2 inflation (extreme covariates) is NOT removed by this — that needs a
        # bounded feature map (phi(x)^T phi(x) ~ const), handled separately.
        n_features = self._kernel_self_cov(psi_x)
        intercept = jnp.transpose(c_max)[..., None]
        core_tensor = (
            phi_t_max[:, :, None, ...] - phi_time[None, None]
            + phi_prime_t_max[:, :, None, ...]
            * (((shifted_x_time - L_time)[None, None] - t_max[..., None])[..., None, None])
        )
        projected_weights = jnp.einsum("nm,mdk->nkd", psi_x, weights) / jnp.sqrt(n_features) + weight_offset
        gamma_phi_gamma_x = jnp.einsum("nkd,nktdz,nkz->knt", projected_weights, core_tensor, projected_weights)
        # Per-player x metric curvature amplitude: scales how hard the curve bends below c_max.
        # curve_amp (n,k) -> (k,n,1) broadcasts over the (k,n,t) descent. a>0 preserves mu <= c_max.
        curve_amp = self._resolve_curve_amp()
        if curve_amp is not None:
            gamma_phi_gamma_x = jnp.transpose(curve_amp)[:, :, None] * gamma_phi_gamma_x
        mu_value = intercept + gamma_phi_gamma_x
        return mu_value  # no deterministic recording

    def _compute_curve_derivatives(self, psi_x, weights, t_max, phi_prime_t_max, L_time, M_time, shifted_x_time):
        """1st/2nd/3rd derivatives of the convex-max curve w.r.t. age, for model_export. Uses the
        SAME projection normalization as _compute_convex_mu: the 5-operand einsum with weights =
        beta*spd (no r**0.25) and an explicit /r equals (gamma/sqrt(n_features))^T [.] (gamma/...),
        i.e. /r == (1/sqrt(r))^2 — so derivatives are consistent with mu by construction. curve_amp
        (t-independent) factors through, so it multiplies all derivatives exactly as it does mu.
        Returns first_deriv (k,n,j), second_deriv (n,k), third_deriv (n,k)."""
        r = self._kernel_self_cov(psi_x)
        phi_double_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_double_prime(t, L_time, M_time))(t_max)
        phi_triple_prime_tmax = jax.vmap(lambda t: vmap_make_convex_phi_triple_prime(t, L_time, M_time))(t_max)
        phi_prime_t = vmap_make_convex_phi_prime(jnp.squeeze(shifted_x_time) - jnp.squeeze(L_time), jnp.squeeze(L_time), M_time)
        second_deriv = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> nk", psi_x, weights, phi_double_prime_tmax, weights, psi_x) / r
        third_deriv  = -1 * jnp.einsum("nm, mdk, nkdz, jzk, nj -> nk", psi_x, weights, phi_triple_prime_tmax, weights, psi_x) / r
        first_deriv  = jnp.einsum("nm, mdk, nktdz, jzk, nj -> knt", psi_x, weights,
                                  phi_prime_t_max[:, :, None, ...] - phi_prime_t[None, None], weights, psi_x) / r
        curve_amp = self._resolve_curve_amp()
        if curve_amp is not None:
            first_deriv  = jnp.transpose(curve_amp)[:, :, None] * first_deriv
            second_deriv = curve_amp * second_deriv
            third_deriv  = curve_amp * third_deriv
        return first_deriv, second_deriv, third_deriv

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            dist = Normal(linear_predictor[mask], expanded_sigmas[mask] / exposure[mask])
            return dist, obs[mask]
        if family == "poisson":
            log_rate = linear_predictor[mask] + exposure[mask]
            return Poisson(jnp.exp(log_rate)), obs[mask]
        if family == "negative-binomial":
            expanded_sigma_neg_bin = context["expanded_sigma_neg_bin"]
            log_rate = linear_predictor + jnp.where(mask, exposure, 0)
            return NegativeBinomial2(mean=jnp.exp(log_rate[mask]), concentration=expanded_sigma_neg_bin[mask]), obs[mask]
        if family == "binomial":
            return BinomialLogits(logits=linear_predictor[mask], total_count=exposure[mask].astype(int)), obs[mask]
        if family == "beta-binomial":
            expanded_sigma_beta_bin = context["expanded_sigma_beta_bin"]
            dist = BetaBinomial(
                concentration0=(1 - jsci.special.expit(linear_predictor[mask])) * expanded_sigma_beta_bin[mask],
                concentration1=jsci.special.expit(linear_predictor[mask]) * expanded_sigma_beta_bin[mask],
                total_count=exposure[mask].astype(int),
            )
            return dist, obs[mask]
        if family == "beta":
            expanded_sigma_beta = context["expanded_sigma_beta"]
            return BetaProportion(jsci.special.expit(linear_predictor[mask]), expanded_sigma_beta[mask] * jnp.square(exposure[mask])), obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexMaxTVLinearLVM likelihood builder")

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["sigma_beta"] = Uniform(0, 1)
        self.prior["sigma"] = InverseGamma(3, 2000)
        self.prior["sigma_negative_binomial"] = Exponential()
        self.prior["sigma_beta_binomial"] = Exponential()
        self.prior["lengthscale_deriv"] = LogNormal(jnp.log(3.0), 0.7)  # flexible prior (NO +3 floor); median 3, 95%~[0.7,12]: lets ls go short (peak-tracking) or long (smooth)
        # alpha base=1 (hard constant): avoids the alpha<->beta ridge / overshoot (learned alpha
        # inflated to ~10 and blew up holdout). sigma_c/sigma_t base=1 here but LEARNED via prior_knobs.
        self.prior["alpha"] = jnp.ones((self.k, 1))
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["X_free"] = Normal()
        self.prior["sigma_c"] = 1.0
        self.prior["sigma_t"] = 1.0
        self.prior["exit"] = Normal(0.0, 1.0)
        # sigma_exit_scale LEARNED across all models (was hard-fixed to 1): HalfNormal(0.5) per the
        # survival prior elicitation. Estimated at MAP and plugged in at MCMC via fixed_params, like the
        # other scale hyperparameters (sigma_c/sigma_t/sigma_X). Sets the per-player spread of the log
        # baseline hazard (exit_raw = make_psi_gamma(psi_x, exit)/sqrt(norm) * sigma_exit_scale).
        self.prior["sigma_exit_scale"] = HalfNormal(0.5)
        self.prior["eta_global_log"] = Normal(jnp.log(0.04), 0.5)
        self.prior["gamma_global_log"] = Normal(jnp.log(0.15), 0.3)
        self.prior["exit_rate"] = Normal(0.0, 0.1)
        # self.prior["t_offset"] = Uniform(-5, 5)
        # self.prior["c_offset"] = Normal(0, .1)
        # Player x metric level random effect on c_max (hierarchical, non-centered).
        # Active only when self.use_c_offset_re is set (base + AR models); see _resolve_c_offset.
        # Lets idiosyncratic peaks (e.g. Curry obpm) break from the shared-latent prediction
        # without re-introducing an age-varying AR.
        self.prior["sigma_c_offset"] = 1.0  # FIXED to 1: RE scale = sqrt(c_max_var); c_offset_re~N(0,1) regularizes per player
        self.prior["c_offset_re"]    = Normal()          # (n, k) unit-normal residual
        # Player x metric peak-AGE random effect (hierarchical, non-centered). Active only when
        # self.use_t_offset_re is set. Lets a metric peak at a different age than the player's shared
        # latent arc (e.g. dbpm peaks early ~24 for athletic bigs while obpm peaks late) — the shared
        # latent alone forces one peak age across all 17 metrics. RE scale = sqrt(t_max_var) (years).
        self.prior["sigma_t_offset"] = 1.0  # loose: peak-age RE scale = sqrt(t_max_var); lets metrics
        # (esp. dbpm) decouple their peak age from the shared latent arc (relocates apex to true age).
        # Player x metric CURVATURE amplitude random effect on the quadratic descent (active only when
        # self.use_curve_re is set). a[n,k] = exp(sigma_curve * curve_re) multiplies gamma_phi_gamma:
        # a>1 bends harder below c_max (sharp peak/crash), a<1 flatter. Multiplicative/log-normal keeps
        # a>0 so the convex-max guarantee mu <= c_max is preserved. Centered at 1 (curve_re=0 -> a=1).
        self.prior["sigma_curve"] = 0.5   # loose-ish log-amplitude scale; a in ~[0.37, 2.7] at +-2sd
        self.prior["curve_re"]    = Normal()   # (n, k) unit-normal residual on log curvature amplitude
        self.prior["t_offset_re"]    = Normal()          # (n, k) unit-normal residual on peak AGE (centered years)
        self.prior["rho_year_ar"]   = Uniform(0.9, 0.99)  # positive persistence only; allows near-unit-root for 40-yr era trends
        self.prior["sigma_year_ar"] = HalfNormal(.05)  # loosened: z~N(0,1) needs to give ~0.05/yr to track era trends (e.g. fg3a +1.6 log units over 43 yrs)
        self.prior["beta_year_ar"]  = Normal()
        self.prior["ar_0_year"]     = Normal()
        # Structured prior for X: X = Z @ W_proj + sigma_X * X_raw (non-centered)
        self.prior["sigma_W_proj"] = 1.0  # FIXED to 1: covariates are standardized, so W_proj ~ N(0,1) is the unit-scale projection (loses ARD shrink-to-0, but removes the scale ridge)
        self.prior["W_proj"]       = Normal()          # sampled as (2, r) with scale sigma_W_proj
        # FIXED latent scale (was HalfNormal(0.5)). Left free, the non-centered scale degeneracy
        # inflates it (~3.6): shrinking X_raw across all n*r elements saves more prior cost than the
        # sigma_X penalty, so the per-player X_raw penalty vanishes and data-poor players overfit to
        # the latent extreme (e.g. Tyler Davis, 0.9 min, |X|=max). Fixing it restores a real N(0,1)
        # penalty per player so sparse players pool toward their covariate mean x_loc. Reconstruction
        # paths already default sigma_X->1.0 when the site is absent.
        self.prior["sigma_X"]      = 1.0
        # Heavy-tailed latent prior: StudentT(nu) on the latent residual X lets genuine outlier
        # players (e.g. Curry obpm ~3 sigma) escape the population without inflating the bulk.
        # scale = sqrt((nu-2)/nu) keeps the marginal variance at 1 (same calibrated width as N(0,1)),
        # only the tails get fatter. This overrides the Normal prior["X"] from RFLVMBase and is the
        # site actually sampled on the structured (has_free=False) path via _resolve_prior("X").
        _nu_x = getattr(self, "x_latent_df", 4.0)
        self.prior["X"] = StudentT(_nu_x, jnp.zeros((self.n, self.r)), jnp.sqrt((_nu_x - 2.0) / _nu_x))

    def _resolve_latent_X_structured(self, x_loc, x_scale, sample_free_indices, sample_fixed_indices):
        """Non-centered parameterization: sample unit-normal residuals, then shift and scale."""
        has_free  = np.asarray(sample_free_indices).size > 0
        has_fixed = np.asarray(sample_fixed_indices).size > 0
        if has_free:
            n_free = len(sample_free_indices)
            # Heavy-tailed latent: StudentT(nu) lets genuine outlier players (e.g. Curry obpm ~3 sigma)
            # escape the population without inflating the bulk. Scale = sqrt((nu-2)/nu) makes the
            # marginal variance exactly 1 (same calibrated width as N(0,1)), only the tails get fatter.
            _nu = getattr(self, "x_latent_df", 4.0)
            _unit_scale = jnp.sqrt((_nu - 2.0) / _nu)
            X_free_raw = numpyro.sample(
                "X_free", StudentT(_nu, jnp.zeros((n_free, self.r)), _unit_scale)
            )
            X_free = x_loc[sample_free_indices] + x_scale * X_free_raw
            X = x_loc.at[sample_free_indices].set(X_free)
            if has_fixed:
                X = X.at[sample_fixed_indices].set(
                    self.prior["X"].at[sample_fixed_indices].get()
                )
            return X
        X_raw = self._resolve_prior("X")
        return x_loc + x_scale * X_raw

    def _resolve_c_offset(self, offsets):
        """Population c_max anchor, optionally plus a hierarchical player x metric level
        random effect. The random effect is active only when self.use_c_offset_re is True
        (set for the plain base and AR models in main.py); otherwise this reproduces the
        previous behaviour exactly. Scaled by sqrt(c_max_var) so the unit prior applies on
        each metric's natural level scale; non-centered via c_offset_re ~ N(0, 1)."""
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.n, self.k))
        if c_offset is None:
            c_offset = offsets["c_max"]
        if not getattr(self, "use_c_offset_re", False):
            return c_offset
        c_max_var = offsets.get("c_max_var", None)
        sigma_c_offset_unit = self._resolve_prior("sigma_c_offset", sample_shape=(self.k,))
        sigma_c_offset = (
            sigma_c_offset_unit * jnp.sqrt(jnp.asarray(c_max_var))
            if c_max_var is not None else sigma_c_offset_unit
        )
        c_offset_re = self._resolve_prior("c_offset_re", sample_shape=(self.n, self.k))
        return c_offset + sigma_c_offset[None, :] * c_offset_re

    def _resolve_t_offset(self, offsets):
        """Population peak-age anchor, optionally plus a hierarchical player x metric peak-age
        random effect (active only when self.use_t_offset_re is True). Scaled by sqrt(t_max_var)
        so the unit prior applies on each metric's natural peak-age spread (years); non-centered
        via t_offset_re ~ N(0, 1). Reproduces previous behaviour exactly when the RE is off."""
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        if not getattr(self, "use_t_offset_re", False):
            return t_offset
        t_max_var = offsets.get("t_max_var", None)
        sigma_t_offset_unit = self._resolve_prior("sigma_t_offset", sample_shape=(self.k,))
        sigma_t_offset = (
            sigma_t_offset_unit * jnp.sqrt(jnp.asarray(t_max_var))
            if t_max_var is not None else sigma_t_offset_unit
        )
        t_offset_re = self._resolve_prior("t_offset_re", sample_shape=(self.n, self.k))
        return t_offset + sigma_t_offset[None, :] * t_offset_re

    def _resolve_curve_amp(self):
        """Per-player x metric multiplicative curvature amplitude on the quadratic descent term,
        active only when self.use_curve_re is True. a[n,k] = exp(sigma_curve * curve_re[n,k]),
        curve_re ~ N(0,1), centered at 1. Lets a player bend harder/softer below c_max than the
        shared latent implies, without breaking convexity (a > 0 keeps gamma_phi_gamma <= 0).
        Returns None when off (callers then skip the multiply)."""
        if not getattr(self, "use_curve_re", False):
            return None
        sigma_curve = self._resolve_prior("sigma_curve")
        curve_re = self._resolve_prior("curve_re", sample_shape=(self.n, self.k))
        return jnp.exp(sigma_curve * curve_re)   # (n, k)

    def _build_curve_weights(self, effective_r, M_time, spd_time):
        """HSGP curve weights beta, shape (effective_r, M_time, k), scaled by the per-metric
        spectral density. Base: each metric's weights are i.i.d. N(0,1) (no cross-metric coupling).
        Subclasses (e.g. ConvexMaxLKJTVLinearLVM) override this to correlate across the metric axis
        while keeping unit-variance-normal marginals so the alpha/spectral calibration is unchanged."""
        weights = self._resolve_prior("beta", sample_shape=(effective_r, M_time, self.k))
        return weights * spd_time.T[None]

    def _compute_player_ar(self):
        """Per-player AR(1) contribution to the linear predictor, shape (k, n, j). Zero for non-AR
        models; AR subclasses override to sample and return the process. Used by prior_check's
        per-player draws so the plotted trajectory includes the AR — the likelihood adds it via the
        family linear predictor, so _compute_mu (the aging curve) alone omits it."""
        return jnp.zeros((self.k, self.n, self.j))

    def _survival_rates(self, X):
        """Per-player Gompertz hazard params from the survival latents: returns (eta, gamma), each
        (n,1). Hazard h(t) = eta * exp(gamma * t). Shared by compute_survival_likelihood and
        prior_check's survival predictive check (single source of truth for the survival forward)."""
        # Project X through the kernel feature map so the hazard uses the SAME representation as the
        # curve: cosine bounds it (||phi||^2 = r), RFF is the unit-norm SE map (||phi||^2 = 1), identity
        # for the linear model (psi_x = X). feat_dim sizes the exit weights to contract with psi_x;
        # the normalization divides by the kernel self-covariance (1 for RFF, r for linear/cosine) so
        # exit_raw/gamma_base have unit variance regardless of ||X|| and of the feature count.
        psi_x = self._project_X(X)
        feat_dim = self._projected_feature_dim()
        norm = self._kernel_self_cov(psi_x)
        exit = self._resolve_prior("exit", sample_shape=(feat_dim,))
        sigma_exit_scale = self._resolve_prior("sigma_exit_scale")
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(feat_dim,))
        exit_raw = make_psi_gamma(psi_x, exit) / jnp.sqrt(norm) * sigma_exit_scale  # (n,)
        eta_global_log = self._resolve_prior("eta_global_log")
        eta = jnp.exp(eta_global_log + exit_raw)[:, None]   # (n, 1) — baseline hazard
        gamma_base = make_psi_gamma(psi_x, exit_rate)[:, None] / jnp.sqrt(norm)  # (n, 1): kernel-self-cov scaled-dot-product, matching exit_raw/eta
        gamma_global_log = self._resolve_prior("gamma_global_log")
        gamma = jnp.exp(gamma_global_log + gamma_base)       # (n, 1) — aging rate
        return eta, gamma

    def compute_survival_likelihood(self, X, offsets = {}) -> None:
        required_keys = ("entrance_times", "exit_times", "right_censor")
        if not all(key in offsets for key in required_keys):
            return

        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_latent = jnp.maximum(entrance_times, 1e-6)

        eta, gamma = self._survival_rates(X)   # (n,1),(n,1) Gompertz hazard params

        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        interval_starts = jnp.arange(self.t, dtype=exit_times.dtype)[None, :]  # (1, t)
        interval_ends = interval_starts + 1.0
        entry = entrance_latent[:, None]   # (n, 1)
        stop  = exit_times[:, None]        # (n, 1)
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end   = jnp.minimum(interval_ends,   stop)
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.where(valid_seg, seg_start, 0.0)
        seg_end_safe   = jnp.where(valid_seg, seg_end,   0.0)
        ratio = eta / gamma  # (n, 1)
        valid_seg_float = valid_seg.astype(exit_times.dtype)
        delta_H = valid_seg_float * ratio * (
            jnp.exp(gamma * seg_end_safe) - jnp.exp(gamma * seg_start_safe)
        )
        cumulative_H = delta_H.sum(axis=-1)  # (n,)

        event_time = exit_times
        log_h_event = jnp.log(eta.squeeze(-1)) + gamma.squeeze(-1) * event_time

        log_lik_exit_event    = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)


    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        trend_ar = context.get("trend_ar", jnp.zeros_like(mu))
        return self._build_linear_predictor(mu, k_indices, trend_ar[k_indices])

    def compute_curves(self, hsgp_params, offsets={}, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0, include_derivs: bool = False):
        """SINGLE SOURCE OF TRUTH for the aging-curve forward. Resolves the curve latents ONCE and
        returns a dict: {mu (k,n,j), t_max (n,k), c_max (n,k), trend_ar (k,n,j), X (n,r)} plus, when
        include_derivs=True, first_deriv (k,n,j) / second_deriv (n,k) / third_deriv (n,k).

        The per-player AR(1) is NOT included here — callers add `self._compute_player_ar()` under the
        same substitute (model_fn, prior_check, model_export), so this method never touches the AR
        sites (avoids double-sampling when wrapped by _compute_mu). model_export reconstructs ALL
        exported curves through this method, so any change here propagates with no parallel edits."""
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_max = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))

        sigma_W_proj = self._resolve_prior("sigma_W_proj")
        _n_cov = self.player_covariates.shape[1] if self.player_covariates is not None else 2
        W_proj = self._resolve_prior("W_proj", sample_shape=(_n_cov, self.r),
                                     dist_override=Normal(0, sigma_W_proj))
        sigma_X = self._resolve_prior("sigma_X")
        Z = jnp.asarray(self.player_covariates) if self.player_covariates is not None \
            else jnp.zeros((self.n, 2))
        x_loc = Z @ W_proj
        X = self._resolve_latent_X_structured(x_loc, sigma_X, sample_free_indices, sample_fixed_indices)
        effective_r = self._projected_feature_dim()

        t_max_raw, c_max_raw = self._sample_max_raw_parameters(effective_r)

        psi_x = self._project_X(X)
        t_offset = self._resolve_t_offset(offsets)
        c_offset = self._resolve_c_offset(offsets)

        t_max, c_max = self._build_max_curves(
            psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max,
            t_offset, c_offset, False, amplitude=hsgp_params["t_amplitude"],
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._build_curve_weights(effective_r, M_time, spd_time)

        mu = self._compute_convex_mu(
            psi_x, weights, phi_t_max, phi_prime_t_max, phi_time,
            shifted_x_time, L_time, t_max, c_max, False,
        )
        # Calendar year AR(3) trend — only for metrics in ar_metric_indices
        if num_de_trend > 0:
            ar_global_indices = ar_metric_indices
            sigma_year_ar = self._resolve_prior("sigma_year_ar", sample_shape=(num_de_trend, 1))
            rho_year_ar   = self._resolve_prior("rho_year_ar",   sample_shape=(num_de_trend, 1))
            z_year        = self._resolve_prior("beta_year_ar",  sample_shape=(num_years, num_de_trend))
            ar_0_year_raw = self._resolve_prior("ar_0_year", sample_shape=(1, num_de_trend), site_name="AR_0_year")
            ar_0_year     = ar_0_year_raw * sigma_year_ar[None, :, 0]
            trend_ar_years = self._compute_ar1_calendar_process(sigma_year_ar, rho_year_ar, z_year, ar_0_year, ref_year_idx=ref_year_idx)
            # Gather: year_indices (n, j) → trend_ar_nj (num_ar, n, j)
            trend_ar_nj = trend_ar_years[:, year_indices]
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
            TREND_AR = TREND_AR.at[ar_global_indices].set(trend_ar_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))

        out = {"mu": mu, "t_max": t_max, "c_max": c_max, "trend_ar": TREND_AR, "X": X, "psi_x": psi_x}
        if include_derivs:
            fd, sd, td = self._compute_curve_derivatives(psi_x, weights, t_max, phi_prime_t_max, L_time, M_time, shifted_x_time)
            out["first_deriv"], out["second_deriv"], out["third_deriv"] = fd, sd, td
        return out

    def _compute_mu(self, hsgp_params, offsets={}, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0):
        """Thin wrapper over compute_curves preserving the (mu, TREND_AR, X) contract used by model_fn
        and prior_check. The full export set (peaks + derivatives) is available via compute_curves."""
        d = self.compute_curves(hsgp_params, offsets, sample_free_indices, sample_fixed_indices,
                                ar_metric_indices, year_indices, num_years, num_de_trend, ref_year_idx)
        return d["mu"], d["trend_ar"], d["X"]

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "map", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0) -> None:
        # Explicit prior-predictive flag (set by prior_check.py); when True the likelihood is
        # drawn (obs=None). Defaults False so MAP/MCMC condition on the observed data.
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0

        mu, TREND_AR, X = self._compute_mu(
            hsgp_params, offsets, sample_free_indices, sample_fixed_indices,
            ar_metric_indices, year_indices, num_years, num_de_trend, ref_year_idx,
        )

        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_taus = None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            trend_ar=TREND_AR,
        )
        self.compute_survival_likelihood(X, offsets=offsets)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
    def _debug_map_init_failure(self, model_args, rng_key):
        print("[MAP DEBUG] Inspecting sampled sites for invalid values")
        try:
            model_trace = trace(seed(self.model_fn, rng_key)).get_trace(**model_args)
        except Exception as error:
            print(f"[MAP DEBUG] Could not build model trace: {error}")
            return

        data_set = model_args.get("data_set", {})

        def _nonfinite_summary(x):
            try:
                x_arr = jnp.asarray(x)
            except Exception:
                return None, []
            bad_mask = ~jnp.isfinite(x_arr)
            bad_count = int(jnp.sum(bad_mask))
            if bad_count == 0:
                return 0, []
            bad_idx = np.asarray(jnp.argwhere(bad_mask))
            preview = [tuple(int(i) for i in row) for row in bad_idx[:10]]
            return bad_count, preview

        def _support_summary(dist, value):
            try:
                support_ok_mask = dist.support.check(value)
            except Exception:
                return None, []
            bad_mask = ~jnp.asarray(support_ok_mask)
            bad_count = int(jnp.sum(bad_mask))
            if bad_count == 0:
                return 0, []
            bad_idx = np.asarray(jnp.argwhere(bad_mask))
            preview = [tuple(int(i) for i in row) for row in bad_idx[:10]]
            return bad_count, preview

        def _fmt_value(x):
            if x is None:
                return "None"
            try:
                arr = np.asarray(x)
            except Exception:
                return str(x)
            if arr.size == 1:
                scalar = arr.item()
                if isinstance(scalar, (float, np.floating)):
                    return f"{float(scalar):.8g}"
                return str(scalar)
            return f"array(shape={arr.shape})"

        def _safe_take(x, idx):
            if x is None:
                return None
            try:
                arr = jnp.asarray(x)
            except Exception:
                return None

            if arr.ndim == 0:
                return arr

            try:
                return arr[idx]
            except Exception:
                pass

            if len(idx) == 1:
                try:
                    flat_idx = idx[0]
                    if 0 <= flat_idx < arr.size:
                        return jnp.ravel(arr)[flat_idx]
                except Exception:
                    return None
            return None

        def _map_likelihood_location(site_name, bad_index):
            if not site_name.startswith("likelihood_"):
                return None
            if len(bad_index) != 1:
                return None
            family = site_name.replace("likelihood_", "")
            family_data = data_set.get(family)
            if family_data is None or "mask" not in family_data:
                return None
            try:
                masked_coords = np.asarray(jnp.argwhere(family_data["mask"]))
                flat_idx = bad_index[0]
                if 0 <= flat_idx < len(masked_coords):
                    return tuple(int(i) for i in masked_coords[flat_idx])
            except Exception:
                return None
            return None

        def _family_snapshot(family, coord):
            family_data = data_set.get(family)
            if family_data is None:
                return None

            out = {}
            for key in ("Y", "exposure", "de_trend", "mask"):
                if key in family_data:
                    out[key] = _safe_take(family_data[key], coord)
            return out

        def _distribution_snapshot(dist, bad_index):
            param_names = (
                "rate",
                "logits",
                "total_count",
                "concentration",
                "concentration0",
                "concentration1",
                "loc",
                "scale",
            )
            params = {}
            for name in param_names:
                try:
                    value = getattr(dist, name)
                except Exception:
                    continue
                params[name] = _safe_take(value, bad_index)
            return params

        def _audit_family_masking():
            if not data_set:
                return
            print("[MAP DEBUG] Mask audit by family (masked entries only):")
            for family, family_data in data_set.items():
                if not all(k in family_data for k in ("Y", "exposure", "mask")):
                    continue
                mask = jnp.asarray(family_data["mask"]).astype(bool)
                y = jnp.asarray(family_data["Y"])
                exposure = jnp.asarray(family_data["exposure"])

                masked_count = int(jnp.sum(mask))
                y_masked = y[mask]
                exp_masked = exposure[mask]

                y_nonfinite = int(jnp.sum(~jnp.isfinite(y_masked)))
                exp_nonfinite = int(jnp.sum(~jnp.isfinite(exp_masked)))
                exp_le_zero = int(jnp.sum(exp_masked <= 0))

                support_issue_count = 0
                if family == "beta":
                    support_issue_count = int(jnp.sum((y_masked <= 0) | (y_masked >= 1)))
                elif family in ("poisson", "negative-binomial"):
                    support_issue_count = int(jnp.sum((y_masked < 0) | (jnp.floor(y_masked) != y_masked)))
                elif family in ("binomial", "beta-binomial"):
                    total = exp_masked.astype(int)
                    support_issue_count = int(jnp.sum((y_masked < 0) | (jnp.floor(y_masked) != y_masked) | (y_masked > total)))

                print(
                    f"  - {family}: masked_n={masked_count}, "
                    f"Y_nonfinite={y_nonfinite}, exposure_nonfinite={exp_nonfinite}, "
                    f"exposure<=0={exp_le_zero}, support_issues={support_issue_count}"
                )

        def _audit_gradients():
            print("[MAP DEBUG] Gradient audit at sampled latent parameters:")
            latent_params = {
                name: site["value"]
                for name, site in model_trace.items()
                if site.get("type") == "sample" and not site.get("is_observed", False)
            }

            if not latent_params:
                print("  - No latent sample sites found for gradient audit")
                return

            try:
                log_joint, _ = log_density(self.model_fn, (), model_args, latent_params)
                log_joint_finite = bool(jnp.isfinite(log_joint))
                print(f"  - log_joint finite={log_joint_finite}, value={float(log_joint):.6g}")
            except Exception as error:
                print(f"  - Could not evaluate log_density: {error}")
                return

            try:
                grad_fn = jax.grad(lambda p: -log_density(self.model_fn, (), model_args, p)[0])
                grads = grad_fn(latent_params)
            except Exception as error:
                print(f"  - Could not evaluate gradients: {error}")
                return

            for name, grad_value in grads.items():
                grad_arr = jnp.asarray(grad_value)
                nonfinite = int(jnp.sum(~jnp.isfinite(grad_arr)))
                total = int(grad_arr.size)
                grad_abs = jnp.abs(jnp.nan_to_num(grad_arr, nan=0.0, posinf=0.0, neginf=0.0))
                max_abs = float(jnp.max(grad_abs)) if total > 0 else 0.0
                mean_abs = float(jnp.mean(grad_abs)) if total > 0 else 0.0
                near_zero = int(jnp.sum(grad_abs < 1e-12)) if total > 0 else 0
                print(
                    f"  - {name}: shape={grad_arr.shape}, nonfinite={nonfinite}/{total}, "
                    f"max|g|={max_abs:.6g}, mean|g|={mean_abs:.6g}, near_zero(<1e-12)={near_zero}/{total}"
                )

            try:
                offsets = model_args.get("offsets", {})
                surv_keys = ("entrance_times", "exit_times", "left_censor", "right_censor")
                if isinstance(offsets, dict) and any(k in offsets for k in surv_keys):
                    model_args_no_surv = dict(model_args)
                    offsets_no_surv = dict(offsets)
                    for key in surv_keys:
                        offsets_no_surv.pop(key, None)
                    model_args_no_surv["offsets"] = offsets_no_surv

                    grad_fn_no_surv = jax.grad(lambda p: -log_density(self.model_fn, (), model_args_no_surv, p)[0])
                    grads_no_surv = grad_fn_no_surv(latent_params)

                    print("[MAP DEBUG] Gradient audit without survival offsets:")
                    for name, grad_value in grads_no_surv.items():
                        grad_arr = jnp.asarray(grad_value)
                        nonfinite = int(jnp.sum(~jnp.isfinite(grad_arr)))
                        total = int(grad_arr.size)
                        print(f"  - {name}: nonfinite={nonfinite}/{total}")
            except Exception as error:
                print(f"  - Could not evaluate no-survival gradient audit: {error}")

        def _audit_survival_likelihood_nans():
            offsets = model_args.get("offsets", {})
            required_keys = ("entrance_times", "exit_times", "left_censor", "right_censor")
            if not isinstance(offsets, dict) or not all(k in offsets for k in required_keys):
                return

            sampled = {
                name: site["value"]
                for name, site in model_trace.items()
                if site.get("type") == "sample" and not site.get("is_observed", False)
            }

            required_sites = ("X", "entrance", "sigma_entrance", "entrance_global_offset", "entrance_latent", "exit", "exit_rate")
            missing = [name for name in required_sites if name not in sampled]
            if missing:
                print(f"[MAP DEBUG] Survival NaN audit skipped; missing sites: {missing}")
                return

            try:
                X = sampled["X"]
                entrance = sampled["entrance"]
                sigma_entrance = sampled["sigma_entrance"]
                entrance_global_offset = sampled["entrance_global_offset"]
                entrance_latent_sampled = sampled["entrance_latent"]
                exit_param = sampled["exit"]
                exit_rate = sampled["exit_rate"]

                lc = jnp.ravel(jnp.asarray(offsets["left_censor"]).astype(bool))
                rc = jnp.ravel(jnp.asarray(offsets["right_censor"]).astype(bool))
                entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
                exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

                observed_mask = (~lc).astype(entrance_times.dtype)
                observed_count = jnp.maximum(observed_mask.sum(), 1.0)
                empirical_log_mean = jnp.sum(jnp.log(entrance_times) * observed_mask) / observed_count

                entrance_raw = make_psi_gamma(X, entrance)
                entrance_loc = entrance_global_offset + entrance_raw + empirical_log_mean
                sigma_effective = sigma_entrance + 0.35
                entrance_dist = LogNormal(entrance_loc, sigma_effective)
                z_entrance = (jnp.log(entrance_times) - entrance_loc) / sigma_effective
                log_cdf_entrance = jsci.stats.norm.logcdf(z_entrance)
                log_pdf_entrance = entrance_dist.log_prob(entrance_times)

                lc_float = lc.astype(entrance_times.dtype)
                entrance_latent = lc_float * entrance_latent_sampled + (1.0 - lc_float) * entrance_times

                psi_x = self._project_X(X)   # kernel feature map (cosine bounds the hazard; identity for linear)
                exit_raw = make_psi_gamma(psi_x, exit_param) / jnp.sqrt(X.shape[-1]) * sampled.get("sigma_exit_scale", 1.0)
                gamma_base = make_psi_gamma(psi_x, exit_rate)[:, None] / jnp.sqrt(X.shape[-1])
                gamma_global_log = sampled.get("gamma_global_log", jnp.log(0.15))
                gamma = jnp.exp(gamma_global_log + gamma_base)  # (n, 1)
                eta_global_log = sampled.get("eta_global_log", jnp.log(0.04))
                eta = jnp.exp(eta_global_log + exit_raw)[:, None]  # (n, 1)

                event_time = exit_times
                gamma_event = gamma.squeeze(-1)
                eta_event = eta.squeeze(-1)
                entry_effective = jnp.maximum(entrance_latent, 0.0)
                stop_effective = jnp.maximum(event_time, 0.0)
                has_exposure_window = stop_effective > entry_effective
                ratio = eta_event / gamma_event
                cumulative_H = jnp.where(
                    has_exposure_window,
                    ratio * (jnp.exp(gamma_event * stop_effective) - jnp.exp(gamma_event * entry_effective)),
                    0.0,
                )
                log_h_event = jnp.log(eta_event) + gamma_event * event_time

                log_lik_exit_event = log_h_event - cumulative_H
                log_lik_exit_censored = -cumulative_H

                def _print_nonfinite(name, arr, mask=None):
                    arr = jnp.asarray(arr)
                    bad = ~jnp.isfinite(arr)
                    if mask is not None:
                        bad = bad & jnp.asarray(mask)
                    count = int(jnp.sum(bad))
                    total = int(arr.size) if mask is None else int(jnp.sum(jnp.asarray(mask)))
                    print(f"[MAP DEBUG][SURV] {name}: nonfinite={count}/{total}")
                    if count == 0:
                        return []
                    idx = np.asarray(jnp.argwhere(bad)).reshape(-1)
                    preview = [int(i) for i in idx[:10]]
                    print(f"[MAP DEBUG][SURV] {name} bad idx (first 10): {preview}")
                    return preview

                print("[MAP DEBUG] Survival-likelihood NaN audit:")
                _print_nonfinite("entrance_times", entrance_times)
                _print_nonfinite("entrance_loc", entrance_loc)
                _print_nonfinite("z_entrance", z_entrance)
                _print_nonfinite("log_lik_entrance_censored", log_cdf_entrance, mask=lc)
                entrance_bad = _print_nonfinite("log_lik_entrance_observed", log_pdf_entrance, mask=(~lc))

                _print_nonfinite("event_time", event_time)
                gamma_bad = _print_nonfinite("gamma", jnp.ravel(gamma))
                _print_nonfinite("eta", eta_event)
                _print_nonfinite("log_h_event", log_h_event, mask=(~rc))
                _print_nonfinite("cumulative_H", cumulative_H)
                _print_nonfinite("log_lik_exit_censored", log_lik_exit_censored, mask=rc)
                exit_bad = _print_nonfinite("log_lik_exit_observed", log_lik_exit_event, mask=(~rc))

                print(
                    "[MAP DEBUG][SURV] quick ranges: "
                    f"entrance_loc=[{float(jnp.nanmin(entrance_loc)):.6g}, {float(jnp.nanmax(entrance_loc)):.6g}], "
                    f"sigma_eff={float(jnp.asarray(sigma_effective)):.6g}, "
                    f"event_time=[{float(jnp.nanmin(event_time)):.6g}, {float(jnp.nanmax(event_time)):.6g}], "
                    f"gamma=[{float(jnp.nanmin(gamma_event)):.6g}, {float(jnp.nanmax(gamma_event)):.6g}], "
                    f"eta=[{float(jnp.nanmin(eta_event)):.6g}, {float(jnp.nanmax(eta_event)):.6g}]"
                )

                if gamma_bad:
                    print("[MAP DEBUG][SURV] gamma bad-point snapshots:")
                    flat = jnp.ravel(gamma)
                    for i in gamma_bad[:5]:
                        print(
                            f"  idx={i}, gamma={_fmt_value(flat[i])}, "
                            f"x_norm={_fmt_value(jnp.linalg.norm(X[i]))}"
                        )

                if entrance_bad:
                    print("[MAP DEBUG][SURV] entrance observed bad-point snapshots:")
                    for i in entrance_bad[:5]:
                        print(
                            f"  idx={i}, lc={bool(lc[i])}, entrance_time={_fmt_value(entrance_times[i])}, "
                            f"entrance_time={_fmt_value(entrance_times[i])}, "
                            f"entrance_loc={_fmt_value(entrance_loc[i])}, z_entrance={_fmt_value(z_entrance[i])}, "
                            f"log_pdf={_fmt_value(log_pdf_entrance[i])}, log_cdf={_fmt_value(log_cdf_entrance[i])}"
                        )

                if exit_bad:
                    print("[MAP DEBUG][SURV] exit observed bad-point snapshots:")
                    for i in exit_bad[:5]:
                        entry_i = entrance_latent[i]
                        print(
                            f"  idx={i}, rc={bool(rc[i])}, event_time={_fmt_value(event_time[i])}, "
                            f"entry={_fmt_value(entry_i)}, "
                            f"gamma={_fmt_value(gamma_event[i])}, eta={_fmt_value(eta_event[i])}, "
                            f"log_h={_fmt_value(log_h_event[i])}, cum_H={_fmt_value(cumulative_H[i])}, "
                            f"log_lik_exit={_fmt_value(log_lik_exit_event[i])}, "
                            f"entry_eff={_fmt_value(entry_effective[i])}, stop_eff={_fmt_value(stop_effective[i])}, "
                            f"has_window={bool(has_exposure_window[i])}"
                        )
            except Exception as error:
                print(f"[MAP DEBUG] Survival NaN audit failed: {error}")

        _audit_family_masking()
        _audit_gradients()
        _audit_survival_likelihood_nans()

        latent_issues = []
        observed_issues = []
        for name, site in model_trace.items():
            if site.get("type") != "sample":
                continue

            value = site["value"]
            dist = site["fn"]
            value_bad_count, value_bad_idx = _nonfinite_summary(value)

            support_bad_count, support_bad_idx = _support_summary(dist, value)

            log_prob_bad_count = None
            log_prob_bad_idx = []
            log_prob = None
            try:
                log_prob = dist.log_prob(value)
                log_prob_bad_count, log_prob_bad_idx = _nonfinite_summary(log_prob)
            except Exception:
                log_prob_bad_count = None

            has_issue = any(
                count not in (None, 0)
                for count in (value_bad_count, support_bad_count, log_prob_bad_count)
            )
            if not has_issue:
                continue

            issue = {
                "name": name,
                "shape": getattr(value, "shape", None),
                "value_bad_count": value_bad_count,
                "value_bad_idx": value_bad_idx,
                "support_bad_count": support_bad_count,
                "support_bad_idx": support_bad_idx,
                "log_prob_bad_count": log_prob_bad_count,
                "log_prob_bad_idx": log_prob_bad_idx,
                "value": value,
                "dist": dist,
                "log_prob": log_prob,
            }

            if site.get("is_observed", False):
                observed_issues.append(issue)
            else:
                latent_issues.append(issue)

        if not latent_issues and not observed_issues:
            print("[MAP DEBUG] No invalid values in direct model sampling trace")
        

        if latent_issues:
            print("[MAP DEBUG] Problematic latent sample sites:")
            for issue in latent_issues:
                print(
                    f"  - {issue['name']}: shape={issue['shape']}, "
                    f"value_nonfinite={issue['value_bad_count']}, "
                    f"support_violations={issue['support_bad_count']}, "
                    f"log_prob_nonfinite={issue['log_prob_bad_count']}"
                )
                if issue["value_bad_idx"]:
                    print(f"      value bad idx (first 10): {issue['value_bad_idx']}")
                if issue["support_bad_idx"]:
                    print(f"      support bad idx (first 10): {issue['support_bad_idx']}")
                if issue["log_prob_bad_idx"]:
                    print(f"      log_prob bad idx (first 10): {issue['log_prob_bad_idx']}")

        if observed_issues:
            print("[MAP DEBUG] Problematic observed/factor sites:")
            for issue in observed_issues:
                print(
                    f"  - {issue['name']}: shape={issue['shape']}, "
                    f"value_nonfinite={issue['value_bad_count']}, "
                    f"support_violations={issue['support_bad_count']}, "
                    f"log_prob_nonfinite={issue['log_prob_bad_count']}"
                )
                if issue["value_bad_idx"]:
                    print(f"      value bad idx (first 10): {issue['value_bad_idx']}")
                if issue["support_bad_idx"]:
                    print(f"      support bad idx (first 10): {issue['support_bad_idx']}")
                if issue["log_prob_bad_idx"]:
                    print(f"      log_prob bad idx (first 10): {issue['log_prob_bad_idx']}")
                    mapped = [
                        _map_likelihood_location(issue["name"], idx)
                        for idx in issue["log_prob_bad_idx"]
                    ]
                    mapped = [coord for coord in mapped if coord is not None]
                    if mapped:
                        print(f"      mapped data indices (first 10): {mapped}")

                    print("      numeric snapshots (first 5 bad points):")
                    family = issue["name"].replace("likelihood_", "")
                    for bad_idx in issue["log_prob_bad_idx"][:5]:
                        mapped_coord = _map_likelihood_location(issue["name"], bad_idx)
                        obs_value = _safe_take(issue["value"], bad_idx)
                        log_prob_value = _safe_take(issue["log_prob"], bad_idx)
                        print(
                            f"        idx={bad_idx}, mapped={mapped_coord}, "
                            f"obs={_fmt_value(obs_value)}, log_prob={_fmt_value(log_prob_value)}"
                        )

                        dist_params = _distribution_snapshot(issue["dist"], bad_idx)
                        if dist_params:
                            param_text = ", ".join(
                                f"{k}={_fmt_value(v)}" for k, v in dist_params.items()
                            )
                            print(f"          dist params: {param_text}")

                        if mapped_coord is not None:
                            fam = _family_snapshot(family, mapped_coord)
                            if fam is not None:
                                fam_text = ", ".join(
                                    f"{k}={_fmt_value(v)}" for k, v in fam.items()
                                )
                                print(f"          data_set snapshot: {fam_text}")

        for survival_name in ("log_lik_entrance", "log_lik_exit"):
            if survival_name in model_trace:
                site = model_trace[survival_name]
                value = site.get("value", None)
                if value is None:
                    print(f"[MAP DEBUG] {survival_name}: present, value=None")
                    continue
                finite = bool(jnp.all(jnp.isfinite(value)))
                try:
                    value_sum = float(jnp.sum(value))
                except Exception:
                    value_sum = float("nan")
                print(f"[MAP DEBUG] {survival_name}: finite={finite}, sum={value_sum:.6g}")

        if type(self) is ConvexMaxTVLinearLVM:
            print("[MAP DEBUG] Prior influence diagnostics (ConvexMaxTVLinearLVM):")
            try:
                sampled = {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site.get("type") == "sample" and not site.get("is_observed", False)
                }

                required_sites = ["alpha", "lengthscale_deriv", "sigma_c", "sigma_t", "t_max_raw", "c_max", "beta", "X"]
                missing = [name for name in required_sites if name not in sampled]
                if missing:
                    print(f"  [MAP DEBUG] Skipping prior influence report; missing sites: {missing}")
                    return

                hsgp_params = model_args.get("hsgp_params", {})
                offsets = model_args.get("offsets", {})
                data_set = model_args.get("data_set", {})

                phi_time = hsgp_params["phi_x_time"]
                L_time = hsgp_params["L_time"]
                M_time = hsgp_params["M_time"]
                shifted_x_time = hsgp_params["shifted_x_time"]

                alpha_time = sampled["alpha"]
                ls_deriv = 3 +  sampled["lengthscale_deriv"]
                spd_time = jnp.squeeze(
                    jnp.sqrt(
                        jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)
                    )
                )

                _sigma_c_unit = sampled["sigma_c"]
                _dbg_c_max_var = model_args.get("offsets", {}).get("c_max_var", None)
                sigma_c_max = _sigma_c_unit * jnp.sqrt(jnp.asarray(_dbg_c_max_var)) if _dbg_c_max_var is not None else _sigma_c_unit
                sigma_t_max = sampled["sigma_t"]
                t_max_raw = sampled["t_max_raw"]
                c_max_raw = sampled["c_max"]
                X = sampled["X"]
                obs_cov = model_args.get("observed_covariates")
                if obs_cov is not None:
                    X = jnp.concatenate([X, jnp.asarray(obs_cov)], axis=-1)

                psi_x = self._project_X(X)

                t_offset = sampled.get("t_offset",offsets["t_max"])
                c_offset = sampled.get("c_offset", offsets["c_max"])


                t_max, c_max = self._build_max_curves(
                    psi_x,
                    t_max_raw,
                    c_max_raw,
                    sigma_t_max,
                    sigma_c_max,
                    t_offset,
                    c_offset,
                    False,
                    amplitude=hsgp_params["t_amplitude"],
                )

                phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)
                weights_raw = sampled["beta"]
                weights_scaled = weights_raw * spd_time.T[None]

                mu = self._compute_convex_mu(
                    psi_x,
                    weights_scaled,
                    phi_t_max,
                    phi_prime_t_max,
                    phi_time,
                    shifted_x_time,
                    L_time,
                    t_max,
                    c_max,
                    False,
                )

                def _qstats(x):
                    x = jnp.asarray(x)
                    return {
                        "min": float(jnp.nanmin(x)),
                        "q01": float(jnp.nanpercentile(x, 1)),
                        "q50": float(jnp.nanpercentile(x, 50)),
                        "q99": float(jnp.nanpercentile(x, 99)),
                        "max": float(jnp.nanmax(x)),
                    }

                def _stats_text(name, stats):
                    print(
                        f"  - {name}: min={stats['min']:.6g}, q01={stats['q01']:.6g}, "
                        f"median={stats['q50']:.6g}, q99={stats['q99']:.6g}, max={stats['max']:.6g}"
                    )

                def _k_range_text(name, x, k_axis=0, k_indices=None):
                    x = jnp.asarray(x)
                    if x.ndim == 0:
                        return
                    if k_axis < 0:
                        k_axis = x.ndim + k_axis
                    if k_axis < 0 or k_axis >= x.ndim:
                        return

                    moved = jnp.moveaxis(x, k_axis, 0)
                    mins = jnp.nanmin(moved, axis=tuple(range(1, moved.ndim)))
                    maxs = jnp.nanmax(moved, axis=tuple(range(1, moved.ndim)))

                    k_size = int(mins.shape[0])
                    print(f"  - {name} per-k ranges:")
                    for i in range(k_size):
                        k_label = int(k_indices[i]) if k_indices is not None else i
                        print(
                            f"      k={k_label}: min={float(mins[i]):.6g}, max={float(maxs[i]):.6g}"
                        )

                _stats_text("alpha", _qstats(alpha_time))
                _k_range_text("alpha", alpha_time, k_axis=0)
                _stats_text("lengthscale_deriv(+3)", _qstats(ls_deriv))
                _k_range_text("lengthscale_deriv(+3)", ls_deriv, k_axis=0)
                _stats_text("spd_time", _qstats(spd_time))
                _k_range_text("spd_time", spd_time, k_axis=0)
                _stats_text("beta_raw", _qstats(weights_raw))
                _k_range_text("beta_raw", weights_raw, k_axis=2)
                _stats_text("beta_scaled", _qstats(weights_scaled))
                _k_range_text("beta_scaled", weights_scaled, k_axis=2)
                _stats_text("sigma_t", _qstats(sigma_t_max))
                _k_range_text("sigma_t", sigma_t_max, k_axis=0)
                _stats_text("sigma_c", _qstats(sigma_c_max))
                _k_range_text("sigma_c", sigma_c_max, k_axis=0)
                _stats_text("t_max", _qstats(t_max))
                _k_range_text("t_max", t_max, k_axis=1)
                _stats_text("c_max", _qstats(c_max))
                _k_range_text("c_max", c_max, k_axis=1)
                _stats_text("mu", _qstats(mu))
                _k_range_text("mu", mu, k_axis=0)

                for family, family_data in data_set.items():
                    if "indices" not in family_data:
                        continue
                    k_indices = family_data["indices"]
                    linear_predictor = mu[k_indices]
                    mask = family_data["mask"]
                    exposure = family_data["exposure"]

                    _stats_text(f"{family}.linear_predictor(masked)", _qstats(linear_predictor[mask]))
                    _k_range_text(
                        f"{family}.linear_predictor(masked)",
                        linear_predictor,
                        k_axis=0,
                        k_indices=np.asarray(k_indices),
                    )

                    if family in ("poisson", "negative-binomial"):
                        log_rate = linear_predictor + jnp.where(mask, exposure, 0)
                        _stats_text(f"{family}.log_rate(masked)", _qstats(log_rate[mask]))
                        _k_range_text(
                            f"{family}.log_rate(masked)",
                            log_rate,
                            k_axis=0,
                            k_indices=np.asarray(k_indices),
                        )
                        underflow_count = int(jnp.sum(log_rate[mask] < -745.0))
                        print(f"  - {family}.log_rate_underflow_count(<-745): {underflow_count}")
                    elif family in ("beta", "beta-binomial", "binomial"):
                        logit_rate = linear_predictor
                        _stats_text(f"{family}.logit_rate(masked)", _qstats(logit_rate[mask]))
                        _k_range_text(
                            f"{family}.logit_rate(masked)",
                            logit_rate,
                            k_axis=0,
                            k_indices=np.asarray(k_indices),
                        )
                        p = jax.nn.sigmoid(logit_rate[mask])
                        near_zero = int(jnp.sum(p < 1e-12))
                        print(f"  - {family}.sigmoid_near_zero_count(<1e-12): {near_zero}")
            except Exception as error:
                print(f"  [MAP DEBUG] Prior influence diagnostics failed: {error}")

    def _debug_map_nonfinite_run(self, guide, params, model_args, rng_key=jax.random.PRNGKey(0)):
        print("[MAP DEBUG] Non-finite MAP loss detected; auditing final guide state")

        try:
            latent_params = guide.median(params)
        except Exception as error:
            print(f"[MAP DEBUG] Could not extract constrained guide median params: {error}")
            return

        def _nonfinite_summary(x):
            try:
                x_arr = jnp.asarray(x)
            except Exception:
                return None, []
            bad_mask = ~jnp.isfinite(x_arr)
            bad_count = int(jnp.sum(bad_mask))
            if bad_count == 0:
                return 0, []
            bad_idx = np.asarray(jnp.argwhere(bad_mask))
            preview = [tuple(int(i) for i in row) for row in bad_idx[:10]]
            return bad_count, preview

        def _safe_take(x, idx):
            if x is None:
                return None
            try:
                arr = jnp.asarray(x)
            except Exception:
                return None

            if arr.ndim == 0:
                return arr

            try:
                return arr[idx]
            except Exception:
                pass

            if len(idx) == 1:
                try:
                    flat_idx = idx[0]
                    if 0 <= flat_idx < arr.size:
                        return jnp.ravel(arr)[flat_idx]
                except Exception:
                    return None
            return None

        def _fmt_value(x):
            if x is None:
                return "None"
            try:
                arr = np.asarray(x)
            except Exception:
                return str(x)
            if arr.size == 1:
                scalar = arr.item()
                if isinstance(scalar, (float, np.floating)):
                    return f"{float(scalar):.8g}"
                return str(scalar)
            return f"array(shape={arr.shape})"

        print("[MAP DEBUG] Gradient audit at final constrained latent values:")
        try:
            log_joint, _ = log_density(self.model_fn, (), model_args, latent_params)
            log_joint_finite = bool(jnp.isfinite(log_joint))
            print(f"  - final log_joint finite={log_joint_finite}, value={float(log_joint):.6g}")
        except Exception as error:
            print(f"  - Could not evaluate final log_density: {error}")
            log_joint = None

        try:
            grad_fn = jax.grad(lambda p: -log_density(self.model_fn, (), model_args, p)[0])
            grads = grad_fn(latent_params)
        except Exception as error:
            print(f"  - Could not evaluate gradients at final state: {error}")
            grads = None

        if grads is not None:
            any_bad = False
            for name, grad_value in grads.items():
                grad_arr = jnp.asarray(grad_value)
                grad_bad_count, grad_bad_idx = _nonfinite_summary(grad_arr)
                total = int(grad_arr.size)
                if grad_bad_count and grad_bad_count > 0:
                    any_bad = True
                    value_arr = latent_params.get(name)
                    print(f"  - {name}: nonfinite_grad={grad_bad_count}/{total}")
                    print(f"      grad bad idx (first 10): {grad_bad_idx}")
                    if value_arr is not None:
                        for bad_idx in grad_bad_idx[:5]:
                            grad_val = _safe_take(grad_arr, bad_idx)
                            param_val = _safe_take(value_arr, bad_idx)
                            print(
                                f"      idx={bad_idx}, latent={_fmt_value(param_val)}, grad={_fmt_value(grad_val)}"
                            )
            if not any_bad:
                print("  - All final latent gradients are finite (NaN likely occurred earlier in optimization)")

        print("[MAP DEBUG] Final-trace log_prob audit at constrained latent values:")
        try:
            model_trace = trace(substitute(seed(self.model_fn, rng_key), data=latent_params)).get_trace(**model_args)
        except Exception as error:
            print(f"  - Could not build substituted model trace: {error}")
            return

        found_problem = False
        for name, site in model_trace.items():
            if site.get("type") != "sample":
                continue

            value = site.get("value", None)
            dist = site.get("fn", None)
            if value is None or dist is None:
                continue

            try:
                log_prob = dist.log_prob(value)
                bad_count, bad_idx = _nonfinite_summary(log_prob)
            except Exception as error:
                found_problem = True
                print(f"  - {name}: log_prob evaluation failed: {error}")
                continue

            if bad_count and bad_count > 0:
                found_problem = True
                observed_flag = bool(site.get("is_observed", False))
                print(
                    f"  - {name} (observed={observed_flag}): nonfinite_log_prob={bad_count}, "
                    f"bad idx (first 10)={bad_idx}"
                )
                param_names = (
                    "rate",
                    "logits",
                    "total_count",
                    "concentration",
                    "concentration0",
                    "concentration1",
                    "loc",
                    "scale",
                )
                for bad in bad_idx[:5]:
                    value_bad = _safe_take(value, bad)
                    print(f"      idx={bad}, site_value={_fmt_value(value_bad)}")
                    snapshots = []
                    for param_name in param_names:
                        try:
                            param_val = getattr(dist, param_name)
                        except Exception:
                            continue
                        snapshots.append(f"{param_name}={_fmt_value(_safe_take(param_val, bad))}")
                    if snapshots:
                        print(f"      dist params: {', '.join(snapshots)}")

        if not found_problem:
            print("  - No non-finite site log_prob values in final substituted trace")

        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        guide = AutoDelta(self.model_fn, prefix="", init_loc_fn=init_to_median, **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state

        init_key = jax.random.PRNGKey(0)
        try:
            state = svi.init(init_key, **model_args)
        except Exception as error:
            print(f"[MAP DEBUG] svi.init failed: {error}")
            self._debug_map_init_failure(model_args, init_key)
            raise

        init_params = None
        init_state_bytes = None
        if isinstance(initial_state, dict) and (("state" in initial_state) or ("samples" in initial_state)):
            if initial_state.get("state") is not None:
                init_state_bytes = initial_state["state"]
            else:
                init_params = initial_state.get("samples")
        elif isinstance(initial_state, (bytes, bytearray)):
            init_state_bytes = initial_state
        elif initial_state is not None:
            init_params = initial_state

        if init_state_bytes is not None:
            state = ser.from_bytes(state, init_state_bytes)
            result = svi.run(
                jax.random.PRNGKey(0),
                num_steps=num_steps,
                progress_bar=True,
                init_state=state,
                stable_update=True,
                **model_args,
            )
        else:
            if isinstance(init_params, dict) and (len(init_params) > 0) and all((isinstance(key, str) and (not key.endswith("__loc"))) for key in init_params.keys()):
                init_params = {f"{key}__loc": value for key, value in init_params.items()}
            result = svi.run(
                jax.random.PRNGKey(0),
                num_steps=num_steps,
                progress_bar=True,
                init_state=state if init_params is None else None,
                init_params=init_params,
                stable_update=True,
                **model_args,
            )
        try:
            losses = np.asarray(result.losses)
            nonfinite_steps = np.where(~np.isfinite(losses))[0]
            if len(nonfinite_steps) > 0:
                first_bad = int(nonfinite_steps[0])
                print(
                    f"[MAP DEBUG] Non-finite MAP loss at step {first_bad + 1}/{len(losses)}; "
                    f"loss={losses[first_bad]}"
                )
                self._debug_map_nonfinite_run(guide=guide, params=result.params, model_args=model_args)
        except Exception as error:
            print(f"[MAP DEBUG] Could not evaluate MAP loss diagnostics: {error}")
        _print_svi_loss_breakdown(
            model_fn=self.model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.PRNGKey(1),
            debug_label=f"{type(self).__name__} MAP",
        )
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        effective_init_state = None if (initial_state is None or (isinstance(initial_state, dict) and len(initial_state) == 0)) else initial_state
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
            init_state=effective_init_state,
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexMaxTVCosineLVM(ConvexMaxTVLinearLVM):
    """Cosine (normalized dot-product) kernel variant of ConvexMaxTVLinearLVM. Identical in every
    respect except the latent feature map is the direction of X scaled to norm sqrt(r):
        phi(x) = sqrt(r) * X / ||X||.
    The concave-process kernel phi(x)^T phi(x') is then r * cosine(x, x'), whose self-covariance
    phi(x)^T phi(x) = r is a BOUNDED CONSTANT (vs the linear kernel's ||X||^2). So a player's curve
    depends on the DIRECTION of their latent embedding, not its magnitude — removing the ||X||^2
    inflation that let extreme-covariate players (e.g. high draft picks) get pathological
    curvature/peak ages. Cosine is a dot-product kernel, so it keeps the exact weight-space feature
    map (W/beta stay iid Normal) — no RFF / spectral density; it is the linear kernel on (scaled)
    unit-normalized X.

    The sqrt(r) scale is deliberate: it makes ||phi||^2 = r match the linear kernel's E||X||^2 = r,
    so the inherited 1/sqrt(r) normalizations in _compute_convex_mu / _build_{t,c}_max_curve land at
    the SAME effective amplitude as a typical convex player. Without it (plain X/||X||), ||phi||=1 and
    the inherited 1/sqrt(r) over-shrinks the level/peak ~1/sqrt(r) and the (quadratic) curvature ~1/r.
    With it, the convex-inherited alpha / sigma_c / sigma_t transfer directly — no amplitude re-tuning.
    Normalizes ALL three processes (level c_max, peak age t_max, curvature), since they share psi_x."""

    def _project_X(self, X: jnp.ndarray, *, W: jnp.ndarray = None, lengthscale: jnp.ndarray = None):
        norm = jnp.sqrt(jnp.sum(jnp.square(X), axis=-1, keepdims=True))
        return X / (norm + 1e-8) * jnp.sqrt(X.shape[-1])


class ConvexMaxLKJTVLinearLVM(ConvexMaxTVLinearLVM):
    """ConvexMaxTVLinearLVM with an LKJ correlation prior across the METRIC (k) axis of the HSGP
    curve weights beta, so metrics (e.g. obpm/dbpm) share curvature structure instead of each having
    independent N(0,1) weights.

    Construction (keeps beta[r,m,k] marginally N(0,1) — preserves the alpha/spectral calibration):
        L_corr ~ LKJCholesky(k, eta)                  # Cholesky of a k x k unit-diagonal correlation
        z      ~ Normal(), shape (r, M_time, k)       # the "beta" site, i.i.d. standard normal
        beta[r,m,:] = L_corr @ z[r,m,:]               # Var(beta[r,m,k]) = (L_corr L_corrᵀ)[k,k] = 1
    Because beta is a linear combination of standard normals it is Gaussian; the unit diagonal of the
    correlation makes each marginal exactly N(0,1); the off-diagonals carry the cross-metric sharing.
    Only _build_curve_weights changes — everything else (the convex-max forward, REs, survival) is
    inherited, so the LKJ variant is a drop-in for the curve weights.

    `lkj_concentration` (eta) is an attribute knob (set before initialize_priors): eta>1 shrinks toward
    independence (identity correlation), eta->1 is uniform over correlation matrices."""

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        eta = float(getattr(self, "lkj_concentration", 2.0))
        self.prior["beta_corr_chol"] = LKJCholesky(self.k, concentration=eta)

    def _build_curve_weights(self, effective_r, M_time, spd_time):
        L_corr = self._resolve_prior("beta_corr_chol")                       # (k, k) lower-tri Cholesky
        z = self._resolve_prior("beta", sample_shape=(effective_r, M_time, self.k))   # i.i.d. N(0,1)
        weights = jnp.einsum("rmj,kj->rmk", z, L_corr)                       # unit-variance, k-correlated
        return weights * spd_time.T[None]


class NaiveLinearLVM(ConvexMaxTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        for _k in ("sigma_t", "sigma_c", "t_max_raw", "c_max", "beta", "alpha",
                   "lengthscale_deriv", "exit_rate", "entrance", "exit"):
            self.prior.pop(_k, None)
        self.prior["c_offset"] = Normal(0, 1)
        # AR(1) per player per metric, parameters shared across players within metric
        self.prior["rho_ar"]   = Uniform(-.5, .5)
        self.prior["sigma_ar"] = HalfNormal()
        self.prior["beta_ar"]  = StudentT(3)
        self.prior["ar_0"]     = Normal(0, 1)
    def _build_max_curves(self, *args, **kwargs):
        raise NotImplementedError("NaiveLinearLVM does not use max curves")
    def _build_t_max_curve(self, psi_x, t_max_raw, sigma_t_max, t_offset, prior, *, amplitude = 10, offset_mode = "arctanh", scale_inside = True, deterministic_amplitude = None):
        raise NotImplementedError("NaiveLinearLVM does not use max curves")
    def _build_c_max_curve(self, psi_x, c_max_raw, sigma_c_max, c_offset, prior, *, scale_inside = True):
        raise NotImplementedError("NaiveLinearLVM does not use max curves")
    def _project_X(self, X, *, W = None, lengthscale = None):
        raise NotImplementedError("NaiveLinearLVM does not use projected X")
    def _compute_phi_at_max(self, t_max, L_time, M_time):
        raise NotImplementedError("NaiveLinearLVM does not use max curves")
    def _compute_convex_mu(self, psi_x, weights_scaled, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max, c_max, prior):
        raise NotImplementedError("NaiveLinearLVM does not use convex mu computation")

    def _compute_mu(self, hsgp_params, offsets={}, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0):
        """Naive forward: a per player x metric level (c_offset) plus a per-series AR(1). There is no
        latent aging curve. Returns (mu, TREND_AR, X) for interface parity with the convex family so
        prior_check.py's per-player draws can call it uniformly (TREND_AR is zero; X is None)."""
        mean = self._resolve_prior("c_offset", sample_shape=(self.k, self.n, 1))
        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar   = self._resolve_prior("rho_ar",   sample_shape=(self.k, 1))
        z        = self._resolve_prior("beta_ar",  sample_shape=(self.j, self.k, self.n))
        ar_0     = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
        ar       = self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)
        mu = jnp.repeat(mean, repeats=self.t, axis=-1) + ar
        return mu, jnp.zeros((self.k, self.n, self.j)), None

    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method = "prior", sample_free_indices = jnp.array([]), sample_fixed_indices = jnp.array([]), **kwargs):
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        mu, _trend_ar, _X = self._compute_mu(hsgp_params, offsets, sample_free_indices, sample_fixed_indices)
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_taus = None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
        )
        self.compute_survival_likelihood(offsets=offsets)
    
    def compute_survival_likelihood(self, offsets={}):
        required_keys = ("entrance_times", "exit_times", "right_censor")
        if not all(key in offsets for key in required_keys):
            return
        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_latent = jnp.maximum(entrance_times, 1e-6)

        eta_global_log = self._resolve_prior("eta_global_log", sample_shape=(self.n, 1))
        eta = jnp.exp(eta_global_log)        # (n, 1) — per-player baseline hazard
        gamma_global_log = self._resolve_prior("gamma_global_log", sample_shape=(self.n, 1))
        gamma = jnp.exp(gamma_global_log)    # (n, 1) — per-player aging rate

        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        entry_effective = jnp.maximum(entrance_latent, 0.0)
        stop_effective = jnp.maximum(exit_times, 0.0)
        has_exposure_window = stop_effective > entry_effective
        ratio = eta / gamma  # (n, 1)
        cumulative_H = jnp.where(
            has_exposure_window,
            ratio.squeeze(-1) * (
                jnp.exp(gamma.squeeze(-1) * stop_effective)
                - jnp.exp(gamma.squeeze(-1) * entry_effective)
            ),
            0.0,
        )
        log_h_event = jnp.log(eta.squeeze(-1)) + gamma.squeeze(-1) * exit_times

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)

class ConvexMaxLinearTrendTVLinearLVM(ConvexMaxTVLinearLVM):
    """Like ConvexMaxTVLinearLVM but replaces the calendar-year AR(1) trend with a
    ReLU-clamped linear trend: f(t) = beta * [relu(t-t_min) - relu(t-t_max)].

    Linear within the observed year range, constant at the boundary values outside.
    Eliminates the AR's sign ambiguity and prevents the calendar-year component from
    absorbing within-career age curvature (e.g. fg3a). No player-level AR process.
    """

    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, output_shape, basis)

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        for key in ("rho_year_ar", "sigma_year_ar", "beta_year_ar", "ar_0_year"):
            self.prior.pop(key, None)
        self.prior["beta_slope_year"] = HalfNormal(1.0)

    @staticmethod
    def _compute_linear_clamped_trend(beta_slope, num_years, ref_year_idx=0, year_max_idx=None):
        """Leaky-ReLU-clamped linear trend, with t normalised to [0, 1].

        f(t_norm) = beta * [leaky_relu(t_norm) - leaky_relu(t_norm - 1)]

        t_norm = (t - ref_year_idx) / right, so t_norm=0 at ref_year_idx and
        t_norm=1 at year_max_idx.  beta therefore represents the total change
        over the observed data range, independent of how many years that spans.

        beta_slope   : (num_de_trend, 1)
        year_max_idx : 0-based index of the last observed year; defaults to num_years-1
        returns      : (num_de_trend, num_years)
        """
        t = jnp.arange(num_years, dtype=jnp.float32) - ref_year_idx
        right = float((num_years - 1 if year_max_idx is None else year_max_idx) - ref_year_idx)
        t_norm = t / right
        return beta_slope * (jax.nn.leaky_relu(t_norm[None, :]) - jax.nn.leaky_relu(t_norm[None, :] - 1.0))

    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, observed_covariates=None, ref_year_idx: int = 0, year_max_idx: int = None) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_max = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))

        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        X = self._resolve_latent_X(sample_free_indices, sample_fixed_indices)
        if observed_covariates is not None:
            X = jnp.concatenate([X, jnp.asarray(observed_covariates)], axis=-1)
        effective_r = X.shape[1]

        t_max_raw, c_max_raw = self._sample_max_raw_parameters(effective_r)
        psi_x = self._project_X(X)
        t_offset = self._resolve_t_offset(offsets)
        c_offset = self._resolve_c_offset(offsets)

        t_max, c_max = self._build_max_curves(
            psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max, t_offset, c_offset, prior,
            amplitude=hsgp_params["t_amplitude"],
        )
        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(effective_r, M_time, self.k))
        weights *= spd_time.T[None]

        mu = self._compute_convex_mu(
            psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max, c_max, prior,
        )

        if num_de_trend > 0:
            beta_slope_year = self._resolve_prior("beta_slope_year", sample_shape=(num_de_trend, 1))
            trend_years = self._compute_linear_clamped_trend(beta_slope_year, num_years, ref_year_idx, year_max_idx)
            trend_nj = trend_years[:, year_indices]   # (num_de_trend, n, j)
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
            TREND_AR = TREND_AR.at[ar_metric_indices].set(trend_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))

        self._sample_family_likelihoods(
            data_set, mu, prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            trend_ar=TREND_AR,
        )
        self.compute_survival_likelihood(X, offsets=offsets)


class ConvexMaxARTVLinearLVM(ConvexMaxTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis, player_covariates=None) -> None:
        super().__init__(latent_rank, output_shape, basis, player_covariates)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-.5, .5)
        self.prior["sigma_ar"] = HalfNormal(0.3)
        self.prior["beta_ar"] = Normal(0, 1)
        self.prior["ar_0"] = Normal(0, 1)
        # Calendar year AR(3) trend priors (only sampled for de_trend_metrics)
        self.prior["rho_year_ar"]   = Uniform(0.0, 0.99)  # positive persistence only; allows near-unit-root for 40-yr era trends
        self.prior["sigma_year_ar"] = HalfNormal(.05)  # loosened: z~N(0,1) needs to give ~0.05/yr to track era trends (e.g. fg3a +1.6 log units over 43 yrs)
        self.prior["beta_year_ar"]  = Normal()
        self.prior["ar_0_year"]     = Normal()

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        ar = context["ar"]
        trend_ar = context["trend_ar"]
        return self._build_linear_predictor(mu, k_indices, ar[k_indices] + trend_ar[k_indices])

    def _compute_player_ar(self):
        """Per-player AR(1) process (k, n, j) — sampled + assembled. Used by both model_fn (likelihood)
        and prior_check's per-player draws so the plotted trajectory includes the AR."""
        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
        return self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)


    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_max = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))

        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)

        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        sigma_W_proj = self._resolve_prior("sigma_W_proj")
        _n_cov = self.player_covariates.shape[1] if self.player_covariates is not None else 2
        W_proj = self._resolve_prior("W_proj", sample_shape=(_n_cov, self.r),
                                     dist_override=Normal(0, sigma_W_proj))
        sigma_X = self._resolve_prior("sigma_X")
        Z = jnp.asarray(self.player_covariates) if self.player_covariates is not None \
            else jnp.zeros((self.n, 2))
        x_loc = Z @ W_proj
        X = self._resolve_latent_X_structured(x_loc, sigma_X, sample_free_indices, sample_fixed_indices)
        effective_r = self._projected_feature_dim()

        t_max_raw, c_max_raw = self._sample_max_raw_parameters(effective_r)

        psi_x = self._project_X(X)
        t_offset = self._resolve_t_offset(offsets)
        c_offset = self._resolve_c_offset(offsets)

        t_max, c_max = self._build_max_curves(
            psi_x,
            t_max_raw,
            c_max_raw,
            sigma_t_max,
            sigma_c_max,
            t_offset,
            c_offset,
            prior,
            amplitude=hsgp_params["t_amplitude"],
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._build_curve_weights(effective_r, M_time, spd_time)

        AR = self._compute_player_ar()

        # Calendar year AR(3) trend — only for metrics in ar_metric_indices
        if num_de_trend > 0:
            ar_global_indices = ar_metric_indices   # (num_ar,)
            sigma_year_ar = self._resolve_prior("sigma_year_ar", sample_shape=(num_de_trend, 1))
            rho_year_ar   = self._resolve_prior("rho_year_ar",   sample_shape=(num_de_trend, 1))
            z_year        = self._resolve_prior("beta_year_ar",  sample_shape=(num_years, num_de_trend))
            ar_0_year_raw = self._resolve_prior("ar_0_year", sample_shape=(1, num_de_trend), site_name="AR_0_year")
            ar_0_year     = ar_0_year_raw * sigma_year_ar[None, :, 0]
            trend_ar_years = self._compute_ar1_calendar_process(sigma_year_ar, rho_year_ar, z_year, ar_0_year, ref_year_idx=ref_year_idx)
            # Gather: year_indices (n, j) → trend_ar_nj (num_ar, n, j)
            trend_ar_nj = trend_ar_years[:, year_indices]
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
            TREND_AR = TREND_AR.at[ar_global_indices].set(trend_ar_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))

        mu = self._compute_convex_mu(
            psi_x,
            weights,
            phi_t_max,
            phi_prime_t_max,
            phi_time,
            shifted_x_time,
            L_time,
            t_max,
            c_max,
            prior,
        )
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            ar=AR,
            trend_ar=TREND_AR,
        )
        self.compute_survival_likelihood(X, offsets=offsets)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)
    
        
    
    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        guide = AutoDelta(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1)
                  )
        print("Setup MAP")
        # result = svi.run(jax.random.PRNGKey(0),
        #                 num_steps = num_steps,progress_bar = True, init_state = initial_state, stable_update=True, **model_args)
        # return result.params, result.state

        state = svi.init(jax.random.PRNGKey(0), **model_args)
        init_params = None
        init_state_bytes = None
        if isinstance(initial_state, dict) and (("state" in initial_state) or ("samples" in initial_state)):
            if initial_state.get("state") is not None:
                init_state_bytes = initial_state["state"]
            else:
                init_params = initial_state.get("samples")
        elif isinstance(initial_state, (bytes, bytearray)):
            init_state_bytes = initial_state
        elif initial_state is not None:
            init_params = initial_state

        if init_state_bytes is not None:
            state = ser.from_bytes(state, init_state_bytes)
            result = svi.run(
                jax.random.PRNGKey(0),
                num_steps=num_steps,
                progress_bar=True,
                init_state=state,
                stable_update=True,
                **model_args,
            )
        else:
            if isinstance(init_params, dict) and (len(init_params) > 0) and all((isinstance(key, str) and (not key.endswith("__loc"))) for key in init_params.keys()):
                init_params = {f"{key}__loc": value for key, value in init_params.items()}
            result = svi.run(
                jax.random.PRNGKey(0),
                num_steps=num_steps,
                progress_bar=True,
                init_state=state if init_params is None else None,
                init_params=init_params,
                stable_update=True,
                **model_args,
            )
        _print_svi_loss_breakdown(
            model_fn=self.model_fn,
            guide=guide,
            params=result.params,
            model_args=model_args,
            rng_key=jax.random.PRNGKey(1),
            debug_label=f"{type(self).__name__} MAP",
        )
        return result.params, result.state

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
       
        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1), 
                  )
        print("Setup SVI")
        effective_init_state = None if (initial_state is None or (isinstance(initial_state, dict) and len(initial_state) == 0)) else initial_state
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
            init_state=effective_init_state,
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)


class ConvexMaxARLKJTVLinearLVM(ConvexMaxARTVLinearLVM, ConvexMaxLKJTVLinearLVM):
    """ConvexMaxARTVLinearLVM + LKJ correlation across the metric axis of the curve weights.
    Inherits the AR(1) forward from ConvexMaxARTVLinearLVM and the LKJ beta construction +
    beta_corr_chol prior from ConvexMaxLKJTVLinearLVM. Cooperative super() in initialize_priors adds
    both the AR priors and beta_corr_chol; the AR model_fn builds weights via _build_curve_weights,
    which the MRO resolves to the LKJ override. No new body needed."""
    pass


class ConvexMaxARTVCosineLVM(ConvexMaxARTVLinearLVM, ConvexMaxTVCosineLVM):
    """ConvexMaxARTVLinearLVM with the cosine (normalized dot-product) latent kernel.
    Inherits the AR(1) forward from ConvexMaxARTVLinearLVM and the L2-normalized _project_X from
    ConvexMaxTVCosineLVM; the MRO resolves _project_X to the cosine override, so the AR model_fn's
    `psi_x = self._project_X(X)` uses X/||X||. No new body needed."""
    pass


class ConvexMaxRFFTVLinearLVM(ConvexMaxTVLinearLVM):
    """Random Fourier Feature latent kernel variant of ConvexMaxTVLinearLVM. Identical in every
    respect (structured X prior, level/peak/curvature REs, survival, compute_curves) except the
    latent feature map is the standard RFF approximation of a stationary (SE) kernel:
        phi(x) = [cos(W x), sin(W x)] / sqrt(m),   W ~ N(0, I)_{m x r}.
    This is the canonical norm-1 RFF: ||phi||^2 = 1 and phi(x)^T phi(x') ~ K(x,x') with K(x,x)=1, so
    the kernel approximation is faithful (no amplitude baked into the feature map). The curve/hazard
    amplitude is kept correct by overriding _kernel_self_cov -> 1.0 (the SE self-covariance): the
    inherited 1/sqrt(K(x,x)) normalizations then divide by 1 for RFF, matching the linear/cosine
    amplitude without re-tuning alpha / sigma_c / sigma_t. _projected_feature_dim stays 2*m (it sizes
    the weights to contract with the 2*m features). Single shared W across level/peak-age/peak-value/
    curvature (compute_curves calls _project_X(X) once)."""

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis, player_covariates=None) -> None:
        super().__init__(latent_rank, output_shape, basis, player_covariates)
        self.m = rff_dim   # number of RFF frequencies; projected feature dim = 2*m
        self._rff_W = None          # per-forward cache of the sampled frequency / inverse-bandwidth,
        self._rff_lengthscale = None  # populated in _resolve_latent_X_structured (single sample point)

    def _projected_feature_dim(self):
        return 2 * self.m

    def _kernel_self_cov(self, psi_x):
        # Standard RFF is norm-1: ||phi||^2 = 1 ~ K(x,x) for the unit-variance SE kernel. Normalizing
        # by 1 (not the 2*m feature count) keeps phi norm-1 AND the curves correctly scaled.
        return 1.0

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        # RFF frequency + inverse-bandwidth sites (sampled once per forward in _resolve_latent_X_structured).
        # The structured StudentT prior on the latent X (set by the parent) is kept; W/lengthscale
        # parameterize the RFF map on top of the latent. W is iid N(0,I) (the SE spectral density).
        self.prior["W"] = Normal()                # resolved with sample_shape (m, r)
        self.prior["lengthscale"] = HalfNormal()  # resolved with sample_shape (r,)

    def _resolve_latent_X_structured(self, x_loc, x_scale, sample_free_indices, sample_fixed_indices):
        # Sample W and lengthscale ONCE here, at the single X-resolution point that always runs before
        # every _project_X call in a given forward (the curve in compute_curves / AR model_fn, then the
        # hazard in _survival_rates). Sampling inside _project_X instead would register a duplicate "W"
        # site (it's invoked once for the curve and again for survival within the same trace).
        X = super()._resolve_latent_X_structured(x_loc, x_scale, sample_free_indices, sample_fixed_indices)
        self._rff_W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        self._rff_lengthscale = self._resolve_prior("lengthscale", sample_shape=(self.r,))[None]
        return X

    def _project_X(self, X: jnp.ndarray, *, W: jnp.ndarray = None, lengthscale: jnp.ndarray = None):
        # Reuse the W/lengthscale sampled in _resolve_latent_X_structured (always run first). Fall back
        # to sampling only if the cache is unset (a standalone projection with no prior X resolution).
        if W is None:
            W = self._rff_W if self._rff_W is not None else self._resolve_prior("W", sample_shape=(self.m, self.r))
        if lengthscale is None:
            lengthscale = self._rff_lengthscale if self._rff_lengthscale is not None \
                else self._resolve_prior("lengthscale", sample_shape=(self.r,))[None]
        _, phi = self._build_rff_features(X, W, lengthscale)   # (n, 2m), ||phi||^2 = 1 (standard RFF)
        return phi                                             # norm-1; amplitude handled by _kernel_self_cov


class ConvexMaxARRFFTVLinearLVM(ConvexMaxARTVLinearLVM, ConvexMaxRFFTVLinearLVM):
    """ConvexMaxARTVLinearLVM with the RFF latent kernel. Inherits the AR(1) + calendar-year-trend
    forward from ConvexMaxARTVLinearLVM and the RFF _project_X / _projected_feature_dim / W+lengthscale
    priors from ConvexMaxRFFTVLinearLVM; the MRO resolves _project_X to the RFF override, so the AR
    model_fn's `psi_x = self._project_X(X)` uses the RFF map. Mirrors ConvexMaxARTVCosineLVM."""

    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis, player_covariates=None) -> None:
        ConvexMaxRFFTVLinearLVM.__init__(self, latent_rank, rff_dim, output_shape, basis, player_covariates)


class TVLinearLVM(ConvexMaxTVLinearLVM):
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        del self.prior["lengthscale_deriv"]
        del self.prior["t_max_raw"]
        del self.prior["sigma_t"]
        self.prior["lengthscale"] = InverseGamma(1.0, 1.0)

    def _build_tv_beta(self, alpha_time, kernel_base):
        """GP curve weights beta (k, r, T): a per (metric, latent-dim) MVN(0, kernel_base) draw over
        the age basis, scaled by the per-metric amplitude alpha. Base: metrics are i.i.d. Subclasses
        (LKJTVLinearLVM) correlate the metric axis while preserving each metric's MVN marginal."""
        beta_raw = self._resolve_prior(
            "beta", site_name="beta",
            dist_override=MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel_base),
            sample_shape=(self.k, self.r),
        )
        return jnp.reshape(alpha_time, (self.k, 1, 1)) * beta_raw

    def _compute_mu(self, hsgp_params, offsets={}, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0):
        """GPLVM forward: a per-metric GP-over-time curve beta (MVN with the `lengthscale` kernel)
        projected onto the latent X, plus the c_max level. Returns (mu, TREND_AR, X) for interface
        parity with the convex family; mu is the (k,n,j) aging curve WITHOUT the per-player AR /
        calendar trend (those enter the likelihood), matching ConvexMaxTVLinearLVM._compute_mu."""
        prior = getattr(self, "_prior_predictive", False)
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls = self._resolve_prior("lengthscale")
        kernel_base = self.make_kernel(ls)
        beta = self._build_tv_beta(alpha_time, kernel_base)

        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_eff = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        c_raw = self._resolve_prior("c_max", sample_shape=(self.r, self.k))

        sigma_W_proj = self._resolve_prior("sigma_W_proj")
        _n_cov = self.player_covariates.shape[1] if self.player_covariates is not None else 2
        W_proj = self._resolve_prior("W_proj", sample_shape=(_n_cov, self.r),
                                     dist_override=Normal(0, sigma_W_proj))
        sigma_X = self._resolve_prior("sigma_X")
        Z = jnp.asarray(self.player_covariates) if self.player_covariates is not None \
            else jnp.zeros((self.n, 2))
        x_loc = Z @ W_proj
        X = self._resolve_latent_X_structured(x_loc, sigma_X, sample_free_indices, sample_fixed_indices)

        c_offset = offsets.get("c_mean", 0.0)
        intercept = self._build_c_max_curve(X, c_raw, sigma_c_eff, c_offset, prior)

        mu = (
            jnp.transpose(intercept)[:, :, None]
            + jnp.einsum("nr, krt -> knt", X, beta) / jnp.sqrt(self.r)
        )

        if num_de_trend > 0:
            sigma_year_ar = self._resolve_prior("sigma_year_ar", sample_shape=(num_de_trend, 1))
            rho_year_ar   = self._resolve_prior("rho_year_ar",   sample_shape=(num_de_trend, 1))
            z_year        = self._resolve_prior("beta_year_ar",  sample_shape=(num_years, num_de_trend))
            ar_0_year_raw = self._resolve_prior("ar_0_year", sample_shape=(1, num_de_trend), site_name="AR_0_year")
            ar_0_year     = ar_0_year_raw * sigma_year_ar[None, :, 0]
            trend_ar_years = self._compute_ar1_calendar_process(sigma_year_ar, rho_year_ar, z_year, ar_0_year, ref_year_idx=ref_year_idx)
            trend_ar_nj = trend_ar_years[:, year_indices]
            TREND_AR = jnp.zeros((self.k, self.n, self.j)).at[ar_metric_indices].set(trend_ar_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
        return mu, TREND_AR, X

    def compute_curves(self, hsgp_params, offsets={}, sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0, include_derivs: bool = False):
        """Single-source export forward for the non-convex GPLVM. mu comes from this class's own
        _compute_mu; the GPLVM has no analytic peak, so peaks are argmax/max over age and derivatives
        are finite differences (matching the retired make_mu_tvlinearlvm_mcmc). Per-player AR is added
        by the caller via _compute_player_ar() (zero for the non-AR variant)."""
        mu, TREND_AR, X = self._compute_mu(
            hsgp_params, offsets, sample_free_indices, sample_fixed_indices,
            ar_metric_indices, year_indices, num_years, num_de_trend, ref_year_idx)
        basis = jnp.asarray(self.basis)
        peak_idx = jnp.argmax(mu, axis=-1)                                   # (k, n)
        t_max = jnp.swapaxes(basis[peak_idx] - basis.mean(), -1, -2)         # (n, k) centered, matches convex convention
        c_max = jnp.swapaxes(jnp.max(mu, axis=-1), -1, -2)                   # (n, k)
        out = {"mu": mu, "t_max": t_max, "c_max": c_max, "trend_ar": TREND_AR, "X": X}
        if include_derivs:
            first_deriv = jnp.gradient(mu, axis=-1)                          # (k, n, t)
            d2 = jnp.gradient(first_deriv, axis=-1)
            d3 = jnp.gradient(d2, axis=-1)
            pe = peak_idx[..., None]
            second_deriv = jnp.swapaxes(jnp.take_along_axis(d2, pe, axis=-1).squeeze(-1), -1, -2)  # (n, k)
            third_deriv  = jnp.swapaxes(jnp.take_along_axis(d3, pe, axis=-1).squeeze(-1), -1, -2)  # (n, k)
            out["first_deriv"], out["second_deriv"], out["third_deriv"] = first_deriv, second_deriv, third_deriv
        return out

    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior",
                 sample_free_indices: jnp.ndarray = jnp.array([]),
                 sample_fixed_indices: jnp.ndarray = jnp.array([]),
                 ar_metric_indices: jnp.ndarray = jnp.array([]),
                 year_indices: jnp.ndarray = jnp.array([]),
                 num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0) -> None:
        prior = getattr(self, "_prior_predictive", False)
        mu, TREND_AR, X = self._compute_mu(
            hsgp_params, offsets, sample_free_indices, sample_fixed_indices,
            ar_metric_indices, year_indices, num_years, num_de_trend, ref_year_idx,
        )
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            trend_ar=TREND_AR,
        )
        self.compute_survival_likelihood(X, offsets=offsets)


class TVLinearLVM_AR(TVLinearLVM):
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"]      = Uniform(-.5, .5)
        self.prior["sigma_ar"]    = HalfNormal(0.3)
        self.prior["beta_ar"]     = Normal(0, 1)
        self.prior["ar_0"]        = Normal(0, 1)
        self.prior["rho_year_ar"] = Uniform(0.0, 0.99)

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        ar = context["ar"]
        trend_ar = context["trend_ar"]
        return self._build_linear_predictor(mu, k_indices, ar[k_indices] + trend_ar[k_indices])

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)

    def _compute_player_ar(self):
        """Per-player AR(1) process (k, n, j) — used by model_fn and prior_check's per-player draws."""
        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar   = self._resolve_prior("rho_ar",   sample_shape=(self.k, 1))
        z        = self._resolve_prior("beta_ar",  sample_shape=(self.j, self.k, self.n))
        ar_0     = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") \
                   * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
        return self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)

    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method: str = "prior",
                 sample_free_indices: jnp.ndarray = jnp.array([]),
                 sample_fixed_indices: jnp.ndarray = jnp.array([]),
                 ar_metric_indices: jnp.ndarray = jnp.array([]),
                 year_indices: jnp.ndarray = jnp.array([]),
                 num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0) -> None:
        prior = getattr(self, "_prior_predictive", False)
        # mu (aging curve) + calendar TREND_AR are the shared GPLVM forward (inherited from
        # TVLinearLVM); this subclass adds the per-player AR(1), folded into the likelihood.
        mu, TREND_AR, X = self._compute_mu(
            hsgp_params, offsets, sample_free_indices, sample_fixed_indices,
            ar_metric_indices, year_indices, num_years, num_de_trend, ref_year_idx,
        )
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        AR = self._compute_player_ar()

        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            ar=AR,
            trend_ar=TREND_AR,
        )
        self.compute_survival_likelihood(X, offsets=offsets)


class LKJTVLinearLVM(TVLinearLVM):
    """TVLinearLVM (non-convex GPLVM) + LKJ correlation across the metric axis of the GP curve
    weights. beta_raw[k] = Σ_j L_corr[k,j] z[j] with z[j] ~ MVN(0, kernel_base): each metric's
    marginal stays MVN(0, kernel_base) (amplitude/lengthscale calibration unchanged) while metrics
    share curvature structure. eta via the attribute knob lkj_concentration."""

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        eta = float(getattr(self, "lkj_concentration", 2.0))
        self.prior["beta_corr_chol"] = LKJCholesky(self.k, concentration=eta)

    def _build_tv_beta(self, alpha_time, kernel_base):
        L_corr = self._resolve_prior("beta_corr_chol")   # (k, k) lower-tri Cholesky of correlation
        z = self._resolve_prior(
            "beta", site_name="beta",
            dist_override=MultivariateNormal(loc=jnp.zeros_like(self.basis), covariance_matrix=kernel_base),
            sample_shape=(self.k, self.r),
        )   # (k, r, T) i.i.d. across metrics
        beta_raw = jnp.einsum("kj,jrt->krt", L_corr, z)   # correlate metric axis; MVN marginal preserved
        return jnp.reshape(alpha_time, (self.k, 1, 1)) * beta_raw


class LKJTVLinearLVM_AR(TVLinearLVM_AR, LKJTVLinearLVM):
    """TVLinearLVM_AR + LKJ metric-correlated curve weights: AR forward from TVLinearLVM_AR, LKJ
    beta + beta_corr_chol prior from LKJTVLinearLVM (cooperative super()). No new body needed."""
    pass


class ConvexMaxInjuryTVLinearLVM(ConvexMaxARTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis, injury_rank: int, num_injury_types: int, player_covariates=None) -> None:
        super().__init__(latent_rank, output_shape, basis, player_covariates)
        self.i = num_injury_types
        self.p = injury_rank

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["injury_factor"] = Normal(0, 1)
        self.prior["injury_loading"] = Normal(0, 1)
        self.prior["injury_global_offset"] = Normal(0, 1)
        self.prior["sigma_c"] = HalfNormal(1.5)           # unit-scale: multiplied by sqrt(c_max_var) per metric in model_fn; loosened to allow elite-player peaks (LeBron, KG, etc.)
        self.prior["sigma_t"] = HalfNormal()
        self.prior["injury_exit_loading"] = Normal(0, 1)
        self.prior["injury_exit_global_offset"] = Normal(0, 1)
        self.prior["sigma_injury_exit"] = HalfNormal()
        self.prior["injury_exit_raw"] = Normal()
        self.prior["injury_time_raw"] = Normal()
        self.prior["sigma_injury"] = HalfNormal()

    def compute_survival_likelihood(self, X, injury_factor, offsets = {}) -> None:
        required_keys = ("entrance_times", "exit_times", "right_censor", "injury_indicator", "injury_type")
        if not all(key in offsets for key in required_keys):
            return

        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_latent = jnp.maximum(entrance_times, 1e-6)

        effective_r = X.shape[-1]
        psi_x = self._project_X(X)   # kernel feature map (cosine bounds the hazard; identity for linear)
        exit = self._resolve_prior("exit", sample_shape=(effective_r,))
        sigma_exit_scale = self._resolve_prior("sigma_exit_scale")
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(effective_r,))
        exit_raw = make_psi_gamma(psi_x, exit) / jnp.sqrt(effective_r) * sigma_exit_scale

        # Baseline hazard η — from latent X only, injury does not elevate it
        eta_global_log = self._resolve_prior("eta_global_log")
        eta = jnp.exp(eta_global_log + exit_raw)[:, None]   # (n, 1)

        # Injury effect on aging rate γ
        injury_exit_loading = self._resolve_prior("injury_exit_loading", sample_shape=(self.p,))
        injury_exit_global_offset = self._resolve_prior("injury_exit_global_offset")
        injury_exit_raw = (
            injury_exit_global_offset
            + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)[None, None, :]
        )  # (1, 1, i)

        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        if injury_indicator.ndim == 3:
            injury_indicator = injury_indicator[0]
        if injury_type.ndim == 3:
            injury_type = injury_type[0]

        injury_exit_padded = jnp.concatenate(
            [jnp.zeros(injury_exit_raw.shape[:-1] + (1,), dtype=injury_exit_raw.dtype),
             injury_exit_raw],
            axis=-1,
        )  # (1, 1, i+1) — take_along_axis broadcasts over (n, t)
        injury_effect_exit = jnp.take_along_axis(
            injury_exit_padded, injury_type[..., None], -1
        ).squeeze(-1)  # (n, t)

        # Aging rate γ — time-varying due to injury type at each interval
        gamma_base = make_psi_gamma(psi_x, exit_rate)[:, None] / jnp.sqrt(effective_r)  # (n, 1): 1/sqrt(r) scaled-dot-product, matching exit_raw/eta
        gamma_global_log = self._resolve_prior("gamma_global_log")
        gamma = jnp.exp(gamma_global_log + gamma_base + injury_effect_exit)  # (n, t)

        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        interval_starts = jnp.arange(self.t, dtype=exit_times.dtype)[None, :]  # (1, t)
        interval_ends = interval_starts + 1.0
        entry = entrance_latent[:, None]   # (n, 1)
        stop = exit_times[:, None]         # (n, 1)
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop)
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.where(valid_seg, seg_start, 0.0)
        seg_end_safe = jnp.where(valid_seg, seg_end, 0.0)
        valid_seg_float = valid_seg.astype(exit_times.dtype)
        ratio = eta / gamma  # (n, t) — time-varying
        delta_H = valid_seg_float * ratio * (
            jnp.exp(gamma * seg_end_safe) - jnp.exp(gamma * seg_start_safe)
        )
        cumulative_H = delta_H.sum(axis=-1)  # (n,)

        event_time = exit_times
        event_interval = jnp.clip(jnp.floor(event_time).astype(jnp.int32), 0, self.t - 1)
        gamma_event = jnp.take_along_axis(gamma, event_interval[:, None], axis=1).squeeze(-1)
        log_h_event = jnp.log(eta.squeeze(-1)) + gamma_event * event_time

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        return self._build_linear_predictor(mu, k_indices)
        


    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([]), ar_metric_indices: jnp.ndarray = jnp.array([]), year_indices: jnp.ndarray = jnp.array([]), num_years: int = 1, num_de_trend: int = 0, ref_year_idx: int = 0) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_max = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))

        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)

        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        sigma_W_proj = self._resolve_prior("sigma_W_proj")
        _n_cov = self.player_covariates.shape[1] if self.player_covariates is not None else 2
        W_proj = self._resolve_prior("W_proj", sample_shape=(_n_cov, self.r),
                                     dist_override=Normal(0, sigma_W_proj))
        sigma_X = self._resolve_prior("sigma_X")
        Z = jnp.asarray(self.player_covariates) if self.player_covariates is not None \
            else jnp.zeros((self.n, 2))
        x_loc = Z @ W_proj
        X = self._resolve_latent_X_structured(x_loc, sigma_X, sample_free_indices, sample_fixed_indices)
        effective_r = self.r

        t_max_raw, c_max_raw = self._sample_max_raw_parameters(effective_r)

        psi_x = self._project_X(X)
        t_offset = self._resolve_t_offset(offsets)
        c_offset = self._resolve_c_offset(offsets)
        t_max, c_max = self._build_max_curves(
            psi_x,
            t_max_raw,
            c_max_raw,
            sigma_t_max,
            sigma_c_max,
            t_offset,
            c_offset,
            prior,
            amplitude=hsgp_params["t_amplitude"],
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(effective_r, M_time, self.k))
        weights *= spd_time.T[None]

        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        AR = self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)
        # Calendar year AR(3) trend — only for metrics in ar_metric_indices

        if num_de_trend > 0:
            ar_global_indices = ar_metric_indices   # (num_ar,)
            sigma_year_ar = self._resolve_prior("sigma_year_ar", sample_shape=(num_de_trend, 1))
            rho_year_ar   = self._resolve_prior("rho_year_ar",   sample_shape=(num_de_trend, 1))
            z_year        = self._resolve_prior("beta_year_ar",  sample_shape=(num_years, num_de_trend))
            ar_0_year_raw = self._resolve_prior("ar_0_year", sample_shape=(1, num_de_trend), site_name="AR_0_year")
            ar_0_year     = ar_0_year_raw * sigma_year_ar[None, :, 0]
            trend_ar_years = self._compute_ar1_calendar_process(sigma_year_ar, rho_year_ar, z_year, ar_0_year, ref_year_idx=ref_year_idx)
            trend_ar_nj = trend_ar_years[:, year_indices]
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
            TREND_AR = TREND_AR.at[ar_global_indices].set(trend_ar_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
        injury_loading = self._resolve_prior("injury_loading", sample_shape=(self.k, self.p))
        injury_factor = self._resolve_prior("injury_factor", sample_shape=(self.i, self.p))
        injury_global_offset = self._resolve_prior("injury_global_offset", sample_shape=(self.k,))
        injury_mean_prior = jnp.einsum("ip, kp -> ki", injury_factor, injury_loading)
        sigma_injury = self._resolve_prior("sigma_injury", sample_shape=(self.k,))  # (k,)
        injury_time_raw = self._resolve_prior("injury_time_raw", sample_shape=(self.j, self.i))  # (j, i)
        injury_effect_raw = (
            injury_global_offset[:, None, None, None]                              # (k, 1, 1, 1)
            + injury_mean_prior[:, None, None, :]                                  # (k, 1, 1, i)
            + sigma_injury[:, None, None, None] * injury_time_raw[None, None, :, :]  # (k, 1, j, i)
        )  # (k, 1, j, i) — uniform over players, time-varying per injury type
        injury_effect_padded = jnp.concatenate(
            [jnp.zeros(injury_effect_raw.shape[:-1] + (1,), dtype=injury_effect_raw.dtype),
             injury_effect_raw],
            axis=-1
        )  # (k, 1, j, i+1) — take_along_axis broadcasts over n
        injury_effect = jnp.take_along_axis(injury_effect_padded, injury_type[..., None], -1).squeeze(-1)
        mu_base = self._compute_convex_mu(
            psi_x,
            weights,
            phi_t_max,
            phi_prime_t_max,
            phi_time,
            shifted_x_time,
            L_time,
            t_max,
            c_max,
            False,
        )
        
        mu = self._build_mu_from_base(mu_base, prior, injury_effect, AR, TREND_AR)
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
        )
        self.compute_survival_likelihood(
            X,
            injury_factor,
            offsets=offsets)


    def run_inference(self, num_warmup, num_samples, num_chains, vectorized: bool, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning = thinning)



    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state = None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):

        guide = AutoLaplaceApproximation(self.model_fn, prefix="", **guide_kwargs)
        print("Setup guide")
        svi = SVI(self.model_fn, guide, optim=adam(.0003), loss=Trace_ELBO(num_particles=1),
                  )
        print("Setup SVI")
        effective_init_state = None if (initial_state is None or (isinstance(initial_state, dict) and len(initial_state) == 0)) else initial_state
        result = _run_svi_with_nan_checks(
            svi=svi,
            guide=guide,
            model_fn=self.model_fn,
            model_args=model_args,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(0),
            progress_bar=True,
            debug_nan=debug_nan,
            debug_label=f"{type(self).__name__} SVI",
            init_state=effective_init_state,
        )
        params = result.params
        posterior_samples = guide.sample_posterior(jax.random.PRNGKey(1), params = params, sample_shape=sample_shape)
        return posterior_samples


    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)



class ConvexMaxDecayInjuryTVLinearLVM(ConvexMaxInjuryTVLinearLVM):
    """Injury effects with pure exponential decay back to baseline.

    Replaces the static per-season injury effect of ConvexMaxInjuryTVLinearLVM
    with a two-parameter decay:

        delta_{k,n,t} = beta0_{k,n,i} * exp(-lambda_{k,i} * (t - T0_n))
                        for t >= T0_n, else 0

    where
        beta0_{k,n,i} -- initial performance shock, player- and injury-type-specific
        lambda_{k,i}  -- decay rate > 0, shared across players (per metric x injury type)
        T0_n          -- time-index of first injury for player n

    Both beta_0 and lambda are shared across players (per metric x injury type
    only) so they are jointly identified: beta_0 sets the average initial shock
    and lambda controls the recovery speed. Per-player heterogeneity in injury
    response is absorbed by the baseline LVM trajectory.
    """

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        # Performance decay-rate priors (log-scale; softplus maps to R+)
        self.prior["lambda_global_offset"] = Normal(0, 1)       # (k, i) -- per metric x injury type
        # Exit-hazard decay-rate priors
        self.prior["lambda_exit_global_offset"] = Normal(0, 1)  # (i,) -- per injury type
        # beta_0 is (k, i) in this model — decay handles time variation; nullify parent's noise sites
        self.prior["injury_raw"] = None
        self.prior["sigma_injury"] = None
        self.prior["injury_time_raw"] = None

    def compute_survival_likelihood(self, X, injury_factor, offsets={}) -> None:
        required_keys = ("entrance_times", "exit_times", "right_censor", "injury_indicator", "injury_type")
        if not all(key in offsets for key in required_keys):
            return

        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_latent = jnp.maximum(entrance_times, 1e-6)

        effective_r = X.shape[-1]
        psi_x = self._project_X(X)   # kernel feature map (cosine bounds the hazard; identity for linear)
        exit = self._resolve_prior("exit", sample_shape=(effective_r,))
        sigma_exit_scale = self._resolve_prior("sigma_exit_scale")
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(effective_r,))
        exit_raw = make_psi_gamma(psi_x, exit) / jnp.sqrt(effective_r) * sigma_exit_scale

        # Baseline hazard η — from latent X only, injury does not elevate it
        eta_global_log = self._resolve_prior("eta_global_log")
        eta = jnp.exp(eta_global_log + exit_raw)[:, None]   # (n, 1)

        # ---- Decayed injury effect on aging rate γ ---- #
        injury_exit_loading = self._resolve_prior("injury_exit_loading", sample_shape=(self.p,))
        injury_exit_global_offset = self._resolve_prior("injury_exit_global_offset")
        sigma_injury_exit = self._resolve_prior("sigma_injury_exit")

        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        if injury_indicator.ndim == 3:
            injury_indicator = injury_indicator[0]
        if injury_type.ndim == 3:
            injury_type = injury_type[0]

        # beta0_exit: initial aging-rate shock — (n, i)
        injury_exit_raw_beta0 = self._resolve_prior("injury_exit_raw", sample_shape=(self.n, self.i))
        beta0_exit = (
            injury_exit_global_offset
            + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)[None, :]  # (1, i)
            + injury_exit_raw_beta0 * sigma_injury_exit
        )  # (n, i)

        # lambda_exit: decay rate of the aging-rate shock — (i,), strictly > 0
        lambda_exit_global_offset = self._resolve_prior("lambda_exit_global_offset", sample_shape=(self.i,))
        lambda_exit = jax.nn.softplus(lambda_exit_global_offset)  # (i,)

        # Time since injury: delta_t[n, t] = max(t - T0_n, 0)
        t0_index = jnp.argmax(injury_indicator, axis=-1).astype(jnp.float32)  # (n,)
        t_grid = jnp.arange(self.t, dtype=jnp.float32)                        # (t,)
        delta_t = jnp.maximum(t_grid[None, :] - t0_index[:, None], 0.0)       # (n, t)

        # Decayed effect on log(γ): beta0 * exp(-lambda * delta_t) → (n, t, i)
        exit_decay_factor = jnp.exp(-lambda_exit[None, None, :] * delta_t[:, :, None])
        injury_exit_raw_decayed = beta0_exit[:, None, :] * exit_decay_factor   # (n, t, i)

        injury_effect_exit = jnp.take_along_axis(
            jnp.concatenate([jnp.zeros_like(injury_indicator)[..., None], injury_exit_raw_decayed], -1),
            injury_type[..., None],
            -1,
        ).squeeze(-1)  # (n, t)

        # Aging rate γ — time-varying via decayed injury effect
        gamma_base = make_psi_gamma(psi_x, exit_rate)[:, None] / jnp.sqrt(effective_r)  # (n, 1): 1/sqrt(r) scaled-dot-product, matching exit_raw/eta
        gamma_global_log = self._resolve_prior("gamma_global_log")
        gamma = jnp.exp(gamma_global_log + gamma_base + injury_effect_exit)  # (n, t)

        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        interval_starts = jnp.arange(self.t, dtype=exit_times.dtype)[None, :]  # (1, t)
        interval_ends = interval_starts + 1.0
        entry = entrance_latent[:, None]   # (n, 1)
        stop = exit_times[:, None]         # (n, 1)
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop)
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.where(valid_seg, seg_start, 0.0)
        seg_end_safe = jnp.where(valid_seg, seg_end, 0.0)
        valid_seg_float = valid_seg.astype(exit_times.dtype)
        ratio = eta / gamma  # (n, t)
        delta_H = valid_seg_float * ratio * (
            jnp.exp(gamma * seg_end_safe) - jnp.exp(gamma * seg_start_safe)
        )
        cumulative_H = delta_H.sum(axis=-1)  # (n,)

        event_time = exit_times
        event_interval = jnp.clip(jnp.floor(event_time).astype(jnp.int32), 0, self.t - 1)
        gamma_event = jnp.take_along_axis(gamma, event_interval[:, None], axis=1).squeeze(-1)
        log_h_event = jnp.log(eta.squeeze(-1)) + gamma_event * event_time

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)

    def model_fn(
        self,
        data_set,
        hsgp_params,
        offsets={},
        inference_method: str = "prior",
        sample_free_indices: jnp.ndarray = jnp.array([]),
        sample_fixed_indices: jnp.ndarray = jnp.array([]),
        ar_metric_indices: jnp.ndarray = jnp.array([]),
        year_indices: jnp.ndarray = jnp.array([]),
        num_years: int = 1,
        num_de_trend: int = 0,
        ref_year_idx: int = 0,
    ) -> None:
        prior = getattr(self, "_prior_predictive", False)
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0

        phi_time = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]

        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(
            jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(
                alpha_time, ls_deriv
            )
        )
        sigma_c_unit = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        _c_max_var = offsets.get("c_max_var", None)
        sigma_c_max = sigma_c_unit * jnp.sqrt(jnp.asarray(_c_max_var)) if _c_max_var is not None else sigma_c_unit
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))

        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        if num_beta > 0:
            sigma_beta = self._resolve_prior("sigma_beta", sample_shape=(num_beta,))
            expanded_sigma_beta = jnp.tile(sigma_beta[:, None, None], (1, self.n, self.j))
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        if num_beta_bins > 0:
            sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial", sample_shape=(num_beta_bins,))
            expanded_sigma_beta_bin = jnp.tile(sigma_beta_binomial[:, None, None], (1, self.n, self.j))

        sigma_W_proj = self._resolve_prior("sigma_W_proj")
        _n_cov = self.player_covariates.shape[1] if self.player_covariates is not None else 2
        W_proj = self._resolve_prior("W_proj", sample_shape=(_n_cov, self.r),
                                     dist_override=Normal(0, sigma_W_proj))
        sigma_X = self._resolve_prior("sigma_X")
        Z = jnp.asarray(self.player_covariates) if self.player_covariates is not None \
            else jnp.zeros((self.n, 2))
        x_loc = Z @ W_proj
        X = self._resolve_latent_X_structured(x_loc, sigma_X, sample_free_indices, sample_fixed_indices)
        effective_r = self.r

        t_max_raw, c_max_raw = self._sample_max_raw_parameters(effective_r)

        psi_x = self._project_X(X)

        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_c_offset(offsets)

        t_max, c_max = self._build_max_curves(
            psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max, t_offset, c_offset, prior,
            amplitude=hsgp_params["t_amplitude"],
        )
        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(effective_r, M_time, self.k))
        weights *= spd_time.T[None]

        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = (
            self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0")
            * (sigma_ar / jnp.sqrt(1 - rho_ar**2))
        )
        AR = self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)
        # Calendar year AR(3) trend — only for metrics in ar_metric_indices

        if num_de_trend > 0:
            ar_global_indices = ar_metric_indices 
            sigma_year_ar = self._resolve_prior("sigma_year_ar", sample_shape=(num_de_trend, 1))
            rho_year_ar   = self._resolve_prior("rho_year_ar",   sample_shape=(num_de_trend, 1))
            z_year        = self._resolve_prior("beta_year_ar",  sample_shape=(num_years, num_de_trend))
            ar_0_year_raw = self._resolve_prior("ar_0_year", sample_shape=(1, num_de_trend), site_name="AR_0_year")
            ar_0_year     = ar_0_year_raw * sigma_year_ar[None, :, 0]
            trend_ar_years = self._compute_ar1_calendar_process(sigma_year_ar, rho_year_ar, z_year, ar_0_year, ref_year_idx=ref_year_idx)
            trend_ar_nj = trend_ar_years[:, year_indices]
            TREND_AR = jnp.zeros((self.k, self.n, self.j))
            TREND_AR = TREND_AR.at[ar_global_indices].set(trend_ar_nj)
        else:
            TREND_AR = jnp.zeros((self.k, self.n, self.j))

        # ------------------------------------------------------------------ #
        # Injury effect: exponential decay                                    #
        # ------------------------------------------------------------------ #

        injury_loading = self._resolve_prior("injury_loading", sample_shape=(self.k, self.p))
        injury_factor = self._resolve_prior("injury_factor", sample_shape=(self.i, self.p))
        injury_global_offset = self._resolve_prior("injury_global_offset", sample_shape=(self.k,))
        sigma_injury = self._resolve_prior("sigma_injury", sample_shape=(self.k,))

        # Shared low-rank structure across injury types: (k, i)
        injury_mean_prior = jnp.einsum("ip,kp->ki", injury_factor, injury_loading)

        # --- beta_0: initial shock -- (k, i), shared across players ---
        # No per-player noise so that lambda receives a genuine gradient from
        # within-player temporal recovery patterns. Player heterogeneity in
        # injury response is captured by the baseline LVM trajectory.
        beta_0 = injury_global_offset[:, None] + injury_mean_prior  # (k, i)

        # --- lambda: decay rate -- (k, i), constrained > 0 via softplus ---
        lambda_global_offset = self._resolve_prior("lambda_global_offset", sample_shape=(self.k, self.i))
        decay_rate = jax.nn.softplus(lambda_global_offset)     # (k, i), strictly > 0

        # --- Time since injury: delta_t[n, t] = max(t - T0_n, 0) ---
        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        if injury_indicator.ndim == 3:
            injury_indicator = injury_indicator[0]
        if injury_type.ndim == 3:
            injury_type = injury_type[0]

        # argmax over the time axis gives the first injured step per player.
        # For uninjured players argmax returns 0, but injury_type stays 0 so
        # the effect is zeroed out by the take_along_axis selection below.
        t0_index = jnp.argmax(injury_indicator, axis=-1).astype(jnp.float32)  # (n,)
        t_grid = jnp.arange(self.t, dtype=jnp.float32)                        # (t,)
        delta_t = jnp.maximum(t_grid[None, :] - t0_index[:, None], 0.0)       # (n, t)

        # --- Pure exponential decay: beta_0 * exp(-lambda * delta_t) ---
        # beta_0: (k, i), decay_rate: (k, i), delta_t: (n, t)
        decay_factor = jnp.exp(
            -decay_rate[:, None, None, :] * delta_t[None, :, :, None]
        )  # (k, n, t, i)
        injury_effect_decayed = beta_0[:, None, None, :] * decay_factor  # (k, n, t, i)

        # Select active injury type; zeros prepended at index 0 handle d=0 (uninjured)
        # injury_effect_decayed: (k, n, t, i); need zeros (k, n, t, 1) and index (k, n, t, 1)
        zeros_ktni = jnp.zeros(injury_effect_decayed.shape[:-1] + (1,), dtype=injury_effect_decayed.dtype)
        injury_effect = jnp.take_along_axis(
            jnp.concatenate([zeros_ktni, injury_effect_decayed], -1),
            injury_type[None, :, :, None],   # (1, n, t, 1) broadcasts to (k, n, t, i+1)
            -1,
        ).squeeze(-1)  # (k, n, t)

        # ------------------------------------------------------------------ #
        # Base trajectory + likelihoods + survival                           #
        # ------------------------------------------------------------------ #

        mu_base = self._compute_convex_mu(
            psi_x, weights, phi_t_max, phi_prime_t_max, phi_time, shifted_x_time, L_time, t_max, c_max, False
        )
        mu = self._build_mu_from_base(mu_base, prior, injury_effect, AR, TREND_AR)
        self._sample_family_likelihoods(
            data_set,
            mu,
            prior=prior,
            expanded_sigmas=expanded_sigmas if num_gaussians > 0 else None,
            expanded_taus=None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
        )
        self.compute_survival_likelihood(X, injury_factor, offsets=offsets)

    def run_inference(self, num_warmup, num_samples, num_chains, vectorized, model_args, initial_values={}, thinning=1):
        return super().run_inference(num_warmup, num_samples, num_chains, vectorized, model_args, initial_values, thinning=thinning)

    def run_map_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state=None):
        return super().run_map_inference(num_steps, guide_kwargs, model_args, initial_state)

    def run_svi_inference(self, num_steps, guide_kwargs: dict = {}, model_args: dict = {}, initial_state: dict = {}, sample_shape=(4, 2000), debug_nan: bool = False):
        return super().run_svi_inference(num_steps, guide_kwargs, model_args, initial_state, sample_shape, debug_nan)

    def predict(self, posterior_samples: dict, model_args, num_samples=1000):
        return super().predict(posterior_samples, model_args, num_samples)
