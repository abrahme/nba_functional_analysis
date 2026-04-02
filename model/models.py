from abc import abstractmethod, ABC
from types import SimpleNamespace
import jax 
import flax.serialization as ser
import numpyro
import numpy as np
from numpyro import sample 
from numpyro.infer.util import log_density
from numpyro.distributions import  InverseGamma, Normal, Exponential, Poisson, Weibull, StudentT, Independent, Beta, HalfCauchy, LogNormal, Binomial, HalfNormal, Categorical, MultivariateNormal, BetaProportion, Distribution, Uniform, BetaBinomial, Gamma, BinomialLogits, NegativeBinomial2, Dirichlet, MixtureSameFamily
from numpyro.infer import MCMC, NUTS, init_to_median, SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoNormal,  AutoLaplaceApproximation
from numpyro.handlers import substitute, seed, trace, mask, replay
import optax
from optax import linear_onecycle_schedule, adam
from jaxopt import LBFGS
from .hsgp import make_convex_f, make_psi_gamma, make_spectral_mixture_density, diag_spectral_density, make_convex_phi,  vmap_make_convex_phi, vmap_make_convex_phi_prime, eigenfunctions_multivariate, vmap_make_convex_phi_double_prime
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
        if isinstance(prior_value, Distribution):
            dist_to_sample = dist_override if dist_override is not None else prior_value
            sample_kwargs = {}
            if sample_shape is not None:
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
        kernel = NUTS(self.model_fn, init_strategy=init_to_value(values=initial_values))
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

        return t_max_value if not prior else numpyro.deterministic("t_max", t_max_det_value)

    def _build_c_max_curve(self, psi_x, c_max_raw, sigma_c_max, c_offset, prior: bool, *, scale_inside: bool = True):
        c_offset_value = c_offset
        c_max_value = (
            make_psi_gamma(psi_x, c_max_raw * sigma_c_max)
            if scale_inside
            else make_psi_gamma(psi_x, c_max_raw) * sigma_c_max
        ) + c_offset_value
        return c_max_value if not prior else numpyro.deterministic("c_max_", c_max_value)

    def _build_max_curves(self, psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max, t_offset, c_offset, prior: bool):
        t_max = self._build_t_max_curve(
            psi_x,
            t_max_raw,
            sigma_t_max,
            t_offset,
            prior,
            amplitude=10.0,
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
        return mu_value if not prior else numpyro.deterministic("mu", mu_value)

    def _build_mu_from_base(self, mu_base, prior: bool, *effects):
        mu_value = self._compose_additive(mu_base, *effects)
        return mu_value if not prior else numpyro.deterministic("mu", mu_value)

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
        mu = make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None]) if not prior else numpyro.deterministic("mu", make_convex_f(gamma_phi_gamma_x, shifted_x_time, slope, (intercept + offsets)[..., None]))
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
        de_trend = family_data["de_trend"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return Normal(rate[mask], expanded_sigmas[mask] / exposure[mask]), obs[mask]
        if family == "poisson":
            log_rate = linear_predictor
            return Poisson(jnp.exp(log_rate[mask]+ exposure[mask] + de_trend[mask])), obs[mask]
        if family == "negative-binomial":
            expanded_sigma_neg_bin = context["expanded_sigma_neg_bin"]
            log_rate = linear_predictor
            dist = NegativeBinomial2(
                mean=jnp.exp(log_rate[mask] + exposure[mask] + de_trend[mask]),
                concentration=expanded_sigma_neg_bin[mask],
            )
            return dist, obs[mask]
        if family == "binomial":
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return BinomialLogits(logits=rate[mask], total_count=exposure[mask].astype(int)), obs[mask]
        if family == "beta-binomial":
            logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
            dist = BetaBinomial(
                concentration0=(1 - jsci.special.expit(logit_rate[mask])),
                concentration1=jsci.special.expit(logit_rate[mask]),
                total_count=exposure[mask].astype(int),
            )
            return dist, obs[mask]
        if family == "beta":
            logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return BetaProportion(jsci.special.expit(logit_rate[mask]), jnp.square(exposure[mask])), obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexMaxTVRFLVM likelihood builder")

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
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
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
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












class ConvexMaxBoundaryTVRFLVM(ConvexTVRFLVM, RFLVMMaxBase):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        and boundary conditions
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["lengthscale"] = InverseGamma(2, 1)
        self.prior["sigma_beta"] = Gamma(5000, 100.0)
        self.prior["sigma_beta_binomial"] = Gamma(5000, 100.0)
        self.prior["sigma_negative_binomial"] =  Gamma(5000, 100)
        self.prior["sigma_c"] = InverseGamma(2.0, 1.0)
        self.prior["sigma_boundary_r"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_boundary_l"] = InverseGamma(2.0, 3.0)
        self.prior["sigma_t"] = InverseGamma(2.0, 1.0)
        self.prior["alpha"] = LogNormal(jnp.log(3.0), 0.35)
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["boundary_l"] = Normal()
        self.prior["boundary_r"] = Normal()
        self.prior["X_free"] = Normal()

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        de_trend = family_data["de_trend"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return Normal(rate, expanded_sigmas / jnp.where(mask, exposure, 1.0)).mask(mask), obs
        if family == "poisson":
            rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
            return Poisson(rate).mask(mask), obs
        if family == "negative-binomial":
            expanded_sigma_neg_bin = context["expanded_sigma_neg_bin"]
            rate = jnp.exp(linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0))
            return NegativeBinomial2(mean=rate, concentration=expanded_sigma_neg_bin).mask(mask), obs
        if family == "binomial":
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return BinomialLogits(logits=rate, total_count=jnp.where(mask, exposure.astype(int), 0)).mask(mask), obs
        if family == "beta-binomial":
            sigma_beta_binomial = context["sigma_beta_binomial"]
            rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
            dist = BetaBinomial(
                concentration0=(1 - rate) * sigma_beta_binomial,
                concentration1=rate * sigma_beta_binomial,
                total_count=jnp.where(mask, exposure.astype(int), 0),
            )
            return dist.mask(mask), obs
        if family == "beta":
            sigma_beta = context["sigma_beta"]
            rate = jsci.special.expit(linear_predictor + jnp.where(mask, de_trend, 0))
            return BetaProportion(rate, jnp.where(mask, exposure, 1.0) * sigma_beta).mask(mask), obs
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexMaxBoundaryTVRFLVM likelihood builder")
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self._resolve_prior("lengthscale")
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k,))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        

        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(1, self.k))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(1, self.k))
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(2 * self.m)
        
        sigma_boundary_r = self._resolve_prior("sigma_boundary_r", sample_shape=(1, self.k))
        sigma_boundary_l = self._resolve_prior("sigma_boundary_l", sample_shape=(1, self.k))
        boundary_r_raw = self._resolve_prior("boundary_r", sample_shape=(2 * self.m, self.k))
        boundary_l_raw = self._resolve_prior("boundary_l", sample_shape=(2 * self.m, self.k))
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        sigma_beta = self._resolve_prior("sigma_beta")
        sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial")
        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))
        X = self._resolve_latent_X(sample_free_indices, sample_fixed_indices)


        psi_x = self._project_X(X, W=W, lengthscale=lengthscale)
        

        t_max = self._build_t_max_curve(
            psi_x,
            t_max_raw,
            sigma_t_max,
            offsets,
            prior,
            amplitude=5.0,
            offset_mode="additive",
            scale_inside=False,
            deterministic_amplitude=2.0,
        )
        c_max = self._build_c_max_curve(
            psi_x,
            c_max_raw,
            sigma_c_max,
            offsets,
            prior,
            scale_inside=False,
        )

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw ) * sigma_boundary_r + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x,boundary_l_raw) * sigma_boundary_l + offsets["boundary_l"])
        
        if prior:
            boundary_l = numpyro.deterministic("boundary_l_", boundary_l)
            boundary_r = numpyro.deterministic("boundary_r_", boundary_r)

        boundary_diff = boundary_l - boundary_r
        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        if prior:
            G = (make_convex_phi(t_r, L_time, M_time) - make_convex_phi(t_0, L_time, M_time))[None, None] + (t_0 - t_r) * phi_prime_t_max
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, self.k) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self._resolve_prior("beta", sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd

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
            sigma_beta=sigma_beta,
            sigma_beta_binomial=sigma_beta_binomial,
        )

        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())   

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 

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
    
    @staticmethod
    def solve_for_weights(A, rho, phi, X, b, spd):
        def loss_fn(A, rho, phi, X, b, s):
            A = A * s
            def residual(n, k):
                phi_nk = phi[n][None]  # shape (M,)
                A_k = A[..., k]          # (D, M)
                X_nk = X[n, k]      # (D, D)
                v = jnp.dot(phi_nk, A_k)    # (D,)

                return jnp.einsum("jd, dd, kd - > jk", v, X_nk, v) - b[n, k]

            # Vectorize residual over n and k
            r = jax.vmap(lambda n: jax.vmap(lambda k: residual(n, k))(jnp.arange(b.shape[-1])))(jnp.arange(b.shape[0]))
            penalty = jnp.sum(jnp.square(r))
            reg = jnp.sum(jnp.square(A))
            return reg + rho * penalty

        solver = LBFGS(fun=loss_fn, maxiter=500, implicit_diff=True, verbose=False)

        result = solver.run(A, rho, phi, X, b, spd)
        A_opt = result.params
        return A_opt

class ConvexMaxBoundaryARTVRFLVM(ConvexMaxBoundaryTVRFLVM):
    """
        model for time varying functional enforcing convexity in the shape parameters. specifies a max, 
        and boundary conditions
    """
    def __init__(self, latent_rank: int, rff_dim: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, rff_dim, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-1,1)
        self.prior["sigma_ar"] = InverseGamma(2, 1)
        # self.prior["sigma_ar"] = Type2Gumbel(alpha=.05, scale=.001)
        self.prior["beta_ar"] = Normal()
        self.prior["ar_0"] = Normal()
        self.prior["X_free"] = Normal()

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        ar = context["ar"]
        return self._build_linear_predictor(mu, k_indices, ar[k_indices])

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)
    
        
    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior", sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        lengthscale = self._resolve_prior("lengthscale")
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k,))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k,))
        spd = jnp.sqrt(diag_spectral_density(1, alpha_time, ls_deriv, L_time, M_time))
        t_0 = hsgp_params["t_0"]
        t_r = hsgp_params["t_r"]
        W = self._resolve_prior("W", sample_shape=(self.m, self.r))
        

        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(1, self.k))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(1, self.k))
        t_max_raw = self._resolve_prior("t_max_raw", sample_shape=(2 * self.m, self.k))
        c_max_raw = self._resolve_prior("c_max", sample_shape=(2 * self.m, self.k))
        
        sigma_boundary_r = self._resolve_prior("sigma_boundary_r", sample_shape=(1, self.k))
        sigma_boundary_l = self._resolve_prior("sigma_boundary_l", sample_shape=(1, self.k))
        boundary_r_raw = self._resolve_prior("boundary_r", sample_shape=(2 * self.m, self.k))
        boundary_l_raw = self._resolve_prior("boundary_l", sample_shape=(2 * self.m, self.k))
        if num_gaussians > 0:
            expanded_sigmas = self._sample_gaussian_sigmas(num_gaussians)
        sigma_beta = self._resolve_prior("sigma_beta")
        sigma_beta_binomial = self._resolve_prior("sigma_beta_binomial")
        X = self._resolve_latent_X(sample_free_indices, sample_fixed_indices)


        if num_neg_bins > 0:
            sigma_negative_binomial = self._resolve_prior("sigma_negative_binomial", sample_shape=(num_neg_bins,))
            expanded_sigma_neg_bin = jnp.tile(sigma_negative_binomial[:, None, None], (1, self.n, self.j))

        psi_x = self._project_X(X, W=W, lengthscale=lengthscale)
        

        t_max = self._build_t_max_curve(
            psi_x,
            t_max_raw,
            sigma_t_max,
            offsets,
            prior,
            amplitude=2.0,
            offset_mode="additive",
            scale_inside=False,
        )
        c_max = self._build_c_max_curve(
            psi_x,
            c_max_raw,
            sigma_c_max,
            offsets,
            prior,
            scale_inside=False,
        )

        boundary_r = c_max - jax.nn.softplus(make_psi_gamma(psi_x, boundary_r_raw ) * sigma_boundary_r + offsets["boundary_r"])
        boundary_l = c_max - jax.nn.softplus(make_psi_gamma(psi_x,boundary_l_raw) * sigma_boundary_l + offsets["boundary_l"])
        
        if prior:
            boundary_l = numpyro.deterministic("boundary_l_", boundary_l)
            boundary_r = numpyro.deterministic("boundary_r_", boundary_r)

        boundary_diff = boundary_l - boundary_r
        

        phi_prime_t_max = jax.vmap(lambda t: vmap_make_convex_phi_prime(t, L_time, M_time))(t_max)
        phi_t_max = jax.vmap(lambda t: vmap_make_convex_phi(t, L_time, M_time))(t_max)

        if prior:
            G = (make_convex_phi(t_r, L_time, M_time) - make_convex_phi(t_0, L_time, M_time))[None, None] + (t_0 - t_r) * phi_prime_t_max
            init_weights =  Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=((self.m * 2, M_time, self.k) ))
            weights = self.solve_for_weights(init_weights, 1e-1, psi_x, G, boundary_diff, spd)
            self.prior["beta"] = weights
        else:
            weights = self._resolve_prior("beta", sample_shape=(self.m * 2, M_time, self.k))
        weights *= spd

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
            sigma_beta=sigma_beta,
            sigma_beta_binomial=sigma_beta_binomial,
            ar=AR,
        )


        value = (mu[..., 0] - mu[..., -1]) - jnp.transpose(boundary_diff)
        logp = Normal(0, 0.1).log_prob(value)
        numpyro.factor(f"boundary_conditions", logp.sum())   

        value = (boundary_r - jnp.transpose(mu[..., -1]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_r", logp.sum()) 

        value = (boundary_l - jnp.transpose(mu[..., 0]))
        logp = Normal(0, .1).log_prob(value)
        numpyro.factor(f"boundary_conditions_l", logp.sum()) 


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
    
    @staticmethod
    def solve_for_weights(A, rho, phi, X, b, spd):
        def loss_fn(A, rho, phi, X, b, s):
            A = A * s
            def residual(n, k):
                phi_nk = phi[n][None]  # shape (M,)
                A_k = A[..., k]          # (D, M)
                X_nk = X[n, k]      # (D, D)
                v = jnp.dot(phi_nk, A_k)    # (D,)

                return jnp.einsum("jd, dd, kd - > jk", v, X_nk, v) - b[n, k]

            # Vectorize residual over n and k
            r = jax.vmap(lambda n: jax.vmap(lambda k: residual(n, k))(jnp.arange(b.shape[-1])))(jnp.arange(b.shape[0]))
            penalty = jnp.sum(jnp.square(r))
            reg = jnp.sum(jnp.square(A))
            return reg + rho * penalty

        solver = LBFGS(fun=loss_fn, maxiter=500, implicit_diff=True, verbose=False)

        result = solver.run(A, rho, phi, X, b, spd)
        A_opt = result.params
        return A_opt






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
        prior = (inference_method == "prior")
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
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
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
        if isinstance(prior_value, Distribution):
            dist_to_sample = dist_override if dist_override is not None else prior_value
            sample_kwargs = {}
            if sample_shape is not None:
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
        kernel = NUTS(self.model_fn, init_strategy=init_to_value(values=initial_values))
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
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        self.r = latent_rank 
        self.n, self.j, self.k = output_shape
        self.basis = basis ### basis for time dimension
        self.t = len(basis)
        self.prior = {}

    def _project_X(self, X: jnp.ndarray, *, W: jnp.ndarray = None, lengthscale: jnp.ndarray = None):
        return X

    def _build_family_distribution(self, family: str, linear_predictor, family_data: dict, **context):
        exposure = family_data["exposure"]
        de_trend = family_data["de_trend"]
        obs = family_data["Y"]
        mask = family_data["mask"]
        if family == "gaussian":
            expanded_sigmas = context["expanded_sigmas"]
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            dist = Normal(rate[mask], expanded_sigmas[mask] / exposure[mask])
            return dist, obs[mask]
        if family == "poisson":
            log_rate = linear_predictor[mask] + exposure[mask] + de_trend[mask]
            return Poisson(jnp.exp(log_rate)), obs[mask]
        if family == "negative-binomial":
            expanded_sigma_neg_bin = context["expanded_sigma_neg_bin"]
            log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
            return NegativeBinomial2(mean=jnp.exp(log_rate[mask]), concentration=expanded_sigma_neg_bin[mask]), obs[mask]
        if family == "binomial":
            rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return BinomialLogits(logits=rate[mask], total_count=exposure[mask].astype(int)), obs[mask]
        if family == "beta-binomial":
            expanded_sigma_beta_bin = context["expanded_sigma_beta_bin"]
            logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
            dist = BetaBinomial(
                concentration0=(1 - jsci.special.expit(logit_rate[mask])) * expanded_sigma_beta_bin[mask],
                concentration1=jsci.special.expit(logit_rate[mask]) * expanded_sigma_beta_bin[mask],
                total_count=exposure[mask].astype(int),
            )
            return dist, obs[mask]
        if family == "beta":
            expanded_sigma_beta = context["expanded_sigma_beta"]
            logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
            return BetaProportion(jsci.special.expit(logit_rate[mask]), expanded_sigma_beta[mask] * jnp.square(exposure[mask])), obs[mask]
        raise NotImplementedError(f"Unsupported family '{family}' for ConvexMaxTVLinearLVM likelihood builder")

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["sigma_beta"] = Uniform(0, 1)
        self.prior["sigma"] = InverseGamma(300, 6000)
        self.prior["sigma_negative_binomial"] = Exponential()
        self.prior["sigma_beta_binomial"] = Exponential()
        self.prior["lengthscale_deriv"] = HalfNormal(2.0)   # loosened: allows ls in [3,10] for smoother trajectories
        self.prior["alpha"] = HalfNormal(0.3)               # tightened: reduces spectral weight variance ~10x
        self.prior["t_max_raw"] = Normal()
        self.prior["c_max"] = Normal()
        self.prior["X_free"] = Normal()
        self.prior["sigma_c"] = HalfNormal(0.5)             # tightened: keeps c_max near debiased anchor
        self.prior["sigma_t"] = HalfNormal()
        self.prior["exit"] = Normal(0.0, 0.5)
        self.prior["entrance"] = Normal()
        self.prior["entrance_global_offset"] = Normal(jnp.log(2.0), 0.5)
        self.prior["exit_global_offset"] = Normal(jnp.log(10.0), 0.5)
        self.prior["exit_rate"] = Normal(0.0, 0.1)
        self.prior["sigma_entrance"] = HalfNormal(.2)
        # self.prior["t_offset"] = Uniform(-5, 5)
        # self.prior["c_offset"] = Normal(0, .1)

    def compute_survival_likelihood(self, X, offsets = {}) -> None:
        required_keys = ("entrance_times", "exit_times", "left_censor", "right_censor")
        if not all(key in offsets for key in required_keys):
            return

        entrance = self._resolve_prior("entrance", sample_shape=(self.r,))
        lc = jnp.ravel(offsets["left_censor"].astype(bool))
        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        observed_mask = (~lc).astype(entrance_times.dtype)
        observed_count = jnp.maximum(observed_mask.sum(), 1.0)
        empirical_log_mean = jnp.sum(jnp.log(entrance_times) * observed_mask) / observed_count
        sigma_entrance = self._resolve_prior("sigma_entrance")
        entrance_global_offset = self._resolve_prior("entrance_global_offset")
        entrance_raw = make_psi_gamma(X, entrance)
        entrance_loc = entrance_global_offset + entrance_raw + empirical_log_mean
        entrance_dist = LogNormal(entrance_loc, sigma_entrance + .35)
        entrance_latent_sampled = numpyro.sample("entrance_latent", entrance_dist)
        z_entrance = (jnp.log(entrance_times) - entrance_loc) / (sigma_entrance + .35)
        log_cdf_entrance = jsci.stats.norm.logcdf(z_entrance)
        log_pdf_entrance = entrance_dist.log_prob(entrance_times)
        with mask(mask=lc):
            numpyro.factor("log_lik_entrance_censored", log_cdf_entrance)
        with mask(mask=(~lc)):
            numpyro.factor("log_lik_entrance_observed", log_pdf_entrance)

        lc_float = lc.astype(entrance_times.dtype)
        entrance_latent = lc_float * entrance_latent_sampled + (1.0 - lc_float) * jax.lax.stop_gradient(entrance_times)

        exit = self._resolve_prior("exit", sample_shape=(self.r,))
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(self.r,))
        exit_global_offset = self._resolve_prior("exit_global_offset")
        exit_rate_raw = make_psi_gamma(X, exit_rate)[:, None] + exit_global_offset
        concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_raw)


        scale_min = 8
        scale_max = 15
        exit_raw = make_psi_gamma(X, exit)
        scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)[:, None]
        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        event_time = exit_times
        concentration_event = concentration.squeeze(-1)
        scale_event = scale.squeeze(-1)
        entry_effective = jnp.maximum(entrance_latent, 0.0)
        stop_effective = jnp.maximum(event_time, 0.0)
        has_exposure_window = stop_effective > entry_effective
        cumulative_H = jnp.where(
            has_exposure_window,
            jnp.power(stop_effective / scale_event, concentration_event)
            - jnp.power(entry_effective / scale_event, concentration_event),
            0.0,
        )
        log_h_event = (
            jnp.log(concentration_event)
            - concentration_event * jnp.log(scale_event)
            + (concentration_event - 1.0) * jnp.log(event_time)
        )

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)
    

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))
        
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(self.r)
        

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


        psi_x = self._project_X(X)
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.n, self.k))
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
        )
        
        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.r, M_time, self.k))
        weights *= spd_time.T[None]

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
            prior
        )
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
                exit_global_offset = sampled.get("exit_global_offset", 0.0)

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

                exit_raw = make_psi_gamma(X, exit_param)
                exit_rate_raw = make_psi_gamma(X, exit_rate)[:, None]
                concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_raw)
                scale = 8.0 + 7.0 * jax.nn.sigmoid(exit_raw + exit_global_offset)[:, None]

                event_time = exit_times
                concentration_event = concentration.squeeze(-1)
                scale_event = scale.squeeze(-1)
                entry_effective = jnp.maximum(entrance_latent, 0.0)
                stop_effective = jnp.maximum(event_time, 0.0)
                has_exposure_window = stop_effective > entry_effective
                cumulative_H = jnp.where(
                    has_exposure_window,
                    jnp.power(stop_effective / scale_event, concentration_event)
                    - jnp.power(entry_effective / scale_event, concentration_event),
                    0.0,
                )
                log_h_event = (
                    jnp.log(concentration_event)
                    - concentration_event * jnp.log(scale_event)
                    + (concentration_event - 1.0) * jnp.log(event_time)
                )

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
                exit_rate_raw_bad = _print_nonfinite("exit_rate_raw", jnp.ravel(exit_rate_raw))
                _print_nonfinite("concentration", concentration_event)
                _print_nonfinite("scale", scale_event)
                _print_nonfinite("log_h_event", log_h_event, mask=(~rc))
                _print_nonfinite("cumulative_H", cumulative_H)
                _print_nonfinite("log_lik_exit_censored", log_lik_exit_censored, mask=rc)
                exit_bad = _print_nonfinite("log_lik_exit_observed", log_lik_exit_event, mask=(~rc))

                print(
                    "[MAP DEBUG][SURV] quick ranges: "
                    f"entrance_loc=[{float(jnp.nanmin(entrance_loc)):.6g}, {float(jnp.nanmax(entrance_loc)):.6g}], "
                    f"sigma_eff={float(jnp.asarray(sigma_effective)):.6g}, "
                    f"event_time=[{float(jnp.nanmin(event_time)):.6g}, {float(jnp.nanmax(event_time)):.6g}], "
                    f"exit_rate_raw=[{float(jnp.nanmin(exit_rate_raw)):.6g}, {float(jnp.nanmax(exit_rate_raw)):.6g}], "
                    f"concentration=[{float(jnp.nanmin(concentration_event)):.6g}, {float(jnp.nanmax(concentration_event)):.6g}], "
                    f"scale=[{float(jnp.nanmin(scale_event)):.6g}, {float(jnp.nanmax(scale_event)):.6g}]"
                )

                if exit_rate_raw_bad:
                    print("[MAP DEBUG][SURV] exit_rate_raw bad-point snapshots:")
                    flat = jnp.ravel(exit_rate_raw)
                    for i in exit_rate_raw_bad[:5]:
                        print(
                            f"  idx={i}, exit_rate_raw={_fmt_value(flat[i])}, "
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
                            f"conc={_fmt_value(concentration_event[i])}, scale={_fmt_value(scale_event[i])}, "
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

                sigma_c_max = sampled["sigma_c"]
                sigma_t_max = sampled["sigma_t"]
                t_max_raw = sampled["t_max_raw"]
                c_max_raw = sampled["c_max"]
                X = sampled["X"]

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
                    de_trend = family_data["de_trend"]

                    _stats_text(f"{family}.linear_predictor(masked)", _qstats(linear_predictor[mask]))
                    _k_range_text(
                        f"{family}.linear_predictor(masked)",
                        linear_predictor,
                        k_axis=0,
                        k_indices=np.asarray(k_indices),
                    )

                    if family in ("poisson", "negative-binomial"):
                        log_rate = linear_predictor + jnp.where(mask, exposure, 0) + jnp.where(mask, de_trend, 0)
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
                        logit_rate = linear_predictor + jnp.where(mask, de_trend, 0)
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

class NaiveLinearLVM(ConvexMaxTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        del self.prior["sigma_t"]
        del self.prior["sigma_c"]
        del self.prior["t_max_raw"]
        del self.prior["c_max"]
        # del self.prior["t_offset"]
        del self.prior["beta"]
        del self.prior["alpha"]
        del self.prior["lengthscale_deriv"]
        del self.prior["exit_rate"]
        del self.prior["entrance"]
        del self.prior["exit"]
        self.prior["exit_scale_global_offset"] = Normal(0, .1)
        self.prior["c_offset"] = Normal(0, 1)
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

    def model_fn(self, data_set, hsgp_params, offsets={}, inference_method = "prior", sample_free_indices = jnp.array([]), sample_fixed_indices = jnp.array([])):
        prior = (inference_method == "prior")
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

        mu = jnp.repeat(self._resolve_prior("c_offset", sample_shape=(self.k, self.n, 1)), repeats = self.t, axis = -1)
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
        required_keys = ("entrance_times", "exit_times", "left_censor", "right_censor")
        if not all(key in offsets for key in required_keys):
            return
        lc = jnp.ravel(offsets["left_censor"].astype(bool))
        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        observed_mask = (~lc).astype(entrance_times.dtype)
        observed_count = jnp.maximum(observed_mask.sum(), 1.0)
        empirical_log_mean = jnp.sum(jnp.log(entrance_times) * observed_mask) / observed_count
        sigma_entrance = self._resolve_prior("sigma_entrance")
        entrance_global_offset = self._resolve_prior("entrance_global_offset", sample_shape=(self.n,))
        entrance_loc = entrance_global_offset + empirical_log_mean
        entrance_dist = LogNormal(entrance_loc, sigma_entrance + .35)
        entrance_latent_sampled = numpyro.sample("entrance_latent", entrance_dist)
        z_entrance = (jnp.log(entrance_times) - entrance_loc) / (sigma_entrance + .35)
        log_cdf_entrance = jsci.stats.norm.logcdf(z_entrance)
        log_pdf_entrance = entrance_dist.log_prob(entrance_times)
        with mask(mask=lc):
            numpyro.factor("log_lik_entrance_censored", log_cdf_entrance)
        with mask(mask=(~lc)):
            numpyro.factor("log_lik_entrance_observed", log_pdf_entrance)

        lc_float = lc.astype(entrance_times.dtype)
        entrance_latent = lc_float * entrance_latent_sampled + (1.0 - lc_float) * jax.lax.stop_gradient(entrance_times)


        exit_global_offset = self._resolve_prior("exit_global_offset", sample_shape=(self.n,1))
        exit_rate_raw = exit_global_offset
        concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_raw)
        scale_min = 8
        scale_max = 15
        exit_scale_raw = self._resolve_prior("exit_scale_global_offset", sample_shape=(self.n,1))
        scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_scale_raw)[:, None]
        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        event_time = exit_times
        concentration_event = concentration.squeeze(-1)
        scale_event = scale.squeeze(-1)
        entry_effective = jnp.maximum(entrance_latent, 0.0)
        stop_effective = jnp.maximum(event_time, 0.0)
        has_exposure_window = stop_effective > entry_effective
        cumulative_H = jnp.where(
            has_exposure_window,
            jnp.power(stop_effective / scale_event, concentration_event)
            - jnp.power(entry_effective / scale_event, concentration_event),
            0.0,
        )
        log_h_event = (
            jnp.log(concentration_event)
            - concentration_event * jnp.log(scale_event)
            + (concentration_event - 1.0) * jnp.log(event_time)
        )

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)

class ConvexMaxARTVLinearLVM(ConvexMaxTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis) -> None:
        super().__init__(latent_rank, output_shape, basis)
    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["rho_ar"] = Uniform(-.5, .5)
        self.prior["sigma_ar"] = HalfNormal(1)
        self.prior["beta_ar"] = StudentT(3)
        self.prior["ar_0"] = Normal(0, 1)

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        ar = context["ar"]
        return self._build_linear_predictor(mu, k_indices, ar[k_indices])

    def _build_ar_process(self, sigma_ar=None, rho_ar=None, z=None, ar_0=None):
        return self._compute_ar_process_from_parameters(sigma_ar, rho_ar, z, ar_0)
    

    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))
        
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(self.r)
        

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


        psi_x = self._project_X(X)
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.n, self.k))
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
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.r, M_time, self.k))
        weights *= spd_time.T[None]

        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
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
            expanded_taus=None,
            expanded_sigma_beta=expanded_sigma_beta if num_beta > 0 else None,
            expanded_sigma_beta_bin=expanded_sigma_beta_bin if num_beta_bins > 0 else None,
            expanded_sigma_neg_bin=expanded_sigma_neg_bin if num_neg_bins > 0 else None,
            ar=AR,
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


class ConvexMaxInjuryTVLinearLVM(ConvexMaxARTVLinearLVM):
    def __init__(self, latent_rank: int, output_shape: tuple, basis, injury_rank: int, num_injury_types: int) -> None:
        super().__init__(latent_rank, output_shape, basis)
        self.i = num_injury_types
        self.p = injury_rank

    def initialize_priors(self, *args, **kwargs) -> None:
        super().initialize_priors(*args, **kwargs)
        self.prior["injury_factor"] = Normal(0, 1)
        self.prior["injury_loading"] = Normal(0, 1)
        self.prior["injury_global_offset"] = Normal(0, 1)
        self.prior["injury_raw"] = Normal()
        self.prior["sigma_injury"] = HalfNormal()
        self.prior["sigma_c"] = HalfNormal()
        self.prior["sigma_t"] = HalfNormal()
        self.prior["injury_exit_loading"] = Normal(0, 1)
        self.prior["injury_exit_global_offset"] = Normal(0, 1)
        self.prior["sigma_injury_exit"] = HalfNormal()
        self.prior["injury_exit_raw"] = Normal()

    def compute_survival_likelihood(self, X, injury_factor, offsets = {}) -> None:
        required_keys = ("entrance_times", "exit_times", "left_censor", "right_censor", "injury_indicator", "injury_type")
        if not all(key in offsets for key in required_keys):
            return

        entrance = self._resolve_prior("entrance", sample_shape=(self.r,))
        lc = jnp.ravel(offsets["left_censor"].astype(bool))
        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_valid = jnp.isfinite(entrance_times)
        observed_mask = (~lc & entrance_valid).astype(entrance_times.dtype)
        observed_count = jnp.maximum(observed_mask.sum(), 1.0)
        empirical_log_mean = jnp.sum(jnp.log(entrance_times) * observed_mask) / observed_count
        sigma_entrance = self._resolve_prior("sigma_entrance")
        entrance_global_offset = self._resolve_prior("entrance_global_offset")
        entrance_raw = make_psi_gamma(X, entrance)
        entrance_loc = entrance_global_offset + entrance_raw + empirical_log_mean
        entrance_dist = LogNormal(entrance_loc, sigma_entrance + .35)
        entrance_latent_sampled = numpyro.sample("entrance_latent", entrance_dist)
        z_entrance = (jnp.log(entrance_times) - entrance_loc) / (sigma_entrance + .35)
        log_cdf_entrance = jsci.stats.norm.logcdf(z_entrance)
        log_pdf_entrance = entrance_dist.log_prob(entrance_times)
        with mask(mask=(lc & entrance_valid)):
            numpyro.factor("log_lik_entrance_censored", log_cdf_entrance)
        with mask(mask=(~lc & entrance_valid)):
            numpyro.factor("log_lik_entrance_observed", log_pdf_entrance)

        lc_float = lc.astype(entrance_times.dtype)
        entrance_latent = lc_float * entrance_latent_sampled + (1.0 - lc_float) * jax.lax.stop_gradient(entrance_times)


        exit = self._resolve_prior("exit", sample_shape=(self.r,))
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(self.r,))
        exit_raw = make_psi_gamma(X, exit)
        injury_exit_loading = self._resolve_prior("injury_exit_loading", sample_shape=(self.p,))
        injury_exit_global_offset = self._resolve_prior("injury_exit_global_offset")
        sigma_injury_exit = self._resolve_prior("sigma_injury_exit")
        injury_exit_raw = (
            injury_exit_global_offset
            + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)[None, None, :]
            + self._resolve_prior("injury_exit_raw", sample_shape=(self.n, self.t, self.i)) * sigma_injury_exit
        )

        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        if injury_indicator.ndim == 3:
            injury_indicator = injury_indicator[0]
        if injury_type.ndim == 3:
            injury_type = injury_type[0]

        injury_effect_exit = jnp.take_along_axis(
            jnp.concatenate([jnp.zeros_like(injury_indicator)[..., None], injury_exit_raw], -1),
            injury_type[..., None],
            -1,
        ).squeeze(-1)

        exit_rate_base = make_psi_gamma(X, exit_rate)[:, None]
        exit_global_offset = self._resolve_prior("exit_global_offset")
        exit_rate_raw = exit_rate_base + exit_global_offset + injury_effect_exit
        concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_raw)
        scale_min = 8
        scale_max = 15
        scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)[:, None]
        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        interval_starts = jnp.arange(self.t, dtype=exit_times.dtype)[None, :]
        interval_ends = interval_starts + 1.0
        entry = entrance_latent[:, None]
        stop = exit_times[:, None]
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop)
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.where(valid_seg, seg_start, 1.0)
        seg_end_safe = jnp.where(valid_seg, seg_end, 1.0)
        log_scale = jnp.log(scale)
        seg_start_exp = concentration * (jnp.log(seg_start_safe) - log_scale)
        seg_end_exp = concentration * (jnp.log(seg_end_safe) - log_scale)
        valid_seg_float = valid_seg.astype(exit_times.dtype)
        delta_H = valid_seg_float * (jnp.exp(seg_end_exp) - jnp.exp(seg_start_exp))
        cumulative_H = delta_H.sum(axis=-1)

        event_time = exit_times
        event_interval = jnp.clip(jnp.floor(event_time).astype(jnp.int32), 0, self.t - 1)
        concentration_event = jnp.take_along_axis(concentration, event_interval[:, None], axis=1).squeeze(-1)
        scale_event = scale.squeeze(-1)

        log_h_event = (
            jnp.log(concentration_event)
            - concentration_event * jnp.log(scale_event)
            + (concentration_event - 1.0) * jnp.log(event_time)
        )

        log_lik_exit_event = log_h_event - cumulative_H
        log_lik_exit_censored = -cumulative_H
        with mask(mask=rc):
            numpyro.factor("log_lik_exit_censored", log_lik_exit_censored)
        with mask(mask=(~rc)):
            numpyro.factor("log_lik_exit_observed", log_lik_exit_event)

    def _compute_family_linear_predictor(self, family: str, mu, family_data: dict, **context):
        k_indices = family_data["indices"]
        return self._build_linear_predictor(mu, k_indices)
        


    def model_fn(self, data_set, hsgp_params, offsets = {}, inference_method:str = "prior",sample_free_indices: jnp.ndarray = jnp.array([]), sample_fixed_indices: jnp.ndarray = jnp.array([])) -> None:
        prior = (inference_method == "prior")
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0
        phi_time  = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]
        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = 3 +  self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(jnp.sqrt(jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(alpha_time, ls_deriv)))
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))
        
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(self.r)
        

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


        psi_x = self._project_X(X)
        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.n, self.k))
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
        )

        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.r, M_time, self.k))
        weights *= spd_time.T[None]


        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0") * (sigma_ar / jnp.sqrt(1 - rho_ar ** 2))
        AR = self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)
        injury_loading = self._resolve_prior("injury_loading", sample_shape=(self.k, self.p))
        injury_factor = self._resolve_prior("injury_factor", sample_shape=(self.i, self.p))
        injury_global_offset = self._resolve_prior("injury_global_offset", sample_shape=(self.k,))
        sigma_injury = self._resolve_prior("sigma_injury", sample_shape=(self.k,))
        injury_mean_prior = jnp.einsum("ip, kp -> ki", injury_factor, injury_loading)
        injury_raw = self._resolve_prior("injury_raw", sample_shape=(self.k, self.n, self.t, self.i))
        injury_effect_raw = (
            injury_global_offset[:, None, None, None]
            + injury_mean_prior[:, None, None, :]
            + injury_raw * sigma_injury[:, None, None, None]
        )
        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"] 
        injury_effect = jnp.take_along_axis(jnp.concatenate([jnp.zeros_like(injury_indicator)[..., None], injury_effect_raw ], -1), injury_type[..., None], -1).squeeze(-1) 
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
        
        mu = self._build_mu_from_base(mu_base, prior, injury_effect, AR)
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
        # beta_0 is (k, i) in this model -- nullify per-player noise sites from parent
        self.prior["injury_raw"] = None
        self.prior["sigma_injury"] = None

    def compute_survival_likelihood(self, X, injury_factor, offsets={}) -> None:
        required_keys = ("entrance_times", "exit_times", "left_censor", "right_censor", "injury_indicator", "injury_type")
        if not all(key in offsets for key in required_keys):
            return

        # ---- Entrance model (unchanged from parent) ---- #
        entrance = self._resolve_prior("entrance", sample_shape=(self.r,))
        lc = jnp.ravel(offsets["left_censor"].astype(bool))
        entrance_times = jnp.ravel(jnp.asarray(offsets["entrance_times"]))
        entrance_valid = jnp.isfinite(entrance_times)
        observed_mask = (~lc & entrance_valid).astype(entrance_times.dtype)
        observed_count = jnp.maximum(observed_mask.sum(), 1.0)
        empirical_log_mean = jnp.sum(jnp.log(entrance_times) * observed_mask) / observed_count
        sigma_entrance = self._resolve_prior("sigma_entrance")
        entrance_global_offset = self._resolve_prior("entrance_global_offset")
        entrance_raw = make_psi_gamma(X, entrance)
        entrance_loc = entrance_global_offset + entrance_raw + empirical_log_mean
        entrance_dist = LogNormal(entrance_loc, sigma_entrance + .35)
        entrance_latent_sampled = numpyro.sample("entrance_latent", entrance_dist)
        z_entrance = (jnp.log(entrance_times) - entrance_loc) / (sigma_entrance + .35)
        log_cdf_entrance = jsci.stats.norm.logcdf(z_entrance)
        log_pdf_entrance = entrance_dist.log_prob(entrance_times)
        with mask(mask=(lc & entrance_valid)):
            numpyro.factor("log_lik_entrance_censored", log_cdf_entrance)
        with mask(mask=(~lc & entrance_valid)):
            numpyro.factor("log_lik_entrance_observed", log_pdf_entrance)

        lc_float = lc.astype(entrance_times.dtype)
        entrance_latent = lc_float * entrance_latent_sampled + (1.0 - lc_float) * jax.lax.stop_gradient(entrance_times)

        # ---- Exit hazard: decayed injury effect ---- #
        exit = self._resolve_prior("exit", sample_shape=(self.r,))
        exit_rate = self._resolve_prior("exit_rate", sample_shape=(self.r,))
        exit_raw = make_psi_gamma(X, exit)

        injury_exit_loading = self._resolve_prior("injury_exit_loading", sample_shape=(self.p,))
        injury_exit_global_offset = self._resolve_prior("injury_exit_global_offset")
        sigma_injury_exit = self._resolve_prior("sigma_injury_exit")

        injury_indicator = offsets["injury_indicator"]
        injury_type = offsets["injury_type"]
        if injury_indicator.ndim == 3:
            injury_indicator = injury_indicator[0]
        if injury_type.ndim == 3:
            injury_type = injury_type[0]

        # beta0_exit: initial exit-hazard shock — (n, i)
        # Reuses "injury_exit_raw" site at shape (n, i) instead of parent's (n, t, i)
        injury_exit_raw_beta0 = self._resolve_prior("injury_exit_raw", sample_shape=(self.n, self.i))
        beta0_exit = (
            injury_exit_global_offset
            + jnp.einsum("ip,p->i", injury_factor, injury_exit_loading)[None, :]  # (1, i)
            + injury_exit_raw_beta0 * sigma_injury_exit
        )  # (n, i)

        # lambda_exit: exit-hazard decay rate — (i,), constrained > 0 via softplus
        lambda_exit_global_offset = self._resolve_prior("lambda_exit_global_offset", sample_shape=(self.i,))
        lambda_exit = jax.nn.softplus(lambda_exit_global_offset)  # (i,), strictly > 0

        # Time since injury: delta_t[n, t] = max(t - T0_n, 0)
        t0_index = jnp.argmax(injury_indicator, axis=-1).astype(jnp.float32)  # (n,)
        t_grid = jnp.arange(self.t, dtype=jnp.float32)                        # (t,)
        delta_t = jnp.maximum(t_grid[None, :] - t0_index[:, None], 0.0)       # (n, t)

        # Decayed exit effect: beta0 * exp(-lambda * delta_t) -> (n, t, i)
        exit_decay_factor = jnp.exp(-lambda_exit[None, None, :] * delta_t[:, :, None])  # (n, t, i)
        injury_exit_raw_decayed = beta0_exit[:, None, :] * exit_decay_factor             # (n, t, i)

        # Select active injury type; zeros at index 0 handle uninjured players
        injury_effect_exit = jnp.take_along_axis(
            jnp.concatenate([jnp.zeros_like(injury_indicator)[..., None], injury_exit_raw_decayed], -1),
            injury_type[..., None],
            -1,
        ).squeeze(-1)  # (n, t)

        # ---- Weibull hazard (unchanged from parent) ---- #
        exit_rate_base = make_psi_gamma(X, exit_rate)[:, None]
        exit_global_offset = self._resolve_prior("exit_global_offset")
        exit_rate_raw = exit_rate_base + exit_global_offset + injury_effect_exit
        concentration = 1.0 + 2.0 * jax.nn.sigmoid(exit_rate_raw)
        scale_min = 8
        scale_max = 15
        scale = scale_min + (scale_max - scale_min) * jax.nn.sigmoid(exit_raw)[:, None]
        rc = jnp.ravel(offsets["right_censor"].astype(bool))
        exit_times = jnp.ravel(jnp.asarray(offsets["exit_times"]))

        interval_starts = jnp.arange(self.t, dtype=exit_times.dtype)[None, :]
        interval_ends = interval_starts + 1.0
        entry = entrance_latent[:, None]
        stop = exit_times[:, None]
        seg_start = jnp.maximum(interval_starts, entry)
        seg_end = jnp.minimum(interval_ends, stop)
        valid_seg = seg_end > seg_start
        seg_start_safe = jnp.where(valid_seg, seg_start, 1.0)
        seg_end_safe = jnp.where(valid_seg, seg_end, 1.0)
        log_scale = jnp.log(scale)
        seg_start_exp = concentration * (jnp.log(seg_start_safe) - log_scale)
        seg_end_exp = concentration * (jnp.log(seg_end_safe) - log_scale)
        valid_seg_float = valid_seg.astype(exit_times.dtype)
        delta_H = valid_seg_float * (jnp.exp(seg_end_exp) - jnp.exp(seg_start_exp))
        cumulative_H = delta_H.sum(axis=-1)

        event_time = exit_times
        event_interval = jnp.clip(jnp.floor(event_time).astype(jnp.int32), 0, self.t - 1)
        concentration_event = jnp.take_along_axis(concentration, event_interval[:, None], axis=1).squeeze(-1)
        scale_event = scale.squeeze(-1)

        log_h_event = (
            jnp.log(concentration_event)
            - concentration_event * jnp.log(scale_event)
            + (concentration_event - 1.0) * jnp.log(event_time)
        )

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
    ) -> None:
        prior = inference_method == "prior"
        num_gaussians = data_set["gaussian"]["Y"].shape[0] if "gaussian" in data_set else 0
        num_neg_bins = data_set["negative-binomial"]["Y"].shape[0] if "negative-binomial" in data_set else 0
        num_beta = data_set["beta"]["Y"].shape[0] if "beta" in data_set else 0
        num_beta_bins = data_set["beta-binomial"]["Y"].shape[0] if "beta-binomial" in data_set else 0

        phi_time = hsgp_params["phi_x_time"]
        L_time = hsgp_params["L_time"]
        M_time = hsgp_params["M_time"]
        shifted_x_time = hsgp_params["shifted_x_time"]

        alpha_time = self._resolve_prior("alpha", sample_shape=(self.k, 1))
        ls_deriv = 3 + self._resolve_prior("lengthscale_deriv", sample_shape=(self.k, 1))
        spd_time = jnp.squeeze(
            jax.vmap(lambda alpha, ls: diag_spectral_density(1, alpha, ls, L_time, M_time))(
                alpha_time, ls_deriv
            )
        )
        sigma_c_max = self._resolve_prior("sigma_c", sample_shape=(self.k,))
        sigma_t_max = self._resolve_prior("sigma_t", sample_shape=(self.k,))
        t_max_raw, c_max_raw = self._sample_max_raw_parameters(self.r)

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
        psi_x = self._project_X(X)

        t_offset = self._resolve_prior("t_offset", sample_shape=(self.n, self.k))
        if t_offset is None:
            t_offset = offsets["t_max"]
        c_offset = self._resolve_prior("c_offset", sample_shape=(self.n, self.k))
        if c_offset is None:
            c_offset = offsets["c_max"]

        t_max, c_max = self._build_max_curves(
            psi_x, t_max_raw, c_max_raw, sigma_t_max, sigma_c_max, t_offset, c_offset, prior
        )
        phi_t_max, phi_prime_t_max = self._compute_phi_at_max(t_max, L_time, M_time)

        weights = self._resolve_prior("beta", sample_shape=(self.r, M_time, self.k))
        weights *= spd_time.T[None]

        sigma_ar = self._resolve_prior("sigma_ar", sample_shape=(self.k, 1))
        rho_ar = self._resolve_prior("rho_ar", sample_shape=(self.k, 1))
        z = self._resolve_prior("beta_ar", sample_shape=(self.j, self.k, self.n))
        ar_0 = (
            self._resolve_prior("ar_0", sample_shape=(self.k, self.n), site_name="AR_0")
            * (sigma_ar / jnp.sqrt(1 - rho_ar**2))
        )
        AR = self._build_ar_process(sigma_ar=sigma_ar, rho_ar=rho_ar, z=z, ar_0=ar_0)

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
        mu = self._build_mu_from_base(mu_base, prior, injury_effect, AR)
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
