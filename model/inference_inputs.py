"""Shared, model-generic construction of the data + model + model_args used for inference.

`build_inference_inputs(args)` is a faithful port of the setup block in `main.py` (the span that
loads the data, builds the survival/FDA datasets, dispatches the model class, computes the offsets,
the HSGP basis, and assembles `model_args`). It is the single entry point used by `prior_check.py`
so the prior-predictive diagnostics run against the EXACT same inputs the real fit uses.

`apply_prior_knobs(model, knobs)` overrides entries of `model.prior` after `initialize_priors`,
model-agnostically (every model class exposes a `self.prior` dict), so prior knobs can be tuned
from config without editing each class's `initialize_priors`.

NOTE: this mirrors the setup in `main.py`; keep the two in sync. The inference-specific pieces
(MAP/MCMC param loading, the AR MAP-reconstruction debug block) are intentionally NOT included —
they are not needed to draw from the prior predictive.
"""
from dataclasses import dataclass, field
from typing import Any

import re
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from config.config_utils import parse_metrics
from model.hsgp import diag_spectral_density, make_psi_gamma, vmap_make_convex_phi, vmap_make_convex_phi_prime, sqrt_eigenvalues
from model.model_utils import compute_priors, apply_detrend_for_offsets, compute_linear_predictor_mean_offsets
from data.data_utils import create_fda_data, create_surv_data, create_validation_mask
from model.models import (
    ConvexMaxInjuryTVLinearLVM, ConvexMaxTVLinearLVM, ConvexMaxDecayInjuryTVLinearLVM,
    ConvexMaxTVRFLVM, NaiveLinearLVM, ConvexMaxARTVRFLVM, ConvexMaxARTVLinearLVM,
    ConvexMaxLinearTrendTVLinearLVM, ConvexMaxLKJTVLinearLVM, ConvexMaxARLKJTVLinearLVM,
    ConvexMaxTVCosineLVM, ConvexMaxARTVCosineLVM,
    ConvexMaxRFFTVLinearLVM, ConvexMaxARRFFTVLinearLVM, ConvexMaxInjuryRFFTVLinearLVM,
    TVLinearLVM, TVLinearLVM_AR, LKJTVLinearLVM, LKJTVLinearLVM_AR,
)


@dataclass
class InferenceInputs:
    model: Any
    model_args: dict
    data: Any
    metrics: list
    metric_output: list
    exposure_list: list
    basis: Any
    covariate_X: Any
    Y: Any
    exposures: Any
    masks: Any
    de_trend: Any
    distribution_indices: dict
    age_min: int
    age_max: int
    names: list
    model_dir: str
    holdout_indices_path: str | None
    scale_values: Any = None
    extras: dict = field(default_factory=dict)


# Knobs that are read as instance ATTRIBUTES inside initialize_priors (not via self.prior),
# so they must be set on the model BEFORE initialize_priors is called.
_ATTRIBUTE_KNOBS = {"x_latent_df", "lkj_concentration"}

# Priors that are sampled with sample_shape=(k, 1) and fed to the per-metric spectral-density
# vmap (alpha = GP amplitude, lengthscale_deriv = derivative lengthscale). A per-metric constant
# array (k,) or a scalar broadcast must be reshaped to (k, 1) to match that axis. Distribution
# values are left untouched (resolved at (k, 1) by _resolve_prior).
_COL_SHAPED_KNOBS = {"alpha", "lengthscale_deriv"}


def _parse_scalar(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _parse_knob_value(v):
    """Scalar ('0.5') or distribution spec 'Dist:a,b' (e.g. HalfNormal:1.0, LogNormal:1.0986,1.2)."""
    v = v.strip()
    if ":" in v:
        name, _, rest = v.partition(":")
        if name.strip().isalpha():
            return {"dist": name.strip(), "args": [float(x) for x in rest.split(",") if x.strip() != ""]}
    return _parse_scalar(v)


def knobs_from_overrides(overrides):
    """Parse repeated --set strings into a knobs dict. Supports key=value (scalar/dist) and
    key@metric=value (per-metric override, e.g. sigma_c@usg=0.3). RE toggles: use_*_re=0/1.

    A plain `key=value` and per-metric `key@metric=value` for the SAME key are MERGED regardless
    of order: the scalar becomes the per-metric `default` (the value for unlisted metrics) and the
    @-entries become the per-metric overrides. (Previously the later assignment clobbered the
    earlier, so `alpha=1.0` after `alpha@obpm=0.1` silently dropped the per-metric override.)"""
    knobs = {}
    for item in overrides:
        k, _, v = item.partition("=")
        k = k.strip()
        if "@" in k:
            base, _, metric = k.partition("@")
            base = base.strip()
            entry = knobs.get(base)
            if not (isinstance(entry, dict) and "per_metric" in entry):
                # Promote any existing scalar default into the new per-metric spec.
                new_entry = {"per_metric": {}}
                if entry is not None and np.isscalar(entry):
                    new_entry["default"] = entry
                entry = new_entry
                knobs[base] = entry
            entry["per_metric"][metric.strip()] = _parse_scalar(v.strip())
        else:
            val = _parse_knob_value(v.strip())
            existing = knobs.get(k)
            # If per-metric overrides already exist for this key, keep them and set the scalar
            # as their `default` rather than overwriting the whole spec.
            if isinstance(existing, dict) and "per_metric" in existing and np.isscalar(val):
                existing["default"] = val
            else:
                knobs[k] = val
    return knobs


def _build_knob_value(spec):
    """A knob value is either a fixed scalar/array, or a distribution spec
    {"dist": "LogNormal", "args": [...], "kwargs": {...}} -> numpyro Distribution."""
    if isinstance(spec, dict) and "dist" in spec:
        cls = getattr(dist, spec["dist"])
        return cls(*spec.get("args", []), **spec.get("kwargs", {}))
    return spec


def _build_per_metric_array(spec, metrics, model):
    """Per-metric CONSTANT scale: {"per_metric": {metric: value, ...}, "default": value} ->
    length-k array (metric order = `metrics`). Default for unspecified metrics is spec['default']
    or the model's current scalar prior value (else 1.0). Distribution-per-metric (batched) is built
    here too when 'default'/values are dist specs of the SAME family with per-metric args."""
    pm = spec["per_metric"]
    # default: explicit, else current scalar prior, else 1.0
    if "default" in spec:
        default = spec["default"]
    else:
        cur = model.prior.get(_metric_default_key(spec), None)
        default = float(cur) if np.isscalar(cur) else 1.0
    if any(isinstance(v, dict) and "dist" in v for v in list(pm.values()) + [default]):
        raise ValueError("per-metric distribution priors not supported yet; use per-metric constants")
    arr = np.full(len(metrics), float(default))
    for m, v in pm.items():
        if m not in metrics:
            raise ValueError(f"per_metric knob references unknown metric '{m}'")
        arr[metrics.index(m)] = float(v)
    return jnp.asarray(arr)


def _build_per_metric_dist(spec, metrics, model):
    """Per-metric distribution of a SINGLE family, e.g.
        {"dist": "HalfNormal", "per_metric": {"obpm": 0.1, "dbpm": 0.1}, "default": 1.0}
    The per-metric value sets the distribution's FIRST positional arg (e.g. HalfNormal scale) as a
    length-k batched parameter; any `args` are shared, trailing fixed positional args. For col-shaped
    knobs (alpha / lengthscale_deriv) the parameter is reshaped to (k, 1) so the sampled site is
    (k, 1) with per-metric parameters. The marginal family is identical across metrics; only the
    parameter varies — and _resolve_prior samples the already-batched dist without re-expanding."""
    cls = getattr(dist, spec["dist"])
    arr = _build_per_metric_array(spec, metrics, model)   # length-k float array of the varying param
    if spec.get("_key") in _COL_SHAPED_KNOBS:
        arr = arr.reshape(model.k, 1)
    return cls(jnp.asarray(arr), *spec.get("args", []))


def _metric_default_key(spec):
    return spec.get("_key")


def apply_prior_knobs(model, knobs: dict | None, metrics=None):
    """Override entries of model.prior after initialize_priors (model-agnostic).
    A knob value may be a scalar, a distribution spec {dist,args}, or a per-metric spec
    {"per_metric": {metric: value}, "default": value} -> length-k constant array (needs `metrics`)."""
    if not knobs:
        return
    for key, spec in knobs.items():
        if key in _ATTRIBUTE_KNOBS or key.startswith("use_"):
            continue  # attribute knobs / RE toggles are applied directly, not via model.prior
        if isinstance(spec, dict) and "per_metric" in spec:
            if metrics is None:
                raise ValueError("per-metric knob requires `metrics`")
            spec = {**spec, "_key": key}
            if "dist" in spec:
                # per-metric DISTRIBUTION (same family, per-metric parameter) -> batched dist
                val = _build_per_metric_dist(spec, list(metrics), model)
            else:
                val = _build_per_metric_array(spec, list(metrics), model)
                if key in _COL_SHAPED_KNOBS:
                    val = val.reshape(model.k, 1)  # (k,1) to match the spectral-density vmap axis
        else:
            val = _build_knob_value(spec)
            # alpha / lengthscale_deriv are per-metric shaped (k, 1); a scalar knob must broadcast
            # (feeds a vmap over the metric axis in the spectral density). Distribution-valued knobs
            # are left as-is and sampled at (k, 1) by _resolve_prior.
            if key in _COL_SHAPED_KNOBS and np.isscalar(val):
                val = jnp.full((model.k, 1), float(val))
        model.prior[key] = val


def fda_injury_flag(model_name: str, injury: bool) -> bool:
    """The injury flag passed to create_fda_data. The non-convex GPLVM (tvlinearlvm without 'convex')
    never masks injury observations; every other family follows the run's `injury` setting."""
    return False if ("tvlinearlvm" in model_name and "convex" not in model_name) else injury


def dispatch_model(model_name, *, latent_rank, output_shape, basis, player_covariates=None,
                   injury=False, num_injury_types=0, rff_dim=0, prior_knobs=None):
    """SINGLE SOURCE for model-name -> model-class dispatch, shared by build_inference_inputs,
    main.py, and model_export. Returns the instantiated model with the per-player RE toggles set.
    Data creation (create_fda_data) stays with the caller; this only selects + builds the class."""
    prior_knobs = prior_knobs or {}
    common = dict(latent_rank=latent_rank, output_shape=output_shape, basis=basis, player_covariates=player_covariates)
    if "tvrflvm" in model_name:
        if "convex" in model_name and "max" in model_name:
            # RFF leaves of the full-featured ConvexMaxTVLinearLVM family (structured prior, REs,
            # compute_curves, AR/calendar trend) — see ConvexMaxRFFTVLinearLVM. Same treatment as the
            # "linear" branch below: pass player_covariates + rff_dim and set the per-player RE toggles.
            if injury and ("injury" in model_name):
                model = ConvexMaxInjuryRFFTVLinearLVM(
                    latent_rank=latent_rank, rff_dim=rff_dim, output_shape=output_shape,
                    basis=basis, player_covariates=player_covariates,
                    injury_rank=5, num_injury_types=num_injury_types)
            else:
                cls = ConvexMaxARRFFTVLinearLVM if "AR" in model_name else ConvexMaxRFFTVLinearLVM
                model = cls(latent_rank=latent_rank, rff_dim=rff_dim, output_shape=output_shape,
                            basis=basis, player_covariates=player_covariates)
            _re_default = (not injury) and ("linear_trend" not in model_name)
            model.use_c_offset_re = _re_default
            model.use_t_offset_re = _re_default
            model.use_curve_re = _re_default
            for _re in ("use_c_offset_re", "use_t_offset_re", "use_curve_re"):
                if _re in prior_knobs:
                    setattr(model, _re, bool(prior_knobs[_re]))
            return model
        raise ValueError("Model not implemented")
    if "naive" in model_name:
        return NaiveLinearLVM(latent_rank=latent_rank, output_shape=output_shape, basis=basis)
    if "tvlinearlvm" in model_name and "convex" not in model_name:
        if "lkj" in model_name:
            cls = LKJTVLinearLVM_AR if "AR" in model_name else LKJTVLinearLVM
        else:
            cls = TVLinearLVM_AR if "AR" in model_name else TVLinearLVM
        return cls(**common)
    if "linear" in model_name:
        if "lkj" in model_name:
            model = (ConvexMaxARLKJTVLinearLVM if "AR" in model_name else ConvexMaxLKJTVLinearLVM)(**common)
        elif "cosine" in model_name:
            model = (ConvexMaxARTVCosineLVM if "AR" in model_name else ConvexMaxTVCosineLVM)(**common)
        else:
            model = ConvexMaxTVLinearLVM(**common)
            if injury and ("injury" in model_name):
                model = ConvexMaxInjuryTVLinearLVM(**common, injury_rank=5, num_injury_types=num_injury_types)
                if "decay" in model_name:
                    model = ConvexMaxDecayInjuryTVLinearLVM(**common, injury_rank=5, num_injury_types=num_injury_types)
            elif "linear_trend" in model_name:
                model = ConvexMaxLinearTrendTVLinearLVM(**common)
            elif "AR" in model_name:
                model = ConvexMaxARTVLinearLVM(**common)
        # Per-player level/peak-age/curvature REs: on for the plain base + AR only; off for
        # injury/causal/linear_trend. prior_knobs can override (e.g. use_c_offset_re=0).
        _re_default = (not injury) and ("linear_trend" not in model_name)
        model.use_c_offset_re = _re_default
        model.use_t_offset_re = _re_default
        model.use_curve_re = _re_default
        for _re in ("use_c_offset_re", "use_t_offset_re", "use_curve_re"):
            if _re in prior_knobs:
                setattr(model, _re, bool(prior_knobs[_re]))
        return model
    raise ValueError("Model not implemented")


def build_inference_inputs(args: dict) -> InferenceInputs:
    inference_method = args.get("inference_method", "map")
    model_name = args["model_name"]
    basis_dims = args["basis_dims"]
    approx_x_dim = args["approx_x_dim"]
    injury = args["injury"]
    censor_survival_at_injury = args.get("censor_survival_at_injury", False)
    age_min = args["age_min"]
    age_max = args["age_max"]
    start_year = args.get("start_year")
    end_year = args.get("end_year")
    m_time = args.get("m_time") or 15
    players = args["player_names"]
    de_trend_metrics = args["de_trend_metrics"]
    validation_year = args["validation_year"]
    validation_scheme = args["validation_scheme"]
    if validation_scheme == "none":
        validation_scheme = None
    holdout_fraction = args["holdout_fraction"]
    holdout_k = args["holdout_k"]
    holdout_seed = args["holdout_seed"]
    position_group = args["position_group"]
    prior_knobs = args.get("prior_knobs", {}) or {}

    model_dir = args.get("model_dir") or f"model_output/{model_name}/{inference_method}"
    os.makedirs(model_dir, exist_ok=True)

    _year_filter = f"age <= {age_max} & name != 'Brandon Williams'"
    if start_year is not None:
        _year_filter += f" & year >= {start_year}"
    if end_year is not None:
        _year_filter += f" & year <= {end_year}"
    data_all = pd.read_csv("data/injury_player_cleaned.csv").query(_year_filter)
    data_all["split"] = np.random.choice(["train", "test"], size=len(data_all), p=[0.8, 0.2])
    data_all["first_major_injury"] = (
        data_all["first_major_injury"].fillna("None").astype(str).str.strip()
        .replace({"Quad Tendon": "Quad/Patellar", "Patellar Tendon": "Quad/Patellar"})
    )
    data_all["first_major_injury"] = (
        data_all["first_major_injury"].astype("category").cat.set_categories(
            ["None"] + [c for c in pd.unique(data_all["first_major_injury"]) if c != "None"], ordered=False))
    data_all["injury_code"] = data_all["first_major_injury"].cat.codes
    data_all["log_min"] = np.log(data_all["minutes"])
    data_all["usg"] /= 100
    data_all["usg"] += .01
    data_all["simple_exposure"] = 1
    data_all["games_exposure"] = np.maximum(data_all["total_games"], data_all["games"])
    data_all["pct_minutes"] = (data_all["minutes"] / data_all["games"]) / 48
    data_all["retirement"] = 1

    metrics, metric_output, exposure_list = parse_metrics(args)
    scale_values = jnp.ones((len(metrics), 1))

    for metric, metric_type, exposure in zip(metrics, metric_output, exposure_list):
        if metric_type in ["gaussian", "beta"]:
            league_avg_broadcasted = data_all.groupby(["year"]).apply(
                lambda g: (g[metric] * g[exposure]).sum() / g[exposure].sum()).reset_index().rename(columns={0: f"{metric}_league_avg"})
            data_all = data_all.merge(league_avg_broadcasted)
        elif metric_type in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli"]:
            data_all[f"{metric}_league_avg"] = data_all.groupby("year")[metric].transform("sum") / data_all.groupby("year")[exposure].transform("sum")

    data = data_all

    _fake_n = age_max - age_min + 1
    fake_data = pd.DataFrame({"age": range(age_min, age_max + 1), "id": 99999999, "year": range(2000, 2000 + _fake_n), "name": "No Name"})
    fake_data = fake_data.reindex(columns=data.columns)
    fake_data["draft_position_adj"] = 39
    fake_data["height_inches"] = 79
    data = pd.concat([data, fake_data], ignore_index=True)
    names = data.groupby("id")["name"].first().values.tolist()
    validation_mask = data[["year", "age", "id"]].pivot(columns="age", index="id", values="year").reindex(columns=range(age_min, age_max + 1)).apply(
        lambda r: r.dropna().iloc[0] + (np.array(range(age_min, age_max + 1)) - r.dropna().index[0]) if r.notna().any() else r,
        axis=1, result_type="expand").to_numpy() > validation_year

    if players:
        pattern = r"class-of-(\d{4})"
        player_indices = []
        for item in players:
            match = re.fullmatch(pattern, item)
            if item == "low-minutes":
                total_mins = data.groupby("id")["minutes"].sum().reset_index()
                for index, val in enumerate(total_mins["minutes"].values.tolist()):
                    if val <= np.percentile(total_mins["minutes"], 25) and index not in player_indices:
                        player_indices.append(index)
            elif match:
                year = int(match.group(1))
                subset = data.groupby("id")["year"].min().reset_index()
                for index, val in enumerate(subset["year"].values.tolist()):
                    if index not in player_indices and val == year:
                        player_indices.append(index)
            else:
                if names.index(item) not in player_indices:
                    player_indices.append(names.index(item))
    elif position_group in ["G", "F", "C"]:
        all_indices = data.drop_duplicates(subset=["position_group", "name", "id"]).reset_index()
        player_indices = all_indices[all_indices["position_group"] == position_group].index.values.tolist()
    else:
        player_indices = []

    de_trend_indices = [True if metric in de_trend_metrics else False for metric in metrics]

    _, surv_data_set, basis = create_surv_data(data, basis_dims, ["left", "right"], ["retirement"] * 2, [], validation_year=validation_year, age_min=age_min, age_max=age_max)
    surv_masks = jnp.stack([d["censored"] for d in surv_data_set], -1)
    censor = jnp.stack([d["censor_type"] for d in surv_data_set], -1)
    Y_surv = jnp.stack([d["observations"] for d in surv_data_set], -1)

    if censor_survival_at_injury:
        _onset = data_all[data_all["injury_period"] != "pre-injury"].groupby("id")["age"].min()
        _player_ids = data.groupby("id").apply(lambda g: g["id"].iloc[0]).index.tolist()
        _Y_surv_np = np.array(Y_surv)
        _surv_masks_np = np.array(surv_masks)
        for _i, _pid in enumerate(_player_ids):
            if _pid in _onset.index:
                _onset_age = float(_onset[_pid])
                if _onset_age < _Y_surv_np[_i, 1]:
                    _Y_surv_np[_i, 1] = _onset_age
                    _surv_masks_np[_i, 1] = True
        Y_surv = jnp.array(_Y_surv_np)
        surv_masks = jnp.array(_surv_masks_np)

    # ── model-class dispatch (shared dispatch_model; same in main.py / model_export) ──────────
    covariate_X, data_set, basis = create_fda_data(
        data, basis_dims, metric_output, metrics, exposure_list, [],
        injury=fda_injury_flag(model_name, injury), validation_year=validation_year, age_min=age_min, age_max=age_max)
    model = dispatch_model(
        model_name, latent_rank=basis_dims, output_shape=(covariate_X.shape[0], len(basis), len(metrics)),
        basis=basis, injury=injury, num_injury_types=int(data["injury_code"].max()),
        rff_dim=approx_x_dim, prior_knobs=prior_knobs)

    # attribute knobs (e.g. x_latent_df) must be set before initialize_priors
    for _k in _ATTRIBUTE_KNOBS:
        if _k in prior_knobs:
            setattr(model, _k, _build_knob_value(prior_knobs[_k]))
    model.initialize_priors(scale_values=scale_values)
    apply_prior_knobs(model, prior_knobs, metrics=metrics)

    # ── masks / holdout / offsets / data_dict / hsgp / model_args ────────────
    distribution_families = set([d["output"] for d in data_set])
    distribution_indices = {family: jnp.array([i for i, d in enumerate(data_set) if family == d["output"]]) for family in distribution_families}
    masks = jnp.stack([d["mask"] for d in data_set]) * (~validation_mask[None])
    _ref = np.array(data_set[0]["output_data"])
    _train_scheme = None if validation_scheme == "all" else validation_scheme
    holdout_mask = jnp.array(create_validation_mask(_ref, _train_scheme, holdout_fraction, holdout_k, holdout_seed, age_min=age_min))
    masks = masks * (~holdout_mask[None])

    holdout_indices_path = None
    if _train_scheme is not None:
        _hm = np.asarray(holdout_mask)
        _pids = data[["year", "age", "id"]].pivot(columns="age", index="id", values="year").index.values
        _ages = list(range(age_min, age_max + 1))
        _hi_rows, _hi_cols = np.where(_hm)
        _ho_ages_arr = np.array(_ages)[_hi_cols]
        _ho_df_dict: dict = {"player": _pids[_hi_rows], "age": _ho_ages_arr}
        if _train_scheme == "stratified_next_k":
            _score_mask = np.zeros_like(_hm)
            _ho_start_age = np.full(_hm.shape[0], -1, dtype=int)
            for _r in range(_hm.shape[0]):
                _cols = np.where(_hm[_r])[0]
                if len(_cols):
                    _start = int(_cols[0])
                    _score_mask[_r, _start:_start + holdout_k] = _hm[_r, _start:_start + holdout_k]
                    _ho_start_age[_r] = _ages[_start]
            _strata_ends = [24, 26, 28, 30, 32, 34]
            def _stratum_of(_a):
                return next((s for s, end in enumerate(_strata_ends, 1) if int(_a) <= end), None)
            _ho_df_dict["stratum"] = [_stratum_of(_ho_start_age[_r]) for _r in _hi_rows]
            _ho_df_dict["score_window"] = _score_mask[_hi_rows, _hi_cols].astype(int)
        holdout_indices_path = os.path.join(model_dir, "holdout_indices.csv")
        pd.DataFrame(_ho_df_dict).to_csv(holdout_indices_path, index=False)
        # Censor survival at last in-sample age for holdout players
        _Y_surv_h = np.array(Y_surv)
        _smasks_h = np.array(surv_masks)
        _ages_arr = np.array(_ages)
        for _pi in range(_hm.shape[0]):
            _ho_ages = _ages_arr[_hm[_pi]]
            if len(_ho_ages) == 0:
                continue
            _last_in = max(float(_ho_ages.min()) - 1.0, float(_Y_surv_h[_pi, 0]))
            if _last_in < float(_Y_surv_h[_pi, 1]):
                _Y_surv_h[_pi, 1] = _last_in
                _smasks_h[_pi, 1] = True
        Y_surv = jnp.array(_Y_surv_h)
        surv_masks = jnp.array(_smasks_h)

    injury_masks = jnp.stack([d["injury_mask"] for d in data_set])
    using_injury_model = isinstance(model, (ConvexMaxInjuryTVLinearLVM))
    should_mask_injury_values = ((injury and not using_injury_model) or ("counterfactual" in model_name))
    if should_mask_injury_values:
        masks = masks * (~injury_masks)
    injury_types = jnp.stack([d["injury_type"] for d in data_set]).astype(jnp.int32)
    exposures = jnp.stack([d["exposure_data"] for d in data_set])
    Y = jnp.stack([d["output_data"] for d in data_set])
    de_trend = jnp.stack([d["de_trend"] for d in data_set])
    de_trend = jnp.where(jnp.array(de_trend_indices)[..., None, None], de_trend, 0.0)

    year_matrix = jnp.array(data_set[0]["year_matrix"])
    min_year = int(jnp.nanmin(year_matrix))
    max_year = int(jnp.nanmax(year_matrix))
    num_years = max_year - min_year + 1
    year_indices = jnp.nan_to_num(year_matrix - min_year, nan=0).astype(int)
    first_observed_year = int(data_all["year"].min())
    ref_year_idx = first_observed_year - min_year

    Y_for_offsets = apply_detrend_for_offsets(Y_obs=Y, exposures_obs=exposures, metric_families=metric_output, de_trend_values=de_trend, de_trend_mask=jnp.array(de_trend_indices))
    offset_mask = masks
    family_requires_positive_exposure = jnp.array(
        [family in ["poisson", "negative-binomial", "binomial", "beta-binomial", "bernoulli", "beta", "gaussian"] for family in metric_output], dtype=bool)[:, None, None]
    finite_mask = jnp.isfinite(Y_for_offsets) & jnp.isfinite(exposures)
    positive_exposure_mask = (~family_requires_positive_exposure) | (exposures > 0)
    offset_valid_mask = offset_mask & finite_mask & positive_exposure_mask
    Y_for_offsets_masked = jnp.where(offset_valid_mask, Y_for_offsets, jnp.nan)
    exposures_for_offsets = jnp.where(offset_valid_mask, exposures, jnp.nan)
    offset_linear_predictor_mean = compute_linear_predictor_mean_offsets(Y_for_offsets_masked, exposures_for_offsets, metric_output)
    offset_max, offset_max_var, offset_peak_absolute, offset_peak_absolute_var, offset_mean = compute_priors(Y_for_offsets_masked, exposures_for_offsets, metric_output, exposure_list)
    offset_peak_absolute = offset_peak_absolute + age_min - basis.mean() + 2.0
    print("offset_max:", offset_max)
    print("offset_peak_absolute:", offset_peak_absolute)
    offset_boundary_r = jnp.log(jnp.exp(2) - 1)
    offset_boundary_l = jnp.log(jnp.exp(2) - 1)

    data_dict = {}
    for family in distribution_families:
        indices = distribution_indices[family]
        data_dict[family] = {"Y": Y[indices], "exposure": exposures[indices], "mask": masks[indices], "indices": indices, "de_trend": de_trend[indices]}

    hsgp_params = {}
    if "convex" in model_name:
        x_time = basis - basis.mean()
        L_time = 2 * jnp.max(jnp.abs(x_time), 0, keepdims=True)
        t_amplitude = float(jnp.squeeze(L_time)) / 2
        M_time = m_time
        phi_time = vmap_make_convex_phi(jnp.squeeze(x_time), jnp.squeeze(L_time), M_time)
        hsgp_params["phi_x_time"] = phi_time
        hsgp_params["M_time"] = M_time
        hsgp_params["L_time"] = L_time
        hsgp_params["t_amplitude"] = t_amplitude
        hsgp_params["shifted_x_time"] = x_time + L_time
        hsgp_params["t_0"] = jnp.min(x_time)
        hsgp_params["t_r"] = jnp.max(x_time)
        if "hsgp" in model_name:
            hsgp_params["eigenvalues_X"] = sqrt_eigenvalues(2 * jnp.ones(basis_dims)[..., None], approx_x_dim, basis_dims)

    player_obs = data.groupby("id")[["draft_position_adj", "height_inches", "position_group"]].first()
    neg_log_draft = -np.log(player_obs["draft_position_adj"].values.astype(float))
    height_vals = player_obs["height_inches"].values.astype(float)
    obs_numeric = np.stack([neg_log_draft, height_vals], axis=1)
    obs_mean = np.nanmean(obs_numeric, axis=0)
    obs_std = np.nanstd(obs_numeric, axis=0) + 1e-8
    obs_numeric_std = (obs_numeric - obs_mean) / obs_std
    pos_dummies = pd.get_dummies(player_obs["position_group"], drop_first=True).astype(float).values
    obs_covariates = jnp.array(np.concatenate([obs_numeric_std, pos_dummies], axis=1))
    model.player_covariates = obs_covariates

    model_args = {"data_set": data_dict, "inference_method": inference_method,
                  "sample_free_indices": jnp.array(player_indices),
                  "sample_fixed_indices": jnp.setdiff1d(jnp.arange(covariate_X.shape[0]), jnp.array(player_indices), assume_unique=True)}
    last_observed_year = int(data_all["year"].max())
    year_max_idx = last_observed_year - min_year
    if "linear" in model_name:
        model_args["ar_metric_indices"] = jnp.where(jnp.array(de_trend_indices))[0]
        model_args["year_indices"] = year_indices
        model_args["num_years"] = num_years
        model_args["num_de_trend"] = len(de_trend_metrics)
        model_args["ref_year_idx"] = ref_year_idx
    if "linear_trend" in model_name:
        model_args["year_max_idx"] = year_max_idx

    model_args["offsets"] = {}
    entrance_times = Y_surv[:, 0]
    exit_times = Y_surv[:, 1]
    right_censor = surv_masks[:, 1]
    model_args["offsets"].update({"exit_times": exit_times - age_min + 1e-6, "entrance_times": entrance_times - age_min + 1e-6, "right_censor": right_censor})
    model_args["offsets"]["injury_indicator"] = injury_masks
    model_args["offsets"]["injury_type"] = injury_types
    model_args.update({"hsgp_params": hsgp_params})
    if "tvlinearlvm" in model_name and "convex" not in model_name:
        model_args["offsets"].update({"c_mean": offset_mean, "c_max_var": offset_max_var})
    if "convex" in model_name:
        if "max" in model_name:
            model_args["offsets"].update({"t_max": offset_peak_absolute, "c_max": offset_max, "boundary_r": offset_boundary_r, "boundary_l": offset_boundary_l, "t_max_var": offset_peak_absolute_var, "c_max_var": offset_max_var})

    return InferenceInputs(
        model=model, model_args=model_args, data=data, metrics=metrics, metric_output=metric_output,
        exposure_list=exposure_list, basis=basis, covariate_X=covariate_X, Y=Y, exposures=exposures,
        masks=masks, de_trend=de_trend, distribution_indices=distribution_indices, age_min=age_min,
        age_max=age_max, names=names, model_dir=model_dir, holdout_indices_path=holdout_indices_path,
        scale_values=scale_values,
        extras={"validation_mask": validation_mask, "Y_surv": Y_surv, "surv_masks": surv_masks,
                "player_indices": player_indices, "offset_linear_predictor_mean": offset_linear_predictor_mean,
                "year_indices": year_indices, "num_years": num_years, "ref_year_idx": ref_year_idx},
    )
