import yaml

_SCHEME_SUFFIXES = (
    "_holdout_last_k", "_holdout_first_k", "_random_interior", "_holdout_peak",
)


def _lookup(models: dict, name: str) -> dict:
    """Exact match, then longest-prefix match. Returns a shallow copy."""
    if name in models:
        return dict(models[name])
    matches = [k for k in models if name.startswith(k)]
    return dict(models[max(matches, key=len)]) if matches else {}


def resolve_model_config(config_path: str, model_name: str, inference_method: str | None = None) -> dict:
    """Load model_config.yaml and return merged config for model_name + inference_method.

    Merge order (later wins):
      defaults → base_entry → regimes[inference_method] → scheme_entry

    'base_entry' is found by stripping a known scheme suffix, so regimes and
    fixed_params defined on the base automatically apply to all scheme variants.
    Scheme entries should only carry fields that differ per scheme
    (model_dir, validation_scheme, injury).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})

    scheme_entry = _lookup(models, model_name)
    _scheme_explicit_keys = set(scheme_entry.keys())

    base_name = model_name
    for suffix in _SCHEME_SUFFIXES:
        if model_name.endswith(suffix):
            base_name = model_name[: -len(suffix)]
            break
    _is_scheme_variant = base_name != model_name
    base_entry = _lookup(models, base_name) if _is_scheme_variant else dict(scheme_entry)

    regimes = base_entry.pop("regimes", {})
    scheme_entry.pop("regimes", None)

    regime_overrides = regimes.get(inference_method, {}) if inference_method else {}

    merged = {**defaults, **base_entry, **regime_overrides, **scheme_entry}
    if inference_method:
        merged["inference_method"] = inference_method

    # For MCMC scheme-variant runs, collocate model_dir and init_path with the
    # scheme's MAP model_dir so each scheme's MCMC initialises from its own
    # MAP samples rather than the base model's.
    if inference_method == "mcmc" and _is_scheme_variant:
        _map_dir = scheme_entry.get("model_dir") or ""
        if _map_dir:
            _mcmc_dir = (
                _map_dir[:-3] + "mcmc" if _map_dir.endswith("/map")
                else f"{_map_dir}/mcmc"
            )
            merged["model_dir"] = _mcmc_dir
            if "init_path" not in _scheme_explicit_keys:
                merged["init_path"] = f"{_map_dir}/samples.pkl"
            if "fixed_param_path" not in _scheme_explicit_keys:
                merged["fixed_param_path"] = f"{_map_dir}/samples.pkl"

    for _list_key in ("player_names", "de_trend_metrics", "fixed_params"):
        val = merged.get(_list_key)
        if val is None:
            merged[_list_key] = []
        elif isinstance(val, str):
            merged[_list_key] = [v.strip() for v in val.split(",") if v.strip()]
    return merged


def parse_metrics(cfg: dict) -> tuple:
    """Unzip the metrics list-of-dicts into three parallel lists.

    Returns (metric_names, metric_output, exposure_list).
    """
    entries = cfg.get("metrics", [])
    return (
        [e["name"] for e in entries],
        [e["distribution"] for e in entries],
        [e["exposure"] for e in entries],
    )
