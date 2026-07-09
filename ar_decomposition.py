#!/usr/bin/env python
"""f-vs-AR variance decomposition from a MAP fit (CPU-only).

For an AR model, the linear predictor is  mu_ptk = f^k(X_p,t) + r^k_p(t) (+ calendar trend),
where f^k is the concave aging curve (shared latent) and r^k_p(t) is the per-player AR(1)
residual. This reconstructs f and r from the MAP point estimate (samples.pkl) via the model's
own forward, then reports, per metric, how much of the *player-specific* signal lives in the
structural curve vs. the AR — i.e. how much the AR "washes out".

    AR share = Var_{obs}(r) / [ Var_{obs}(f - f_pop) + Var_{obs}(r) ]

computed over in-sample (observed, non-holdout) (player, age) cells, where f_pop is the
population-average curve for that metric. Lower AR share  ->  curve holds more signal.

Usage (CPU, won't touch the GPUs):
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python ar_decomposition.py \
        nba_convex_max_tvrflvm_AR_stratified_next_k \
        nba_convex_max_tvlinearlvm_AR_stratified_next_k
"""
import os
# Force CPU BEFORE importing jax so we never grab a GPU (MCMC is running there).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys, pickle
import numpy as np
import jax, jax.numpy as jnp
import numpyro
from numpyro.handlers import seed, substitute
jax.config.update("jax_enable_x64", True)

from config.config_utils import resolve_model_config
from model.inference_inputs import build_inference_inputs


def reconstruct_f_and_r(model_name, config="config/model_config.yaml"):
    """Return (f, r, mask, metrics): f,r are (k,n,t) MAP arrays, mask (k,n,t) in-sample bool."""
    args = resolve_model_config(config, model_name, "map")
    args["model_name"] = model_name

    map_dir = args["model_dir"]
    with open(os.path.join(map_dir, "samples.pkl"), "rb") as fh:
        samples = pickle.load(fh)
    params = {(k[:-5] if k.endswith("__loc") else k): jnp.asarray(v) for k, v in samples.items()}

    # Match the rebuilt model to the saved fit's dims (the fit may predate a config change):
    #   r = latent rank (X cols), m = RFF frequency count (W rows, RFF only).
    if "X" in params:
        args["basis_dims"] = int(params["X"].shape[1])
    if "W" in params:
        args["approx_x_dim"] = int(params["W"].shape[0])

    inp = build_inference_inputs(args)              # single-source data + model setup
    model, ma = inp.model, inp.model_args

    ca = (ma["hsgp_params"], ma["offsets"], ma["sample_free_indices"], ma["sample_fixed_indices"],
          ma.get("ar_metric_indices", jnp.array([])), ma.get("year_indices", jnp.array([])),
          ma.get("num_years", 1), ma.get("num_de_trend", 0), ma.get("ref_year_idx", 0))

    def fwd():
        d = model.compute_curves(*ca, include_derivs=False)
        return d["mu"], d["trend_ar"], model._compute_player_ar()   # f, trend, r  (k,n,t)

    f, trend, r = substitute(seed(fwd, jax.random.PRNGKey(0)), data=params)()
    return np.asarray(f), np.asarray(r), np.asarray(inp.masks).astype(bool), list(inp.metrics)


def decompose(model_name):
    f, r, mask, metrics = reconstruct_f_and_r(model_name)
    print(f"\n=== {model_name} ===")
    print(f"{'metric':10s} {'Var(f_dev)':>11s} {'Var(AR)':>10s} {'AR share':>9s}")
    rows = []
    for ki, m in enumerate(metrics):
        obs = mask[ki]
        n_obs = obs.sum()
        if n_obs < 2:
            continue
        fk, rk = f[ki], r[ki]
        denom = obs.sum(axis=0).clip(min=1)                 # players observed per age
        f_pop = (np.where(obs, fk, 0.0).sum(axis=0)) / denom  # (t,) population-avg curve
        f_dev = (fk - f_pop[None, :])[obs]                  # player-specific structural deviation
        rr = rk[obs]
        Vf, Vr = float(f_dev.var()), float(rr.var())
        share = Vr / (Vf + Vr + 1e-12)
        rows.append((m, Vf, Vr, share))
        print(f"{m:10s} {Vf:11.4f} {Vr:10.4f} {share:8.1%}")
    # signal-weighted overall AR share across metrics
    tot_f = sum(v for _, v, _, _ in rows)
    tot_r = sum(v for _, _, v, _ in rows)
    print(f"{'OVERALL':10s} {tot_f:11.4f} {tot_r:10.4f} {tot_r/(tot_f+tot_r+1e-12):8.1%}")
    return rows


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "nba_convex_max_tvrflvm_AR_stratified_next_k",
        "nba_convex_max_tvlinearlvm_AR_stratified_next_k",
    ]
    print(f"jax devices (should be CPU): {jax.devices()}")
    summary = []
    for nm in names:
        try:
            rows = decompose(nm)
            tf = sum(v for _, v, _, _ in rows); tr = sum(v for _, _, v, _ in rows)
            summary.append((nm, tr / (tf + tr + 1e-12)))
        except FileNotFoundError as e:
            print(f"\n=== {nm} ===\n  skipped — no MAP samples.pkl ({e})")
        except Exception as e:
            print(f"\n=== {nm} ===\n  skipped — {type(e).__name__}: {e}")
    if summary:
        print("\n================ OVERALL AR share by model ================")
        for nm, s in sorted(summary, key=lambda x: x[1]):
            print(f"  {s:6.1%}   {nm}")
