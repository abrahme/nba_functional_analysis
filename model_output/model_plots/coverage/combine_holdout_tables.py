"""Combine per-model holdout CSVs into a single LaTeX comparison table.

Reads every  <model>_vs_<scheme>_f<frac>_k<k>_s<seed>.csv  file from this
directory and produces  combined_holdout_table.tex  with one subtable per
holdout scheme.  Each subtable has rows = metrics and columns = model families,
each cell showing  RMSE (bias).  Only the "holdout" split is reported.

Usage (run from repo root):
    python model_output/model_plots/coverage/combine_holdout_tables.py
"""

import math
import os
import re
import pandas as pd

COVERAGE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TEX    = os.path.join(COVERAGE_DIR, "combined_holdout_table.tex")
OUTPUT_BODY   = os.path.join(COVERAGE_DIR, "combined_holdout_body.tex")

SCHEME_ORDER = ["holdout_last_k", "holdout_first_k", "random_interior", "holdout_peak"]
SCHEME_LABEL = {
    "holdout_last_k":   "Hold-out Last $k$",
    "holdout_first_k":  "Hold-out First $k$",
    "random_interior":  "Random Interior",
    "holdout_peak":     "Hold-out Peak",
}

MODEL_ORDER = [
    "nba_naive",
    "nba_convex_max_tvlinearlvm",
    "nba_convex_max_tvlinearlvm_AR",
    "nba_convex_max_tvlinearlvm_injury"

]
MODEL_LABEL = {
    "nba_convex_max_tvlinearlvm":         "Concave",
    "nba_convex_max_tvlinearlvm_AR":      "Concave + AR",
    "nba_convex_max_tvlinearlvm_injury":  "Concave + AR + Injury",
    "nba_naive":                          "AR",
}

METRIC_ORDER = [
    "games", "usg", "pct_minutes",
    "obpm", "dbpm",
    "blk", "stl", "ast", "dreb", "oreb", "tov",
    "fta", "fg2a", "fg3a",
    "ftm", "fg2m", "fg3m",
]
METRIC_LABEL = {
    "games": "GP\\%", "usg": "USG\\%", "pct_minutes": "MPG",
    "obpm": "OBPM", "dbpm": "DBPM",
    "blk": "BLK", "stl": "STL", "ast": "AST",
    "dreb": "DREB", "oreb": "OREB", "tov": "TOV",
    "fta": "FTA", "fg2a": "FG2A", "fg3a": "FG3A",
    "ftm": "FT\\%", "fg2m": "FG2\\%", "fg3m": "FG3\\%",
}

# ── Load all CSVs ─────────────────────────────────────────────────────────────

_SCHEME_SUFFIXES = ("_holdout_last_k", "_holdout_first_k", "_random_interior", "_holdout_peak")

records = []
pattern = re.compile(r"^(.+)_vs_(.+?)_f\d+_k\d+_s\d+\.csv$")

for fname in os.listdir(COVERAGE_DIR):
    m = pattern.match(fname)
    if not m:
        continue
    model_name, scheme = m.group(1), m.group(2)
    # Strip scheme suffix so the key matches MODEL_ORDER base names
    for suffix in _SCHEME_SUFFIXES:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
            break
    df = pd.read_csv(os.path.join(COVERAGE_DIR, fname))
    df = df[df["split"] == "holdout"].copy()
    df["model"]  = model_name
    df["scheme"] = scheme
    records.append(df)

if not records:
    print("No coverage CSVs found — run the holdout jobs first.")
    raise SystemExit(1)

data = pd.concat(records, ignore_index=True)

# ── Build LaTeX ───────────────────────────────────────────────────────────────

def _sci(val, decimals=2):
    """Format val in LaTeX scientific notation: $m.dd\times10^{e}$."""
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / 10 ** exp
    sign = "+" if mantissa >= 0 else ""
    return f"${mantissa:.{decimals}f}\\times10^{{{exp}}}$"


def _fmt_num(val, decimals=3):
    """Fixed decimal for small values, scientific notation for |val| >= 1e4."""
    if pd.isna(val):
        return "---"
    if abs(val) >= 1e4:
        return _sci(val)
    return f"{val:.{decimals}f}"


def _fmt_signed(val, decimals=3):
    """Like _fmt_num but always shows a leading sign."""
    if pd.isna(val):
        return "---"
    if abs(val) >= 1e4:
        return ("+" if val >= 0 else "") + _sci(val)
    return f"{val:+.{decimals}f}"


def fmt_cell(rmse, bias, bold=False):
    """Format as  RMSE (bias)  using scientific notation for large values."""
    if pd.isna(rmse) and pd.isna(bias):
        return "---"
    r = _fmt_num(rmse)
    b = _fmt_signed(bias)
    text = f"{r} ({b})"
    return f"\\textbf{{{text}}}" if bold else text


def best_model_for_metric(sub, metric, models, col):
    """Return the model name with the lowest finite value of col for metric."""
    vals = {}
    for m in models:
        cd = sub[(sub["model"] == m) & (sub["metric"] == metric)]
        if not cd.empty and not pd.isna(cd[col].iloc[0]):
            vals[m] = cd[col].iloc[0]
    return min(vals, key=vals.__getitem__) if vals else None


lines = []
lines.append(r"\documentclass{article}")
lines.append(r"\usepackage{booktabs}")
lines.append(r"\usepackage{multirow}")
lines.append(r"\usepackage{graphicx}")
lines.append(r"\usepackage[margin=1in]{geometry}")
lines.append(r"\begin{document}")
lines.append("")
lines.append(r"\newcommand{\tblnote}[1]{\smallskip\noindent\footnotesize #1}")
lines.append("")

present_models  = [m for m in MODEL_ORDER  if m in data["model"].unique()]
present_schemes = [s for s in SCHEME_ORDER if s in data["scheme"].unique()]

n_models = len(present_models)
col_spec = "l" + "r" * n_models    # metric col + one col per model

for scheme in present_schemes:
    sub = data[data["scheme"] == scheme]
    if sub.empty:
        continue

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{")

    # column spec: metric | model1 | model2 | model3
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header_cols = ["Metric"] + [MODEL_LABEL.get(m, m) for m in present_models]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"  & " + " & ".join([r"\small RMSE (bias)"] * n_models) + r" \\")
    lines.append(r"\midrule")

    present_metrics = [m for m in METRIC_ORDER if m in sub["metric"].unique()]
    # add any unexpected metrics at the end
    extra = [m for m in sub["metric"].unique() if m not in METRIC_ORDER]
    for metric in present_metrics + extra:
        best = best_model_for_metric(sub, metric, present_models, "rmse")
        row_cells = [METRIC_LABEL.get(metric, metric.upper())]
        for model in present_models:
            cell_data = sub[(sub["model"] == model) & (sub["metric"] == metric)]
            if cell_data.empty:
                row_cells.append("---")
            else:
                row_cells.append(fmt_cell(
                    cell_data["rmse"].iloc[0],
                    cell_data["bias"].iloc[0],
                    bold=(model == best),
                ))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close \resizebox
    scheme_label = SCHEME_LABEL.get(scheme, scheme)
    lines.append(
        f"\\caption{{Holdout RMSE and bias (in parentheses) by metric — "
        f"{scheme_label} scheme.}}"
    )
    lines.append(f"\\label{{tab:coverage_{scheme}}}")
    lines.append(r"\end{table}")
    lines.append("")

# ── Log-loss tables ───────────────────────────────────────────────────────────

lines.append(r"\bigskip")
lines.append(r"{\centering\large\textbf{Log-Loss Tables}\\[4pt]}")
lines.append("")

for scheme in present_schemes:
    sub = data[data["scheme"] == scheme]
    if sub.empty:
        continue

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header_cols = ["Metric"] + [MODEL_LABEL.get(m, m) for m in present_models]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"  & " + " & ".join([r"\small Avg log-loss"] * n_models) + r" \\")
    lines.append(r"\midrule")

    present_metrics = [m for m in METRIC_ORDER if m in sub["metric"].unique()]
    extra = [m for m in sub["metric"].unique() if m not in METRIC_ORDER]
    for metric in present_metrics + extra:
        best = best_model_for_metric(sub, metric, present_models, "avg_log_loss")
        row_cells = [METRIC_LABEL.get(metric, metric.upper())]
        for model in present_models:
            cell_data = sub[(sub["model"] == model) & (sub["metric"] == metric)]
            if cell_data.empty or pd.isna(cell_data["avg_log_loss"].iloc[0]):
                row_cells.append("---")
            else:
                raw = cell_data["avg_log_loss"].iloc[0]
                val = _fmt_num(raw, decimals=4)
                row_cells.append(f"\\textbf{{{val}}}" if model == best else val)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close \resizebox
    scheme_label = SCHEME_LABEL.get(scheme, scheme)
    lines.append(
        f"\\caption{{Holdout average log-loss by metric — "
        f"{scheme_label} scheme.}}"
    )
    lines.append(f"\\label{{tab:logloss_{scheme}}}")
    lines.append(r"\end{table}")
    lines.append("")

# ── Coverage tables ───────────────────────────────────────────────────────────
# Reads coverage_basic.tex from <model_output_root>/<model>/<scheme>/mcmc/plots/coverage/

MODEL_OUTPUT_ROOT = os.path.dirname(os.path.dirname(COVERAGE_DIR))

_COV_ROW = re.compile(r'^\s*(.+?) & ([\d.]+)\\% & ([\d.]+)\\%', re.MULTILINE)

def parse_coverage_tex(path):
    """Return dict: display_label -> (val_pct, in_sample_pct)."""
    with open(path) as f:
        content = f.read()
    result = {}
    for m in _COV_ROW.finditer(content):
        label = m.group(1).strip()
        result[label] = (float(m.group(2)), float(m.group(3)))
    return result


# {scheme: {model_key: {display_label: (val_pct, in_pct)}}}
cov_data = {}
for model_key in MODEL_ORDER:
    for scheme in SCHEME_ORDER:
        tex_path = os.path.join(
            MODEL_OUTPUT_ROOT, model_key, scheme, "mcmc", "plots", "coverage", "coverage_basic.tex"
        )
        if not os.path.exists(tex_path):
            continue
        cov_data.setdefault(scheme, {})[model_key] = parse_coverage_tex(tex_path)

# Display label -> metric key (for ordering rows in METRIC_ORDER order)
LABEL_TO_KEY = {v: k for k, v in METRIC_LABEL.items()}

if cov_data:
    lines.append(r"\bigskip")
    lines.append(r"{\centering\large\textbf{Coverage Tables (95\% HDI)}\\[4pt]}")
    lines.append("")

    cov_schemes = [s for s in SCHEME_ORDER if s in cov_data]
    for scheme in cov_schemes:
        scheme_cov = cov_data[scheme]
        present_cov_models = [m for m in MODEL_ORDER if m in scheme_cov]
        if not present_cov_models:
            continue

        # Collect all metric display labels present across any model
        all_labels = set()
        for mc in scheme_cov.values():
            all_labels.update(mc.keys())

        # Order: METRIC_LABEL display values in METRIC_ORDER, then extras, EXIT_AGE last
        ordered_labels = [METRIC_LABEL[k] for k in METRIC_ORDER if METRIC_LABEL[k] in all_labels]
        exit_labels    = [l for l in all_labels if "EXIT" in l]
        extra_labels   = sorted(all_labels - set(ordered_labels) - set(exit_labels))
        ordered_labels += extra_labels + exit_labels

        n_cov   = len(present_cov_models)
        col_cov = "l" + "r" * n_cov

        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\resizebox{\linewidth}{!}{")
        lines.append(f"\\begin{{tabular}}{{{col_cov}}}")
        lines.append(r"\toprule")
        header_cols = ["Metric"] + [MODEL_LABEL.get(m, m) for m in present_cov_models]
        lines.append(" & ".join(header_cols) + r" \\")
        lines.append(r"  & " + " & ".join([r"\small Val\% (In-Samp\%)"] * n_cov) + r" \\")
        lines.append(r"\midrule")

        for label in ordered_labels:
            # Best model = highest validation coverage for this metric
            best, best_val = None, -1.0
            for mk in present_cov_models:
                pcts = scheme_cov[mk].get(label)
                if pcts and pcts[0] > best_val:
                    best_val, best = pcts[0], mk

            row_cells = [label]
            for mk in present_cov_models:
                pcts = scheme_cov[mk].get(label)
                if pcts is None:
                    row_cells.append("---")
                else:
                    v, i = pcts
                    text = f"{v:.1f}\\% ({i:.1f}\\%)"
                    row_cells.append(f"\\textbf{{{text}}}" if mk == best else text)
            lines.append(" & ".join(row_cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")  # close \resizebox
        scheme_label = SCHEME_LABEL.get(scheme, scheme)
        lines.append(
            f"\\caption{{Holdout (val) and in-sample 95\\% HDI coverage by metric --- "
            f"{scheme_label} scheme. Best validation coverage per metric in bold.}}"
        )
        lines.append(f"\\label{{tab:hdi_coverage_{scheme}}}")
        lines.append(r"\end{table}")
        lines.append("")

# ── Posterior interval tables (bias and MSE) ──────────────────────────────────
# Reads coverage_bias_intervals.csv and coverage_mse_intervals.csv from
# <model_output_root>/<model>/<scheme>/mcmc/plots/coverage/

def load_interval_csv(model_key, scheme, filename):
    """Return DataFrame with columns metric, mean, lower, upper, split; or None."""
    path = os.path.join(
        MODEL_OUTPUT_ROOT, model_key, scheme, "mcmc", "plots", "coverage", filename
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def fmt_interval(mean, lower, upper, decimals=3):
    """Format as  mean [lower, upper]."""
    if pd.isna(mean):
        return "---"
    fmt = f"{{:+.{decimals}f}}"
    fmtu = f"{{:.{decimals}f}}"
    if abs(mean) >= 1e4:
        return _sci(mean)
    return f"{fmt.format(mean)} [{fmtu.format(lower)}, {fmtu.format(upper)}]"


def _build_interval_tables(csv_filename, section_title, caption_template,
                           label_prefix, best_by="abs_mean"):
    """
    Emit LaTeX table blocks for a given interval CSV file.

    best_by: "abs_mean" (bias — best = closest to 0),
             "mean"     (MSE/log-loss — best = smallest mean)
    """
    # {scheme: {model_key: DataFrame(metric, mean, lower, upper, split)}}
    iv_data = {}
    for mk in MODEL_ORDER:
        for sc in SCHEME_ORDER:
            df = load_interval_csv(mk, sc, csv_filename)
            if df is None:
                continue
            iv_data.setdefault(sc, {})[mk] = df

    if not iv_data:
        return

    lines.append(r"\bigskip")
    lines.append(f"{{\\centering\\large\\textbf{{{section_title}}}\\\\[4pt]}}")
    lines.append("")

    for scheme in SCHEME_ORDER:
        if scheme not in iv_data:
            continue
        scheme_iv = iv_data[scheme]
        present_iv_models = [m for m in MODEL_ORDER if m in scheme_iv]
        if not present_iv_models:
            continue

        n_iv = len(present_iv_models)
        col_iv = "l" + "r" * n_iv

        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\resizebox{\linewidth}{!}{")
        lines.append(f"\\begin{{tabular}}{{{col_iv}}}")
        lines.append(r"\toprule")
        header_cols = ["Metric"] + [MODEL_LABEL.get(m, m) for m in present_iv_models]
        lines.append(" & ".join(header_cols) + r" \\")
        subhdr = r"\small Val [95\% HDI] / (In-Samp [95\% HDI])"
        lines.append(r"  & " + " & ".join([subhdr] * n_iv) + r" \\")
        lines.append(r"\midrule")

        # Collect all metric labels across models
        all_metrics_raw = set()
        for df in scheme_iv.values():
            all_metrics_raw.update(df["metric"].unique())
        # Order by METRIC_LABEL display values; extras appended
        metric_label_inv = {v: k for k, v in METRIC_LABEL.items()}
        ordered_ms = [METRIC_LABEL[k] for k in METRIC_ORDER if METRIC_LABEL[k] in all_metrics_raw]
        extra_ms   = sorted(all_metrics_raw - set(ordered_ms))
        ordered_ms += extra_ms

        for label in ordered_ms:
            # Identify best model for this metric on validation mean
            best = None
            best_score = float("inf")
            for mk in present_iv_models:
                df = scheme_iv[mk]
                row = df[(df["metric"] == label) & (df["split"] == "holdout")]
                if row.empty or pd.isna(row["mean"].iloc[0]):
                    continue
                score = abs(row["mean"].iloc[0]) if best_by == "abs_mean" else row["mean"].iloc[0]
                if score < best_score:
                    best_score, best = score, mk

            row_cells = [label]
            for mk in present_iv_models:
                df = scheme_iv[mk]
                val_row = df[(df["metric"] == label) & (df["split"] == "holdout")]
                is_row  = df[(df["metric"] == label) & (df["split"] == "in_sample")]
                if val_row.empty and is_row.empty:
                    row_cells.append("---")
                    continue
                val_str = fmt_interval(
                    val_row["mean"].iloc[0]  if not val_row.empty else float("nan"),
                    val_row["lower"].iloc[0] if not val_row.empty else float("nan"),
                    val_row["upper"].iloc[0] if not val_row.empty else float("nan"),
                ) if not val_row.empty else "---"
                is_str = fmt_interval(
                    is_row["mean"].iloc[0]  if not is_row.empty else float("nan"),
                    is_row["lower"].iloc[0] if not is_row.empty else float("nan"),
                    is_row["upper"].iloc[0] if not is_row.empty else float("nan"),
                ) if not is_row.empty else "---"
                cell = f"\\shortstack{{{val_str} \\\\\\\\ ({is_str})}}"
                row_cells.append(f"\\textbf{{{cell}}}" if mk == best else cell)
            lines.append(" & ".join(row_cells) + r" \\[2pt]")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        scheme_label = SCHEME_LABEL.get(scheme, scheme)
        lines.append(f"\\caption{{{caption_template.format(scheme=scheme_label)}}}")
        lines.append(f"\\label{{tab:{label_prefix}_{scheme}}}")
        lines.append(r"\end{table}")
        lines.append("")


_build_interval_tables(
    csv_filename="coverage_bias_intervals.csv",
    section_title="Posterior Bias Interval Tables (Mean [95\\% HDI])",
    caption_template="Posterior bias (mean [95\\% HDI]) by metric --- {scheme} scheme. "
                     "Validation interval shown; in-sample in parentheses. "
                     "Best-calibrated model (smallest $|$bias$|$) per metric in bold.",
    label_prefix="bias_interval",
    best_by="abs_mean",
)

_build_interval_tables(
    csv_filename="coverage_log_loss_intervals.csv",
    section_title="Posterior Log-Loss Interval Tables (Avg NLL [95\\% HDI])",
    caption_template="Posterior average negative log-likelihood (mean [95\\% HDI]) by metric "
                     "--- {scheme} scheme. Per-family NLL: Gaussian NLL for OBPM/DBPM/USG/MPG, "
                     "Poisson NLL for count metrics, Binomial NLL for FT\\%/FG2\\%/FG3\\%, "
                     "NB NLL for FTA/FG2A/FG3A, Beta-Binomial NLL for GP\\%. "
                     "Validation interval shown; in-sample in parentheses. "
                     "Best model (lowest validation NLL) per metric in bold.",
    label_prefix="log_loss_interval",
    best_by="mean",
)

# ── ELPPD tables ──────────────────────────────────────────────────────────────
# Reads posterior_elppd.parquet from <model_output_root>/<model>/<scheme>/mcmc/
# Displays ΔELPPD per observation vs. best model per metric row.
# † marks pairs where |ΔELPPD| > 1.96 × SE (significantly different at 95%).

elppd_data = {}
for _mk in MODEL_ORDER:
    for _sc in SCHEME_ORDER:
        _ep = os.path.join(MODEL_OUTPUT_ROOT, _mk, _sc, "mcmc", "posterior_elppd.parquet")
        if not os.path.exists(_ep):
            continue
        elppd_data.setdefault(_sc, {})[_mk] = pd.read_parquet(_ep)

if elppd_data:
    lines.append(r"\bigskip")
    lines.append(r"{\centering\large\textbf{ELPPD Tables (Holdout --- $\Delta$ from best)}\\[4pt]}")
    lines.append("")

    for _sc in SCHEME_ORDER:
        if _sc not in elppd_data:
            continue
        _sc_ep = elppd_data[_sc]
        _ep_models = [_m for _m in MODEL_ORDER if _m in _sc_ep]
        if not _ep_models:
            continue

        _n_ep = len(_ep_models)
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\resizebox{\linewidth}{!}{")
        lines.append(f"\\begin{{tabular}}{{l{'r' * _n_ep}}}")
        lines.append(r"\toprule")
        lines.append(" & ".join(["Metric"] + [MODEL_LABEL.get(_m, _m) for _m in _ep_models]) + r" \\")
        lines.append(r"  & " + " & ".join([r"\small $\Delta$ELPPD/obs"] * _n_ep) + r" \\")
        lines.append(r"\midrule")

        for _mkey in METRIC_ORDER:
            _mlabel = METRIC_LABEL.get(_mkey, _mkey)
            # Collect holdout elppd_per_obs and se for each model
            _ep_vals = {}
            for _m in _ep_models:
                _df = _sc_ep[_m]
                _hr = _df[(_df["metric"] == _mkey) & (_df["split"] == "holdout")]
                _h_val = _hr["elppd_per_obs"].iloc[0] if not _hr.empty else float("nan")
                _h_se  = _hr["elppd_se"].iloc[0]      if not _hr.empty else float("nan")
                _h_n   = _hr["n_obs"].iloc[0]          if not _hr.empty else 1
                _ep_vals[_m] = (_h_val, _h_se, _h_n)

            # Best = highest holdout elppd_per_obs
            _finite_vals = [(_m, v[0]) for _m, v in _ep_vals.items() if not pd.isna(v[0])]
            if not _finite_vals:
                lines.append(_mlabel + " & " + " & ".join(["---"] * _n_ep) + r" \\")
                continue
            _best_m, _best_val = max(_finite_vals, key=lambda x: x[1])
            _best_se, _best_n = _ep_vals[_best_m][1], _ep_vals[_best_m][2]

            row_cells = [_mlabel]
            for _m in _ep_models:
                _h_val, _h_se, _h_n = _ep_vals[_m]
                if pd.isna(_h_val):
                    row_cells.append("---")
                    continue
                _delta_h = _h_val - _best_val
                # SE of Δ(elppd_per_obs) = sqrt((se_A/n_A)² + (se_best/n_best)²)
                # where elppd_se = sqrt(n * Var_i), so elppd_se/n = sqrt(Var_i/n) = SEM
                _se_diff = (
                    ((_h_se / _h_n) ** 2 + (_best_se / _best_n) ** 2) ** 0.5
                    if not pd.isna(_h_se) and not pd.isna(_best_se) and _h_n > 0 and _best_n > 0
                    else float("nan")
                )
                _sig = (not pd.isna(_se_diff)) and (abs(_delta_h) > 1.96 * _se_diff) and (_m != _best_m)
                _h_str = f"{_delta_h:+.4f}" + (r"$\dagger$" if _sig else "")
                cell = _h_str
                row_cells.append(f"\\textbf{{{cell}}}" if _m == _best_m else cell)
            lines.append(" & ".join(row_cells) + r" \\[2pt]")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        _sc_label = SCHEME_LABEL.get(_sc, _sc)
        lines.append(
            f"\\caption{{ELPPD per observation difference vs.\\ best model (holdout) --- "
            f"{_sc_label} scheme. Higher is better; 0 = best model. "
            f"$\\dagger$ = significantly different at 95\\% ($|\\Delta|>1.96\\times\\mathrm{{SE}}$).}}"
        )
        lines.append(f"\\label{{tab:elppd_{_sc}}}")
        lines.append(r"\end{table}")
        lines.append("")

lines.append(r"\end{document}")

with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(lines))

print(f"Written: {OUTPUT_TEX}")

# Write body-only version for \input in main.tex (no preamble)
body_start = next(i for i, l in enumerate(lines) if l.strip() == r"\begin{document}") + 1
body_end   = next(i for i, l in enumerate(lines) if l.strip() == r"\end{document}")
body_lines = lines[body_start:body_end]
with open(OUTPUT_BODY, "w") as f:
    f.write("\n".join(body_lines))
print(f"Written: {OUTPUT_BODY}")
print(f"  Schemes:           {present_schemes}")
print(f"  Models:            {present_models}")
print(f"  Coverage schemes:  {list(cov_data.keys())}")

# ── Standalone 4-model holdout coverage comparison table ─────────────────────
# Written to coverage_comparison_table.tex for \input in paper/main.tex.
# Shows ONLY holdout validation coverage for the holdout_last_k scheme,
# one column per model variant.  Values >= 95% are bolded.

OUTPUT_COV_COMPARISON = os.path.join(COVERAGE_DIR, "coverage_comparison_table.tex")
_PRIMARY_SCHEME = "holdout_last_k"

if _PRIMARY_SCHEME in cov_data:
    _scheme_cov = cov_data[_PRIMARY_SCHEME]
    _present_cov_models = [m for m in MODEL_ORDER if m in _scheme_cov]

    # Collect and order metric display labels
    _all_labels: set = set()
    for _mc in _scheme_cov.values():
        _all_labels.update(_mc.keys())
    _ordered_labels = [METRIC_LABEL[k] for k in METRIC_ORDER if METRIC_LABEL[k] in _all_labels]
    _exit_labels    = [l for l in _all_labels if "EXIT" in l]
    _extra_labels   = sorted(_all_labels - set(_ordered_labels) - set(_exit_labels))
    _ordered_labels += _extra_labels + _exit_labels

    _n_cov   = len(_present_cov_models)
    _col_cov = "l" + "r" * _n_cov

    ctab = []
    ctab.append(r"\begin{table}[htbp]")
    ctab.append(r"\centering")
    ctab.append(
        r"\caption{95\% HDI holdout validation coverage by metric across model variants "
        r"(Hold-out Last $k$ scheme, 727 player-seasons). "
        r"\textbf{Bold} indicates the model closest to the nominal 95\% level per metric.}"
    )
    ctab.append(r"\label{tab:coverage_basic}")
    ctab.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{" + _col_cov + "}")
    ctab.append(r"\toprule")
    _header_cols = ["Metric"] + [MODEL_LABEL.get(m, m) for m in _present_cov_models]
    ctab.append(" & ".join(_header_cols) + r" \\")
    ctab.append(r"\midrule\addlinespace[2.5pt]")

    for _label in _ordered_labels:
        # Best model = closest to nominal 95% coverage (minimise |coverage - 95|)
        _best_mk, _best_dist = None, float("inf")
        for _mk in _present_cov_models:
            _pcts = _scheme_cov[_mk].get(_label)
            if _pcts is not None:
                _dist = abs(_pcts[0] - 95.0)
                if _dist < _best_dist:
                    _best_dist, _best_mk = _dist, _mk

        _row_cells = [_label]
        for _mk in _present_cov_models:
            _pcts = _scheme_cov[_mk].get(_label)
            if _pcts is None:
                _row_cells.append("---")
            else:
                _v = _pcts[0]
                _text = f"{_v:.1f}\\%"
                _row_cells.append(f"\\textbf{{{_text}}}" if _mk == _best_mk else _text)
        ctab.append(" & ".join(_row_cells) + r" \\")

    ctab.append(r"\bottomrule")
    ctab.append(r"\end{tabular}}")
    ctab.append(r"\end{table}")

    with open(OUTPUT_COV_COMPARISON, "w") as f:
        f.write("\n".join(ctab) + "\n")
    print(f"Written: {OUTPUT_COV_COMPARISON}")
else:
    print(f"No coverage data for {_PRIMARY_SCHEME} — skipping {OUTPUT_COV_COMPARISON}")
