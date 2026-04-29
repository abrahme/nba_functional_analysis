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
    "nba_convex_max_tvlinearlvm",
    "nba_convex_max_tvlinearlvm_AR",
    "nba_convex_max_tvlinearlvm_injury",
    "nba_naive",
]
MODEL_LABEL = {
    "nba_convex_max_tvlinearlvm":         "Base",
    "nba_convex_max_tvlinearlvm_AR":      "AR",
    "nba_convex_max_tvlinearlvm_injury":  "Injury",
    "nba_naive":                          "Naive AR",
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

    lines.append(r"\begin{table}[ht]")
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

    lines.append(r"\begin{table}[ht]")
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

        lines.append(r"\begin{table}[ht]")
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
