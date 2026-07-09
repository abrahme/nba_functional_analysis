"""Combine q-ablation holdout CSVs into LaTeX comparison tables.

Reads every  nba_convex_max_tvlinearlvm_q{q}_vs_<scheme>_*.csv  file from this
directory and produces  ablation_holdout_table.tex  with one pair of subtables
(RMSE/bias and log-loss) per holdout scheme — metrics as rows, q values as
columns.

Usage (run from repo root):
    python model_output/model_plots/coverage/combine_ablation_tables.py
"""

import math
import os
import re
import pandas as pd

COVERAGE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TEX   = os.path.join(COVERAGE_DIR, "ablation_holdout_table.tex")
OUTPUT_BODY  = os.path.join(COVERAGE_DIR, "ablation_holdout_body.tex")

SCHEME_ORDER = ["holdout_last_k", "holdout_first_k", "random_interior", "holdout_peak"]
SCHEME_LABEL = {
    "holdout_last_k":  "Hold-out Last $k$",
    "holdout_first_k": "Hold-out First $k$",
    "random_interior": "Random Interior",
    "holdout_peak":    "Hold-out Peak",
}

Q_VALUES = [5, 10, 15, 20]
Q_KEYS   = [f"nba_convex_max_tvlinearlvm_q{q}" for q in Q_VALUES]
Q_LABEL  = {f"nba_convex_max_tvlinearlvm_q{q}": f"$q={q}$" for q in Q_VALUES}

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

# ── Load all q-ablation CSVs ──────────────────────────────────────────────────

_SCHEME_SUFFIXES = ("_holdout_last_k", "_holdout_first_k", "_random_interior", "_holdout_peak")

records = []
pattern = re.compile(r"^(.+)_vs_(.+?)_f\d+_k\d+_s\d+\.csv$")

for fname in os.listdir(COVERAGE_DIR):
    m = pattern.match(fname)
    if not m:
        continue
    model_name, scheme = m.group(1), m.group(2)
    for suffix in _SCHEME_SUFFIXES:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
            break
    if model_name not in Q_KEYS:
        continue
    df = pd.read_csv(os.path.join(COVERAGE_DIR, fname))
    df = df[df["split"] == "holdout"].copy()
    df["model"]  = model_name
    df["scheme"] = scheme
    records.append(df)

if not records:
    print("No q-ablation coverage CSVs found — run the ablation jobs first.")
    raise SystemExit(1)

data = pd.concat(records, ignore_index=True)
present_q       = [k for k in Q_KEYS    if k in data["model"].unique()]
present_schemes = [s for s in SCHEME_ORDER if s in data["scheme"].unique()]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sci(val, decimals=2):
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / 10 ** exp
    return f"${mantissa:.{decimals}f}\\times10^{{{exp}}}$"

def _fmt_num(val, decimals=3):
    if pd.isna(val):
        return "---"
    if abs(val) >= 1e4:
        return _sci(val)
    return f"{val:.{decimals}f}"

def _fmt_signed(val, decimals=3):
    if pd.isna(val):
        return "---"
    if abs(val) >= 1e4:
        return ("+" if val >= 0 else "") + _sci(val)
    return f"{val:+.{decimals}f}"

def fmt_cell(rmse, bias, bold=False):
    if pd.isna(rmse) and pd.isna(bias):
        return "---"
    text = f"{_fmt_num(rmse)} ({_fmt_signed(bias)})"
    return f"\\textbf{{{text}}}" if bold else text

def best_q_for_metric(sub, metric, col):
    vals = {k: sub[(sub["model"] == k) & (sub["metric"] == metric)][col].iloc[0]
            for k in present_q
            if not sub[(sub["model"] == k) & (sub["metric"] == metric)].empty
            and not pd.isna(sub[(sub["model"] == k) & (sub["metric"] == metric)][col].iloc[0])}
    return min(vals, key=vals.__getitem__) if vals else None

# ── Build LaTeX ───────────────────────────────────────────────────────────────

n_q      = len(present_q)
col_spec = "l" + "r" * n_q
header   = ["Metric"] + [Q_LABEL[k] for k in present_q]

lines = []
lines.append(r"\documentclass{article}")
lines.append(r"\usepackage{booktabs}")
lines.append(r"\usepackage{multirow}")
lines.append(r"\usepackage{graphicx}")
lines.append(r"\usepackage[margin=1in]{geometry}")
lines.append(r"\begin{document}")
lines.append("")

for scheme in present_schemes:
    sub = data[data["scheme"] == scheme]
    if sub.empty:
        continue

    scheme_lbl       = SCHEME_LABEL.get(scheme, scheme)
    present_metrics  = [m for m in METRIC_ORDER if m in sub["metric"].unique()]
    extra_metrics    = [m for m in sub["metric"].unique() if m not in METRIC_ORDER]

    # ── RMSE / bias ───────────────────────────────────────────────────────────
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"  & " + " & ".join([r"\small RMSE (bias)"] * n_q) + r" \\")
    lines.append(r"\midrule")

    for metric in present_metrics + extra_metrics:
        best = best_q_for_metric(sub, metric, "rmse")
        row_cells = [METRIC_LABEL.get(metric, metric.upper())]
        for k in present_q:
            row = sub[(sub["model"] == k) & (sub["metric"] == metric)]
            row_cells.append("---" if row.empty else
                             fmt_cell(row["rmse"].iloc[0], row["bias"].iloc[0], bold=(k == best)))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        f"\\caption{{Holdout RMSE and bias (in parentheses) by metric and latent dimension $q$ --- "
        f"{scheme_lbl} scheme. Best $q$ per metric in bold.}}"
    )
    lines.append(f"\\label{{tab:ablation_rmse_{scheme}}}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Log-loss ──────────────────────────────────────────────────────────────
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"  & " + " & ".join([r"\small Avg log-loss"] * n_q) + r" \\")
    lines.append(r"\midrule")

    for metric in present_metrics + extra_metrics:
        best = best_q_for_metric(sub, metric, "avg_log_loss")
        row_cells = [METRIC_LABEL.get(metric, metric.upper())]
        for k in present_q:
            row = sub[(sub["model"] == k) & (sub["metric"] == metric)]
            if row.empty or pd.isna(row["avg_log_loss"].iloc[0]):
                row_cells.append("---")
            else:
                val = _fmt_num(row["avg_log_loss"].iloc[0], decimals=4)
                row_cells.append(f"\\textbf{{{val}}}" if k == best else val)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        f"\\caption{{Holdout average log-loss by metric and latent dimension $q$ --- "
        f"{scheme_lbl} scheme. Best $q$ per metric in bold.}}"
    )
    lines.append(f"\\label{{tab:ablation_logloss_{scheme}}}")
    lines.append(r"\end{table}")
    lines.append("")

lines.append(r"\end{document}")

with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(lines))
print(f"Written: {OUTPUT_TEX}")

body_start = next(i for i, l in enumerate(lines) if l.strip() == r"\begin{document}") + 1
body_end   = next(i for i, l in enumerate(lines) if l.strip() == r"\end{document}")
with open(OUTPUT_BODY, "w") as f:
    f.write("\n".join(lines[body_start:body_end]))
print(f"Written: {OUTPUT_BODY}")
print(f"  Q values present : {[Q_LABEL[k] for k in present_q]}")
print(f"  Schemes present  : {present_schemes}")
