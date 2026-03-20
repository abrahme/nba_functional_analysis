from __future__ import annotations

from pathlib import Path
import re
import math

import pandas as pd


TARGET_SPLITS = (
    "holdout",
    "holdout_injured",
    "holdout_non_injured",
    "non_holdout",
    "non_holdout_injured",
    "non_holdout_non_injured",
)
SPLIT_TITLES = {
    "holdout": "Holdout",
    "holdout_injured": "Holdout (injured)",
    "holdout_non_injured": "Holdout (non-injured)",
    "non_holdout": "Non-holdout",
    "non_holdout_injured": "Non-holdout (injured)",
    "non_holdout_non_injured": "Non-holdout (non-injured)",
}


def _clean_cell(value: str) -> str:
    return value.strip().replace("\\_", "_")


def _parse_table_rows(tex_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in tex_text.splitlines():
        line = raw_line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue

        parts = [_clean_cell(part) for part in line[:-2].split("&")]
        if len(parts) < 5:
            continue

        split_value = parts[0]
        if split_value not in TARGET_SPLITS:
            continue

        metric = parts[1]
        try:
            bias = float(parts[2])
            rmse = float(parts[3])
        except ValueError:
            continue

        # avg_log_loss = math.nan
        # n_obs = 0
        avg_log_loss = float(parts[4])
        n_obs = int(float(parts[5]))

        # try:
        #     if len(parts) >= 9:
        #         avg_log_loss = float(parts[4])
        #         n_obs = int(float(parts[5]))
        #         avg_log_loss = float(parts[8])
        #     elif len(parts) >= 7:
        #         n_obs = int(float(parts[4]))

        #     else:
        #         n_obs = int(float(parts[4]))
        # except ValueError:
        #     continue

        rows.append(
            {
                "split": split_value,
                "metric": metric,
                "bias": bias,
                "rmse": rmse,
                "avg_log_loss": avg_log_loss,
                "n_obs": n_obs,
            }
        )
    return rows


def _label_model(model_name: str) -> str | None:
    if "tvlinearlvm_injury" in model_name:
        return "concave + AR + injury"
    if "tvlinearlvm_AR" in model_name:
        return "concave + AR"
    if "tvlinearlvm" in model_name:
        return "concave"
    if "naive" in model_name:
        return "naive"


def _format_wide_table(df: pd.DataFrame, value_column: str, model_order: list[str], use_abs_for_best: bool) -> pd.DataFrame:
    metric_order = df["metric"].drop_duplicates().tolist()
    wide = (
        df.pivot_table(index="metric", columns="model", values=value_column, aggfunc="mean")
        .reindex(metric_order)
        .reindex(columns=[col for col in model_order if col in df["model"].unique()])
    )

    formatted = wide.copy().astype(object)
    for metric_name, row in wide.iterrows():
        if use_abs_for_best:
            best_value = row.abs().min(skipna=True)
        else:
            best_value = row.min(skipna=True)
        for model_col, value in row.items():
            if pd.isna(value):
                formatted.at[metric_name, model_col] = ""
            else:
                value_str = f"{value:.4f}"
                if use_abs_for_best:
                    is_best = abs(abs(value) - best_value) <= 1e-12
                else:
                    is_best = abs(value - best_value) <= 1e-12
                if is_best:
                    value_str = f"\\textbf{{{value_str}}}"
                formatted.at[metric_name, model_col] = value_str

    return formatted.reset_index().rename(columns={"metric": "Metric"})


def _format_scalar(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def _format_pair(left_value: float, right_value: float, use_abs_for_best: bool) -> tuple[str, str]:
    left_str = _format_scalar(left_value)
    right_str = _format_scalar(right_value)

    if pd.isna(left_value) or pd.isna(right_value):
        return left_str, right_str

    left_score = abs(left_value) if use_abs_for_best else left_value
    right_score = abs(right_value) if use_abs_for_best else right_value
    best_score = min(left_score, right_score)

    if abs(left_score - best_score) <= 1e-12:
        left_str = f"\\textbf{{{left_str}}}"
    if abs(right_score - best_score) <= 1e-12:
        right_str = f"\\textbf{{{right_str}}}"
    return left_str, right_str


def _build_non_holdout_side_by_side_table(df_rows: pd.DataFrame) -> pd.DataFrame | None:
    injured_df = (
        df_rows[df_rows["split"] == "non_holdout_injured"]
        .drop_duplicates(subset=["metric"], keep="last")
        .set_index("metric")
    )
    non_injured_df = (
        df_rows[df_rows["split"] == "non_holdout_non_injured"]
        .drop_duplicates(subset=["metric"], keep="last")
        .set_index("metric")
    )

    metric_order = (
        df_rows[df_rows["split"].isin(["non_holdout_injured", "non_holdout_non_injured"])]["metric"]
        .drop_duplicates()
        .tolist()
    )
    if not metric_order:
        return None

    table_rows: list[dict[str, str]] = []
    for metric in metric_order:
        inj_bias = injured_df["bias"].get(metric, math.nan) if "bias" in injured_df.columns else math.nan
        non_inj_bias = non_injured_df["bias"].get(metric, math.nan) if "bias" in non_injured_df.columns else math.nan
        inj_rmse = injured_df["rmse"].get(metric, math.nan) if "rmse" in injured_df.columns else math.nan
        non_inj_rmse = non_injured_df["rmse"].get(metric, math.nan) if "rmse" in non_injured_df.columns else math.nan
        inj_log = injured_df["avg_log_loss"].get(metric, math.nan) if "avg_log_loss" in injured_df.columns else math.nan
        non_inj_log = non_injured_df["avg_log_loss"].get(metric, math.nan) if "avg_log_loss" in non_injured_df.columns else math.nan

        bias_inj_str, bias_non_inj_str = _format_pair(inj_bias, non_inj_bias, use_abs_for_best=True)
        rmse_inj_str, rmse_non_inj_str = _format_pair(inj_rmse, non_inj_rmse, use_abs_for_best=False)
        log_inj_str, log_non_inj_str = _format_pair(inj_log, non_inj_log, use_abs_for_best=False)

        table_rows.append(
            {
                "Metric": metric,
                "Bias (Injured)": bias_inj_str,
                "Bias (Non-Injured)": bias_non_inj_str,
                "RMSE (Injured)": rmse_inj_str,
                "RMSE (Non-Injured)": rmse_non_inj_str,
                "LogLoss (Injured)": log_inj_str,
                "LogLoss (Non-Injured)": log_non_inj_str,
            }
        )

    return pd.DataFrame(table_rows)


def main() -> None:
    coverage_dir = Path(__file__).resolve().parent

    metric_specs = {
        "rmse": {
            "value_column": "rmse",
            "use_abs_for_best": False,
            "caption_suffix": "RMSE",
            "best_suffix": "lowest value",
        },
        "bias": {
            "value_column": "bias",
            "use_abs_for_best": True,
            "caption_suffix": "bias",
            "best_suffix": "lowest absolute value",
        },
        "log_loss": {
            "value_column": "avg_log_loss",
            "use_abs_for_best": False,
            "caption_suffix": "average log loss",
            "best_suffix": "lowest value",
        },
    }

    output_files: dict[tuple[str, str], Path] = {}
    for split_name in TARGET_SPLITS:
        for metric_name in metric_specs:
            output_files[(split_name, metric_name)] = coverage_dir / f"all_models_{split_name}_{metric_name}.tex"

    # Keep the original "holdout" file names for backward compatibility.
    output_files[("holdout", "rmse")] = coverage_dir / "all_models_holdout_rmse.tex"
    output_files[("holdout", "bias")] = coverage_dir / "all_models_holdout_bias.tex"
    output_files[("holdout", "log_loss")] = coverage_dir / "all_models_holdout_log_loss.tex"

    legacy_rmse_file = coverage_dir / "all_models_holdout_bias_rmse.tex"
    generated_outputs = {path.name for path in output_files.values()}
    generated_outputs.add(legacy_rmse_file.name)

    records: list[dict[str, object]] = []
    side_by_side_written: list[Path] = []
    skipped_no_rows: list[str] = []
    skipped_unmapped_files: list[str] = []

    for tex_path in sorted(coverage_dir.glob("*.tex")):
        if tex_path.name in generated_outputs:
            continue
        if tex_path.name.endswith("_non_holdout_injury_side_by_side.tex"):
            continue

        tex_text = tex_path.read_text(encoding="utf-8")
        holdout_rows = _parse_table_rows(tex_text)
        if not holdout_rows:
            skipped_no_rows.append(tex_path.name)
            continue

        model_name = re.sub(r"\.tex$", "", tex_path.name)
        model_rows_df = pd.DataFrame(holdout_rows)
        side_by_side_df = _build_non_holdout_side_by_side_table(model_rows_df)
        if side_by_side_df is not None:
            side_by_side_path = coverage_dir / f"{model_name}_non_holdout_injury_side_by_side.tex"
            side_by_side_latex = side_by_side_df.to_latex(
                index=False,
                caption=(
                    f"Non-holdout injured vs non-injured side-by-side metrics for {model_name} "
                    "(bold = better within each injured/non-injured pair; bias by absolute value)"
                ),
                label=f"tab:{model_name}_non_holdout_injury_side_by_side",
                escape=False,
            )
            side_by_side_path.write_text(side_by_side_latex, encoding="utf-8")
            side_by_side_written.append(side_by_side_path)

        model_label = _label_model(model_name)
        if model_label is None:
            skipped_unmapped_files.append(tex_path.name)
            continue
        for row in holdout_rows:
            records.append(
                {
                    "split": row["split"],
                    "model": model_label,
                    "metric": row["metric"],
                    "bias": row["bias"],
                    "rmse": row["rmse"],
                    "avg_log_loss": row["avg_log_loss"],
                    "n_obs": row["n_obs"],
                }
            
            )
            
            

    if not records:
        raise RuntimeError("No holdout rows were found in any .tex coverage tables.")

    df = pd.DataFrame(records)
    model_order = ["concave", "concave + AR", "concave + AR + injury", "naive"]

    written_files: list[Path] = []
    for split_name in TARGET_SPLITS:
        df_split = df[df["split"] == split_name]
        if df_split.empty:
            print(f"No rows found for split '{split_name}', skipping output generation for this split.")
            continue

        split_title = SPLIT_TITLES[split_name]
        for metric_name, metric_spec in metric_specs.items():
            formatted = _format_wide_table(
                df_split,
                metric_spec["value_column"],
                model_order,
                use_abs_for_best=metric_spec["use_abs_for_best"],
            )
            latex_text = formatted.to_latex(
                index=False,
                caption=(
                    f"{split_title} {metric_spec['caption_suffix']} by metric across models "
                    f"(bold = {metric_spec['best_suffix']})"
                ),
                label=f"tab:{split_name}_{metric_name}_all_models",
                escape=False,
            )
            output_path = output_files[(split_name, metric_name)]
            output_path.write_text(latex_text, encoding="utf-8")
            written_files.append(output_path)

    if legacy_rmse_file.exists():
        legacy_rmse_file.unlink()

    for output_path in written_files:
        print(f"Wrote {output_path}")
    for output_path in side_by_side_written:
        print(f"Wrote {output_path}")
    if not legacy_rmse_file.exists():
        print(f"Removed legacy file {legacy_rmse_file}")
    if skipped_no_rows:
        print("Skipped files without target split rows:")
        for name in skipped_no_rows:
            print(f"  - {name}")
    if skipped_unmapped_files:
        print("Skipped files for cross-model aggregation (unmapped model label):")
        for name in skipped_unmapped_files:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
