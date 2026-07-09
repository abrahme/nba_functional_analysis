# Prior-predictive-check plots (r-new container).
# Consumes the parquet files written by prior_check.py (mcmc-analysis container) and renders the
# diagnostic plots. Plotting is separated from the Python ELPPD computation so it runs in the R
# container, matching the rest of the pipeline (run_pipeline.sh phase 0).
#
# Usage: Rscript data_analysis/prior_check.r <prior_dir>
#   <prior_dir> is the model's prior output dir, e.g.
#   model_output/nba_convex_max_tvlinearlvm/stratified_next_k/prior
# For a sweep, point it at the sweep base dir (containing sweep_summary.parquet and <knob>=<val>/ subdirs).

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(glue)
  library(arrow)
})

# `posterior_plot_names` — the canonical per-player set model_diagnostics.r uses (single source of
# truth in diagnostics_utils.r); prior plots default to the same players.
source("data_analysis/diagnostics_utils.r")

args <- commandArgs(trailingOnly = TRUE)
pc_dir <- if (length(args) >= 1) args[1] else stop("Usage: Rscript data_analysis/prior_check.r <prior_dir> [plot_players]")
# Optional 2nd arg selects which players get per-player PNGs: "all", or a comma-separated list.
# Default: plot all present if few, else a notable subset (the parquet may hold every player).
plot_players_arg <- if (length(args) >= 2) args[2] else NA_character_
plots_dir <- file.path(pc_dir, "plots")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

rd <- function(name) {
  p <- file.path(pc_dir, name)
  if (file.exists(p)) suppressMessages(read_parquet(p)) else NULL
}
save_plot <- function(g, name, w = 9, h = 6) {
  ggsave(file.path(plots_dir, name), g, width = w, height = h, dpi = 120)
  message(glue("[prior_check.r] wrote {file.path(plots_dir, name)}"))
}

# ── 1. Per-player prior-predictive production curves (the main view) ──────────
# The prior analog of the posterior-predictive career-trajectory plots in model_diagnostics.r
# (plot_posterior): per metric, a gray prior-predictive interval band + mean line + observed points,
# faceted, with the same metric relabeling (GP%, MPG, FG2%…) and pct_minutes scaled to MPG (×48).
# A spaghetti variant (one line per draw) is also written for players who want the raw-draw view.

# Match model_diagnostics.r's facet labels exactly so prior and posterior panels line up.
relabel_metric <- function(m) {
  m <- toupper(m)
  dplyr::case_when(
    m == "GAMES"       ~ "GP%",
    m == "FG2M"        ~ "FG2%",
    m == "FG3M"        ~ "FG3%",
    m == "FTM"         ~ "FT%",
    m == "PCT_MINUTES" ~ "MPG",
    .default = m)
}

draws <- rd("prior_player_draws.parquet")
pobs  <- rd("prior_player_obs.parquet")
if (!is.null(draws)) {
  ppdir <- file.path(plots_dir, "player_prior")
  dir.create(ppdir, recursive = TRUE, showWarnings = FALSE)
  present <- unique(as.character(draws$name))
  # Which players to render PNGs for: explicit 2nd arg ("all" or comma list), else the
  # model_diagnostics player set (posterior_plot_names) intersected with what's in the parquet.
  to_plot <- if (!is.na(plot_players_arg)) {
    if (identical(plot_players_arg, "all")) present else trimws(strsplit(plot_players_arg, ",")[[1]])
  } else {
    intersect(posterior_plot_names, present)
  }
  to_plot <- intersect(to_plot, present)
  message(glue("[prior_check.r] per-player plots for {length(to_plot)} players (of {length(present)} in parquet)"))

  # pct_minutes is exported as a proportion; ×48 -> MPG to match the posterior plots' MPG panel.
  draws  <- draws |> mutate(value = if_else(metric == "pct_minutes", value * 48, value),
                            metric = relabel_metric(metric))
  if (!is.null(pobs))
    pobs <- pobs  |> mutate(obs_value = if_else(metric == "pct_minutes", obs_value * 48, obs_value),
                            metric = relabel_metric(metric))

  # Per (name, metric, age): prior-predictive 95% band + mean — the band/line analog of plot_posterior.
  band <- draws |>
    group_by(name, metric, age) |>
    summarize(lower = quantile(value, 0.025, na.rm = TRUE),
              upper = quantile(value, 0.975, na.rm = TRUE),
              pred_mean = mean(value, na.rm = TRUE), .groups = "drop")

  # OBPM/DBPM have an identity link, so the unbounded concave decay sends the prior band to
  # absurd values far from the peak. Zoom those two panels to a fixed [-15, 15] window: clamp the
  # band into range so the ribbon fills the view, and (with free_y) pin the axis via a geom_blank
  # at +/-15 (no ggh4x available for per-facet scales). Spaghetti draws outside the window are set
  # to NA so the lines simply leave the panel rather than piling on the edge.
  BOUNDED <- c("OBPM", "DBPM"); YLO <- -15; YHI <- 15
  clampw <- function(x) pmin(pmax(x, YLO), YHI)
  band <- band |> mutate(
    lower     = if_else(metric %in% BOUNDED, pmax(lower, YLO), lower),
    upper     = if_else(metric %in% BOUNDED, pmin(upper, YHI), upper),
    pred_mean = if_else(metric %in% BOUNDED, clampw(pred_mean), pred_mean))
  if (!is.null(pobs))
    pobs <- pobs |> mutate(obs_value = if_else(metric %in% BOUNDED, clampw(obs_value), obs_value))
  # geom_blank rows pin the OBPM/DBPM y-axis to exactly [-15, 15] (age is a shared/fixed scale).
  blank_df <- data.frame(metric = rep(BOUNDED, each = 2), age = 25, y = c(YLO, YHI, YLO, YHI))

  for (nm in to_plot) {
    bd <- filter(band, name == nm)
    # (a) posterior-style: gray prior-predictive band + mean line + observed points
    g <- ggplot(bd, aes(x = age)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), fill = "gray", alpha = 0.4) +
      geom_line(aes(y = pred_mean)) +
      geom_blank(data = blank_df, aes(x = age, y = y), inherit.aes = FALSE) +
      facet_wrap(~ metric, scales = "free_y") + theme_bw(base_size = 14) +
      labs(x = "Age", y = "Metric Value") +
      ggtitle(glue("Prior Predictive Career Trajectory: {nm}"))
    if (!is.null(pobs)) {
      po <- filter(pobs, name == nm)
      if (nrow(po) > 0) g <- g + geom_point(data = po, aes(x = age, y = obs_value), colour = "black")
    }
    fn <- file.path(ppdir, glue("{gsub(' ', '_', nm)}.png"))
    ggsave(fn, g, width = 14, height = 10, dpi = 120)
    message(glue("[prior_check.r] wrote {fn}"))

    # (b) spaghetti: one faint line per prior draw (the raw-draw view), same facets
    dd <- filter(draws, name == nm) |>
      mutate(value = if_else(metric %in% BOUNDED & (value < YLO | value > YHI), NA_real_, value))
    gs <- ggplot(dd, aes(x = age, y = value)) +
      geom_line(aes(group = draw), alpha = 0.15, colour = "steelblue") +
      geom_blank(data = blank_df, aes(x = age, y = y), inherit.aes = FALSE) +
      facet_wrap(~ metric, scales = "free_y") + theme_bw(base_size = 14) +
      labs(x = "Age", y = "Metric Value") +
      ggtitle(glue("Prior Predictive Draws: {nm}"))
    if (!is.null(pobs)) {
      po <- filter(pobs, name == nm)
      if (nrow(po) > 0) gs <- gs + geom_point(data = po, aes(x = age, y = obs_value), colour = "black", size = 1)
    }
    fns <- file.path(ppdir, glue("{gsub(' ', '_', nm)}_spaghetti.png"))
    ggsave(fns, gs, width = 14, height = 10, dpi = 120)
    message(glue("[prior_check.r] wrote {fns}"))
  }
}

# ── 2. ELPPD per metric ───────────────────────────────────────────────────────
pm <- rd("prior_elppd_per_metric.parquet")
if (!is.null(pm)) {
  g <- pm |>
    mutate(metric = fct_reorder(metric, elppd_per_obs)) |>
    ggplot(aes(elppd_per_obs, metric, fill = family)) +
    geom_col() +
    labs(title = "Prior predictive ELPPD per observation", x = "ELPPD / obs (higher = better)", y = NULL) +
    theme_minimal()
  save_plot(g, "elppd_per_metric.png")
}

# ── 3. Coverage per metric ────────────────────────────────────────────────────
cov <- rd("prior_ppc_coverage.parquet")
if (!is.null(cov)) {
  g <- cov |>
    pivot_longer(c(cover90, cover50), names_to = "level", values_to = "coverage") |>
    mutate(metric = fct_reorder(metric, coverage)) |>
    ggplot(aes(coverage, metric, fill = level)) +
    geom_col(position = "dodge") +
    geom_vline(xintercept = c(0.5, 0.9), linetype = "dashed", colour = "grey40") +
    labs(title = "Prior-predictive coverage of observed data", x = "fraction of observed inside interval", y = NULL) +
    theme_minimal()
  save_plot(g, "coverage.png")
}

# ── 4. Prior-predictive ribbon vs observed, by age, per metric (pooled) ───────
ba <- rd("prior_ppc_by_age.parquet")
if (!is.null(ba)) {
  g <- ggplot(ba, aes(age)) +
    geom_ribbon(aes(ymin = prior_q05, ymax = prior_q95), fill = "steelblue", alpha = 0.20) +
    geom_ribbon(aes(ymin = prior_q25, ymax = prior_q75), fill = "steelblue", alpha = 0.35) +
    geom_line(aes(y = prior_q50), colour = "steelblue") +
    geom_line(aes(y = obs_mean), colour = "firebrick") +
    facet_wrap(~ metric, scales = "free_y") +
    labs(title = "Prior predictive (blue band/line) vs observed mean (red), by age",
         subtitle = "bands = prior-predictive 50%/90% intervals pooled over players & draws", y = "value") +
    theme_minimal()
  save_plot(g, "ppc_ribbon_by_age.png", w = 14, h = 10)
}

# ── 5. Peak check: prior mu-peak interval vs empirical peak ───────────────────
pk <- rd("prior_peak_check.parquet")
if (!is.null(pk)) {
  g <- pk |>
    mutate(metric = fct_reorder(metric, prior_peak_val_med),
           flag = ifelse(empirical_outside_90, "outside 90%", "inside 90%")) |>
    ggplot(aes(y = metric)) +
    geom_linerange(aes(xmin = prior_peak_val_q05, xmax = prior_peak_val_q95), colour = "steelblue") +
    geom_point(aes(x = prior_peak_val_med), colour = "steelblue") +
    geom_point(aes(x = empirical_peak_val, colour = flag), shape = 4, size = 3) +
    scale_colour_manual(values = c("inside 90%" = "grey30", "outside 90%" = "firebrick")) +
    labs(title = "Prior peak-value interval (blue) vs empirical peak (x)",
         x = "peak value (linear-predictor scale)", y = NULL, colour = NULL) +
    theme_minimal()
  save_plot(g, "peak_check.png")
}

# ── 6. Survival predictive check: prior-pred survival band vs empirical KM ─────
sc <- rd("prior_survival_curve.parquet")
if (!is.null(sc)) {
  g <- ggplot(sc, aes(age)) +
    geom_ribbon(aes(ymin = prior_S_q05, ymax = prior_S_q95), fill = "darkgreen", alpha = 0.20) +
    geom_line(aes(y = prior_S_q50), colour = "darkgreen") +
    geom_line(aes(y = empirical_S), colour = "black", linetype = "dashed") +
    labs(title = "Prior-predictive survival curve (green) vs empirical KM (dashed)",
         subtitle = "P(still active | age); band = prior-predictive 90% interval over draws",
         x = "age", y = "survival S(age)") +
    theme_minimal()
  save_plot(g, "survival_curve.png")
}

# ── 7. ELPPD breakdowns ───────────────────────────────────────────────────────
for (bk in c("age", "position", "minutes", "stratum")) {
  d <- rd(glue("prior_elppd_by_{bk}.parquet"))
  if (is.null(d) || nrow(d) == 0) next
  xcol <- names(d)[1]
  g <- ggplot(d, aes(x = .data[[xcol]], y = mean)) + geom_col(fill = "darkslategray") +
    labs(title = glue("Mean prior ELPPD per obs by {bk}"), x = xcol, y = "mean lppd") + theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_plot(g, glue("elppd_by_{bk}.png"))
}

# ── 8. Sweep comparison (if present) ──────────────────────────────────────────
sw <- rd("sweep_summary.parquet")
if (!is.null(sw)) {
  knob_cols <- names(sw)[startsWith(names(sw), "knob_")]
  if (length(knob_cols) >= 1) {
    kc <- knob_cols[1]
    save_plot(ggplot(sw, aes(x = .data[[kc]], y = elppd_total)) + geom_line() + geom_point() +
                labs(title = "Sweep: total prior ELPPD vs knob", x = kc, y = "ELPPD total") + theme_minimal(),
              "sweep_elppd.png")
    save_plot(ggplot(sw, aes(x = .data[[kc]], y = mean_cover90)) + geom_line() + geom_point() +
                geom_hline(yintercept = 0.9, linetype = "dashed", colour = "firebrick") +
                labs(title = "Sweep: mean 90% coverage vs knob", x = kc, y = "mean cover90") + theme_minimal(),
              "sweep_coverage.png")
  }
}

message("[prior_check.r] done")
