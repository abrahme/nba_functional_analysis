library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
library(purrr)
library(glue)
library(ggrepel)
library(ggridges)
library(pheatmap)
library(gt)
library(ggbeeswarm)
library(ggdist)
library(uwot)
library(patchwork)
library(arrow)
library(fdasrvf)
library(dbscan)

options(expressions = 500000)

source("data_analysis/diagnostics_utils.r")

args             <- commandArgs(trailingOnly = TRUE)
model_dir        <- if (length(args) >= 1) args[1] else stop("Usage: Rscript data_analysis/coverage.r <model_dir> [validation_year]")
validation_year  <- if (length(args) >= 2) as.integer(args[2]) else 2021L
min_minutes_threshold <- 100L   # discard player-seasons with fewer than this many minutes

# ── Inputs ────────────────────────────────────────────────────────────────────
posterior_data <- read_parquet(file.path(model_dir, "posterior_ar.parquet")) |>
  mutate(value = if_else(metric == "pct_minutes", value * 48, value))

conditional_parquet_path <- file.path(model_dir, "posterior_ar_conditional.parquet")
has_conditional <- file.exists(conditional_parquet_path)
if (has_conditional) {
  posterior_conditional_data <- read_parquet(conditional_parquet_path) |>
    mutate(value = if_else(metric == "pct_minutes", value * 48, value)) |>
    rename(value_conditional = value)
}

posterior_peaks     <- read_parquet_if_exists(file.path(model_dir, "posterior_peaks_ar.parquet"))
posterior_metric_ll <- read_parquet_if_exists(file.path(model_dir, "posterior_metric_log_loss.parquet"))

posterior_retirement_data <- open_dataset(file.path(model_dir, "posterior_exit_age_sample.parquet")) |>
  filter(measure == "exit_age_sample", scenario == "observed", exit_censored == 0,
         conditioning_label == "entrance") |>
  select(player, value, observed_exit_age) |>
  collect()

posterior_survival_data <- open_dataset(file.path(model_dir, "posterior_exit_survival.parquet")) |>
  filter(measure == "exit_survival", scenario == "observed") |>
  select(player, age, value, observed_exit_age, exit_censored) |>
  collect()

# ── Shared posterior + injury data prep (see diagnostics_utils.r) ─────────────
data        <- make_player_data(posterior_data)
injury_data <- build_injury_data(data)

cov_ctx <- build_joined_data(posterior_data, injury_data, data, posterior_peaks,
                             model_dir, validation_year)
joined_data     <- cov_ctx$joined_data
train_years_df  <- cov_ctx$train_years_df
holdout_players <- cov_ctx$holdout_players
holdout_idx     <- cov_ctx$holdout_idx

plots_dir <- file.path(model_dir, "plots")
dir.create(file.path(plots_dir, "coverage"), recursive = TRUE, showWarnings = FALSE)

# ── Coverage / bias / MSE / log-loss diagnostics ──────────────────────────────
validation_coverage_df <- joined_data |> filter(split == "holdout", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |> group_by(metric, player, age) |>
                          summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"], upper = HDInterval::hdi(value, credMass = 0.95)["upper"], obs_value = first(obs_value), year = min(year), posterior_mean = mean(value, na.rm = TRUE) ) |>
                          ungroup() |>

    mutate(
    validation_coverage = between(obs_value, lower, upper)) |> filter(!is.na(obs_value)) |>
    mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))

in_sample_coverage_df <- joined_data |> filter(split == "train", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |> group_by(metric, player, age) |> summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(value, credMass = 0.95)["upper"], obs_value = first(obs_value), year = min(year)) |> ungroup() |>
    mutate(
    in_sample_coverage = between(obs_value, lower, upper)) |> ungroup() |> filter(!is.na(obs_value)) |>
    mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))

player_exit_meta <- data |>
  group_by(id) |>
  summarize(observed_exit_year = max(year, na.rm = TRUE),
            years_played = n(),
            .groups = "drop")

exit_age_coverage_df <- posterior_retirement_data |>
  group_by(player) |>
  summarize(
    lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
    obs_value = first(observed_exit_age),
    posterior_mean = mean(value, na.rm = TRUE),
    .groups = "drop"
  ) |>
  inner_join(player_exit_meta, by = c("player" = "id")) |>
  mutate(
    metric = "EXIT_AGE",
    exit_age_coverage = between(obs_value, lower, upper)
  ) |>
  (\(df) if (!is.null(holdout_players))
    left_join(df, holdout_players, by = "player") |> mutate(exit_split = replace_na(exit_split, "train"))
  else
    mutate(df, exit_split = if_else(observed_exit_year > validation_year, "holdout", "train"))
  )()

retirement_validation_summary <- joined_data |>
  filter(split == "holdout" & metric == "retirement") |>
  filter(!is.na(obs_value)) |>
  group_by(metric) |>
  summarize(validation_coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop") |>
  mutate(metric = toupper(metric))

retirement_in_sample_summary <- joined_data |>
  filter(split == "train" & metric == "retirement") |>
  filter(!is.na(obs_value)) |>
  group_by(metric) |>
  summarize(in_sample_coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop") |>
  mutate(metric = toupper(metric))

exit_age_validation_summary <- exit_age_coverage_df |>
  filter(exit_split == "holdout") |>
  summarize(metric = "EXIT_AGE", validation_coverage = mean(exit_age_coverage, na.rm = TRUE))

exit_age_in_sample_summary <- exit_age_coverage_df |>
  filter(exit_split == "train") |>
  summarize(metric = "EXIT_AGE", in_sample_coverage = mean(exit_age_coverage, na.rm = TRUE))

validation_coverage_summary <- validation_coverage_df |>
  group_by(metric) |>
  summarize(validation_coverage = mean(validation_coverage, na.rm = TRUE), .groups = "drop") |>
  bind_rows(retirement_validation_summary, exit_age_validation_summary)

in_sample_coverage_summary <- in_sample_coverage_df |>
  group_by(metric) |>
  summarize(in_sample_coverage = mean(in_sample_coverage, na.rm = TRUE), .groups = "drop") |>
  bind_rows(retirement_in_sample_summary, exit_age_in_sample_summary)

# Per-stratum coverage (stratified_next_k only — detected via stratum column in holdout_indices.csv)
if (!is.null(holdout_idx) && "stratum" %in% names(holdout_idx)) {
  stratum_age_labels <- c("1" = "Ages 23-24", "2" = "Ages 25-26", "3" = "Ages 27-28",
                          "4" = "Ages 29-30", "5" = "Ages 31-32", "6" = "Ages 33-34")
  stratum_coverage_df <- validation_coverage_df |>
    left_join(holdout_idx |> select(player, age, stratum), by = c("player", "age")) |>
    filter(!is.na(stratum)) |>
    mutate(stratum_label = stratum_age_labels[as.character(stratum)]) |>
    group_by(stratum, stratum_label, metric) |>
    summarize(validation_coverage = mean(validation_coverage, na.rm = TRUE),
              n_obs = n(), .groups = "drop") |>
    arrange(stratum, metric)

  write.csv(stratum_coverage_df,
            file.path(model_dir, "stratum_coverage.csv"), row.names = FALSE)

  stratum_cov_plt <- stratum_coverage_df |>
    ggplot(aes(x = metric, y = validation_coverage, fill = stratum_label)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 0.95, linetype = "dashed", colour = "black") +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Validation Coverage by Age Stratum",
         x = "Metric", y = "95% HDI Coverage", fill = "Age Stratum") +
    theme_bw(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(model_dir, "stratum_coverage.png"), stratum_cov_plt, width = 14, height = 6)
}

# Per-stratum ELPPD (written by model_export.py for stratified_next_k)
stratum_elppd_path <- file.path(model_dir, "stratum_elppd.csv")
if (file.exists(stratum_elppd_path)) {
  stratum_elppd <- read.csv(stratum_elppd_path)
  stratum_age_labels <- c("1" = "Ages 23-24", "2" = "Ages 25-26", "3" = "Ages 27-28",
                          "4" = "Ages 29-30", "5" = "Ages 31-32", "6" = "Ages 33-34")
  stratum_elppd_all <- stratum_elppd |>
    filter(metric == "all") |>
    mutate(
      stratum      = as.integer(sub("holdout_stratum_", "", split)),
      stratum_label = stratum_age_labels[as.character(stratum)],
      se_per_obs   = elppd_se / n_obs
    ) |>
    arrange(stratum)

  stratum_elppd_plt <- stratum_elppd_all |>
    ggplot(aes(x = stratum_label, y = elppd_per_obs)) +
    geom_col(fill = "#4D9DE0") +
    geom_errorbar(aes(ymin = elppd_per_obs - se_per_obs,
                      ymax = elppd_per_obs + se_per_obs), width = 0.3) +
    labs(title = "ELPPD per Observation by Age Stratum",
         x = "Age Stratum", y = "ELPPD / obs") +
    theme_bw(base_size = 14) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(model_dir, "stratum_elppd.png"), stratum_elppd_plt, width = 8, height = 5)

  write.csv(stratum_elppd_all |> select(stratum, stratum_label, elppd_per_obs, elppd_se, n_obs),
            file.path(model_dir, "stratum_elppd_summary.csv"), row.names = FALSE)
}

# Conditional coverage: conditions on observed holdout games and pct_minutes as exposures,
# isolating metric-rate predictive uncertainty from exposure uncertainty.
if (has_conditional) {
  rename_metrics <- function(df) {
    df |> mutate(metric = toupper(metric),
                 metric = case_when(metric == "GAMES" ~ "GP%",
                                    metric == "FG2M"  ~ "FG2%",
                                    metric == "FG3M"  ~ "FG3%",
                                    metric == "FTM"   ~ "FT%",
                                    metric == "PCT_MINUTES" ~ "MPG",
                                    .default = metric))
  }

  joined_data_conditional <- joined_data |>
    left_join(
      posterior_conditional_data |>
        select(chain, sample, player, metric, age, value_conditional),
      by = c("chain", "sample", "player", "metric", "age")
    )

  conditional_coverage_df <- joined_data_conditional |>
    filter(split == "holdout", metric != "retirement",
           is.na(minutes) | minutes >= min_minutes_threshold) |>
    group_by(metric, player, age) |>
    summarize(
      lower          = HDInterval::hdi(value_conditional, credMass = 0.95)["lower"],
      upper          = HDInterval::hdi(value_conditional, credMass = 0.95)["upper"],
      obs_value      = first(obs_value),
      year           = min(year),
      posterior_mean = mean(value_conditional, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(conditional_coverage = between(obs_value, lower, upper)) |>
    filter(!is.na(obs_value)) |>
    rename_metrics()

  conditional_coverage_summary <- conditional_coverage_df |>
    group_by(metric) |>
    summarize(conditional_coverage = mean(conditional_coverage, na.rm = TRUE), .groups = "drop")
}

coverage_base_tbl <- validation_coverage_summary |>
  inner_join(in_sample_coverage_summary, by = "metric")
if (has_conditional) {
  coverage_base_tbl <- coverage_base_tbl |>
    left_join(conditional_coverage_summary, by = "metric")
  plt_cols <- c(in_sample_coverage   = "In-Sample",
                validation_coverage  = "Validation (marginal)",
                conditional_coverage = "Validation (conditional)")
} else {
  plt_cols <- c(in_sample_coverage  = "In-Sample",
                validation_coverage = "Validation")
}
coverage_plt_basic <- coverage_base_tbl |>
                      pivot_longer(cols = names(plt_cols), names_to = "coverage_type", values_to = "Coverage") |>
                      mutate(coverage_type = recode(coverage_type, !!!plt_cols)) |>
                      filter(!is.na(Coverage)) |>
                      ggplot(aes(x = coverage_type, y = Coverage, fill = coverage_type)) +
                      geom_col(position = "dodge") + facet_wrap(~ metric, scales = "fixed") +
                      coord_cartesian(ylim = c(0, 1)) +
                      theme_bw(base_size = 14) + scale_fill_brewer(palette = "Set1") +
                      ggtitle("Per Metric Coverage") +
                      labs(x = NULL, fill = "Coverage Type") +
                      theme(axis.text.x = element_blank())
ggsave(file.path(plots_dir, "coverage", "coverage_basic.png"), coverage_plt_basic)

# "All" = POOLED coverage over every held-out observation (sum(covered)/sum(n)),
# which weights each metric by its observation count -- NOT the unweighted mean of
# per-metric rates. Equal to the mean-of-rates only when every metric has the same
# n_obs. EXIT_AGE indicators are pooled in alongside the per-(player,age) metric hits.
val_all_pool <- c(validation_coverage_df$validation_coverage,
                  exit_age_coverage_df$exit_age_coverage[exit_age_coverage_df$exit_split == "holdout"])
is_all_pool  <- c(in_sample_coverage_df$in_sample_coverage,
                  exit_age_coverage_df$exit_age_coverage[exit_age_coverage_df$exit_split == "train"])
coverage_all_row <- tibble(
  metric              = "All",
  validation_coverage = mean(val_all_pool, na.rm = TRUE),
  in_sample_coverage  = mean(is_all_pool,  na.rm = TRUE)
)
if (has_conditional && exists("conditional_coverage_df")) {
  # conditional coverage is performance-metric only (no EXIT_AGE); pool its per-obs hits
  coverage_all_row$conditional_coverage <- mean(conditional_coverage_df$conditional_coverage, na.rm = TRUE)
}
coverage_base_tbl_tex <- bind_rows(coverage_base_tbl, coverage_all_row)

latex_tbl <- coverage_base_tbl_tex |>
  mutate(in_sample_coverage  = paste0(round(in_sample_coverage  * 100, 1), "%"),
         validation_coverage = paste0(round(validation_coverage * 100, 1), "%"))
latex_col_labels <- list(metric              = "Metric",
                         in_sample_coverage  = "In-Sample Coverage",
                         validation_coverage = "Validation Coverage")
if (has_conditional) {
  latex_tbl <- latex_tbl |>
    mutate(conditional_coverage = paste0(round(as.numeric(conditional_coverage) * 100, 1), "%"))
  latex_col_labels[["conditional_coverage"]] <- "Conditional Coverage"
}
latex_code <- latex_tbl |>
  gt() |>
  cols_label(!!!latex_col_labels) |>
  as_latex()
latex_code <- gsub(
  "\\\\begin\\{tabular\\*\\}\\{\\\\linewidth\\}\\{@\\{\\\\extracolsep\\{\\\\fill\\}\\}([^}]+)\\}",
  "\\\\resizebox{\\\\linewidth}{!}{\\\\begin{tabular}{\\1}",
  latex_code
)
latex_code <- gsub("\\\\end\\{tabular\\*\\}", "\\\\end{tabular}}", latex_code)
latex_code <- gsub(
  "\\\\begin\\{table\\}\\[t\\]",
  "\\\\begin{table}[htbp]\n\\\\centering\n\\\\caption{95\\\\% HDI coverage by metric: validation (hold-out), in-sample, and conditional.}\n\\\\label{tab:coverage_basic}",
  latex_code
)
latex_code <- gsub("\\\\fontsize\\{[0-9.]+\\}\\{[0-9.]+\\}\\\\selectfont[^\n]*\n?", "", latex_code)
writeLines(latex_code, file.path(plots_dir, "coverage", "coverage_basic.tex"))

# Posterior Bias Table (mean [95% HDI]) — rows: metrics, cols: in-sample / validation
# Compute per-sample mean bias for continuous metrics
validation_bias_samples <- joined_data |>
  filter(split == "holdout", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |>
  filter(!is.na(obs_value)) |>
  group_by(chain, sample, metric) |>
  summarize(mean_bias = mean(value - obs_value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = toupper(metric),
         metric = case_when(metric == "GAMES" ~ "GP%",
                            metric == "FG2M" ~ "FG2%",
                            metric == "FG3M" ~ "FG3%",
                            metric == "FTM" ~ "FT%",
                            metric == "PCT_MINUTES" ~ "MPG",
                            .default = metric))

in_sample_bias_samples <- joined_data |>
  filter(split == "train", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |>
  filter(!is.na(obs_value)) |>
  group_by(chain, sample, metric) |>
  summarize(mean_bias = mean(value - obs_value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = toupper(metric),
         metric = case_when(metric == "GAMES" ~ "GP%",
                            metric == "FG2M" ~ "FG2%",
                            metric == "FG3M" ~ "FG3%",
                            metric == "FTM" ~ "FT%",
                            metric == "PCT_MINUTES" ~ "MPG",
                            .default = metric))

validation_bias_summary <- validation_bias_samples |>
  group_by(metric) |>
  summarize(
    val_mean  = mean(mean_bias, na.rm = TRUE),
    val_lower = HDInterval::hdi(mean_bias, credMass = 0.95)["lower"],
    val_upper = HDInterval::hdi(mean_bias, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(validation_bias = paste0(round(val_mean, 3), " [", round(val_lower, 3), ", ", round(val_upper, 3), "]")) |>
  select(metric, validation_bias)

in_sample_bias_summary <- in_sample_bias_samples |>
  group_by(metric) |>
  summarize(
    is_mean  = mean(mean_bias, na.rm = TRUE),
    is_lower = HDInterval::hdi(mean_bias, credMass = 0.95)["lower"],
    is_upper = HDInterval::hdi(mean_bias, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(in_sample_bias = paste0(round(is_mean, 3), " [", round(is_lower, 3), ", ", round(is_upper, 3), "]")) |>
  select(metric, in_sample_bias)

# Exit age bias (already has posterior_mean and obs_value per player)
exit_age_bias_validation <- exit_age_coverage_df |>
  filter(exit_split == "holdout") |>
  summarize(
    metric        = "EXIT_AGE",
    val_mean      = mean(posterior_mean - obs_value, na.rm = TRUE),
    val_lower     = HDInterval::hdi(posterior_mean - obs_value, credMass = 0.95)["lower"],
    val_upper     = HDInterval::hdi(posterior_mean - obs_value, credMass = 0.95)["upper"]
  ) |>
  mutate(validation_bias = paste0(round(val_mean, 3), " [", round(val_lower, 3), ", ", round(val_upper, 3), "]")) |>
  select(metric, validation_bias)

exit_age_bias_in_sample <- exit_age_coverage_df |>
  filter(exit_split == "train") |>
  summarize(
    metric    = "EXIT_AGE",
    is_mean   = mean(posterior_mean - obs_value, na.rm = TRUE),
    is_lower  = HDInterval::hdi(posterior_mean - obs_value, credMass = 0.95)["lower"],
    is_upper  = HDInterval::hdi(posterior_mean - obs_value, credMass = 0.95)["upper"]
  ) |>
  mutate(in_sample_bias = paste0(round(is_mean, 3), " [", round(is_lower, 3), ", ", round(is_upper, 3), "]")) |>
  select(metric, in_sample_bias)

bias_latex_code <- in_sample_bias_summary |>
  bind_rows(exit_age_bias_in_sample) |>
  inner_join(
    bind_rows(validation_bias_summary, exit_age_bias_validation),
    by = "metric"
  ) |>
  gt() |>
  cols_label(
    metric         = "Metric",
    in_sample_bias = "In-Sample Bias (95\\% HDI)",
    validation_bias = "Validation Bias (95\\% HDI)"
  ) |>
  tab_header(title = "Posterior Bias Summary: Mean [95\\% HDI]") |>
  as_latex()

writeLines(bias_latex_code, file.path(plots_dir, "coverage", "coverage_bias.tex"))

# Export bias intervals as machine-readable CSV for combine_holdout_tables.py
bias_intervals_val <- validation_bias_samples |>
  group_by(metric) |>
  summarise(
    mean  = mean(mean_bias, na.rm = TRUE),
    lower = HDInterval::hdi(mean_bias, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(mean_bias, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(split = "holdout")

bias_intervals_is <- in_sample_bias_samples |>
  group_by(metric) |>
  summarise(
    mean  = mean(mean_bias, na.rm = TRUE),
    lower = HDInterval::hdi(mean_bias, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(mean_bias, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(split = "in_sample")

write.csv(bind_rows(bias_intervals_val, bias_intervals_is),
          file.path(plots_dir, "coverage", "coverage_bias_intervals.csv"),
          row.names = FALSE)

# ── Exit age histogram: observed retirements vs right-censored ────────────────
exit_age_meta <- posterior_survival_data |>
  distinct(player, observed_exit_age, exit_censored) |>
  mutate(status = if_else(exit_censored == 1L, "Right-censored", "Observed retirement"))

exit_age_hist_plt <- ggplot(exit_age_meta, aes(x = observed_exit_age, fill = status)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.55, color = "white", linewidth = 0.2) +
  scale_fill_manual(
    values = c("Observed retirement" = "#e74c3c", "Right-censored" = "#2980b9"),
    name   = NULL
  ) +
  scale_x_continuous(breaks = seq(18, 46, by = 2)) +
  labs(
    title    = "Distribution of Exit Ages",
    subtitle = "Observed retirements vs right-censored players",
    x        = "Exit age",
    y        = "Count"
  ) +
  theme_bw(base_size = 14) +
  theme(legend.position = "top", panel.grid.minor = element_blank())

ggsave(file.path(plots_dir, "coverage", "exit_age_histogram.png"), exit_age_hist_plt,
       width = 8, height = 5, dpi = 150)

# ── Data availability by age (18–38) ─────────────────────────────────────────
real_data        <- data |> filter(!(id %in% c("99999999", 99999999)))
n_total_players  <- n_distinct(real_data$id)

data_availability <- tibble(age = 18L:38L) |>
  mutate(
    n_with_data   = map_int(age, \(a) sum(real_data$age == a, na.rm = TRUE)),
    pct_available = 100 * n_with_data / n_total_players
  )

data_avail_plt <- ggplot(data_availability, aes(x = factor(age), y = pct_available)) +
  geom_col(fill = "#2980b9", color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.0f%%", pct_available)), vjust = -0.4, size = 3) +
  scale_y_continuous(limits = c(0, 110), expand = c(0, 0)) +
  labs(
    title    = "Data Availability by Age",
    subtitle = sprintf("Percentage of %d players with observed data at each age (18–38)", n_total_players),
    x        = "Age",
    y        = "% of players with data"
  ) +
  theme_bw(base_size = 14) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank())

ggsave(file.path(plots_dir, "coverage", "data_availability_by_age.png"), data_avail_plt,
       width = 10, height = 5, dpi = 150)

# ── Empirical peak-max PCA colored by position ───────────────────────────────
empirical_metric_cols <- c("GP_pct", "MPG", "BLK", "AST", "TOV", "OREB", "DREB",
                           "STL", "FT_pct", "FG2_pct", "FG3_pct",
                           "FG3A", "FG2A", "FTA", "USG", "OBPM", "DBPM")

empirical_max_df <- real_data |>
  filter(!is.na(minutes), minutes >= min_minutes_threshold) |>
  mutate(
    GP_pct  = games / pmax(games, total_games, na.rm = TRUE),
    MPG     = (minutes / games) / 48,
    BLK     = 36 * (blk  / minutes),
    AST     = 36 * (ast  / minutes),
    TOV     = 36 * (tov  / minutes),
    OREB    = 36 * (oreb / minutes),
    DREB    = 36 * (dreb / minutes),
    STL     = 36 * (stl  / minutes),
    FT_pct  = ftm  / fta,
    FG2_pct = fg2m / fg2a,
    FG3_pct = fg3m / fg3a,
    FG3A    = 36 * (fg3a / minutes),
    FG2A    = 36 * (fg2a / minutes),
    FTA     = 36 * (fta  / minutes),
    USG     = usg,
    OBPM    = obpm,
    DBPM    = dbpm
  ) |>
  group_by(id) |>
  summarise(
    across(all_of(empirical_metric_cols), \(x) max(x, na.rm = TRUE)),
    position_group = first(na.omit(position_group)),
    name           = first(na.omit(name)),
    total_minutes  = sum(minutes, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(if_all(all_of(empirical_metric_cols), is.finite))

emp_pca      <- prcomp(empirical_max_df |> select(all_of(empirical_metric_cols)), scale. = TRUE, center = TRUE)
pct_var      <- round(100 * emp_pca$sdev^2 / sum(emp_pca$sdev^2), 1)
emp_pca_df   <- tibble(PC1 = emp_pca$x[, 1], PC2 = emp_pca$x[, 2]) |>
  bind_cols(empirical_max_df |> select(id, name, position_group, total_minutes))

emp_pca_plt <- ggplot(emp_pca_df, aes(x = PC1, y = PC2, color = position_group)) +
  geom_point(aes(alpha = total_minutes), size = 1.8) +
  scale_alpha_continuous(range = c(0.15, 0.9), name = "Career minutes") +
  scale_colour_brewer(palette = "Set1", name = "Position") +
  labs(
    title    = "PCA of Empirical Career-Max Metrics",
    subtitle = "Each point is a player; axes are top-2 PCs of per-36 / percentage stats at career peak",
    x        = glue("PC1 ({pct_var[1]}% var)"),
    y        = glue("PC2 ({pct_var[2]}% var)")
  ) +
  theme_bw(base_size = 14) +
  theme(legend.position = "right", panel.grid.minor = element_blank())

ggsave(file.path(plots_dir, "coverage", "empirical_max_pca_position.png"), emp_pca_plt,
       width = 9, height = 6, dpi = 150)

# Per-sample MSE (proportional to Gaussian NLL; posterior interval for predictive accuracy)
rename_metric_vec <- function(m) {
  m <- toupper(m)
  dplyr::case_when(m == "GAMES" ~ "GP%", m == "FG2M" ~ "FG2%", m == "FG3M" ~ "FG3%",
                   m == "FTM" ~ "FT%", m == "PCT_MINUTES" ~ "MPG", .default = m)
}

validation_mse_samples <- joined_data |>
  filter(split == "holdout", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |>
  filter(!is.na(obs_value)) |>
  group_by(chain, sample, metric) |>
  summarise(mean_mse = mean((value - obs_value)^2, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = rename_metric_vec(metric))

in_sample_mse_samples <- joined_data |>
  filter(split == "train", metric != "retirement", is.na(minutes) | minutes >= min_minutes_threshold) |>
  filter(!is.na(obs_value)) |>
  group_by(chain, sample, metric) |>
  summarise(mean_mse = mean((value - obs_value)^2, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = rename_metric_vec(metric))

mse_intervals_val <- validation_mse_samples |>
  group_by(metric) |>
  summarise(
    mean  = mean(mean_mse, na.rm = TRUE),
    lower = HDInterval::hdi(mean_mse, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(mean_mse, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(split = "holdout")

mse_intervals_is <- in_sample_mse_samples |>
  group_by(metric) |>
  summarise(
    mean  = mean(mean_mse, na.rm = TRUE),
    lower = HDInterval::hdi(mean_mse, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(mean_mse, credMass = 0.95)["upper"],
    .groups = "drop"
  ) |>
  mutate(split = "in_sample")

write.csv(bind_rows(mse_intervals_val, mse_intervals_is),
          file.path(plots_dir, "coverage", "coverage_mse_intervals.csv"),
          row.names = FALSE)

# Log-loss posterior interval from posterior_metric_log_loss.parquet (exact NLL per family)
if (!is.null(posterior_metric_ll)) {
  ll_renamed <- posterior_metric_ll |>
    mutate(metric = rename_metric_vec(metric))

  ll_intervals_val <- ll_renamed |>
    filter(split == "holdout") |>
    group_by(metric) |>
    summarise(
      mean  = mean(avg_log_loss, na.rm = TRUE),
      lower = HDInterval::hdi(avg_log_loss, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(avg_log_loss, credMass = 0.95)["upper"],
      .groups = "drop"
    ) |>
    mutate(split = "holdout")

  ll_intervals_is <- ll_renamed |>
    filter(split == "in_sample") |>
    group_by(metric) |>
    summarise(
      mean  = mean(avg_log_loss, na.rm = TRUE),
      lower = HDInterval::hdi(avg_log_loss, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(avg_log_loss, credMass = 0.95)["upper"],
      .groups = "drop"
    ) |>
    mutate(split = "in_sample")

  ll_all_val <- ll_renamed |>
    filter(split == "holdout") |>
    group_by(chain, draw) |>
    summarise(avg_log_loss = mean(avg_log_loss, na.rm = TRUE), .groups = "drop") |>
    summarise(
      metric = "All",
      mean  = mean(avg_log_loss, na.rm = TRUE),
      lower = HDInterval::hdi(avg_log_loss, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(avg_log_loss, credMass = 0.95)["upper"]
    ) |>
    mutate(split = "holdout")

  ll_all_is <- ll_renamed |>
    filter(split == "in_sample") |>
    group_by(chain, draw) |>
    summarise(avg_log_loss = mean(avg_log_loss, na.rm = TRUE), .groups = "drop") |>
    summarise(
      metric = "All",
      mean  = mean(avg_log_loss, na.rm = TRUE),
      lower = HDInterval::hdi(avg_log_loss, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(avg_log_loss, credMass = 0.95)["upper"]
    ) |>
    mutate(split = "in_sample")

  write.csv(bind_rows(ll_intervals_val, ll_intervals_is, ll_all_val, ll_all_is),
            file.path(plots_dir, "coverage", "coverage_log_loss_intervals.csv"),
            row.names = FALSE)
}

coverage_plt_yearly <- validation_coverage_df |> group_by(metric, year) |> summarize(Coverage = mean(validation_coverage, na.rm = TRUE), .groups = "drop") |>
                        bind_rows(joined_data %>% filter(split == "holdout" & metric == "retirement") %>% mutate(metric = toupper(metric)) %>% filter(!is.na(obs_value)) %>% group_by(metric, year) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop")) |>
                        bind_rows(exit_age_coverage_df %>% filter(exit_split == "holdout") %>% mutate(year = observed_exit_year, Coverage = as.numeric(exit_age_coverage)) %>% group_by(metric, year) %>% summarize(Coverage = mean(Coverage, na.rm = TRUE), .groups = "drop")) |>
                        ggplot(aes(x = year, y = Coverage)) +
                        geom_col() + facet_wrap(~metric, scales = "free_y") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
                        theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Time Horizon") +xlab("Year")
ggsave(file.path(plots_dir, "coverage", "coverage_validation_yearly.png"), coverage_plt_yearly)
coverage_plt_minutes <- validation_coverage_df |> inner_join(train_years_df, by = c("player" = "id")) |> group_by(metric, years_played) |> summarize(Coverage = mean(validation_coverage, na.rm = TRUE), .groups = "drop") |> ungroup() |> bind_rows(
  joined_data %>%
  filter(split == "holdout" & metric == "retirement") %>%
  mutate(metric = toupper(metric)) %>%
  filter(!is.na(obs_value)) %>%
  inner_join(train_years_df, by = c("player" = "id")) %>%
  group_by(metric, years_played) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop") %>% ungroup()) |>
                        bind_rows(exit_age_coverage_df %>% filter(exit_split == "holdout") %>% mutate(Coverage = as.numeric(exit_age_coverage)) %>% group_by(metric, years_played) %>% summarize(Coverage = mean(Coverage, na.rm = TRUE), .groups = "drop")) %>%
                        ggplot(aes(x = years_played, y = Coverage)) +
                        geom_point() + facet_wrap(~metric, scales = "free_y") +
                        theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Years of Training Data Available") + xlab("Years Played")
ggsave(file.path(plots_dir, "coverage", "coverage_validation_minutes.png"), coverage_plt_minutes)

coverage_plt_minutes <- in_sample_coverage_df |> inner_join(train_years_df, by = c("player" = "id")) |> group_by(metric, years_played) |> summarize(Coverage = mean(in_sample_coverage, na.rm = TRUE), .groups = "drop") |> ungroup() |> bind_rows(
  joined_data %>%
  filter(split == "train" & metric == "retirement") %>%
  mutate(metric = toupper(metric)) %>%
  filter(!is.na(obs_value)) %>%
  inner_join(train_years_df, by = c("player" = "id")) %>%
  group_by(metric, years_played) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop") %>% ungroup()) |>
                        bind_rows(exit_age_coverage_df %>% filter(exit_split == "train") %>% mutate(Coverage = as.numeric(exit_age_coverage)) %>% group_by(metric, years_played) %>% summarize(Coverage = mean(Coverage, na.rm = TRUE), .groups = "drop")) %>%
                        ggplot(aes(x = years_played, y = Coverage)) +
                        geom_point() + facet_wrap(~metric, scales = "free_y") +
                        theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric In-Sample Coverage by Years of Training Data Available") + xlab("Years Played")
ggsave(file.path(plots_dir, "coverage", "coverage_in_sample_minutes.png"), coverage_plt_minutes)
