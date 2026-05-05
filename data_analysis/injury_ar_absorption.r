library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(HDInterval)

model_dir <- "model_output/nba_convex_max_tvlinearlvm_injury/holdout_last_k/mcmc"
plots_dir <- file.path(model_dir, "plots")
dir.create(file.path(plots_dir, "injury"), recursive = TRUE, showWarnings = FALSE)

rename_metrics <- function(df) {
  df |> mutate(metric = toupper(metric),
               metric = case_when(metric == "GAMES"       ~ "GP%",
                                  metric == "FG2M"        ~ "FG2%",
                                  metric == "FG3M"        ~ "FG3%",
                                  metric == "FTM"         ~ "FT%",
                                  metric == "PCT_MINUTES" ~ "MPG",
                                  .default = metric))
}

ar_comp    <- read_parquet(file.path(model_dir, "posterior_latent_ar.parquet"))
inj_effect <- read_parquet(file.path(model_dir, "posterior_injury_effect.parquet"))

inj_data <- read.csv("data/injury_player_cleaned.csv") |>
  filter(!is.na(first_major_injury)) |>
  select(id, age, injury_period, first_major_injury)

ar_summary <- ar_comp |>
  group_by(player, metric, age) |>
  summarise(
    ar_mean  = mean(value, na.rm = TRUE),
    ar_lower = hdi(value, credMass = 0.95)["lower"],
    ar_upper = hdi(value, credMass = 0.95)["upper"],
    .groups  = "drop"
  )

inj_summary <- inj_effect |>
  group_by(player, metric, age) |>
  summarise(
    injury_mean  = mean(value, na.rm = TRUE),
    injury_lower = hdi(value, credMass = 0.95)["lower"],
    injury_upper = hdi(value, credMass = 0.95)["upper"],
    .groups      = "drop"
  )

combined <- ar_summary |>
  inner_join(inj_summary, by = c("player", "metric", "age")) |>
  inner_join(inj_data, by = c("player" = "id", "age")) |>
  rename_metrics()

# ── Plot 1: mean AR vs injury effect by injury_period, faceted by metric ──────
focus_metrics <- c("GP%", "MPG", "OBPM", "AST", "DREB", "FG2%")

plot_df <- combined |>
  filter(metric %in% focus_metrics) |>
  pivot_longer(
    cols      = c(ar_mean, injury_mean),
    names_to  = "component",
    values_to = "value"
  ) |>
  mutate(component = recode(component,
                             ar_mean     = "AR",
                             injury_mean = "Injury effect"))

absorption_plt <- plot_df |>
  group_by(metric, age, injury_period, component) |>
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") |>
  ggplot(aes(x = age, y = value, colour = component)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.8) +
  facet_grid(metric ~ injury_period, scales = "free_y") +
  scale_colour_manual(values = c("AR" = "#E41A1C", "Injury effect" = "#377EB8")) +
  labs(
    title    = "AR component vs injury effect by career period",
    subtitle = "If AR is negative post-injury while injury effect ≈ 0, AR is absorbing the injury signal",
    x        = "Age", y = "Effect (model units)", colour = NULL
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(plots_dir, "injury", "ar_vs_injury_effect_by_period.png"),
       absorption_plt, width = 14, height = 12)

# ── Plot 2: per-player AR trajectory around injury onset ─────────────────────
# Centre age on injury onset (age 0 = injury year)
injury_onset <- inj_data |>
  group_by(id) |>
  summarise(injury_age = min(age[injury_period == "post-injury"], na.rm = TRUE),
            first_major_injury = first(first_major_injury),
            .groups = "drop")

centered <- ar_summary |>
  inner_join(injury_onset, by = c("player" = "id")) |>
  rename_metrics() |>
  mutate(seasons_from_injury = age - injury_age) |>
  filter(seasons_from_injury >= -5, seasons_from_injury <= 5)

onset_plt <- centered |>
  group_by(metric, seasons_from_injury) |>
  summarise(
    mean  = mean(ar_mean, na.rm = TRUE),
    lower = mean(ar_lower, na.rm = TRUE),
    upper = mean(ar_upper, na.rm = TRUE),
    .groups = "drop"
  ) |>
  ggplot(aes(x = seasons_from_injury, y = mean)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(
    title    = "Mean AR component centred on injury onset (season 0)",
    subtitle = "A dip at/after season 0 indicates AR absorbing the post-injury performance drop",
    x        = "Seasons from injury", y = "AR component (model units)"
  ) +
  theme_bw()

ggsave(file.path(plots_dir, "injury", "ar_centred_on_injury_onset.png"),
       onset_plt, width = 18, height = 14)

message("Saved to: ", file.path(plots_dir, "injury"))
