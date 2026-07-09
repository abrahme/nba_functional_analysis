library(tidyverse)
library(readr)
library(glue)
library(ggrepel)
library(patchwork)
library(arrow)

args      <- commandArgs(trailingOnly = TRUE)
model_dir <- if (length(args) >= 1) args[1] else
  stop("Usage: Rscript data_analysis/team_window.r <model_dir> [horizon]")
horizon   <- if (length(args) >= 2) as.integer(args[2]) else 10L

min_minutes_threshold <- 0L     # include any player with recorded minutes
window_frac           <- 0.85   # window = years within this fraction of peak quality
mins_per_game         <- 240.0  # 5 players × 48 min (team minutes per game)

plots_dir <- file.path(model_dir, "plots", "team_window")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. Rosters: most recent year in the data ──────────────────────────────────
data         <- read.csv("data/injury_player_cleaned.csv")
roster_year  <- max(data$year, na.rm = TRUE)

message(glue("Model dir   : {model_dir}"))
message(glue("Roster year : {roster_year}  |  Horizon: {horizon} seasons"))

rosters <- data |>
  filter(year == roster_year, !is.na(team), !is.na(minutes),
         minutes >= min_minutes_threshold) |>
  select(id, name, team, age, minutes) |>
  distinct(id, team, .keep_all = TRUE)

active_teams <- sort(unique(rosters$team))
player_ids   <- unique(rosters$id)
message(glue("{length(active_teams)} teams, {length(player_ids)} players on rosters"))

# ── 2. Posterior curves: OBPM, GP%, MPG ──────────────────────────────────────
# posterior_ar.parquet is present for all model variants; the choice of
# model_dir determines which model's trajectories are used.
message("Loading posterior curves (obpm, games, pct_minutes)...")

posterior_curves <- open_dataset(file.path(model_dir, "posterior_ar.parquet")) |>
  filter(metric %in% c("obpm", "games", "pct_minutes")) |>
  select(player, metric, age, value) |>
  collect() |>
  filter(player %in% player_ids)

curve_means <- posterior_curves |>
  group_by(player, metric, age) |>
  summarise(
    mean_val = mean(value,           na.rm = TRUE),
    q10_val  = quantile(value, 0.10, na.rm = TRUE),
    q90_val  = quantile(value, 0.90, na.rm = TRUE),
    .groups  = "drop"
  ) |>
  pivot_wider(names_from  = metric,
              values_from = c(mean_val, q10_val, q90_val))

# ── 3. Survival: P(exit_age > future_age) per player ─────────────────────────
message("Loading exit-age samples...")

exit_samples <- open_dataset(file.path(model_dir, "posterior_exit_age_sample.parquet")) |>
  filter(measure == "exit_age_sample", scenario == "observed") |>
  select(player, value) |>
  collect() |>
  filter(player %in% player_ids)

age_grid <- seq(18L, 45L)

survival_prob <- exit_samples |>
  group_by(player) |>
  summarise(
    age      = list(age_grid),
    p_active = list(map_dbl(age_grid, ~ mean(value > .x))),
    .groups  = "drop"
  ) |>
  unnest(c(age, p_active))

# ── 4. Team quality at each future season ────────────────────────────────────
# Q(team, Y) = (1 / mins_per_game) *
#              sum_p [ P(active_p at Y) * GP_frac_p(Y) * MPG_p(Y) * OBPM_p(Y) ]
message("Computing team quality curves...")
future_years <- seq(roster_year, roster_year + horizon)

player_projections <- rosters |>
  crossing(calendar_year = future_years) |>
  mutate(future_age = age + (calendar_year - roster_year)) |>
  filter(future_age >= 18L, future_age <= 45L) |>
  left_join(curve_means,   by = c("id" = "player", "future_age" = "age")) |>
  left_join(survival_prob, by = c("id" = "player", "future_age" = "age")) |>
  mutate(
    # pct_minutes is fraction of individual 48 min; minutes/3936 = (season min/82)/48
    # Unmodeled players get replacement-level defaults: gp_frac=0.75, mpg from
    # their observed 2026 season minutes, obpm=-2.0
    gp_frac  = coalesce(mean_val_games,       0.75),
    mpg      = coalesce(mean_val_pct_minutes, minutes / 3936) * 48,
    obpm     = coalesce(mean_val_obpm,        -2.0),
    gp_q10   = coalesce(q10_val_games,        0.75),
    mpg_q10  = coalesce(q10_val_pct_minutes,  minutes / 3936) * 48,
    obpm_q10 = coalesce(q10_val_obpm,         -2.0),
    gp_q90   = coalesce(q90_val_games,        0.75),
    mpg_q90  = coalesce(q90_val_pct_minutes,  minutes / 3936) * 48,
    obpm_q90 = coalesce(q90_val_obpm,         -2.0),
    p_active = coalesce(p_active,             0),
    contrib     = p_active * gp_frac * mpg * obpm     / mins_per_game,
    contrib_q10 = p_active * gp_q10  * mpg_q10 * obpm_q10 / mins_per_game,
    contrib_q90 = p_active * gp_q90  * mpg_q90 * obpm_q90 / mins_per_game
  )

team_quality <- player_projections |>
  group_by(team, calendar_year) |>
  summarise(
    team_obpm     = sum(contrib,     na.rm = TRUE),
    team_obpm_q10 = sum(contrib_q10, na.rm = TRUE),
    team_obpm_q90 = sum(contrib_q90, na.rm = TRUE),
    n_active_exp  = sum(p_active * gp_frac, na.rm = TRUE),
    .groups       = "drop"
  )

# ── 5. Window identification ──────────────────────────────────────────────────
window_summary <- team_quality |>
  group_by(team) |>
  mutate(peak_obpm = max(team_obpm)) |>
  filter(peak_obpm < 0 | team_obpm >= window_frac * peak_obpm) |>
  summarise(
    peak_year    = calendar_year[which.max(team_obpm)],
    window_open  = min(calendar_year),
    window_close = max(calendar_year),
    peak_obpm    = max(team_obpm),
    .groups      = "drop"
  ) |>
  arrange(peak_year, desc(peak_obpm))

print(window_summary, n = 30)

# ── 5b. Peak-year distribution from posterior draws ──────────────────────────
# Restrict to a 5-season window: beyond that, team roster composition changes
# too much to be meaningful.
message("Loading posterior draws for 5-year peak-year CI computation...")

ci_years <- seq(roster_year, roster_year + 5L)

ci_relevant_ages <- rosters |>
  crossing(calendar_year = ci_years) |>
  mutate(future_age = age + (calendar_year - roster_year)) |>
  filter(future_age >= 18L, future_age <= 45L) |>
  pull(future_age) |>
  unique()

posterior_draws <- open_dataset(file.path(model_dir, "posterior_ar.parquet")) |>
  filter(metric %in% c("obpm", "games", "pct_minutes"),
         player %in% player_ids,
         age    %in% ci_relevant_ages) |>
  select(chain, sample, player, metric, age, value) |>
  collect()

draw_wide <- posterior_draws |>
  mutate(draw_id = paste(chain, sample, sep = "_")) |>
  select(draw_id, player, metric, age, value) |>
  pivot_wider(names_from = metric, values_from = value) |>
  rename(gp_frac_d = games, mpg_d = pct_minutes, obpm_d = obpm)

draw_proj_ci <- rosters |>
  crossing(calendar_year = ci_years) |>
  mutate(future_age = age + (calendar_year - roster_year)) |>
  filter(future_age >= 18L, future_age <= 45L) |>
  left_join(draw_wide,     by = c("id" = "player", "future_age" = "age"),
            relationship  = "many-to-many") |>
  left_join(survival_prob, by = c("id" = "player", "future_age" = "age")) |>
  mutate(
    contrib_d = coalesce(p_active, 0) *
                coalesce(gp_frac_d, 0.75) *
                coalesce(mpg_d, minutes / 3936) * 48 *
                coalesce(obpm_d, -2.0) / mins_per_game
  )

peak_year_ci <- draw_proj_ci |>
  group_by(draw_id, team, calendar_year) |>
  summarise(team_obpm_d = sum(contrib_d, na.rm = TRUE), .groups = "drop") |>
  group_by(draw_id, team) |>
  slice_max(team_obpm_d, n = 1, with_ties = FALSE) |>
  group_by(team) |>
  summarise(
    window_q025 = quantile(calendar_year, 0.025),
    window_q975 = quantile(calendar_year, 0.975),
    .groups     = "drop"
  )

# ── 6. Plots ──────────────────────────────────────────────────────────────────
yr_labels <- function(x) paste0("'", formatC(x %% 100, width = 2, flag = "0"))

# Plot A: faceted quality curves (all teams)
facet_plt <- team_quality |>
  left_join(window_summary |> select(team, peak_year), by = "team") |>
  ggplot(aes(x = calendar_year)) +
  geom_ribbon(aes(ymin = team_obpm_q10, ymax = team_obpm_q90),
              alpha = 0.20, fill = "steelblue") +
  geom_line(aes(y = team_obpm), colour = "steelblue", linewidth = 0.9) +
  geom_vline(aes(xintercept = peak_year),
             linetype = "dashed", colour = "firebrick", linewidth = 0.6) +
  facet_wrap(~ team, ncol = 6) +
  scale_x_continuous(
    breaks = seq(roster_year, roster_year + horizon, by = 2),
    labels = yr_labels
  ) +
  labs(
    title    = glue("Projected team quality — all {length(active_teams)} teams ({roster_year} rosters)"),
    subtitle = "Q(Y) = Σ P(active) × GP% × MPG × OBPM / 240  |  ribbon = 10–90th pct  |  dashed = peak",
    x        = "Season",
    y        = "Expected team OBPM"
  ) +
  theme_bw(base_size = 9) +
  theme(strip.text       = element_text(face = "bold", size = 8),
        axis.text.x      = element_text(size = 6),
        panel.grid.minor = element_blank())

ggsave(file.path(plots_dir, "team_quality_all.png"),
       facet_plt, width = 18, height = 14, dpi = 150)
message("Saved: team_quality_all.png")

# Plot B: faceted quality curves with 95% CI of peak year as a gold band
# One rectangle per team (distinct to avoid stacking with alpha)
window_rects <- team_quality |>
  left_join(peak_year_ci,                                by = "team") |>
  left_join(window_summary |> select(team, peak_year),  by = "team") |>
  distinct(team, window_q025, window_q975, peak_year) |>
  mutate(team = fct_reorder(team, coalesce(peak_year, roster_year)))

timeline_plt <- team_quality |>
  left_join(window_summary |> select(team, peak_year), by = "team") |>
  mutate(team = fct_reorder(team, coalesce(peak_year, roster_year))) |>
  ggplot(aes(x = calendar_year)) +
  geom_rect(
    data         = window_rects,
    aes(xmin = window_q025 - 0.5, xmax = window_q975 + 0.5,
        ymin = -Inf, ymax = Inf),
    fill         = "gold", alpha = 0.35, inherit.aes = FALSE
  ) +
  geom_vline(
    data         = window_rects,
    aes(xintercept = peak_year),
    linetype     = "dashed", colour = "firebrick", linewidth = 0.5,
    inherit.aes  = FALSE
  ) +
  geom_ribbon(aes(ymin = team_obpm_q10, ymax = team_obpm_q90),
              alpha = 0.20, fill = "steelblue") +
  geom_line(aes(y = team_obpm), colour = "steelblue", linewidth = 0.9) +
  facet_wrap(~ team, ncol = 6) +
  scale_x_continuous(
    breaks = seq(roster_year, roster_year + horizon, by = 2),
    labels = yr_labels
  ) +
  labs(
    title    = glue("NBA competitive windows — {length(active_teams)} teams ({roster_year} rosters)"),
    subtitle = "Blue ribbon = 10–90th pct of quality  |  gold band = 95% CI of peak season  |  dashed = MAP peak",
    x        = "Season",
    y        = "Expected team OBPM"
  ) +
  theme_bw(base_size = 9) +
  theme(strip.text       = element_text(face = "bold", size = 8),
        axis.text.x      = element_text(size = 6),
        panel.grid.minor = element_blank())

ggsave(file.path(plots_dir, "team_windows_timeline.png"),
       timeline_plt, width = 18, height = 14, dpi = 150)
message("Saved: team_windows_timeline.png")

# ── 7. LaTeX summary table ────────────────────────────────────────────────────
tier_breaks <- quantile(window_summary$peak_obpm, c(0.25, 0.50, 0.75))

window_tex <- knitr::kable(
  window_summary |>
    arrange(peak_year, desc(peak_obpm)) |>
    mutate(
      tier = case_when(
        peak_obpm >= tier_breaks[3] ~ "Contender",
        peak_obpm >= tier_breaks[2] ~ "Playoff",
        peak_obpm >= tier_breaks[1] ~ "Fringe",
        TRUE                        ~ "Rebuilding"
      ),
      Window    = glue("{window_open}--{window_close}"),
      peak_obpm = sprintf("%.2f", peak_obpm)
    ) |>
    select(Team = team, Peak = peak_year,
           Window, OBPM = peak_obpm, Tier = tier),
  format   = "latex",
  booktabs = TRUE,
  escape   = FALSE,
  caption  = glue(
    "Projected competitive windows for all {length(active_teams)} NBA teams",
    " based on {roster_year} rosters. ",
    "$Q(Y)=\\\\frac{{1}}{{240}}\\\\sum_p P(\\\\text{{active}}_p)\\\\cdot",
    "\\\\mathrm{{GP}}_p(Y)\\\\cdot\\\\mathrm{{MPG}}_p(Y)\\\\cdot",
    "\\\\mathrm{{OBPM}}_p(Y)$. ",
    "Window = years within {window_frac * 100}\\\\%% of peak $Q$;",
    " tier thresholds are quartiles of peak OBPM across all teams."
  ),
  label = "tab:team_windows"
)

writeLines(as.character(window_tex),
           file.path(plots_dir, "team_windows.tex"))
message("Saved: team_windows.tex")
message(glue("All outputs in: {plots_dir}"))
