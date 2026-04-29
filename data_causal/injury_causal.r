library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
library(purrr)
library(glue)
library(ggrepel)
library(ggnewscale)
library(ggdist)
library(patchwork)
library(uwot)
library(arrow)


args      <- commandArgs(trailingOnly = TRUE)
model_dir <- if (length(args) >= 1) args[1] else stop("Usage: Rscript injury_causal.r <model_dir>")
plots_dir <- file.path(model_dir, "plots")
dir.create(file.path(plots_dir, "causal"), recursive = TRUE, showWarnings = FALSE)

data <- read.csv("data/injury_player_cleaned.csv") |>    mutate( pct_games = games / pmax(games, total_games, na.rm = TRUE),
            mpg = minutes / games,
            blk_rate = 36 * (blk / minutes),
            ast_rate = 36 * (ast / minutes),
            tov_rate = 36 * (tov / minutes),
            oreb_rate = 36 * (oreb / minutes),
            dreb_rate = 36 * (dreb / minutes),
            stl_rate = 36 * (stl / minutes),
            fg3a_rate = 36 * (fg3a / minutes),
            fg2a_rate = 36 * (fg2a / minutes), 
            fta_rate = 36 * (fta / minutes), 
            ft_pct =  (ftm / fta), 
            fg2_pct =  (fg2m / fg2a), 
            fg3_pct =  (fg3m / fg3a),
            usg = (usg / 100) + .01 )
injury_data <- data |> 
    group_by(id) |>
    filter(!any(is.na(first_major_injury))) |> 
    ungroup() |> 
    select(name, id, obpm, dbpm, pct_games, usg, mpg, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period, year) |>
    rename(pct_minutes = mpg, games = pct_games, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov,
             oreb, dreb, stl, fg3a, fg2a,
             fta, ftm, fg2m, fg3m, usg),
    names_to = "metric",
    values_to = "obs_value")
print("pivoted the original data")

non_injury_data <- data |> 
    group_by(id) |>
    filter(any(is.na(first_major_injury))) |> 
    ungroup() |> 
    select(name, id, obpm, dbpm, pct_games, mpg, usg, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period, year) |>
    rename(pct_minutes = mpg, games = pct_games, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov,
             oreb, dreb, stl, fg3a, fg2a,
             fta, ftm, fg2m, fg3m, usg),
    names_to = "metric",
    values_to = "obs_value")

causal_empirical <- ggplot(injury_data |> group_by(id, injury_period, metric, first_major_injury) |> summarize(max_val = mean(obs_value)) |> ungroup() |> pivot_wider(
    names_from = injury_period,   # the values in this column become column names
    values_from = max_val           # the values to fill in those new columns
) |> mutate(empirical_change = `pre-injury` - `post-injury`) |>  mutate(metric = toupper(metric),
                              metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric)) |> filter(first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Foot Fracture", "Meniscus")), 
                              aes(x = first_major_injury, y = empirical_change)) + 
                              geom_boxplot(outlier.shape = NA, width = 0.6) + facet_wrap(~metric, scales = "free_y") + 
                              scale_fill_brewer(palette = "Set1") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + 
  labs(x = "Injury Type", y = "Empirical Average Treatment Effect (ATT)", title = "Empirical Distribution of Average Treatment Effect by Metric")

ggsave(file.path(plots_dir, "causal", "empirical_att_causal_plot.png"), causal_empirical)

posterior_data <- read_parquet(file.path(model_dir, "posterior_counterfactual_ar.parquet"))
posterior_injury_effect <- read_parquet(file.path(model_dir, "posterior_injury_effect.parquet"))
print("loaded the posterior data")
posterior_peaks <- read_parquet(file.path(model_dir, "posterior_peaks_ar.parquet"))
latent_space <- read_parquet(file.path(model_dir, "phi_X.parquet")) |>
  rename_with(~ gsub(" ", "", .), starts_with("Dim"))

exit_age_data <- read_parquet(
  file.path(model_dir, "posterior_exit_age_sample.parquet"),
  col_select = c("player", "chain", "sample", "value",
                 "observed_entrance_age", "observed_exit_age", "exit_censored", "scenario")
)

posterior_data <- posterior_data |> mutate(value = case_when(metric == "pct_minutes" ~ value * 48, 
                                                             metric == "games" ~ value,
                                                             .default = value)) |> 
                                    inner_join(posterior_peaks |> 
                                    rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
joined_data <- posterior_data |> 
                left_join(injury_data, by = c("player" = "id", "metric", "age")) |> 
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, first_major_injury, .direction  = "downup") |> 
                fill(injury_period, .direction = "up") |> mutate(injury_period = replace_na(injury_period, "post-injury")) |> 
                mutate(
                  base_age = if_else(!is.na(year), age, NA_integer_),
                  base_year = if_else(!is.na(year), year, NA_integer_)) |>
                
                fill(base_age, base_year, .direction = "downup") |>
                mutate(
                  year = if_else(is.na(year), base_year + (age - base_age), year)) |> 
                select(-base_age, -base_year) |> ungroup() |> 
                mutate(metric = toupper(metric),
                              metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))
joined_data_uninjured <- posterior_data |> 
                left_join(non_injury_data, by = c("player" = "id", "metric", "age")) |> 
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, first_major_injury, .direction  = "downup") |> 
                fill(injury_period, .direction = "up") |> mutate(injury_period = replace_na(injury_period, "post-injury")) |> 
                mutate(
                  base_age = if_else(!is.na(year), age, NA_integer_),
                  base_year = if_else(!is.na(year), year, NA_integer_)) |>
                
                fill(base_age, base_year, .direction = "downup") |>
                mutate(
                  year = if_else(is.na(year), base_year + (age - base_age), year)) |> 
                select(-base_age, -base_year) |> ungroup() |> 
                mutate(metric = toupper(metric),
                              metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))
print("joined the data with predictions")

# Observed post-injury ages: only ages where the player was actually seen post-injury
observed_post_injury_pts <- injury_data |>
  filter(!is.na(year), injury_period == "post-injury") |>
  select(id, metric, age) |>
  distinct()

# Full latent baseline = mu + AR + TREND_AR (no injury effect).
# posterior_mu_ar = mu + TREND_AR; posterior_latent_ar = player AR component.
posterior_baseline_latent <- read_parquet(file.path(model_dir, "posterior_mu_ar.parquet")) |>
  semi_join(observed_post_injury_pts, by = c("player" = "id", "metric", "age")) |>
  inner_join(
    read_parquet(file.path(model_dir, "posterior_latent_ar.parquet")) |>
      semi_join(observed_post_injury_pts, by = c("player" = "id", "metric", "age")) |>
      rename(ar_val = value),
    by = c("player", "chain", "sample", "metric", "age")
  ) |>
  mutate(mu_ar = value + ar_val) |>
  select(player, chain, sample, metric, age, mu_ar)

# Model-based ATT: relative injury effect at observed post-injury ages only.
# Log-link:   exp(effect)                          — rate multiplier relative to baseline
# Logit-link: plogis(mu_ar + effect)/plogis(mu_ar) — probability ratio relative to baseline
# Gaussian:   raw additive offset
att_base <- posterior_injury_effect |>
  semi_join(observed_post_injury_pts, by = c("player" = "id", "metric", "age")) |>
  inner_join(posterior_baseline_latent, by = c("player", "chain", "sample", "metric", "age")) |>
  mutate(
    relative_effect = case_when(
      metric %in% c("blk", "ast", "tov", "oreb", "dreb", "stl", "fg3a", "fg2a", "fta") ~
        exp(value),
      metric %in% c("fg2m", "ftm", "games", "fg3m", "usg", "pct_minutes") ~
        plogis(mu_ar + value) / plogis(mu_ar),
      .default = value
    )
  ) |>
  inner_join(
    posterior_peaks |> rename(peak_age = value),
    by = c("player", "chain", "sample", "metric")
  ) |>
  left_join(
    injury_data |> select(id, metric, age, first_major_injury, name) |> distinct(),
    by = c("player" = "id", "metric", "age")
  ) |>
  group_by(player, chain, sample, metric) |>
  arrange(age) |>
  fill(first_major_injury, name, .direction = "downup") |>
  mutate(
    injury_peak_type = if_else(min(age) > first(peak_age), "post-peak", "pre-peak"),
    relative_effect  = if_else(is.finite(relative_effect), relative_effect, NA_real_)
  ) |>
  summarize(
    player_att         = mean(relative_effect, na.rm = TRUE),
    injury_peak_type   = first(injury_peak_type),
    first_major_injury = first(na.omit(first_major_injury)),
    .groups = "drop"
  ) |>
  mutate(
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES"       ~ "GP%",
      metric == "FG2M"        ~ "FG2%",
      metric == "FG3M"        ~ "FG3%",
      metric == "FTM"         ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    )
  )

injury_types_filter <- c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon",
                          "Quad Tendon", "Foot Fracture", "Lower Body Fracture", "Meniscus")

# Null-effect reference lines per metric (log/logit → 1, Gaussian → 0)
null_effect_df <- tibble(
  metric    = c("BLK", "AST", "TOV", "OREB", "DREB", "STL", "FG3A", "FG2A", "FTA",
                "GP%", "FG2%", "FG3%", "FT%", "USG", "MPG",
                "OBPM", "DBPM"),
  null_val  = c(rep(1, 15), 0, 0)
)

att_plot <- att_base |>
  filter(first_major_injury %in% injury_types_filter) |>
  group_by(first_major_injury, metric, injury_peak_type) |>
  summarize(
    mean_att = mean(player_att, na.rm = TRUE),
    lower    = HDInterval::hdi(player_att, credMass = 0.95)["lower"],
    upper    = HDInterval::hdi(player_att, credMass = 0.95)["upper"],
    .groups  = "drop"
  ) |>
  ggplot(aes(x = first_major_injury, y = mean_att, color = injury_peak_type)) +
  geom_hline(data = null_effect_df, aes(yintercept = null_val), linetype = "dashed", color = "gray50") +
  geom_point(size = 3) + scale_colour_brewer(palette = "Set1") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .2) +
  facet_wrap(~metric, scales = "free_y") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(y = "Injury Effect Ratio vs Baseline (95% CI)",
       x = "Injury", color = "Injury Timing") +
  ggtitle("Injury Effect Ratio per Injury Type, by Metric")
ggsave(file.path(plots_dir, "causal", "att_causal_plot.png"), att_plot)

att_plot_total <- att_base |>
  filter(first_major_injury %in% injury_types_filter) |>
  group_by(metric, injury_peak_type) |>
  summarize(
    mean_att = mean(player_att, na.rm = TRUE),
    lower    = HDInterval::hdi(player_att, credMass = 0.95)["lower"],
    upper    = HDInterval::hdi(player_att, credMass = 0.95)["upper"],
    .groups  = "drop"
  ) |>
  ggplot(aes(x = injury_peak_type, y = mean_att, color = injury_peak_type)) +
  geom_hline(data = null_effect_df, aes(yintercept = null_val), linetype = "dashed", color = "gray50") +
  geom_point(size = 3) + scale_colour_brewer(palette = "Set1") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .2) +
  facet_wrap(~metric, scales = "free_y") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(y = "Injury Effect Ratio vs Baseline (95% CI)",
       x = "Injury Timing", color = "Injury Timing") +
  ggtitle("Injury Effect Ratio by Metric")
ggsave(file.path(plots_dir, "causal", "att_causal_plot_total.png"), att_plot_total)

umap_latent_space <- umap(latent_space |> select(starts_with("Dim")), n_neighbors = 15, min_dist = 0.001, verbose = TRUE)

umap_df <- latent_space |> select(-starts_with("Dim")) |> cbind(as.tibble(umap_latent_space, .name_repair = "minimal") |> rename(Dim1 = 1, Dim2 = 2))

plot_names <-  c("Stephen Curry", "Kevin Durant", "LeBron James", "Isaiah Thomas", "Kobe Bryant", "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                "Chris Paul", "Shaquille O'Neal","Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd","Marcus Camby", "Rudy Gobert", "Tim Duncan",
                 "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", 
                 "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", "Giannis Antetokounmpo", "Jrue Holiday")

umap_df <- umap_df |> mutate(plot_name = if_else(name %in% plot_names, TRUE, FALSE))




itt_df <- joined_data |> filter(injury_period == "post-injury") |> group_by(player, metric, chain, sample) |> 
mutate(injury_type = if_else(min(age) > peak_age, "post-peak", "pre-peak")) |> ungroup() |>
mutate(obs_value = if_else(year <= 2026 & metric %in% c("games") & is.na(obs_value), 0, obs_value),
value = if_else(is.finite(value), value, NA_real_)) |>
filter(!is.na(obs_value) & is.finite(obs_value)) |> 
mutate(injury_change =  obs_value - value) |> ungroup() |> group_by(metric, player, first_major_injury, injury_type, age) |> 
summarize(mean_itt = mean(injury_change, na.rm = TRUE), lower = HDInterval::hdi(injury_change, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(injury_change, credMass = 0.95)["upper"]) |> ungroup()


metrics <- na.omit(unique(itt_df$metric))
all_ids <- unique(umap_df$id)

# Create full grid of id × metric
expanded <- expand_grid(id = all_ids, metric = metrics)


joined_itt_df <- itt_df |> right_join(umap_df |> inner_join(expanded), by = c("player" = "id", "metric"))

plot_umap <- function(grouped_data_set){
  # Load libraries


# Plot
 plt <- ggplot() +
  # Background: non-injured players faded
    geom_point(
    data = filter(grouped_data_set, is.na(first_major_injury)),
    aes(x = Dim1, y = Dim2, color = position_group),
    size = 1.5,
    alpha = 0.1,
    
  ) +
    new_scale_color() +
  # Injured players with shape by injury type, color by ITE
  geom_point(
    data = filter(grouped_data_set, first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Foot Fracture", "Lower Body Fracture", "Meniscus") & injury_type == "pre-peak"),
    aes(shape = first_major_injury, color = mean_itt, x =Dim1, y = Dim2),
    size = 4,
    alpha = 1
  ) +


  # Labels for chosen players only
  geom_text_repel(
    data = filter(grouped_data_set, !is.na(first_major_injury) & plot_name),
    aes(label = name, x = Dim1, y = Dim2),
    size = 3,
    max.overlaps = 15,
    box.padding = 0.3
  ) +
  # Manual shapes for injury types
  scale_shape_manual(values = c(
    "ACL" = 4,       # x
    "Achilles" = 1, # filled circle
    "Meniscus" = 8 ,  # star
    "Patellar Tendon" = 0, 
    "Quad Tendon" = 3,
    "Back/Spine" = 2,
    "Lower Body Fracture" = 11,
    "Foot Fracture" = 9,
    "Hip" = 5
  )) +
  # Color gradient for ITE values (diverging)
  scale_color_gradient2(
    low = "blue", mid = "white", high = "red",
    name = "ITE magnitude"
  ) +
  theme_bw() +
  labs(
    title = "Player ITE and Injury Type in Latent Space",
    x = "Latent Dimension 1",
    y = "Latent Dimension 2",
    shape = "Injury Type"
  ) +
  theme(legend.position = "right")
  return(plt)
}


plot_counterfactual_metrics <- function(grouped_data_set) {

  injury_type <- unique(grouped_data_set$first_major_injury)
  group_name <- unique(grouped_data_set$name)
  

  raw_df <- grouped_data_set |>
    group_by(injury_period, chain, sample, metric) |> summarize(
      counterfactual_value = case_when(metric == "GP%" ~ 82*sum(value, na.rm = TRUE),
                                       .default = mean(value, na.rm = TRUE)),
      total_obs_value = case_when(metric == "GP%" ~ 82*sum(if_else(is.na(obs_value),0,obs_value)),
                                  .default = mean(obs_value, na.rm = TRUE)),
      first_major_injury = first(first_major_injury),
      player = first(player),
      name = first(name)
    ) |> ungroup() 
  obs_df <- raw_df %>%
    group_by(injury_period, metric) %>%
    summarize(total_obs_value = first(total_obs_value), .groups = "drop")

  # 2. compute densities per facet to get y max
  dens_df <- raw_df %>%
    group_by(injury_period, metric) %>%
    summarize(d = list(density(counterfactual_value, na.rm = TRUE)), .groups = "drop") %>%
    mutate(y_max = map_dbl(d, ~ max(.x$y))) %>%
    select(-d)

  # 3. join obs values with y positions
  label_df <- obs_df %>%
    left_join(dens_df, by = c("injury_period", "metric")) %>%
    mutate(
      y = y_max * 0.4,
      label = case_when(metric == "GP%" ~ glue("Observed Total: {round(total_obs_value)}"),
                        .default = glue("Observed: {round(total_obs_value, 2)}"))
    )
  # plt_gp <- ggplot(raw_df |> filter(metric == "GP%"), aes(x = counterfactual_value)) + 
  #   geom_density(adjust=3, from = 0, to = max(raw_df |> filter(metric == "GP%") |> select(counterfactual_value))*1.5) + 
  #   geom_vline(data = obs_df |> filter(metric == "GP%"),
  #            aes(xintercept = total_obs_value),
  #            color = "blue", size = 1, linetype = "dashed") +
  #   geom_text(data = label_df |> filter(metric == "GP%"),
  #           aes(x = total_obs_value, y = y, label = label),
  #           inherit.aes = FALSE, vjust = -0.5, color = "black", angle = 270) + scale_x_continuous(limits = c(0, NA)) + coord_cartesian(clip = "off") +
  #   labs(y = "Posterior Predictive Density", x = "Total Games Played (Post-Inury)") +  theme_classic() # start at 0, no extra padding 
  
  # plt_fta <- ggplot(raw_df |> filter(metric == "FTA"), aes(x = counterfactual_value)) + 
  #   geom_density(adjust=3, from = 0, to = max(raw_df |> filter(metric == "FTA") |> select(counterfactual_value))*1.5) + 
  #   geom_vline(data = obs_df |> filter(metric == "FTA"),
  #            aes(xintercept = total_obs_value),
  #            color = "blue", size = 1, linetype = "dashed") +
  #   geom_text(data = label_df |> filter(metric == "FTA"),
  #           aes(x = total_obs_value, y = y, label = label),
  #           inherit.aes = FALSE, vjust = -0.5, color = "black", angle = 270) + 
  #   labs(y = "", x = "Avg. FTA Per 36 (Post-Injury)") +  theme_classic() + scale_x_continuous(limits = c(0, NA)) + coord_cartesian(clip = "off") 

  # plt <- plt_gp + plt_fta + plot_annotation(title = glue("Counterfactual Posterior Distribution Post {injury_type} Injury: {group_name}"))

  plt <- raw_df %>% ggplot(aes(x = counterfactual_value)) + geom_density(adjust = 3) + 
         geom_vline(data = obs_df, aes(xintercept = total_obs_value), color = "blue", size = 1, linetype = "dashed") + geom_text(data = label_df, aes(x = total_obs_value, y = y, label = label), 
         inherit.aes = FALSE, vjust = -0.5, color = "black", angle = 270) + facet_wrap(~ metric, scales = "free") + scale_x_continuous(limits = c(0, NA)) + coord_cartesian(clip = "off") +
    labs(y = "Posterior Predictive Density", x = "Counterfactual Value", title = glue("Counterfactual Posterior Distribution Post {injury_type} Injury: {group_name}")) +  theme_classic() # start at 0, no extra padding 
  return(plt)
}



plot_counterfactual <- function(grouped_data_set) {

  group_name <- unique(grouped_data_set$name)
  raw_plt <-
  grouped_data_set |> mutate(
    age_of_injury = if_else(injury_period == "post-injury", age, Inf),
    age_of_injury = min(age_of_injury)
  ) 
  
  injury_label <- raw_plt |> group_by(metric) |> 
    summarise(
      upper = HDInterval::hdi(value, credMass = .95)["upper"],
      max_upper = max(upper),
      age_of_injury = first(age_of_injury),
      first_major_injury = first(first_major_injury)
    )
  
  plt <- raw_plt |>
    group_by(metric, age) |> summarize(
      posterior_mean = mean(value, na.rm = TRUE),
      upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
      max_upper = max(upper),
      lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
      obs_value = first(obs_value),
      first_major_injury = first(first_major_injury),
      injury_period = first(injury_period),
      age_of_injury = first(age_of_injury),
      player = first(player),
      name = first(name)
    ) |> ungroup() |>
    ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.3) +
    geom_line(aes(x = age, y = posterior_mean)) +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_vline(aes(xintercept = age_of_injury),
               linetype = "dashed",
               color = "red") +
    geom_text(
      data = injury_label, 
      size = 2,
      aes(x = age_of_injury, y = max_upper, label = first_major_injury),
      
      color = "black"
    )  +
    facet_wrap( ~ metric, scales = "free_y") + 
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Counterfactual Career Trajectory: ", group_name)) + theme_bw()
  return(plt)
}


plots_list <- joined_data %>% filter(metric %in% c("MPG", "OBPM", "FTA", "USG")) %>% 
  mutate(value = if_else(is.finite(value), value, NA_real_)) %>% 
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_counterfactual(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = file.path(plots_dir, "causal", glue("{name}.png")),
      plot = plt,
    )
    })

plots_list <- joined_data %>%
filter(injury_period == "post-injury" & year <= 2026 & metric %in% c("MPG", "FTA", "OBPM", "USG")) %>%
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_counterfactual_metrics(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = file.path(plots_dir, "causal", glue("{name}_metrics_itt.png")),
      plot = plt,
    )
    })


plots_list <- joined_itt_df %>%
  group_by(metric) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_umap(.x)
    name <- gsub("%", "",unique( .x$metric))
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = file.path(plots_dir, "causal", glue("{name}_itt.png")),
      plot = plt,
    )
    })





#### ACTUAL PLOTS FOR PAPER 

min_data <- data  %>% mutate(normalized_min_played = replace_na(82 * pct_games * mpg,0), normalized_games_played = replace_na(82 * pct_games, 0))  %>% select(age, id, normalized_min_played, normalized_games_played, year)

# isaiah_thomas_metrics <- plot_counterfactual_metrics(joined_data %>%
# filter(injury_period == "post-injury" & year <= 2026 & metric %in% c("FTA", "OBPM") & name == "Isaiah Thomas"))

# derrick_rose_metrics <- plot_counterfactual_metrics(joined_data %>%
# filter(injury_period == "post-injury" & year <= 2026 & metric %in% c("FTA", "OBPM") & name == "Derrick Rose"))

# isaiah_thomas_mpg_since <- joined_data %>% filter(metric %in% c("GP%", "MPG") & name == "Isaiah Thomas") %>% 
#                            pivot_wider(names_from = metric, values_from = c(value, obs_value, peak_age)) %>% mutate(minutes_played_sample = 82 * value_MPG * `value_GP%`,
#                           age_of_injury = if_else(injury_period == "post-injury", age, Inf),
#                           age_of_injury = min(age_of_injury))  %>% left_join(min_data, by = c("age" = "age", "player" = "id")) %>% filter(age >= age_of_injury) %>% group_by(chain, sample) %>%  arrange(age) %>% mutate(cum_min_played_sample = cumsum(replace_na(minutes_played_sample,0)),
#                           cum_min_played_obs = cumsum(replace_na(normalized_min_played,0)))  %>% group_by(age) %>% 
#                           summarize(posterior_mean = mean(cum_min_played_sample, na.rm = TRUE),
#                                       upper = HDInterval::hdi(cum_min_played_sample, credMass = 0.95)["upper"],
#                                       max_upper = max(upper),
#                                       lower = HDInterval::hdi(cum_min_played_sample, credMass = 0.95)["lower"],
#                                       obs_value = first(cum_min_played_obs),
#                                       first_major_injury = first(first_major_injury),
#                                       injury_period = first(injury_period),
#                                       age_of_injury = first(age_of_injury),
#                                       player = first(player),
#                                       name = first(name)) %>% ungroup()
# derrick_rose_mpg_since <- joined_data %>% filter(metric %in% c("GP%", "MPG") & name == "Derrick Rose") %>% 
#                            pivot_wider(names_from = metric, values_from = c(value, obs_value, peak_age)) %>% mutate(minutes_played_sample = 82 * value_MPG * `value_GP%`,
#                           age_of_injury = if_else(injury_period == "post-injury", age, Inf),
#                           age_of_injury = min(age_of_injury)) %>% left_join(min_data, by = c("age" = "age", "player" = "id")) %>% filter(age >= age_of_injury) %>% group_by(chain, sample) %>%  arrange(age) %>% mutate(cum_min_played_sample = cumsum(replace_na(minutes_played_sample,0)),
#                           cum_min_played_obs = cumsum(replace_na(normalized_min_played,0)))  %>% group_by(age) %>% 
#                           summarize(posterior_mean = mean(cum_min_played_sample, na.rm = TRUE),
#                                       upper = HDInterval::hdi(cum_min_played_sample, credMass = 0.95)["upper"],
#                                       max_upper = max(upper),
#                                       lower = HDInterval::hdi(cum_min_played_sample, credMass = 0.95)["lower"],
#                                       obs_value = first(cum_min_played_obs),
#                                       first_major_injury = first(first_major_injury),
#                                       injury_period = first(injury_period),
#                                       age_of_injury = first(age_of_injury),
#                                       player = first(player),
#                                       name = first(name)) %>% ungroup()

# derrick_rose_mpg <- ggplot(derrick_rose_mpg_since, aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
#                                        fill = "gray",
#                                        alpha = 0.3) +
#               geom_line(aes(x = age, y = posterior_mean)) +
#               geom_line(aes(x = age, y = obs_value), color = "blue", linetype = "dashed") +
#               geom_vline(aes(xintercept = age_of_injury),
#                linetype = "dashed",
#                color = "red") +
#                 geom_text(
#                   size = 4,
#                   aes(x = age_of_injury, y = mean(max_upper), label = unique(first_major_injury)),
                  
#                   color = "black"
#                 ) + 
#               annotate("text", x = mean(derrick_rose_mpg_since$age), y = tail(derrick_rose_mpg_since$posterior_mean, 1),
#                             label = "Counterfactual Posterior Mean", hjust = -0.1, vjust = 0.5) +
#               annotate("text", x = mean(derrick_rose_mpg_since$age), y = tail(derrick_rose_mpg_since$obs_value, 1),
#                       label = "Observed", hjust = -0.1, vjust = 0.5, color = "blue") +

#               labs(x = "Age", y = "Cumulative Minutes Played Since Injury") + theme_classic() + ggtitle("Counterfactual Cumulative Minutes Played: Derrick Rose")

# isaiah_thomas_mpg <- ggplot(isaiah_thomas_mpg_since, aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
#                                        fill = "gray",
#                                        alpha = 0.3) +
#               geom_line(aes(x = age, y = posterior_mean)) +
#               geom_line(aes(x = age, y = obs_value), color = "blue", linetype = "dashed") +
#               geom_vline(aes(xintercept = age_of_injury),
#                linetype = "dashed",
#                color = "red") +
#                 geom_text(
#                   size = 4,
#                   aes(x = age_of_injury, y = mean(max_upper), label = unique(first_major_injury)),
                  
#                   color = "black"
#                 ) + 
#               annotate("text", x = mean(isaiah_thomas_mpg_since$age), y = tail(isaiah_thomas_mpg_since$posterior_mean, 1),
#                             label = "Counterfactual Posterior Mean", hjust = -0.1, vjust = 0.5) +
#               annotate("text", x = mean(isaiah_thomas_mpg_since$age), y = tail(isaiah_thomas_mpg_since$obs_value, 1),
#                       label = "Observed", hjust = -0.1, vjust = 0.5, color = "blue") +

#               labs(x = "Age", y = "Cumulative Minutes Played Since Injury") + theme_classic() + ggtitle("Counterfactual Cumulative Minutes Played: Isaiah Thomas")


# derrick_rose <- derrick_rose_mpg + derrick_rose_metrics
# isaiah_thomas <- isaiah_thomas_mpg + isaiah_thomas_metrics 

# total_plot <- (derrick_rose / isaiah_thomas) 
# ggsave(file.path(plots_dir, "causal", "player_comparison.png", total_plot, width = 21, height = 14)




minutes_lost <- joined_data %>% filter(metric %in% c("GP%", "MPG") & !is.na(first_major_injury) & year <= 2026) %>% 
                           pivot_wider(names_from = metric, values_from = c(value, obs_value, peak_age)) %>% mutate(minutes_played_sample = 82 * value_MPG * `value_GP%`, games_played_sample = 82 * `value_GP%`) %>% group_by(player, chain, sample) %>% mutate(
                          age_of_injury = if_else(injury_period == "post-injury", age, Inf),
                          age_of_injury = min(age_of_injury)) %>% ungroup () %>% left_join(min_data, by = c("age" = "age", "player" = "id")) %>% filter(age > age_of_injury + 1) %>% group_by(first_major_injury, chain, sample) %>%
                          summarize(total_pred_games = sum(games_played_sample), total_game_obs = sum(replace_na(normalized_games_played,0)), total_pred_min = sum(minutes_played_sample), total_min_obs = sum(replace_na(normalized_min_played,0)), ratio = total_min_obs / total_pred_min, ratio_games = total_game_obs / total_pred_games) %>% ungroup()

minutes_lost_player <- joined_data %>% filter(metric %in% c("GP%", "MPG") & !is.na(first_major_injury) & year <= 2026) %>% 
                           pivot_wider(names_from = metric, values_from = c(value, obs_value, peak_age)) %>% mutate(minutes_played_sample = 82 * value_MPG * `value_GP%`, games_played_sample = 82 * `value_GP%`) %>% group_by(player, chain, sample) %>% mutate(
                          age_of_injury = if_else(injury_period == "post-injury", age, Inf),
                          age_of_injury = min(age_of_injury)) %>% ungroup () %>% left_join(min_data, by = c("age" = "age", "player" = "id")) %>% filter(age > age_of_injury + 1) %>% group_by(first_major_injury, chain, sample, player) %>%
                          summarize(total_pred_games = sum(games_played_sample), total_game_obs = sum(replace_na(normalized_games_played,0)), total_pred_min = sum(minutes_played_sample), total_min_obs = sum(replace_na(normalized_min_played,0)), ratio = (total_min_obs + 1) / (total_pred_min + 1), ratio_games = (total_game_obs + 1) / (total_pred_games + 1), diff_games = total_pred_games - total_game_obs, diff = total_pred_min - total_min_obs, age_of_injury = min(age_of_injury)) %>% ungroup()



uninjured <- joined_data_uninjured %>% filter(metric %in% c("GP%", "MPG")  & year <= 2026 ) %>% 
                           pivot_wider(names_from = metric, values_from = c(value, obs_value, peak_age)) %>% mutate(minutes_played_sample = 82 * value_MPG * `value_GP%`, games_played_sample = 82 * `value_GP%`) %>% left_join(min_data, by = c("age" = "age", "player" = "id", "year")) %>% filter(is.na(first_major_injury))

n_uninjured <- min_data %>% group_by(id) %>% summarize(enter_age = min(age), exit_age = if_else(max(year) == 2025, max(age), 38), range = exit_age - enter_age) %>% ungroup() %>% filter(range >= 1) %>% rowwise() %>%
  mutate(randomized_age_of_injury = sample(seq(enter_age, exit_age), 1)) %>% ungroup()
uninjured_randomized <- uninjured %>% inner_join(n_uninjured, by = c("player" = "id"))
minutes_lost_contrast <- uninjured_randomized %>% filter(age > randomized_age_of_injury + 1) %>% group_by(chain, sample) %>%
                          summarize(first_major_injury = "Placebo", total_pred_games = sum(games_played_sample), total_game_obs = sum(replace_na(normalized_games_played,0)), ratio_games = total_game_obs / total_pred_games, total_pred_min = sum(minutes_played_sample), total_min_obs = sum(replace_na(normalized_min_played,0)), ratio = total_min_obs / total_pred_min) %>% ungroup() 

minutes_lost_total <- minutes_lost %>% bind_rows(minutes_lost %>% filter(!is.na(first_major_injury)) %>% group_by(chain, sample) %>%
      summarize(
        first_major_injury = "All Injuries",
        total_pred_min = sum(total_pred_min, na.rm = TRUE),
        total_min_obs = sum(total_min_obs, na.rm = TRUE),
        ratio = total_min_obs / total_pred_min,
        total_pred_games = sum(total_pred_games, na.rm = TRUE),
        total_game_obs = sum(total_game_obs, na.rm = TRUE),
        ratio = total_min_obs / total_pred_min,
        ratio_games = total_game_obs / total_pred_games,
        .groups = "drop"
      ), minutes_lost_contrast ) 


injury_summary <- injury_data |> group_by(id) |> summarize(first_major_injury = first(first_major_injury)) |> ungroup() |> group_by(first_major_injury) |> summarize(n = n()) |> ungroup() %>% bind_rows(summarize(., across(where(is.numeric), sum, na.rm = TRUE)) %>%
      mutate(first_major_injury = "All Injuries"), n_uninjured %>% count() %>% mutate(first_major_injury = "Placebo"))
test_plt <- minutes_lost_total %>% inner_join(injury_summary, by = "first_major_injury") %>% filter(n >= 1) %>% mutate(first_major_injury = fct_reorder(first_major_injury, ratio, .fun = mean, .desc = TRUE))  %>% 
  ggplot(aes(x = ratio, y = first_major_injury, )) + stat_pointinterval() + geom_vline(aes(xintercept = mean(minutes_lost_contrast$ratio)), linetype = "dashed", color = "red") + 

 geom_text(data = injury_summary %>% filter(n >= 1), aes(x = .35, y = first_major_injury, label = glue("N = {n}")))  + xlim(c(.3, 1.5)) +
theme_classic()  + labs(x = "Ratio of Observed Minutes Played to Predicted Minutes Played", y = "First Major Injury", title = "Injury Impact on Reduction in Minutes Played") 
ggsave(file.path(plots_dir, "causal", "minutes_lost.png"), test_plt)
test_plt_games <- minutes_lost_total %>% inner_join(injury_summary, by = "first_major_injury") %>% filter(n >= 1) %>% mutate(first_major_injury = fct_reorder(first_major_injury, ratio_games, .fun = mean, .desc = TRUE))  %>% 
  ggplot(aes(x = ratio_games, y = first_major_injury, )) + stat_pointinterval() + geom_vline(aes(xintercept = mean(minutes_lost_contrast$ratio_games)), linetype = "dashed", color = "red") + 


 geom_text(data = injury_summary %>% filter(n >= 1), aes(x = .35, y = first_major_injury, label = glue("N = {n}")))  + xlim(c(.3, 1.5)) +
theme_classic()  + labs(x = "Ratio of Observed Games Played to Predicted Games Played", y = "First Major Injury", title = "Injury Impact on Reduction in Games Played") 


ggsave(file.path(plots_dir, "causal", "games_lost.png"), test_plt_games)



latent_injuries <- latent_space %>% left_join(minutes_lost_player %>% group_by(player) %>% summarize(avg_log_ratio = log(mean(ratio)), first_major_injury = first(first_major_injury), age_of_injury = mean(age_of_injury)), 
                                              by = c("id" = "player")) %>% mutate(first_major_injury = if_else(is.na(first_major_injury), "No Injury", first_major_injury))

latent_injuries_pca <- latent_injuries %>% select(starts_with("Dim")) %>%  prcomp(center = TRUE, scale. = TRUE) %>%       # perform PCA
  .$x %>%                                       # extract principal component scores
  as.data.frame() %>%                            # convert to data frame
  as_tibble(.name_repair = "unique") 

latent_injuries_pca$id =  latent_injuries$id
latent_injuries_pca <- latent_injuries_pca %>% inner_join(latent_injuries %>% select(id, first_major_injury, age_of_injury, position_group, avg_log_ratio, minutes, name))

injury_plot <- latent_injuries_pca %>% ggplot(aes(x = PC1, y = PC2, alpha = if_else(first_major_injury == "No Injury", .1, 1))) + geom_point( aes(color = first_major_injury,)) + theme_classic() + scale_color_brewer(palette = "Set1") +
labs(alpha = NULL, color = "First Major Injury", title = "PCA of Latent Embedding") + guides(alpha = "none") + coord_cartesian(xlim = c(-max(abs(latent_injuries_pca$PC1)), max(abs(latent_injuries_pca$PC1))),
                  ylim = c(-max(abs(latent_injuries_pca$PC2)), max(abs(latent_injuries_pca$PC2))))


injury_plot_2 <- latent_injuries_pca %>% filter(first_major_injury %in% c("Achilles")) %>% ggplot(aes(x = PC1, y = PC2)) + geom_point( aes(color = avg_log_ratio)) + theme_classic() + scale_color_gradient(low = "blue", high = "green") +
geom_text_repel(aes(label = name), size = 3, max.overlaps = 20) +
labs(alpha = NULL, color = "Avg. Log Ratio (Observed / Predicted)", title = "PCA of Latent Embedding (Achilles Injuries)") + coord_cartesian(xlim = c(-max(abs(latent_injuries_pca$PC1)), max(abs(latent_injuries_pca$PC1))),
                  ylim = c(-max(abs(latent_injuries_pca$PC2)), max(abs(latent_injuries_pca$PC2))))


injury_plot_3 <- latent_injuries_pca %>% filter(first_major_injury %in% c("Achilles")) %>% ggplot() + geom_boxplot(aes(x = factor(age_of_injury), y = avg_log_ratio)) + geom_hline(aes(yintercept = log(.8)), linetype = "dashed", color = "red") + theme_classic() +
labs(x = "Age of Achilles Injury", y = "Avg. Log Ratio (Observed / Predicted)", title = "Log (Observed / Predicted) Minutes vs. Age of Injury")


ggsave(file.path(plots_dir, "causal", "minutes_lost_latent_space.png"), (injury_plot + injury_plot_2), width = 14)
ggsave(file.path(plots_dir, "causal", "achilles_vs_age.png"), injury_plot_3)



### injury latent factor analysis

injury_mean_posterior <- read_parquet(file.path(model_dir, "posterior_injury_prior_mean.parquet"))

gaussian_metrics <- c("obpm", "dbpm")
count_metrics <- c("blk", "stl", "ast", "dreb", "oreb", "tov", "fta", "fg2a", "fg3a")
proportion_metrics <- c("games", "retirement", "pct_minutes", "usg", "ftm", "fg2m", "fg3m")

injury_mean_posterior <- injury_mean_posterior %>%
  mutate(
    value_link = value,
    effect_scale = case_when(
      metric %in% gaussian_metrics ~ "additive",
      metric %in% count_metrics ~ "rate_multiplier",
      metric %in% proportion_metrics ~ "odds_ratio",
      TRUE ~ "additive"
    ),
    value = case_when(
      metric %in% gaussian_metrics ~ value,
      metric %in% count_metrics ~ exp(value),
      metric %in% proportion_metrics ~ exp(value),
      TRUE ~ value
    ),
    value = case_when(
      effect_scale %in% c("rate_multiplier", "odds_ratio") ~ value - 1,
      TRUE ~ value
    )
  )

injury_effect_plot_data <- injury_mean_posterior %>%
  mutate(
    sample = if ("sample" %in% names(.)) sample else samples,
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES" ~ "GP%",
      metric == "FG2M" ~ "FG2%",
      metric == "FG3M" ~ "FG3%",
      metric == "FTM" ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    ),
    injury_type = as.factor(injury_type)
  ) %>%
  filter(!is.na(injury_type), !is.na(value), is.finite(value))

injury_effect_interval_data <- injury_effect_plot_data %>%
  group_by(metric, injury_type) %>%
  ggdist::median_qi(value, .width = 0.95) %>%
  ungroup() %>%
  mutate(
    interval_color = case_when(
      .lower > 0 ~ "Above 0",
      .upper < 0 ~ "Below 0",
      TRUE ~ "Overlaps 0"
    )
  )

injury_effect_interval_plot <- injury_effect_interval_data %>%  filter(metric != "RETIREMENT") %>%
  ggplot(aes(x = injury_type, y = value, color = interval_color)) +
  geom_linerange(aes(ymin = .lower, ymax = .upper), linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  scale_y_continuous(limits = function(lims) {
    lim <- max(abs(lims), na.rm = TRUE)
    c(-lim, lim)
  }) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(values = c("Above 0" = "blue", "Below 0" = "red", "Overlaps 0" = "grey50")) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(
    x = "Injury Type",
    y = "Posterior Injury Effect",
    color = "95% Interval",
    title = "Posterior Injury Effect by Injury Type and Metric"
  )

ggsave(
  file.path(plots_dir, "causal", "injury_mean_posterior_interval_by_metric.png"),
  injury_effect_interval_plot,
  width = 14,
  height = 9
)

# ── Trace plots: posterior prior mean per metric × injury type ──────────────
# x = sample index within chain, y = value, colour = chain.
# Facet grid: injury_type (rows) × metric (columns).
injury_trace_data <- injury_effect_plot_data |>
  filter(metric != "RETIREMENT") |>
  mutate(chain = factor(chain))

injury_trace_metrics <- sort(unique(injury_trace_data$metric))
injury_trace_injuries <- sort(unique(as.character(injury_trace_data$injury_type)))

injury_trace_plot <- injury_trace_data |>
  ggplot(aes(x = sample, y = value, colour = chain, group = chain)) +
  geom_line(alpha = 0.7, linewidth = 0.3) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.3, linetype = "dashed") +
  facet_grid(
    injury_type ~ metric,
    scales = "free_y",
    switch = "y"
  ) +
  scale_colour_brewer(palette = "Set1", name = "Chain") +
  theme_bw(base_size = 7) +
  theme(
    axis.text.x    = element_blank(),
    axis.ticks.x   = element_blank(),
    strip.text.x   = element_text(size = 6, angle = 0),
    strip.text.y   = element_text(size = 5, angle = 0),
    panel.spacing  = unit(0.15, "lines"),
    legend.position = "bottom"
  ) +
  labs(
    x     = "Sample",
    y     = "Posterior Prior Mean Effect",
    title = "MCMC Trace: Posterior Injury Prior Mean by Metric × Injury Type"
  )

ggsave(
  file.path(plots_dir, "causal", "injury_mean_posterior_trace.png"),
  injury_trace_plot,
  width  = length(injury_trace_metrics) * 1.6,
  height = length(injury_trace_injuries) * 0.9
)

injury_mean_for_pca <- injury_mean_posterior %>%
  mutate(sample = if ("sample" %in% names(.)) sample else samples) %>%
  filter(!is.na(chain), !is.na(sample), !is.na(metric), !is.na(injury_type), !is.na(value), is.finite(value))

pca_by_draw <- injury_mean_for_pca %>%
  group_by(chain, sample) %>%
  group_split() %>%
  map(function(df_draw) {
    mat_df <- df_draw %>%
      select(metric, injury_type, value) %>%
      distinct() %>%
      pivot_wider(names_from = injury_type, values_from = value)

    metric_names <- mat_df$metric
    value_mat <- mat_df %>% select(-metric) %>% as.matrix()

    if (nrow(value_mat) < 2 || ncol(value_mat) < 2) {
      return(NULL)
    }

    pca_fit <- prcomp(value_mat, center = TRUE, scale. = TRUE)
    if (ncol(pca_fit$x) < 2 || ncol(pca_fit$rotation) < 2) {
      return(NULL)
    }

    scores <- as_tibble(pca_fit$x[, 1:2, drop = FALSE]) %>%
      rename(PC1 = 1, PC2 = 2) %>%
      mutate(metric = metric_names)

    loadings <- as_tibble(pca_fit$rotation[, 1:2, drop = FALSE], rownames = "injury_type") %>%
      rename(PC1 = 2, PC2 = 3)

    list(
      chain = first(df_draw$chain),
      sample = first(df_draw$sample),
      scores = scores,
      loadings = loadings
    )
  }) %>%
  compact()

metric_score_draws <- pca_by_draw %>%
  map_df(~ .x$scores %>% mutate(chain = .x$chain, sample = .x$sample))

injury_loading_draws <- pca_by_draw %>%
  map_df(~ .x$loadings %>% mutate(chain = .x$chain, sample = .x$sample))

if (nrow(metric_score_draws) > 0 && nrow(injury_loading_draws) > 0) {
  first_draw <- metric_score_draws %>%
    distinct(chain, sample) %>%
    arrange(chain, sample) %>%
    slice(1)

  ref_scores <- metric_score_draws %>%
    semi_join(first_draw, by = c("chain", "sample")) %>%
    select(metric, ref_PC1 = PC1, ref_PC2 = PC2)

  metric_score_draws_aligned <- metric_score_draws %>%
    group_by(chain, sample) %>%
    group_modify(~ {
      d <- .x %>% left_join(ref_scores, by = "metric")
      s1 <- if_else(cor(d$PC1, d$ref_PC1, use = "complete.obs") < 0, -1, 1)
      s2 <- if_else(cor(d$PC2, d$ref_PC2, use = "complete.obs") < 0, -1, 1)
      tibble(metric = d$metric, PC1 = d$PC1 * s1, PC2 = d$PC2 * s2, sign1 = s1, sign2 = s2)
    }) %>%
    ungroup()

  draw_signs <- metric_score_draws_aligned %>%
    distinct(chain, sample, sign1, sign2)

  injury_loading_draws_aligned <- injury_loading_draws %>%
    left_join(draw_signs, by = c("chain", "sample")) %>%
    mutate(PC1 = PC1 * sign1, PC2 = PC2 * sign2)

  metric_centroids <- metric_score_draws_aligned %>%
    group_by(metric) %>%
    summarize(PC1 = median(PC1), PC2 = median(PC2), .groups = "drop")

  loading_centroids <- injury_loading_draws_aligned %>%
    group_by(injury_type) %>%
    summarize(PC1 = median(PC1), PC2 = median(PC2), .groups = "drop")

  metric_scores_plot <- ggplot(metric_score_draws_aligned , aes(x = PC1, y = PC2, color = metric)) +
    geom_point(alpha = 0.08, size = 0.7) +
    stat_ellipse(aes(group = metric), level = 0.8, linewidth = 0.5, alpha = 0.8) +
    geom_point(data = metric_centroids, size = 2.2) +
    geom_text_repel(data = metric_centroids, aes(label = metric), size = 3, show.legend = FALSE) +
    theme_bw() +
    labs(
      x = "PC1",
      y = "PC2",
      color = "Metric",
      title = "Posterior PCA of Injury Mean Effects: Metric Scores",
      # subtitle = "Point cloud over (chain, sample); ellipses summarize posterior uncertainty"
    )

  injury_loadings_plot <- ggplot(injury_loading_draws_aligned, aes(x = PC1, y = PC2, color = injury_type)) +
    geom_point(alpha = 0.08, size = 0.7) +
    stat_ellipse(aes(group = injury_type), level = 0.8, linewidth = 0.5, alpha = 0.8) +
    geom_point(data = loading_centroids, size = 2.2) +
    geom_text_repel(data = loading_centroids, aes(label = injury_type), size = 3, show.legend = FALSE) +
    theme_bw() +
    labs(
      x = "PC1",
      y = "PC2",
      color = "Injury Type",
      title = "Posterior PCA of Injury Mean Effects: Injury-Type Loadings",
      # subtitle = "Point cloud over (chain, sample); ellipses summarize posterior uncertainty"
    )

  ggsave(
    file.path(plots_dir, "causal", "injury_mean_pca_metric_scores.png"),
    metric_scores_plot,
    width = 11,
    height = 8
  )

  ggsave(
    file.path(plots_dir, "causal", "injury_mean_pca_injury_loadings.png"),
    injury_loadings_plot,
    width = 11,
    height = 8
  )
}


injury_effect_posterior <- read_parquet(
  file.path(model_dir, "posterior_injury_samples.parquet"),
  col_select = c("metric", "value", "age", "player", "injury_type", "sample", "chain", "injured")
)

injury_effect_time_player_means <- injury_effect_posterior %>%
  mutate(
    sample = if ("sample" %in% names(.)) sample else samples,
    value_link = value,
    effect_scale = case_when(
      metric %in% gaussian_metrics ~ "additive",
      metric %in% count_metrics ~ "rate_multiplier",
      metric %in% proportion_metrics ~ "odds_ratio",
      TRUE ~ "additive"
    ),
    value = case_when(
      metric %in% gaussian_metrics ~ value,
      metric %in% count_metrics ~ exp(value) - 1,
      metric %in% proportion_metrics ~ exp(value) - 1,
      TRUE ~ value
    ),
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES" ~ "GP%",
      metric == "FG2M" ~ "FG2%",
      metric == "FG3M" ~ "FG3%",
      metric == "FTM" ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    ),
    injury_type = as.factor(injury_type)
  ) %>%
  filter(
    !is.na(injury_type),
    !is.na(age),
    !is.na(player),
    !is.na(value),
    is.finite(value)
  ) %>%
  group_by(injury_type, metric, age, player) %>%
  summarize(posterior_mean = mean(value, na.rm = TRUE), .groups = "drop")

plot_injury_effect_over_time <- function(df_injury, injury_name) {
  ggplot(df_injury, aes(x = factor(age), y = posterior_mean)) +
    geom_violin(fill = "steelblue", alpha = 0.35, color = "steelblue", scale = "width") +
    stat_summary(fun = median, geom = "point", size = 0.8, color = "black") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    facet_wrap(~metric, scales = "free_y") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      x = "Age",
      y = "Posterior Mean Injury Effect Across Players (Negative = Decrease)",
      title = glue("Posterior Injury Effect Over Time: {injury_name}"),
      subtitle = "Density (violin) of player-level posterior means at each age"
    )
}

injury_effect_time_player_means %>%
  group_split(injury_type) %>%
  walk(function(df_injury) {
    injury_name <- as.character(first(df_injury$injury_type))
    safe_name <- str_replace_all(injury_name, "[^A-Za-z0-9]+", "_")

    ggsave(
      file.path(plots_dir, "causal", glue("injury_effect_over_time_{safe_name}.png")),
      plot_injury_effect_over_time(df_injury, injury_name),
      width = 14,
      height = 9
    )
  })

injury_age_anchor <- injury_effect_posterior %>%
  filter(injured == 1, !is.na(age), !is.na(player), !is.na(injury_type)) %>%
  group_by(player, injury_type) %>%
  summarize(injury_age = min(age, na.rm = TRUE), .groups = "drop")

injury_effect_years_since_player_means <- injury_effect_posterior %>%
  left_join(injury_age_anchor, by = c("player", "injury_type")) %>%
  mutate(
    sample = if ("sample" %in% names(.)) sample else samples,
    years_since_injury = age - injury_age,
    value_link = value,
    effect_scale = case_when(
      metric %in% gaussian_metrics ~ "additive",
      metric %in% count_metrics ~ "rate_multiplier",
      metric %in% proportion_metrics ~ "odds_ratio",
      TRUE ~ "additive"
    ),
    value = case_when(
      metric %in% gaussian_metrics ~ value,
      metric %in% count_metrics ~ exp(value) - 1,
      metric %in% proportion_metrics ~ exp(value) - 1,
      TRUE ~ value
    ),
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES" ~ "GP%",
      metric == "FG2M" ~ "FG2%",
      metric == "FG3M" ~ "FG3%",
      metric == "FTM" ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    ),
    injury_type = as.factor(injury_type)
  ) %>%
  filter(
    !is.na(injury_type),
    !is.na(years_since_injury),
    years_since_injury >= 0,
    !is.na(player),
    !is.na(value),
    is.finite(value)
  ) %>%
  group_by(injury_type, metric, years_since_injury, player) %>%
  summarize(posterior_mean = mean(value, na.rm = TRUE), .groups = "drop")

plot_injury_effect_since_injury <- function(df_injury, injury_name) {
  ggplot(df_injury, aes(x = factor(years_since_injury), y = posterior_mean)) +
    geom_violin(fill = "steelblue", alpha = 0.35, color = "steelblue", scale = "width") +
    stat_summary(fun = median, geom = "point", size = 0.8, color = "black") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    facet_wrap(~metric, scales = "free_y") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      x = "Years Since Injury",
      y = "Posterior Mean Injury Effect Across Players (Negative = Decrease)",
      title = glue("Posterior Injury Effect by Years Since Injury: {injury_name}"),
      subtitle = "Density (violin) of player-level posterior means by years since injury"
    )
}

injury_effect_years_since_player_means %>%
  group_split(injury_type) %>%
  walk(function(df_injury) {
    injury_name <- as.character(first(df_injury$injury_type))
    safe_name <- str_replace_all(injury_name, "[^A-Za-z0-9]+", "_")

    ggsave(
      file.path(plots_dir, "causal", glue("injury_effect_years_since_injury_{safe_name}.png")),
      plot_injury_effect_since_injury(df_injury, injury_name),
      width = 14,
      height = 9
    )
  })


### -----------------------------------------------------------------------
### Causal effect of injury type on career length (exit age)
### -----------------------------------------------------------------------

focal_injuries <- c("ACL", "Achilles", "Hip", "Back/Spine",
                    "Patellar Tendon", "Quad Tendon",
                    "Lower Body Fracture", "Foot Fracture", "Meniscus")

# injury type lookup (one row per player)
injury_type_by_player <- data |>
  group_by(id) |>
  summarize(
    first_major_injury = first(na.omit(first_major_injury)),
    name = first(name),
    .groups = "drop"
  )

n_per_injury_type <- injury_type_by_player |>
  filter(first_major_injury %in% focal_injuries) |>
  count(first_major_injury, name = "n_players")

# Injury age: first post-injury season age per player.
# We need this to correct for delayed-entry bias: a player observed as injured at
# age T_injury must have survived in the league until T_injury. Comparing total
# career length (from age 18) conflates the injury effect with the selection that
# kept the player in the league long enough to be injured.
# Correction: in the counterfactual, clip T_cf to max(T_cf, T_injury) so we
# evaluate the counterfactual only among "always-survivors to T_injury". The
# estimand becomes E[max(T_cf, T_injury) - T_obs], i.e. the effect on remaining
# career from the injury age onward.
injury_age_by_player <- data |>
  filter(!is.na(first_major_injury), injury_period == "post-injury") |>
  group_by(id) |>
  summarize(injury_age = min(age, na.rm = TRUE), .groups = "drop")

# ATT: for each player restrict to the row matching their actual injury type,
# then average over post-injury ages to get one draw per (chain, sample, metric, injury_type).
# This is the Average Treatment Effect on the Treated — the correct causal quantity.
# Observed player-seasons: restrict to ages where the player was actually in the league
observed_player_seasons <- data |>
  select(id, age) |>
  distinct()

injury_att_posterior <- injury_effect_posterior |>
  inner_join(
    injury_type_by_player |> select(id, first_major_injury),
    by = c("player" = "id")
  ) |>
  filter(
    injury_type == first_major_injury,
    first_major_injury %in% focal_injuries
  ) |>
  inner_join(injury_age_by_player, by = c("player" = "id")) |>
  filter(age >= injury_age) |>
  # Restrict to ages where the player was actually observed playing
  semi_join(observed_player_seasons, by = c("player" = "id", "age")) |>
  mutate(
    value_link = value,
    effect_scale = case_when(
      metric %in% gaussian_metrics   ~ "additive",
      metric %in% count_metrics      ~ "rate_multiplier",
      metric %in% proportion_metrics ~ "odds_ratio",
      TRUE                           ~ "additive"
    ),
    value = case_when(
      metric %in% gaussian_metrics   ~ value,
      metric %in% count_metrics      ~ exp(value) - 1,
      metric %in% proportion_metrics ~ exp(value) - 1,
      TRUE                           ~ value
    ),
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES"       ~ "GP%",
      metric == "FG2M"        ~ "FG2%",
      metric == "FG3M"        ~ "FG3%",
      metric == "FTM"         ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    )
  ) |>
  filter(!is.na(value), is.finite(value))

# One posterior draw per (chain, sample, metric, injury_type): mean over post-injury ages and players
injury_att_draws <- injury_att_posterior |>
  rename(any_of(c(sample = "samples"))) |>
  group_by(chain, sample, metric, injury_type) |>
  summarize(value = mean(value, na.rm = TRUE), .groups = "drop")

# exit_hazard is not in the samples CSV; pull it from the prior mean file (injury_mean_posterior)
# and append as an additional metric so it appears in the same plot.
exit_hazard_att_draws <- injury_mean_posterior |>
  filter(metric == "exit_hazard", injury_type %in% focal_injuries) |>
  rename(any_of(c(sample = "samples"))) |>
  select(chain, sample, injury_type, value) |>
  mutate(metric = "Exit Hazard")

injury_att_draws_all <- bind_rows(
  injury_att_draws |> filter(metric != "RETIREMENT"),
  exit_hazard_att_draws
)

injury_att_intervals <- injury_att_draws_all |>
  group_by(metric, injury_type) |>
  ggdist::median_qi(value, .width = 0.95) |>
  ungroup() |>
  mutate(
    interval_color = case_when(
      .lower > 0 ~ "Above 0",
      .upper < 0 ~ "Below 0",
      TRUE       ~ "Overlaps 0"
    )
  )

injury_att_plot <- injury_att_intervals |>
  ggplot(aes(x = injury_type, y = value, color = interval_color)) +
  geom_linerange(aes(ymin = .lower, ymax = .upper), linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  scale_y_continuous(limits = function(lims) {
    lim <- max(abs(lims), na.rm = TRUE)
    c(-lim, lim)
  }) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(
    values = c("Above 0" = "blue", "Below 0" = "red", "Overlaps 0" = "grey50")
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(
    x        = "Injury Type",
    y        = "ATT Posterior Effect",
    color    = "95% Interval",
    title    = "ATT: Posterior Injury Effect on Performance Metrics + Exit Hazard",
    subtitle = "Post-injury observed seasons only; each player matched to their actual injury type"
  )

ggsave(
  file.path(plots_dir, "causal", "injury_att_posterior_interval_by_metric.png"),
  injury_att_plot,
  width = 14,
  height = 9
)

# ── Latent-space ATT panel ─────────────────────────────────────────────────
# For each (injury_type × metric) show PC1/PC2 of the latent X space.
# Background = all players (grey). Foreground = players who actually had that
# injury type, colored by their posterior mean ATT effect on the metric.
# This reveals which regions of the latent X space drive differential injury
# responses — approximating E[X @ injury_player_x] after averaging out noise.

focal_metrics_latent <- c("GP%", "MPG", "OBPM", "DBPM", "AST", "BLK", "Exit Hazard")

# Player-level posterior mean ATT: average over MCMC draws and post-injury ages
injury_att_player_effect <- injury_att_posterior |>
  rename(any_of(c(sample = "samples"))) |>
  filter(metric %in% focal_metrics_latent) |>
  group_by(player, metric, injury_type) |>
  summarize(effect = mean(value, na.rm = TRUE), .groups = "drop")

# exit_hazard is only available at injury-type level (prior mean); compute
# per-injury-type means and give every player in that group the same value so
# it appears on the same panel grid
exit_hazard_player_effect <- injury_mean_posterior |>
  filter(metric == "exit_hazard", injury_type %in% focal_injuries) |>
  rename(any_of(c(sample = "samples"))) |>
  group_by(injury_type) |>
  summarize(effect = mean(value_link, na.rm = TRUE), .groups = "drop") |>
  inner_join(
    injury_type_by_player |>
      filter(first_major_injury %in% focal_injuries) |>
      select(id, first_major_injury),
    by = c("injury_type" = "first_major_injury")
  ) |>
  rename(player = id) |>
  mutate(metric = "Exit Hazard")

injury_att_player_all <- bind_rows(
  injury_att_player_effect,
  exit_hazard_player_effect
)

# Latent space coordinates: phi_X has one row per (chain, sample, player) so
# latent_injuries_pca inherits duplicates. Average PC1/PC2 across draws so
# each player has exactly one position before joining.
latent_coords <- latent_injuries_pca |>
  select(id, PC1, PC2) |>
  group_by(id) |>
  summarize(PC1 = mean(PC1, na.rm = TRUE), PC2 = mean(PC2, na.rm = TRUE), .groups = "drop") |>
  rename(player = id)

# Normalise effect within each metric: divide by SD so colour encodes
# standard deviations from the metric mean. Zero stays at zero (mid-colour);
# cross-metric comparison is meaningful because all panels share the same unit.
cat(sprintf(
  "latent space panel diagnostics:\n  injury_att_player_all: %d rows, players: %d unique\n  latent_coords: %d rows\n",
  nrow(injury_att_player_all),
  n_distinct(injury_att_player_all$player),
  nrow(latent_coords)
))
cat("  sample player IDs from att:", paste(head(unique(injury_att_player_all$player), 3), collapse=", "), "\n")
cat("  sample player IDs from latent:", paste(head(latent_coords$player, 3), collapse=", "), "\n")

att_latent_df <- injury_att_player_all |>
  inner_join(latent_coords, by = "player") |>
  group_by(metric) |>
  mutate(
    effect_sd     = sd(effect, na.rm = TRUE),
    effect_clamped = effect / pmax(effect_sd, 1e-8)
  ) |>
  ungroup() |>
  mutate(
    metric      = factor(metric, levels = focal_metrics_latent),
    injury_type = factor(injury_type, levels = sort(unique(injury_type)))
  )

cat(sprintf("  att_latent_df after join: %d rows, metrics: %s\n",
  nrow(att_latent_df),
  paste(unique(as.character(att_latent_df$metric)), collapse=", ")
))

# Background: all players in latent space
latent_bg <- latent_coords |>
  filter(!is.na(PC1), !is.na(PC2))

# Each metric gets its own colour scale; build one column-plot per metric and
# combine with patchwork so limits are independently fitted.
injury_att_latent_panel <- map(focal_metrics_latent, function(m) {
  df_m <- att_latent_df |> filter(metric == m)
  if (nrow(df_m) == 0) return(NULL)
  lim  <- max(abs(df_m$effect_clamped), na.rm = TRUE)
  if (!is.finite(lim) || lim < 1e-8) lim <- 1
  show_strip <- m == focal_metrics_latent[[1]]

  ggplot() +
    geom_point(
      data = latent_bg,
      aes(x = PC1, y = PC2),
      color = "grey82", size = 0.6, alpha = 0.5
    ) +
    geom_point(
      data = df_m,
      aes(x = PC1, y = PC2, color = effect_clamped),
      size = 1.8, alpha = 0.85
    ) +
    scale_color_gradient2(
      low      = "red",
      mid      = "grey92",
      high     = "blue",
      midpoint = 0,
      limits   = c(-lim, lim),
      name     = "SD units"
    ) +
    facet_wrap(~injury_type, ncol = 1, drop = FALSE,
               strip.position = if (show_strip) "left" else "right") +
    theme_bw(base_size = 9) +
    theme(
      axis.text        = element_blank(),
      axis.ticks       = element_blank(),
      axis.title       = element_blank(),
      panel.grid       = element_blank(),
      strip.text       = if (show_strip)
                           element_text(angle = 0, hjust = 1, size = 7)
                         else
                           element_blank(),
      strip.background = if (show_strip) element_rect() else element_blank(),
      legend.position  = "bottom",
      legend.key.width = unit(1.2, "cm"),
      plot.title       = element_text(size = 8, hjust = 0.5)
    ) +
    labs(title = m)
}) |>
  purrr::compact()

if (length(injury_att_latent_panel) == 0) {
  warning("injury_att_latent_panel: no plots were built — att_latent_df is empty or join failed")
} else {
  injury_att_latent_panel <- patchwork::wrap_plots(injury_att_latent_panel, nrow = 1)
  n_injury_rows <- max(length(levels(att_latent_df$injury_type)), 1L)
  ggsave(
    file.path(plots_dir, "causal", "injury_att_latent_space_panel.png"),
    injury_att_latent_panel,
    width  = length(focal_metrics_latent) * 2.2,
    height = n_injury_rows * 1.8
  )
}

# pivot scenarios wide so each row is one (chain, sample, player) draw.
# Right-censored players (exit_censored == 1) have a known lower bound on their
# exit age (observed_exit_age). The utility samples T unconditionally, so any
# draw below the censoring time is inconsistent with what we observed. We clip
# both scenarios to pmax(sampled, observed_exit_age) for censored players; this
# gives a conservative lower-bound estimate of career_years_lost for those players.
career_length_causal <- exit_age_data |>
  select(chain, sample, player, value, observed_entrance_age, observed_exit_age,
         exit_censored, scenario) |>
  pivot_wider(
    id_cols     = c(chain, sample, player, observed_entrance_age,
                    observed_exit_age, exit_censored),
    names_from  = scenario,
    values_from = value,
    names_prefix = "exit_age_"
  ) |>
  inner_join(injury_age_by_player, by = c("player" = "id")) |>
  mutate(
    # (1) right-censoring: enforce T >= last observed age
    exit_age_observed       = if_else(exit_censored == 1,
                                      pmax(exit_age_observed,      observed_exit_age),
                                      exit_age_observed),
    exit_age_counterfactual = if_else(exit_censored == 1,
                                      pmax(exit_age_counterfactual, observed_exit_age),
                                      exit_age_counterfactual),
    # (2) delayed-entry correction: counterfactual must also survive to T_injury
    #     (always-survivors principal stratum)
    exit_age_counterfactual = pmax(exit_age_counterfactual, injury_age),
    career_years_lost = exit_age_counterfactual - exit_age_observed,
    career_length_obs = exit_age_observed       - observed_entrance_age,
    career_length_cf  = exit_age_counterfactual - observed_entrance_age,
    is_censored       = exit_censored == 1
  ) |>
  inner_join(injury_type_by_player, by = c("player" = "id")) |>
  filter(first_major_injury %in% focal_injuries)

# fraction of censored players per injury type (for annotation)
censored_frac <- career_length_causal |>
  distinct(player, first_major_injury, is_censored) |>
  group_by(first_major_injury) |>
  summarize(pct_censored = round(100 * mean(is_censored)), .groups = "drop")

# ATT: average years of career lost per injury type
career_att_by_type <- career_length_causal |>
  group_by(first_major_injury, chain, sample) |>
  summarize(sample_att = mean(career_years_lost, na.rm = TRUE), .groups = "drop") |>
  group_by(first_major_injury) |>
  summarize(
    mean_att   = mean(sample_att),
    lower      = HDInterval::hdi(sample_att, credMass = 0.95)["lower"],
    upper      = HDInterval::hdi(sample_att, credMass = 0.95)["upper"],
    .groups    = "drop"
  ) |>
  left_join(n_per_injury_type, by = "first_major_injury") |>
  left_join(censored_frac,     by = "first_major_injury") |>
  mutate(
    first_major_injury = fct_reorder(first_major_injury, mean_att, .desc = TRUE),
    label = glue("N={n_players} ({pct_censored}% active)")
  )

career_length_att_plot <- ggplot(career_att_by_type,
                                 aes(x = first_major_injury, y = mean_att)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  geom_text(aes(label = label, y = upper + 0.15), size = 2.8, hjust = 0) +
  coord_flip(clip = "off") +
  labs(
    x = "Injury Type",
    y = "Career Years Lost (ATT, 95% CI)",
    title = "Causal Effect of Injury Type on Career Length",
    subtitle = paste0("Estimand: E[max(T_cf, T_injury) \u2212 T_obs | injury type] among always-survivors to T_injury\n",
                      "Corrects for delayed-entry bias; right-censored players further clipped to last observed season")
  ) +
  theme_bw() +
  theme(plot.margin = margin(r = 80))

ggsave(file.path(plots_dir, "causal", "career_length_att_by_injury_type.png"),
       career_length_att_plot, width = 10, height = 6)

# per-player posterior distribution of career years lost;
# triangles = right-censored players (career_years_lost is a lower bound for them)
career_player_dist <- career_length_causal |>
  group_by(first_major_injury, player, name, is_censored) |>
  summarize(
    mean_years_lost = mean(career_years_lost, na.rm = TRUE),
    lower           = HDInterval::hdi(career_years_lost, credMass = 0.95)["lower"],
    upper           = HDInterval::hdi(career_years_lost, credMass = 0.95)["upper"],
    .groups         = "drop"
  ) |>
  mutate(first_major_injury = factor(first_major_injury,
                                     levels = levels(career_att_by_type$first_major_injury)))

career_length_player_plot <- ggplot(career_player_dist,
                                    aes(x = first_major_injury, y = mean_years_lost)) +
  geom_jitter(aes(color = first_major_injury,
                  shape = is_censored),
              width = 0.2, alpha = 0.7, size = 2) +
  geom_boxplot(outlier.shape = NA, width = 0.4, fill = NA) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_brewer(palette = "Set1") +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17),
                     labels = c("FALSE" = "Observed exit", "TRUE" = "Right-censored (\u2265 lower bound)")) +
  coord_flip() +
  guides(color = "none") +
  labs(
    x = "Injury Type",
    y = "Career Years Lost (posterior mean per player)",
    shape = NULL,
    title = "Distribution of Career Years Lost by Injury Type",
    subtitle = "Triangles = still-active players; their estimate is a conservative lower bound"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(plots_dir, "causal", "career_length_distribution_by_injury_type.png"),
       career_length_player_plot, width = 9, height = 6)

# survival curves by injury type: observed vs counterfactual
exit_survival_data <- read_parquet(
  file.path(model_dir, "posterior_exit_survival.parquet"),
  col_select = c("player", "scenario", "value", "age", "chain", "sample")
)

# median injury age per type — used to condition the survival curves
median_injury_age <- injury_age_by_player |>
  inner_join(injury_type_by_player |> select(id, first_major_injury), by = "id") |>
  filter(first_major_injury %in% focal_injuries) |>
  group_by(first_major_injury) |>
  summarize(median_injury_age = median(injury_age, na.rm = TRUE), .groups = "drop")

survival_by_injury_type <- exit_survival_data |>
  inner_join(injury_type_by_player, by = c("player" = "id")) |>
  filter(first_major_injury %in% focal_injuries) |>
  group_by(first_major_injury, scenario, age, chain, sample) |>
  summarize(mean_survival = mean(value, na.rm = TRUE), .groups = "drop") |>
  group_by(first_major_injury, scenario, age) |>
  summarize(
    posterior_mean = mean(mean_survival, na.rm = TRUE),
    lower          = HDInterval::hdi(mean_survival, credMass = 0.95)["lower"],
    upper          = HDInterval::hdi(mean_survival, credMass = 0.95)["upper"],
    .groups        = "drop"
  ) |>
  mutate(first_major_injury = factor(first_major_injury, levels = levels(career_att_by_type$first_major_injury)))

# Conditional survival: S(t | T > T_injury) = S(t) / S(T_injury).
# Before T_injury observed = counterfactual, so the unconditional curves overlap
# for most of the x-axis, diluting the visible gap. Conditioning rescales both
# curves to start at 1.0 at the injury age, making the post-injury divergence clear.
survival_conditional <- survival_by_injury_type |>
  inner_join(median_injury_age, by = "first_major_injury") |>
  filter(age >= floor(median_injury_age)) |>
  group_by(first_major_injury, scenario) |>
  mutate(
    s0             = posterior_mean[which.min(age)],
    posterior_mean = posterior_mean / s0,
    lower          = lower          / s0,
    upper          = upper          / s0
  ) |>
  ungroup()

survival_curve_plot <- ggplot(survival_conditional,
                              aes(x = age, y = posterior_mean,
                                  color = scenario, fill = scenario)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = c("observed" = "steelblue", "counterfactual" = "tomato"),
                     labels = c("observed" = "Observed (injured)", "counterfactual" = "Counterfactual (no injury)")) +
  scale_fill_manual(values  = c("observed" = "steelblue", "counterfactual" = "tomato"),
                    labels  = c("observed" = "Observed (injured)", "counterfactual" = "Counterfactual (no injury)")) +
  facet_wrap(~first_major_injury, nrow = 3) +
  labs(
    x = "Age", y = "Conditional survival P(active at age t | active at injury age)",
    color = NULL, fill = NULL,
    title = "Exit Survival Curves by Injury Type (conditional on surviving to injury age)",
    subtitle = "Curves rescaled to S=1 at median injury age — pre-injury overlap removed; shaded band = 95% CI"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(plots_dir, "causal", "exit_survival_curves_by_injury_type.png"),
       survival_curve_plot, width = 14, height = 10)


### -----------------------------------------------------------------------
### Identification test: is injury type predictable from X and/or volume?
### -----------------------------------------------------------------------
# Two stages, each run with two predictor sets to isolate the role of volume:
#   (a) X only  — rate-based latent style factors
#   (b) X + volume — adds cumulative minutes played, which captures load
#
# The key comparison is stage 2 (type | injured):
#   - Low lift from X alone is EXPECTED (rate metrics don't contain volume)
#   - If X+volume also shows low lift → type is close to random given load+style
#     → current model assumption is reasonable
#   - If X+volume shows substantial lift over X alone → volume is the missing
#     confounder → time-to-injury model with volume covariate is needed

library(nnet)

dim_cols <- grep("^Dim", names(latent_space), value = TRUE)

# ---- Volume features ----
# Stage 1: avg minutes/season + career seasons (rate × exposure, avoids
#   endogeneity from total minutes being correlated with career length)
# Stage 2: cumulative minutes *before* injury (the accumulated load at injury time)
volume_career <- min_data |>
  group_by(id) |>
  summarize(
    avg_min_per_season  = mean(normalized_min_played,  na.rm = TRUE),
    avg_gp_per_season   = mean(normalized_games_played, na.rm = TRUE),
    career_seasons      = n(),
    .groups = "drop"
  )

cum_vol_at_injury <- min_data |>
  inner_join(injury_age_by_player, by = "id") |>
  filter(age < injury_age) |>
  group_by(id) |>
  summarize(
    cum_min_before_injury = sum(normalized_min_played,  na.rm = TRUE),
    cum_gp_before_injury  = sum(normalized_games_played, na.rm = TRUE),
    seasons_before_injury = n(),
    .groups = "drop"
  )

vol_inc_cols  <- c("avg_min_per_season", "avg_gp_per_season", "career_seasons")
vol_type_cols <- c("cum_min_before_injury", "cum_gp_before_injury", "seasons_before_injury")

id_injury_base <- latent_space |>
  left_join(injury_type_by_player, by = "id") |>
  left_join(volume_career, by = "id") |>
  mutate(injured = !is.na(first_major_injury))

id_injury_x  <- id_injury_base |> drop_na(all_of(dim_cols))
id_injury_xv <- id_injury_base |> drop_na(all_of(c(dim_cols, vol_inc_cols)))
# use the X-only frame as the canonical `id_injury` for PCA / plots
id_injury <- id_injury_x

injured_base <- id_injury_x |>
  filter(injured, first_major_injury %in% focal_injuries) |>
  left_join(cum_vol_at_injury, by = "id") |>
  mutate(injury_type = factor(first_major_injury))

injured_pred_df_x  <- injured_base |> drop_na(all_of(dim_cols))
injured_pred_df_xv <- injured_base |> drop_na(all_of(c(dim_cols, vol_type_cols)))
injured_pred_df    <- injured_pred_df_xv  # used downstream for plots

# ---- Fit models: X only and X + volume ----
null_incidence   <- glm(injured ~ 1,                                          data = id_injury_x,  family = binomial)
fit_inc_x        <- glm(reformulate(dim_cols,                   "injured"),   data = id_injury_x,  family = binomial)
fit_inc_xv       <- glm(reformulate(c(dim_cols, vol_inc_cols),  "injured"),   data = id_injury_xv, family = binomial)

null_type        <- nnet::multinom(injury_type ~ 1,                                                        data = injured_pred_df_x,  trace = FALSE)
fit_type_x       <- nnet::multinom(reformulate(dim_cols,                   "injury_type"),                 data = injured_pred_df_x,  trace = FALSE, MaxNWts = 10000)
fit_type_xv      <- nnet::multinom(reformulate(c(dim_cols, vol_type_cols), "injury_type"),                 data = injured_pred_df_xv, trace = FALSE, MaxNWts = 10000)

mcfadden <- function(fit, null) round(1 - fit$deviance / null$deviance, 3)

# ---- k-fold CV (k = 5) for all four models ----
set.seed(42)
k <- 5

cv_glm <- function(df, preds, outcome = "injured", folds) {
  map_dbl(1:k, function(i) {
    tr <- df[folds != i, ]; te <- df[folds == i, ]
    fit <- glm(reformulate(preds, outcome), data = tr, family = binomial)
    mean((predict(fit, te, type = "response") > 0.5) == te[[outcome]])
  }) |> mean()
}

cv_mn <- function(df, preds, folds) {
  map_dbl(1:k, function(i) {
    tr <- df[folds != i, ]; te <- df[folds == i, ]
    fit <- nnet::multinom(reformulate(preds, "injury_type"), data = tr, trace = FALSE, MaxNWts = 10000)
    mean(predict(fit, te) == te$injury_type)
  }) |> mean()
}

folds_inc_x  <- sample(rep(1:k, length.out = nrow(id_injury_x)))
folds_inc_xv <- sample(rep(1:k, length.out = nrow(id_injury_xv)))
folds_type_x  <- sample(rep(1:k, length.out = nrow(injured_pred_df_x)))
folds_type_xv <- sample(rep(1:k, length.out = nrow(injured_pred_df_xv)))

null_acc_inc  <- max(mean(id_injury_x$injured), mean(!id_injury_x$injured))
null_acc_type <- max(prop.table(table(injured_pred_df_x$injury_type)))

cv_results <- tibble(
  stage    = rep(c("1 — incidence (any vs none)", "2 — type (given injured)"), each = 2),
  features = rep(c("X only", "X + volume"), 2),
  mcfadden_r2 = c(
    mcfadden(fit_inc_x,   null_incidence),
    mcfadden(fit_inc_xv,  null_incidence),
    mcfadden(fit_type_x,  null_type),
    mcfadden(fit_type_xv, null_type)
  ),
  cv_accuracy = c(
    cv_glm(id_injury_x,       dim_cols,                   folds = folds_inc_x),
    cv_glm(id_injury_xv,      c(dim_cols, vol_inc_cols),  folds = folds_inc_xv),
    cv_mn(injured_pred_df_x,  dim_cols,                   folds = folds_type_x),
    cv_mn(injured_pred_df_xv, c(dim_cols, vol_type_cols), folds = folds_type_xv)
  ),
  null_accuracy = c(null_acc_inc, null_acc_inc, null_acc_type, null_acc_type)
) |> mutate(cv_lift = round(cv_accuracy - null_accuracy, 3),
            cv_accuracy = round(cv_accuracy, 3))

print(cv_results)

# ---- Plot: CV lift comparison X vs X+volume ----
lift_comparison_plot <- cv_results |>
  ggplot(aes(x = features, y = cv_lift, fill = features)) +
  geom_col(width = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~stage, scales = "free_y") +
  scale_fill_manual(values = c("X only" = "steelblue", "X + volume" = "tomato")) +
  guides(fill = "none") +
  labs(
    x = NULL, y = "CV accuracy lift over null (base rates)",
    title = "Does volume accumulation explain injury selection beyond latent style (X)?",
    subtitle = paste0(
      "Stage 2 is the identification-relevant comparison.\n",
      "Large lift from adding volume \u2192 cumulative load is a missing confounder in the current model."
    )
  ) +
  theme_bw()

ggsave(file.path(plots_dir, "causal", "identification_test_lift_comparison.png"),
       lift_comparison_plot, width = 8, height = 5)

# ---- Plot: cumulative minutes before injury by type ----
# Direct visualization of the volume-type relationship
vol_by_type_plot <- injured_pred_df |>
  ggplot(aes(x = fct_reorder(injury_type, cum_min_before_injury, .fun = median),
             y = cum_min_before_injury)) +
  geom_violin(fill = "steelblue", alpha = 0.35, color = "steelblue", scale = "width") +
  stat_summary(fun = median, geom = "point", size = 2) +
  stat_summary(fun.min = \(x) HDInterval::hdi(x, credMass = 0.8)["lower"],
               fun.max = \(x) HDInterval::hdi(x, credMass = 0.8)["upper"],
               geom = "linerange", linewidth = 0.8) +
  coord_flip() +
  labs(
    x = NULL, y = "Cumulative minutes played before injury",
    title = "Volume accumulation at time of injury, by injury type",
    subtitle = "Spread across types indicates volume is a confounder for injury-type selection"
  ) +
  theme_bw()

ggsave(file.path(plots_dir, "causal", "identification_test_volume_by_type.png"),
       vol_by_type_plot, width = 8, height = 6)

# ---- Visual: PCA coloured by injury type, sized by volume ----
pca_id <- latent_space |>
  select(all_of(dim_cols)) |>
  prcomp(center = TRUE, scale. = TRUE)

pct_var <- round(100 * summary(pca_id)$importance[2, 1:2], 1)

pca_id_df <- as_tibble(pca_id$x[, 1:2], .name_repair = "minimal") |>
  setNames(c("PC1", "PC2")) |>
  bind_cols(latent_space |> select(id, name, position_group)) |>
  left_join(injury_type_by_player |> select(id, first_major_injury), by = "id") |>
  left_join(volume_career, by = "id") |>
  mutate(
    injury_label = case_when(
      is.na(first_major_injury)              ~ "No injury",
      first_major_injury %in% focal_injuries ~ first_major_injury,
      TRUE                                   ~ "Other injury"
    ),
    is_focal = first_major_injury %in% focal_injuries & !is.na(first_major_injury)
  )

stage2_lift_x  <- cv_results |> filter(stage == "2 — type (given injured)", features == "X only")  |> pull(cv_lift)
stage2_lift_xv <- cv_results |> filter(stage == "2 — type (given injured)", features == "X + volume") |> pull(cv_lift)

label_players <- c(
  "Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant",
  "Dwight Howard", "Nikola Jokic", "Shaquille O'Neal", "Chris Paul",
  "Derrick Rose", "Giannis Antetokounmpo", "Tim Duncan", "Dirk Nowitzki",
  "Allen Iverson", "Russell Westbrook", "James Harden", "Rudy Gobert",
  "Kevin Garnett", "Dwyane Wade", "Carmelo Anthony", "Karl-Anthony Towns"
)

# ---- panel 1: injury type (colour) + volume (size), no position ----
pca_panel1 <- ggplot() +
  geom_point(
    data = filter(pca_id_df, !is_focal),
    aes(x = PC1, y = PC2),
    color = "grey75", size = 0.9, alpha = 0.35
  ) +
  geom_point(
    data = filter(pca_id_df, is_focal),
    aes(x = PC1, y = PC2, fill = injury_label, size = avg_min_per_season),
    shape = 21, color = "white", stroke = 0.3, alpha = 0.9
  ) +
  scale_fill_brewer(palette = "Set1", name = "Injury type") +
  scale_size_continuous(range = c(1.5, 5), name = "Avg min/season") +
  labs(
    x     = glue("PC1 ({pct_var[1]}% var)"),
    y     = glue("PC2 ({pct_var[2]}% var)"),
    title = "Injury type & volume"
  ) +
  theme_bw() +
  guides(fill = guide_legend(override.aes = list(shape = 21, size = 3, color = "white")))

# ---- panel 2: DBSCAN clusters + representative labels ----
library(dbscan)

pca_complete <- pca_id_df |> filter(!is.na(PC1), !is.na(PC2))
pc_scaled    <- scale(pca_complete |> select(PC1, PC2))
# eps ~ 0.35 captures the within-cluster density while spanning the gap between
# the two visible clusters; minPts = 10 avoids labelling sparse edge points as noise.
db <- dbscan(pc_scaled, eps = 0.35, minPts = 10)
# Noise points (cluster 0) are assigned to the nearest core-point cluster.
cluster_raw <- db$cluster
if (any(cluster_raw == 0)) {
  core_idx  <- which(cluster_raw != 0)
  noise_idx <- which(cluster_raw == 0)
  nn        <- kNN(pc_scaled[core_idx, ], k = 1, query = pc_scaled[noise_idx, , drop = FALSE])
  cluster_raw[noise_idx] <- cluster_raw[core_idx][nn$id[, 1]]
}
pca_complete <- pca_complete |> mutate(db_cluster = as.character(cluster_raw))
pca_id_df <- pca_id_df |>
  select(-any_of("db_cluster")) |>
  left_join(pca_complete |> select(id, db_cluster), by = "id")

# ---- Qualitative cluster assessment ----
cat("\n=== DBSCAN cluster composition ===\n")
cluster_summary <- pca_id_df |>
  filter(!is.na(db_cluster)) |>
  group_by(db_cluster) |>
  summarize(
    n               = n(),
    pct_guard       = round(100 * mean(position_group == "G", na.rm = TRUE)),
    pct_forward     = round(100 * mean(position_group == "F", na.rm = TRUE)),
    pct_center      = round(100 * mean(position_group == "C", na.rm = TRUE)),
    med_PC1         = round(median(PC1), 2),
    med_PC2         = round(median(PC2), 2),
    med_min         = round(median(avg_min_per_season, na.rm = TRUE)),
    .groups = "drop"
  )
print(cluster_summary)

cat("\nTop 10 players by minutes per cluster:\n")
pca_id_df |>
  filter(!is.na(db_cluster), !is.na(avg_min_per_season)) |>
  group_by(db_cluster) |>
  slice_max(avg_min_per_season, n = 10) |>
  select(db_cluster, name, position_group, avg_min_per_season) |>
  print(n = 40)

cat("\nInjury type distribution by cluster (among focal-injury players):\n")
injury_by_cluster <- pca_id_df |>
  filter(!is.na(db_cluster), first_major_injury %in% focal_injuries) |>
  count(db_cluster, first_major_injury) |>
  group_by(db_cluster) |>
  mutate(pct = round(100 * n / sum(n), 1)) |>
  ungroup()
print(injury_by_cluster, n = 40)

# # Test: is injury type independent of cluster?
# # Chi-square approximation is unreliable with sparse cells (cluster 1 has only
# # ~330 players split across 9 injury types). Use Monte Carlo Fisher exact test.
# injury_cluster_tab <- pca_id_df |>
#   filter(!is.na(db_cluster), first_major_injury %in% focal_injuries) |>
#   with(table(db_cluster, first_major_injury))
# cat("\nExpected cell counts (flag cells < 5):\n")
# print(round(chisq.test(injury_cluster_tab)$expected, 1))
# cat("\nFisher exact test (Monte Carlo, B=10000) — injury type ~ cluster:\n")
# print(fisher.test(injury_cluster_tab, simulate.p.value = TRUE, B = 10000))

# # sample 15 players per cluster, weighted by avg minutes per season
# set.seed(42)
# cluster_reps <- pca_id_df |>
#   filter(!is.na(db_cluster), !is.na(avg_min_per_season)) |>
#   group_by(db_cluster) |>
#   slice_sample(n = 15, weight_by = avg_min_per_season) |>
#   ungroup()

# pca_panel2 <- ggplot(
#     pca_id_df |> filter(!is.na(db_cluster)),
#     aes(x = PC1, y = PC2, color = db_cluster)
#   ) +
#   geom_point(size = 1, alpha = 0.45) +
#   geom_point(
#     data = cluster_reps,
#     aes(x = PC1, y = PC2),
#     shape = 21, color = "black", fill = NA, size = 3.5, stroke = 1.1,
#     inherit.aes = FALSE
#   ) +
#   geom_text_repel(
#     data = cluster_reps,
#     aes(label = name),
#     color = "black",
#     size = 2.2, max.overlaps = 30, box.padding = 0.3, show.legend = FALSE
#   ) +
#   scale_color_brewer(palette = "Set2", name = "Cluster") +
#   labs(
#     x     = glue("PC1 ({pct_var[1]}% var)"),
#     y     = glue("PC2 ({pct_var[2]}% var)"),
#     title = "Latent space clusters (DBSCAN)"
#   ) +
#   theme_bw()

# pca_combined <- pca_panel1 + pca_panel2 +
#   plot_annotation(
#     title    = "Latent space (X): injury type vs. latent clusters",
#     subtitle = glue("Stage-2 CV lift — X only: {stage2_lift_x};  X + volume: {stage2_lift_xv}")
#   )

# ggsave(file.path(plots_dir, "causal", "identification_test_pca.png"),
#        pca_combined, width = 16, height = 7)





