library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
library(purrr)
library(glue)
library(uwot)
library(ggrepel)
library(ggnewscale)
library(ggdist)


injury_data <- read.csv("data/injury_player_cleaned.csv") |> 
    mutate( pct_games = games / pmax(games, total_games, na.rm = TRUE),
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
            fg3_pct =  (fg3m / fg3a)) |>
    group_by(id) |>
    filter(!any(is.na(first_major_injury))) |> 
    ungroup() |> 
    select(name, id, obpm, dbpm, pct_games, mpg, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period, year) |>
    rename(pct_minutes = mpg, games = pct_games, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov,
             oreb, dreb, stl, fg3a, fg2a,
             fta, ftm, fg2m, fg3m),
    names_to = "metric",
    values_to = "obs_value")
print("pivoted the original data")

causal_empirical <- ggplot(injury_data |> group_by(id, injury_period, metric, first_major_injury) |> summarize(max_val = mean(obs_value)) |> ungroup() |> pivot_wider(
    names_from = injury_period,   # the values in this column become column names
    values_from = max_val           # the values to fill in those new columns
) |> mutate(empirical_change = `pre-injury` - `post-injury`) |>  mutate(metric = toupper(metric),
                              metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric)) |> filter(first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Meniscus")), 
                              aes(x = first_major_injury, y = empirical_change)) + 
                              geom_boxplot(outlier.shape = NA, width = 0.6) + facet_wrap(~metric, scales = "free_y") + 
                              scale_fill_brewer(palette = "Set1") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + 
  labs(x = "Injury Type", y = "Empirical Average Treatment Effect (ATT)", title = "Empirical Distribution of Average Treatment Effect by Metric")

ggsave("model_output/model_plots/causal/empirical_att_causal_plot.png", causal_empirical)

posterior_data <- read.csv("posterior_ar_injury.csv")
print("loaded the posterior data")
posterior_peaks <- read.csv("posterior_peaks_ar_injury.csv")
latent_space <- read.csv("latent_space_injury.csv")

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
print("joined the data with predictions")

att_plot <- joined_data |> filter(injury_period == "post-injury")  |> 
mutate(obs_value = if_else(year <= 2026 & metric %in% c("games") & is.na(obs_value), 0, obs_value),
        value = if_else(is.finite(value), value, NA_real_)) |>
filter(!is.na(obs_value) & is.finite(obs_value)) |> 
group_by(player, metric, chain, sample) |> 
mutate(injury_type = if_else(min(age) > peak_age, "post-peak", "pre-peak")) |> ungroup() |>
filter(first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Meniscus")) |>
mutate(injury_change =  obs_value - value) |> ungroup() |> 
group_by(first_major_injury, metric, injury_type, chain, sample) |> 
summarize(sample_ate = mean(injury_change, na.rm = TRUE)) |> ungroup() |> group_by(first_major_injury, metric, injury_type) |>
    summarize(mean_ate = mean(sample_ate), lower = HDInterval::hdi(sample_ate, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(sample_ate, credMass = 0.95)["upper"]) |>
    ggplot(aes(x = first_major_injury, y = mean_ate, color = injury_type)) + 
    geom_point( size = 3)  + scale_colour_brewer(palette = "Set1") + 
    geom_errorbar(aes(ymin = lower, ymax = upper), width = .2, ) + 
    facet_wrap(~metric, scales = "free_y") + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Average Treatment Effect for Treated (ATT) with 95% CI", x = "Injury", color = "Injury Time Period") + 
    ggtitle("Average Treatment Effect for Treated (ATT) per Injury Type, by Metric") 
ggsave("model_output/model_plots/causal/att_causal_plot.png", att_plot)

att_plot_total <- joined_data |> filter(injury_period == "post-injury")  |> 
mutate(obs_value = if_else(year <= 2026 & metric %in% c("games") & is.na(obs_value), 0, obs_value),
        value = if_else(is.finite(value), value, NA_real_)) |>
filter(!is.na(obs_value) & is.finite(obs_value)) |> 
group_by(player, metric, chain, sample) |> 
mutate(injury_type = if_else(min(age) > peak_age, "post-peak", "pre-peak")) |> ungroup() |>
filter(first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Meniscus")) |>
mutate(injury_change =  obs_value - value) |> ungroup() |> 
group_by(metric, injury_type, chain, sample) |> 
summarize(sample_ate = mean(injury_change, na.rm = TRUE)) |> ungroup() |> group_by(metric, injury_type) |>
    summarize(mean_ate = mean(sample_ate), lower = HDInterval::hdi(sample_ate, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(sample_ate, credMass = 0.95)["upper"]) |>
    ggplot(aes(x = injury_type, y = mean_ate, color = injury_type)) + 
    geom_point( size = 3)  + scale_colour_brewer(palette = "Set1") + 
    geom_errorbar(aes(ymin = lower, ymax = upper), width = .2, ) + 
    facet_wrap(~metric, scales = "free_y") + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Average Treatment Effect for Treated (ATT) with 95% CI", x = "Injury Time Period", color = "Injury Time Period") + 
    ggtitle("Average Treatment Effect for Treated (ATT) by Metric") 
ggsave("model_output/model_plots/causal/att_causal_plot_total.png", att_plot_total)

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
mutate(injury_change =  obs_value - value) |> ungroup() |> group_by(metric, player, first_major_injury, injury_type) |> 
summarize(mean_itt = mean(injury_change, na.rm = TRUE)) |> ungroup()



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
    data = filter(grouped_data_set, first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Meniscus") & injury_type == "pre-peak"),
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
      y = y_max * 0.9,
      label = case_when(metric == "GP%" ~ glue("Observed Total: {round(total_obs_value)}"),
                        .default = glue("Observed: {round(total_obs_value, 2)}"))
    )
  plt <- ggplot(raw_df, aes(x = counterfactual_value)) + 
    geom_density() + 
    geom_vline(data = obs_df,
             aes(xintercept = total_obs_value),
             color = "red", size = 1, linetype = "dashed") +
    geom_text(data = label_df,
            aes(x = total_obs_value, y = y, label = label),
            inherit.aes = FALSE, vjust = -0.5, color = "black", angle = 270) +
     scale_x_continuous(limits = c(0, NA)) + 
    facet_wrap(~metric, scale = "free") +
    labs(x = "Metric Measurables Post Injury", y = "Posterior Predictive Density") + 
    ggtitle(glue("Counterfactual Metric Values Post {injury_type} Injury: {group_name}")) + theme_bw() 

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
                                       fill = "blue",
                                       alpha = 0.2) +
    geom_line(aes(x = age, y = posterior_mean)) +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_vline(aes(xintercept = age_of_injury),
               linetype = "dashed",
               color = "red") +
    geom_text(
      data = injury_label, 
      size = 2,
      aes(x = age_of_injury, y = max_upper, label = first_major_injury),
      vjust = -1,
      color = "black"
    )  +
    facet_wrap( ~ metric, scales = "free_y") +
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Counterfactual Career Trajectory: ", group_name)) + theme_bw()
  return(plt)
}


plots_list <- joined_data %>% 
  mutate(value = if_else(is.finite(value), value, NA_real_)) %>% 
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_counterfactual(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/causal/{name}.png"),
      plot = plt,
    )
    })

plots_list <- joined_data %>%
filter(injury_period == "post-injury" & year <= 2026 & metric %in% c("GP%", "FTA")) %>%
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_counterfactual_metrics(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/causal/{name}_metrics_itt.png"),
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
      filename = glue("model_output/model_plots/causal/{name}_itt.png"),
      plot = plt,
    )
    })