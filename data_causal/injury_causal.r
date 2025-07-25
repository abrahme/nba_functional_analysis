library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
library(purrr)
library(glue)


injury_data <- read.csv("data/injury_player_cleaned.csv") |> 
    mutate( pct_games = games / max(games, 82),
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
posterior_data <- read.csv("posterior_injury_ar.csv")
print("loaded the posterior data")
posterior_peaks <- read.csv("posterior_peaks_injury_ar.csv")
posterior_data <- posterior_data |> mutate(value = case_when(metric == "pct_minutes" ~ value / 82, 
                                                             metric == "games" ~ value / 82,
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
                select(-base_age, -base_year) |> ungroup()
print("joined the data with predictions")

att_plot <- joined_data |> filter(injury_period == "post-injury")  |> group_by(player, metric, chain, sample) |> 
mutate(injury_type = if_else(min(age) > peak_age, "post-peak", "pre-peak")) |> ungroup() |>
mutate(obs_value = if_else(year <= 2026 & !metric %in% c("obpm","dbpm") & is.na(obs_value), 0, obs_value)) |>
filter(!is.na(obs_value)) |> 
filter(first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon", "Quad Tendon", "Lower Body Fracture", "Meniscus")) |>
mutate(injury_change =  obs_value - value) |> ungroup() |> 
group_by(first_major_injury, metric, injury_type) |> 
summarize(mean_ate = mean(injury_change), lower = hdi(injury_change, credMass = 0.95)["lower"],
    upper = hdi(injury_change, credMass = 0.95)["upper"]) |> ungroup() |> 
    ggplot(aes(x = first_major_injury, y = mean_ate, color = injury_type)) + 
    geom_point( size = 3)  + scale_colour_brewer(palette = "Set1") + 
    geom_errorbar(aes(ymin = lower, ymax = upper), width = .2, ) + 
    facet_wrap(~metric, scales = "free_y") + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Average Treatment Effect for Treated (ATT) with 95% CI", x = "Injury") + 
    ggtitle("Average Treatment Effect for Treated (ATT) per Injury Type, by Metric") 
ggsave("model_output/model_plots/causal/att_causal_plot.png", att_plot)


plot_counterfactual <- function(grouped_data_set) {

  group_name <- unique(grouped_data_set$name)
  raw_plt <-
  grouped_data_set |> mutate(
    age_of_injury = if_else(injury_period == "post-injury", age, Inf),
    age_of_injury = min(age_of_injury)
  ) 
  
  injury_label <- raw_plt |> group_by(metric) |> 
    summarise(
      upper = hdi(value, credMass = 0.95)["upper"],
      max_upper = max(upper),
      age_of_injury = first(age_of_injury),
      first_major_injury = first(first_major_injury)
    )
  
  plt <- raw_plt |>
    group_by(metric, age) |> summarize(
      posterior_mean = mean(value),
      upper = hdi(value, credMass = 0.95)["upper"],
      #max_upper = max(upper),
      lower = hdi(value, credMass = 0.95)["lower"],
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
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Counterfactual Career Trajectory: ", group_name))
  return(plt)
}


plots_list <- joined_data %>%
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