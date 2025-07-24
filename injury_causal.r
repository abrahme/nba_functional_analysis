library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
injury_data <- read.csv("data/injury_player_cleaned.csv") |> 
    mutate(
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
    select(name, id, obpm, dbpm, games, minutes, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period) |>
    rename(pct_minutes = minutes, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov,
             oreb, dreb, stl, fg3a, fg2a,
             fta, ftm, fg2m, fg3m),
    names_to = "metric",
    values_to = "obs_value")
print("pivoted the original data")
posterior_data <- read.csv("posterior_injury.csv")
print("loaded the posterior data")
posterior_peaks <- read.csv("posterior_peaks_injury.csv")
posterior_data <- posterior_data |> inner_join(posterior_peaks |> rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
joined_data <- posterior_data |> 
                left_join(injury_data, by = c("player" = "id", "metric", "age")) |> 
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, first_major_injury, .direction  = "downup") |> 
                fill(injury_period, .direction = "up") |> ungroup() |> filter(injury_period == "post-injury") |> group_by(player, metric, chain, sample) |> mutate(injury_type = if_else(min(age) > peak_age, "post-peak", "pre-peak")) |> ungroup()
print("joined the data with predictions")

att_plot <- joined_data |> filter(!is.na(obs_value)) |> 
group_by(player,first_major_injury,metric,chain, sample, injury_type) |> 
summarize(injury_change = mean((obs_value - value), na.rm = TRUE)) |> ungroup() |> 
group_by(first_major_injury, metric, injury_type) |> 
summarize(mean_ate = mean(injury_change, na.rm = TRUE), lower = hdi(injury_change, credMass = 0.95)["lower"],
    upper = hdi(injury_change, credMass = 0.95)["upper"]) |> ungroup() |> 
    ggplot(aes(x = first_major_injury, y = mean_ate, color = injury_type)) + 
    geom_point(position = position_dodge(width = 0.5), size = 3)  + 
    geom_errorbar(aes(ymin = lower, ymax = upper), width = .2, position = position_dodge(width = 0.5)) + 
    facet_wrap(~metric, scales = "free_y") + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Average Treatment Effect for Treated (ATT) with 95% CI", x = "Injury") + 
    ggtitle("Average Treatment Effect for Treated (ATT) per Injury Type, by Metric") 
ggsave("att_causal_plot.png", att_plot)