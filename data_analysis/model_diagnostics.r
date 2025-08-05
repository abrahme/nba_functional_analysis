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
library(ggridges)
library(pheatmap)

data <- read.csv("data/injury_player_cleaned.csv")

player_corrs <- injury_data <-  data |> 
    mutate(`GP%` = games / pmax(games, total_games, na.rm = TRUE),
            MPG = minutes / games,
            BLK = 36 * (blk / minutes),
            AST = 36 * (ast / minutes),
            TOV = 36 * (tov / minutes),
            OREB = 36 * (oreb / minutes),
            DREB = 36 * (dreb / minutes),
            STL = 36 * (stl / minutes),
            `FT%` =  (ftm / fta),
            `FG2%` =  (fg2m / fg2a),
            `FG3%` =  (fg3m / fg3a),
            FG3A = 36 * (fg3a / minutes),
            FG2A = 36 * (fg2a / minutes),
            FTA = 36 * (fta / minutes),
            OBPM = obpm,
            DBPM = dbpm) |>
  select(c(`GP%`, `FT%`, `FG2%`, `FG3%`, FG2A, FG3A, FTA, OREB, DREB, OBPM, DBPM, STL, TOV, AST, MPG, BLK)) |> cor( use = "pairwise.complete.obs")



pheatmap(player_corrs,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         display_numbers = TRUE,
         main = "Metric Correlation Heatmap", 
         filename =  "model_output/model_plots/empirical_correlation.png"
         )

injury_data <-  data |> 
    mutate( pct_games = games / pmax(games, 82, na.rm = TRUE),
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
    select(name, id, obpm, dbpm, pct_games, mpg, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period, year) |>
    rename(pct_minutes = mpg, games = pct_games, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov,
             oreb, dreb, stl, fg3a, fg2a,
             fta, ftm, fg2m, fg3m),
    names_to = "metric",
    values_to = "obs_value")

print("pivoted the original data")


empirical_player_plt <- injury_data |> filter(name %in% c("Kobe Bryant", "Dwight Howard", "LeBron James")) |> mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric)) |> ggplot(aes(x = age, y = obs_value, color = name, group = name)) +  
                              geom_smooth(method = "loess", se = FALSE) + facet_wrap(~metric, scales = "free_y") + theme_bw() + scale_colour_brewer(palette = "Set1") + 
                              ggtitle("An Empirical Production Curve Comparison by Metric") +xlab("Age") + ylab("Metric Value")
ggsave("model_output/model_plots/empirical_production_player.png", empirical_player_plt)    

empirical_plt <- injury_data  |> mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric)) |> ggplot(aes(x = age, y = obs_value)) + 
                              geom_smooth(method = "loess", se = TRUE) + facet_wrap(~metric, scales = "free_y") + theme_bw() + scale_colour_brewer(palette = "Set1") + 
                              ggtitle("Empirical Production Curves by Metric") +xlab("Age") + ylab("Metric Value")
ggsave("model_output/model_plots/empirical_production.png", empirical_plt) 



posterior_data <- read.csv("posterior_ar.csv")
posterior_mu_data <- read.csv("posterior_mu_ar.csv")
print("loaded the posterior data")
posterior_peaks <- read.csv("posterior_peaks_ar.csv")
latent_space <- read.csv("latent_space.csv")
phi_X <- read.csv("phi_X.csv")
third_deriv <- read.csv("posterior_third_deriv_ar.csv")





peaks_plt_df <- posterior_peaks |>
  inner_join(
    data |>
      group_by(id) |>
      summarize(name = first(name), position_group = first(position_group)) |>
      ungroup(),
    by = c("player" = "id")
  ) |>
  mutate(
    metric = toupper(metric),
    metric = case_when(
      metric == "GAMES" ~ "GP%",
      metric == "FG2M" ~ "FG2%",
      metric == "FG3M" ~ "FG3%",
      metric == "FTM" ~ "FT%",
      metric == "PCT_MINUTES" ~ "MPG",
      .default = metric
    ),
    metric_group = case_when(
      metric %in% c("AST", "TOV", "OBPM", "GP%", "DBPM", "MPG", "DREB") ~ "Composite",
      metric %in% c("BLK", "STL", "OREB", "FG2A", "FTA", "FG2%") ~ "Athleticism",
      metric %in% c("FT%", "FG3%", "FG3A") ~ "Skill"
    )
  ) |>
  mutate(
    metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
  )

peaks_plt <- ggplot(peaks_plt_df, aes(y = metric, x = value, fill = metric_group)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  scale_fill_brewer(palette = "Set1") +
  theme_bw() +
  labs(
    title = "Posterior Distribution of Peak Age by Metric",
    x = "Age",
    y = "Metric",
    fill = "Metric Group"
  ) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))


ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvrflvm_max_boundary.png", peaks_plt)


third_deriv_plt <- ggplot(third_deriv |> inner_join(data |> group_by(id) |> summarize(name = first(name), position_group = first(position_group)) |> ungroup(),
                    by = c("player" = "id")) |>
                    mutate(metric = toupper(metric),
                    metric = case_when(metric == "GAMES" ~ "GP%",
                    metric == "FG2M" ~ "FG2%",
                    metric == "FG3M" ~ "FG3%",
                    metric == "FTM" ~ "FT%",
                    metric == "PCT_MINUTES" ~ "MPG",
                    .default = metric)) |> 
                    mutate(metric_group = case_when( metric %in% c("AST", "TOV", "OBPM", "GP%", "DBPM", "MPG", "DREB") ~ "Composite",
                                                     metric %in% c("BLK", "STL", "OREB", "FG2A", "FTA", "FG2%") ~ "Athleticism",
                                                     metric %in% c("FT%", "FG3%", "FG3A") ~ "Skill")) |>
                    group_by(metric) |> summarize(posterior_mean = mean(value), metric_group = first(metric_group),
                    lower = hdi(value, credMass = 0.95)["lower"],
                    upper = hdi(value, credMass = 0.95)["upper"] ) |> ungroup() |> mutate(metric = fct_reorder(metric, posterior_mean, .desc = TRUE)),
               aes(x = metric, y = posterior_mean, color = metric_group)) + geom_point() + geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) + 
  ggtitle("Posterior Distribution of Third Derivative by Metric") +labs(y = "Posterior Value of Third Derivative", x = "Metric", color = "Metric Group") + 
  theme_bw() + scale_fill_brewer(palette = "Set1") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvrflvm_max_boundary_third_deriv.png", third_deriv_plt)

third_deriv_plt <- ggplot(third_deriv |> inner_join(data |> group_by(id) |> summarize(name = first(name), position_group = first(position_group)) |> ungroup(),
                    by = c("player" = "id")) |>
                    mutate(metric = toupper(metric),
                    metric = case_when(metric == "GAMES" ~ "GP%",
                    metric == "FG2M" ~ "FG2%",
                    metric == "FG3M" ~ "FG3%",
                    metric == "FTM" ~ "FT%",
                    metric == "PCT_MINUTES" ~ "MPG",
                    .default = metric)) |> ungroup()
               ) + geom_density_ridges(aes(x = value, y = position_group, fill = position_group)) + facet_wrap(~metric, scales = "free_x") + 
  ggtitle("Posterior Distribution of Third Derivative by Metric, Across Position") +labs(x = "Third Derivative", fill = "Position Group", y = "Position Group") + 
  theme_bw() + scale_fill_brewer(palette = "Set1") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvrflvm_max_boundary_third_deriv_density.png", third_deriv_plt)

curve_third_deriv_plt <- ggplot(third_deriv |> filter(metric == "obpm") |> mutate(quantile = cut(value, breaks = c(-Inf, -0.05, 0.05, Inf),
                                                                     labels = c("Left-Skew Symmetry", "Symmetric", "Right-Skew Symmetry"))) |> group_by(quantile) |> slice_sample(n = 5) |> ungroup() |> group_by(metric, sample,chain, player) |> select(-value) |> 
                                inner_join(posterior_mu_data) |> mutate(value = value - value[age == 18]) , aes(x = age, y = value, group = interaction(chain, sample, player, metric), color = quantile)) + geom_line(alpha = .9)  + theme_bw() + scale_fill_brewer(palette = "Set1") +
                                labs(x = "Age", y = "Latent Curve Value", color = "Skew Type") + ggtitle("Illustration of Third Derivative Influence on Latent Curve Symmetry")

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvrflvm_max_boundary_third_deriv_curves.png", curve_third_deriv_plt)

posterior_data <- posterior_data |> mutate(value = case_when(metric == "pct_minutes" ~ value / 82, 
                                                             metric == "games" ~ value / 82,
                                                             .default = value)) |> 
                                    inner_join(posterior_peaks |> 
                                    rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
joined_data <- posterior_data |> 
                left_join(injury_data |> 
                                semi_join(posterior_data, by = c("id" = "player")
                                ),
                                 by = c("player" = "id", "metric", "age")
                                 ) |> 
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, .direction  = "downup") |> 
                mutate(
                  base_age = if_else(!is.na(year), age, NA_integer_),
                  base_year = if_else(!is.na(year), year, NA_integer_)) |>
                
                fill(base_age, base_year, .direction = "downup") |>
                mutate(
                  year = if_else(is.na(year), base_year + (age - base_age), year)) |> 
                select(-base_age, -base_year) |> ungroup()
print("joined the data with predictions")

validation_coverage_df <- joined_data |> filter(year >= 2022 & year <= 2025) |> group_by(metric, player, age) |> summarize(lower = hdi(value, credMass = 0.95)["lower"],
    upper = hdi(value, credMass = 0.95)["upper"], obs_value = first(obs_value), year = min(year), posterior_mean = mean(value) ) |> ungroup() |>

    mutate(obs_value = if_else(year <= 2025 & !metric %in% c("obpm", "dbpm") & is.na(obs_value), 0 , obs_value), 
    validation_coverage = between(obs_value, lower, upper)) |> filter(!is.na(obs_value)) |> 
    mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))

in_sample_coverage_df <- joined_data |> filter(year <= 2021) |> group_by(metric, player, age) |> summarize(lower = hdi(value, credMass = 0.95)["lower"],
    upper = hdi(value, credMass = 0.95)["upper"], obs_value = first(obs_value), year = min(year)) |> ungroup() |>  
    mutate(obs_value = if_else(!metric %in% c("obpm", "dbpm") & is.na(obs_value), 0 , obs_value), 
    in_sample_coverage = between(obs_value, lower, upper)) |> ungroup() |> filter(!is.na(obs_value)) |> 
    mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))

coverage_plt_basic <- validation_coverage_df |> group_by(metric) |> summarize(validation_coverage = mean(validation_coverage)) |> ungroup() |>
                      inner_join(in_sample_coverage_df |> group_by(metric) |> summarize(in_sample_coverage = mean(in_sample_coverage))) |> 
                      pivot_longer(cols = c(in_sample_coverage,validation_coverage),  names_to = "coverage_type", values_to = "Coverage") |>
                      mutate(coverage_type = case_when(coverage_type == "in_sample_coverage" ~ "In-Sample Coverage",
                                                        coverage_type == "validation_coverage" ~ "Validation Coverage")) |>
                      ggplot(aes(x  = coverage_type, y = Coverage, fill = coverage_type)) + geom_col(position = "dodge") +facet_wrap(~ metric, scales = "fixed") +
                      coord_cartesian(ylim = c(0, 1)) +
                      theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Coverage (In-Sample vs. Validation)") + labs(x = NULL, fill = "Coverage Type") + theme(axis.text.x = element_blank())
ggsave("model_output/model_plots/coverage/nba_convex_tvrflvm_max_boundary_ar.png", coverage_plt_basic)
coverage_plt_yearly <- validation_coverage_df |> group_by(metric, year) |> summarize(Coverage = mean(validation_coverage)) |> ungroup() |> ggplot(aes(x = year, y = Coverage)) + 
                        geom_col() + facet_wrap(~metric, scales = "fixed") + coord_cartesian(ylim = c(0, 1)) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Time Horizon") +xlab("Year")
ggsave("model_output/model_plots/coverage/nba_convex_tvrflvm_max_boundary_ar_validation_yearly.png", coverage_plt_yearly)
coverage_plt_minutes <- validation_coverage_df |> inner_join(data |> filter(year <= 2021) |> group_by(id) |> summarize(years_played = n()) |> ungroup(), by = c("player" = "id")) |> group_by(metric, years_played) |> summarize(Coverage = mean(validation_coverage)) |> ungroup() |>
                        ggplot(aes(x = years_played, y = Coverage)) + 
                        geom_point() + facet_wrap(~metric, scales = "fixed") + coord_cartesian(ylim = c(0, 1)) +
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Years of Training Data Available") + xlab("Years Played")
ggsave("model_output/model_plots/coverage/nba_convex_tvrflvm_max_boundary_ar_validation_minutes.png", coverage_plt_minutes)
posterior_plot_names <-  c("Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                "Chris Paul", "Shaquille O'Neal","Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd","Marcus Camby", "Rudy Gobert", "Tim Duncan",
                 "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", 
                 "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", "Giannis Antetokounmpo", "Jrue Holiday")



latent_space_umap <- latent_space |> select(starts_with("Dim")) |> umap() |> as.tibble(.name_repair = "unique") |> cbind(latent_space |> select(-starts_with("Dim"))) |> rename(UMAP1 = `...1`, UMAP2 = `...2`)

latent_space_plot  <- latent_space_umap |> ggplot(aes(x = UMAP1, y = UMAP2)) + geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(
                      data = filter(latent_space_umap, name %in% posterior_plot_names),
                      aes(label = name),
                      size = 4,
                      fontface = "bold",
                      max.overlaps = Inf) +
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("UMAP Visualization of Learned Latent Embedding") + labs(x = "UMAP 1", y = "UMAP 2", alpha = "Minutes", color = "Position Group")

ggsave("model_output/model_plots/latent_space/map/nba_convex_tvrflvm_max_boundary.png", latent_space_plot)





plot_posterior <- function(grouped_data_set, hold_out_year) {

  group_name <- unique(grouped_data_set$name)
  raw_plt <-
  grouped_data_set |> mutate(
    age_of_holdout = if_else(year ==  hold_out_year, age, Inf),
    age_of_holdout = min(age_of_holdout)
  ) 
  
  validation_label <- "Hold-Out"
  
  plt <- raw_plt |>
    ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "blue",
                                       alpha = 0.2) +
    geom_line(aes(x = age, y = posterior_mean)) +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") +
    facet_wrap( ~ metric, scales = "free_y") +
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Posterior Predictive Career Trajectory: ", group_name))
  return(plt)
}


plots_list <- joined_data |> 
              group_by(metric, player, age) |> 
              summarize(lower = hdi(value, credMass = 0.95)["lower"],
              upper = hdi(value, credMass = 0.95)["upper"], 
              obs_value = first(obs_value), 
              year = min(year), 
              posterior_mean = mean(value)) |> ungroup() |> 
            mutate(metric = toupper(metric),
                    metric = case_when(metric == "GAMES" ~ "GP%",
                    metric == "FG2M" ~ "FG2%",
                    metric == "FG3M" ~ "FG3%",
                    metric == "FTM" ~ "FT%",
                    metric == "PCT_MINUTES" ~ "MPG",
                    .default = metric)) |> inner_join(latent_space |> filter(name %in% posterior_plot_names) |> select(name,id), by = c("player" = "id")) %>%
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_posterior(.x, 2021)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/player_plots/predictions/mcmc/nba_convex_tvrflvm_max_boundary_AR_{name}.png"),
      plot = plt
    )
    })




X <- latent_space |> select(starts_with("Dim")) |> as.matrix()
D <- dist(X)^2  # default: Euclidean

# 2. Convert to similarity matrix
K <- exp(-as.matrix(D) / 2)


K_subset <- K[latent_space$name %in% posterior_plot_names, latent_space$name %in% posterior_plot_names]

hr <- hclust(as.dist(1 - K_subset))

# Step 2: reorder full matrix by clustering
mat_subset <- K_subset[hr$order, hr$order]

# Step 5: create subset labels (optional)
labels_subset <- posterior_plot_names[hr$order]

# Step 6: plot subset matrix, no reclustering
pheatmap(mat_subset,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         labels_row = labels_subset,
         labels_col = labels_subset,
         main = "Clustered Covariance of the Learned Latent Embedding",
         filename =  "model_output/model_plots/latent_space/nba_injury_tvrflvm_max_boundary_K_X.png")



