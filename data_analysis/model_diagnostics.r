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
library(umap)
library(dbscan)
library(ggforce)
library(patchwork)


posterior_plot_names <-  c("Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Dwight Howard",  "Nikola Jokic", "Kevin Garnett", "Steve Nash", 
                "Chris Paul", "Shaquille O'Neal","Anthony Edwards", "Jamal Murray", "Donovan Mitchell", "Ray Allen", "Klay Thompson",
                "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki", "Jason Kidd","Marcus Camby", "Rudy Gobert", "Tim Duncan",
                 "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic", "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton", 
                 "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose", "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis", "Giannis Antetokounmpo", "Jrue Holiday", "No Name")


fake_data <- data.frame(age = 18:38, name = "No Name", id = "99999999", year = 2000:2020)
data <- read.csv("data/injury_player_cleaned.csv") %>% mutate(retirement = 1) %>% bind_rows(fake_data)
player_year_bounds <- data %>%
  group_by(id) %>% arrange(age) %>%
  summarise(
    first_obs = min(year[!is.na(age)], na.rm = TRUE),
    last_obs  = max(year[!is.na(age)], na.rm = TRUE)
  )


player_corrs <- injury_data <-  data |> 
    mutate(`GP%` = games / pmax(games, total_games, na.rm = TRUE),
            MPG = (minutes / games) / 48,
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
            USG = usg,
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
    mutate( pct_games = games / pmax(games, total_games, na.rm = TRUE),
            mpg = (minutes / games) ,
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
            usg = (usg / 100) + .01, 
            fg2_pct =  (fg2m / fg2a), 
            fg3_pct =  (fg3m / fg3a)) |>
    select(name, id, obpm, dbpm, pct_games, mpg, usg, blk_rate, ast_rate, tov_rate, oreb_rate, dreb_rate, stl_rate, fg3a_rate, fg2a_rate, fta_rate, ft_pct, fg2_pct, fg3_pct, age, first_major_injury, injury_period, year, retirement) |>
    rename(pct_minutes = mpg, games = pct_games, blk = blk_rate, ast = ast_rate, tov = tov_rate, oreb = oreb_rate, dreb = dreb_rate, stl = stl_rate, fg3a = fg3a_rate, fg2a = fg2a_rate, fta = fta_rate, ftm = ft_pct, fg2m = fg2_pct, fg3m = fg3_pct) |>
    pivot_longer( cols = c(obpm, dbpm, games, pct_minutes, blk, ast, tov, retirement,
             oreb, dreb, stl, usg, fg2a, fg3a,
             fta, ftm, fg2m, fg3m, retirement),
    names_to = "metric",
    values_to = "obs_value")

print("pivoted the original data")


empirical_player_plt <- injury_data |> filter(metric != "retirement") |> filter(name %in% c("Kobe Bryant", "Dwight Howard", "LeBron James")) |> mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric), 
                              
          metric = case_when(metric %in% c("OBPM", "DBPM") ~ paste0(metric, " (𝓖)"),
                              metric %in% c("FG2%", "FT%", "FG3%") ~ paste0(metric, " (𝓑)"),
                              metric %in% c("MPG", "GP%") ~ metric,
                              .default =  paste0(metric, " (𝓡)")
                    )) |>
                              ggplot(aes(x = age, y = obs_value, color = name, group = name)) +  
                              geom_smooth(method = "loess", se = FALSE) + facet_wrap(~metric, scales = "free_y") + theme_bw() + scale_colour_brewer(palette = "Set1") + 
                              ggtitle("An Empirical Production Curve Comparison by Metric") +xlab("Age") + ylab("Metric Value") +theme(legend.position = "bottom",
                              legend.justification = "center",
                              legend.title = element_blank())

ggsave("model_output/model_plots/empirical_production_player.png", empirical_player_plt)    

empirical_plt <- injury_data  |> filter(metric != "retirement") |> mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric),
          metric = case_when(metric %in% c("OBPM", "DBPM") ~ paste0(metric, " (𝓖)"),
                              metric %in% c("FG2%", "FT%", "FG3%") ~ paste0(metric, " (𝓑)"),
                              metric %in% c("MPG", "GP%") ~ metric,
                              .default =  paste0(metric, " (𝓡)"))) |> ggplot(aes(x = age, y = obs_value)) + 
                              geom_smooth(method = "loess", se = TRUE) + facet_wrap(~metric, scales = "free_y") + theme_bw() + scale_colour_brewer(palette = "Set1") + 
                              ggtitle("Empirical Production Curves by Metric") +xlab("Age") + ylab("Metric Value")
ggsave("model_output/model_plots/empirical_production.png", empirical_plt) 


posterior_data <- read.csv("posterior_ar_linear.csv") |> mutate(value = if_else(metric == "pct_minutes", value * 48, value))
posterior_mu_data <- read.csv("posterior_mu_ar_linear.csv") 
posterior_data_no_ar <- read.csv("posterior_no_ar_linear.csv") |> mutate(value = if_else(metric == "pct_minutes", value * 48, value))
print("loaded the posterior data")
posterior_peaks <- read.csv("posterior_peaks_ar_linear.csv")
posterior_peak_vals <- read.csv("posterior_peak_vals_ar_linear.csv")
latent_space <- read.csv("latent_space_linear.csv")
phi_X <- read.csv("phi_X_linear.csv")
third_deriv <- read.csv("posterior_third_deriv_ar_linear.csv")
first_deriv = read.csv("posterior_first_deriv_ar_linear.csv")


# obpm_curves <- posterior_mu_data |> filter(metric == "dbpm") |> group_by(player, age) |> summarize(value = mean(value)) |> ungroup() |> group_by(player) |> arrange(age, .by_group = TRUE) |>
# mutate(value_normalized = (value - first(value)) / sd(value)) |> select(-c(value)) |> pivot_wider(names_from = age, values_from = value_normalized, names_prefix = "age_")
# obpm_curves_corr <- obpm_curves |> select(-player) |> as.matrix() |> t() |> cor(use = "pairwise.complete.obs")

# dist_mat_curves <- as.dist(1 - obpm_curves_corr)
# hc_curves <- hclust(dist_mat_curves, method = "ward.D2")  # or "complete", "ward.D2" etc
# k <- 3
# cluster_labels <- cutree(hc_curves, k)
# obpm_curves$cluster <- factor(cluster_labels)


# obpm_curves_cluster <- obpm_curves %>%
#   select(-c(player)) %>%
#   pivot_longer(
#     cols = starts_with("age_"),
#     names_to = "age",
#     values_to = "value"
#   ) %>%
#   mutate(age = as.numeric(sub("age_", "", age))) %>%
#   group_by(cluster, age) %>%
#   summarise(mean_value = mean(value, na.rm = TRUE)) %>%
#   ungroup()

# cluster_curves_plt <- ggplot(obpm_curves_cluster, aes(x = age, y = mean_value, color = cluster)) + geom_line() + theme_bw() +  scale_colour_brewer(palette = "Set1") + 
#                               ggtitle("Clustered Normalized OBPM Curves") +xlab("Age") + ylab("Metric Value")
# ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_cluster.png", cluster_curves_plt) 



peaks_plt_df <- posterior_peaks |>
  inner_join(
    data |>
      group_by(id) |>
      summarize(name = first(name), position_group = first(position_group), minutes = sum(minutes)) |>
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
    )) |>
  mutate(
    metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
  )

peak_vals_plt_df <- posterior_peak_vals |> 
    inner_join(
    data |>
      group_by(id) |>
      summarize(name = first(name), position_group = first(position_group), minutes = sum(minutes)) |>
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
    )) |>
  mutate(
    metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
  )


peak_2019_class <- peaks_plt_df |> rename(peak_age = value) |> inner_join(peak_vals_plt_df |> rename(peak_val = value)) |> filter(metric == "OBPM") |> inner_join(data |> group_by(id) |> arrange(year, .by_group = TRUE) |> filter(2016 == first(year)) |> ungroup() |> distinct(id) , by = c("player" = "id")) |>
  group_by(player) |> summarize(peak_val = mean(peak_val), peak_age = mean(peak_age), position_group = first(position_group), name = first(name)) |> 
  ggplot(aes(x = peak_age, y = peak_val, color = position_group)) + 
  geom_point() + 
  geom_text_repel(aes(label = name),
                  color = "black",
                  fontface = "bold",
                  max.overlaps = 5) + theme_bw() + 
  scale_colour_brewer(palette = "Set1") + ggtitle("Posterior Mean of Peak OBPM Age, Value for 2016 Draft Class") + labs(x = "Peak Age", color = "Position Group", y = "Peak OBPM Value")

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_2018.png", peak_2019_class)



# dbpm_df <- peaks_plt_df %>%
#   filter(metric == "DBPM") %>%
#   group_by(player) %>%
#   summarize(
#     Name = first(name),
#     `Position Group` = first(position_group),
#     `Posterior Peak Age` = round(mean(value),1)
#   ) %>%
#   ungroup() %>%
#   arrange(`Posterior Peak Age`) %>% select(-player)

# latex_code <- rbind(dbpm_df |> slice_head(n = 5), 
#                     dbpm_df |> slice_tail(n = 5), 
#                     peaks_plt_df |> filter(metric == "DBPM") |> summarize(Name = 'Average',`Posterior Peak Age` = round(mean(value),1), `Position Group` = '----')) |>
#               arrange(`Posterior Peak Age`) |> 
#                gt() %>%
#                tab_header(title = "DBPM Posterior Peak") %>%
#                as_latex()   # get LaTeX code

# writeLines(latex_code, "model_output/model_plots/peaks/mcmc/dbpm_peak_table.tex")



# obpm_df <- peaks_plt_df %>%
#   filter(metric == "OBPM") %>%
#   group_by(player) %>%
#   summarize(
#     Name = first(name),
#     `Position Group` = first(position_group),
#     `Posterior Peak Age` = round(mean(value),1)
#   ) %>%
#   ungroup() %>%
#   arrange(`Posterior Peak Age`) %>% select(-player)

# latex_code <- rbind(obpm_df |> slice_head(n = 5), 
#                     obpm_df |> slice_tail(n = 5), 
#                     peaks_plt_df |> filter(metric == "OBPM") |> summarize(Name = 'Average',`Posterior Peak Age` = round(mean(value),1), `Position Group` = '----')) |>
#               arrange(`Posterior Peak Age`) |> 
#                gt() %>%
#                tab_header(title = "OBPM Posterior Peak") %>%
#                as_latex()   # get LaTeX code

# writeLines(latex_code, "model_output/model_plots/peaks/mcmc/obpm_peak_table.tex")




peaks_players <- peaks_plt_df %>% group_by(metric, player) %>% summarize(value = mean(value)) %>% ungroup() %>% pivot_wider(names_from = metric, values_from = value) %>% inner_join(latent_space %>% filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) %>% select(id), by = c("player" = "id"))

peaks_pca <- prcomp(peaks_players  %>% select(-c(player,RETIREMENT)) %>% data.matrix() , scale. = TRUE, center = TRUE)

peaks_pca_df <- tibble(PC1 = peaks_pca$x[,1], PC2 = peaks_pca$x[,2], id = peaks_players$player) %>% inner_join(latent_space %>% select(id, name, position_group, minutes))

max_range <- max(abs(range(peaks_pca_df$PC1)),
                 abs(range(peaks_pca_df$PC2 )))



tops <- c(peaks_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(10))
bottoms <- c(peaks_pca_df %>% arrange(PC1) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(PC2) %>% pull(name) %>% head(10))

pca_outlier_names <- unique(c(tops,bottoms, posterior_plot_names))

peaks_pca_plot  <-filter(peaks_pca_df, name %in% pca_outlier_names) %>% ggplot(aes(x = PC1, y = PC2)) + 
                      geom_text_repel(
                      aes(label = name, x = PC1, y = PC2),
                      size = 3,
                      fontface = "bold",
                      max.overlaps = 20,
                      inherit.aes = FALSE) + 
  coord_fixed() +
  xlim(-max_range, max_range) +
  ylim(-max_range, max_range) + 
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("PCA Visualization of Learned Metric Peaks") + labs(x = "PC 1", y = "PC 2")

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_pca.png", peaks_pca_plot)

# Extract loadings
loadings <- as.data.frame(peaks_pca$rotation[, 1:2])
loadings$metric <- rownames(loadings)
db <- kmeans(loadings %>% select(-c(metric)), center = 3)
loadings$metric_group <- as.factor(db$cluster) 
peaks_pca_loadings_plt <- ggplot(loadings, aes(x = PC1, y = PC2)) +
  geom_text(aes(label = metric, color = metric_group), size = 4,
                      fontface = "bold", show.legend = FALSE) +
  geom_point(aes(color = metric_group), alpha = 0) + 
  theme_bw() + 
  guides(color = guide_legend(override.aes = list(shape = 16, alpha = 1))) + 
  scale_colour_brewer(palette = "Set1") + 
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
  # Make equal scaling so 0,0 is visually centered
  coord_cartesian(xlim = c(-max(abs(loadings$PC1)), max(abs(loadings$PC1))), 
                  ylim = c(-max(abs(loadings$PC2)), max(abs(loadings$PC2)))) +
  ggtitle("PCA Visualization of Learned Metric Peaks Factor Loadings") + labs(x = "PC 1", y = "PC 2", color = "Metric Group")
ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_pca_loadings.png", peaks_pca_loadings_plt)


peak_vals_players <- peak_vals_plt_df %>% group_by(metric, player) %>% summarize(value = mean(value)) %>% ungroup() %>% pivot_wider(names_from = metric, values_from = value) %>% inner_join(latent_space %>% filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) %>% select(id), by = c("player" = "id"))

peak_vals_pca <- prcomp(peak_vals_players  %>% select(-c(player,RETIREMENT)) %>% data.matrix() , scale. = TRUE, center = TRUE)

peak_vals_pca_df <- tibble(PC1 = peak_vals_pca$x[,1], PC2 = peak_vals_pca$x[,2], id = peak_vals_players$player) %>% inner_join(latent_space %>% select(id, name, position_group, minutes))

tops <- c(peak_vals_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(10), peak_vals_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(10))
bottoms <- c(peak_vals_pca_df %>% arrange(PC1) %>% pull(name) %>% head(10), peak_vals_pca_df %>% arrange(PC2) %>% pull(name) %>% head(10))

pca_outlier_names <- unique(c(tops,bottoms, posterior_plot_names))

peak_vals_pca_plot  <-  peak_vals_pca_df %>% ggplot(aes(x = PC1, y = PC2)) +  geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(data = filter(peak_vals_pca_df, name %in% posterior_plot_names), 
                      aes(label = name, x = PC1, y = PC2),
                      size = 3,
                      fontface = "bold",
                      max.overlaps = 20,
                      inherit.aes = FALSE) + 
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("PCA Visualization of Learned Metric Peak Values") + labs(x = "PC 1", y = "PC 2", color = "Position Group", alpha = "Minutes")

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_pca_vals.png", peak_vals_pca_plot)

# Extract loadings
loadings_vals <- as.data.frame(peak_vals_pca$rotation[, 1:2])
loadings_vals$metric <- rownames(loadings_vals)
# db <- dbscan(loadings %>% select(-c(metric)), eps = .2, minPts = 2)
# loadings$metric_group <- as.factor(db$cluster) 
peak_vals_pca_loadings_plt <- ggplot(loadings_vals, aes(x = PC1, y = PC2)) +
  geom_text_repel(aes(label = metric), size = 4,
                      fontface = "bold", show.legend = FALSE) +
  theme_bw() + 
  guides(color = guide_legend(override.aes = list(shape = 16, alpha = 1))) + 
  scale_colour_brewer(palette = "Set1") + 
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
  # Make equal scaling so 0,0 is visually centered
  coord_cartesian(xlim = c(-max(abs(loadings_vals$PC1)), max(abs(loadings_vals$PC1))), 
                  ylim = c(-max(abs(loadings_vals$PC2)), max(abs(loadings_vals$PC2)))) +
  ggtitle("PCA Visualization of Learned Metric Peak Values Factor Loadings") + labs(x = "PC 1", y = "PC 2")
ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_pca_vals_loadings.png", peak_vals_pca_loadings_plt)


skew_plt_df <- posterior_peaks |> inner_join(
    data |>
      group_by(id) |>
      summarize(name = first(name), position_group = first(position_group), minutes = sum(minutes)) |>
      ungroup(),
    by = c("player" = "id")
  ) |> rename(peak_age = value) |> inner_join(posterior_mu_data |> rename(mu = value)) |>
  mutate(peak_int = ceiling(peak_age)) |> group_by(metric, chain, sample, player) |> arrange(age) |> 
  summarize(first_deriv_pre = (mu[age==peak_int] - first(mu))/(peak_int - 18),
  first_deriv_post = (last(mu) - mu[age == peak_int]) / (38 - peak_int), minutes = first(minutes)) |> ungroup() |>
  mutate(
      metric = toupper(metric),
      metric = case_when(
        metric == "GAMES" ~ "GP%",
        metric == "FG2M" ~ "FG2%",
        metric == "FG3M" ~ "FG3%",
        metric == "FTM" ~ "FT%",
        metric == "PCT_MINUTES" ~ "MPG",
        .default = metric
      )) |>
      inner_join(loadings, by = "metric") |> 
       group_by(metric, player) |> summarize(first_deriv_pre = mean(first_deriv_pre), first_deriv_post = mean(first_deriv_post), minutes = first(minutes),
    metric_group = first(metric_group)) |> ungroup()


skew_plt <- ggplot(skew_plt_df, aes( x = first_deriv_pre / first_deriv_post, color = metric_group, y = metric, )) +
      stat_pointinterval() + 
  scale_color_brewer(palette = "Set1") + scale_y_discrete(expand = expansion(mult = c(0.2, 0.2))) + 
  theme_bw() +
  labs(
    title = "Posterior Mean of Pre vs. Post Peak First Deriv. by Metric",
    x = "Ratio of Pre-Peak to Post-Peak First Derivative",
    color = "Metric Group"
  ) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) 


ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_skew.png", skew_plt)


peaks_plt <- ggplot(peaks_plt_df |> inner_join(loadings, by = "metric") |> group_by(metric, player) |> summarize(minutes = first(minutes), metric_group = first(metric_group), value = mean(value)) |> ungroup() |>   mutate(
    metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
  ), 

  aes(y = metric, x = value, fill = metric_group, color = metric_group)) +
  stat_pointinterval() + 
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") + 
  theme_bw() +
  labs(
    title = "Posterior Mean of Peak Age by Metric",
    x = "Age",
    y = "Metric",
    fill = "Metric Group",
    color = "Metric Group",
  ) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + scale_y_discrete(expand = expansion(mult = c(0.2, 0.2)))



ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max.png", peaks_plt)




third_deriv_plt <- ggplot(third_deriv |> inner_join(data |> group_by(id) |> summarize(name = first(name), position_group = first(position_group), minutes = sum(minutes)) |> ungroup(),
                    by = c("player" = "id")) |>
                    mutate(metric = toupper(metric),
                    metric = case_when(metric == "GAMES" ~ "GP%",
                    metric == "FG2M" ~ "FG2%",
                    metric == "FG3M" ~ "FG3%",
                    metric == "FTM" ~ "FT%",
                    metric == "PCT_MINUTES" ~ "MPG",
                    .default = metric)) |> 
                    inner_join(loadings, by = "metric") |>
                    
                    group_by(metric, player) |> summarize(posterior_mean = mean(value), metric_group = first(metric_group), minutes = first(minutes)) |> ungroup() |>
                     mutate(metric = fct_reorder(metric, posterior_mean, .fun = median, .desc = TRUE))
                     ,
               aes(x = metric, y = posterior_mean, color = metric_group)) + 
                stat_pointinterval() + 
  ggtitle("Posterior Mean of Third Derivative by Metric") +labs(y = "Posterior Mean of Third Derivative", x = "Metric", color = "Metric Group") + 
  theme_bw() + scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) 


ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_third_deriv.png", third_deriv_plt)



curve_third_deriv_plt <- ggplot(third_deriv |> filter(metric == "obpm") |> mutate(quantile = cut(value, breaks = c(-Inf, -0.05, 0.05, Inf),
                                                                     labels = c("Left-Skew Symmetry", "Symmetric", "Right-Skew Symmetry"))) |> group_by(quantile) |> slice_sample(n = 5) |> ungroup() |> group_by(metric, sample,chain, player) |> select(-value) |> 
                                inner_join(posterior_mu_data) |> mutate(value = value - value[age == 18]) , aes(x = age, y = value, group = interaction(chain, sample, player, metric), color = quantile)) + geom_line(alpha = .9)  + theme_bw() + scale_fill_brewer(palette = "Set1") +
                                labs(x = "Age", y = "Latent Curve Value", color = "Skew Type") + ggtitle("Illustration of Third Derivative Influence on Latent Curve Symmetry")

ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_max_third_deriv_curves.png", curve_third_deriv_plt)

posterior_data <- posterior_data  |> 
                                    inner_join(posterior_peaks |> 
                                    rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
posterior_data_no_ar <- posterior_data_no_ar |> 
                                    inner_join(posterior_peaks |> 
                                    rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
joined_data <- posterior_data |> 
                left_join(injury_data |> inner_join(player_year_bounds) |>
                                semi_join(posterior_data, by = c("id" = "player")
                                ),
                                 by = c("player" = "id", "metric", "age")
                                 )  |>
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, first_major_injury, .direction  = "downup") |> 
                fill(injury_period, .direction = "up") |> mutate(injury_period = replace_na(injury_period, "post-injury")) |>
                mutate(
                  base_age = if_else(!is.na(year), age, NA_integer_),
                  base_year = if_else(!is.na(year), year, NA_integer_)) |>
                
                fill(base_age, base_year, .direction = "downup") |>
                mutate(
                  year = if_else(is.na(year), base_year + (age - base_age), year)) |> 
                select(-base_age, -base_year) |> ungroup() |> mutate(
                  obs_value = case_when(
                    # Rule A: year <= target_year & year >= last_obs for the target metric
                    metric == "retirement" & is.na(obs_value) & year <= 2026  & year > last_obs  ~ 0,
                    # Rule B: year >= target_min_year & year <= first_obs for the target metric
                    metric == "retirement" & is.na(obs_value) & year > 1997 & year < first_obs ~ 0,
                    TRUE ~ obs_value
                  )
                ) |>
                select(-first_obs, -last_obs)
joined_data_no_ar <- posterior_data_no_ar |> 
                left_join(injury_data |> 
                                semi_join(posterior_data, by = c("id" = "player")
                                ),
                                 by = c("player" = "id", "metric", "age")
                                 ) |> inner_join(player_year_bounds, by = c("player" = "id")) |>
                group_by(player, chain, sample, metric) |> 
                arrange(age) |> fill(name, .direction  = "downup") |> 
                mutate(
                  base_age = if_else(!is.na(year), age, NA_integer_),
                  base_year = if_else(!is.na(year), year, NA_integer_)) |>
                
                fill(base_age, base_year, .direction = "downup") |>
                mutate(
                  year = if_else(is.na(year), base_year + (age - base_age), year)) |> 
                select(-base_age, -base_year) |> ungroup() |> mutate(
                  obs_value = case_when(
                    # Rule A: year <= target_year & year >= last_obs for the target metric
                    metric == "retirement" & is.na(obs_value) & year <= 2026  & year > last_obs  ~ 0,
                    # Rule B: year >= target_min_year & year <= first_obs for the target metric
                    metric == "retirement" & is.na(obs_value) & year > 1997 & year < first_obs ~ 0,
                    TRUE ~ obs_value
                  )
                ) |>
                select(-first_obs, -last_obs)
print("joined the data with predictions")

validation_coverage_df <- joined_data |> filter(year >= 2022 & year <= 2025 & metric != "retirement") |> group_by(metric, player, age) |> summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(value, credMass = 0.95)["upper"], obs_value = first(obs_value), year = min(year), posterior_mean = mean(value, na.rm = TRUE) ) |> ungroup() |>

    mutate(
    validation_coverage = between(obs_value, lower, upper)) |> filter(!is.na(obs_value)) |> 
    mutate(metric = toupper(metric),
           metric = case_when(metric == "GAMES" ~ "GP%",
                              metric == "FG2M" ~ "FG2%",
                              metric == "FG3M" ~ "FG3%",
                              metric == "FTM" ~ "FT%",
                              metric == "PCT_MINUTES" ~ "MPG",
                              .default = metric))
validation_coverage_df_retirement <- joined_data |> filter(year >= 2022 & year <= 2025 & metric == "retirement") %>% mutate(metric = toupper(metric)) %>% filter(!is.na(obs_value)) %>% group_by(metric) %>% summarize(validation_coverage = mean(if_else(obs_value == 1, value, 1 - value))) %>% ungroup()

in_sample_coverage_df <- joined_data |> filter(year <= 2021 & metric != "retirement") |> group_by(metric, player, age) |> summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
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

in_sample_coverage_df_retirement <- joined_data |> filter(year <= 2021 & metric == "retirement") %>% mutate(metric = toupper(metric)) %>% filter(!is.na(obs_value)) %>% group_by(metric) %>% summarize(in_sample_coverage = mean(if_else(obs_value == 1, value, 1 - value))) %>% ungroup()


coverage_plt_basic <- validation_coverage_df |> group_by(metric) |> summarize(validation_coverage = mean(validation_coverage, na.rm = TRUE)) |> ungroup() |>
                      bind_rows(validation_coverage_df_retirement) |> 
                      inner_join(in_sample_coverage_df |> group_by(metric) |> summarize(in_sample_coverage = mean(in_sample_coverage, na.rm = TRUE)) |> ungroup() |> bind_rows(in_sample_coverage_df_retirement)) |> 
                      pivot_longer(cols = c(in_sample_coverage,validation_coverage),  names_to = "coverage_type", values_to = "Coverage") |>
                      mutate(coverage_type = case_when(coverage_type == "in_sample_coverage" ~ "In-Sample Coverage",
                                                        coverage_type == "validation_coverage" ~ "Validation Coverage")) |>
                      ggplot(aes(x  = coverage_type, y = Coverage, fill = coverage_type)) + geom_col(position = "dodge") +facet_wrap(~ metric, scales = "fixed") +
                      coord_cartesian(ylim = c(0, 1)) +
                      theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Coverage (In-Sample vs. Validation)") + labs(x = NULL, fill = "Coverage Type") + theme(axis.text.x = element_blank())
ggsave("model_output/model_plots/coverage/nba_convex_tvlinearlvm_max_ar.png", coverage_plt_basic)

latex_code <- validation_coverage_df |> group_by(metric) |> summarize(validation_coverage = mean(validation_coverage, na.rm = TRUE)) |> ungroup() |>
                      bind_rows(validation_coverage_df_retirement) |> 
                      inner_join(in_sample_coverage_df |> group_by(metric) |> summarize(in_sample_coverage = mean(in_sample_coverage, na.rm = TRUE)) |> ungroup() |> bind_rows(in_sample_coverage_df_retirement)) |>
                      mutate(
                          in_sample_coverage = paste0(round(in_sample_coverage * 100, 1), "%"),
                          validation_coverage = paste0(round(validation_coverage * 100, 1), "%")) |> gt() |>
                        cols_label(
                          metric = "Metric",
                          in_sample_coverage = "In-Sample Coverage",
                          validation_coverage = "Validation Coverage"
                        ) |>
                        tab_header(title = "Coverage Summary") |>
                        as_latex()
writeLines(latex_code, "model_output/model_plots/coverage/nba_convex_tvlinearlvm_max_ar.tex")

coverage_plt_yearly <- validation_coverage_df |> group_by(metric, year) |> summarize(Coverage = mean(validation_coverage, na.rm = TRUE)) |> ungroup() |> 
                        bind_rows(joined_data %>% filter(year >= 2022 & year <= 2025 & metric == "retirement") %>% mutate(metric = toupper(metric)) %>% filter(!is.na(obs_value)) %>% group_by(metric, year) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value))) %>% ungroup()) |> ggplot(aes(x = year, y = Coverage)) + 
                        geom_col() + facet_wrap(~metric, scales = "free_y") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Time Horizon") +xlab("Year")
ggsave("model_output/model_plots/coverage/nba_convex_tvlinearlvm_max_ar_validation_yearly.png", coverage_plt_yearly)
coverage_plt_minutes <- validation_coverage_df |> inner_join(data |> filter(year <= 2021) |> group_by(id) |> summarize(years_played = n()) |> ungroup(), by = c("player" = "id")) |> group_by(metric, years_played) |> summarize(Coverage = mean(validation_coverage, na.rm = TRUE)) |> ungroup() |> bind_rows(
  joined_data %>% 
  filter(year >= 2022 & year <= 2025 & metric == "retirement") %>% 
  mutate(metric = toupper(metric)) %>% 
  filter(!is.na(obs_value)) %>% 
  inner_join(data %>% 
  filter(year <= 2021) %>% 
  group_by(id) %>% 
  summarize(years_played = n()) %>% 
  ungroup(), by = c("player" = "id")) %>% 
  group_by(metric, years_played) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value))) %>% ungroup()) %>%
                        ggplot(aes(x = years_played, y = Coverage)) + 
                        geom_point() + facet_wrap(~metric, scales = "free_y") +
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Years of Training Data Available") + xlab("Years Played")
ggsave("model_output/model_plots/coverage/nba_convex_tvlinearlvm_max_ar_validation_minutes.png", coverage_plt_minutes)






latent_space_umap <- latent_space %>% select(starts_with("Dim")) %>% umap(n_neighbors = 50, min_dist = 0.001, verbose = TRUE) %>% .$layout %>% as.tibble(.name_repair = "unique") %>% cbind(latent_space %>% select(-starts_with("Dim"))) %>% rename(UMAP1 = `...1`, UMAP2 = `...2`)

latent_space_plot  <- latent_space_umap |> ggplot(aes(x = UMAP1, y = UMAP2)) + geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(
                      data = filter(latent_space_umap, name %in% posterior_plot_names),
                      aes(label = name, x = UMAP1, y = UMAP2),
                      size = 4,
                      fontface = "bold",
                      max.overlaps = 5,
                      inherit.aes = FALSE) +
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("UMAP Visualization of Learned Latent Embedding") + labs(x = "UMAP 1", y = "UMAP 2", alpha = "Minutes", color = "Position Group")

ggsave("model_output/model_plots/latent_space/map/nba_convex_tvlinearlvm_max.png", latent_space_plot)

functional_pca_result <-  posterior_mu_data %>% group_by(age, player, metric) %>% summarize(value = mean(value, na.rm = TRUE)) %>% ungroup() %>% pivot_wider(
  names_from = c(metric, age),
    values_from = value,
    names_sep = "_"
) 
ids <- functional_pca_result$player 



functional_pca_embedding <- functional_pca_result %>% 
  select(-player) %>% 
  prcomp(center = TRUE, scale. = TRUE) %>%       # perform PCA
  .$x %>%                                       # extract principal component scores
  as.data.frame() %>%                            # convert to data frame
  as_tibble(.name_repair = "unique") %>%        # repair names
  rename(PCA1 = `PC1`, PCA2 = `PC2`)            # rename first two components

functional_pca_embedding$player = ids 
functional_pca_embedding <- functional_pca_embedding %>% inner_join( latent_space %>% select(id, name, position_group, minutes), by = c("player" = "id"))
functional_pca_plt <- functional_pca_embedding %>% filter(PCA1 <= 20 & PCA2 <=20) %>%
                       ggplot(aes(x = PCA1, y = PCA2)) + geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(
                      data = filter(functional_pca_embedding, name %in% posterior_plot_names),
                      aes(label = name, x = PCA1, y = PCA2),
                      size = 2,
                      fontface = "bold",
                      max.overlaps = 20,
                      inherit.aes = FALSE) +
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("PCA Visualization of Learned Metric Functionals") + labs(x = "PC 1", y = "PC 2", alpha = "Minutes", color = "Position Group")
ggsave("model_output/model_plots/latent_space/map/nba_convex_tvlinearlvm_max_functional.png", functional_pca_plt)



plot_posterior <- function(grouped_data_set, hold_out_year, plot_obs = TRUE) {

  group_name <- unique(grouped_data_set$name)
  raw_plt <-
  grouped_data_set |> mutate(
    age_of_holdout = if_else(year ==  hold_out_year, age, Inf),
    age_of_holdout = min(age_of_holdout)
  ) 
  
  validation_label <- "Hold-Out"
  
  plt <- raw_plt |>
    ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = posterior_mean)) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) + 
    
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") +
    facet_wrap( ~ metric, scales = "free_y") + theme_bw() + 
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Posterior Predictive Career Trajectory: ", group_name))
  if (plot_obs) {
    plt <- plt + geom_point(aes(x = age, y = obs_value), color = "black")
  }
  return(plt)
}


plot_posterior_mu_spaghetti <- function(grouped_data_set){
  group_name <- unique(grouped_data_set$name)
  hold_out_year <- 2021
  raw_plt <- grouped_data_set |> mutate(
    age_of_holdout = if_else(year ==  hold_out_year, age, Inf),
    age_of_holdout = min(age_of_holdout)
  ) 
    
  validation_label <- "Hold-Out"
    
  
  
  plt <- raw_plt |>
    ggplot(aes(x = age))  +
    geom_line(aes(x = age, y = mu, group = sample),  color = "grey", alpha = .2) + 
    geom_point(aes(x = age, y = obs_value), color = "black") + 
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") + 
    geom_line(aes(x = age, y = posterior_mean)) + 
    
    facet_wrap( ~ metric, scales = "free_y") + theme_bw() + 
    labs(x = "Age", y = "Metric Value") + ggtitle(paste("Posterior Latent Career Trajectory: ", group_name))
  return(plt)
}

plots_list <- joined_data |> 
              group_by(metric, player, age) |> 
              summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
              upper = HDInterval::hdi(value, credMass = 0.95)["upper"], 
              obs_value = first(obs_value), 
              year = min(year), 
              posterior_mean = mean(value, na.rm = TRUE)) |> ungroup() |> inner_join(posterior_mu_data |> group_by(metric, player, age) |> summarize(mu = mean(value)) |> ungroup() |> 
              mutate(mu = case_when(metric %in% c("fg2m", "ftm", "games", "fg3m", "retirement") ~ plogis(mu),
                                                                                 metric %in% c("obpm", "dbpm") ~ mu, 
                                                                                 metric %in% c("pct_minutes") ~ plogis(mu) * 48,
                                                                                 metric %in% c("usg") ~ plogis(mu),
                                                                                 .default = exp(mu) * 36)) ) |>
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
      filename = glue("model_output/model_plots/player_plots/predictions/mcmc/nba_convex_tvlinearlvm_max_AR_{name}.png"),
      plot = plt
    )
    })


plots_list <- joined_data |> 
              group_by(metric, player, age) |> 
              summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
              upper = HDInterval::hdi(value, credMass = 0.95)["upper"], 
              obs_value = first(obs_value), 
              year = min(year), 
              posterior_mean = mean(value, na.rm = TRUE)) |> ungroup() |> inner_join(
              posterior_mu_data |> 
              mutate(mu = case_when(metric %in% c("fg2m", "ftm", "games", "fg3m", "retirement") ~ plogis(value),
                                                                                 metric %in% c("obpm", "dbpm") ~ value, 
                                                                                 metric %in% c("pct_minutes") ~ plogis(value) * 48,
                                                                                 metric %in% c("usg") ~ plogis(value),
                                                                                 .default = exp(value) * 36)))  |>
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
    plt <- plot_posterior_mu_spaghetti(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/player_plots/predictions/mcmc/nba_convex_tvlinearlvm_max_AR_{name}_spaghetti.png"),
      plot = plt
    )
    })



phi_X_mat <- phi_X |> select(starts_with("Dim")) |> as.matrix()

# 2. Convert to similarity matrix
K <- phi_X_mat %*% t(phi_X_mat)


K_subset <- K[phi_X$name %in% posterior_plot_names, phi_X$name %in% posterior_plot_names]

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
         filename =  "model_output/model_plots/latent_space/nba_tvlinearlvm_max_K_X.png")







##### ACTUAL PLOTS FOR THE PAPER 
### PLAYER PLOT
injury_summary <- joined_data |>  group_by(player) |> mutate(age_of_holdout = if_else(year ==  2021, age, Inf),
                    age_of_holdout = min(age_of_holdout),
                  age_of_injury = if_else(injury_period == "post-injury", age, Inf),
                  age_of_injury = min(age_of_injury)) |> ungroup() |> group_by(player) |> 
                  summarize(first_major_injury = first(first_major_injury), 
                            age_of_injury = first(age_of_injury), 
                            age_of_holdout = first(age_of_holdout), 
                            name = first(name)) |> filter(name %in% c("Kevin Durant", "Stephen Curry", "Derrick Rose")) |>
                  mutate(age_of_injury = if_else(is.na(first_major_injury), NA_real_, age_of_injury))
player_plot_df <- joined_data |> 
              group_by(metric, player, age)  |> 
              summarize(lower = HDInterval::hdi(if_else(is.finite(value), value, NA_real_), credMass = 0.95)["lower"],
              upper = HDInterval::hdi(if_else(is.finite(value), value, NA_real_), credMass = 0.95)["upper"], 
              obs_value = first(obs_value), 
              year = min(year), 
              posterior_mean = mean(if_else(is.finite(value), value, NA_real_), na.rm = TRUE)) |> ungroup() |> inner_join(posterior_mu_data |> group_by(metric, player, age) |> summarize(mu = mean(if_else(is.finite(value), value, NA_real_), na.rm = TRUE)) |> ungroup() |> 
              mutate(mu = case_when(metric %in% c("fg2m", "ftm", "games", "fg3m") ~ plogis(mu),
                                                                                 metric %in% c("obpm", "dbpm") ~ mu, 
                                                                                 metric %in% c("pct_minutes") ~ plogis(mu) * 48,
                                                                                 metric %in% c("usg") ~ plogis(mu),
                                                                                 .default = exp(mu) * 36)) ) |>
            mutate(metric = toupper(metric),
                    metric = case_when(metric == "GAMES" ~ "GP%",
                    metric == "FG2M" ~ "FG2%",
                    metric == "FG3M" ~ "FG3%",
                    metric == "FTM" ~ "FT%",
                    metric == "PCT_MINUTES" ~ "MPG",
                    .default = metric)) |> filter(metric %in% c("GP%", "FTA", "OBPM")) |> inner_join(latent_space |> filter(name %in% c("Stephen Curry", "Kevin Durant", "Derrick Rose")) |> select(name,id), by = c("player" = "id"))  |>
            inner_join(injury_summary)        




label_df <- player_plot_df |> group_by(metric, player) |> summarize(max_upper = max(upper), min_lower = min(lower), first_major_injury = first(first_major_injury), age_of_holdout = first(age_of_holdout),
                                                                    age_of_injury = first(age_of_injury), name = first(name)) |> ungroup()
derrick_rose <- player_plot_df |> filter(name == "Derrick Rose") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) + 
    
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +  
    geom_point(aes(x = age, y = obs_value), color = "black") + 
    geom_text(data = label_df |> filter(name == "Derrick Rose"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) + 
    geom_text( 
      data = label_df |> filter(name == "Derrick Rose"),
      size = 3,
      aes(x = age_of_holdout, y = .65*min_lower, label = "Hold-Out Year")) + 
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "") 

kevin_durant <- player_plot_df |> filter(name == "Kevin Durant") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) + 
    
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +  
    geom_point(aes(x = age, y = obs_value), color = "black") + 
    geom_text(data = label_df |> filter(name == "Kevin Durant"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) + 
    geom_text( 
      data = label_df |> filter(name == "Kevin Durant"),
      size = 3,
      aes(x = age_of_holdout, y = .65*min_lower, label = "Hold-Out Year")) + 
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "")  

stephen_curry <- player_plot_df |> filter(name == "Stephen Curry") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) + 
    
    geom_vline(aes(xintercept = age_of_holdout),
               linetype = "dashed",
               color = "red") +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +  
    geom_point(aes(x = age, y = obs_value), color = "black") + 
    geom_text(data = label_df |> filter(name == "Stephen Curry"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) + 
    geom_text( 
      data = label_df |> filter(name == "Stephen Curry"),
      size = 3,
      aes(x = age_of_holdout, y = .65*min_lower, label = "Hold-Out Year")) + 
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "") 

player_plots <- (derrick_rose / kevin_durant / stephen_curry) + plot_annotation(title = "Posterior Predictive Production Curves", tag_levels = list(c("Derrick Rose", "Kevin Durant", "Stephen Curry"))) 


ggsave("test_plot.png", player_plots)