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


source("data_analysis/diagnostics_utils.r")

options(expressions = 500000)  # raise R call-stack limit for deep dendrogram traversal

args             <- commandArgs(trailingOnly = TRUE)
model_dir        <- if (length(args) >= 1) args[1] else stop("Usage: Rscript model_diagnostics.r <model_dir> [validation_year]")
validation_year  <- if (length(args) >= 2) as.integer(args[2]) else 2021L
min_minutes_threshold <- 100L   # discard player-seasons with fewer than this many minutes




posterior_data <- read_parquet(file.path(model_dir, "posterior_ar.parquet")) |>
  mutate(value = if_else(metric == "pct_minutes", value * 48, value))
conditional_parquet_path <- file.path(model_dir, "posterior_ar_conditional.parquet")
has_conditional <- file.exists(conditional_parquet_path)
if (has_conditional) {
  posterior_conditional_data <- read_parquet(conditional_parquet_path) |>
    mutate(value = if_else(metric == "pct_minutes", value * 48, value)) |>
    rename(value_conditional = value)
}
age_min   <- min(posterior_data$age)
age_max   <- max(posterior_data$age)
fake_data <- data.frame(age = age_min:age_max, name = "No Name", id = "99999999",
                        year = seq(2000, 2000 + age_max - age_min))
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


empirical_player_plt <- injury_data |> filter(name %in% c("Kobe Bryant", "Dwight Howard", "LeBron James")) |> mutate(metric = toupper(metric),
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
                              legend.title = element_blank()) +
                              scale_colour_brewer(palette = "Set1", labels = plot_name)

ggsave("model_output/model_plots/empirical_production_player.png", empirical_player_plt)    

empirical_plt <- injury_data |> mutate(metric = toupper(metric),
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


plots_dir        <- file.path(model_dir, "plots")
dir.create(file.path(plots_dir, "peaks",        "mcmc", "trace"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "coverage"),                      recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "latent_space", "map"),           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "latent_space", "mcmc"),          recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "latent_space", "tables"),        recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "player_plots", "mcmc"),          recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "mcmc"),                          recursive = TRUE, showWarnings = FALSE)

# posterior_data already loaded above (hoisted for age_min/age_max derivation)
print("loaded the posterior data")

posterior_mu_data    <- read_parquet(file.path(model_dir, "posterior_mu_ar.parquet"))
posterior_peaks      <- read_parquet_if_exists(file.path(model_dir, "posterior_peaks_ar.parquet"))
posterior_peak_vals  <- read_parquet_if_exists(file.path(model_dir, "posterior_peak_vals_ar.parquet"))
latent_space         <- read_parquet_if_exists(file.path(model_dir, "latent_space.parquet"))
injury_global_offset <- read_parquet_if_exists(file.path(model_dir, "posterior_injury_global_offset.parquet"))
injury_prior_mean    <- read_parquet_if_exists(file.path(model_dir, "posterior_injury_prior_mean.parquet"))
phi_X                <- read_parquet_if_exists(file.path(model_dir, "phi_X.parquet"))
third_deriv              <- read_parquet_if_exists(file.path(model_dir, "posterior_third_deriv_ar.parquet"))
log_posterior            <- read_parquet_if_exists(file.path(model_dir, "log_posterior.parquet"))
posterior_metric_ll      <- read_parquet_if_exists(file.path(model_dir, "posterior_metric_log_loss.parquet"))

posterior_retirement_data <- open_dataset(file.path(model_dir, "posterior_exit_age_sample.parquet")) |>
  filter(measure == "exit_age_sample", scenario == "observed", exit_censored == 0) |>
  select(player, value, observed_exit_age) |>
  collect()

posterior_survival_data <- open_dataset(file.path(model_dir, "posterior_exit_survival.parquet")) |>
  filter(measure == "exit_survival", scenario == "observed") |>
  select(player, age, value, observed_exit_age, exit_censored) |>
  collect()

posterior_latent_X <- if (file.exists(file.path(model_dir, "posterior_latent_X.parquet"))) {
  open_dataset(file.path(model_dir, "posterior_latent_X.parquet")) |>
    select(id, name, chain, sample, starts_with("Dim")) |>
    collect()
} else NULL

posterior_latent_X_peak_age <- read_parquet_if_exists(file.path(model_dir, "posterior_latent_X_peak_age.parquet"))
posterior_latent_X_peak_value <- read_parquet_if_exists(file.path(model_dir, "posterior_latent_X_peak_value.parquet"))
phi_X_peak_age <- read_parquet_if_exists(file.path(model_dir, "phi_X_peak_age.parquet"))
phi_X_peak_value <- read_parquet_if_exists(file.path(model_dir, "phi_X_peak_value.parquet"))

curvature_post_paths <- list.files(
  model_dir,
  pattern = "^posterior_latent_X_curvature_m[0-9]+\\.parquet$",
  full.names = TRUE
)


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
# ggsave("model_output/model_plots/peaks/mcmc/nba_convex_tvlinearlvm_ar_max_cluster.png", cluster_curves_plt) 



if (!is.null(posterior_peaks)) {

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
  geom_text_repel(aes(label = plot_name(name)),
                  color = "black",
                  fontface = "bold",
                  max.overlaps = 5) + theme_bw() + 
  scale_colour_brewer(palette = "Set1") + ggtitle("Posterior Mean of Peak OBPM Age, Value for 2016 Draft Class") + labs(x = "Peak Age", color = "Position Group", y = "Peak OBPM Value")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peak_class_2018.png"), peak_2019_class)



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




if (!is.null(latent_space)) {

peaks_players <- peaks_plt_df %>% group_by(metric, player) %>% summarize(value = mean(value)) %>% ungroup() %>% pivot_wider(names_from = metric, values_from = value) %>% inner_join(latent_space %>% filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) %>% select(id), by = c("player" = "id"))

peaks_pca <- prcomp(peaks_players  %>% select(-c(player)) %>% data.matrix() , scale. = TRUE, center = TRUE)

peaks_pca_df <- tibble(PC1 = peaks_pca$x[,1], PC2 = peaks_pca$x[,2], id = peaks_players$player) %>% inner_join(latent_space %>% select(id, name, position_group, minutes))

max_range <- max(abs(range(peaks_pca_df$PC1)),
                 abs(range(peaks_pca_df$PC2 )))



tops <- c(peaks_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(10))
bottoms <- c(peaks_pca_df %>% arrange(PC1) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(PC2) %>% pull(name) %>% head(10))

pca_outlier_names <- unique(c(tops,bottoms, posterior_plot_names))

# Extract loadings
loadings <- as.data.frame(peaks_pca$rotation[, 1:2])
loadings$metric <- rownames(loadings)
loadings_mat <- loadings %>% select(-c(metric)) %>% as.matrix()
k_dist <- kNNdist(loadings_mat, k = 2)
eps_grid <- quantile(k_dist, probs = seq(0.3, 0.95, length.out = 30), na.rm = TRUE)
target_k <- 3L

db_candidates <- lapply(eps_grid, function(eps_val) {
  db_try <- dbscan(loadings_mat, eps = eps_val, minPts = 2)
  list(
    db         = db_try,
    n_clusters = length(setdiff(unique(db_try$cluster), 0L)),
    n_noise    = sum(db_try$cluster == 0L)
  )
})

# Prefer exact target_k, then minimise noise; break ties by choosing larger eps
exact_3 <- Filter(function(x) x$n_clusters == target_k, db_candidates)
if (length(exact_3) > 0) {
  best <- exact_3[[which.min(sapply(exact_3, `[[`, "n_noise"))]]
} else {
  # Fall back: pick the eps closest to target_k
  diffs <- sapply(db_candidates, function(x) abs(x$n_clusters - target_k))
  best  <- db_candidates[[which.min(diffs)]]
}
db <- best$db
loadings$metric_group <- as.factor(db$cluster)

loadings_summary <- loadings |>
  group_by(metric_group) |>
  summarize(mean_pc1 = mean(PC1), mean_pc2 = mean(PC2), .groups = "drop")

pc1_pos_group <- loadings_summary |> slice_max(mean_pc1, n = 1) |> pull(metric_group)
pc1_neg_group <- loadings_summary |> slice_min(mean_pc1, n = 1) |> pull(metric_group)
pc2_pos_group <- loadings_summary |> slice_max(mean_pc2, n = 1) |> pull(metric_group)
pc2_neg_group <- loadings_summary |> slice_min(mean_pc2, n = 1) |> pull(metric_group)

pc1_pos_label <- glue("Later cluster {pc1_pos_group} metric peak")
pc1_neg_label <- glue("Earlier cluster {pc1_neg_group} metric peak")
pc2_pos_label <- glue("Later cluster {pc2_pos_group} metric peak")
pc2_neg_label <- glue("Earlier cluster {pc2_neg_group} metric peak")

avg_minutes_df <- data |>
  group_by(id) |>
  summarize(avg_minutes = mean(minutes, na.rm = TRUE), .groups = "drop")

peaks_pca_df <- peaks_pca_df |>
  left_join(avg_minutes_df, by = c("id" = "id"))

peaks_pca_plot  <-filter(peaks_pca_df, name %in% pca_outlier_names, avg_minutes > 2000) %>% ggplot(aes(x = PC1, y = PC2)) +
  geom_text_repel(
    aes(label = plot_name(name), x = PC1, y = PC2),
    size = 3,
    fontface = "bold",
    max.overlaps = 20,
    inherit.aes = FALSE
  ) +
  coord_fixed(clip = "off") +
  xlim(-max_range, max_range) +
  ylim(-max_range, max_range) +
  geom_segment(
    aes(x = -max_range, xend = max_range, y = 0, yend = 0),
    arrow = arrow(ends = "both", length = grid::unit(0.12, "inches")),
    color = "black",
    linewidth = 0.4
  ) +
  geom_segment(
    aes(x = 0, xend = 0, y = -max_range, yend = max_range),
    arrow = arrow(ends = "both", length = grid::unit(0.12, "inches")),
    color = "black",
    linewidth = 0.4
  ) +
  annotate("text", x = max_range * 0.95, y = 0, label = pc1_pos_label, hjust = 0, vjust = -0.5, size = 3) +
  annotate("text", x = -max_range * 0.95, y = 0, label = pc1_neg_label, hjust = 1, vjust = -0.5, size = 3) +
  annotate("text", x = 0, y = max_range * 0.95, label = pc2_pos_label, hjust = -0.1, vjust = 0, size = 3) +
  annotate("text", x = 0, y = -max_range * 0.95, label = pc2_neg_label, hjust = -0.1, vjust = 1, size = 3) +
  theme_bw() +
  theme(panel.grid = element_blank(), plot.margin = margin(10, 60, 10, 60)) +
  ggtitle("PCA Visualization of Learned Metric Peaks") +
  labs(x = "PC 1", y = "PC 2")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_pca.png"), peaks_pca_plot)
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
ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_pca_loadings.png"), peaks_pca_loadings_plt)


peak_vals_players <- peak_vals_plt_df %>% group_by(metric, player) %>% summarize(value = mean(value)) %>% ungroup() %>% pivot_wider(names_from = metric, values_from = value) %>% inner_join(latent_space %>% filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) %>% select(id), by = c("player" = "id"))

peak_vals_pca <- prcomp(peak_vals_players  %>% select(-c(player)) %>% data.matrix() , scale. = TRUE, center = TRUE)

peak_vals_pca_df <- tibble(PC1 = peak_vals_pca$x[,1], PC2 = peak_vals_pca$x[,2], id = peak_vals_players$player) %>% inner_join(latent_space %>% select(id, name, position_group, minutes))

tops <- c(peak_vals_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(10), peak_vals_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(10))
bottoms <- c(peak_vals_pca_df %>% arrange(PC1) %>% pull(name) %>% head(10), peak_vals_pca_df %>% arrange(PC2) %>% pull(name) %>% head(10))

pca_outlier_names <- unique(c(tops,bottoms, posterior_plot_names))

peak_vals_pca_plot  <-  peak_vals_pca_df %>% ggplot(aes(x = PC1, y = PC2)) +  geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(data = filter(peak_vals_pca_df, name %in% posterior_plot_names), 
                      aes(label = plot_name(name), x = PC1, y = PC2),
                      size = 3,
                      fontface = "bold",
                      max.overlaps = 20,
                      inherit.aes = FALSE) + 
  theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("PCA Visualization of Learned Metric Peak Values") + labs(x = "PC 1", y = "PC 2", color = "Position Group", alpha = "Minutes")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peak_vals_pca.png"), peak_vals_pca_plot)

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
ggsave(file.path(plots_dir, "peaks", "mcmc", "peak_vals_pca_loadings.png"), peak_vals_pca_loadings_plt)


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


ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_skew.png"), skew_plt)


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



ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks.png"), peaks_plt)

if (!is.null(third_deriv)) {

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


ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_third_deriv.png"), third_deriv_plt)



curve_third_deriv_plt <- ggplot(third_deriv |> filter(metric == "obpm") |> mutate(quantile = cut(value, breaks = c(-Inf, -0.05, 0.05, Inf),
                                                                     labels = c("Left-Skew Symmetry", "Symmetric", "Right-Skew Symmetry"))) |> group_by(quantile) |> slice_sample(n = 5) |> ungroup() |> group_by(metric, sample,chain, player) |> select(-value) |> 
                                inner_join(posterior_mu_data) |> mutate(value = value - value[age == 18]) , aes(x = age, y = value, group = interaction(chain, sample, player, metric), color = quantile)) + geom_line(alpha = .9)  + theme_bw() + scale_fill_brewer(palette = "Set1") +
                                labs(x = "Age", y = "Latent Curve Value", color = "Skew Type") + ggtitle("Illustration of Third Derivative Influence on Latent Curve Symmetry")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_third_deriv_curves.png"), curve_third_deriv_plt)

} # end if (!is.null(third_deriv))

} # end if (!is.null(latent_space))

posterior_data <- posterior_data  |>
                                    inner_join(posterior_peaks |>
                                    rename(peak_age = value), by = c("player", "chain", "sample", "metric"))

} # end if (!is.null(posterior_peaks))

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

# Join per-season minutes so we can filter low-minutes player-seasons out of
# coverage and bias computations.
player_season_minutes <- data |>
  filter(!is.na(minutes)) |>
  select(id, age, minutes)

joined_data <- joined_data |>
  left_join(player_season_minutes, by = c("player" = "id", "age"))

# Attach split label — use explicit holdout mask when available (scheme variants),
# fall back to year-based split for base MCMC models.
holdout_csv <- file.path(model_dir, "holdout_indices.csv")
if (file.exists(holdout_csv)) {
  holdout_idx <- read.csv(holdout_csv) |> mutate(split = "holdout")
  joined_data <- joined_data |>
    left_join(holdout_idx, by = c("player", "age")) |>
    mutate(split = replace_na(split, "train"))
  # years of training data per player = seasons not in holdout mask
  train_years_df <- data |>
    anti_join(holdout_idx, by = c("id" = "player", "age" = "age")) |>
    group_by(id) |>
    summarize(years_played = n(), .groups = "drop")
  # holdout flag for exit-age coverage: player appears in holdout mask
  holdout_players <- holdout_idx |> distinct(player) |> mutate(exit_split = "holdout")
} else {
  joined_data <- joined_data |>
    mutate(split = if_else(year > validation_year, "holdout", "train"))
  train_years_df <- data |>
    filter(year <= validation_year) |>
    group_by(id) |>
    summarize(years_played = n(), .groups = "drop")
  holdout_players <- NULL
}

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
                      theme_bw() + scale_fill_brewer(palette = "Set1") +
                      ggtitle("Per Metric Coverage") +
                      labs(x = NULL, fill = "Coverage Type") +
                      theme(axis.text.x = element_blank())
ggsave(file.path(plots_dir, "coverage", "coverage_basic.png"), coverage_plt_basic)

latex_tbl <- coverage_base_tbl |>
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

  write.csv(bind_rows(ll_intervals_val, ll_intervals_is),
            file.path(plots_dir, "coverage", "coverage_log_loss_intervals.csv"),
            row.names = FALSE)
}

coverage_plt_yearly <- validation_coverage_df |> group_by(metric, year) |> summarize(Coverage = mean(validation_coverage, na.rm = TRUE), .groups = "drop") |> 
                        bind_rows(joined_data %>% filter(split == "holdout" & metric == "retirement") %>% mutate(metric = toupper(metric)) %>% filter(!is.na(obs_value)) %>% group_by(metric, year) %>% summarize(Coverage = mean(if_else(obs_value == 1, value, 1 - value), na.rm = TRUE), .groups = "drop")) |>
                        bind_rows(exit_age_coverage_df %>% filter(exit_split == "holdout") %>% mutate(year = observed_exit_year, Coverage = as.numeric(exit_age_coverage)) %>% group_by(metric, year) %>% summarize(Coverage = mean(Coverage, na.rm = TRUE), .groups = "drop")) |>
                        ggplot(aes(x = year, y = Coverage)) +
                        geom_col() + facet_wrap(~metric, scales = "free_y") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Time Horizon") +xlab("Year")
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
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric Validation Coverage by Years of Training Data Available") + xlab("Years Played")
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
                        theme_bw() + scale_colour_brewer(palette = "Set1") + ggtitle("Per Metric In-Sample Coverage by Years of Training Data Available") + xlab("Years Played")
ggsave(file.path(plots_dir, "coverage", "coverage_in_sample_minutes.png"), coverage_plt_minutes)




# ── Injury effect decomposition: global offset + type-specific ───────────────
if (!is.null(injury_global_offset) && !is.null(injury_prior_mean)) {
  dir.create(file.path(plots_dir, "injury"), recursive = TRUE, showWarnings = FALSE)

  rename_injury_metrics <- function(df) {
    df |> mutate(metric = toupper(metric),
                 metric = case_when(metric == "GAMES"       ~ "GP%",
                                    metric == "FG2M"        ~ "FG2%",
                                    metric == "FG3M"        ~ "FG3%",
                                    metric == "FTM"         ~ "FT%",
                                    metric == "PCT_MINUTES" ~ "MPG",
                                    metric == "EXIT_HAZARD" ~ "Exit Hazard",
                                    metric == "EXIT_SCALE"  ~ "Exit Scale",
                                    .default = metric))
  }

  # Global offset: one row per (chain, sample, metric) — HDI per metric
  global_summary <- injury_global_offset |>
    rename_injury_metrics() |>
    group_by(metric) |>
    summarize(
      mean  = mean(value, na.rm = TRUE),
      lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
      .groups = "drop"
    ) |>
    mutate(metric = fct_reorder(metric, mean))

  global_plt <- global_summary |>
    ggplot(aes(x = mean, y = metric)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.3) +
    geom_point(size = 2) +
    labs(title = "Global injury offset per metric (95% HDI)",
         subtitle = "Applied uniformly to all injury types during post-injury seasons",
         x = "Effect on metric (model units)", y = NULL) +
    theme_bw()
  ggsave(file.path(plots_dir, "injury", "injury_global_offset.png"), global_plt,
         width = 8, height = 7)

  # Type-specific component: one row per (chain, sample, metric, injury_type)
  # filter to performance metrics only (exclude exit_hazard/exit_scale — those bake in global already)
  type_summary <- injury_prior_mean |>
    filter(!metric %in% c("exit_hazard", "exit_scale")) |>
    rename_injury_metrics() |>
    group_by(metric, injury_type) |>
    summarize(
      mean  = mean(value, na.rm = TRUE),
      lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
      upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
      .groups = "drop"
    )

  type_plt <- type_summary |>
    ggplot(aes(x = injury_type, y = mean, colour = injury_type)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3) +
    geom_point(size = 2) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_colour_brewer(palette = "Set1") +
    labs(title = "Type-specific injury effect per metric (95% HDI)",
         subtitle = "Factor-model component — incremental over global offset",
         x = NULL, y = "Effect (model units)", colour = "Injury type") +
    theme_bw() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          legend.position = "bottom")
  ggsave(file.path(plots_dir, "injury", "injury_type_specific.png"), type_plt,
         width = 14, height = 10)

  # Survival-specific: exit_hazard and exit_scale global + type-specific side by side
  surv_metrics <- c("Exit Hazard", "Exit Scale")
  surv_global <- global_summary |> filter(metric %in% surv_metrics)
  surv_type   <- injury_prior_mean |>
    filter(metric %in% c("exit_hazard", "exit_scale")) |>
    rename_injury_metrics() |>
    group_by(metric, injury_type) |>
    summarize(mean  = mean(value, na.rm = TRUE),
              lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
              upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
              .groups = "drop")

  if (nrow(surv_global) > 0 && nrow(surv_type) > 0) {
    surv_type_plt <- surv_type |>
      ggplot(aes(x = injury_type, y = mean, colour = injury_type)) +
      geom_hline(data = surv_global, aes(yintercept = mean), linetype = "dashed") +
      geom_hline(yintercept = 0, colour = "grey70") +
      geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3) +
      geom_point(size = 2) +
      facet_wrap(~ metric, scales = "free_y") +
      scale_colour_brewer(palette = "Set1") +
      labs(title = "Survival injury effects: global (dashed) + type-specific (95% HDI)",
           x = NULL, y = "Effect (log-hazard / log-scale units)", colour = "Injury type") +
      theme_bw() +
      theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
            legend.position = "bottom")
    ggsave(file.path(plots_dir, "injury", "injury_survival_effects.png"), surv_type_plt,
           width = 10, height = 6)
  }
}
plot_posterior <- function(grouped_data_set, hold_out_year, plot_obs = TRUE) {
  group_name <- unique(grouped_data_set$name)

  plt <- grouped_data_set |>
    ggplot(aes(x = age)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "gray", alpha = 0.4) +
    geom_line(aes(x = age, y = posterior_mean)) +
    geom_line(aes(x = age, y = mu), color = "#4DAF4AFF", linewidth = 1) +
    facet_wrap(~ metric, scales = "free_y") + theme_bw() +
    labs(x = "Age", y = "Metric Value") +
    ggtitle(paste("Posterior Predictive Career Trajectory:", group_name))

  # Observed value points for non-survival metrics only
  if (plot_obs) {
    non_surv <- filter(grouped_data_set, metric != "EXIT_SURVIVAL")
    if (nrow(non_surv) > 0)
      plt <- plt + geom_point(data = non_surv, aes(x = age, y = obs_value), color = "black")
  }

  # Survival panel: vertical line at exit age (solid = retired, dashed = censored)
  surv_meta <- grouped_data_set |>
    filter(metric == "EXIT_SURVIVAL", !is.na(observed_exit_age)) |>
    slice(1)
  if (nrow(surv_meta) > 0) {
    plt <- plt + geom_vline(
      data = surv_meta |> select(metric, observed_exit_age, exit_censored),
      aes(xintercept = observed_exit_age),
      linetype = if_else(surv_meta$exit_censored == 1L, "dashed", "solid"),
      color = "black"
    )
  }

  return(plt)
}


plot_posterior_mu_spaghetti <- function(grouped_data_set){
  group_name <- unique(grouped_data_set$name)

  plt <- grouped_data_set |>
    ggplot(aes(x = age)) +
    geom_line(aes(x = age, y = mu, group = interaction(chain, sample), color = factor(chain)), alpha = .2) +
    scale_color_brewer(palette = "Set1", name = "Chain") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_line(aes(x = age, y = posterior_mean)) +
    facet_wrap(~ metric, scales = "free_y") + theme_bw() +
    labs(x = "Age", y = "Metric Value") +
    ggtitle(paste("Posterior Latent Career Trajectory:", group_name))
  return(plt)
}
player_age_year_map <- data |> 
  group_by(id) |> 
  summarize(base_year = min(year, na.rm = TRUE),
            base_age = min(age, na.rm = TRUE),
            .groups = "drop")

metric_plot_df <- joined_data |> filter(metric != "retirement") |>
  group_by(metric, player, age) |> 
  summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
            upper = HDInterval::hdi(value, credMass = 0.95)["upper"], 
            obs_value = first(obs_value), 
            year = min(year), 
            posterior_mean = mean(value, na.rm = TRUE),
            .groups = "drop") |>
  inner_join(posterior_mu_data |> group_by(metric, player, age) |> summarize(mu = mean(value), .groups = "drop") |> 
               mutate(mu = case_when(metric %in% c("fg2m", "ftm", "games", "fg3m", "retirement") ~ plogis(mu),
                                     metric %in% c("obpm", "dbpm") ~ mu, 
                                     metric %in% c("pct_minutes") ~ plogis(mu) * 48,
                                     metric %in% c("usg") ~ plogis(mu),
                                     .default = exp(mu) * 36))) |>
  mutate(metric = toupper(metric),
         metric = case_when(metric == "GAMES" ~ "GP%",
                            metric == "FG2M" ~ "FG2%",
                            metric == "FG3M" ~ "FG3%",
                            metric == "FTM" ~ "FT%",
                            metric == "PCT_MINUTES" ~ "MPG",
                            .default = metric))

survival_plot_df <- posterior_survival_data |>
  filter(!is.na(value)) |>
  group_by(player, age) |>
  summarize(lower = HDInterval::hdi(value, credMass = 0.95)["lower"],
            upper = HDInterval::hdi(value, credMass = 0.95)["upper"],
            posterior_mean = mean(value, na.rm = TRUE),
            observed_exit_age = first(observed_exit_age),
            exit_censored = first(exit_censored),
            .groups = "drop") |>
  inner_join(player_age_year_map, by = c("player" = "id")) |>
  mutate(
    metric = "EXIT_SURVIVAL",
    obs_value = case_when(
      age <= observed_exit_age ~ 1,
      exit_censored == 1 ~ NA_real_,
      TRUE ~ 0
    ),
    year = base_year + (age - base_age),
    mu = posterior_mean
  ) |>
  select(metric, player, age, lower, upper, obs_value, year, posterior_mean, mu, observed_exit_age, exit_censored)

plots_list <- bind_rows(metric_plot_df, survival_plot_df) |>
  inner_join(data |> distinct(id, name) |> filter(name %in% posterior_plot_names), by = c("player" = "id")) %>%
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_posterior(.x, 2021)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = file.path(plots_dir, "player_plots", "mcmc", glue("{name}.png")),
      plot = plt
    )
    })


plots_list <- joined_data |> filter(metric != "retirement") |>
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
                    .default = metric)) |> inner_join(data |> distinct(id, name) |> filter(name %in% posterior_plot_names), by = c("player" = "id")) %>%
  group_by(player) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plot_posterior_mu_spaghetti(.x)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = file.path(plots_dir, "player_plots", "mcmc", glue("{name}_spaghetti.png")),
      plot = plt
    )
    })
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
                    .default = metric)) |> filter(metric %in% c("GP%", "FTA", "OBPM")) |> inner_join(data |> distinct(id, name) |> filter(name %in% c("Stephen Curry", "Kevin Durant", "Derrick Rose")), by = c("player" = "id"))  |>
            inner_join(injury_summary)        




label_df <- player_plot_df |> group_by(metric, player) |> summarize(max_upper = max(upper), min_lower = min(lower), first_major_injury = first(first_major_injury), age_of_holdout = first(age_of_holdout),
                                                                    age_of_injury = first(age_of_injury), name = first(name)) |> ungroup()
derrick_rose <- player_plot_df |> filter(name == "Derrick Rose") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_text(data = label_df |> filter(name == "Derrick Rose"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "")

kevin_durant <- player_plot_df |> filter(name == "Kevin Durant") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_text(data = label_df |> filter(name == "Kevin Durant"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "")

stephen_curry <- player_plot_df |> filter(name == "Stephen Curry") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_text(data = label_df |> filter(name == "Stephen Curry"),
      size = 3,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw() +
    labs(x = "Age", y = "")

player_plots <- (derrick_rose / kevin_durant / stephen_curry) + plot_annotation(title = "Posterior Predictive Production Curves", tag_levels = list(c("Derrick Rose", "Kevin Durant", "Stephen Curry")))

ggsave(file.path(plots_dir, "player_plots", "mcmc", "KD_Rose_Curry_comparison.png"),
       player_plots, width = 14, height = 18)



### Causal Gap Plot: ATT by time-since-treatment (Injury vs Non-Injured control)

# # Treatment onset: injury age for injured players
# injury_age_tbl <- injury_data |>
#   filter(!is.na(first_major_injury), injury_period == "post-injury") |>
#   select(id, age, first_major_injury) |>
#   distinct() |>
#   group_by(id) |>
#   summarize(
#     injury_age         = min(age, na.rm = TRUE),
#     first_major_injury = first(first_major_injury),
#     .groups = "drop"
#   )

# # Treatment onset: hold-out year (2021) for non-injured players
# noninjured_ref_tbl <- injury_data |>
#   filter(is.na(first_major_injury), year == 2021) |>
#   select(id, age) |>
#   distinct() |>
#   rename(ref_age = age)

# # ATT = obs_value - posterior_mean, centred at treatment onset.
# # All pre-treatment time points (years_since < 0) included for all players
# # as a parallel-trends / placebo check.
# # obs_value is constant across samples so mean(obs_value - value) = obs_value - mean(value).
# injured_gap <- joined_data |>
#   inner_join(injury_age_tbl |> select(id, injury_age), by = c("player" = "id")) |>
#   filter(
#     first_major_injury %in% c("ACL", "Achilles", "Hip", "Back/Spine",
#                                "Patellar Tendon", "Quad Tendon",
#                                "Foot Fracture", "Lower Body Fracture", "Meniscus"),
#     !is.na(obs_value), is.finite(value)
#   ) |>
#   mutate(years_since = age - injury_age) |>
#   group_by(player, metric, years_since, first_major_injury) |>
#   summarise(gap = first(obs_value) - mean(value), .groups = "drop")

# non_injured_gap <- joined_data |>
#   inner_join(noninjured_ref_tbl, by = c("player" = "id")) |>
#   filter(!is.na(obs_value), is.finite(value)) |>
#   mutate(years_since = age - ref_age) |>
#   group_by(player, metric, years_since) |>
#   summarise(gap = first(obs_value) - mean(value), .groups = "drop")

# plot_causal_gap <- function(injury_name) {
#   df_injured <- injured_gap |> filter(first_major_injury == injury_name)

#   median_injured <- df_injured |>
#     group_by(metric, years_since) |>
#     summarise(gap = median(gap, na.rm = TRUE), .groups = "drop")

#   median_control <- non_injured_gap |>
#     group_by(metric, years_since) |>
#     summarise(gap = median(gap, na.rm = TRUE), .groups = "drop")

#   ggplot() +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.4) +
#     geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.4) +
#     # geom_line(
#     #   data = non_injured_gap,
#     #   aes(x = years_since, y = gap, group = player),
#     #   color = "grey60", alpha = 0.08
#     # ) +
#     # geom_line(
#     #   data = df_injured,
#     #   aes(x = years_since, y = gap, group = player),
#     #   color = "red", alpha = 0.12
#     # ) +
#     geom_line(
#       data = median_control,
#       aes(x = years_since, y = gap),
#       color = "blue", linewidth = 1.0
#     ) +
#     geom_line(
#       data = median_injured,
#       aes(x = years_since, y = gap),
#       color = "black", linewidth = 1.0
#     ) +
#     facet_wrap(~metric, scales = "free_y") +
#     theme_bw() +
#     labs(
#       x        = "Years Since Treatment Onset",
#       y        = "Observed − Posterior Mean (ATT)",
#       title    = glue("Causal Gap Plot: {injury_name} vs Non-Injured"),
#       subtitle = "Black: median injured  |  Blue: median non-injured  |  x < 0: pre-treatment (placebo check)"
#     )
# }

# unique(injured_gap$first_major_injury) |>
#   walk(function(injury_name) {
#     safe_name <- str_replace_all(injury_name, "[^A-Za-z0-9]+", "_")
#     plt <- plot_causal_gap(injury_name)
#     ggsave(
#       glue("model_output/model_plots/causal/gap_{safe_name}.png"),
#       plt,
#       width = 14,
#       height = 9
#     )
#   })

# # Combined: all injury types pooled — one line per player regardless of injury type
# all_injured_gap <- injured_gap |>
#   group_by(player, metric, years_since) |>
#   summarise(gap = mean(gap), .groups = "drop")

# median_all_injured <- all_injured_gap |>
#   group_by(metric, years_since) |>
#   summarise(gap = median(gap, na.rm = TRUE), .groups = "drop")

# median_all_control <- non_injured_gap |>
#   group_by(metric, years_since) |>
#   summarise(gap = median(gap, na.rm = TRUE), .groups = "drop")

# plt_all <- ggplot() +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.4) +
#   geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.4) +
#   # geom_line(
#   #   data = non_injured_gap,
#   #   aes(x = years_since, y = gap, group = player),
#   #   color = "grey60", alpha = 0.08
#   # ) +
#   # geom_line(
#   #   data = all_injured_gap,
#   #   aes(x = years_since, y = gap, group = player),
#   #   color = "red", alpha = 0.12
#   # ) +
#   geom_line(data = median_all_control, aes(x = years_since, y = gap),
#             color = "blue", linewidth = 1.0) +
#   geom_line(data = median_all_injured, aes(x = years_since, y = gap),
#             color = "black", linewidth = 1.0) +
#   facet_wrap(~metric, scales = "free_y") +
#   theme_bw() +
#   labs(
#     x        = "Years Since Treatment Onset",
#     y        = "Observed - Posterior Mean (ATT)",
#     title    = "Causal Gap Plot: All Injury Types vs Non-Injured",
#     subtitle = "Black: median injured  |  Blue: median non-injured  |  x < 0: pre-treatment (placebo check)"
#   )

# ggsave("model_output/model_plots/causal/gap_all_injuries.png", plt_all, width = 14, height = 9)

# # ATT grid: average over post-treatment period (years_since >= 0), per player first,
# # then summarise across players. Non-injured placebo is replicated into every
# # injury-type row so both CIs appear in the same facet cell.

# injured_post_player <- injured_gap |>
#   filter(years_since >= 0) |>
#   group_by(player, metric, first_major_injury) |>
#   summarise(gap = mean(gap, na.rm = TRUE), .groups = "drop")

# non_injured_post_player <- non_injured_gap |>
#   filter(years_since >= 0) |>
#   group_by(player, metric) |>
#   summarise(gap = mean(gap, na.rm = TRUE), .groups = "drop")

# ci_injured <- injured_post_player |>
#   group_by(metric, first_major_injury) |>
#   summarise(
#     mean_gap = mean(gap, na.rm = TRUE),
#     lower    = quantile(gap, 0.025, na.rm = TRUE),
#     upper    = quantile(gap, 0.975, na.rm = TRUE),
#     .groups  = "drop"
#   ) |>
#   mutate(group = "Injured")

# ci_non_injured <- non_injured_post_player |>
#   group_by(metric) |>
#   summarise(
#     mean_gap = mean(gap, na.rm = TRUE),
#     lower    = quantile(gap, 0.025, na.rm = TRUE),
#     upper    = quantile(gap, 0.975, na.rm = TRUE),
#     .groups  = "drop"
#   ) |>
#   mutate(group = "Non-Injured")

# # Replicate non-injured CI across all injury types for facet_grid alignment
# att_grid <- bind_rows(
#   ci_injured,
#   ci_non_injured |>
#     cross_join(ci_injured |> distinct(first_major_injury))
# )

# plt_att_grid <- ggplot(att_grid, aes(x = group, y = mean_gap, color = group)) +
#   geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3, color = "black") +
#   geom_pointrange(aes(ymin = lower, ymax = upper), size = 0.4, linewidth = 0.7) +
#   facet_grid(first_major_injury ~ metric, scales = "free_y") +
#   scale_color_manual(values = c("Injured" = "#E41A1C", "Non-Injured" = "grey40")) +
#   theme_bw() +
#   theme(
#     axis.text.x  = element_text(angle = 45, hjust = 1, size = 7),
#     strip.text.x = element_text(size = 6),
#     strip.text.y = element_text(size = 7),
#     legend.position = "none"
#   ) +
#   labs(
#     x        = NULL,
#     y        = "Mean ATT (Observed - Posterior Mean)",
#     title    = "ATT Summary: Injury Type × Metric",
#     subtitle = "Mean ± 95% CI across players, averaged over post-treatment period"
#   )

# ggsave("model_output/model_plots/causal/gap_att_grid.png", plt_att_grid, width = 24, height = 14)


# ##### CONDITIONAL ATT REGRESSION WITH POSTERIOR X UNCERTAINTY
# # For each posterior draw of X: fit gap ~ X + injury_type per metric.
# # Collecting coefficients across draws gives a posterior distribution
# # of the conditional ATT that propagates X estimation uncertainty.

# x_dim_cols <- names(posterior_latent_X)[startsWith(names(posterior_latent_X), "Dim")]

# # Fixed per-player gap (averaged over samples already) — response is the same
# # across regressions; only the X predictor matrix changes per draw
# gap_fixed <- bind_rows(
#   injured_post_player  |> rename(injury_type = first_major_injury),
#   non_injured_post_player |> mutate(injury_type = "Non-Injured")
# ) |>
#   mutate(injury_type = factor(injury_type, levels = c("Non-Injured",
#     "ACL", "Achilles", "Hip", "Back/Spine", "Patellar Tendon",
#     "Quad Tendon", "Foot Fracture", "Lower Body Fracture", "Meniscus")))

# injury_formula <- as.formula(
#   paste("gap ~", paste(x_dim_cols, collapse = " + "), "+ injury_type")
# )

# # Run one regression per (chain, sample) draw of X
# posterior_coefs <- posterior_latent_X |>
#   select(chain, sample, player = id, all_of(x_dim_cols)) |>
#   group_by(chain, sample) |>
#   group_modify(~ {
#     x_draw <- .x |> rename_with(~ x_dim_cols, all_of(x_dim_cols))
#     reg_data <- gap_fixed |>
#       inner_join(x_draw, by = "player")

#     reg_data |>
#       group_by(metric) |>
#       group_modify(~ {
#         fit <- lm(injury_formula, data = .x)
#         broom::tidy(fit, conf.int = FALSE) |>
#           filter(startsWith(term, "injury_type")) |>
#           mutate(term = str_remove(term, "^injury_type")) |>
#           select(injury_type = term, estimate)
#       }) |>
#       ungroup()
#   }) |>
#   ungroup()

# # Summarise posterior distribution of ATT per injury_type x metric
# att_posterior_summary <- posterior_coefs |>
#   group_by(metric, injury_type) |>
#   summarise(
#     mean_att = mean(estimate, na.rm = TRUE),
#     lower    = quantile(estimate, 0.025, na.rm = TRUE),
#     upper    = quantile(estimate, 0.975, na.rm = TRUE),
#     .groups  = "drop"
#   ) |>
#   mutate(
#     metric = toupper(metric),
#     metric = case_when(
#       metric == "GAMES"       ~ "GP%",
#       metric == "FG2M"        ~ "FG2%",
#       metric == "FG3M"        ~ "FG3%",
#       metric == "FTM"         ~ "FT%",
#       metric == "PCT_MINUTES" ~ "MPG",
#       .default = metric
#     ),
#     significant = if_else(sign(lower) == sign(upper), "p < 0.05", "n.s.")
#   )

# plt_att_posterior <- ggplot(att_posterior_summary,
#                             aes(x = mean_att, y = injury_type, color = significant)) +
#   geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3, color = "black") +
#   geom_pointrange(aes(xmin = lower, xmax = upper), size = 0.35, linewidth = 0.6) +
#   facet_wrap(~ metric, scales = "free_x") +
#   scale_color_manual(values = c("p < 0.05" = "#E41A1C", "n.s." = "grey50")) +
#   theme_bw() +
#   theme(legend.position = "bottom") +
#   labs(
#     x        = "Conditional ATT (gap | player archetype X)",
#     y        = NULL,
#     color    = NULL,
#     title    = "Injury Effect Conditional on Latent Player Type",
#     subtitle = "Posterior distribution over X draws: gap ~ X + injury_type  |  Reference = Non-Injured  |  95% CI"
#   )

# ggsave("model_output/model_plots/causal/gap_conditional_att.png",
#        plt_att_posterior, width = 20, height = 14)


# ##### CONDITIONAL ATT REGRESSION WITH POSTERIOR X + INJURY_TIME
# # Same as above but adds injury_time (pre-peak vs post-peak) as a covariate.
# # Both X and peak_age vary per (chain, sample), so we join everything upfront
# # and group by (chain, sample, metric) — avoids filtering inside the loop.

# # Per-player reference age: injury_age for injured, holdout age for non-injured
# ref_age_tbl <- bind_rows(
#   injury_age_tbl |> select(id, ref_age = injury_age),
#   noninjured_ref_tbl
# )

# # Per-draw gap at EVERY observed time point — gap varies per (chain, sample, player, metric, age).
# # joined_data already contains (chain, sample, player, metric, age, value, obs_value,
# # first_major_injury, peak_age). We normalise chain type here once.
# injury_levels <- c("No Injury", "ACL", "Achilles", "Hip", "Back/Spine",
#                    "Patellar Tendon", "Quad Tendon", "Foot Fracture",
#                    "Lower Body Fracture", "Meniscus")

# all_time_gap <- joined_data |>
#   filter(!is.na(obs_value), is.finite(value), metric != "retirement") |>
#   mutate(chain = as.integer(as.character(chain))) |>
#   left_join(ref_age_tbl, by = c("player" = "id")) |>
#   filter(!is.na(ref_age)) |>
#   mutate(
#     gap         = obs_value - value,
#     years_since = age - ref_age,
#     injury_type = case_when(
#       first_major_injury %in% injury_levels[-1] ~ first_major_injury,
#       is.na(first_major_injury)                 ~ "No Injury",
#       TRUE                                       ~ NA_character_
#     )
#   ) |>
#   filter(!is.na(injury_type), years_since >= 0) |>
#   mutate(injury_type = factor(injury_type, levels = injury_levels))

# # peak_age draws: one row per (chain, sample, player, metric)
# peak_age_draws <- joined_data |>
#   distinct(chain, sample, player, metric, peak_age) |>
#   mutate(chain = as.integer(as.character(chain)))

# # Formula now includes years_since to control for career-time trend across all time points
# injury_formula_with_time <- as.formula(
#   paste("gap ~", paste(dim_cols, collapse = " + "),
#         "+ injury_type + injury_time + years_since")
# )

# # Procrustes-aligned X draws — normalise chain to integer
# rotated_x_draws <- rotated_posterior_df |>
#   select(chain, sample, player = id, all_of(dim_cols)) |>
#   mutate(chain = as.integer(as.character(chain)))

# # Build full regression data: all time points × all draws.
# # X (per player-draw) fans out across each player's time points via the join.
# full_reg_data_with_time <- all_time_gap |>
#   select(chain, sample, player, metric, age, years_since, gap, injury_type, ref_age) |>
#   inner_join(rotated_x_draws, by = c("chain", "sample", "player")) |>
#   left_join(peak_age_draws,   by = c("chain", "sample", "player", "metric")) |>
#   mutate(
#     injury_time = factor(
#       if_else(ref_age < peak_age, "pre-peak", "post-peak"),
#       levels = c("post-peak", "pre-peak")
#     )
#   )

# # One regression per (chain, sample, metric) over all time points.
# # Captures injury_type, injury_time, years_since, and Dim coefficients.
# posterior_coefs_with_time <- full_reg_data_with_time |>
#   group_by(chain, sample, metric) |>
#   group_modify(~ {
#     fit <- lm(injury_formula_with_time, data = .x)
#     broom::tidy(fit, conf.int = FALSE) |>
#       filter(startsWith(term, "injury_type") | startsWith(term, "injury_time") |
#                startsWith(term, "Dim") | term == "years_since") |>
#       mutate(term = str_remove(term, "^injury_type|^injury_time")) |>
#       select(term, estimate)
#   }) |>
#   ungroup()

# # Summarise — injury_type terms only (injury_time, years_since, and Dim are controls)
# att_posterior_summary_with_time <- posterior_coefs_with_time |>
#   filter(!term %in% c("pre-peak", "post-peak", "years_since") & !startsWith(term, "Dim")) |>
#   rename(injury_type = term) |>
#   group_by(metric, injury_type) |>
#   summarise(
#     mean_att = mean(estimate, na.rm = TRUE),
#     lower    = quantile(estimate, 0.025, na.rm = TRUE),
#     upper    = quantile(estimate, 0.975, na.rm = TRUE),
#     .groups  = "drop"
#   ) |>
#   mutate(
#     metric = toupper(metric),
#     metric = case_when(
#       metric == "GAMES"       ~ "GP%",
#       metric == "FG2M"        ~ "FG2%",
#       metric == "FG3M"        ~ "FG3%",
#       metric == "FTM"         ~ "FT%",
#       metric == "PCT_MINUTES" ~ "MPG",
#       .default = metric
#     ),
#     significant = if_else(sign(lower) == sign(upper), "p < 0.05", "n.s.")
#   )

# plt_att_posterior_with_time <- ggplot(att_posterior_summary_with_time,
#                                       aes(x = mean_att, y = injury_type, color = significant)) +
#   geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3, color = "black") +
#   geom_pointrange(aes(xmin = lower, xmax = upper), size = 0.35, linewidth = 0.6) +
#   facet_wrap(~ metric, scales = "free_x") +
#   scale_color_manual(values = c("p < 0.05" = "#E41A1C", "n.s." = "grey50")) +
#   theme_bw() +
#   theme(legend.position = "bottom") +
#   labs(
#     x        = "Conditional ATT (gap | player archetype X + injury timing)",
#     y        = NULL,
#     color    = NULL,
#     title    = "Injury Effect Conditional on Latent Player Type + Injury Timing",
#     subtitle = "Per-draw, per-time-point gap ~ X + injury_type + injury_time + years_since  |  Reference = No Injury, post-peak  |  95% CI"
#   )

# ggsave("model_output/model_plots/causal/gap_conditional_att_with_peak_timing.png",
#        plt_att_posterior_with_time, width = 20, height = 14)


# ### PLOT: injury_time effect (pre-peak vs post-peak marginal coefficient)
# att_injury_time_summary <- posterior_coefs_with_time |>
#   filter(term == "pre-peak") |>
#   group_by(metric) |>
#   summarise(
#     mean_effect = mean(estimate, na.rm = TRUE),
#     lower       = quantile(estimate, 0.025, na.rm = TRUE),
#     upper       = quantile(estimate, 0.975, na.rm = TRUE),
#     .groups     = "drop"
#   ) |>
#   mutate(
#     metric = toupper(metric),
#     metric = case_when(
#       metric == "GAMES"       ~ "GP%",
#       metric == "FG2M"        ~ "FG2%",
#       metric == "FG3M"        ~ "FG3%",
#       metric == "FTM"         ~ "FT%",
#       metric == "PCT_MINUTES" ~ "MPG",
#       .default = metric
#     ),
#     significant = if_else(sign(lower) == sign(upper), "p < 0.05", "n.s.")
#   )

# plt_injury_time_effect <- ggplot(att_injury_time_summary,
#                                   aes(x = mean_effect, y = metric, color = significant)) +
#   geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3, color = "black") +
#   geom_pointrange(aes(xmin = lower, xmax = upper), size = 0.4, linewidth = 0.7) +
#   scale_color_manual(values = c("p < 0.05" = "#E41A1C", "n.s." = "grey50")) +
#   theme_bw() +
#   theme(legend.position = "bottom") +
#   labs(
#     x        = "Coefficient on injury_time (pre-peak vs post-peak)",
#     y        = "Metric",
#     color    = NULL,
#     title    = "Effect of Injury Timing on Post-Injury Gap",
#     subtitle = "Posterior 95% CI: positive = pre-peak injuries have larger gap  |  Reference = post-peak"
#   )

# ggsave("model_output/model_plots/causal/gap_injury_time_effect.png",
#        plt_injury_time_effect, width = 10, height = 6)


# ### PLOT: X influence on injury effect — PCA of aligned mean X, coloured by X @ β_X
# # Mean aligned X per player (averaged over all Procrustes-aligned draws)
# mean_X_aligned <- rotated_posterior_df |>
#   group_by(id) |>
#   summarise(across(all_of(dim_cols), mean), .groups = "drop") |>
#   left_join(phi_X |> select(id, name), by = "id")

# X_mat_aligned <- mean_X_aligned |> select(all_of(dim_cols)) |> as.matrix()
# pca_res       <- prcomp(X_mat_aligned, center = TRUE, scale. = FALSE)
# pct_var       <- round(100 * pca_res$sdev^2 / sum(pca_res$sdev^2), 1)

# pca_df <- as_tibble(pca_res$x[, 1:2]) |>
#   rename(PC1 = PC1, PC2 = PC2) |>
#   mutate(player = mean_X_aligned$id, name = mean_X_aligned$name)

# # Mean Dim coefficients per metric (averaged over draws) — consistent because X is aligned
# mean_dim_coefs <- posterior_coefs_with_time |>
#   filter(startsWith(term, "Dim")) |>
#   rename(dimension = term) |>
#   group_by(metric, dimension) |>
#   summarise(mean_coef = mean(estimate, na.rm = TRUE), .groups = "drop")

# # X @ β_X per player per metric using mean aligned X and mean β_X
# x_influence_df <- mean_X_aligned |>
#   select(player = id, all_of(dim_cols)) |>
#   pivot_longer(all_of(dim_cols), names_to = "dimension", values_to = "x_val") |>
#   inner_join(mean_dim_coefs, by = "dimension", relationship = "many-to-many") |>
#   group_by(player, metric) |>
#   summarise(x_effect = sum(x_val * mean_coef), .groups = "drop") |>
#   mutate(
#     metric = toupper(metric),
#     metric = case_when(
#       metric == "GAMES"       ~ "GP%",
#       metric == "FG2M"        ~ "FG2%",
#       metric == "FG3M"        ~ "FG3%",
#       metric == "FTM"         ~ "FT%",
#       metric == "PCT_MINUTES" ~ "MPG",
#       .default = metric
#     )
#   )

# pca_influence_df <- pca_df |>
#   inner_join(x_influence_df, by = "player") |>
#   group_by(metric) |>
#   mutate(x_effect_scaled = x_effect / max(abs(x_effect), na.rm = TRUE)) |>
#   ungroup()

# plt_x_influence <- ggplot(pca_influence_df, aes(x = PC1, y = PC2, color = x_effect_scaled)) +
#   geom_point(alpha = 0.85, size = 1.8) +
#   scale_color_gradient2(low = "#2166AC", mid = "#D9D9D9", high = "#D6604D", midpoint = 0,
#                         limits = c(-1, 1), name = "X-predicted\ngap\n(scaled)") +
#   facet_wrap(~ metric) +
#   theme_bw() +
#   theme(
#     strip.text      = element_text(size = 8),
#     legend.position = "right"
#   ) +
#   labs(
#     x        = glue("PC1 ({pct_var[1]}% var)"),
#     y        = glue("PC2 ({pct_var[2]}% var)"),
#     title    = "Latent Player Archetype vs X-Predicted Injury Gap",
#     subtitle = "PCA of Procrustes-aligned mean X; color = X @ β_X (archetype-driven gap contribution)"
#   )

# ggsave("model_output/model_plots/causal/gap_x_influence_pca.png",
#        plt_x_influence, width = 20, height = 14)
