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




posterior_data <- read_parquet(file.path(model_dir, "posterior_ar.parquet")) |>
  mutate(value = if_else(metric == "pct_minutes", value * 48, value))
data <- make_player_data(posterior_data)


player_corrs <- data |>
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

injury_data <- build_injury_data(data)

print("pivoted the original data")


empirical_player_plt <- injury_data |> filter(name %in% c("Kobe Bryant", "Dwight Howard", "LeBron James")) |> filter(metric %in% c("obpm", "pct_minutes", "fta")) |> mutate(metric = toupper(metric),
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
                              geom_smooth(method = "loess", se = FALSE) + facet_wrap(~metric, scales = "free_y") + theme_bw(base_size = 18) + scale_colour_brewer(palette = "Set1") +
                              ggtitle("An Empirical Production Curve Comparison by Metric") +xlab("Age") + ylab("Metric Value") +theme(legend.position = "bottom",
                              legend.justification = "center",
                              legend.title = element_blank()) +
                              scale_colour_brewer(palette = "Set1", labels = plot_name)

ggsave("model_output/model_plots/empirical_production_player.png", empirical_player_plt,
       width = 10, height = 4, dpi = 150)

empirical_plt <- injury_data |> filter(metric %in% c("obpm", "pct_minutes", "fta")) |> mutate(metric = toupper(metric),
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
                              geom_smooth(method = "loess", se = TRUE) + facet_wrap(~metric, scales = "free_y") + theme_bw(base_size = 18) + scale_colour_brewer(palette = "Set1") +
                              ggtitle("Empirical Production Curves by Metric") +xlab("Age") + ylab("Metric Value")
ggsave("model_output/model_plots/empirical_production.png", empirical_plt,
       width = 10, height = 4, dpi = 150)


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

posterior_exit_samples <- open_dataset(file.path(model_dir, "posterior_exit_age_sample.parquet")) |>
  filter(measure == "exit_age_sample", scenario == "observed",
         conditioning_label %in% c("entrance", "last_observed")) |>
  select(player, value, observed_exit_age, exit_censored, conditioning_label) |>
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

# cluster_curves_plt <- ggplot(obpm_curves_cluster, aes(x = age, y = mean_value, color = cluster)) + geom_line() + theme_bw(base_size = 14) +  scale_colour_brewer(palette = "Set1") + 
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
                  max.overlaps = 5) + theme_bw(base_size = 14) + 
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



# Label only the spatial extremes along each PC: these sit at the periphery and
# repel cleanly. The dense central cluster of stars is left unlabeled so player
# names do not overlap.
tops <- c(peaks_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(10))
bottoms <- c(peaks_pca_df %>% arrange(PC1) %>% pull(name) %>% head(10), peaks_pca_df %>% arrange(PC2) %>% pull(name) %>% head(10))

pca_outlier_names <- unique(c(tops, bottoms))

# Extract loadings
loadings <- as.data.frame(peaks_pca$rotation[, 1:2])
loadings$metric <- rownames(loadings)

# Fixed semantic metric clusters (replaces the prior DBSCAN grouping). The level
# order is chosen so scale_*_brewer("Set1") keeps the paper's colors:
# Skill = red, Both = blue, Athleticism = green. FG2% (two-point FG%) groups with
# "Both" alongside FG2A.
metric_group_map <- c(
  OREB = "Athleticism", STL = "Athleticism", BLK = "Athleticism",
  FTA  = "Athleticism", `GP%` = "Athleticism",
  `FT%` = "Skill", `FG3%` = "Skill", FG3A = "Skill",
  AST = "Both", TOV = "Both", DBPM = "Both", OBPM = "Both", MPG = "Both",
  USG = "Both", DREB = "Both", FG2A = "Both", `FG2%` = "Both"
)
loadings$metric_group <- factor(unname(metric_group_map[loadings$metric]),
                                levels = c("Skill", "Both", "Athleticism"))
if (any(is.na(loadings$metric_group)))
  warning("Metrics with no fixed cluster assignment: ",
          paste(loadings$metric[is.na(loadings$metric_group)], collapse = ", "))

# Axis interpretation for the direction labels. PC1 is a "general" peak-timing
# factor: essentially all metric loadings share one sign, so moving along PC1
# shifts every metric's peak age together (overall earlier vs. later peaking).
# PC2 is a contrast between defensive-impact/rebounding metrics and
# scoring-volume/usage metrics peaking at different relative ages (empirically it
# separates defensive bigs from high-usage scorers). Both axes are oriented from
# the data (PCA signs are arbitrary) so the labels stay correct if a sign flips
# across model runs.
pc1_later_is_pos <- mean(loadings$PC1) >= 0     # + direction = later-peaking?
# DBPM is the dominant PC2 loader; its sign marks the defense/rebounding end.
pc2_defreb_is_pos <- loadings$PC2[loadings$metric == "DBPM"] >= 0

later_lab    <- "Later-peaking\n(overall)"
earlier_lab  <- "Earlier-peaking\n(overall)"
defreb_lab   <- "Defense & rebounding\npeak later"
scoreuse_lab <- "Scoring & usage\npeak later"

pc1_pos_label <- if (pc1_later_is_pos) later_lab else earlier_lab
pc1_neg_label <- if (pc1_later_is_pos) earlier_lab else later_lab
pc2_pos_label <- if (pc2_defreb_is_pos) defreb_lab else scoreuse_lab
pc2_neg_label <- if (pc2_defreb_is_pos) scoreuse_lab else defreb_lab

avg_minutes_df <- data |>
  group_by(id) |>
  summarize(avg_minutes = mean(minutes, na.rm = TRUE), .groups = "drop")

peaks_pca_df <- peaks_pca_df |>
  left_join(avg_minutes_df, by = c("id" = "id"))

# Label a few representative players per quadrant: the 4 furthest from the
# origin (most characteristic peak-age profiles) among established players, so
# each region has a handful of well-separated names rather than a dense cloud.
peaks_pca_labels <- peaks_pca_df |>
  filter(avg_minutes > 1200) |>
  mutate(quadrant = interaction(PC1 >= 0, PC2 >= 0),
         dist = sqrt(PC1^2 + PC2^2)) |>
  group_by(quadrant) |>
  slice_max(dist, n = 4, with_ties = FALSE) |>
  ungroup() |>
  pull(name)

label_pos <- max_range * 0.99
peaks_pca_plot  <- filter(peaks_pca_df, name %in% peaks_pca_labels) %>% ggplot(aes(x = PC1, y = PC2)) +
  # Faint zero cross marks the quadrants (the bold arrowed axes were redundant
  # with the PC1/PC2 tick axes and have been removed).
  geom_hline(yintercept = 0, colour = "grey85", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey85", linewidth = 0.3) +
  geom_text_repel(
    aes(label = plot_name(name), x = PC1, y = PC2),
    size = 3.2,
    fontface = "bold",
    max.overlaps = Inf,
    min.segment.length = 0,
    box.padding = 0.4,
    inherit.aes = FALSE
  ) +
  coord_fixed(clip = "off") +
  xlim(-max_range, max_range) +
  ylim(-max_range, max_range) +
  # Direction labels sit inside the panel near each edge, justified inward.
  annotate("text", x = label_pos,  y = 0, label = pc1_pos_label, hjust = 1,   vjust = -0.4, size = 3.2, lineheight = 0.9, colour = "grey30", fontface = "italic") +
  annotate("text", x = -label_pos, y = 0, label = pc1_neg_label, hjust = 0,   vjust = -0.4, size = 3.2, lineheight = 0.9, colour = "grey30", fontface = "italic") +
  annotate("text", x = 0, y = label_pos,  label = pc2_pos_label, hjust = 0.5, vjust = 1,    size = 3.2, lineheight = 0.9, colour = "grey30", fontface = "italic") +
  annotate("text", x = 0, y = -label_pos, label = pc2_neg_label, hjust = 0.5, vjust = 0,    size = 3.2, lineheight = 0.9, colour = "grey30", fontface = "italic") +
  theme_bw(base_size = 14) +
  theme(panel.grid = element_blank(), plot.margin = margin(10, 14, 10, 14),
        plot.title = element_text(hjust = 0.5)) +
  ggtitle("Player Peak Ages") +
  labs(x = "PC 1", y = "PC 2")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_pca.png"), peaks_pca_plot, width = 6, height = 6)
peaks_pca_loadings_plt <- ggplot(loadings, aes(x = PC1, y = PC2)) +
  # geom_text_repel (not geom_text) so co-located metric labels separate instead of stacking.
  geom_text_repel(aes(label = metric, color = metric_group), size = 4,
                      fontface = "bold", show.legend = FALSE, max.overlaps = Inf,
                      min.segment.length = 0, box.padding = 0.35, seed = 1) +
  geom_point(aes(color = metric_group), alpha = 0) +
  theme_bw(base_size = 14) +
  guides(color = guide_legend(override.aes = list(shape = 16, alpha = 1))) +
  scale_colour_brewer(palette = "Set1") +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
  # Make equal scaling so 0,0 is visually centered
  coord_cartesian(xlim = c(-max(abs(loadings$PC1)), max(abs(loadings$PC1))),
                  ylim = c(-max(abs(loadings$PC2)), max(abs(loadings$PC2)))) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom") +
  ggtitle("Metric Loadings (Peak Age)") + labs(x = "PC 1", y = "PC 2", color = "Metric Group")
ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_pca_loadings.png"), peaks_pca_loadings_plt, width = 5, height = 5)


peak_vals_players <- peak_vals_plt_df %>% group_by(metric, player) %>% summarize(value = mean(value)) %>% ungroup() %>% pivot_wider(names_from = metric, values_from = value) %>% inner_join(latent_space %>% filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) %>% select(id), by = c("player" = "id"))

peak_vals_pca <- prcomp(peak_vals_players  %>% select(-c(player)) %>% data.matrix() , scale. = TRUE, center = TRUE)

peak_vals_pca_df <- tibble(PC1 = peak_vals_pca$x[,1], PC2 = peak_vals_pca$x[,2], id = peak_vals_players$player) %>% inner_join(latent_space %>% select(id, name, position_group, minutes))

# Label only the spatial extremes (periphery) so player names do not overlap.
tops <- c(peak_vals_pca_df %>% arrange(desc(PC1)) %>% pull(name) %>% head(5), peak_vals_pca_df %>% arrange(desc(PC2)) %>% pull(name) %>% head(5))
bottoms <- c(peak_vals_pca_df %>% arrange(PC1) %>% pull(name) %>% head(5), peak_vals_pca_df %>% arrange(PC2) %>% pull(name) %>% head(5))

pca_outlier_names <- unique(c(tops, bottoms))

peak_vals_pca_plot  <-  peak_vals_pca_df %>% ggplot(aes(x = PC1, y = PC2)) +  geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(data = filter(peak_vals_pca_df, name %in% pca_outlier_names),
                      aes(label = plot_name(name), x = PC1, y = PC2),
                      size = 3,
                      fontface = "bold",
                      max.overlaps = Inf,
                      min.segment.length = 0,
                      box.padding = 0.4,
                      seed = 1,
                      inherit.aes = FALSE) +
  theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") +
  # Legend to the bottom so the scatter uses the full panel width (it was crushed to half-width before).
  guides(color = guide_legend(nrow = 1, override.aes = list(alpha = 1)), alpha = guide_legend(nrow = 1)) +
  theme(legend.position = "bottom", legend.box = "horizontal",
        plot.title = element_text(hjust = 0.5)) +
  ggtitle("Player Peak Values") + labs(x = "PC 1", y = "PC 2", color = "Position Group", alpha = "Minutes")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peak_vals_pca.png"), peak_vals_pca_plot, width = 4.8, height = 6)

# Extract loadings
loadings_vals <- as.data.frame(peak_vals_pca$rotation[, 1:2])
loadings_vals$metric <- rownames(loadings_vals)
# db <- dbscan(loadings %>% select(-c(metric)), eps = .2, minPts = 2)
# loadings$metric_group <- as.factor(db$cluster) 
peak_vals_pca_loadings_plt <- ggplot(loadings_vals, aes(x = PC1, y = PC2)) +
  geom_text_repel(aes(label = metric), size = 4,
                      fontface = "bold", show.legend = FALSE, max.overlaps = Inf,
                      min.segment.length = 0, box.padding = 0.35, seed = 1) +
  theme_bw(base_size = 14) +
  guides(color = guide_legend(override.aes = list(shape = 16, alpha = 1))) +
  scale_colour_brewer(palette = "Set1") +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
  coord_cartesian(xlim = c(-max(abs(loadings_vals$PC1)), max(abs(loadings_vals$PC1))),
                  ylim = c(-max(abs(loadings_vals$PC2)), max(abs(loadings_vals$PC2)))) +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle("Metric Loadings (Peak Value)") + labs(x = "PC 1", y = "PC 2")
ggsave(file.path(plots_dir, "peaks", "mcmc", "peak_vals_pca_loadings.png"), peak_vals_pca_loadings_plt, width = 5, height = 5)


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


skew_plt <- ggplot(skew_plt_df, aes(x = abs(first_deriv_pre / first_deriv_post), color = metric_group, y = metric)) +
  stat_pointinterval() +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_x_log10() +
  scale_color_brewer(palette = "Set1") +
  scale_y_discrete(expand = expansion(mult = c(0.2, 0.2))) +
  theme_bw(base_size = 14) +
  labs(
    title = "Posterior Mean of Pre vs. Post Peak First Deriv. by Metric",
    x = "|Pre-Peak| / |Post-Peak| First Derivative (log scale; 1 = symmetric)",
    color = "Metric Group"
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_skew.png"), skew_plt)

skew_obpm_df <- skew_plt_df |>
  filter(metric == "OBPM") |>
  inner_join(
    data |>
      filter(id != "99999999") |>
      group_by(id) |>
      summarize(name = first(name), position_group = first(position_group), .groups = "drop"),
    by = c("player" = "id")
  )

obpm_skew_scatter <- ggplot(skew_obpm_df,
    aes(x = first_deriv_pre, y = first_deriv_post, color = position_group)) +
  geom_point(aes(size = minutes), alpha = 0.55) +
  geom_text_repel(
    data = filter(skew_obpm_df, name %in% posterior_plot_names),
    aes(label = plot_name(name)),
    size = 3, fontface = "bold", max.overlaps = 20
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_colour_brewer(palette = "Set1") +
  theme_bw(base_size = 14) +
  labs(
    title = "OBPM Career Arc: Pre-Peak vs Post-Peak First Derivative",
    subtitle = "Each point is a player (posterior mean); x > 0 = ascending, y < 0 = declining",
    x = "Pre-Peak First Derivative (latent units / year)",
    y = "Post-Peak First Derivative (latent units / year)",
    color = "Position",
    size = "Career Minutes"
  )

ggsave(file.path(plots_dir, "peaks", "mcmc", "obpm_skew_scatter.png"), obpm_skew_scatter,
       width = 10, height = 7)


peaks_plt <- ggplot(peaks_plt_df |> inner_join(loadings, by = "metric") |> group_by(metric, player) |> summarize(minutes = first(minutes), metric_group = first(metric_group), value = mean(value)) |> ungroup() |>   mutate(
    metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
  ), 

  aes(y = metric, x = value, fill = metric_group, color = metric_group)) +
  stat_pointinterval() + 
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") + 
  theme_bw(base_size = 14) +
  labs(
    title = str_wrap("Posterior Mean of Peak Age by Metric", 24),
    x = "Age",
    y = "Metric",
    fill = "Metric Group",
    color = "Metric Group",
  ) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            plot.title = element_text(hjust = 0.5),
            legend.position = "bottom") + scale_y_discrete(expand = expansion(mult = c(0.2, 0.2)))



ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks.png"), peaks_plt, width = 5, height = 5.4)

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
  theme_bw(base_size = 14) + scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) 


ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_third_deriv.png"), third_deriv_plt)



curve_third_deriv_plt <- ggplot(third_deriv |> filter(metric == "obpm") |> mutate(quantile = cut(value, breaks = c(-Inf, -0.05, 0.05, Inf),
                                                                     labels = c("Left-Skew Symmetry", "Symmetric", "Right-Skew Symmetry"))) |> group_by(quantile) |> slice_sample(n = 5) |> ungroup() |> group_by(metric, sample,chain, player) |> select(-value) |> 
                                inner_join(posterior_mu_data) |> mutate(value = value - value[age == 18]) , aes(x = age, y = value, group = interaction(chain, sample, player, metric), color = quantile)) + geom_line(alpha = .9)  + theme_bw(base_size = 14) + scale_fill_brewer(palette = "Set1") +
                                labs(x = "Age", y = "Latent Curve Value", color = "Skew Type") + ggtitle("Illustration of Third Derivative Influence on Latent Curve Symmetry")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_third_deriv_curves.png"), curve_third_deriv_plt)

} # end if (!is.null(third_deriv))

} # end if (!is.null(latent_space))

} # end if (!is.null(posterior_peaks))

# Posterior + injury merge and train/holdout split (shared with coverage.r;
# see build_joined_data in diagnostics_utils.r). Only joined_data is needed by
# the player/paper plots below.
cov_ctx <- build_joined_data(posterior_data, injury_data, data, posterior_peaks,
                             model_dir, validation_year)
joined_data <- cov_ctx$joined_data

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
    theme_bw(base_size = 14)
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
    theme_bw(base_size = 14) +
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
      theme_bw(base_size = 14) +
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
    facet_wrap(~ metric, scales = "free_y") + theme_bw(base_size = 14) +
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
    facet_wrap(~ metric, scales = "free_y") + theme_bw(base_size = 14) +
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

# Wire the exported exit data into the player-trajectory frame: posterior_ar (and thus
# metric_plot_df) carries no exit info, but posterior_exit_survival.parquet does. Build
# EXIT_SURVIVAL rows (survival curve summarized over draws + observed exit metadata) so
# plot_posterior renders the survival facet with an exit-age vline.
surv_plot_ids <- data |> distinct(id, name) |> filter(name %in% posterior_plot_names) |> pull(id)
survival_metric_df <- posterior_survival_data |>
  filter(player %in% surv_plot_ids) |>
  group_by(player, age) |>
  summarize(
    lower             = HDInterval::hdi(value, credMass = 0.95)["lower"],
    upper             = HDInterval::hdi(value, credMass = 0.95)["upper"],
    posterior_mean    = mean(value, na.rm = TRUE),
    observed_exit_age = first(observed_exit_age),
    exit_censored     = first(exit_censored),
    .groups = "drop"
  ) |>
  mutate(metric = "EXIT_SURVIVAL", mu = posterior_mean,
         obs_value = NA_real_, year = NA_integer_)

metric_plot_df <- bind_rows(metric_plot_df, survival_metric_df)

# Per-player exit age density: two overlaid curves (entrance-conditioned vs last_observed-conditioned)
exit_density_splits <- posterior_exit_samples |>
  inner_join(data |> distinct(id, name) |> filter(name %in% posterior_plot_names),
             by = c("player" = "id")) |>
  group_by(player) |>
  group_split()

exit_density_by_player <- set_names(
  map(exit_density_splits, ~ {
    t_obs     <- first(.x$observed_exit_age)
    exit_cens <- first(.x$exit_censored)
    ggplot(.x, aes(x = value, fill = conditioning_label, color = conditioning_label)) +
      geom_density(alpha = 0.35, linewidth = 0.6) +
      scale_fill_manual(
        values = c("entrance" = "#2980b9", "last_observed" = "#e74c3c"),
        labels = c("entrance" = "P(exit | T > entrance)", "last_observed" = glue("P(exit | T > {t_obs})")),
        name = NULL
      ) +
      scale_color_manual(
        values = c("entrance" = "#2980b9", "last_observed" = "#e74c3c"),
        labels = c("entrance" = "P(exit | T > entrance)", "last_observed" = glue("P(exit | T > {t_obs})")),
        name = NULL
      ) +
      geom_vline(xintercept = t_obs,
                 linetype = if_else(exit_cens == 1L, "dashed", "solid"),
                 color = "black") +
      labs(x = "Exit Age", y = "Density") +
      theme_bw(base_size = 14) +
      theme(legend.position = "bottom")
  }),
  map_chr(exit_density_splits, ~ as.character(first(.x$player)))
)

plots_list <- metric_plot_df |>
  inner_join(data |> distinct(id, name) |> filter(name %in% posterior_plot_names), by = c("player" = "id")) %>%
  group_by(player) %>%
  group_split() %>%
  map(~ {
    player_id   <- as.character(first(.x$player))
    name        <- unique(.x$name)
    plt_metrics <- plot_posterior(.x, 2021)
    plt_exit    <- exit_density_by_player[[player_id]]
    plt <- if (!is.null(plt_exit))
      patchwork::wrap_plots(plt_metrics, plt_exit, ncol = 1, heights = c(3, 1))
    else
      plt_metrics
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
      size = 6,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw(base_size = 22) +
    labs(x = "Age", y = "")

kevin_durant <- player_plot_df |> filter(name == "Kevin Durant") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_text(data = label_df |> filter(name == "Kevin Durant"),
      size = 6,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw(base_size = 22) +
    labs(x = "Age", y = "")

stephen_curry <- player_plot_df |> filter(name == "Stephen Curry") |> ggplot(aes(x = age)) + geom_ribbon(aes(ymin = lower, ymax = upper),
                                       fill = "gray",
                                       alpha = 0.4) +
    geom_line(aes(x = age, y = mu),  color = "#4DAF4AFF", linewidth = 1) +
    geom_vline(aes(xintercept = age_of_injury), linetype = "dashed", color = "blue") +
    geom_point(aes(x = age, y = obs_value), color = "black") +
    geom_text(data = label_df |> filter(name == "Stephen Curry"),
      size = 6,
      aes(x = age_of_injury, y = .65*max_upper, label = first_major_injury)) +
    facet_wrap(~ metric, scales = "free") + theme_bw(base_size = 22) +
    labs(x = "Age", y = "")

player_plots <- (derrick_rose / kevin_durant / stephen_curry) + plot_annotation(title = "Posterior Predictive Production Curves", tag_levels = list(c("Derrick Rose", "Kevin Durant", "Stephen Curry")))

ggsave(file.path(plots_dir, "player_plots", "mcmc", "KD_Rose_Curry_comparison.png"),
       player_plots, width = 14, height = 18)



### Jokic / Curry side-by-side posterior predictive (mu uncertainty only)

jc_player_ids <- data |> distinct(id, name) |> filter(name %in% c("Nikola Jokic", "Stephen Curry"))
jc_metrics    <- c("obpm", "fg2m", "fg2a", "fta")

jokic_curry_mu_df <- posterior_mu_data |>
  filter(metric %in% jc_metrics) |>
  inner_join(jc_player_ids, by = c("player" = "id")) |>
  mutate(mu_t = case_when(
    metric %in% c("fg2m", "ftm", "games", "fg3m") ~ plogis(value),
    metric %in% c("obpm", "dbpm")                  ~ value,
    metric %in% c("pct_minutes")                   ~ plogis(value) * 48,
    metric %in% c("usg")                           ~ plogis(value),
    .default                                        = exp(value) * 36
  )) |>
  mutate(metric = toupper(metric), metric = if_else(metric == "FG2M", "FG2%", metric)) |>
  group_by(metric, player, name, age) |>
  summarize(
    lower = HDInterval::hdi(mu_t, credMass = 0.95)["lower"],
    upper = HDInterval::hdi(mu_t, credMass = 0.95)["upper"],
    mu    = mean(mu_t, na.rm = TRUE),
    .groups = "drop"
  )

jokic_curry_obs_df <- joined_data |>
  filter(metric %in% jc_metrics) |>
  inner_join(jc_player_ids, by = c("player" = "id")) |>
  mutate(metric = toupper(metric), metric = if_else(metric == "FG2M", "FG2%", metric)) |>
  distinct(player, metric, age, obs_value)

jokic_curry_plot_df <- jokic_curry_mu_df |>
  left_join(jokic_curry_obs_df, by = c("player", "metric", "age"))

# Grid layout: players are rows, metrics are columns (one panel per player x metric).
jc_metric_order <- c("OBPM", "FG2%", "FG2A", "FTA")
jc_player_order <- c("Nikola Jokic", "Stephen Curry")

# Shared y-range per metric (across both players) so the two player rows are
# directly comparable within a metric column; OBPM is clamped to a fixed window.
jc_ylims <- jokic_curry_plot_df |>
  group_by(metric) |>
  summarize(ymin = min(c(lower, obs_value), na.rm = TRUE),
            ymax = max(c(upper, obs_value), na.rm = TRUE), .groups = "drop")

jc_cell <- function(player_name, m) {
  yl       <- jc_ylims |> filter(metric == m)
  ylim_use <- if (m == "OBPM") c(-10, 10) else c(yl$ymin, yl$ymax)
  top_row  <- player_name == jc_player_order[1]
  bot_row  <- player_name == jc_player_order[length(jc_player_order)]
  left_col <- m == jc_metric_order[1]

  ggplot(jokic_curry_plot_df |> filter(name == player_name, metric == m), aes(x = age)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "gray", alpha = 0.4) +
    geom_line(aes(y = mu), color = "#4DAF4AFF", linewidth = 1) +
    geom_point(aes(y = obs_value), color = "black") +
    coord_cartesian(ylim = ylim_use) +
    theme_bw(base_size = 22) +
    labs(
      title = if (top_row)  m           else NULL,  # metric column headers (top row)
      x     = if (bot_row)  "Age"       else NULL,  # Age label (bottom row)
      y     = if (left_col) player_name else NULL   # player row labels (left column)
    ) +
    theme(plot.title = element_text(hjust = 0.5))
}

jc_plot <- wrap_plots(
  lapply(jc_player_order, function(pl)
    wrap_plots(lapply(jc_metric_order, function(m) jc_cell(pl, m)), nrow = 1)),
  ncol = 1
)

ggsave(
  file.path(plots_dir, "player_plots", "mcmc", "Jokic_Curry_comparison.png"),
  jc_plot,
  width = 18, height = 9, dpi = 150
)


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
#     theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
#   theme_bw(base_size = 14) +
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
