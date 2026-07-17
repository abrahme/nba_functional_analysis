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
library(cluster)

options(expressions = 500000)

source("data_analysis/diagnostics_utils.r")

args      <- commandArgs(trailingOnly = TRUE)
model_dir <- if (length(args) >= 1) args[1] else
  stop("Usage: Rscript data_analysis/latent_space.r <model_dir>")

plots_dir <- file.path(model_dir, "plots")
dir.create(file.path(plots_dir, "latent_space", "map"),    recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "latent_space", "mcmc"),   recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "latent_space", "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(plots_dir, "mcmc"),                   recursive = TRUE, showWarnings = FALSE)

# ── Data loads ────────────────────────────────────────────────────────────────
data <- read.csv("data/injury_player_cleaned.csv")

posterior_mu_data   <- read_parquet(file.path(model_dir, "posterior_mu_ar.parquet"))
posterior_peaks     <- read_parquet_if_exists(file.path(model_dir, "posterior_peaks_ar.parquet"))
posterior_peak_vals <- read_parquet_if_exists(file.path(model_dir, "posterior_peak_vals_ar.parquet"))
latent_space        <- read_parquet_if_exists(file.path(model_dir, "latent_space.parquet"))
phi_X               <- read_parquet_if_exists(file.path(model_dir, "phi_X.parquet"))
log_posterior       <- read_parquet_if_exists(file.path(model_dir, "log_posterior.parquet"))

posterior_latent_X <- if (file.exists(file.path(model_dir, "posterior_latent_X.parquet"))) {
  open_dataset(file.path(model_dir, "posterior_latent_X.parquet")) |>
    select(id, name, chain, sample, starts_with("Dim")) |>
    collect()
} else NULL

posterior_latent_X_peak_age   <- read_parquet_if_exists(file.path(model_dir, "posterior_latent_X_peak_age.parquet"))
posterior_latent_X_peak_value <- read_parquet_if_exists(file.path(model_dir, "posterior_latent_X_peak_value.parquet"))
phi_X_peak_age    <- read_parquet_if_exists(file.path(model_dir, "phi_X_peak_age.parquet"))
phi_X_peak_value  <- read_parquet_if_exists(file.path(model_dir, "phi_X_peak_value.parquet"))

# Load first derivative lazily (98M rows — do not collect until after aggregation)
posterior_first_deriv_ds <- {
  fd_path <- file.path(model_dir, "posterior_first_deriv_ar.parquet")
  if (file.exists(fd_path)) arrow::open_dataset(fd_path) else NULL
}

curvature_post_paths <- list.files(
  model_dir,
  pattern   = "^posterior_latent_X_curvature_m[0-9]+\\.parquet$",
  full.names = TRUE
)

player_summary <- data |>
  group_by(id) |>
  summarize(name = first(name), position_group = first(position_group), minutes = sum(minutes)) |>
  ungroup()

peaks_plt_df <- if (!is.null(posterior_peaks)) {
  posterior_peaks |>
    inner_join(player_summary, by = c("player" = "id")) |>
    mutate(
      metric = toupper(metric),
      metric = case_when(
        metric == "GAMES"       ~ "GP%",
        metric == "FG2M"        ~ "FG2%",
        metric == "FG3M"        ~ "FG3%",
        metric == "FTM"         ~ "FT%",
        metric == "PCT_MINUTES" ~ "MPG",
        .default = metric
      ),
      metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
    )
} else NULL

peak_vals_plt_df <- if (!is.null(posterior_peak_vals)) {
  posterior_peak_vals |>
    inner_join(player_summary, by = c("player" = "id")) |>
    mutate(
      metric = toupper(metric),
      metric = case_when(
        metric == "GAMES"       ~ "GP%",
        metric == "FG2M"        ~ "FG2%",
        metric == "FG3M"        ~ "FG3%",
        metric == "FTM"         ~ "FT%",
        metric == "PCT_MINUTES" ~ "MPG",
        .default = metric
      ),
      metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)
    )
} else NULL

if (!is.null(latent_space)) {

latent_space_umap <- latent_space %>% select(starts_with("Dim")) %>% umap(n_neighbors = 50, min_dist = 0.001, verbose = TRUE) %>% as_tibble(.name_repair = "unique") %>% cbind(latent_space %>% select(-starts_with("Dim"))) %>% rename(UMAP1 = `...1`, UMAP2 = `...2`)

latent_space_plot  <- latent_space_umap |> ggplot(aes(x = UMAP1, y = UMAP2)) + geom_point(aes(alpha = minutes, color = position_group)) + scale_alpha(range = c(0,1)) +
                      geom_text_repel(
                      data = filter(latent_space_umap, name %in% posterior_plot_names),
                      aes(label = plot_name(name), x = UMAP1, y = UMAP2),
                      size = 4,
                      fontface = "bold",
                      max.overlaps = 5,
                      inherit.aes = FALSE) +
  theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") + ggtitle("UMAP Visualization of Learned Latent Embedding") + labs(x = "UMAP 1", y = "UMAP 2", alpha = "Minutes", color = "Position Group")

ggsave(file.path(plots_dir, "latent_space", "map", "latent_space_umap.png"), latent_space_plot)

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
                      aes(label = plot_name(name), x = PCA1, y = PCA2),
                      size = 2,
                      fontface = "bold",
                      max.overlaps = 20,
                      inherit.aes = FALSE) +
  theme_bw(base_size = 14) + scale_colour_brewer(palette = "Set1") + ggtitle("PCA Visualization of Learned Metric Functionals") + labs(x = "PC 1", y = "PC 2", alpha = "Minutes", color = "Position Group")
ggsave(file.path(plots_dir, "latent_space", "map", "latent_space_functional_pca.png"), functional_pca_plt)



phi_X_mat <- phi_X |> select(starts_with("Dim")) |> as.matrix()

# 2. Convert to similarity matrix
K      <- phi_X_mat %*% t(phi_X_mat)
D_inv  <- 1 / sqrt(diag(K))
K_corr <- diag(D_inv) %*% K %*% diag(D_inv)

K_subset <- K_corr[phi_X$name %in% posterior_plot_names, phi_X$name %in% posterior_plot_names]

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
         filename = file.path(plots_dir, "latent_space", "K_X_heatmap.png"))









##### LOG POSTERIOR TRACE PLOT

lp_trace_plt <- log_posterior |>
  mutate(chain = factor(chain)) |>
  ggplot(aes(x = draw, y = log_joint, color = chain, group = chain)) +
  geom_line(alpha = 0.7, linewidth = 0.4) +
  theme_bw(base_size = 14) +
  scale_colour_brewer(palette = "Set1") +
  labs(title = "Log Joint Trace by Chain", x = "Draw", y = "Log Joint", color = "Chain")

ggsave(file.path(plots_dir, "mcmc", "log_posterior_trace.png"), lp_trace_plt)

##### LATENT SPACE MCMC DIAGNOSTICS (trace plots per player colored by chain)

# Helper: compute R-hat (Gelman-Rubin) for a single parameter
compute_rhat <- function(chain, value) {
  chains <- split(value, chain)
  n <- min(lengths(chains))
  chains <- lapply(chains, function(x) x[seq_len(n)])
  W <- mean(sapply(chains, var))
  B <- n * var(sapply(chains, mean))
  sqrt(((n - 1) / n * W + B / n) / W)
}

# 1. Pivot long, filter to players of interest
trace_df <- posterior_latent_X |>
  filter(name %in% posterior_plot_names) |>
  mutate(chain = factor(chain)) |>
  pivot_longer(cols = starts_with("Dim"), names_to = "dimension", values_to = "value")

# 2. Compute R-hat per player x dimension, build strip label
rhat_df <- trace_df |>
  group_by(name, id, dimension) |>
  summarise(rhat = compute_rhat(chain, value), .groups = "drop") |>
  mutate(strip_label = glue("{dimension} (R-hat = {round(rhat, 3)})"))

trace_df <- trace_df |>
  left_join(rhat_df |> select(name, dimension, strip_label), by = c("name", "dimension")) |>
  mutate(strip_label = factor(strip_label, levels = unique(rhat_df$strip_label)))

# 3. Plot function: trace plots faceted by dimension for a single player
plt_trace <- function(player_trace_df) {
  player_name <- unique(player_trace_df$name)
  ggplot(player_trace_df, aes(x = sample, y = value, color = chain)) +
    geom_line(alpha = 0.6, linewidth = 0.3) +
    facet_wrap(~ strip_label, scales = "free_y", ncol = 3) +
    theme_bw(base_size = 14) +
    theme(strip.text = element_text(size = 7)) +
    labs(title = glue("MCMC Trace Plots: {player_name}"), x = "Sample", y = "Value", color = "Chain")
}

# 4. Generate and save plots for all posterior_plot_names

trace_df |>
  group_by(id) |>
  group_split() |>
  walk(~ {
    plt <- plt_trace(.x)
    player_name <- unique(.x$name)
    ggsave(
      filename = file.path(plots_dir, "latent_space", "mcmc", glue("trace_{player_name}.png")),
      plot = plt,
      width = 14,
      height = 18
    )
  })


##### PEAK AGE / PEAK VALUE MCMC TRACE PLOTS

rename_metrics <- function(df) {
  df |>
    mutate(
      metric = toupper(metric),
      metric = case_when(
        metric == "GAMES"        ~ "GP%",
        metric == "FG2M"         ~ "FG2%",
        metric == "FG3M"         ~ "FG3%",
        metric == "FTM"          ~ "FT%",
        metric == "PCT_MINUTES"  ~ "MPG",
        .default = metric
      )
    )
}

player_lookup <- data |>
  group_by(id) |>
  summarize(name = first(name), .groups = "drop")

peaks_trace_df <- posterior_peaks |>
  inner_join(player_lookup, by = c("player" = "id")) |>
  filter(name %in% posterior_plot_names) |>
  mutate(chain = factor(chain)) |>
  rename_metrics()

peak_vals_trace_df <- posterior_peak_vals |>
  inner_join(player_lookup, by = c("player" = "id")) |>
  filter(name %in% posterior_plot_names) |>
  mutate(chain = factor(chain)) |>
  rename_metrics()

# R-hat per player x metric
rhat_peaks_df <- peaks_trace_df |>
  group_by(name, player, metric) |>
  summarise(rhat = compute_rhat(chain, value), .groups = "drop") |>
  mutate(strip_label = glue("{metric} (R-hat = {round(rhat, 3)})"))

rhat_peak_vals_df <- peak_vals_trace_df |>
  group_by(name, player, metric) |>
  summarise(rhat = compute_rhat(chain, value), .groups = "drop") |>
  mutate(strip_label = glue("{metric} (R-hat = {round(rhat, 3)})"))

peaks_trace_df <- peaks_trace_df |>
  left_join(rhat_peaks_df |> select(name, metric, strip_label), by = c("name", "metric"))

peak_vals_trace_df <- peak_vals_trace_df |>
  left_join(rhat_peak_vals_df |> select(name, metric, strip_label), by = c("name", "metric"))

plt_peaks_trace <- function(player_df, title_suffix) {
  player_name <- unique(player_df$name)
  ordered_labels <- player_df |>
    distinct(metric, strip_label) |>
    arrange(metric) |>
    pull(strip_label)
  player_df <- player_df |>
    mutate(strip_label = factor(strip_label, levels = ordered_labels))
  ggplot(player_df, aes(x = sample, y = value, color = chain)) +
    geom_line(alpha = 0.6, linewidth = 0.3) +
    facet_wrap(~ strip_label, scales = "free_y", ncol = 3) +
    theme_bw(base_size = 14) +
    theme(strip.text = element_text(size = 7)) +
    labs(title = glue("{title_suffix}: {player_name}"), x = "Sample", y = "Value", color = "Chain")
}


peaks_trace_df |>
  group_by(player) |>
  group_split() |>
  walk(~ {
    plt <- plt_peaks_trace(.x, "Peak Age Trace")
    player_name <- unique(.x$name)
    ggsave(
      filename = file.path(plots_dir, "peaks", "mcmc", "trace", glue("peak_age_{player_name}.png")),
      plot = plt, width = 14, height = 18
    )
  })

peak_vals_trace_df |>
  group_by(player) |>
  group_split() |>
  walk(~ {
    plt <- plt_peaks_trace(.x, "Peak Value Trace")
    player_name <- unique(.x$name)
    ggsave(
      filename = file.path(plots_dir, "peaks", "mcmc", "trace", glue("peak_val_{player_name}.png")),
      plot = plt, width = 14, height = 18
    )
  })

##### PROCRUSTES-ALIGNED TRACE PLOTS

dim_cols <- names(posterior_latent_X)[startsWith(names(posterior_latent_X), "Dim")]

# phi_X is the MAP estimate — use it as the Procrustes reference target.
# Order rows by id to ensure correspondence with posterior snapshots.
phi_ref_ordered <- phi_X |>
  arrange(id) |>
  select(all_of(dim_cols)) |>
  as.matrix()

# Helper: find orthogonal R minimising ||B %*% R - A||_F
procrustes_rotation <- function(B, A) {
  sv <- svd(t(B) %*% A)
  sv$u %*% t(sv$v)
}

# Align every (chain, sample) snapshot to phi_ref_ordered
rotated_posterior_df <- posterior_latent_X |>
  mutate(chain = factor(chain)) |>
  group_by(chain, sample) |>
  group_modify(~ {
    B <- .x |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
    R <- procrustes_rotation(B, phi_ref_ordered)
    B_rot <- B %*% R
    colnames(B_rot) <- dim_cols
    .x |>
      arrange(id) |>
      select(-all_of(dim_cols)) |>
      bind_cols(as_tibble(B_rot))
  }) |>
  ungroup()

# Pivot long, filter to players of interest
rotated_trace_df <- rotated_posterior_df |>
  filter(name %in% posterior_plot_names) |>
  pivot_longer(cols = all_of(dim_cols), names_to = "dimension", values_to = "value")

# R-hat on aligned samples
rhat_rotated_df <- rotated_trace_df |>
  group_by(name, id, dimension) |>
  summarise(rhat = compute_rhat(chain, value), .groups = "drop") |>
  mutate(strip_label = glue("{dimension} (R-hat = {round(rhat, 3)})"))

rotated_trace_df <- rotated_trace_df |>
  left_join(rhat_rotated_df |> select(name, dimension, strip_label), by = c("name", "dimension"))

# Save aligned trace plots (reuses plt_trace which handles strip label ordering)
rotated_trace_df |>
  group_by(id) |>
  group_split() |>
  walk(~ {
    plt <- plt_trace(.x)
    player_name <- unique(.x$name)
    ggsave(
      filename = file.path(plots_dir, "latent_space", "mcmc", glue("rotated_trace_{player_name}.png")),
      plot = plt,
      width = 14,
      height = 18
    )
  })

##### ARCHETYPE CLUSTERING + NEAREST-NEIGHBOR TABLES + UNCERTAINTY

library(cluster)  # silhouette — part of base R recommended packages

# ── 1. Posterior mean of Procrustes-aligned samples ──────────────────────────
posterior_mean_latent <- rotated_posterior_df |>
  group_by(id, name) |>
  summarise(across(all_of(dim_cols), mean), .groups = "drop")

phi_mat   <- posterior_mean_latent |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
rownames(phi_mat) <- posterior_mean_latent |> arrange(id) |> pull(name)
phi_mat <- scale(phi_mat)
phi_dist  <- dist(phi_mat)
hc_latent <- hclust(phi_dist, method = "ward.D2")

# ── 2. k diagnostics (WSS + silhouette, k = 2..10) ──────────────────────────
wss_sil <- map_dfr(2:10, function(k) {
  labs <- cutree(hc_latent, k)
  wss  <- sum(sapply(unique(labs), function(cl) {
    sub <- phi_mat[labs == cl, , drop = FALSE]
    sum(scale(sub, scale = FALSE)^2)
  }))
  sil <- mean(silhouette(labs, phi_dist)[, "sil_width"])
  tibble(k = k, wss = wss, silhouette = sil)
})

wss_plt <- ggplot(wss_sil, aes(x = k, y = wss)) +
  geom_line() + geom_point() +
  labs(title = "Within-cluster SS vs. k", x = "k", y = "WSS") + theme_bw(base_size = 14)
sil_plt <- ggplot(wss_sil, aes(x = k, y = silhouette)) +
  geom_line() + geom_point() +
  labs(title = "Mean silhouette vs. k", x = "k", y = "Avg silhouette") + theme_bw(base_size = 14)
ggsave(file.path(plots_dir, "latent_space", "map", "archetype_k_diagnostics.png"),
       wss_plt + sil_plt, width = 12, height = 5)

# ── 3. Dendrogram utilities ──────────────────────────────────────────────────
# Extract ggplot-ready segment data from an hclust object by iterating over
# the merge matrix. No dendrapply, no C-stack exhaustion on 2000+ leaf trees.
hclust_segments <- function(hc) {
  n      <- length(hc$order)
  leaf_y <- integer(n)
  leaf_y[hc$order] <- seq_len(n)

  node_y <- numeric(n - 1L)
  node_x <- hc$height

  get_cy <- function(idx) if (idx < 0L) leaf_y[-idx] else node_y[idx]
  get_cx <- function(idx) if (idx < 0L) 0            else node_x[idx]

  mat <- matrix(0.0, nrow = (n - 1L) * 3L, ncol = 4L)
  k   <- 0L
  for (i in seq_len(n - 1L)) {
    ya <- get_cy(hc$merge[i, 1L]); xa <- get_cx(hc$merge[i, 1L])
    yb <- get_cy(hc$merge[i, 2L]); xb <- get_cx(hc$merge[i, 2L])
    node_y[i] <- (ya + yb) / 2.0
    xi <- node_x[i]
    k <- k + 1L; mat[k, ] <- c(xa, xi, ya, ya)   # left arm
    k <- k + 1L; mat[k, ] <- c(xb, xi, yb, yb)   # right arm
    k <- k + 1L; mat[k, ] <- c(xi, xi, ya, yb)   # vertical bar
  }

  list(
    segments = data.frame(x = mat[,1], xend = mat[,2], y = mat[,3], yend = mat[,4]),
    labels   = data.frame(y = as.integer(leaf_y), label = hc$labels,
                          stringsAsFactors = FALSE),
    n        = n,
    max_h    = max(node_x)
  )
}

# Horizontal dendrogram with geom_text_repel — no label overlap.
# keep_labels: character vector of leaf names to annotate.
plot_dendrogram_gg <- function(hc, keep_labels, title = "") {
  ddata    <- hclust_segments(hc)
  max_h    <- ddata$max_h
  label_df <- ddata$labels |> filter(label %in% keep_labels)

  ggplot() +
    geom_segment(
      data = ddata$segments,
      aes(x = x, xend = xend, y = y, yend = yend),
      linewidth = 0.25, color = "grey40"
    ) +
    ggrepel::geom_text_repel(
      data               = label_df,
      aes(x = 0, y = y, label = label),
      direction          = "y",
      nudge_x            = -max_h * 0.35,
      hjust              = 0,
      size               = 3.2,
      segment.size       = 0.2,
      segment.color      = "grey60",
      max.overlaps       = Inf,
      force              = 0.3,
      min.segment.length = 0,
      box.padding        = 0.1
    ) +
    scale_x_reverse(expand = expansion(mult = c(0.01, 0.01))) +
    scale_y_continuous(limits = c(0, ddata$n + 1), breaks = NULL) +
    coord_cartesian(clip = "off") +
    labs(title = title, x = "Height", y = NULL) +
    theme_bw(base_size = 14) +
    theme(
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid   = element_blank(),
      plot.margin  = unit(c(0.5, 6, 0.5, 0.5), "cm")
    )
}

notable_names <- posterior_plot_names

ggsave(
  file.path(plots_dir, "latent_space", "map", "archetype_dendrogram.png"),
  plot_dendrogram_gg(hc_latent, notable_names,
                     "Ward hierarchical clustering — posterior mean latent dims"),
  width = 14, height = 18
)

# ── 4. Cut tree and assign archetypes ────────────────────────────────────────
find_k_dendrogram <- function(hc, default_k = 4L, max_k = 50L) {
  heights <- hc$height
  if (length(heights) < 2L) return(default_k)
  rel_gaps <- diff(heights) / pmax(heights[-1], .Machine$double.eps)
  idx <- which.max(rel_gaps)
  if (length(idx) == 0L) return(default_k)
  k <- length(heights) - idx
  # If k is implausibly large (signal-poor modality), fall back to default
  if (k < 2L || k > max_k) return(default_k)
  k
}

k_archetypes <- find_k_dendrogram(hc_latent)
message(glue("Optimal k by dendrogram gap: {k_archetypes}"))

# Palettes that scale with k_archetypes — Set1 capped at 9, hue_pal beyond
arch_colour_pal <- if (k_archetypes <= 9L) {
  RColorBrewer::brewer.pal(max(3L, k_archetypes), "Set1")[seq_len(k_archetypes)]
} else {
  scales::hue_pal()(k_archetypes)
}
arch_shape_pool <- c(15L, 17L, 18L, 19L, 16L, 8L, 3L, 4L, 7L, 10L, 11L, 13L, 14L, 25L)
arch_shape_pal  <- arch_shape_pool[seq_len(k_archetypes)]

archetype_labels <- cutree(hc_latent, k = k_archetypes)

posterior_mean_latent <- posterior_mean_latent |>
  arrange(id) |>
  mutate(archetype = factor(unname(archetype_labels)))

phi_X        <- phi_X        |> left_join(posterior_mean_latent |> select(id, archetype), by = "id")
latent_space <- latent_space |> left_join(posterior_mean_latent |> select(id, archetype), by = "id")

archetype_reps <- latent_space |>
  filter(!is.na(archetype)) |>
  group_by(archetype) |>
  slice_max(minutes, n = 5) |>
  select(archetype, name, position_group, minutes)
print(archetype_reps)

# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT PROBABILITY BY ENTRY COHORT
#
# Players who entered the league in year Y have Y-specific amounts of data,
# so their posterior X_p has Y-specific uncertainty. Grouping by entry cohort
# (first season observed) gives an apples-to-apples comparison within each
# group and shows how archetype certainty accumulates across cohorts — without
# requiring sequential model refits.
#
# Two measures computed per player:
#   (A) P(peak OBPM > threshold) from posterior_peak_vals draws
#   (B) P(elite archetype) — fraction of MCMC X_p draws that nearest-centroid
#       classify to the archetype with the highest mean peak OBPM
# ══════════════════════════════════════════════════════════════════════════════

if (!is.null(posterior_peak_vals) && !is.null(posterior_latent_X)) {

  # Entry cohort: draft_year + 1 gives the first playing season (the draft
  # occurs in June/July, so a 2023 draftee's first season is 2023-24, recorded
  # as year = 2024 in the data).  Undrafted players fall back to min(year).
  player_info <- read.csv("data/player_info.csv") |>
    select(id, draft_year) |>
    mutate(draft_year = suppressWarnings(as.integer(draft_year)))

  player_birth_year <- data |>
    filter(!is.na(age), !is.na(year), id != "99999999") |>
    group_by(id) |>
    summarise(birth_year = as.integer(round(median(year - age, na.rm = TRUE))), .groups = "drop") |>
    mutate(age_2026 = 2026L - birth_year)

  entry_cohort <- data |>
    filter(!is.na(year)) |>
    group_by(id) |>
    summarise(
      n_seasons      = n_distinct(year),
      first_obs_year = min(year, na.rm = TRUE),
      name           = first(name),
      position_group = first(position_group),
      .groups        = "drop"
    ) |>
    left_join(player_info, by = "id") |>
    mutate(cohort = coalesce(draft_year + 1L, first_obs_year))

  # ── Peak OBPM per posterior draw, including observation/sampling variance.
  # The peak AGE is the concave curve's peak (posterior_peaks); the peak VALUE is
  # the posterior PREDICTIVE OBPM at that age (posterior_ar.parquet = latent
  # mu + AR(1) residual + sampling noise). So P(peak > threshold) is the
  # probability of a breakout SEASON at the projected peak, integrating parameter
  # AND sampling uncertainty. NB: we fix the age rather than argmax the predictive
  # over all ages, because the predictive variance sigma^2/(minutes+1) blows up at
  # low-minute far-future ages and an argmax there just tracks noise spikes. OBPM
  # uses the identity link, so predictive values are already in OBPM units. ─────
  obpm_peak_age_draw <- posterior_peaks |>
    filter(metric == "obpm") |>
    transmute(player, chain, sample,
              peak_age = pmin(pmax(round(value), 18L), 38L))

  obpm_pred_peak <- open_dataset(file.path(model_dir, "posterior_ar.parquet")) |>
    filter(metric == "obpm") |>
    select(player, chain, sample, age, value) |>
    collect() |>
    inner_join(obpm_peak_age_draw, by = c("player", "chain", "sample")) |>
    filter(age == peak_age, is.finite(value)) |>
    transmute(player, chain, sample, peak_age, peak_val = value)

  # ── (A) P(peak OBPM > threshold) ───────────────────────────────────────────
  obpm_breakout <- obpm_pred_peak |>
    group_by(player) |>
    summarise(
      p_breakout_obpm  = mean(peak_val > 2.0),
      p_breakout_obpm3 = mean(peak_val > 3.0),
      p_breakout_obpm4 = mean(peak_val > 4.0),
      mean_peak_obpm   = mean(peak_val),
      sd_peak_obpm     = sd(peak_val),
      .groups          = "drop"
    )

  # ── (B) Expected years to peak (OBPM peak age minus last observed age) ───────
  # Peak age comes from the concave curve (continuous), not the rounded lookup
  # age above — sampling variance affects the peak VALUE, not where it occurs.
  obpm_peak_age <- posterior_peaks |>
    filter(metric == "obpm") |>
    group_by(player) |>
    summarise(mean_peak_age = mean(value), .groups = "drop")

  player_last_age <- data |>
    group_by(id) |>
    summarise(last_obs_age = max(age, na.rm = TRUE), .groups = "drop")

  yrs_to_peak_df <- obpm_peak_age |>
    left_join(player_last_age, by = c("player" = "id")) |>
    mutate(yrs_to_peak = pmax(mean_peak_age - last_obs_age, 0))

  # ── (C) Fraction of players whose career-peak OBPM exceeded each threshold ───
  empirical_peak_obpm <- data |>
    filter(!is.na(obpm), id != "99999999") |>
    group_by(id) |>
    summarise(peak_obpm = max(obpm, na.rm = TRUE), .groups = "drop") |>
    filter(is.finite(peak_obpm)) |>
    pull(peak_obpm)
  pct_peaked_above <- function(thresh) round(mean(empirical_peak_obpm > thresh) * 100, 0)
  pct_above_2 <- pct_peaked_above(2)
  pct_above_3 <- pct_peaked_above(3)
  pct_above_4 <- pct_peaked_above(4)

  arch_memb_probs <- rotated_posterior_df |>
    select(id, name) |>
    distinct() |>
    left_join(entry_cohort,      by = c("id", "name")) |>
    left_join(obpm_breakout,     by = c("id" = "player")) |>
    left_join(yrs_to_peak_df,    by = c("id" = "player")) |>
    left_join(player_birth_year, by = "id")

  # ── LaTeX table: top 5 per cohort ───────────────────────────────────────────
  recent_cohorts <- sort(unique(arch_memb_probs$cohort[!is.na(arch_memb_probs$cohort)]),
                         decreasing = TRUE)[1:6]

  cohort_data <- arch_memb_probs |>
    filter(cohort %in% recent_cohorts, !is.na(name)) |>
    mutate(cohort = as.integer(cohort))

  top5_per_cohort <- cohort_data |>
    filter(!is.na(mean_peak_obpm)) |>
    group_by(cohort) |>
    slice_max(mean_peak_obpm, n = 5, with_ties = FALSE) |>
    arrange(cohort, desc(mean_peak_obpm)) |>
    mutate(rank = row_number()) |>
    ungroup()

  cohort_max_year <- cohort_data |>
    group_by(cohort) |>
    summarise(max_obs_year = max(first_obs_year, na.rm = TRUE), .groups = "drop")

  # Build table body: cohort groups separated by \midrule + header row
  cohort_blocks <- map_chr(sort(unique(top5_per_cohort$cohort), decreasing = TRUE),
    function(yr) {
      block        <- top5_per_cohort |> filter(cohort == yr)
      max_obs_year <- cohort_max_year$max_obs_year[cohort_max_year$cohort == yr]
      ns           <- max(2026L - yr + 1L, 1L)
      season_label <- if (ns == 1) "1 season" else glue("{ns} seasons")
      header_row <- glue(
        "    \\multicolumn{{9}}{{l}}{{\\textit{{Entry {yr} ({season_label})}}}}"
      )
      data_rows <- pmap_chr(block, function(rank, name, position_group, age_2026,
                                            mean_peak_obpm, p_breakout_obpm,
                                            p_breakout_obpm3, p_breakout_obpm4,
                                            yrs_to_peak, ...) {
        fmt_p   <- function(x) if (is.na(x)) "---" else sprintf("%.0f\\%%", x * 100)
        ytp     <- if (is.na(yrs_to_peak)) "---" else sprintf("%.1f", yrs_to_peak)
        age_str <- if (is.na(age_2026)) "---" else as.character(age_2026)
        glue(
          "    {rank} & {name} & {age_str} & {position_group}",
          " & {sprintf('%.1f', mean_peak_obpm)}",
          " & {fmt_p(p_breakout_obpm)}",
          " & {fmt_p(p_breakout_obpm3)}",
          " & {fmt_p(p_breakout_obpm4)}",
          " & {ytp} \\\\"
        )
      })
      paste(c(paste0(header_row, " \\\\"), "    \\midrule", data_rows), collapse = "\n")
    }
  )

  tex_body <- paste(cohort_blocks, collapse = "\n    \\midrule[0.4pt]\n")

  breakout_tex <- paste0(
    "\\begin{table}[htbp]\n",
    "  \\centering\\small\n",
    "  \\begin{tabular}{rlrlrrrrr}\n",
    "    \\toprule\n",
    "     & & & & OBPM & & & & \\\\\n",
    "    \\cmidrule(lr){5-5}\n",
    "    Rank & Player & Age & Pos. & Expected Peak",
    " & P($>$2) & P($>$3) & P($>$4) & Yrs to Peak \\\\\n",
    "    \\midrule\n",
    tex_body, "\n",
    "    \\bottomrule\n",
    "  \\end{tabular}\n",
    "  \\caption{Top 5 players by posterior mean peak OBPM per entry cohort.",
    " P($>$2), P($>$3), P($>$4) are posterior probabilities that peak OBPM exceeds",
    " the threshold; empirically, ", pct_above_2, "\\% of players peaked above",
    " OBPM\\,$=$\\,2, ", pct_above_3, "\\% above OBPM\\,$=$\\,3, and ",
    pct_above_4, "\\% above OBPM\\,$=$\\,4 (career peak per player).",
    " Yrs to Peak is the posterior mean OBPM peak age minus last observed age",
    " (0.0 = already at peak).}\n",
    "  \\label{tab:breakout_cohort}\n",
    "\\end{table}\n"
  )

  writeLines(
    breakout_tex,
    file.path(plots_dir, "latent_space", "map", "breakout_by_cohort.tex")
  )

  # JSON export consumed by the JSM deck's interactive draft-cohort widget
  # (presentation/jsm/jsm_presentation.qmd) — same rows as the LaTeX table,
  # with the actual posterior mean peak age and years-to-peak (no "<1" bucket).
  cohort_json <- top5_per_cohort |>
    transmute(cohort, rank, name, age = age_2026, pos = position_group,
              peak_obpm   = round(mean_peak_obpm, 1),
              peak_age    = round(mean_peak_age, 1),
              p2          = round(p_breakout_obpm  * 100),
              p3          = round(p_breakout_obpm3 * 100),
              p4          = round(p_breakout_obpm4 * 100),
              yrs_to_peak = round(yrs_to_peak, 1))
  jsonlite::write_json(
    cohort_json,
    file.path(plots_dir, "latent_space", "map", "breakout_by_cohort.json")
  )

  # Plot 2: mean peak OBPM distribution within each cohort (violin)
  cohort_violin_plt <- ggplot(cohort_data |> filter(!is.na(mean_peak_obpm)),
      aes(x = factor(cohort), y = mean_peak_obpm, fill = factor(cohort))) +
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), alpha = 0.7) +
    geom_jitter(aes(colour = position_group), width = 0.1, alpha = 0.4, size = 1) +
    scale_fill_brewer(palette = "Blues") +
    scale_colour_brewer(palette = "Set1") +
    labs(
      title    = "Distribution of posterior mean peak OBPM by entry cohort",
      subtitle = "Narrowing spread reflects posterior contraction with more seasons observed",
      x        = "Entry cohort (first season)",
      y        = "Posterior mean peak OBPM",
      fill     = "Cohort",
      colour   = "Position"
    ) +
    theme_bw(base_size = 14)
  ggsave(
    file.path(plots_dir, "latent_space", "map", "breakout_cohort_distributions.png"),
    cohort_violin_plt, width = 12, height = 6
  )

  # Print summary table: median mean peak OBPM and spread per cohort
  cohort_summary <- arch_memb_probs |>
    filter(!is.na(cohort)) |>
    group_by(cohort) |>
    summarise(
      n_players        = n(),
      median_mean_obpm = median(mean_peak_obpm, na.rm = TRUE),
      iqr_mean_obpm    = IQR(mean_peak_obpm, na.rm = TRUE),
      median_p_obpm    = median(p_breakout_obpm, na.rm = TRUE),
      .groups          = "drop"
    ) |>
    arrange(desc(cohort))
  print(cohort_summary)

}  # end if (!is.null(posterior_peak_vals) && !is.null(posterior_latent_X))

# ── peaks with archetype mean overlays ───────────────────────────────────────
if (!is.null(peaks_plt_df)) {

# Recompute peaks PCA loadings with metric_group assignments (mirrors model_diagnostics.r)
peaks_players_ls <- peaks_plt_df |>
  group_by(metric, player) |>
  summarize(value = mean(value), .groups = "drop") |>
  pivot_wider(names_from = metric, values_from = value) |>
  inner_join(
    latent_space |> filter(minutes >= quantile(minutes, .75, na.rm = TRUE)) |> select(id),
    by = c("player" = "id")
  )

peaks_pca_ls  <- prcomp(peaks_players_ls |> select(-player) |> data.matrix(),
                        scale. = TRUE, center = TRUE)

loadings <- as.data.frame(peaks_pca_ls$rotation[, 1:2])
loadings$metric <- rownames(loadings)
loadings_mat_ls <- loadings |> select(-metric) |> as.matrix()
k_dist_ls  <- kNNdist(loadings_mat_ls, k = 2)
eps_grid_ls <- quantile(k_dist_ls, probs = seq(0.3, 0.95, length.out = 30), na.rm = TRUE)

db_cands_ls <- lapply(eps_grid_ls, function(eps_val) {
  db_try <- dbscan(loadings_mat_ls, eps = eps_val, minPts = 2)
  list(db = db_try,
       n_clusters = length(setdiff(unique(db_try$cluster), 0L)),
       n_noise    = sum(db_try$cluster == 0L))
})
# Priority 1: exactly 3 clusters with no noise points
zero_noise_3_ls <- Filter(function(x) x$n_clusters == 3L && x$n_noise == 0L, db_cands_ls)
exact_3_ls      <- Filter(function(x) x$n_clusters == 3L, db_cands_ls)
best_ls <- if (length(zero_noise_3_ls) > 0) {
  zero_noise_3_ls[[1L]]
} else if (length(exact_3_ls) > 0) {
  exact_3_ls[[which.min(sapply(exact_3_ls, `[[`, "n_noise"))]]
} else {
  db_cands_ls[[which.min(sapply(db_cands_ls, function(x) abs(x$n_clusters - 3L)))]]
}
# Reassign any remaining noise points (cluster == 0) to their nearest cluster centroid
if (any(best_ls$db$cluster == 0L)) {
  cl    <- best_ls$db$cluster
  k_ids <- setdiff(unique(cl), 0L)
  cents <- sapply(k_ids, function(ki)
    colMeans(loadings_mat_ls[cl == ki, , drop = FALSE]))
  for (i in which(cl == 0L)) {
    dists_i <- colSums((cents - loadings_mat_ls[i, ])^2)
    best_ls$db$cluster[i] <- k_ids[which.min(dists_i)]
  }
}
loadings$metric_group <- as.factor(best_ls$db$cluster)

peaks_plt <- ggplot(
  peaks_plt_df |>
    inner_join(loadings, by = "metric") |>
    group_by(metric, player) |>
    summarize(minutes = first(minutes), metric_group = first(metric_group),
              value = mean(value), .groups = "drop") |>
    mutate(metric = fct_reorder(metric, value, .fun = median, .desc = TRUE)),
  aes(y = metric, x = value, fill = metric_group, color = metric_group)
) +
  stat_pointinterval() +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  theme_bw(base_size = 14) +
  labs(title = "Posterior Mean of Peak Age by Metric",
       x = "Age", y = "Metric",
       fill = "Metric Group", color = "Metric Group") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  scale_y_discrete(expand = expansion(mult = c(0.2, 0.2)))

archetype_mean_peaks <- peaks_plt_df |>
  inner_join(loadings, by = "metric") |>
  left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id")) |>
  filter(!is.na(archetype)) |>
  group_by(metric, archetype, metric_group) |>
  summarise(mean_peak_age = mean(value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = fct_reorder(metric, mean_peak_age, .fun = median, .desc = TRUE))

peaks_plt_arch <- peaks_plt +
  geom_point(
    data = archetype_mean_peaks,
    aes(x = mean_peak_age, y = metric, shape = archetype),
    color = "black", size = 2.5, inherit.aes = FALSE
  ) +
  scale_shape_manual(values = arch_shape_pal) +
  labs(shape = "Archetype mean")

ggsave(file.path(plots_dir, "peaks", "mcmc", "peaks_with_archetypes.png"),
       peaks_plt_arch, width = 10, height = 7)

} # end if (!is.null(peaks_plt_df))

# ── 5. Archetype characterization table and bullet descriptions ──────────────
# Placed before the Frechet clustering section so it runs even if hclust fails
# on empty curve_players_common for models with fewer player-metric combinations.
if (!is.null(peaks_plt_df) && !is.null(peak_vals_plt_df)) {

archetype_peak_ages <- peaks_plt_df |>
  left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id")) |>
  filter(!is.na(archetype)) |>
  group_by(archetype, metric) |>
  summarise(mean_peak_age = mean(value, na.rm = TRUE), .groups = "drop")

archetype_peak_vals <- peak_vals_plt_df |>
  left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id")) |>
  filter(!is.na(archetype)) |>
  group_by(archetype, metric) |>
  summarise(mean_peak_val = mean(value, na.rm = TRUE), .groups = "drop")

# ── 5b. Archetype quantitative summary table (LaTeX) ─────────────────────────
# Inverse link map: display metric name → link used in model
metric_inv_link <- c(
  "GP%"  = "plogis", "USG"  = "plogis", "MPG"  = "plogis",
  "FT%"  = "plogis", "FG2%" = "plogis", "FG3%" = "plogis",
  "OBPM" = "identity", "DBPM" = "identity",
  "BLK"  = "exp", "STL" = "exp", "AST" = "exp",
  "DREB" = "exp", "OREB" = "exp", "TOV" = "exp",
  "FTA"  = "exp", "FG2A" = "exp", "FG3A" = "exp"
)
apply_inv_link <- function(x, m) {
  link <- metric_inv_link[as.character(m)]
  link <- ifelse(is.na(link), "identity", link)
  dplyr::case_when(link == "plogis" ~ plogis(x), link == "exp" ~ exp(x), TRUE ~ x)
}

archetype_summary_wide <- archetype_peak_ages |>
  inner_join(archetype_peak_vals, by = c("archetype", "metric")) |>
  mutate(
    peak_val_out = apply_inv_link(mean_peak_val, metric),
    cell = sprintf("%.1f (%.2f)", mean_peak_age, peak_val_out)
  ) |>
  select(metric, archetype, cell) |>
  pivot_wider(names_from = archetype, values_from = cell) |>
  arrange(metric) |>
  mutate(metric = gsub("%", "\\\\%", as.character(metric)))

archetype_reps_min <- posterior_mean_latent |>
  left_join(latent_space |> select(id, minutes), by = "id") |>
  filter(!is.na(archetype)) |>
  group_by(archetype) |>
  slice_max(order_by = minutes, n = 3, with_ties = FALSE) |>
  summarise(reps = paste(name, collapse = ", "), .groups = "drop") |>
  pivot_wider(names_from = archetype, values_from = reps) |>
  mutate(metric = "\\textit{Representatives}")

arch_cols <- sort(unique(archetype_peak_ages$archetype))
col_headers <- c("Metric", paste0("Archetype ", arch_cols))

archetype_table_tex <- knitr::kable(
  archetype_summary_wide |> mutate(across(everything(), as.character)),
  format    = "latex",
  booktabs  = TRUE,
  escape    = FALSE,
  col.names = col_headers,
  caption   = "Posterior mean peak age and peak value (in parentheses) by archetype and metric.",
  label     = "archetype_summary"
)

# Wrap tabular in \resizebox so the table fits within the text block width
archetype_table_str <- as.character(archetype_table_tex) |>
  stringr::str_replace(
    stringr::fixed("\\begin{tabular"),
    "\\resizebox{\\linewidth}{!}{\\begin{tabular"
  ) |>
  stringr::str_replace(
    stringr::fixed("\\end{tabular}"),
    "\\end{tabular}}"
  )

writeLines(
  archetype_table_str,
  file.path(plots_dir, "latent_space", "mcmc", "archetype_summary_table.tex")
)

# ── 5c. Dynamic archetype bullet descriptions (LaTeX) ────────────────────────
arch_metric_zscored <- archetype_peak_vals |>
  group_by(metric) |>
  mutate(
    gm  = mean(mean_peak_val, na.rm = TRUE),
    gsd = sd(mean_peak_val, na.rm = TRUE)
  ) |>
  ungroup() |>
  mutate(z = ifelse(is.na(gsd) | gsd == 0, 0, (mean_peak_val - gm) / gsd)) |>
  select(-gm, -gsd)

arch_distinctive <- arch_metric_zscored |>
  group_by(archetype) |>
  slice_max(z, n = 4, with_ties = FALSE) |>
  summarise(
    top_metrics = paste0("\\textit{", gsub("%", "\\\\%", as.character(metric)), "}", collapse = ", "),
    .groups = "drop"
  )

archetype_reps_long <- posterior_mean_latent |>
  left_join(latent_space |> select(id, minutes), by = "id") |>
  filter(!is.na(archetype)) |>
  group_by(archetype) |>
  slice_max(order_by = minutes, n = 3, with_ties = FALSE) |>
  summarise(reps = paste(name, collapse = ", "), .groups = "drop")

arch_bullets_df <- archetype_reps_long |>
  left_join(arch_distinctive, by = "archetype") |>
  arrange(archetype) |>
  mutate(item = glue(
    "\\item \\textbf{{Archetype {archetype}}} --- ",
    "\\emph{{Representatives (by career minutes):}} {reps}. ",
    "Relative to the other archetypes, this group shows the most distinctive peak ",
    "production in {top_metrics}; see Table~\\ref{{tab:archetype_summary}} for ",
    "complete peak-age and peak-value comparisons across all 17 metrics."
  ))

writeLines(
  c("\\begin{itemize}", arch_bullets_df$item, "\\end{itemize}"),
  file.path(plots_dir, "latent_space", "mcmc", "archetype_bullets.tex")
)

} # end if (!is.null(peaks_plt_df) && !is.null(peak_vals_plt_df))

# ── 5d. Pre/post-peak slope by archetype ────────────────────────────────────
# Uses the GP first derivative (posterior_first_deriv_ar.parquet) loaded lazily
# above. Joins with posterior_peaks to split each player's derivative series at
# their posterior peak age, then averages the pre-peak and post-peak slopes
# within each archetype.
if (!is.null(posterior_first_deriv_ds) && !is.null(posterior_peaks) &&
    !is.null(posterior_mean_latent)) {

  peaks_ref <- posterior_peaks |>
    transmute(player, metric,
              chain  = as.integer(chain),
              sample = as.integer(sample),
              peak_age = value)

  # Lazy arrow pipeline: join, classify phase, aggregate — collect only once
  slope_by_draw <- posterior_first_deriv_ds |>
    mutate(chain = as.integer(chain), sample = as.integer(sample)) |>
    left_join(peaks_ref, by = c("player", "metric", "chain", "sample")) |>
    filter(!is.na(peak_age)) |>
    mutate(
      phase = case_when(
        age < peak_age ~ "pre",
        age > peak_age ~ "post",
        TRUE           ~ NA_character_
      )
    ) |>
    filter(!is.na(phase)) |>
    group_by(player, metric, chain, sample, phase) |>
    summarise(mean_slope = mean(value, na.rm = TRUE), .groups = "drop") |>
    collect()

  metric_display <- function(m) {
    m <- toupper(m)
    dplyr::case_when(
      m == "GAMES"       ~ "GP%",
      m == "FG2M"        ~ "FG2%",
      m == "FG3M"        ~ "FG3%",
      m == "FTM"         ~ "FT%",
      m == "PCT_MINUTES" ~ "MPG",
      .default = m
    )
  }

  arch_slope_df <- slope_by_draw |>
    left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id")) |>
    filter(!is.na(archetype)) |>
    mutate(metric_disp = metric_display(metric)) |>
    group_by(archetype, metric_disp, phase) |>
    summarise(
      lo_slope   = quantile(mean_slope, 0.1, na.rm = TRUE),
      hi_slope   = quantile(mean_slope, 0.9, na.rm = TRUE),
      mean_slope = mean(mean_slope, na.rm = TRUE),
      .groups    = "drop"
    ) |>
    mutate(
      phase       = factor(if_else(phase == "pre", "Pre-peak", "Post-peak"),
                           levels = c("Pre-peak", "Post-peak")),
      metric_disp = fct_reorder(metric_disp, mean_slope, .fun = median, .desc = TRUE)
    )

  # ── Plot 1: faceted pre / post slope per metric, archetype-coloured
  plt_arch_slopes <- ggplot(
    arch_slope_df,
    aes(x = mean_slope, y = metric_disp, color = archetype)
  ) +
    geom_pointrange(
      aes(xmin = lo_slope, xmax = hi_slope),
      position = position_dodge(0.6),
      size = 0.35, linewidth = 0.5
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    facet_wrap(~phase, scales = "free_x") +
    scale_color_brewer(palette = "Set1") +
    theme_bw(base_size = 14) +
    labs(
      x     = "Mean GP first derivative (latent units / year)",
      y     = NULL,
      color = "Archetype"
    ) +
    scale_y_discrete(expand = expansion(mult = c(0.1, 0.1))) +
    theme(
      legend.position  = "right",
      strip.background = element_blank(),
      strip.text       = element_text(face = "bold")
    )

  ggsave(
    file.path(plots_dir, "latent_space", "mcmc", "archetype_slopes.png"),
    plt_arch_slopes, width = 12, height = 7
  )

  # ── Plot 2: asymmetry ratio (|pre| / |post|) — > 1 means faster decline
  arch_slope_wide <- arch_slope_df |>
    select(archetype, metric_disp, phase, mean_slope) |>
    pivot_wider(names_from = phase, values_from = mean_slope) |>
    mutate(
      skew_ratio  = `Pre-peak` / abs(pmin(`Post-peak`, -1e-6)),
      metric_disp = fct_reorder(metric_disp, skew_ratio, .fun = median, .desc = TRUE)
    )

  plt_arch_ratio <- ggplot(
    arch_slope_wide,
    aes(x = skew_ratio, y = metric_disp, color = archetype)
  ) +
    geom_point(position = position_dodge(0.5), size = 2.5) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
    scale_color_brewer(palette = "Set1") +
    theme_bw(base_size = 14) +
    labs(
      x     = "Pre-peak slope / |Post-peak slope|  (>1 = faster decline than ascent)",
      y     = NULL,
      color = "Archetype"
    ) +
    scale_y_discrete(expand = expansion(mult = c(0.1, 0.1))) +
    theme(legend.position = "right")

  ggsave(
    file.path(plots_dir, "latent_space", "mcmc", "archetype_slope_ratio.png"),
    plt_arch_ratio, width = 10, height = 7
  )

} # end archetype slopes

# Archetype average curves via Frechet mean (fdasrvf)
frechet_mean_curve_basic <- function(time, mat) {
  if (ncol(mat) == 0) {
    return(rep(NA_real_, length(time)))
  }
  rowMeans(mat, na.rm = TRUE)
}

frechet_mean_curve_elastic <- function(time, mat) {
  if (ncol(mat) == 0) {
    return(rep(NA_real_, length(time)))
  }
  if (ncol(mat) == 1) {
    return(as.numeric(mat[, 1]))
  }
  # fdasrvf expects time in [0,1]
  t_norm <- (time - min(time)) / (max(time) - min(time))
  tryCatch(
    {
      warp_res <- fdasrvf::time_warping(f = mat, time = t_norm, parallel = FALSE)
      as.numeric(warp_res$fmean)
    },
    error = function(e) {
      warning(paste0("SRV Frechet mean failed: ", e$message, " — falling back to arithmetic mean"))
      rowMeans(mat, na.rm = TRUE)
    }
  )
}

interp_curve_matrix <- function(time, mat) {
  apply(mat, 2, function(col) {
    if (all(is.na(col))) {
      return(rep(NA_real_, length(col)))
    }
    approx(x = time[!is.na(col)], y = col[!is.na(col)], xout = time, rule = 2)$y
  })
}

srvf_transform <- function(time, mat) {
  if (length(time) < 2 || ncol(mat) == 0) {
    return(matrix(numeric(0), nrow = 0, ncol = ncol(mat)))
  }
  dt <- diff(time)
  deriv <- apply(mat, 2, function(col) diff(col) / dt)
  q <- sign(deriv) * sqrt(abs(deriv))
  sweep(q, 1, sqrt(dt), "*")
}

transform_metric_mu <- function(metric, mu) {
  dplyr::case_when(
    metric %in% c("fg2m", "ftm", "games", "fg3m", "retirement") ~ plogis(mu),
    metric %in% c("obpm", "dbpm") ~ mu,
    metric %in% c("pct_minutes") ~ plogis(mu) * 48,
    metric %in% c("usg") ~ plogis(mu),
    .default = exp(mu) * 36
  )
}

metrics_for_archetypes <- c("obpm", "ast", "blk", "pct_minutes")

# Curve-based clustering using Frechet mean curves across 16 metrics
metrics_for_curve_cluster <- c(
  "obpm", "dbpm", "games", "pct_minutes",
  "blk", "ast", "tov", "oreb", "dreb", "stl",
  "usg", "fg2a", "fg3a", "fta", "fg2m", "fg3m"
)

all_srv_metrics <- union(metrics_for_archetypes, metrics_for_curve_cluster)

# Raw posterior mean curves (for basic Frechet mean, which shows observed-scale values).
posterior_mu_means <- posterior_mu_data |>
  filter(metric %in% metrics_for_archetypes) |>
  group_by(player, metric, age) |>
  summarize(mu = mean(value, na.rm = TRUE), .groups = "drop")

# Per-draw normalisation → per-player elastic Frechet mean across draws.
# Each draw is shifted to start at 0 then divided by sqrt(TV) = sqrt(sum|f'(t)|),
# giving unit SRVF L2 norm. The elastic Frechet mean across draws is then computed
# in parallel, so the resulting curve per player is a proper shape summary.
posterior_mu_means_norm <- if (!is.null(posterior_first_deriv_ds)) {
  arc_per_draw <- posterior_first_deriv_ds |>
    filter(metric %in% all_srv_metrics) |>
    group_by(chain, sample, player, metric) |>
    summarize(tv = sum(abs(value)), .groups = "drop") |>
    collect()

  start_per_draw <- posterior_mu_data |>
    filter(metric %in% all_srv_metrics) |>
    group_by(chain, sample, player, metric) |>
    slice_min(age, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(chain, sample, player, metric, start_val = value)

  posterior_mu_data |>
    filter(metric %in% all_srv_metrics) |>
    left_join(start_per_draw, by = c("chain", "sample", "player", "metric")) |>
    left_join(arc_per_draw,   by = c("chain", "sample", "player", "metric")) |>
    mutate(
      tv    = if_else(is.na(tv) | tv == 0, 1, tv),
      value = (value - start_val) / tv
    ) |>
    group_by(player, metric, age) |>
    summarize(mu = mean(value, na.rm = TRUE), .groups = "drop")
} else {
  posterior_mu_means
}

compute_archetype_curves <- function(data, method_fn, method_label, show_observed = TRUE) {
  player_curves <- data |>
    filter(metric %in% metrics_for_archetypes) |>
    left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id")) |>
    filter(!is.na(archetype))

  player_curves |>
    mutate(player = as.character(player)) |>
    group_by(archetype, metric) |>
    group_modify(~ {
      wide <- .x |>
        select(age, player, mu) |>
        pivot_wider(names_from = player, values_from = mu) |>
        arrange(age)
      time <- wide$age
      mat <- wide |>
        select(-age) |>
        as.matrix()
      if (ncol(mat) == 0) {
        return(tibble())
      }
      mat <- interp_curve_matrix(time, mat)
      mu <- method_fn(time, mat)
      tibble(age = time, mu = mu, n_players = ncol(mat))
    }) |>
    ungroup() |>
    mutate(
      observed_mu = if (show_observed) transform_metric_mu(metric, mu) else mu,
      metric = toupper(metric),
      metric = case_when(metric == "PCT_MINUTES" ~ "MPG", .default = metric),
      method = method_label
    )
}

archetype_frechet_basic   <- compute_archetype_curves(posterior_mu_means,      frechet_mean_curve_basic, "Basic Frechet Mean",    show_observed = TRUE)
archetype_frechet_elastic <- compute_archetype_curves(posterior_mu_means_norm,  frechet_mean_curve_basic, "SRV Normalised (mean)", show_observed = FALSE)

archetype_frechet_curves <- bind_rows(archetype_frechet_basic, archetype_frechet_elastic)

frechet_dir <- file.path(plots_dir, "latent_space", "frechet")
frechet_mcmc_dir <- file.path(frechet_dir, "mcmc")
frechet_tables_dir <- file.path(frechet_dir, "tables")
frechet_map_dir <- file.path(frechet_dir, "map")
dir.create(frechet_mcmc_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(frechet_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(frechet_map_dir, recursive = TRUE, showWarnings = FALSE)

archetype_curve_dir <- frechet_mcmc_dir

plot_archetype_curves <- function(curve_df, title_suffix, y_label = "Metric value") {
  curve_df |>
    ggplot(aes(x = age, y = observed_mu, color = archetype)) +
    geom_line(linewidth = 1) +
    facet_wrap(~ metric, scales = "free_y") +
    theme_bw(base_size = 14) +
    labs(
      title = paste("Archetype Average Curves (Frechet mean):", title_suffix),
      x = "Age",
      y = y_label,
      color = "Archetype"
    )
}

archetype_curve_plot_basic <- plot_archetype_curves(
  archetype_frechet_curves |> filter(method == "Basic Frechet Mean"),
  "basic"
)
archetype_curve_plot_elastic <- plot_archetype_curves(
  archetype_frechet_curves |> filter(method == "SRV Normalised (mean)"),
  "elastic",
  y_label = "Normalised shape (TV units)"
)

ggsave(file.path(archetype_curve_dir, "archetype_frechet_curves_basic.png"),
       archetype_curve_plot_basic, width = 12, height = 8)
ggsave(file.path(archetype_curve_dir, "archetype_frechet_curves_elastic.png"),
       archetype_curve_plot_elastic, width = 12, height = 8)

# Curve-based clustering: SRVF distances on normalised per-player Frechet mean shapes.
curve_dist_list <- posterior_mu_means_norm |>
  filter(metric %in% metrics_for_curve_cluster) |>
  group_by(metric) |>
  group_map(~ {
    wide <- .x |>
      mutate(player = as.character(player)) |>
      select(age, player, mu) |>
      pivot_wider(names_from = player, values_from = mu) |>
      arrange(age)
    time <- wide$age
    mat <- wide |>
      select(-age) |>
      as.matrix()
    if (ncol(mat) <= 1) {
      return(list(players = character(0), dist = matrix(0, 0, 0)))
    }
    mat <- interp_curve_matrix(time, mat)
    q <- srvf_transform(time, mat)
    list(
      players = colnames(mat),
      dist = as.matrix(dist(t(q)))
    )
  })

curve_players_common <- Reduce(
  intersect,
  lapply(curve_dist_list, function(x) x$players)
)

curve_dist_avg <- matrix(
  0,
  nrow = length(curve_players_common),
  ncol = length(curve_players_common),
  dimnames = list(curve_players_common, curve_players_common)
)

for (dist_item in curve_dist_list) {
  if (length(dist_item$players) == 0) {
    next
  }
  dist_mat <- dist_item$dist[curve_players_common, curve_players_common, drop = FALSE]
  curve_dist_avg <- curve_dist_avg + dist_mat
}
curve_dist_avg <- curve_dist_avg / length(curve_dist_list)

id_to_name <- latent_space |> distinct(id, name) |> deframe()
curve_dist_names <- curve_dist_avg
dimnames(curve_dist_names) <- lapply(
  dimnames(curve_dist_avg),
  function(ids) ifelse(ids %in% names(id_to_name), id_to_name[ids], ids)
)
if (length(curve_players_common) < 2) {
  message("Frechet curve clustering skipped: fewer than 2 players common across all metrics.")
  curve_archetype_lookup <- tibble(id = character(0), curve_archetype = factor(character(0)))
} else {
curve_hc <- hclust(as.dist(curve_dist_names), method = "ward.D2")
curve_archetype_labels <- cutree(curve_hc, k = find_k_dendrogram(curve_hc))
curve_archetype_lookup <- tibble(
  id = curve_players_common,
  curve_archetype = factor(unname(curve_archetype_labels))
)

ggsave(
  file.path(frechet_map_dir, "frechet_curve_cluster_dendrogram.png"),
  plot_dendrogram_gg(curve_hc, notable_names,
                     "Frechet curve clustering — mean distance across metrics"),
  width = 14, height = 18
)

player_lookup_curve <- data |>
  group_by(id) |>
  summarize(name = first(name), .groups = "drop")

# Define write_neighbor_tex function before use in group_walk()
write_neighbor_tex <- function(df, focal_name, caption, out_path,
                               archetype_col = "archetype") {
  label <- paste0("tab:", tools::file_path_sans_ext(basename(out_path)))
  rows <- df |> arrange(rank_mean)
  body <- pmap_chr(rows, function(neighbor_name, neighbor_pos, rank_mean, ...) {
    arch <- list(...)[[archetype_col]]
    sprintf("    %s & %s & %s & %s %s", rank_mean, neighbor_name, neighbor_pos, arch, "\\\\")
  }) |> paste(collapse = "\n")
  latex <- paste0(
    "\\begin{table}[htbp]\n",
    "  \\centering\\small\n",
    "  \\begin{tabular}{rllc}\n",
    "    \\toprule\n",
    "    Rank & Player & Position & Archetype ", "\\\\", "\n",
    "    \\midrule\n",
    body, "\n",
    "    \\bottomrule\n",
    "  \\end{tabular}\n",
    "  \\caption{", caption, "}\n",
    "  \\label{", label, "}\n",
    "\\end{table}\n"
  )
  writeLines(latex, out_path)
}

focal_ids_curve <- player_lookup_curve |>
  filter(name %in% posterior_plot_names) |>
  filter(id %in% curve_players_common) |>
  pull(id)

curve_neighbors <- map_dfr(focal_ids_curve, function(fid) {
  drow <- curve_dist_avg[as.character(fid), ]
  drow[as.character(fid)] <- Inf
  top5_ids <- names(sort(drow)[1:5])
  top5_dists <- as.numeric(sort(drow)[1:5])
  tibble(
    focal_id = fid,
    neighbor_id = top5_ids,
    rank_mean = 1:5,
    dist_mean = top5_dists
  )
}) |>
  left_join(player_lookup_curve |> rename(focal_name = name),
            by = c("focal_id" = "id")) |>
  left_join(
    latent_space |>
      select(id, neighbor_name = name, neighbor_pos = position_group),
    by = c("neighbor_id" = "id")
  ) |>
  left_join(curve_archetype_lookup, by = c("neighbor_id" = "id"))

curve_neighbors |>
  group_by(focal_id, focal_name) |>
  group_walk(~ {
    fname <- gsub("[^A-Za-z0-9_]", "_", .y$focal_name)
    write_neighbor_tex(
      df            = .x |> select(neighbor_name, neighbor_pos, curve_archetype, rank_mean),
      focal_name    = .y$focal_name,
      caption       = paste0("Five nearest neighbors of ", .y$focal_name, " by mean Fr\\'{e}chet curve distance across metrics. Identified purely from trajectory shape without dimensionality reduction."),
      out_path      = file.path(frechet_tables_dir, glue("{fname}_frechet_curve_neighbors.tex")),
      archetype_col = "curve_archetype"
    )
  })

frechet_neighbor_curve_means <- posterior_mu_means_norm |>
  filter(metric %in% metrics_for_curve_cluster) |>
  mutate(
    observed_mu = mu,
    metric = toupper(metric),
    metric = case_when(metric == "PCT_MINUTES" ~ "MPG", .default = metric)
  )

plot_neighbor_curves_frechet <- function(focal_id, focal_name, neighbor_ids) {
  plot_ids <- unique(c(focal_id, neighbor_ids))
  n_neighbors <- length(neighbor_ids)

  neighbor_rank_df <- tibble(
    id = neighbor_ids,
    neighbor_rank = seq_along(neighbor_ids)
  )

  neighbor_names_ordered <- neighbor_rank_df |>
    left_join(latent_space |> select(id, name), by = "id") |>
    arrange(neighbor_rank) |>
    pull(name)

  plot_df <- frechet_neighbor_curve_means |>
    filter(player %in% plot_ids) |>
    left_join(latent_space |> select(id, name, position_group), by = c("player" = "id")) |>
    left_join(neighbor_rank_df, by = c("player" = "id")) |>
    mutate(
      role = if_else(player == focal_id, "Focal", "Neighbor"),
      name_label = case_when(
        role == "Focal" ~ paste0(name, " (", position_group, ")"),
        TRUE ~ paste0(neighbor_rank, ". ", name, " (", position_group, ")")
      )
    )

  focal_label     <- unique(plot_df$name_label[plot_df$role == "Focal"])
  neighbor_labels <- plot_df |>
    filter(role == "Neighbor") |>
    distinct(neighbor_rank, name_label) |>
    arrange(neighbor_rank) |>
    pull(name_label)
  labels_ordered  <- c(focal_label, neighbor_labels)
  plot_df <- plot_df |>
    mutate(name_label = factor(name_label, levels = labels_ordered))

  line_colors <- setNames(rep("#1a1a1a", length(labels_ordered)), labels_ordered)

  label_df <- plot_df |>
    filter(role == "Neighbor") |>
    group_by(metric, player, name_label, neighbor_rank) |>
    slice_max(age, n = 1, with_ties = FALSE) |>
    ungroup()

  ggplot(plot_df, aes(x = age, y = observed_mu, group = name_label,
                      color = name_label, linetype = role,
                      alpha = role, linewidth = role)) +
    geom_line() +
    geom_text(
      data = label_df,
      aes(label = neighbor_rank),
      size = 6, hjust = -0.2, show.legend = FALSE
    ) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_color_manual(values = line_colors, name = "Player") +
    scale_linetype_manual(values = c(Neighbor = "dashed", Focal = "solid")) +
    scale_alpha_manual(values = c(Neighbor = 0.45, Focal = 1.0)) +
    scale_linewidth_manual(values = c(Neighbor = 0.6, Focal = 1.2)) +
    guides(
      linetype  = "none",
      alpha     = "none",
      linewidth = "none",
      color = guide_legend(
        title = "Player",
        override.aes = list(
          linetype  = c("solid", rep("blank", n_neighbors)),
          linewidth = c(1.2,     rep(0,       n_neighbors))
        )
      )
    ) +
    theme_bw(base_size = 22) +
    labs(
      title = glue("Frechet Neighbor Curves: {focal_name}"),
      x = "Age",
      y = "Metric value"
    )
}

curve_neighbors |>
  group_by(focal_id, focal_name) |>
  group_walk(~ {
    neighbor_ids <- .x |> arrange(rank_mean) |> pull(neighbor_id)
    plt <- plot_neighbor_curves_frechet(.y$focal_id, .y$focal_name, neighbor_ids)
    fname <- gsub("[^A-Za-z0-9_]", "_", .y$focal_name)
    ggsave(
      file.path(frechet_map_dir, glue("frechet_neighbors_curves_{fname}.png")),
      plt,
      width = 12,
      height = 8
    )
  })

} # end else (curve_players_common >= 2)

# ── 6. Archetype-coloured functional PCA plot ────────────────────────────────
fpc_arch <- functional_pca_embedding |>
  left_join(posterior_mean_latent |> select(id, archetype), by = c("player" = "id"))

functional_pca_plt_archetype <- fpc_arch |>
  filter(PCA1 <= 20 & PCA2 <= 20) |>
  ggplot(aes(x = PCA1, y = PCA2, color = archetype)) +
  geom_point(aes(alpha = minutes)) +
  scale_alpha(range = c(0, 1)) +
  geom_text_repel(
    data = fpc_arch |> filter(name %in% posterior_plot_names),
    aes(label = plot_name(name), x = PCA1, y = PCA2),
    size = 2, fontface = "bold", max.overlaps = 20,
    min.segment.length = Inf,
    inherit.aes = FALSE
  ) +
  theme_bw(base_size = 14) + scale_colour_manual(values = arch_colour_pal) +
  labs(title = "Latent Space — Archetype Clusters",
       x = "PC 1", y = "PC 2", color = "Archetype", alpha = "Minutes")
ggsave(file.path(plots_dir, "latent_space", "map", "latent_space_archetypes.png"),
       functional_pca_plt_archetype, width = 12, height = 8)

# ── 6b. PCA of posterior mean latent coords, coloured by archetype ───────────
latent_pca        <- prcomp(phi_mat, center = FALSE, scale. = FALSE)  # phi_mat already scaled
latent_pca_df     <- as_tibble(latent_pca$x[, 1:2]) |>
  bind_cols(posterior_mean_latent |> arrange(id) |> select(id, name, archetype)) |>
  left_join(latent_space |> select(id, position_group, minutes), by = "id")

latent_pca_plt <- latent_pca_df |>
  ggplot(aes(x = PC1, y = PC2, color = archetype)) +
  geom_point(aes(alpha = minutes)) +
  scale_alpha(range = c(0, 1)) +
  geom_text_repel(
    data = latent_pca_df |> filter(name %in% posterior_plot_names),
    aes(label = plot_name(name)),
    size = 2, fontface = "bold", max.overlaps = 20,
    min.segment.length = Inf
  ) +
  theme_bw(base_size = 14) + scale_colour_manual(values = arch_colour_pal) +
  labs(title = "PCA of Posterior Mean Latent Coordinates — Archetype Clusters",
       x = glue("PC 1 ({round(summary(latent_pca)$importance[2,1]*100,1)}% var)"),
       y = glue("PC 2 ({round(summary(latent_pca)$importance[2,2]*100,1)}% var)"),
       color = "Archetype", alpha = "Minutes")

ggsave(file.path(plots_dir, "latent_space", "map", "latent_pca_archetypes.png"),
       latent_pca_plt, width = 12, height = 8)

# ── 6c. Scree plot for latent PCA ────────────────────────────────────────────
pca_var_explained <- latent_pca$sdev^2 / sum(latent_pca$sdev^2)
pca_cumvar        <- cumsum(pca_var_explained)
pca_scree_df      <- tibble(
  PC            = seq_along(pca_var_explained),
  var_explained = pca_var_explained,
  cumvar        = pca_cumvar
)

latent_pca_scree <- ggplot(pca_scree_df, aes(x = PC)) +
  geom_col(aes(y = var_explained), fill = "steelblue", alpha = 0.8) +
  geom_line(aes(y = cumvar), color = "darkred", linewidth = 0.8) +
  geom_point(aes(y = cumvar), color = "darkred", size = 1.5) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    sec.axis = sec_axis(~ ., name = "Cumulative variance", labels = scales::percent_format(accuracy = 1))
  ) +
  labs(
    title = "Latent PCA — Eigenvalue Scree Plot",
    x = "Principal Component", y = "Variance explained"
  ) +
  theme_bw(base_size = 14)

ggsave(file.path(plots_dir, "latent_space", "map", "latent_pca_scree.png"),
       latent_pca_scree, width = 9, height = 5)

# ── 6d. PC3 vs PC4 scatter coloured by archetype ─────────────────────────────
latent_pca_df34 <- as_tibble(latent_pca$x[, 3:4]) |>
  bind_cols(posterior_mean_latent |> arrange(id) |> select(id, name, archetype)) |>
  left_join(latent_space |> select(id, minutes), by = "id")

pc3_var <- round(summary(latent_pca)$importance[2, 3] * 100, 1)
pc4_var <- round(summary(latent_pca)$importance[2, 4] * 100, 1)

latent_pca_pc34 <- latent_pca_df34 |>
  ggplot(aes(x = PC3, y = PC4, color = archetype)) +
  geom_point(aes(alpha = minutes)) +
  scale_alpha(range = c(0, 1)) +
  geom_text_repel(
    data = latent_pca_df34 |> filter(name %in% posterior_plot_names),
    aes(label = plot_name(name)),
    size = 2, fontface = "bold", max.overlaps = 20
  ) +
  theme_bw(base_size = 14) + scale_colour_manual(values = arch_colour_pal) +
  labs(
    title = "PCA of Posterior Mean Latent Coordinates — PC3 vs PC4",
    x = glue("PC 3 ({pc3_var}% var)"),
    y = glue("PC 4 ({pc4_var}% var)"),
    color = "Archetype", alpha = "Minutes"
  )

ggsave(file.path(plots_dir, "latent_space", "map", "latent_pca_pc34_archetypes.png"),
       latent_pca_pc34, width = 12, height = 8)

# ── 7. Embedding uncertainty from aligned MCMC samples ───────────────────────
map_positions <- phi_X |> arrange(id) |> select(id, all_of(dim_cols))

embedding_uncertainty <- rotated_posterior_df |>
  group_by(id, name) |>
  group_modify(~ {
    samp_mat <- .x |> select(all_of(dim_cols)) |> as.matrix()
    map_pos  <- map_positions |> filter(id == .y$id) |> select(all_of(dim_cols)) |> as.matrix()
    dists    <- sqrt(rowSums(sweep(samp_mat, 2, map_pos)^2))
    cov_mat  <- cov(samp_mat)
    tibble(
      mean_dist_to_map = mean(dists),
      sd_dist_to_map   = sd(dists),
      cov_vol          = det(cov_mat)^(1 / ncol(samp_mat))
    )
  }) |>
  ungroup() |>
  left_join(latent_space |> select(id, position_group, archetype, minutes), by = "id")

uncertainty_gt <- embedding_uncertainty |>
  filter(name %in% posterior_plot_names) |>
  arrange(desc(mean_dist_to_map)) |>
  select(name, position_group, archetype, mean_dist_to_map, sd_dist_to_map) |>
  gt() |>
  fmt_number(columns = c(mean_dist_to_map, sd_dist_to_map), decimals = 3) |>
  cols_label(
    name             = "Player",
    position_group   = "Position",
    archetype        = "Archetype",
    mean_dist_to_map = "Mean dist to MAP",
    sd_dist_to_map   = "SD"
  ) |>
  tab_header(title = "Latent Embedding Uncertainty (Procrustes-aligned MCMC samples)")

writeLines(as.character(as_latex(uncertainty_gt)),
           file.path(plots_dir, "latent_space", "tables", "embedding_uncertainty.tex"))

# ── 8. Nearest-neighbor tables ────────────────────────────────────────────────
focal_ids <- latent_space |> filter(name %in% posterior_plot_names) |> pull(id)

# 8A: Point-estimate neighbors from posterior mean positions
pm_mat <- posterior_mean_latent |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
pm_ids <- posterior_mean_latent |> arrange(id) |> pull(id)
D_mean <- as.matrix(dist(pm_mat))
rownames(D_mean) <- colnames(D_mean) <- as.character(pm_ids)

nn_mean <- map_dfr(focal_ids, function(fi) {
  drow <- D_mean[as.character(fi), ]
  drow[as.character(fi)] <- Inf
  top5_ids   <- names(sort(drow)[1:5])
  top5_dists <- as.numeric(sort(drow)[1:5])
  tibble(focal_id    = fi,
         neighbor_id = top5_ids,
         rank_mean   = 1:5,
         dist_mean   = top5_dists)
}) |>
  left_join(latent_space |> select(id, focal_name = name),
            by = c("focal_id" = "id")) |>
  left_join(latent_space |> select(id, neighbor_name = name,
                                    neighbor_pos = position_group, archetype),
            by = c("neighbor_id" = "id"))

# 8B: Per-sample neighbor frequency across aligned MCMC snapshots
n_snapshots <- rotated_posterior_df |> distinct(chain, sample) |> nrow()

nn_freq_raw <- rotated_posterior_df |>
  group_by(chain, sample) |>
  group_modify(~ {
    mat  <- .x |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
    ids  <- .x |> arrange(id) |> pull(id)
    fidx <- which(ids %in% focal_ids)
    D    <- as.matrix(dist(mat))
    map_dfr(fidx, function(i) {
      drow    <- D[i, ]
      drow[i] <- Inf
      top5    <- ids[order(drow)[1:5]]
      tibble(focal_id = ids[i], neighbor_id = top5)
    })
  }) |>
  ungroup()

nn_freq <- nn_freq_raw |>
  group_by(focal_id, neighbor_id) |>
  summarise(freq_top5 = n() / n_snapshots, .groups = "drop")

# 8C: Join point-estimate + frequency
nn_top5 <- nn_mean |>
  left_join(nn_freq, by = c("focal_id", "neighbor_id")) |>
  mutate(freq_top5 = replace_na(freq_top5, 0))

# 8D: Save one .tex file per focal player
nn_top5 |>
  group_by(focal_id, focal_name) |>
  group_walk(~ {
    fname <- gsub("[^A-Za-z0-9_]", "_", .y$focal_name)
    write_neighbor_tex(
      df           = .x |> select(neighbor_name, neighbor_pos, archetype, rank_mean),
      focal_name   = .y$focal_name,
      caption      = glue("Five nearest neighbors of {.y$focal_name} in the Procrustes-aligned posterior latent space. Rank is by posterior mean embedding distance."),
      out_path     = file.path(plots_dir, "latent_space", "tables", glue("{fname}_neighbors.tex")),
      archetype_col = "archetype"
    )
  })

# ── Modality-specific latent analyses (peak age/value + curvature) ─────────
run_modality_latent_analysis <- function(tag, posterior_df, phi_ref, plots_dir,
                                         posterior_plot_names) {
  if (is.null(posterior_df) || is.null(phi_ref)) {
    return(invisible(NULL))
  }
  dim_cols <- names(posterior_df)[startsWith(names(posterior_df), "Dim")]
  if (length(dim_cols) == 0) {
    return(invisible(NULL))
  }

  mod_dir <- file.path(plots_dir, "latent_space", "modalities", tag)
  dir.create(mod_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(mod_dir, "tables"), recursive = TRUE, showWarnings = FALSE)

  phi_ref_ordered <- phi_ref |>
    arrange(id) |>
    select(all_of(dim_cols)) |>
    as.matrix()

  rotated_df <- posterior_df |>
    mutate(chain = factor(chain)) |>
    group_by(chain, sample) |>
    group_modify(~ {
      B <- .x |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
      R <- procrustes_rotation(B, phi_ref_ordered)
      B_rot <- B %*% R
      colnames(B_rot) <- dim_cols
      .x |>
        arrange(id) |>
        select(-all_of(dim_cols)) |>
        bind_cols(as_tibble(B_rot))
    }) |>
    ungroup()

  posterior_mean_latent <- rotated_df |>
    group_by(id, name) |>
    summarise(across(all_of(dim_cols), mean), .groups = "drop")

  phi_mat <- posterior_mean_latent |>
    arrange(id) |>
    select(all_of(dim_cols)) |>
    as.matrix()
  rownames(phi_mat) <- posterior_mean_latent |> arrange(id) |> pull(name)

  # Per-dimension variance across players in the modality-specific (unscaled) coords.
  # High variance = that dim is both variable across players and strongly weighted
  # by this modality's gamma. Used directly for Ward clustering (no re-scaling).
  dim_var     <- apply(phi_mat, 2, var)
  top3_dims   <- names(sort(dim_var, decreasing = TRUE))[1:3]
  dim_var_tbl <- tibble(dimension = names(dim_var), variance = dim_var) |>
    arrange(desc(variance))
  message(glue("  [{tag}] top-3 dims by variance: {paste(top3_dims, collapse = ', ')}"))

  phi_dist <- dist(phi_mat)   # Ward clustering on unscaled modality-weighted coords
  hc_latent <- hclust(phi_dist, method = "ward.D2")

  wss_sil <- map_dfr(2:10, function(k) {
    labs <- cutree(hc_latent, k)
    wss <- sum(sapply(unique(labs), function(cl) {
      sub <- phi_mat[labs == cl, , drop = FALSE]
      sum(scale(sub, scale = FALSE)^2)
    }))
    sil <- mean(silhouette(labs, phi_dist)[, "sil_width"])
    tibble(k = k, wss = wss, silhouette = sil)
  })

  wss_plt <- ggplot(wss_sil, aes(x = k, y = wss)) +
    geom_line() + geom_point() +
    labs(title = glue("Within-cluster SS vs. k ({tag})"), x = "k", y = "WSS") + theme_bw(base_size = 14)
  sil_plt <- ggplot(wss_sil, aes(x = k, y = silhouette)) +
    geom_line() + geom_point() +
    labs(title = glue("Mean silhouette vs. k ({tag})"), x = "k", y = "Avg silhouette") + theme_bw(base_size = 14)
  ggsave(file.path(mod_dir, "archetype_k_diagnostics.png"),
         wss_plt + sil_plt, width = 12, height = 5)

  ggsave(
    file.path(mod_dir, "archetype_dendrogram.png"),
    plot_dendrogram_gg(hc_latent, notable_names, glue("Ward clustering — {tag}")),
    width = 14, height = 18
  )

  k_mod <- find_k_dendrogram(hc_latent)
  message(glue("  [{tag}] optimal k by dendrogram gap: {k_mod}"))
  mod_colour_pal <- if (k_mod <= 9L) {
    RColorBrewer::brewer.pal(max(3L, k_mod), "Set1")[seq_len(k_mod)]
  } else {
    scales::hue_pal()(k_mod)
  }
  archetype_labels <- cutree(hc_latent, k = k_mod)
  posterior_mean_latent <- posterior_mean_latent |>
    arrange(id) |>
    mutate(archetype = factor(unname(archetype_labels)))

  # ── Dimension variance bar chart (modality-level) ────────────────────────────
  dim_var_plt <- ggplot(dim_var_tbl, aes(x = reorder(dimension, variance), y = variance)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(
      title    = glue("Per-dimension variance in modality-specific coords ({tag})"),
      subtitle = glue("Top dims: {paste(top3_dims, collapse=', ')}"),
      x        = "Latent dimension",
      y        = "Variance across players"
    ) +
    theme_bw(base_size = 14)
  ggsave(file.path(mod_dir, "dim_variance.png"), dim_var_plt, width = 7, height = 5)

  # ── Per-archetype player representatives (top-5 by career minutes) ────────────
  arch_reps_mod <- posterior_mean_latent |>
    left_join(
      latent_space |> select(id, position_group, minutes) |>
        group_by(id) |> slice(1L) |> ungroup(),
      by = "id"
    ) |>
    filter(!is.na(archetype)) |>
    group_by(archetype) |>
    arrange(desc(minutes)) |>
    mutate(rank = row_number()) |>
    filter(rank <= 5L) |>
    ungroup()

  arch_rep_blocks_mod <- map_chr(levels(arch_reps_mod$archetype), function(a) {
    block   <- arch_reps_mod |> filter(archetype == a)
    pos_mix <- block |>
      count(position_group, sort = TRUE) |>
      slice_head(n = 2L) |>
      pull(position_group) |>
      paste(collapse = "/")
    header <- glue(
      "    \\multicolumn{{4}}{{l}}{{\\textit{{Archetype {a} ({pos_mix})}}}}"
    )
    rows <- pmap_chr(
      block |> select(rank, name, position_group, minutes),
      function(rank, name, position_group, minutes, ...) {
        mins_str <- if (is.na(minutes)) "---"
                    else formatC(as.integer(minutes), format = "d", big.mark = ",")
        glue("    {rank} & {name} & {position_group} & {mins_str} \\\\")
      }
    )
    paste(c(paste0(header, " \\\\"), "    \\midrule", rows), collapse = "\n")
  })

  tag_tex <- gsub("_", "-", tag)   # underscores invalid in LaTeX text mode

  arch_rep_tex_mod <- paste0(
    "\\begin{table}[htbp]\n",
    "  \\centering\\small\n",
    "  \\begin{tabular}{rlll}\n",
    "    \\toprule\n",
    "    Rank & Player & Pos. & Career Min. \\\\\n",
    "    \\midrule\n",
    paste(arch_rep_blocks_mod, collapse = "\n    \\midrule[0.4pt]\n"), "\n",
    "    \\bottomrule\n",
    "  \\end{tabular}\n",
    "  \\caption{Top 5 players by career minutes per archetype --- ",
    tag_tex, " modality ($k=", k_mod, "$ archetypes).",
    " Top-3 latent dimensions by variance across players in this modality:",
    " ", paste(top3_dims, collapse = ", "), ".}\n",
    "  \\label{tab:arch_reps_", tag, "}\n",
    "\\end{table}\n"
  )
  writeLines(arch_rep_tex_mod, file.path(mod_dir, "archetype_representatives.tex"))

  # ── Per-archetype bullet prose (mirrors global archetype_bullets.tex) ─────────
  arch_reps_summary <- arch_reps_mod |>
    group_by(archetype) |>
    slice_min(rank, n = 3L) |>
    summarise(reps = paste(name, collapse = ", "), .groups = "drop")

  tag_tex <- gsub("_", "-", tag)   # underscores invalid in LaTeX text mode

  arch_bullets_mod <- arch_reps_summary |>
    arrange(archetype) |>
    mutate(item = glue(
      "\\item \\textbf{{Archetype {archetype}}} ---",
      " \\emph{{Representatives (by career minutes):}} {reps}.",
      " Primary separating dimensions in the {tag_tex} modality: {paste(top3_dims, collapse = ', ')}."
    ))

  writeLines(
    c("\\begin{itemize}", arch_bullets_mod$item, "\\end{itemize}"),
    file.path(mod_dir, "archetype_bullets.tex")
  )

  # PCA of modality-specific posterior mean latent coords, coloured by archetype
  latent_pca_mod <- prcomp(phi_mat, center = FALSE, scale. = FALSE)
  pca_df <- as_tibble(latent_pca_mod$x[, 1:2]) |>
    bind_cols(posterior_mean_latent |> arrange(id) |> select(id, name, archetype))
  label_df <- pca_df |> filter(name %in% posterior_plot_names)
  pca_label_layer <- if (
    nrow(label_df) > 0 &&
      is.finite(diff(range(pca_df$PC1, na.rm = TRUE))) && diff(range(pca_df$PC1, na.rm = TRUE)) > 1e-8 &&
      is.finite(diff(range(pca_df$PC2, na.rm = TRUE))) && diff(range(pca_df$PC2, na.rm = TRUE)) > 1e-8
  ) {
    geom_text_repel(
      data = label_df,
      aes(label = plot_name(name)),
      size = 2, fontface = "bold", max.overlaps = 20,
      box.padding = 0.25, point.padding = 0.15,
      min.segment.length = 0
    )
  } else if (nrow(label_df) > 0) {
    geom_text(
      data = label_df,
      aes(label = plot_name(name)),
      size = 2, fontface = "bold",
      vjust = -0.6
    )
  } else {
    NULL
  }
  pca_plt <- pca_df |>
    ggplot(aes(x = PC1, y = PC2, color = archetype)) +
    geom_point(size = 1.5) +
    pca_label_layer +
    theme_bw(base_size = 14) + scale_colour_manual(values = mod_colour_pal) +
    labs(
      title = glue("PCA of Modality-Specific Latent Coords — {tag} (k={k_mod})"),
      x = glue("PC 1 ({round(summary(latent_pca_mod)$importance[2,1]*100,1)}% var)"),
      y = glue("PC 2 ({round(summary(latent_pca_mod)$importance[2,2]*100,1)}% var)"),
      color = "Archetype"
    )
  tryCatch(
    {
      ggsave(file.path(mod_dir, "archetype_pca.png"), pca_plt, width = 10, height = 7)
    },
    error = function(e) {
      warning(glue("Failed to save PCA plot with labels for {tag}; retrying without labels. Error: {e$message}"))
      pca_plt_no_labels <- pca_df |>
        ggplot(aes(x = PC1, y = PC2, color = archetype)) +
        geom_point(size = 1.5) +
        theme_bw(base_size = 14) + scale_colour_manual(values = mod_colour_pal) +
        labs(
          title = glue("PCA of Modality-Specific Latent Coords — {tag} (k={k_mod})"),
          x = glue("PC 1 ({round(summary(latent_pca_mod)$importance[2,1]*100,1)}% var)"),
          y = glue("PC 2 ({round(summary(latent_pca_mod)$importance[2,2]*100,1)}% var)"),
          color = "Archetype"
        )
      ggsave(file.path(mod_dir, "archetype_pca.png"), pca_plt_no_labels, width = 10, height = 7)
    }
  )

  focal_ids <- posterior_mean_latent |>
    filter(name %in% posterior_plot_names) |>
    pull(id)
  if (length(focal_ids) == 0) {
    return(invisible(NULL))
  }

  pm_mat <- posterior_mean_latent |>
    arrange(id) |>
    select(all_of(dim_cols)) |>
    as.matrix()
  pm_ids <- posterior_mean_latent |> arrange(id) |> pull(id)
  D_mean <- as.matrix(dist(pm_mat))
  rownames(D_mean) <- colnames(D_mean) <- as.character(pm_ids)

  nn_mean <- map_dfr(focal_ids, function(fi) {
    drow <- D_mean[as.character(fi), ]
    drow[as.character(fi)] <- Inf
    top5_ids <- names(sort(drow)[1:5])
    top5_dists <- as.numeric(sort(drow)[1:5])
    tibble(
      focal_id = fi,
      neighbor_id = top5_ids,
      rank_mean = 1:5,
      dist_mean = top5_dists
    )
  }) |>
    left_join(phi_ref |> select(id, focal_name = name),
              by = c("focal_id" = "id")) |>
    left_join(phi_ref |> select(id, neighbor_name = name, neighbor_pos = position_group),
              by = c("neighbor_id" = "id")) |>
    left_join(posterior_mean_latent |> select(id, archetype),
              by = c("neighbor_id" = "id"))

  n_snapshots <- rotated_df |> distinct(chain, sample) |> nrow()
  nn_freq_raw <- rotated_df |>
    group_by(chain, sample) |>
    group_modify(~ {
      mat <- .x |> arrange(id) |> select(all_of(dim_cols)) |> as.matrix()
      ids <- .x |> arrange(id) |> pull(id)
      fidx <- which(ids %in% focal_ids)
      if (length(fidx) == 0) {
        return(tibble())
      }
      D <- as.matrix(dist(mat))
      map_dfr(fidx, function(i) {
        drow <- D[i, ]
        drow[i] <- Inf
        top5 <- ids[order(drow)[1:5]]
        tibble(focal_id = ids[i], neighbor_id = top5)
      })
    }) |>
    ungroup()

  nn_freq <- nn_freq_raw |>
    group_by(focal_id, neighbor_id) |>
    summarise(freq_top5 = n() / n_snapshots, .groups = "drop")

  nn_top5 <- nn_mean |>
    left_join(nn_freq, by = c("focal_id", "neighbor_id")) |>
    mutate(freq_top5 = replace_na(freq_top5, 0))

  nn_top5 |>
    group_by(focal_id, focal_name) |>
    group_walk(~ {
      fname <- gsub("[^A-Za-z0-9_]", "_", .y$focal_name)
      write_neighbor_tex(
        df            = .x |> select(neighbor_name, neighbor_pos, archetype, rank_mean),
        focal_name    = .y$focal_name,
        caption       = glue("Five nearest neighbors of {.y$focal_name} ({tag} modality), Procrustes-aligned posterior mean."),
        out_path      = file.path(mod_dir, "tables", glue("{fname}_neighbors_{tag}.tex")),
        archetype_col = "archetype"
      )
    })

  invisible(list(
    rotated = rotated_df,
    posterior_mean = posterior_mean_latent
  ))
}

if (!is.null(posterior_latent_X_peak_age) && !is.null(phi_X_peak_age)) {
  run_modality_latent_analysis(
    "peak_age",
    posterior_latent_X_peak_age |>
      select(id, name, chain, sample, starts_with("Dim")),
    phi_X_peak_age,
    plots_dir,
    posterior_plot_names
  )
}

if (!is.null(posterior_latent_X_peak_value) && !is.null(phi_X_peak_value)) {
  run_modality_latent_analysis(
    "peak_value",
    posterior_latent_X_peak_value |>
      select(id, name, chain, sample, starts_with("Dim")),
    phi_X_peak_value,
    plots_dir,
    posterior_plot_names
  )
}

if (length(curvature_post_paths) > 0) {
  walk(curvature_post_paths, function(post_path) {
    m_id <- str_match(basename(post_path), "curvature_m([0-9]+)\\.parquet")[, 2]
    if (is.na(m_id)) {
      return(NULL)
    }
    phi_path <- file.path(model_dir, glue("phi_X_curvature_m{m_id}.parquet"))
    if (!file.exists(phi_path)) {
      return(NULL)
    }
    post_df <- open_dataset(post_path) |>
      select(id, name, chain, sample, starts_with("Dim")) |>
      collect()
    phi_df <- read_parquet(phi_path)
    run_modality_latent_analysis(
      glue("curvature_m{m_id}"),
      post_df,
      phi_df,
      plots_dir,
      posterior_plot_names
    )
  })
}

# ── 9. Neighbor curve plots (standard latent space) ─────────────────────────
metrics_for_neighbor_curves <- c("obpm", "ast", "blk", "pct_minutes")

neighbor_curve_means <- posterior_mu_data |>
  filter(metric %in% metrics_for_neighbor_curves) |>
  group_by(metric, player, age) |>
  summarize(mu = mean(value, na.rm = TRUE), .groups = "drop") |>
  mutate(
    observed_mu = transform_metric_mu(metric, mu),
    metric = toupper(metric),
    metric = case_when(metric == "PCT_MINUTES" ~ "MPG", .default = metric)
  )

plot_neighbor_curves <- function(focal_id, focal_name, neighbor_ids) {
  plot_ids <- unique(c(focal_id, neighbor_ids))
  n_neighbors <- length(neighbor_ids)

  neighbor_rank_df <- tibble(
    id = neighbor_ids,
    neighbor_rank = seq_along(neighbor_ids)
  )

  neighbor_names_ordered <- neighbor_rank_df |>
    left_join(latent_space |> select(id, name), by = "id") |>
    arrange(neighbor_rank) |>
    pull(name)

  plot_df <- neighbor_curve_means |>
    filter(player %in% plot_ids) |>
    left_join(latent_space |> select(id, name, position_group), by = c("player" = "id")) |>
    left_join(neighbor_rank_df, by = c("player" = "id")) |>
    mutate(
      role = if_else(player == focal_id, "Focal", "Neighbor"),
      name_label = case_when(
        role == "Focal" ~ paste0(name, " (", position_group, ")"),
        TRUE ~ paste0(neighbor_rank, ". ", name, " (", position_group, ")")
      )
    )

  focal_label     <- unique(plot_df$name_label[plot_df$role == "Focal"])
  neighbor_labels <- plot_df |>
    filter(role == "Neighbor") |>
    distinct(neighbor_rank, name_label) |>
    arrange(neighbor_rank) |>
    pull(name_label)
  labels_ordered  <- c(focal_label, neighbor_labels)
  plot_df <- plot_df |>
    mutate(name_label = factor(name_label, levels = labels_ordered))

  line_colors <- setNames(rep("#1a1a1a", length(labels_ordered)), labels_ordered)

  label_df <- plot_df |>
    filter(role == "Neighbor") |>
    group_by(metric, player, name_label, neighbor_rank) |>
    slice_max(age, n = 1, with_ties = FALSE) |>
    ungroup()

  ggplot(plot_df, aes(x = age, y = observed_mu, group = name_label,
                      color = name_label, linetype = role,
                      alpha = role, linewidth = role)) +
    geom_line() +
    geom_text(
      data = label_df,
      aes(label = neighbor_rank),
      size = 6, hjust = -0.2, show.legend = FALSE
    ) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_color_manual(values = line_colors, name = "Player") +
    scale_linetype_manual(values = c(Neighbor = "dashed", Focal = "solid")) +
    scale_alpha_manual(values = c(Neighbor = 0.45, Focal = 1.0)) +
    scale_linewidth_manual(values = c(Neighbor = 0.6, Focal = 1.2)) +
    guides(
      linetype  = "none",
      alpha     = "none",
      linewidth = "none",
      color = guide_legend(
        title = "Player",
        override.aes = list(
          linetype  = c("solid", rep("blank", n_neighbors)),
          linewidth = c(1.2,     rep(0,       n_neighbors))
        )
      )
    ) +
    theme_bw(base_size = 22) +
    labs(
      title = glue("Latent-Space Neighbor Curves: {focal_name}"),
      x = "Age",
      y = "Metric value"
    )
}

neighbor_curve_dir <- file.path(plots_dir, "latent_space", "map")
dir.create(neighbor_curve_dir, recursive = TRUE, showWarnings = FALSE)

nn_top5 |>
  group_by(focal_id, focal_name) |>
  group_walk(~ {
    neighbor_ids <- .x |> arrange(rank_mean) |> pull(neighbor_id)
    plt <- plot_neighbor_curves(.y$focal_id, .y$focal_name, neighbor_ids)
    fname <- gsub("[^A-Za-z0-9_]", "_", .y$focal_name)
    ggsave(
      file.path(neighbor_curve_dir, glue("latent_neighbors_curves_{fname}.png")),
      plt,
      width = 12,
      height = 8
    )
  })

} # end if (!is.null(latent_space))
