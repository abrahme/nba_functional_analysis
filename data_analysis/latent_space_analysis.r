library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)

data <- read.csv("data/injury_player_cleaned.csv")


latent_space <- read.csv("latent_X_cohort_2022_2023_2025.csv")
X_fixed <- latent_space %>% filter(label == "fixed") %>% group_by(id) %>%  summarise(across(starts_with("Dim"), mean, na.rm = TRUE)) %>% ungroup() %>%  select(starts_with("Dim")) %>% data.matrix()

pca_latent_space_fixed <- prcomp(X_fixed, center = TRUE, scale. = TRUE)



pca_fixed_df <- tibble(PC1 = pca_latent_space_fixed$x[,1], PC2 = pca_latent_space_fixed$x[,2]) |> cbind(latent_space %>% filter(label == "fixed") %>% group_by(id) %>% summarise(across(!starts_with("Dim"), first)))

pca_samples_df <- latent_space %>% filter(label != "fixed") %>%
  group_by(chain, sample, val_year) %>%
  group_modify(~ {
    X <- as.matrix(.x %>% select(starts_with("Dim")))
    pc_res <- predict(pca_latent_space_fixed, X)
    tibble(row = .x$row, id = .x$id, PC1 = pc_res[,1], PC2 = pc_res[,2], name = .x$name, minutes = .x$minutes, position_group = .x$position_group)
  }) %>%
  ungroup()


plt <- ggplot() +
  # Fixed points (all players)
  geom_point(data = pca_fixed_df, aes(x = PC1, y = PC2, alpha= minutes)) +

  # Posterior 2D density per player
  geom_polygon(stat = "ellipse", data = pca_samples_df |> filter(name == "LaMelo Ball"), aes(x = PC1, y = PC2, fill = factor(val_year)), alpha = 0.4,
  level = .95
  ) +
  theme_bw() +
  labs(title = "PCA Projection of LaMelo Ball", fill = "Year")

ggsave(filename = "test_plot.png", plot = plt)


# file_paths <- c("posterior_mu_ar_2022.csv", "posterior_mu_ar_2023.csv", "posterior_mu_ar_2025.csv") 


# combined_df <- file_paths %>%
#   lapply(function(path) {
#     df <- read.csv(path) |> inner_join(data |> group_by(id) |> summarize(name = first(name)) |> ungroup() |> filter(name == "Tyrese Haliburton"),
#                                         by = c("player" = "id")) |> 
                                        
#               group_by(metric, player, age) |> summarize(mu = mean(value)) |> ungroup() |> 
#               mutate(mu = case_when(metric %in% c("fg2m", "ftm", "games", "fg3m") ~ plogis(mu),
#                                                                                  metric %in% c("obpm", "dbpm") ~ mu, 
#                                                                                  metric %in% c("pct_minutes") ~ plogis(mu) * 48,
#                                                                                  .default = exp(mu) * 36))   |>
#           mutate(metric = toupper(metric),
#                     metric = case_when(metric == "GAMES" ~ "GP%",
#                     metric == "FG2M" ~ "FG2%",
#                     metric == "FG3M" ~ "FG3%",
#                     metric == "FTM" ~ "FT%",
#                     metric == "PCT_MINUTES" ~ "MPG",
#                     .default = metric))
#     df$val_year <- str_extract(basename(path) , "\\d{4}") # add filename column
#     df
#   }) %>%
#   bind_rows()

# latent_diff_plot <- combined_df |> ggplot(aes(x = age, y = mu, color = val_year)) + geom_line() + 
#       facet_wrap( ~ metric, scales = "free_y") + theme_bw() + 
#     labs(x = "Age", y = "Metric Value") + ggtitle(paste("Posterior Mean Career Trajectory over Different Training Sets: Tyrese Haliburton"))

# ggsave(filename = "test_plot_2.png", plot = latent_diff_plot)