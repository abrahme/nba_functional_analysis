library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
library(HDInterval)
library(glue)

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


latent_mu_space <- read.csv("latent_mu_cohort_2022_2023_2025.csv")


player_info <- latent_space  %>% select(-starts_with("Dim")) %>% group_by(id, val_year) %>% slice_head(n = 1) %>% ungroup() %>% select(-c(chain, sample))

full_latent_mu_space <- latent_mu_space %>% inner_join(player_info, by = c("player" = "id", "val_year" = "val_year"))

latent_mu_fixed <- full_latent_mu_space %>% filter(label == "fixed") %>% group_by(age, player, metric) %>% summarize(value = mean(value, na.rm = TRUE)) %>% ungroup() %>% pivot_wider(
  names_from = c(metric, age),
    values_from = value,
    names_sep = "_"
)  %>% select(-player) %>% data.matrix()

pca_latent_space_mu_fixed <- prcomp(latent_mu_fixed, center = TRUE, scale. = TRUE)

pca_mu_fixed_df <- tibble(PC1 = pca_latent_space_mu_fixed$x[,1], PC2 = pca_latent_space_mu_fixed$x[,2]) |> cbind(full_latent_mu_space %>% filter(label == "fixed") %>% group_by(age, player, metric) %>% summarize(value = mean(value, na.rm = TRUE), minutes = first(minutes), position_group = first(position_group), name = first(name)) %>% ungroup() %>% pivot_wider(
  names_from = c(metric, age),
    values_from = value,
    names_sep = "_"
) %>% select(name, minutes, position_group))


pca_mu_samples_df <- full_latent_mu_space %>% filter(label != "fixed") %>%
  group_by(chain, sample, val_year) %>%
  group_modify(~ {
    df_wide <- .x %>%
      pivot_wider(names_from = c(metric, age), values_from = value)
    X <- as.matrix(df_wide %>% dplyr::select(-c(label, player,name,minutes,position_group)))
    pc_res <- predict(pca_latent_space_mu_fixed, X)
    tibble( id = df_wide$player, PC1 = pc_res[,1], PC2 = pc_res[,2], name = df_wide$name, minutes = df_wide$minutes, position_group = df_wide$position_group)
  }) %>%
  ungroup()


plt_pca <- function(pca_samples_df, pca_fixed_df){
  group_name <- unique(pca_samples_df$name)
  plt <- ggplot() +
  # Fixed points (all players)
  geom_point(data = pca_fixed_df, aes(x = PC1, y = PC2, alpha= minutes)) +

  # Posterior 2D density per player
  geom_polygon(stat = "ellipse", data = pca_samples_df , aes(x = PC1, y = PC2, fill = factor(val_year)), alpha = 0.4,
  level = .95
  ) +
  theme_bw() +
  labs(title = glue("Yearly PCA Projection of {group_name}"), fill = "Year")
  return(plt)
  }

plt_functional_pca <- function(pca_mu_samples_df, pca_mu_fixed_df){
  group_name <- unique(pca_samples_df$name)
  plt <- ggplot() +
  # Fixed points (all players)
  geom_point(data = pca_mu_fixed_df, aes(x = PC1, y = PC2, alpha= minutes)) +

  # Posterior 2D density per player
  geom_polygon(stat = "ellipse", data = pca_mu_samples_df , aes(x = PC1, y = PC2, fill = factor(val_year)), alpha = 0.4,
  level = .95
  ) +
  theme_bw() +
  labs(title = glue("Yearly PCA Functional Projection of {group_name}"), fill = "Year")
  return(plt)
}


plots_list <- pca_mu_samples_df %>% 
  group_by(id) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plt_functional_pca(.x, pca_mu_fixed_df)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/latent_space/evolution_functional_pca_{name}.png"),
      plot = plt,
    )
    })

plots_list <- pca_samples_df %>% 
  group_by(id) %>%
  group_split() %>%               # splits into a list of grouped tibbles
  map(~ {
    plt <- plt_pca(.x, pca_fixed_df)
    name <- unique(.x$name)
    # Save the plot to disk (change path as needed)
    ggsave(
      filename = glue("model_output/model_plots/latent_space/evolution_pca_{name}.png"),
      plot = plt,
    )
    })
