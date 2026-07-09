# Shared utilities sourced by model_diagnostics.r, latent_space.r and coverage.r.
# The pure helpers below define no global state; the data builders at the bottom
# (make_player_data / build_injury_data / build_joined_data) perform file I/O when
# CALLED, so the sourcing script controls when that happens.

plot_name <- function(x) if_else(x == "No Name", "Average", x)

posterior_plot_names <- c(
  "Stephen Curry", "Kevin Durant", "LeBron James", "Kobe Bryant", "Dwight Howard",
  "Nikola Jokic", "Kevin Garnett", "Steve Nash",
  "Chris Paul", "Shaquille O'Neal", "Anthony Edwards", "Jamal Murray",
  "Donovan Mitchell", "Ray Allen", "Klay Thompson",
  "Scottie Pippen", "Amar'e Stoudemire", "Shawn Marion", "Dirk Nowitzki",
  "Jason Kidd", "Marcus Camby", "Rudy Gobert", "Tim Duncan",
  "Manu Ginobili", "James Harden", "Russell Westbrook", "Luka Doncic",
  "Devin Booker", "Paul Pierce", "Allen Iverson", "Tyrese Haliburton",
  "LaMelo Ball", "Carmelo Anthony", "Dwyane Wade", "Derrick Rose",
  "Chris Bosh", "Karl-Anthony Towns", "Kristaps Porzingis",
  "Giannis Antetokounmpo", "Jrue Holiday", "No Name"
)

read_parquet_if_exists <- function(path) if (file.exists(path)) read_parquet(path) else NULL

# ── Shared data builders ──────────────────────────────────────────────────────
# These reproduce the posterior + injury data prep used by both model_diagnostics.r
# (player/paper plots) and coverage.r. Keep their behaviour identical to the inline
# versions they replaced so downstream output is unchanged.

# Cleaned player CSV + a "No Name" average-player row spanning the posterior age range.
make_player_data <- function(posterior_data) {
  age_min   <- min(posterior_data$age)
  age_max   <- max(posterior_data$age)
  fake_data <- data.frame(age = age_min:age_max, name = "No Name", id = "99999999",
                          year = seq(2000, 2000 + age_max - age_min))
  read.csv("data/injury_player_cleaned.csv") %>% mutate(retirement = 1) %>% bind_rows(fake_data)
}

# Long-format per-(player, metric, age) observed values from the cleaned player CSV.
build_injury_data <- function(data) {
  data |>
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
}

# Merge posterior with observed values, fill metadata, label train/holdout split.
# `posterior_peaks` may be NULL (peak_age join is skipped). Returns a named list:
#   joined_data, train_years_df, holdout_players (or NULL), holdout_idx (or NULL).
build_joined_data <- function(posterior_data, injury_data, data, posterior_peaks,
                              model_dir, validation_year) {
  player_year_bounds <- data %>%
    group_by(id) %>% arrange(age) %>%
    summarise(
      first_obs = min(year[!is.na(age)], na.rm = TRUE),
      last_obs  = max(year[!is.na(age)], na.rm = TRUE)
    )

  if (!is.null(posterior_peaks)) {
    posterior_data <- posterior_data |>
      inner_join(posterior_peaks |>
                 rename(peak_age = value), by = c("player", "chain", "sample", "metric"))
  }

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
    holdout_idx <- read.csv(holdout_csv)
    has_score_window <- "score_window" %in% names(holdout_idx)
    # 3-state split. With a score_window flag (stratified_next_k: the full career tail is
    # held out of training but only the next-k window is scored):
    #   score_window == 1     -> "holdout"  (scored)
    #   in CSV, score_window 0 -> "masked"  (held out of training, NOT scored, NOT in-sample)
    #   absent from CSV        -> "train"
    # Without the flag (other schemes) every held-out cell is "holdout" (unchanged behaviour).
    join_cols <- if (has_score_window) c("player", "age", "score_window") else c("player", "age")
    joined_data <- joined_data |>
      left_join(holdout_idx |> select(all_of(join_cols)) |> mutate(in_hold = TRUE),
                by = c("player", "age")) |>
      mutate(split = if (has_score_window)
                       case_when(is.na(in_hold)   ~ "train",
                                 score_window == 1 ~ "holdout",
                                 TRUE              ~ "masked")
                     else
                       if_else(is.na(in_hold), "train", "holdout")) |>
      select(-in_hold, -any_of("score_window"))
    # years of training data per player = seasons never held out of training (full tail excluded)
    train_years_df <- data |>
      anti_join(holdout_idx, by = c("id" = "player", "age" = "age")) |>
      group_by(id) |>
      summarize(years_played = n(), .groups = "drop")
    # holdout flag for exit-age coverage: player appears in holdout mask
    holdout_players <- holdout_idx |> distinct(player) |> mutate(exit_split = "holdout")
  } else {
    holdout_idx <- NULL
    joined_data <- joined_data |>
      mutate(split = if_else(year > validation_year, "holdout", "train"))
    train_years_df <- data |>
      filter(year <= validation_year) |>
      group_by(id) |>
      summarize(years_played = n(), .groups = "drop")
    holdout_players <- NULL
  }

  list(joined_data    = joined_data,
       train_years_df = train_years_df,
       holdout_players = holdout_players,
       holdout_idx    = holdout_idx)
}
