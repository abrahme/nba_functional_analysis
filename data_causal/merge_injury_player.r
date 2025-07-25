library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
injury_data <- read.csv("data/injury_data.csv")
player_data <- read.csv("data/player_data.csv")
validation_data <- read.csv("data/validation_player_data.csv") |> mutate(season = paste(as.character(year - 1), str_sub(year, 3, 4), sep = "-"))

### add klay, paul george, jay williams, etc other injuries which occurred in the off-season

full_data <- player_data |> distinct(name, id) |> inner_join(validation_data, by = "name")  |> select(id, name,age,team,fg3m,fg3a,fg2m,fg2a,ftm,fta,
oreb,dreb,ast,stl,blk,tov,pf,games,minutes,position,year,obpm,dbpm,position_group, season, year) |> bind_rows(player_data |> select(id,name,age,team,fg3m,fg3a,fg2m,fg2a,ftm,fta,oreb,dreb,ast,stl,blk,tov,pf,games,minutes,position,year,obpm,dbpm,position_group, season, year))



full_data <- full_data |> full_join(injury_data |> select(name,year,season,injury_type), by = c("name", "year", "season")) |>
            group_by(name) |> arrange(year) |>
            fill(name, id, position_group) |> ungroup() |>
            group_by(name) |> filter(sum(is.na(id)) == 0) |> arrange(year) |>
            mutate(
                    known_age = if_else(!is.na(age), age, NA_integer_),
                    known_year = if_else(!is.na(age), year, NA_integer_)) |>
            fill(known_age, known_year, .direction = "downup") |>
            mutate(
              age = if_else(is.na(age), known_age + (year - known_year), age)) |>
            ungroup() |> select(-known_age, -known_year) |>
            group_by(name, id) |>
            mutate(
              first_injury_year = min(year[!is.na(injury_type)], na.rm = TRUE),
              first_injury_year = ifelse(is.infinite(first_injury_year), NA, first_injury_year),
              injury_period = case_when(
                is.na(first_injury_year) ~ "pre-injury",
                year < first_injury_year ~ "pre-injury",
                year >= first_injury_year ~ "post-injury"
              )) 
    
major_injury_type <- full_data |> 
    group_by(name,id) |>
    slice_min(order_by = year, n = 1, with_ties = FALSE) |> 
    select(name, id, first_major_injury = injury_type) |> ungroup()

full_data <- full_data |> inner_join(major_injury_type)
write_csv("data/injury_player_cleaned.csv")