library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
library(ggplot2)
injury_data <- read.csv("data/injury_data.csv")
full_data <- read.csv("data/player_advanced_totals_cleaned.csv")
# player_data <- read.csv("data/player_data.csv")
# validation_data <- read.csv("data/validation_player_data.csv") |> mutate(season = paste(as.character(year - 1), str_sub(year, 3, 4), sep = "-"))

### add klay, paul george, jay williams, etc other injuries which occurred in the off-season

# full_data <- validation_data  |> select(name,age,team,fg3m,fg3a,fg2m,fg2a,ftm,fta,
#             oreb,dreb,ast,stl,blk,tov,pf,games,minutes,position,year,obpm,dbpm,position_group, season, year) |>
#             bind_rows(player_data |> select(id,name,age,team,fg3m,fg3a,fg2m,fg2a,ftm,fta,oreb,dreb,ast,stl,blk,tov,pf,games,minutes,position,year,obpm,dbpm,position_group, season, year)) |> 
#             group_by(name) |> arrange(year) |> fill(id, .direction = "downup") |> ungroup() |> 
#             group_by(name, id) |> 
#             mutate(id = if_else(any(is.na(id)), max(player_data$id, na.rm =TRUE) + cur_group_id(), max(id))) |> 
#             ungroup() 




full_data <- full_data |> filter(minutes > 0) |> full_join(injury_data |> select(name,year,season,injury_type), by = c("name", "year", "season")) |>
            group_by(name) |> arrange(year) |>
            fill(name, id, position_group, .direction = "downup") |> ungroup() |>
            group_by(name, id) |> filter(!any(is.na(id))) |> arrange(year) |>
            mutate(
                    known_age = if_else(!is.na(age), age, NA_integer_),
                    known_year = if_else(!is.na(age), year, NA_integer_)) |>
            fill(known_age, known_year, .direction = "downup") |>
            mutate(
              age = if_else(is.na(age), known_age + (year - known_year), age)) |>
            ungroup() |> select(-known_age, -known_year) |> 
            mutate(total_games = case_when(year == 2012 ~ 66, 
                                            year == 1999 ~ 50, 
                                                                                           year == 2021 ~ 72,
                                                                                           year == 2020 ~ if_else(is.infinite(max(games, na.rm = TRUE)),NA_integer_,max(games, na.rm = TRUE)),  
                                                                                           .default = 82,
                                                                                           ),.by = c(year)) |>
            group_by(name, id) |>
            mutate(
              first_injury_year = min(year[!is.na(injury_type)], na.rm = TRUE),
              first_injury_year = if_else(is.infinite(first_injury_year), NA, first_injury_year) ,
              gp_injury_year = coalesce(games[year == first_injury_year],0),
              first_injury_year = if_else((gp_injury_year / max(total_games[year == first_injury_year], gp_injury_year)) > .5, first_injury_year + 1, first_injury_year),
              injury_period = case_when(
                is.na(first_injury_year) ~ "pre-injury",
                is.na(minutes) & !is.na(first_injury_year) ~ "post-injury",
                year < first_injury_year ~ "pre-injury",
                year >= first_injury_year ~ "post-injury"
              )) |> select(-c(gp_injury_year))
    
major_injury_type <- full_data |> 
    group_by(name,id) |> arrange(year) |> fill(injury_type, .direction = "up") |>
    slice_min(order_by = year, n = 1, with_ties = FALSE) |> 
    select(name, id, first_major_injury = injury_type) |> ungroup()

full_data <- full_data |> inner_join(major_injury_type) |> group_by(name, id) |> arrange(year) |> ungroup() |> 
mutate(first_major_injury = case_when(name == "Paul George" ~"Lower Body Fracture", 
                                      name == "Kawhi Leonard" ~ "ACL",
                                      .default = first_major_injury),
       injury_period = case_when(name == "Paul George" & year <= 2014 ~ "pre-injury",
                                 name == "Paul George" & year > 2014 ~ "post-injury",
                                 name == "Kawhi Leonard" & year >= 2022 ~ "post-injury",
                                 name == "Kawhi Leonard" & year < 2022 ~ "pre-injury",
                                 .default = injury_period)                                                                                               )
write.csv(full_data, "data/injury_player_cleaned.csv", row.names = FALSE )