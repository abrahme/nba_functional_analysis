library(tidyverse)
library(readr)
library(stringr)
library(lubridate)

advanced_data <- read.csv("data/nba_advanced_stats_1997-2026.csv")
totals_data <- read.csv("data/nba_totals_stats_1997-2026.csv")

advanced_data_filtered <- advanced_data %>% group_by(id, Age, Season) %>% slice_max(order_by = MP, n = 1, with_ties = FALSE) %>%
  ungroup() %>% select(-c(Rk,Awards)) 

totals_data_filtered <- totals_data %>% group_by(id, Age, Season) %>% slice_max(order_by = MP, n = 1, with_ties = FALSE) %>%
  ungroup() %>% select(-c(Rk,Awards)) 

data <- inner_join(advanced_data_filtered %>% select(id, Age, Season, `USG.`, OBPM, DBPM), totals_data_filtered, by = c("Age", "Season", "id")) %>% select(id, Age, Season, `USG.`, Player, Team, Pos, G, OBPM, DBPM, MP, `X3P`, `X2P`, FT, FTA, `X3PA`, `X2PA`, BLK, TOV, STL, AST, ORB, DRB) %>%
        rename_with(tolower) %>% rename(fg3a = x3pa, name = player, year = season, fg2a = x2pa, games = g, minutes = mp, usg = `usg.`, fg3m = x3p, fg2m = x2p, dreb = drb, oreb = orb, position = pos, ftm = ft) %>%
        mutate(position_group = case_when( str_detect(position, "G") ~ "G",
                                            str_detect(position, "F") ~ "F",
                                            str_detect(position, "C") ~ "C"),
                                            
              season = paste(as.character(year - 1), str_sub(year, 3, 4), sep = "-"))
        
write.csv(data, "data/player_advanced_totals_cleaned.csv", row.names = FALSE)
