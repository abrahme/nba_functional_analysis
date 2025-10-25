library(tidyverse)
library(readr)
library(stringr)
library(lubridate)
injury_data <- read.csv("data/NBA Player Injury Stats(1951 - 2023).csv")
season_ending <- injury_data |> 
  filter(str_detect(Notes, "out for season")) |>
  mutate(name = coalesce(Relinquished, Acquired), year = year(Date)) |>
  mutate(name = gsub("^\\s*|\\s*\\([^)]*\\)\\s*", "", name)) |>
  mutate(year = if_else(month(Date) >= 10, year(Date) + 1, year(Date))) |>
  mutate(season = paste(as.character(as.integer(year) - 1), str_sub(year, 3, 4), sep = "-")) |>
  filter((season >= 1997))  |>
  mutate(injury_type =
  case_when(str_detect(Notes, regex("rupture|torn", ignore_case = TRUE)) & str_detect(Notes, regex("achilles", ignore_case = TRUE)) ~ "Achilles",
    str_detect(Notes, regex("ACL", ignore_case = TRUE)) ~ "ACL",
    str_detect(Notes, regex("meniscus", ignore_case = TRUE)) ~ "Meniscus",
    str_detect(Notes, regex("labrum|labral|injury", ignore_case = TRUE)) & str_detect(Notes, regex("hip|pelvis", ignore_case = TRUE))  ~ "Hip",
    # str_detect(Notes, regex("labrum|labral", ignore_case = TRUE)) & !str_detect(Notes, regex("hip", ignore_case = TRUE))  ~ "Labrum",
    str_detect(Notes, regex("rupture|torn", ignore_case = TRUE)) & str_detect(Notes, regex("patella", ignore_case = TRUE)) ~ "Patellar Tendon",
    str_detect(Notes, regex("rupture|torn", ignore_case = TRUE)) & str_detect(Notes, regex("quad", ignore_case = TRUE)) ~ "Quad Tendon",
    str_detect(Notes, regex("back|spine|disc|hernia|spinal", ignore_case = TRUE)) ~ "Back/Spine",
    # str_detect(Notes, regex("rotator", ignore_case = TRUE))  ~ "Rotator Cuff",
    str_detect(Notes,regex("fracture|broken", ignore_case =  TRUE)) & str_detect(Notes, regex("tibia|ankle|femur|fibula|foot|leg|toe|navicular", ignore_case = TRUE)) ~ "Lower Body Fracture",
    # str_detect(Notes,regex("fracture|broken", ignore_case =  TRUE)) & str_detect(Notes, regex("arm|thumb|hand|wrist|finger|ulna", ignore_case = TRUE)) ~ "Upper Body Fracture",
    TRUE ~ "Other")) |>
    filter(injury_type != "Other") |> 
  group_by(name) |>
  arrange(year, .by_group = TRUE) |>
  ungroup() |>
  distinct(name, year,injury_type, .keep_all = TRUE) |>
  write_csv("data/injury_data.csv")