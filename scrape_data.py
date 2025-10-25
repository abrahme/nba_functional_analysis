import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


years = range(1997, 2026)
advanced_dfs = []
advanced_keep_cols = ["Player", "Age", "Team", "OBPM", "G", "DBPM", "Pos", "MP"]
advanced_keep_cols_mapper = {"Player": "name", "Team": "team", "Age": "age", "OBPM": "obpm", "DBPM":"dbpm", "G":"games", "Pos":"position", "MP": "minutes"}

for year in years:
# URL of the page
    url_advanced = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"

    # Get page content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://google.com",
        # Add other headers if needed
    }

    response_advanced = requests.get(url_advanced, headers=headers)
    print(response_advanced)
    soup_advanced = BeautifulSoup(response_advanced.content, 'html.parser')
    # Directly find the table by ID
    advanced_table = soup_advanced.find("table", {"id": "advanced"})
    if advanced_table is None:
        raise ValueError("Table with id='advanced_post' not found.")

    # Parse the table into a DataFrame
    df_advanced = pd.read_html(str(advanced_table))[0]

    # Drop repeated header rows (e.g., rows where 'Player' == 'Player')
    df_advanced = df_advanced[df_advanced['Player'] != 'Player']

    is_multiteam_summary = df_advanced["Team"].str.match(r"\dTM")
    player_counts = df_advanced["Player"].value_counts()
    is_single_entry = df_advanced["Player"].map(player_counts) == 1
    final_df_advanced = df_advanced[is_multiteam_summary | is_single_entry]

    # Reset index and show
    final_df_advanced.reset_index(drop=True, inplace=True)
    final_df_advanced = final_df_advanced[advanced_keep_cols]
    final_df_advanced.rename(mapper=advanced_keep_cols_mapper, axis=1, inplace=True)
    final_df_advanced["year"] = year
    advanced_dfs.append(final_df_advanced)
    time.sleep(5)

advanced_stats = pd.concat(advanced_dfs)
    
dfs = []
keep_cols = ["Player", "Age", "Team", "3P", "3PA","2P","2PA","FT","FTA","ORB","DRB", "AST", "STL", "BLK","TOV", "PF", "G", "MP", "Pos"]
keep_cols_mapper = {"Player": "name", "3P": "fg3m", "3PA": "fg3a", "2P": "fg2m", "2PA": "fg2a", "FT": "ftm", "ORB": "oreb", "DRB": "dreb", "G": "games", "MP": "minutes", "Pos": "position" }

for year in years:
    print(year)
# URL of the page
    url_totals = f"https://www.basketball-reference.com/leagues/NBA_{year}_totals.html"

    # Get page content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://google.com",
        # Add other headers if needed
    }

    response_totals = requests.get(url_totals, headers=headers)



    soup_totals = BeautifulSoup(response_totals.content, 'html.parser')
    # Directly find the table by ID
    totals_table = soup_totals.find("table", {"id": "totals_stats"})
    if advanced_table is None:
        raise ValueError("Table with id='totals_stats' not found.")

    # Parse the table into a DataFrame
    df_totals = pd.read_html(str(totals_table))[0]

    # Drop repeated header rows (e.g., rows where 'Player' == 'Player')
    df_totals = df_totals[df_totals['Player'] != 'Player']

    is_multiteam_summary = df_totals["Team"].str.match(r"\dTM")
    player_counts = df_totals["Player"].value_counts()
    is_single_entry = df_totals["Player"].map(player_counts) == 1
    final_df_totals = df_totals[is_multiteam_summary | is_single_entry]

    # Reset index and show
    final_df_totals.reset_index(drop=True, inplace=True)
    final_df_totals = final_df_totals[keep_cols]
    final_df_totals.rename(mapper=keep_cols_mapper, axis=1, inplace=True)
    final_df_totals["year"] = year
    final_df_totals.columns = final_df_totals.columns.str.lower()
    dfs.append(final_df_totals)
    time.sleep(5)

total_stats = pd.concat(dfs)

aggregate_data = pd.merge(total_stats, advanced_stats, on = ["name", "team", "age", "year", "games", "minutes", "position"])
position_map = {"SG": "G", "PG": "G", "SF": "F", "PF": "F", "C": "C", "G":"G", "F":"F"}
aggregate_data["position_group"] = aggregate_data["position"].apply(lambda x: position_map[x])
aggregate_data["season"] = aggregate_data["year"].apply(lambda x: str(x - 1) + "-" + str(x)[-2:])
aggregate_data.to_csv("data/validation_player_data.csv", index = False)