import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
import time
import sys

def get_nba_stats(start_year=2000, end_year=2026, stat_type='totals'):
    """
    Scrapes NBA stats from Basketball Reference for a range of years.
    
    stat_type options: 
    - 'totals' (Total stats)
    - 'advanced' (Advanced stats)
    - 'per_game' (Per Game stats)
    """
    
    # Map the URL identifier to the actual HTML table ID
    # Advanced uses id="advanced", Totals uses id="totals_stats"
    table_ids = {
        'totals': 'totals_stats',
        'advanced': 'advanced',
        'per_game': 'per_game_stats'
    }
    
    target_table_id = table_ids.get(stat_type)
    if not target_table_id:
        print(f"Error: Unknown stat_type '{stat_type}'. Supported: {list(table_ids.keys())}")
        return

    all_data = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Starting scrape for '{stat_type}' from {start_year} to {end_year}...")
    print("NOTE: A 4-second delay is enforced between requests to avoid IP bans.")

    for year in range(start_year, end_year + 1):
        # URL structure: https://www.basketball-reference.com/leagues/NBA_2024_totals.html
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_{stat_type}.html"
        
        try:
            print(f"Fetching {year}...", end=" ")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the specific table
                table = soup.find('table', {'id': target_table_id})
                
                if table:
                    df = pd.read_html(str(table))[0]
                    
                    # CLEANUP:
                    
                    
                    # 2. Add Season column
                    df.insert(0, 'Season', year)
                    
                    # 3. Remove Hall of Fame asterisks
                    if 'Player' in df.columns:
                        df['Player'] = df['Player'].str.replace('*', '', regex=False)
                    player_ids = []
                    for row in table.tbody.find_all("tr"):
                        td = row.find("td", {"data-stat": "name_display"})
                        if td and td.has_attr("data-append-csv"):
                            player_ids.append(td["data-append-csv"])
                        else:
                            player_ids.append(None)

                    df["id"] = player_ids
                    # 1. Remove repeating headers
                    df = df[df['Rk'] != 'Rk']
                    df = df[df["Player"] != "League Average"] ## remove league average column
                    
                    all_data.append(df)
                    print(f"Success ({len(df)} rows).")
                else:
                    print(f"Table not found. (Season might not exist yet)")
            
            elif response.status_code == 429:
                print("\nERROR: Rate limit exceeded (429). You are sending requests too fast.")
                sys.exit(1)
            else:
                print(f"Failed. Status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error occurred: {e}")

        # CRITICAL: Sleep to respect rate limits
        time.sleep(10)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        output_filename = f"data/nba_{stat_type}_stats_{start_year}-{end_year}.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"\nDone! Saved {len(final_df)} rows to {output_filename}")
    else:
        print("\nNo data collected.")

if __name__ == "__main__":
    # Change stat_type to 'advanced' or 'totals' as needed
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=1997)
    parser.add_argument("--end_year", type=int, default=2026)
    parser.add_argument("--stat_type", type=str, default="totals")
    args = parser.parse_args()
    start_year = args.start_year
    end_year  = args.end_year
    stat_type = args.stat_type
    get_nba_stats(start_year, end_year, stat_type=stat_type)