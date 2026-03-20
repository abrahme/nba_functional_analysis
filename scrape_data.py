import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
import time
import sys
import re


def _extract_height_from_soup(soup, meta_div=None):
    """
    Extract player height from a Basketball Reference player page.

    Tries the structured `itemprop=height` location first, then falls back
    to parsing `#meta` text patterns used on some player pages.
    """
    height_span = soup.find('span', {'itemprop': 'height'})
    if height_span:
        height_text = height_span.get_text(strip=True)
        if height_text:
            return height_text

    if meta_div is None:
        meta_div = soup.find('div', {'id': 'meta'})

    if meta_div:
        for paragraph in meta_div.find_all('p'):
            text = paragraph.get_text(" ", strip=True)
            match = re.search(r'\b([4-8]-\d{1,2})\b', text)
            if match:
                return match.group(1)

        meta_text = meta_div.get_text(" ", strip=True)
        match = re.search(r'\b([4-8]-\d{1,2})\b', meta_text)
        if match:
            return match.group(1)

    page_text = soup.get_text(" ", strip=True)
    match = re.search(r'\b([4-8]-\d{1,2})\b', page_text)
    if match:
        return match.group(1)

    return "NA"


def fetch_player_draft_and_height(player_id, session=None, headers=None, timeout=20):
    """
    Fetch draft position and height from a Basketball Reference player page.

    Returns a tuple: (draft_position, height)
    Missing values are returned as "NA".
    """
    if pd.isna(player_id):
        return "NA", "NA"

    player_id = str(player_id).strip()
    if not player_id:
        return "NA", "NA"

    if session is None:
        session = requests

    if headers is None:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }

    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"

    try:
        response = session.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return "NA", "NA"
    except Exception:
        return "NA", "NA"

    soup = BeautifulSoup(response.content, 'html.parser')

    draft_position = "NA"
    meta_div = soup.find('div', {'id': 'meta'})
    height = _extract_height_from_soup(soup, meta_div=meta_div)
    if meta_div:
        for paragraph in meta_div.find_all('p'):
            label = paragraph.find('strong')
            if label and 'Draft:' in label.get_text(strip=True):
                draft_text = paragraph.get_text(" ", strip=True)
                match = re.search(r'\((\d+)(?:st|nd|rd|th)\s+pick,\s+(\d+)(?:st|nd|rd|th)\s+overall\)', draft_text)
                if match:
                    draft_position = match.group(2)
                else:
                    undrafted = re.search(r'Undrafted', draft_text, flags=re.IGNORECASE)
                    if undrafted:
                        draft_position = "Undrafted"
                break

    return draft_position, height


def collect_unique_player_ids(input_csv, id_column='id'):
    """
    Read a player CSV and return a list of unique Basketball Reference IDs.

    IDs are normalized by stripping whitespace and dropping empty/NA entries.
    """
    df = pd.read_csv(input_csv)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in input CSV.")

    ids_series = df[id_column].dropna().astype(str).str.strip()
    ids_series = ids_series[ids_series != ""]

    return ids_series.drop_duplicates().tolist()


def get_draft_position_and_height(input_csv, output_csv=None, id_column='id', sleep_seconds=2):
    """
    Given an input CSV of players with Basketball Reference IDs, create a new
    DataFrame with one row per unique ID and columns:
      - id
      - draft_position
      - height

    Missing or unavailable values are filled with "NA".
    """
    if id_column != 'id':
        print(f"Input ID column is '{id_column}'. Output column name will be standardized to 'id'.")

    input_df = pd.read_csv(input_csv)
    if id_column not in input_df.columns:
        raise ValueError(f"Column '{id_column}' not found in input CSV.")

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    unique_player_ids = collect_unique_player_ids(input_csv, id_column=id_column)
    print(f"Collected {len(unique_player_ids)} unique player IDs from {input_csv}")

    rows = []

    with requests.Session() as session:
        total_unique = len(unique_player_ids)
        for idx, player_id in enumerate(unique_player_ids, start=1):
            draft_pos, height = fetch_player_draft_and_height(
                player_id,
                session=session,
                headers=headers
            )
            rows.append({
                'id': player_id,
                'draft_position': draft_pos,
                'height': height
            })

            if idx % 25 == 0 or idx == total_unique:
                print(f"Fetched draft/height for {idx}/{total_unique} unique IDs")

            time.sleep(sleep_seconds)

    df = pd.DataFrame(rows, columns=['id', 'draft_position', 'height'])

    if output_csv is None:
        if input_csv.lower().endswith('.csv'):
            output_csv = input_csv[:-4] + '_id_draft_height.csv'
        else:
            output_csv = input_csv + '_id_draft_height.csv'

    df.to_csv(output_csv, index=False, na_rep='NA')
    print(f"Saved id-level draft/height CSV to {output_csv}")

    return df

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="nba_stats",
        choices=["nba_stats", "player_meta"],
        help="Choose 'nba_stats' to scrape season tables or 'player_meta' to export id/draft_position/height."
    )

    parser.add_argument("--start_year", type=int, default=1997)
    parser.add_argument("--end_year", type=int, default=2026)
    parser.add_argument("--stat_type", type=str, default="totals")

    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--sleep_seconds", type=float, default=2)

    args = parser.parse_args()

    if args.mode == "player_meta":
        if not args.input_csv:
            parser.error("--input_csv is required when --mode player_meta")

        get_draft_position_and_height(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            id_column=args.id_column,
            sleep_seconds=args.sleep_seconds
        )
    else:
        get_nba_stats(args.start_year, args.end_year, stat_type=args.stat_type)