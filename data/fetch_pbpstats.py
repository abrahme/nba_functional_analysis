"""
Fetch PBPStats player totals (1996-97 through 2025-26) and attach BBRef player IDs.

Each season is written to data/pbpstats/YYYY.parquet immediately after fetching.
Already-fetched seasons are skipped automatically (safe to resume after interruption).

Final merge: data/pbpstats_1997_2026.parquet
Columns: id (BBRef), year + all ~202 original PBPStats fields (OffPoss, DefPoss, FG2M, ...)

Usage:
    python data/fetch_pbpstats.py                  # fetch all missing seasons
    python data/fetch_pbpstats.py --start-year 2015  # start from a specific year
    python data/fetch_pbpstats.py --merge-only       # skip fetching, just merge existing files
"""

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import requests
import pandas as pd

ALIASES_PATH = "data/pbpstats_name_aliases.csv"
BBREF_PATH = "data/player_advanced_totals_cleaned.csv"
SEASON_DIR = Path("data/pbpstats")
OUTPUT_PATH = "data/pbpstats_1997_2026.parquet"
FIRST_SEASON = 1997
LAST_SEASON = 2026
REQUEST_DELAY = 15.0
RETRY_DELAYS = [15, 30, 60]

# Characters that NFKD cannot decompose to ASCII (e.g. Turkish dotless-i)
_NFKD_GAPS = str.maketrans({"ı": "i", "İ": "i"})


def normalize_name(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", s.translate(_NFKD_GAPS))
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def season_str(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[2:]}"


def season_path(end_year: int) -> Path:
    return SEASON_DIR / f"{end_year}.parquet"


def fetch_season(end_year: int) -> pd.DataFrame:
    url = (
        "https://api.pbpstats.com/get-totals/nba"
        f"?Season={season_str(end_year)}&SeasonType=Regular%20Season&Type=Player"
    )
    for attempt, wait in enumerate(RETRY_DELAYS, start=1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.ok:
                rows = resp.json().get("multi_row_table_data", [])
                return pd.DataFrame(rows) if rows else pd.DataFrame()
            print(
                f"{season_str(end_year)}: HTTP {resp.status_code} (attempt {attempt}) — waiting {wait}s",
                file=sys.stderr,
            )
        except requests.exceptions.RequestException as exc:
            print(
                f"{season_str(end_year)}: connection error (attempt {attempt}) — {exc} — waiting {wait}s",
                file=sys.stderr,
            )
        time.sleep(wait)

    print(f"{season_str(end_year)}: all retries failed — skipping", file=sys.stderr)
    return pd.DataFrame()


def build_crosswalk(bbref: pd.DataFrame) -> dict[str, str]:
    name_to_id: dict[str, str] = {}
    for _, row in bbref[["id", "name"]].drop_duplicates().iterrows():
        key = normalize_name(str(row["name"]))
        name_to_id[key] = str(row["id"])
    return name_to_id


def merge_seasons(alias_map: dict | None = None, crosswalk: dict | None = None) -> None:
    files = sorted(SEASON_DIR.glob("*.parquet"))
    if not files:
        print("No season files found — nothing to merge.", file=sys.stderr)
        sys.exit(1)
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)

    if alias_map is not None and crosswalk is not None:
        def resolve_id(name_str) -> str | None:
            if not name_str:
                return None
            norm = normalize_name(str(name_str))
            norm = alias_map.get(norm, norm)
            return crosswalk.get(norm)

        out["id"] = out["Name"].apply(resolve_id)
        unmatched = out[out["id"].isna()]["Name"].dropna().unique().tolist()
        for name in sorted(unmatched):
            print(f"reapply: unmatched — '{name}'", file=sys.stderr)
        out = out[out["id"].notna()].reset_index(drop=True)
        print(f"Reapplied crosswalk: {len(out)} matched rows", file=sys.stderr)

    cols = ["id", "year"] + [c for c in out.columns if c not in ("id", "year")]
    out = out[cols]
    out.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print(
        f"Merged {len(files)} seasons → {len(out)} rows × {len(out.columns)} cols → {OUTPUT_PATH}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=FIRST_SEASON,
                        help="First season end-year to fetch (e.g. 2015 = 2014-15). "
                             "Seasons with existing parquet files are always skipped.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip fetching; just merge existing per-season files.")
    parser.add_argument("--reapply-crosswalk", action="store_true",
                        help="Re-run alias+crosswalk from Name column on existing season files "
                             "and write fresh merged output. No API calls.")
    args = parser.parse_args()

    SEASON_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_seasons()
        return

    if args.reapply_crosswalk:
        aliases_df = pd.read_csv(ALIASES_PATH)
        alias_map = {
            normalize_name(r["pbpstats_name"]): normalize_name(r["bbref_name"])
            for _, r in aliases_df.iterrows()
        }
        bbref = pd.read_csv(BBREF_PATH, usecols=["id", "name"])
        crosswalk = build_crosswalk(bbref)
        merge_seasons(alias_map=alias_map, crosswalk=crosswalk)
        return

    aliases_df = pd.read_csv(ALIASES_PATH)
    alias_map: dict[str, str] = {
        normalize_name(r["pbpstats_name"]): normalize_name(r["bbref_name"])
        for _, r in aliases_df.iterrows()
    }

    bbref = pd.read_csv(BBREF_PATH, usecols=["id", "name"])
    crosswalk = build_crosswalk(bbref)

    def resolve_id(name_str) -> str | None:
        if not name_str:
            return None
        norm = normalize_name(str(name_str))
        norm = alias_map.get(norm, norm)
        return crosswalk.get(norm)

    for end_year in range(args.start_year, LAST_SEASON + 1):
        path = season_path(end_year)
        if path.exists():
            print(f"{season_str(end_year)}: already fetched — skipping", file=sys.stderr)
            continue

        raw = fetch_season(end_year)
        if raw.empty:
            print(f"{season_str(end_year)}: 0 rows — skipping", file=sys.stderr)
            time.sleep(REQUEST_DELAY)
            continue

        season_df = raw.copy()
        season_df["id"] = season_df["Name"].apply(resolve_id)
        season_df["year"] = end_year

        unmatched = season_df[season_df["id"].isna()]["Name"].tolist()
        for name in unmatched:
            print(f"{season_str(end_year)}: unmatched — '{name}'", file=sys.stderr)

        season_df = season_df[season_df["id"].notna()].reset_index(drop=True)

        total = len(raw)
        matched = len(season_df)
        pct = 100 * matched / total if total else 0
        print(f"{season_str(end_year)}: {matched}/{total} matched ({pct:.1f}%)", file=sys.stderr)

        season_df.to_parquet(path, index=False, engine="pyarrow")

        time.sleep(REQUEST_DELAY)

    merge_seasons()


if __name__ == "__main__":
    main()
