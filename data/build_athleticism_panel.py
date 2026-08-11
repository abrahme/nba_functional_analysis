"""Build the athleticism panel: PBPStats shot-location / contact metrics joined onto the
existing player-season panel, with POSSESSIONS as the exposure denominator.

Motivation
----------
The main model's metrics (OBPM/DBPM/counts) measure *how much* a player produces. A skilled
player can hold those roughly flat after an injury by changing how he plays -- which is why the
injury effects in paper/main.tex come out centred near zero and hard to interpret. The metrics
here measure *how* he produces: whether he still gets to the rim, finishes there, creates his own
shot, draws contact inside the arc, and jumps for his own miss. Those are much harder to fake with
skill, so the latent space built from them encodes athleticism/explosiveness rather than quality.

Metric selection is empirical, not a priori. Each candidate was scored on
  (a) loading on a rim-pressure factor,
  (b) age slope (a real athleticism measure should decline),
  (c) R^2 against quality controls (obpm/dbpm/usage/TsPct/minutes) -- LOW is good, because a
      metric that is mostly "being good at basketball" re-introduces the very confounding the
      panel exists to avoid.
The six lowest-R^2 metrics are the ANCHORS: they identify the athleticism factor separately from
skill. `at_rim_unast` is the best of them (R^2 = 0.06, second-steepest age decline).

Rejected and why: Steals (r = -0.09 with the factor -- lateral quickness is a different
construct); all 3pt unassisted shares (age slope ~ -0.003, i.e. flat -- a pull-up three is not a
burst skill); `Transition Take Fouls Drawn` (2022 rule, 12% coverage); `BlockingFoulsDrawn` (18%).

Exposure: OffPoss / DefPoss rather than minutes. Minutes conflates with pace, which drifts hard
across the 1997-2026 window; possessions is the natural denominator for a per-opportunity rate.
data/data_utils.py:process_data log-transforms these for poisson/negative-binomial families and
keeps integer counts for binomial, so raw counts are what belongs in the CSV.

Output conforms to the panel contract main.py expects (it is the base panel plus new columns), so
it drops straight into `injury_data_csv:` in config/model_config.yaml with no pipeline changes.

Usage:
    python data/build_athleticism_panel.py
    python data/build_athleticism_panel.py --base data/injury_player_cleaned_symmetric_v2.csv \
        --out data/athleticism_panel_symmetric.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

PBP_PATH = "data/pbpstats_1997_2026.parquet"
BASE_PATH = "data/injury_player_cleaned_v2.csv"
OUT_PATH = "data/athleticism_panel.csv"

# PBPStats source columns summed across a player's stints (mid-season trades produce one row per
# player-team-season; everything below is a raw count so summing is exact).
_SUM_COLS = [
    "OffPoss", "DefPoss",
    "AtRimFGA", "AtRimFGM", "AtRimAssists",
    "FG2A", "Fg2aBlocked", "BlockedAtRim",
    "TwoPtShootingFoulsDrawn", "2pt And 1 Free Throw Trips",
    "PtsPutbacks", "SelfOReb", "Blocks",
]

# (output column, source expression) -- integer counts, since every consumer family is a
# count/binomial likelihood and process_data casts binomial exposures with np.int32.
_DERIVED = {
    "off_poss":       lambda d: d["OffPoss"],
    "def_poss":       lambda d: d["DefPoss"],
    "at_rim_fga":     lambda d: d["AtRimFGA"],
    "at_rim_fgm":     lambda d: d["AtRimFGM"],
    # unassisted rim makes = made at rim minus those that were assisted. The ANCHOR metric:
    # creating your own shot at the rim is the least skill-contaminated athleticism signal found.
    "at_rim_unast":   lambda d: d["AtRimFGM"] - d["AtRimAssists"],
    "sfd_2pt":        lambda d: d["TwoPtShootingFoulsDrawn"],
    "and1_2pt":       lambda d: d["2pt And 1 Free Throw Trips"],
    "putback_pts":    lambda d: d["PtsPutbacks"],
    "self_oreb":      lambda d: d["SelfOReb"],
    # Getting stuffed = less lift. NOTE the PBPStats naming trap: `BlockedAtRim` /
    # `Blocked2s` / `BlockedShortMidRange` are DEFENSIVE (blocks the player made, r=0.97 with
    # Blocks); the OFFENSIVE "my attempts that got blocked" field is `Fg2aBlocked` (r=0.88 with
    # FG2A). Verified empirically -- do not swap these.
    "fg2a_pbp":       lambda d: d["FG2A"],
    "fg2a_blocked":   lambda d: d["Fg2aBlocked"],
    # Defensive verticality (rim protection), on the DefPoss denominator.
    "blocks_at_rim":  lambda d: d["BlockedAtRim"],
    "blocks_pbp":     lambda d: d["Blocks"],
}

# Minimum possessions for a season to carry athleticism information at all. Below this the
# per-possession rates are dominated by sampling noise; the cells are blanked (not dropped) so the
# player keeps his row and the model's mask treats them as missing.
MIN_OFF_POSS = 200


def build(base_path: str, pbp_path: str, out_path: str) -> None:
    pbp = pd.read_parquet(pbp_path)
    missing = [c for c in _SUM_COLS if c not in pbp.columns]
    if missing:
        sys.exit(f"PBPStats file is missing required columns: {missing}")
    pbp["id"] = pbp["id"].astype(str)

    n_raw, n_ps = len(pbp), pbp.groupby(["id", "year"]).ngroups
    agg = pbp.groupby(["id", "year"], as_index=False)[_SUM_COLS].sum()
    print(f"pbpstats: {n_raw} rows -> {n_ps} player-seasons "
          f"({n_raw - n_ps} stint rows merged from mid-season trades)", file=sys.stderr)

    out = pd.DataFrame({"id": agg["id"], "year": agg["year"]})
    for name, fn in _DERIVED.items():
        out[name] = fn(agg)

    # Guard the binomial nesting the config relies on: successes must not exceed trials, or the
    # Binomial likelihood is -inf. Clipping is defensive -- these should already nest.
    for succ, trial in (("at_rim_fgm", "at_rim_fga"), ("at_rim_unast", "at_rim_fgm"),
                        ("fg2a_blocked", "fg2a_pbp")):
        bad = int((out[succ] > out[trial]).sum())
        if bad:
            print(f"  WARNING: {bad} rows with {succ} > {trial} -- clipping", file=sys.stderr)
        out[succ] = np.minimum(out[succ], out[trial])
        out[succ] = out[succ].clip(lower=0)

    thin = out["off_poss"] < MIN_OFF_POSS
    metric_cols = [c for c in _DERIVED if c not in ("off_poss", "def_poss")]
    out.loc[thin, metric_cols] = np.nan
    print(f"  blanked {int(thin.sum())} player-seasons under {MIN_OFF_POSS} offensive possessions",
          file=sys.stderr)

    for c in _DERIVED:
        out[c] = out[c].astype("Int64")   # nullable int: keeps NaN while staying integral

    base = pd.read_csv(base_path)
    base["id"] = base["id"].astype(str)
    merged = base.merge(out, on=["id", "year"], how="left")
    if len(merged) != len(base):
        sys.exit(f"join changed row count: {len(base)} -> {len(merged)} (duplicate id/year in pbpstats?)")

    # Play-by-play starts in 1997, so pre-1997 base rows can never match. Report coverage on the
    # window the model will actually use -- the config MUST set start_year: 1997, or a third of
    # the panel enters as all-missing athleticism rows.
    yrs = merged.loc[merged["at_rim_fga"].notna(), "year"]
    m97 = merged[merged["year"] >= 1997]
    print(f"\nwrote {len(merged)} rows x {len(merged.columns)} cols -> {out_path}", file=sys.stderr)
    print(f"  coverage on 1997+ (the usable window): {m97['at_rim_fga'].notna().mean():.1%} "
          f"of {len(m97)} rows; observed years {int(yrs.min())}-{int(yrs.max())}", file=sys.stderr)
    print(f"  coverage over the whole base panel:    {merged['at_rim_fga'].notna().mean():.1%} "
          f"(the rest is pre-1997, before play-by-play)", file=sys.stderr)
    print(f"  players with >=3 covered seasons: "
          f"{(merged[merged.at_rim_fga.notna()].groupby('id').size() >= 3).sum()}", file=sys.stderr)
    merged.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_PATH, help="player-season panel to join onto")
    ap.add_argument("--pbp", default=PBP_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    a = ap.parse_args()
    build(a.base, a.pbp, a.out)


if __name__ == "__main__":
    main()
