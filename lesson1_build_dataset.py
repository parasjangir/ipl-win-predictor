"""
lesson1_build_dataset.py
========================
LESSON 1 -- From raw files to clean, analysis-ready tables (ETL).

WHAT YOU'LL LEARN
-----------------
Real data almost never arrives in the shape you want. Cricsheet gives us
~1,240 matches as TWO files each:

    data/raw/<id>.csv        -> one row per BALL  (the "deliveries" data)
    data/raw/<id>_info.csv   -> key/value metadata for that match

Analysts think in terms of TIDY TABLES: one clearly-defined "thing" per row.
So our job is to produce exactly two clean files:

    data/deliveries.csv  -> one row per ball, ALL matches stacked together
    data/matches.csv     -> one row per match (teams, toss, venue, winner, ...)

This "raw -> clean tables" step is called ETL (Extract, Transform, Load).
It is one of the most employable skills in data science. Master it and you can
work with ANY dataset, not just this one.

    python lesson1_build_dataset.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd  # pandas is THE workhorse library for tabular data in Python

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"


# ===========================================================================
# PART A -- Build deliveries.csv  (the ball-by-ball table)
# ===========================================================================
def build_deliveries() -> pd.DataFrame:
    """Stack every match's ball-by-ball file into one big table.

    Each <id>.csv already shares the SAME columns and header, so this is mostly
    a matter of reading them all and putting them on top of one another.
    The only files to avoid are the *_info.csv metadata files.
    """
    # .glob() lists files matching a pattern. We want <id>.csv but NOT the
    # <id>_info.csv files, so we filter those out.
    delivery_files = sorted(
        f for f in RAW_DIR.glob("*.csv") if not f.name.endswith("_info.csv")
    )
    print(f"[deliveries] reading {len(delivery_files)} match files ...")

    # Read each file into a small DataFrame (a table in memory), collect them
    # in a list, then pd.concat = "stack these tables vertically into one".
    # ignore_index=True renumbers the rows 0,1,2,... in the combined table.
    frames = [pd.read_csv(f) for f in delivery_files]
    deliveries = pd.concat(frames, ignore_index=True)

    # A useful column the raw data doesn't hand us directly: the TOTAL runs
    # conceded on a ball = runs off the bat + extras (wides, no-balls, etc.).
    deliveries["total_runs"] = deliveries["runs_off_bat"] + deliveries["extras"]

    out = DATA_DIR / "deliveries.csv"
    deliveries.to_csv(out, index=False)  # index=False -> don't write row numbers
    print(f"[deliveries] wrote {len(deliveries):,} rows -> {out.name}")
    return deliveries


# ===========================================================================
# PART B -- Build matches.csv  (one row per match)
# ===========================================================================
# These keys appear at most once per match, so we store them as single values.
SINGLE_VALUE_KEYS = {
    "season", "date", "city", "venue",
    "toss_winner", "toss_decision",
    "winner", "winner_runs", "winner_wickets",
    "outcome", "method", "eliminator",
    "player_of_match", "match_number", "event",
}


def parse_info_file(path: Path) -> dict:
    """Turn ONE <id>_info.csv into a single dict (one match summary row).

    These files are NOT a normal table. Every line looks like:
        info,toss_winner,Royal Challengers Bangalore
        info,team,Sunrisers Hyderabad
        info,player,Sunrisers Hyderabad,DA Warner
    i.e.  "info", <key>, <value>[, <extra>...]

    A few keys (like 'team') appear multiple times, so we gather those into a
    list. Everything in SINGLE_VALUE_KEYS we keep as one value.
    """
    # match_id comes from the filename, e.g. "1082591_info.csv" -> 1082591
    record: dict = {"match_id": int(path.stem.replace("_info", ""))}
    teams: list[str] = []

    # Using the csv module (not a plain split on ",") correctly handles values
    # that contain commas inside quotes, e.g. "Rajiv Gandhi Stadium, Uppal".
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            # Skip blank lines and the leading "version,2.1.0" line.
            if not row or row[0] != "info":
                continue
            key = row[1]
            value = row[2] if len(row) > 2 else None

            if key == "team":
                teams.append(value)
            elif key in SINGLE_VALUE_KEYS:
                # setdefault keeps the FIRST value if a key ever repeats.
                record.setdefault(key, value)

    # Record the two teams in the order Cricsheet lists them.
    record["team1"] = teams[0] if len(teams) > 0 else None
    record["team2"] = teams[1] if len(teams) > 1 else None
    return record


def build_matches() -> pd.DataFrame:
    """Parse every *_info.csv into a row, then assemble the matches table."""
    info_files = sorted(RAW_DIR.glob("*_info.csv"))
    print(f"[matches] parsing {len(info_files)} info files ...")

    # A list of dicts -> a DataFrame. pandas lines up the keys into columns
    # and fills any missing key with NaN automatically.
    records = [parse_info_file(f) for f in info_files]
    matches = pd.DataFrame(records)

    # Convert the text date "2017/04/05" into a real datetime we can sort and
    # do maths on. errors="coerce" turns anything unparseable into NaT (missing).
    matches["date"] = pd.to_datetime(matches["date"], format="%Y/%m/%d", errors="coerce")

    # Order columns in a human-friendly way (only keep those that exist).
    preferred = [
        "match_id", "season", "date", "city", "venue",
        "team1", "team2", "toss_winner", "toss_decision",
        "winner", "winner_runs", "winner_wickets",
        "outcome", "method", "eliminator",
        "player_of_match", "match_number", "event",
    ]
    cols = [c for c in preferred if c in matches.columns]
    matches = matches[cols].sort_values("date").reset_index(drop=True)

    out = DATA_DIR / "matches.csv"
    matches.to_csv(out, index=False)
    print(f"[matches] wrote {len(matches):,} rows -> {out.name}")
    return matches


def main() -> None:
    # Guard clause: fail early with a clear message if the data isn't there.
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.csv")):
        raise SystemExit("No raw data found. Run `python download_data.py` first.")

    deliveries = build_deliveries()
    matches = build_matches()

    # ---- A first look at what we built (the payoff!) ----
    print("\n" + "=" * 64)
    print("DONE -- here is a first look at your clean dataset")
    print("=" * 64)

    print(f"\nMATCHES: {matches.shape[0]} rows x {matches.shape[1]} columns")
    print(matches[["date", "team1", "team2", "winner", "venue"]].head(3).to_string(index=False))

    seasons = sorted(matches["season"].dropna().unique())
    print(f"\nSeasons covered ({len(seasons)}): {seasons}")

    print(f"\nDELIVERIES: {deliveries.shape[0]:,} rows x {deliveries.shape[1]} columns")
    print(deliveries[["innings", "ball", "batting_team", "striker", "bowler", "total_runs"]]
          .head(3).to_string(index=False))

    print("\nNext up -> Lesson 2: exploring this data and finding our first insights.")


if __name__ == "__main__":
    main()
