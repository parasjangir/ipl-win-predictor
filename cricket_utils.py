"""
cricket_utils.py
================
Shared helpers used by every lesson. Putting common logic here (instead of
re-writing it in each script) is the DRY principle -- "Don't Repeat Yourself".
Later lessons just do:  from cricket_utils import load_matches, load_deliveries

What this module handles:
  1. Standardising franchise names (teams that were renamed over the years).
  2. Loading the clean tables and adding a tidy integer `season_year`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Franchise renames.  IMPORTANT NUANCE: we only merge teams that are genuinely
# the SAME franchise under a new name. We deliberately keep distinct franchises
# separate (e.g. Deccan Chargers != Sunrisers Hyderabad; Gujarat Lions was a
# short-lived 2016-17 team, not today's Gujarat Titans).
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",  # the 's' was dropped in 2017
    "Pune Warriors India": "Pune Warriors",
}

# Dismissal types CREDITED to the bowler. Run-outs (and a few rare types) are
# NOT the bowler's wicket, so we must exclude them when ranking bowlers.
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}

# The features our win-probability model uses, in a fixed order. Defining this
# ONCE here means the trainer (lesson 5) and the app (phase 4) can never disagree.
WINPROB_FEATURES = ["runs_left", "balls_left", "wickets_left", "crr", "rrr", "target"]


def standardize_teams(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace old franchise names with their current canonical name."""
    for col in columns:
        df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def load_matches() -> pd.DataFrame:
    """Load matches.csv, clean team names, and add an integer season_year.

    The raw 'season' label is messy ('2007/08', '2020/21', ...). But every IPL
    season is played within a single calendar year, so the cleanest source of
    truth is simply the YEAR of the match date.
    """
    matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
    matches = standardize_teams(matches, ["team1", "team2", "toss_winner", "winner"])
    matches["season_year"] = matches["date"].dt.year
    return matches


def load_deliveries() -> pd.DataFrame:
    """Load the ball-by-ball table, clean team names, add an integer season_year.

    Prefers the compact Parquet store (≈36x smaller, much faster) and falls back
    to the CSV if Parquet isn't present.
    """
    parquet = DATA_DIR / "deliveries.parquet"
    if parquet.exists():
        deliveries = pd.read_parquet(parquet)
        deliveries["start_date"] = pd.to_datetime(deliveries["start_date"])
    else:
        deliveries = pd.read_csv(
            DATA_DIR / "deliveries.csv",
            parse_dates=["start_date"],
            dtype={"season": "string"},  # season mixes '2017' and '2020/21'
        )
    deliveries = standardize_teams(deliveries, ["batting_team", "bowling_team"])
    deliveries["season_year"] = deliveries["start_date"].dt.year
    return deliveries
