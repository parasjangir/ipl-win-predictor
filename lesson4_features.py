"""
lesson4_features.py
===================
PHASE 3 / LESSON 4 -- Feature engineering for a win-probability model.

GOAL
----
Convert raw ball-by-ball data into a TRAINING TABLE where each row is a single
"match state" during a run chase, plus the answer we want to predict:
did the chasing team go on to win?

We only use the 2nd innings (the chase), because only a chase has a well-defined
situation at every ball: a target, runs needed, balls left, wickets in hand.

The features we build for every ball:
    runs_left      -- runs still needed to win
    balls_left     -- legal balls remaining (out of 120)
    wickets_left   -- wickets in hand (10 - wickets lost)
    crr            -- current run rate  (runs so far per over)
    rrr            -- required run rate (runs needed per remaining over)
    target         -- the score to chase
Label:
    won            -- 1 if the chasing team won the match, else 0

TWO TRAPS we handle carefully:
  * "Legal balls": wides and no-balls do NOT use up one of the 120 balls.
  * DLS matches: rain revises the target, which would corrupt runs_left, so we
    drop those matches.

    python lesson4_features.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cricket_utils import load_deliveries, load_matches

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
BALLS_PER_INNINGS = 120  # 20 overs x 6 balls


def build_features() -> pd.DataFrame:
    deliveries = load_deliveries()
    matches = load_matches()

    # ---- 1. The target = first-innings total + 1 ----
    # groupby(match).sum() gives each first innings' total runs.
    innings1_total = (
        deliveries[deliveries["innings"] == 1]
        .groupby("match_id")["total_runs"].sum()
        .rename("innings1_total")
    )

    # ---- 2. Keep only the chase (2nd innings) and attach the target ----
    chase = deliveries[deliveries["innings"] == 2].copy()
    chase = chase.merge(innings1_total, on="match_id", how="left")
    chase["target"] = chase["innings1_total"] + 1

    # ---- 3. Running totals WITHIN each match (rows are already in ball order) ----
    # A ball is "legal" if it is not a wide and not a no-ball.
    chase["legal"] = chase["wides"].isna() & chase["noballs"].isna()
    # A wicket fell on this ball if someone was dismissed.
    chase["wicket"] = chase["player_dismissed"].notna().astype(int)

    grp = chase.groupby("match_id", sort=False)
    chase["cum_runs"] = grp["total_runs"].cumsum()      # score so far
    chase["balls_used"] = grp["legal"].cumsum()         # legal balls bowled so far
    chase["cum_wickets"] = grp["wicket"].cumsum()       # wickets lost so far

    # ---- 4. The features a fan reads off the scoreboard ----
    chase["runs_left"] = chase["target"] - chase["cum_runs"]
    chase["balls_left"] = BALLS_PER_INNINGS - chase["balls_used"]
    chase["wickets_left"] = 10 - chase["cum_wickets"]

    # ---- 5. Keep only genuine "still chasing" states (avoids divide-by-zero
    #         and the trivial already-won/over states) ----
    chase = chase[
        (chase["balls_left"] >= 1)
        & (chase["balls_used"] >= 1)
        & (chase["runs_left"] >= 1)
    ].copy()

    # Run rates (safe now: balls_used >= 1 and balls_left >= 1).
    chase["crr"] = chase["cum_runs"] / (chase["balls_used"] / 6)
    chase["rrr"] = chase["runs_left"] / (chase["balls_left"] / 6)

    # ---- 6. Attach the outcome label, dropping DLS + no-result matches ----
    chase = chase.merge(matches[["match_id", "winner", "method"]], on="match_id", how="left")
    chase = chase[chase["method"].isna()]            # drop rain/DLS-revised matches
    chase = chase.dropna(subset=["winner"])          # drop abandoned/no-result
    chase["won"] = (chase["batting_team"] == chase["winner"]).astype(int)

    # ---- 7. The final tidy training table ----
    feature_cols = [
        "match_id", "batting_team", "bowling_team",
        "balls_used", "runs_left", "balls_left", "wickets_left",
        "crr", "rrr", "target", "won",
    ]
    return chase[feature_cols].reset_index(drop=True)


def sanity_check_plot(data: pd.DataFrame) -> None:
    """Before modelling, confirm the features actually carry signal:
    win rate should fall as more runs are needed, and rise with wickets in hand.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Win rate vs runs still required (bucketed).
    bins = [0, 10, 20, 30, 50, 75, 300]
    labels = ["1-10", "11-20", "21-30", "31-50", "51-75", "76+"]
    data = data.assign(runs_bucket=pd.cut(data["runs_left"], bins=bins, labels=labels))
    win_by_runs = data.groupby("runs_bucket", observed=True)["won"].mean() * 100
    axes[0].bar(win_by_runs.index.astype(str), win_by_runs.values, color="#DD8452")
    axes[0].set_title("Win % vs runs still required")
    axes[0].set_xlabel("runs_left"); axes[0].set_ylabel("win %")
    axes[0].axhline(50, ls="--", color="grey")

    # Win rate vs wickets in hand.
    win_by_wkts = data.groupby("wickets_left", observed=True)["won"].mean() * 100
    axes[1].bar(win_by_wkts.index.astype(int), win_by_wkts.values, color="#55A868")
    axes[1].set_title("Win % vs wickets in hand")
    axes[1].set_xlabel("wickets_left"); axes[1].set_ylabel("win %")
    axes[1].axhline(50, ls="--", color="grey")

    fig.suptitle("Sanity check: do the features carry signal? (yes!)", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "lesson4_feature_check.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Sanity-check chart saved -> {out.relative_to(out.parents[1])}")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    data = build_features()
    out_csv = DATA_DIR / "win_prob_training.csv"
    data.to_csv(out_csv, index=False)

    print(f"Built training table: {len(data):,} rows x {data.shape[1]} columns")
    print(f"  from {data['match_id'].nunique()} matches")
    print(f"  saved -> {out_csv.name}\n")

    # Class balance: what fraction of all in-play states ended in a win?
    print(f"Label balance: {data['won'].mean()*100:.1f}% of states ended in a chasing-team win\n")

    # Eyeball one chase to see the state evolve ball by ball.
    sample_id = data["match_id"].iloc[0]
    sample = data[data["match_id"] == sample_id]
    print(f"Sample chase (match {sample_id}) every ~20 balls:")
    print(sample[["balls_used", "runs_left", "balls_left", "wickets_left", "crr", "rrr", "won"]]
          .iloc[::20].to_string(index=False))

    sanity_check_plot(data)


if __name__ == "__main__":
    main()
