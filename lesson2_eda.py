"""
lesson2_eda.py
==============
LESSON 2 -- Exploratory Data Analysis (EDA).

EDA is the art of *interrogating* a dataset: asking questions and letting the
data answer, usually with summaries and charts. It's how you build intuition
and spot stories worth telling.

In this lesson you'll learn the core pandas "analysis verbs":
    .groupby()      -> split rows into groups and summarise each group
    .value_counts() -> count how often each value appears
    .isin()         -> keep only rows whose value is in a given set
    boolean .mean() -> the % of rows where a condition is True (a neat trick)
    .nlargest()     -> the top-N rows by some value

We answer six questions and draw all six on a single dashboard image.

    python lesson2_eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # "Agg" = render straight to image files (no popup window)
import matplotlib.pyplot as plt
import seaborn as sns

from cricket_utils import BOWLER_WICKETS, load_deliveries, load_matches

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def main() -> None:
    sns.set_theme(style="whitegrid")  # clean, modern chart styling
    FIGURES_DIR.mkdir(exist_ok=True)

    # ---- Load the clean data (cleaning lives in cricket_utils) ----
    matches = load_matches()
    deliveries = load_deliveries()
    print(f"Loaded {len(matches):,} matches and {len(deliveries):,} deliveries.\n")

    # =======================================================================
    # Q1. How many matches were played each season?
    # groupby(season).size() => number of rows (matches) in each season.
    # =======================================================================
    matches_per_season = matches.groupby("season_year").size()

    # =======================================================================
    # Q2. Which franchises have won the most matches?
    # value_counts() counts each winner; it ignores NaN (ties / no-results).
    # =======================================================================
    team_wins = matches["winner"].value_counts()

    # =======================================================================
    # Q3. Does winning the toss help you win the match?
    # Among matches with a result, what % were won by the toss-winner?
    # (toss_winner == winner) is a True/False column; its .mean() is the share
    # of True values -- a very handy idiom.
    # =======================================================================
    decided = matches.dropna(subset=["winner"])
    toss_win_pct = (decided["toss_winner"] == decided["winner"]).mean() * 100

    # =======================================================================
    # Q4. Are teams increasingly choosing to field first (i.e. chase)?
    # Per season, the % of captains who chose "field" at the toss.
    # =======================================================================
    chose_field = matches.assign(is_field=matches["toss_decision"].eq("field"))
    field_pct_by_season = chose_field.groupby("season_year")["is_field"].mean() * 100

    # =======================================================================
    # Q5. Who are the all-time leading run-scorers?
    # Sum runs_off_bat per batter (extras don't count as batting runs).
    # =======================================================================
    top_batters = deliveries.groupby("striker")["runs_off_bat"].sum().nlargest(10)

    # =======================================================================
    # Q6. Who are the all-time leading wicket-takers?
    # Keep only bowler-credited dismissals, then count per bowler.
    # =======================================================================
    bowler_dismissals = deliveries[deliveries["wicket_type"].isin(BOWLER_WICKETS)]
    top_bowlers = bowler_dismissals.groupby("bowler").size().nlargest(10)

    # ----------------------------------------------------------------------
    # Print the headline numbers (great for a README / CV bullet points)
    # ----------------------------------------------------------------------
    print("KEY INSIGHTS")
    print("-" * 60)
    print(f"Most successful team : {team_wins.index[0]} ({team_wins.iloc[0]} wins)")
    print(f"Toss advantage       : toss winners go on to win {toss_win_pct:.1f}% of matches")
    print(f"Leading run-scorer   : {top_batters.index[0]} ({top_batters.iloc[0]:,} runs)")
    print(f"Leading wicket-taker : {top_bowlers.index[0]} ({top_bowlers.iloc[0]} wickets)")
    print(f"Chasing trend        : {field_pct_by_season.iloc[-1]:.0f}% chose to field in the latest season")
    print("-" * 60)

    # ----------------------------------------------------------------------
    # Draw everything onto ONE 2x3 dashboard figure.
    # plt.subplots(2, 3) gives a grid of 6 mini-plots ("axes") to draw on.
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("IPL 2008-2026 — Exploratory Analysis", fontsize=18, fontweight="bold")

    # (0,0) Matches per season
    ax = axes[0, 0]
    ax.bar(matches_per_season.index, matches_per_season.values, color="#4C72B0")
    ax.set_title("Matches per season")
    ax.set_xlabel("Season"); ax.set_ylabel("Matches")
    ax.tick_params(axis="x", rotation=45)

    # (0,1) Most successful teams (top 8 by wins)
    ax = axes[0, 1]
    tw = team_wins.head(8).sort_values()  # sort ascending so the biggest bar is on top
    ax.barh(tw.index, tw.values, color="#55A868")
    ax.set_title("Most successful teams (total wins)")
    ax.set_xlabel("Wins")

    # (0,2) Toss advantage
    ax = axes[0, 2]
    ax.bar(["Toss winner\nWON", "Toss winner\nLOST"],
           [toss_win_pct, 100 - toss_win_pct], color=["#C44E52", "#8172B3"])
    ax.axhline(50, ls="--", color="grey", label="50% (no effect)")
    ax.set_ylim(0, 100); ax.set_ylabel("% of decided matches")
    ax.set_title("Does winning the toss win the match?"); ax.legend()

    # (1,0) Chasing trend
    ax = axes[1, 0]
    ax.plot(field_pct_by_season.index, field_pct_by_season.values, marker="o", color="#4C72B0")
    ax.axhline(50, ls="--", color="grey")
    ax.set_title("Captains choosing to FIELD (chase) %")
    ax.set_xlabel("Season"); ax.set_ylabel("% choosing to field")
    ax.tick_params(axis="x", rotation=45)

    # (1,1) Top run-scorers
    ax = axes[1, 1]
    tb = top_batters.sort_values()
    ax.barh(tb.index, tb.values, color="#DD8452")
    ax.set_title("Top run-scorers (all-time)"); ax.set_xlabel("Runs")

    # (1,2) Top wicket-takers
    ax = axes[1, 2]
    tbo = top_bowlers.sort_values()
    ax.barh(tbo.index, tbo.values, color="#937860")
    ax.set_title("Top wicket-takers (all-time)"); ax.set_xlabel("Wickets")

    fig.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for the suptitle
    out = FIGURES_DIR / "lesson2_eda_overview.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nDashboard saved -> {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
