"""
lesson3_stats.py
================
LESSON 3 -- Hypothesis testing: separating real effects from random noise.

THE BIG IDEA
------------
In Lesson 2 we saw toss winners win 51.6% of matches. Is that a real edge, or
just luck wobbling around 50%? Charts can't answer that -- statistics can.

The recipe of a hypothesis test:
  1. NULL HYPOTHESIS (H0): "there is NO effect" (here: win rate = 50%, a coin flip).
  2. Collect data and compute a test statistic.
  3. P-VALUE: the probability of data THIS extreme *if H0 were true*.
  4. DECISION: if p < 0.05 (our "alpha"), the noise explanation is too unlikely,
     so we REJECT H0 and call the effect "statistically significant".
     If p >= 0.05 we "fail to reject" -- we can't rule out chance.

We run three tests:
  A. Is the toss-winner advantage real?         (binomial test vs 50%)
  B. Does the team CHASING win more than half?   (binomial test vs 50%)
  C. Does the bat-or-field CHOICE change who wins? (chi-square test)

    python lesson3_stats.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cricket_utils import load_matches

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
ALPHA = 0.05  # our significance threshold (the conventional 5%)


def verdict(p_value: float) -> str:
    """Translate a p-value into plain English."""
    if p_value < ALPHA:
        return f"p = {p_value:.4f} < {ALPHA}  ->  SIGNIFICANT (reject H0)"
    return f"p = {p_value:.4f} >= {ALPHA}  ->  not significant (fail to reject H0)"


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    matches = load_matches()

    # Only matches with an actual result can be analysed (drop no-results).
    decided = matches.dropna(subset=["winner"]).copy()
    n = len(decided)
    print(f"Analysing {n} decided matches (out of {len(matches)} total).\n")

    # Pull the columns we need as plain numpy arrays for fast comparisons.
    toss_winner = decided["toss_winner"].to_numpy()
    winner = decided["winner"].to_numpy()
    team1 = decided["team1"].to_numpy()
    team2 = decided["team2"].to_numpy()
    decision = decided["toss_decision"].to_numpy()

    # The team that did NOT win the toss.
    other_team = np.where(toss_winner == team1, team2, team1)
    # Who batted second (chased)? If the toss winner chose to field, it's them;
    # otherwise it's the other team.
    chasing_team = np.where(decision == "field", toss_winner, other_team)

    toss_wins = int((toss_winner == winner).sum())
    chasing_wins = int((winner == chasing_team).sum())

    # =======================================================================
    # TEST A -- Toss-winner advantage.  H0: win probability = 0.5
    # binomtest gives an EXACT p-value for "k successes in n coin flips".
    # =======================================================================
    test_a = stats.binomtest(toss_wins, n, 0.5, alternative="two-sided")
    ci_a = test_a.proportion_ci(confidence_level=0.95)
    print("TEST A -- Does winning the TOSS help you win the match?")
    print(f"  Toss winners won {toss_wins}/{n} = {toss_wins / n * 100:.1f}%  (95% CI {ci_a.low*100:.1f}-{ci_a.high*100:.1f}%)")
    print(f"  {verdict(test_a.pvalue)}\n")

    # =======================================================================
    # TEST B -- Chasing advantage.  H0: chasing win probability = 0.5
    # =======================================================================
    test_b = stats.binomtest(chasing_wins, n, 0.5, alternative="two-sided")
    ci_b = test_b.proportion_ci(confidence_level=0.95)
    print("TEST B -- Does the team CHASING (batting second) win more often?")
    print(f"  Chasing teams won {chasing_wins}/{n} = {chasing_wins / n * 100:.1f}%  (95% CI {ci_b.low*100:.1f}-{ci_b.high*100:.1f}%)")
    print(f"  {verdict(test_b.pvalue)}\n")

    # =======================================================================
    # TEST C -- Does the CHOICE (bat vs field) change the toss-winner's odds?
    # Chi-square test of independence on a 2x2 table.
    # H0: the result is independent of the toss decision.
    # =======================================================================
    decided["toss_winner_won"] = toss_winner == winner
    table = pd.crosstab(decided["toss_decision"], decided["toss_winner_won"])
    chi2, p_c, dof, _ = stats.chi2_contingency(table)
    print("TEST C -- Does choosing to BAT vs FIELD change the toss-winner's win rate?")
    print(table.to_string())
    print(f"  chi-square = {chi2:.2f}, dof = {dof}")
    print(f"  {verdict(p_c)}\n")

    # ----------------------------------------------------------------------
    # Visualise tests A & B: the win % with 95% confidence intervals.
    # A confidence interval that CROSSES the 50% line == not significant.
    # ----------------------------------------------------------------------
    labels = ["Toss winner\nwins match", "Chasing team\nwins match"]
    pcts = [toss_wins / n * 100, chasing_wins / n * 100]
    # Error-bar lengths = distance from the bar height to each CI edge.
    yerr = np.array([
        [pcts[0] - ci_a.low * 100, pcts[1] - ci_b.low * 100],   # lower arms
        [ci_a.high * 100 - pcts[0], ci_b.high * 100 - pcts[1]],  # upper arms
    ])
    # Colour green if significant (CI clears 50%), grey if not.
    colors = ["#55A868" if t.pvalue < ALPHA else "#999999" for t in (test_a, test_b)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, pcts, yerr=yerr, capsize=10, color=colors)
    ax.axhline(50, ls="--", color="crimson", label="50% = pure chance")
    ax.set_ylim(40, 60)
    ax.set_ylabel("Win % (with 95% confidence interval)")
    ax.set_title("Is it skill or chance?  (green = statistically significant)")
    for i, (pct, t) in enumerate(zip(pcts, (test_a, test_b))):
        tag = "significant" if t.pvalue < ALPHA else "not significant"
        ax.text(i, 58.5, f"p={t.pvalue:.3f}\n({tag})", ha="center", fontsize=10)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = FIGURES_DIR / "lesson3_stats.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Chart saved -> {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
