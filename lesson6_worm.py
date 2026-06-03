"""
lesson6_worm.py
===============
PHASE 3 / LESSON 6 -- Gradient boosting + the win-probability "worm".

Part 1: Train a HistGradientBoostingClassifier and compare it, fairly, to the
        logistic regression from Lesson 5 (same match-level train/test split).
        Gradient boosting builds many small decision trees in sequence, each
        correcting the last, so it can learn non-linear "it depends" patterns.

Part 2: Draw the WORM -- the chasing team's win probability after every ball of
        a single match. We let the model pick the most DRAMATIC game it saw
        (the one where the eventual winner was once given the smallest chance).

    python lesson6_worm.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from cricket_utils import WINPROB_FEATURES, load_matches

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "model"


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, (proba >= 0.5).astype(int)),
        "log_loss": log_loss(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_DIR / "win_prob_training.csv")
    X, y, groups = data[WINPROB_FEATURES], data["won"], data["match_id"]

    # Exact same split as Lesson 5 (same seed) -> a fair head-to-head.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ---- Part 1: train gradient boosting, compare with logistic regression ----
    # Regularised + early-stopped, so it stops before it memorises the training
    # matches. This is the FAIR way to compare against logistic regression.
    gb = HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=15,        # shallower trees = less overfitting
        min_samples_leaf=200,     # don't split on tiny, noisy groups
        l2_regularization=1.0,
        early_stopping=True,      # stop when a held-out slice stops improving
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    gb.fit(X_train, y_train)
    print(f"(gradient boosting stopped after {gb.n_iter_} trees)\n")

    logreg = joblib.load(MODEL_DIR / "winprob_logreg.joblib")  # from Lesson 5

    results = {
        "Logistic Regression": evaluate(logreg, X_test, y_test),
        "Gradient Boosting": evaluate(gb, X_test, y_test),
    }
    print("MODEL COMPARISON (same unseen test matches)")
    print("-" * 56)
    print(f"{'model':22} {'accuracy':>9} {'log-loss':>9} {'roc-auc':>9}")
    for name, m in results.items():
        print(f"{name:22} {m['accuracy']*100:8.1f}% {m['log_loss']:9.3f} {m['roc_auc']:9.3f}")
    print("-" * 56)

    estimators = {"Logistic Regression": logreg, "Gradient Boosting": gb}
    best_name = min(results, key=lambda k: results[k]["log_loss"])
    best_model = estimators[best_name]
    print(f"Best (lowest log-loss): {best_name} -> shipping this one to the app.\n")

    # Save the BEST model for the Phase 4 app.
    joblib.dump(best_model, MODEL_DIR / "winprob_model.joblib")
    print(f"Best model saved -> {(MODEL_DIR / 'winprob_model.joblib').relative_to(PROJECT_ROOT)}\n")

    # ---- Part 2: find the most dramatic match in the test set ----
    test_data = data.iloc[test_idx].copy()
    test_data["p"] = best_model.predict_proba(test_data[WINPROB_FEATURES])[:, 1]
    # Probability that the EVENTUAL WINNER wins, at each ball.
    test_data["p_winner"] = np.where(test_data["won"] == 1, test_data["p"], 1 - test_data["p"])

    grouped = test_data.groupby("match_id")
    candidates = pd.DataFrame({"low_point": grouped["p_winner"].min(), "n_balls": grouped.size()})
    candidates = candidates[candidates["n_balls"] >= 100]  # only full, real finishes
    # The biggest comeback = the winner was once given the LOWEST chance.
    dramatic_id = candidates["low_point"].idxmin()

    match = test_data[test_data["match_id"] == dramatic_id].sort_values("balls_used")
    overs = match["balls_used"].to_numpy() / 6
    p = match["p"].to_numpy()
    chased_won = int(match["won"].iloc[0])
    chasing_team = match["batting_team"].iloc[0]
    bowling_team = match["bowling_team"].iloc[0]
    target = int(match["target"].iloc[0])
    match_winner = chasing_team if chased_won == 1 else bowling_team

    meta = load_matches().set_index("match_id").loc[dramatic_id]
    when = pd.Timestamp(meta["date"]).strftime("%d %b %Y")

    # The single most dramatic ball (winner's lowest ebb).
    p_winner_path = np.where(chased_won == 1, p, 1 - p)
    low_i = int(p_winner_path.argmin())

    # ---- Draw the worm ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(overs, p * 100, color="#222222", lw=2, zorder=3)
    # Shade green when the chasing team is favourite, red when not.
    ax.fill_between(overs, 50, p * 100, where=(p >= 0.5), interpolate=True, color="#55A868", alpha=0.35)
    ax.fill_between(overs, 50, p * 100, where=(p < 0.5), interpolate=True, color="#C44E52", alpha=0.35)
    ax.axhline(50, ls="--", color="grey", lw=1)

    ax.annotate(
        f"eventual winner given\nonly {p_winner_path[low_i]*100:.0f}% here",
        xy=(overs[low_i], p[low_i] * 100),
        xytext=(overs[low_i] + 1.5, p[low_i] * 100 + (18 if chased_won else -18)),
        arrowprops=dict(arrowstyle="->", color="black"), fontsize=10, ha="left",
    )

    ax.set_xlim(0, 20); ax.set_ylim(0, 100)
    ax.set_xlabel("Overs (2nd innings)")
    ax.set_ylabel(f"Win probability: {chasing_team} (%)")
    ax.set_title(
        f"Win-probability worm — {chasing_team} chasing {target} vs {bowling_team}\n"
        f"{when} · {meta['venue']}  —  Winner: {match_winner}",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "lesson6_worm.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Worm chart saved -> {out.relative_to(PROJECT_ROOT)}")
    print(f"  (drama: {chasing_team} chasing {target} vs {bowling_team}, {when} — won by {match_winner})")


if __name__ == "__main__":
    main()
