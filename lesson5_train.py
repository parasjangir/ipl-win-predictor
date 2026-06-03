"""
lesson5_train.py
================
PHASE 3 / LESSON 5 -- Train & evaluate a win-probability model.

We fit a LOGISTIC REGRESSION that, given the live state of a chase, outputs the
probability that the chasing team wins.

Why logistic regression first?
  * It outputs a real probability between 0 and 1 (exactly what we want).
  * Its coefficients are INTERPRETABLE -- we can see which features matter.

Key ideas you'll meet:
  * GROUP SPLIT: we split whole MATCHES into train/test so the model can't peek
    at the same game during training and testing (avoids "data leakage").
  * SCALING: we standardise features so their coefficients are comparable.
  * RIGHT METRICS: accuracy is not enough for probabilities -- we also use
    log-loss (penalises confident-but-wrong) and a CALIBRATION curve.

    python lesson5_train.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cricket_utils import WINPROB_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "model"


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_DIR / "win_prob_training.csv")
    X = data[WINPROB_FEATURES]
    y = data["won"]
    groups = data["match_id"]  # used to keep whole matches together in the split

    # ---- Split by MATCH (not by ball) to prevent leakage ----
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train: {len(X_train):,} states from {groups.iloc[train_idx].nunique()} matches")
    print(f"Test : {len(X_test):,} states from {groups.iloc[test_idx].nunique()} matches\n")

    # ---- The model: scale features, then logistic regression ----
    # A Pipeline chains steps so that scaling is learned on train data only and
    # applied automatically at predict time (no manual bookkeeping, no leakage).
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    # predict_proba gives [P(lose), P(win)]; we keep the win column.
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # ---- Metrics ----
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    # Baselines to prove the model actually adds value.
    base_rate = y_train.mean()
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())          # always guess majority class
    baseline_ll = log_loss(y_test, np.full(len(y_test), base_rate))  # always guess base rate

    print("MODEL PERFORMANCE (on unseen matches)")
    print("-" * 52)
    print(f"  Accuracy : {acc*100:5.1f}%   (baseline {baseline_acc*100:.1f}%)")
    print(f"  Log-loss : {ll:5.3f}   (baseline {baseline_ll:.3f}, lower is better)")
    print(f"  ROC-AUC  : {auc:5.3f}   (0.5 = coin flip, 1.0 = perfect)")
    print("-" * 52)

    # ---- Interpret the coefficients (this is logistic regression's superpower) ----
    # Because features are standardised, a larger |coefficient| = bigger influence.
    # Positive pushes win probability UP, negative pushes it DOWN.
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    print("\nWhat the model learned (standardised coefficients):")
    for i in order:
        direction = "raises" if coefs[i] > 0 else "lowers"
        print(f"  {WINPROB_FEATURES[i]:>13}: {coefs[i]:+.3f}  ({direction} win probability)")

    # ---- Evaluation charts: calibration + ROC ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    axes[0].plot(mean_pred, frac_pos, marker="o", color="#4C72B0", label="our model")
    axes[0].set_xlabel("Predicted win probability")
    axes[0].set_ylabel("Actual win fraction")
    axes[0].set_title("Calibration: does 70% really mean 70%?")
    axes[0].legend()

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, color="#55A868", label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve: how well it separates win vs loss")
    axes[1].legend()

    fig.suptitle("Logistic Regression win-probability model — evaluation", fontweight="bold")
    fig.tight_layout()
    out_fig = FIGURES_DIR / "lesson5_model_eval.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    print(f"\nEvaluation chart saved -> {out_fig.relative_to(PROJECT_ROOT)}")

    # ---- Save the trained pipeline for the Phase 4 app ----
    out_model = MODEL_DIR / "winprob_logreg.joblib"
    joblib.dump(model, out_model)
    print(f"Model saved             -> {out_model.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
