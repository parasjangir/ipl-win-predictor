"""
winprob.py
==========
Serving helpers for the win-probability model. Kept SEPARATE from the Streamlit
UI (app.py) so the prediction logic is easy to test and reuse.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from cricket_utils import WINPROB_FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "model" / "winprob_model.joblib"


def load_model():
    """Load the trained scikit-learn pipeline saved in Lesson 6."""
    return joblib.load(MODEL_PATH)


def match_state(target: int, current_score: int, balls_bowled: int, wickets_lost: int) -> pd.DataFrame:
    """Turn a human-described chase into the single feature row the model expects.

    Returns a 1-row DataFrame with columns in the exact WINPROB_FEATURES order.
    """
    balls_bowled = max(0, min(balls_bowled, 119))  # keep at least 1 ball left
    runs_left = max(target - current_score, 0)
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets_lost
    crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0.0
    rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else 0.0

    row = {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "crr": crr,
        "rrr": rrr,
        "target": target,
    }
    return pd.DataFrame([row])[WINPROB_FEATURES]


def win_probability(model, target: int, current_score: int, balls_bowled: int, wickets_lost: int) -> float:
    """Probability (0-1) that the CHASING team wins, given the situation."""
    state = match_state(target, current_score, balls_bowled, wickets_lost)
    return float(model.predict_proba(state)[0, 1])
