"""
app.py -- IPL Win Probability Predictor (Phase 4, Streamlit).

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cricket_utils import WINPROB_FEATURES
from winprob import load_model, match_state, win_probability

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")


@st.cache_resource  # load the model once and reuse it across interactions
def get_model():
    return load_model()


model = get_model()

st.title("🏏 IPL Win Probability Predictor")
st.caption("Logistic-regression model · ROC-AUC 0.875 · trained on 2008–2026 ball-by-ball data")

# ---------------- Inputs ----------------
st.subheader("Set the chase situation")
left, right = st.columns(2)
with left:
    chasing_team = st.text_input("Chasing team", "Kolkata Knight Riders")
    target = st.number_input("Target (runs to win)", 1, 300, 205)
    current_score = st.number_input("Current score", 0, 400, 120)
with right:
    bowling_team = st.text_input("Bowling team", "Gujarat Titans")
    overs = st.number_input("Overs completed", 0, 19, 14)
    balls = st.number_input("Balls in current over", 0, 5, 0)
    wickets = st.number_input("Wickets lost", 0, 9, 3)

balls_bowled = min(int(overs) * 6 + int(balls), 119)

# ---------------- Prediction ----------------
prob = win_probability(model, int(target), int(current_score), balls_bowled, int(wickets))
state = match_state(int(target), int(current_score), balls_bowled, int(wickets))
runs_left = int(state["runs_left"].iloc[0])
balls_left = int(state["balls_left"].iloc[0])

st.subheader("Prediction")
m1, m2 = st.columns(2)
m1.metric(f"{chasing_team or 'Chasing team'} win %", f"{prob * 100:.1f}%")
m2.metric(f"{bowling_team or 'Bowling team'} win %", f"{(1 - prob) * 100:.1f}%")
st.progress(prob)

if runs_left <= 0:
    st.success(f"{chasing_team} have already reached the target! 🎉")
else:
    st.write(
        f"**{chasing_team} need {runs_left} off {balls_left} balls** "
        f"&nbsp;·&nbsp; current RR {state['crr'].iloc[0]:.2f} "
        f"&nbsp;·&nbsp; required RR {state['rrr'].iloc[0]:.2f}",
        unsafe_allow_html=True,
    )

# ---------------- Sensitivity curve ----------------
st.subheader("How runs-still-needed shapes the odds")
runs_axis = np.arange(1, max(runs_left * 2, 60))
grid = pd.concat([state] * len(runs_axis), ignore_index=True)
grid["runs_left"] = runs_axis
grid["rrr"] = runs_axis / (balls_left / 6)
curve = model.predict_proba(grid[WINPROB_FEATURES])[:, 1] * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(runs_axis, curve, color="#4C72B0", lw=2)
ax.axvline(runs_left, ls="--", color="crimson", label=f"now: {runs_left} needed")
ax.axhline(50, ls=":", color="grey")
ax.set_xlabel("Runs still required (this many balls left)")
ax.set_ylabel(f"{chasing_team or 'Chasing'} win %")
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

st.caption("Educational/portfolio project · data: Cricsheet · DLS-affected matches excluded.")
