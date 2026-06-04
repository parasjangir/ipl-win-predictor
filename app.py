"""
app.py -- PitchIQ · an IPL analytics platform (Streamlit).

Always-visible top navigation (wrapping pills) -> ten pages.
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import ui
from views import (battle_page, bowler_page, compare_page, dashboard_page,
                   h2h_page, player_page, predictor_page, season_page,
                   team_page, venue_page)

st.set_page_config(page_title="PitchIQ", page_icon="🏏", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

PAGES = {
    "🛰️ Command Center": dashboard_page,
    "🔬 Team Deep Dive": team_page,
    "⚔️ Head to Head": h2h_page,
    "👤 Player Analytics": player_page,
    "🎳 Bowler Analytics": bowler_page,
    "🥊 Player Battles": battle_page,
    "⚖️ Player Compare": compare_page,
    "🏟️ Venues": venue_page,
    "📅 Seasons": season_page,
    "🎯 Win Predictor": predictor_page,
}

ui.top_header("PITCHIQ", "IPL ball-by-ball analytics · 2008–2026")

labels = list(PAGES)
choice = st.pills("Navigate", labels, selection_mode="single", default=labels[0],
                  label_visibility="collapsed", key="nav")
page = choice or labels[0]

st.markdown(
    "<hr style='margin:6px 0 18px 0;border:none;border-top:1px solid rgba(255,255,255,0.08)'>",
    unsafe_allow_html=True,
)

PAGES[page]()

st.markdown(
    "<div style='text-align:center;color:#4d5874;font-size:0.72rem;margin-top:2.5rem'>"
    "PitchIQ · data: Cricsheet · model: logistic regression (AUC 0.875) · built by Paras Jangir</div>",
    unsafe_allow_html=True,
)
