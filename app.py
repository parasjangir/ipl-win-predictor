"""
app.py -- IPL Intel · a multi-page IPL analytics platform (Streamlit).

Navigation is an always-visible TOP BAR (no collapsible sidebar).
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import ui
from views import (battle_page, dashboard_page, h2h_page, player_page,
                   predictor_page, team_page)

st.set_page_config(page_title="IPL Intel", page_icon="🏏", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

PAGES = {
    "🛰️ Command Center": dashboard_page,
    "🔬 Team Deep Dive": team_page,
    "👤 Player Analytics": player_page,
    "🥊 Player Battles": battle_page,
    "⚔️ Head to Head": h2h_page,
    "🎯 Win Predictor": predictor_page,
}

ui.top_header()

if "page" not in st.session_state or st.session_state.page not in PAGES:
    st.session_state.page = next(iter(PAGES))

# --- Top navigation bar: one button per page, active page highlighted ---
nav_cols = st.columns(len(PAGES))
for col, label in zip(nav_cols, PAGES):
    is_active = st.session_state.page == label
    if col.button(label, width="stretch", type="primary" if is_active else "secondary"):
        st.session_state.page = label
        st.rerun()

st.markdown(
    "<hr style='margin:6px 0 18px 0;border:none;border-top:1px solid rgba(255,255,255,0.08)'>",
    unsafe_allow_html=True,
)

# --- Render the selected page ---
PAGES[st.session_state.page]()

st.markdown(
    "<div style='text-align:center;color:#4d5874;font-size:0.72rem;margin-top:2.5rem'>"
    "IPL Intel · data: Cricsheet · model: logistic regression (AUC 0.875) · built by Paras Jangir</div>",
    unsafe_allow_html=True,
)
