"""
app.py -- IPL Intelligence Platform (Phase 4, Streamlit).

A multi-page analytics platform:
  * Command Center  -- league-wide dashboard
  * Team Deep Dive  -- full report on any franchise
  * Head to Head    -- franchise vs franchise
  * Win Predictor   -- live ball-by-ball chase probability (ML model)

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import ui
from views import (battle_page, dashboard_page, h2h_page, player_page,
                   predictor_page, team_page)

st.set_page_config(page_title="IPL Intelligence", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

with st.sidebar:
    ui.sidebar_brand()

pages = [
    st.Page(dashboard_page, title="Command Center", icon="🛰️", default=True),
    st.Page(team_page, title="Team Deep Dive", icon="🔬"),
    st.Page(player_page, title="Player Analytics", icon="👤"),
    st.Page(battle_page, title="Player Battles", icon="🥊"),
    st.Page(h2h_page, title="Head to Head", icon="⚔️"),
    st.Page(predictor_page, title="Win Predictor", icon="🎯"),
]
st.navigation(pages).run()

with st.sidebar:
    st.markdown("<div style='color:#5b6680;font-size:0.7rem;margin-top:1rem'>"
                "Data: Cricsheet · Model: logistic regression (AUC 0.875)<br>"
                "Built by Paras Jangir</div>", unsafe_allow_html=True)
