"""
ui.py -- the design system.
A small, reusable "futuristic dark" theme: colour palette, injected CSS,
glassmorphism KPI cards, gradient hero headers, and a Plotly styler so every
chart shares the same look.
"""
from __future__ import annotations

import streamlit as st

# ---- Palette ----
BG0 = "#070b16"
BG1 = "#0c1428"
TEXT = "#E8EEF9"
MUTED = "#8a97b1"
CYAN = "#22d3ee"
VIOLET = "#8b5cf6"
PINK = "#f471b5"
GREEN = "#34d399"
RED = "#fb7185"
GOLD = "#fbbf24"
# Categorical colour sequence for multi-series charts.
SEQ = [CYAN, VIOLET, PINK, GREEN, GOLD, "#60a5fa", "#f87171", "#a3e635", "#e879f9", "#2dd4bf"]
# Red -> amber -> green scale for "win %" style metrics.
WIN_SCALE = [[0.0, "#fb7185"], [0.5, "#fbbf24"], [1.0, "#34d399"]]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Orbitron:wght@700;800&display=swap');

.stApp {
    background:
        radial-gradient(1200px 600px at 8% -10%, rgba(139,92,246,0.20), transparent 60%),
        radial-gradient(1000px 520px at 100% -5%, rgba(34,211,238,0.14), transparent 55%),
        linear-gradient(180deg, #070b16 0%, #0c1428 100%);
    color: #E8EEF9;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3, h4 {font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.4px; color: #E8EEF9;}

.kpi {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(139,92,246,0.22);
    border-radius: 18px; padding: 16px 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.02);
    height: 100%;
}
.kpi .label {font-size: 0.70rem; text-transform: uppercase; letter-spacing: 1.6px; color: #8a97b1;}
.kpi .value {font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.1; margin-top: 8px;}
.kpi .sub {font-size: 0.76rem; color: #8a97b1; margin-top: 6px;}

section[data-testid="stSidebar"] {
    background: rgba(7,11,22,0.65);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stMetricValue"] {font-family: 'Space Grotesk', sans-serif;}
.stPlotlyChart {border-radius: 16px; overflow: hidden;}
a {color: #22d3ee;}
hr {border-color: rgba(255,255,255,0.08);}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        """
        <div style="padding:4px 2px 16px 2px">
          <div style="font-family:'Orbitron',sans-serif;font-weight:800;font-size:1.2rem;
               background:linear-gradient(90deg,#22d3ee,#8b5cf6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🏏 IPL&nbsp;INTEL</div>
          <div style="color:#8a97b1;font-size:0.68rem;letter-spacing:2px;margin-top:2px">ANALYTICS&nbsp;PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin:2px 0 18px 0">
          <div style="font-family:'Orbitron',sans-serif;font-size:2.0rem;font-weight:800;
               background:linear-gradient(90deg,#22d3ee,#8b5cf6 55%,#f471b5);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{title}</div>
          <div style="color:#8a97b1;font-size:0.95rem;margin-top:4px">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_html(label: str, value, sub: str = "", accent: str = CYAN) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{accent}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def kpi_row(items: list[dict]) -> None:
    """Render a row of KPI cards. Each item: {label, value, sub, accent}."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(_kpi_html(**item), unsafe_allow_html=True)


def style_fig(fig, height: int = 340, legend: bool = False):
    """Apply the shared dark/transparent Plotly look to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=30, b=8),
        height=height,
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="#0c1428", font_size=12, font_family="Inter"),
        colorway=SEQ,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    return fig
