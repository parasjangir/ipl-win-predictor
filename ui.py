"""
ui.py -- the design system.
A small, reusable "futuristic dark" theme: colour palette, injected CSS,
glassmorphism KPI cards, gradient headers, player avatars, and a Plotly styler
so every chart shares the same look.
"""
from __future__ import annotations

import hashlib

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
SEQ = [CYAN, VIOLET, PINK, GREEN, GOLD, "#60a5fa", "#f87171", "#a3e635", "#e879f9", "#2dd4bf"]
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
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
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

/* Top navigation buttons */
.stButton > button {
    border-radius: 12px; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.15s ease;
}
.stButton > button:hover {border-color: rgba(34,211,238,0.55);}

/* Hide the (now-empty) sidebar — navigation lives in the top bar */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}

.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.72rem;
       border:1px solid rgba(34,211,238,0.4); color:#22d3ee; background:rgba(34,211,238,0.08);}
[data-testid="stMetricValue"] {font-family: 'Space Grotesk', sans-serif;}
.stPlotlyChart {border-radius: 16px; overflow: hidden;}
a {color: #22d3ee;}
hr {border-color: rgba(255,255,255,0.08);}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def top_header(title: str = "IPL INTEL", tag: str = "2008–2026 · ball-by-ball analytics") -> None:
    """Global brand bar rendered above the top navigation."""
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:14px;margin:0 0 8px 0;flex-wrap:wrap">
        <div style="font-family:'Orbitron',sans-serif;font-weight:800;font-size:1.55rem;
        background:linear-gradient(90deg,{CYAN},{VIOLET} 55%,{PINK});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🏏 {title}</div>
        <span class="pill">{tag}</span></div>""",
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div style="margin:2px 0 16px 0">
          <div style="font-family:'Orbitron',sans-serif;font-size:1.9rem;font-weight:800;
               background:linear-gradient(90deg,{CYAN},{VIOLET} 55%,{PINK});
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{title}</div>
          <div style="color:{MUTED};font-size:0.95rem;margin-top:2px">{subtitle}</div></div>""",
        unsafe_allow_html=True,
    )


def _kpi_html(label: str, value, sub: str = "", accent: str = CYAN) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{accent}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def kpi_row(items: list[dict]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(_kpi_html(**item), unsafe_allow_html=True)


def _initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "?"


def player_avatar(name: str, size: int = 72) -> str:
    """A clean gradient 'jersey' avatar (initials) with a stable per-player
    colour — a reliable, licence-free stand-in for a player photo."""
    ini = _initials(name)
    accent = SEQ[int(hashlib.md5(name.encode()).hexdigest(), 16) % len(SEQ)]
    fs = int(size * 0.36)
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f"display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;"
        f"font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:{fs}px;"
        f'color:#08101f;background:linear-gradient(135deg,{accent},#aab6cc);'
        f'box-shadow:0 6px 18px rgba(0,0,0,0.45),inset 0 0 0 2px rgba(255,255,255,0.18)">{ini}</div>'
    )


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
