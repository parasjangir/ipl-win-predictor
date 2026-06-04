"""
views.py -- the four pages of the platform.
Each function renders one page using analytics (data) + ui (design).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import ui
from cricket_utils import WINPROB_FEATURES
from winprob import load_model, match_state


@st.cache_resource(show_spinner=False)
def _model():
    return load_model()


# ==========================================================================
# PAGE 1 -- League Command Center
# ==========================================================================
def dashboard_page() -> None:
    ui.hero("IPL COMMAND CENTER", "every IPL season · 2008–2026 · ball-by-ball intelligence")

    k = an.league_kpis()
    ui.kpi_row([
        dict(label="Matches", value=f"{k['matches']:,}", sub=f"{k['seasons']} seasons · {k['teams']} franchises", accent=ui.CYAN),
        dict(label="Deliveries", value=f"{k['balls'] / 1000:.0f}K", sub="every ball, tracked", accent=ui.VIOLET),
        dict(label="Runs scored", value=f"{k['runs'] / 1000:.0f}K", sub=f"{k['sixes']:,} sixes · {k['fours']:,} fours", accent=ui.PINK),
        dict(label="Highest total", value=f"{k['highest_total']}", sub=k["highest_total_team"], accent=ui.GOLD),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matches per season")
        mps = an.matches_per_season()
        fig = px.bar(mps, x="season", y="matches")
        fig.update_traces(marker_color=ui.CYAN, marker_line_width=0)
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    with c2:
        st.markdown("#### Scoring inflation · avg 1st-innings score")
        a = an.avg_first_innings_by_season()
        fig = px.area(a, x="season", y="avg_score")
        fig.update_traces(line_color=ui.VIOLET, fillcolor="rgba(139,92,246,0.18)")
        st.plotly_chart(ui.style_fig(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Most successful franchises")
        w = an.wins_by_team().head(10).sort_values("wins")
        fig = px.bar(w, x="wins", y="team", orientation="h")
        fig.update_traces(marker_color=ui.GREEN)
        st.plotly_chart(ui.style_fig(fig, height=380), width="stretch")
    with c4:
        st.markdown("#### The rise of chasing · % electing to field")
        t = an.toss_field_pct_by_season()
        fig = px.line(t, x="season", y="field_pct", markers=True)
        fig.update_traces(line_color=ui.PINK, line_width=3)
        fig.add_hline(y=50, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(fig, height=380), width="stretch")

    with st.expander("ℹ️  About PitchIQ"):
        st.markdown(
            "**PitchIQ** turns ~296,000 ball-by-ball IPL deliveries (2008–2026, source: "
            "[Cricsheet](https://cricsheet.org)) into an end-to-end analytics platform — "
            "data engineering, EDA, statistics, and a calibrated machine-learning "
            "**win-probability model** (ROC-AUC 0.875). "
            "Built by **Paras Jangir** as a data-science portfolio project."
        )


# ==========================================================================
# PAGE 2 -- Team Deep Dive
# ==========================================================================
def team_page() -> None:
    ui.hero("TEAM DEEP DIVE", "a full intelligence report on any franchise")

    teams = an.team_list()
    default = teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0
    team = st.selectbox("Select a franchise", teams, index=default)

    k = an.team_kpis(team)
    big = f"by {k['big_runs']} runs" if k["big_runs"] >= k["big_wkts"] else f"by {k['big_wkts']} wkts"
    ui.kpi_row([
        dict(label="Matches", value=f"{k['played']}", sub=f"{k['first_season']}–{k['last_season']}", accent=ui.CYAN),
        dict(label="Win rate", value=f"{k['winpct']:.1f}%", sub=f"{k['won']}W · {k['lost']}L", accent=ui.GREEN),
        dict(label="Batting first", value=f"{k['bf_wp']:.0f}%", sub="win rate", accent=ui.GOLD),
        dict(label="Chasing", value=f"{k['ch_wp']:.0f}%", sub="win rate", accent=ui.VIOLET),
    ])
    st.write("")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Win rate by season")
        s = an.team_by_season(team)
        fig = px.bar(s, x="season_year", y="win_pct", custom_data=["won", "played"])
        fig.update_traces(marker_color=ui.CYAN,
                          hovertemplate="%{x}<br>%{y:.0f}% (%{customdata[0]}/%{customdata[1]})<extra></extra>")
        fig.add_hline(y=50, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    with c2:
        st.markdown("#### Results split")
        fig = go.Figure(go.Pie(
            labels=["Won", "Lost", "No result"], values=[k["won"], k["lost"], k["nr"]],
            hole=0.62, marker_colors=[ui.GREEN, ui.RED, ui.MUTED], sort=False,
        ))
        fig.update_traces(textinfo="percent", textfont_size=13)
        st.plotly_chart(ui.style_fig(fig, height=340, legend=True), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Top run-scorers")
        b = an.team_top_batters(team).sort_values("runs")
        fig = px.bar(b, x="runs", y="striker", orientation="h", custom_data=["sr"])
        fig.update_traces(marker_color=ui.GOLD,
                          hovertemplate="%{y}<br>%{x} runs · SR %{customdata[0]:.1f}<extra></extra>")
        st.plotly_chart(ui.style_fig(fig, height=400), width="stretch")
    with c4:
        st.markdown("#### Top wicket-takers")
        bw = an.team_top_bowlers(team).sort_values("wickets")
        fig = px.bar(bw, x="wickets", y="bowler", orientation="h")
        fig.update_traces(marker_color=ui.PINK)
        st.plotly_chart(ui.style_fig(fig, height=400), width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Record vs every opponent")
        h = an.team_h2h(team).sort_values("win_pct")
        fig = px.bar(h, x="win_pct", y="opponent", orientation="h",
                     color="win_pct", color_continuous_scale=ui.WIN_SCALE, range_color=[0, 100],
                     custom_data=["won", "played"])
        fig.update_traces(hovertemplate="%{y}<br>%{x:.0f}% (%{customdata[0]}/%{customdata[1]})<extra></extra>")
        fig.add_vline(x=50, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(fig, height=420), width="stretch")
    with c6:
        st.markdown("#### Fortress venues · most played")
        v = an.team_venues(team).sort_values("played")
        fig = px.bar(v, x="played", y="venue_short", orientation="h",
                     color="win_pct", color_continuous_scale=ui.WIN_SCALE, range_color=[0, 100],
                     custom_data=["win_pct"])
        fig.update_traces(hovertemplate="%{y}<br>%{x} played · %{customdata[0]:.0f}% won<extra></extra>")
        st.plotly_chart(ui.style_fig(fig, height=420), width="stretch")


# ==========================================================================
# PAGE 3 -- Head to Head
# ==========================================================================
def h2h_page() -> None:
    ui.hero("HEAD TO HEAD", "franchise vs franchise, all-time")

    teams = an.team_list()
    ia = teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0
    ib = teams.index("Chennai Super Kings") if "Chennai Super Kings" in teams else 1
    c1, c2 = st.columns(2)
    a = c1.selectbox("Team A", teams, index=ia)
    b = c2.selectbox("Team B", teams, index=ib)
    if a == b:
        st.warning("Pick two different franchises.")
        return

    s, recent = an.head_to_head(a, b)
    if s["played"] == 0:
        st.info("These two franchises have never met in the IPL.")
        return

    leader = a if s["a_wins"] >= s["b_wins"] else b
    ui.kpi_row([
        dict(label="Meetings", value=s["played"], sub="all-time", accent=ui.CYAN),
        dict(label=f"{a}", value=s["a_wins"], sub="wins", accent=ui.VIOLET),
        dict(label=f"{b}", value=s["b_wins"], sub="wins", accent=ui.PINK),
        dict(label="Edge", value=leader.split()[-1], sub=f"+{abs(s['a_wins'] - s['b_wins'])}", accent=ui.GOLD),
    ])
    st.write("")

    c3, c4 = st.columns([1, 1.2])
    with c3:
        st.markdown("#### Win share")
        fig = go.Figure(go.Bar(
            x=[s["a_wins"], s["b_wins"], s["nr"]], y=["", "", ""], orientation="h",
            marker_color=[ui.VIOLET, ui.PINK, ui.MUTED],
            text=[a, b, "No result"], textposition="inside", insidetextanchor="middle",
        ))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(visible=False)
        st.plotly_chart(ui.style_fig(fig, height=180), width="stretch")
        st.caption("Each segment = number of wins.")
    with c4:
        st.markdown("#### Last 10 meetings")
        st.dataframe(
            recent.rename(columns={"date": "Date", "winner": "Winner", "venue": "Venue"}),
            hide_index=True, width="stretch",
        )


# ==========================================================================
# PAGE 4 -- Win Predictor
# ==========================================================================
def predictor_page() -> None:
    ui.hero("WIN PREDICTOR", "live ball-by-ball chase probability · model ROC-AUC 0.875")
    model = _model()
    teams = an.team_list()

    c1, c2, c3 = st.columns(3)
    with c1:
        chasing = st.selectbox("Chasing team", teams,
                               index=teams.index("Kolkata Knight Riders") if "Kolkata Knight Riders" in teams else 0)
        target = st.number_input("Target (runs to win)", 1, 300, 205)
    with c2:
        bowling = st.selectbox("Bowling team", teams,
                               index=teams.index("Gujarat Titans") if "Gujarat Titans" in teams else 1)
        score = st.number_input("Current score", 0, 400, 120)
    with c3:
        overs = st.number_input("Overs completed", 0, 19, 14)
        balls = st.number_input("Balls this over", 0, 5, 0)
    wickets = st.slider("Wickets lost", 0, 9, 3)

    balls_bowled = min(int(overs) * 6 + int(balls), 119)
    state = match_state(int(target), int(score), balls_bowled, int(wickets))
    prob = float(model.predict_proba(state)[0, 1])
    runs_left = int(state["runs_left"].iloc[0])
    balls_left = int(state["balls_left"].iloc[0])

    g1, g2 = st.columns([1, 1.1])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 46}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": ui.MUTED},
                "bar": {"color": ui.CYAN, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.03)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(251,113,133,0.16)"},
                    {"range": [50, 100], "color": "rgba(52,211,153,0.16)"},
                ],
                "threshold": {"line": {"color": ui.GOLD, "width": 3}, "thickness": 0.85, "value": 50},
            },
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": ui.TEXT, "family": "Inter"},
                            height=300, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(
            f"<div style='text-align:center;color:#8a97b1'>"
            f"<b style='color:{ui.CYAN}'>{chasing}</b> {prob*100:.1f}% &nbsp;|&nbsp; "
            f"<b style='color:{ui.PINK}'>{bowling}</b> {(1-prob)*100:.1f}%</div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown("#### How runs-still-needed shapes the odds")
        runs_axis = np.arange(1, max(runs_left * 2, 60))
        grid = pd.concat([state] * len(runs_axis), ignore_index=True)
        grid["runs_left"] = runs_axis
        grid["rrr"] = runs_axis / (balls_left / 6)
        curve = model.predict_proba(grid[WINPROB_FEATURES])[:, 1] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runs_axis, y=curve, mode="lines",
                                 line=dict(color=ui.CYAN, width=3), fill="tozeroy",
                                 fillcolor="rgba(34,211,238,0.12)", name="win %"))
        fig.add_vline(x=runs_left, line_dash="dash", line_color=ui.GOLD,
                      annotation_text=f"now: {runs_left} needed", annotation_font_color=ui.GOLD)
        fig.add_hline(y=50, line_dash="dot", line_color=ui.MUTED)
        fig.update_xaxes(title="runs still required")
        fig.update_yaxes(title="win %", range=[0, 100])
        st.plotly_chart(ui.style_fig(fig, height=300), width="stretch")

    if runs_left <= 0:
        st.success(f"{chasing} have already reached the target! 🎉")
    else:
        verdict = "cruising" if prob > 0.75 else "in control" if prob > 0.55 else \
            "on a knife's edge" if prob > 0.45 else "up against it" if prob > 0.2 else "almost gone"
        st.markdown(
            f"**{chasing} need {runs_left} off {balls_left} balls** &nbsp;·&nbsp; "
            f"required RR **{state['rrr'].iloc[0]:.2f}** &nbsp;·&nbsp; status: *{verdict}*"
        )


# ==========================================================================
# PAGE 5 -- Player Analytics
# ==========================================================================
def player_page() -> None:
    ui.hero("PLAYER ANALYTICS", "batting deep-dive · strengths, weaknesses & scoring DNA")

    batters = an.batter_list()
    idx = batters.index("V Kohli") if "V Kohli" in batters else 0
    player = st.selectbox("Select a batter", batters, index=idx)

    c = an.player_career(player)
    photo = an.player_photo_url(player)
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:16px;margin:2px 0 14px">
        {ui.player_face(player, photo, 76)}
        <div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#fff">{player}</div>
        <div style="color:{ui.MUTED};font-size:0.85rem">{c['innings']} innings · {c['runs']:,} runs · SR {c['sr']:.1f}</div></div></div>""",
        unsafe_allow_html=True,
    )
    ui.kpi_row([
        dict(label="Runs", value=f"{c['runs']:,}", sub=f"{c['innings']} innings", accent=ui.CYAN),
        dict(label="Strike rate", value=f"{c['sr']:.1f}", sub=f"average {c['avg']:.1f}", accent=ui.VIOLET),
        dict(label="Boundaries", value=f"{c['fours'] + c['sixes']:,}", sub=f"{c['fours']}×4 · {c['sixes']}×6", accent=ui.PINK),
        dict(label="50s / 100s", value=f"{c['fifties']} / {c['hundreds']}", sub=f"high score {c['highest']}", accent=ui.GOLD),
    ])
    st.write("")

    # ---- Auto-detected strengths & weaknesses ----
    ph = an.player_phase(player)
    dis = an.player_dismissals(player)
    nem = an.player_nemeses(player)
    ph_valid = ph[ph["balls"] >= 50]
    strong = ph_valid.loc[ph_valid["sr"].idxmax()] if len(ph_valid) else None
    weak = ph_valid.loc[ph_valid["sr"].idxmin()] if len(ph_valid) else None
    top_dis = dis.iloc[0] if len(dis) else None
    top_nem = nem.iloc[0] if len(nem) else None
    ui.kpi_row([
        dict(label="Strong zone", value=(strong["phase"] if strong is not None else "—"),
             sub=(f"SR {strong['sr']:.0f}" if strong is not None else ""), accent=ui.GREEN),
        dict(label="Quieter zone", value=(weak["phase"] if weak is not None else "—"),
             sub=(f"SR {weak['sr']:.0f}" if weak is not None else ""), accent=ui.GOLD),
        dict(label="Usual downfall", value=(str(top_dis["type"]).title() if top_dis is not None else "—"),
             sub=(f"{int(top_dis['count'])} times" if top_dis is not None else ""), accent=ui.RED),
        dict(label="Nemesis bowler", value=(top_nem["bowler"] if top_nem is not None else "—"),
             sub=(f"{int(top_nem['dismissals'])} dismissals" if top_nem is not None else ""), accent=ui.PINK),
    ])
    st.write("")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### Acceleration curve · strike rate by over")
        op = an.player_over_profile(player)
        fig = px.line(op, x="over", y="sr", markers=True, custom_data=["runs", "balls"])
        fig.update_traces(line_color=ui.CYAN, line_width=3,
                          hovertemplate="over %{x}<br>SR %{y:.0f} (%{customdata[0]} off %{customdata[1]})<extra></extra>")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    with c2:
        st.markdown("#### Scoring DNA")
        dna = an.player_scoring_dna(player)
        fig = go.Figure(go.Pie(
            labels=list(dna.keys()), values=list(dna.values()), hole=0.6, sort=False,
            marker_colors=[ui.MUTED, "#60a5fa", ui.CYAN, ui.VIOLET, ui.GOLD, ui.PINK],
        ))
        fig.update_traces(textinfo="percent")
        st.plotly_chart(ui.style_fig(fig, height=300, legend=True), width="stretch")
        st.caption("Run-type split — an honest stand-in for a wagon wheel (shot-direction data isn't in the free feed).")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Strike rate by phase")
        fig = px.bar(ph, x="phase", y="sr", custom_data=["runs", "balls"])
        fig.update_traces(marker_color=ui.VIOLET,
                          hovertemplate="%{x}<br>SR %{y:.1f} (%{customdata[0]} off %{customdata[1]})<extra></extra>")
        st.plotly_chart(ui.style_fig(fig, height=340), width="stretch")
    with c4:
        st.markdown("#### Nemesis bowlers · most dismissals")
        if len(nem):
            fig = px.bar(nem.sort_values("dismissals"), x="dismissals", y="bowler", orientation="h")
            fig.update_traces(marker_color=ui.RED)
            st.plotly_chart(ui.style_fig(fig, height=340), width="stretch")
        else:
            st.info("No bowler-credited dismissals on record.")

    st.markdown("#### Runs by season")
    sea = an.player_by_season(player)
    fig = px.bar(sea, x="season_year", y="runs", custom_data=["sr"])
    fig.update_traces(marker_color=ui.GOLD,
                      hovertemplate="%{x}<br>%{y} runs · SR %{customdata[0]:.0f}<extra></extra>")
    st.plotly_chart(ui.style_fig(fig, height=300), width="stretch")


# ==========================================================================
# PAGE 6 -- Player Battles (batter vs bowler)
# ==========================================================================
def _battle_verdict(b: dict) -> str:
    sr, d, balls = b["sr"], b["dismissals"], b["balls"]
    bpd = balls / d if d else None
    if d == 0 and sr >= 130:
        return f"🟢 <b>Batter dominates</b> — {sr:.0f} strike rate and never dismissed."
    if sr >= 140 and (bpd is None or bpd >= 18):
        return f"🟢 <b>Batter on top</b> — scoring at {sr:.0f} with few dismissals."
    if sr <= 110 and d >= 2:
        return f"🔴 <b>Bowler on top</b> — kept the batter to {sr:.0f} SR with {d} dismissals."
    return f"🟡 <b>Even contest</b> — {sr:.0f} SR with {d} dismissal(s) across {balls} balls."


def battle_page() -> None:
    ui.hero("PLAYER BATTLES", "batter vs bowler · the matchups that decide games")

    batters = an.batter_list(min_balls=300)
    bowlers = an.bowler_list(min_balls=300)
    c1, c2 = st.columns(2)
    bat = c1.selectbox("Batter", batters, index=batters.index("V Kohli") if "V Kohli" in batters else 0)
    bowl = c2.selectbox("Bowler", bowlers, index=bowlers.index("JJ Bumrah") if "JJ Bumrah" in bowlers else 0)

    b = an.battle(bat, bowl)
    if b["balls"] == 0:
        st.info(f"📭 {bat} and {bowl} have never met in the IPL (in our data). Try another pairing.")
        return

    pbat, pbowl = an.player_photo_url(bat), an.player_photo_url(bowl)
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:center;gap:26px;margin:8px 0 18px">
        <div style="text-align:center">{ui.player_face(bat, pbat, 80)}
        <div style="margin-top:8px;color:{ui.CYAN};font-weight:600">{bat}</div></div>
        <div style="font-family:'Orbitron',sans-serif;font-size:1.2rem;color:{ui.MUTED}">VS</div>
        <div style="text-align:center">{ui.player_face(bowl, pbowl, 80)}
        <div style="margin-top:8px;color:{ui.PINK};font-weight:600">{bowl}</div></div></div>""",
        unsafe_allow_html=True,
    )
    ui.kpi_row([
        dict(label="Balls faced", value=b["balls"], sub=f"{b['dots']} dots", accent=ui.CYAN),
        dict(label="Runs", value=b["runs"], sub=f"{b['fours']}×4 · {b['sixes']}×6", accent=ui.VIOLET),
        dict(label="Strike rate", value=f"{b['sr']:.1f}", sub="in this matchup", accent=ui.PINK),
        dict(label="Dismissals", value=b["dismissals"],
             sub=(f"avg {b['avg']:.1f}" if b["dismissals"] else "never out"), accent=ui.GOLD),
    ])
    st.markdown(
        f"<div style='text-align:center;font-size:1.05rem;padding:10px 0 4px 0'>"
        f"<span style='color:{ui.CYAN};font-weight:700'>{bat}</span> "
        f"<span style='color:{ui.MUTED}'>vs</span> "
        f"<span style='color:{ui.PINK};font-weight:700'>{bowl}</span><br>{_battle_verdict(b)}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c3, c4 = st.columns([1, 1.3])
    with c3:
        st.markdown("#### Every ball, by outcome")
        oc = an.battle_outcomes(bat, bowl)
        fig = px.bar(x=list(oc.keys()), y=list(oc.values()))
        fig.update_traces(marker_color=[ui.MUTED, "#60a5fa", ui.CYAN, ui.VIOLET, ui.GOLD, ui.PINK])
        fig.update_xaxes(title="runs off the ball")
        fig.update_yaxes(title="balls")
        st.plotly_chart(ui.style_fig(fig, height=330), width="stretch")
    with c4:
        st.markdown("#### This battle, season by season")
        bs = an.battle_by_season(bat, bowl)
        if len(bs) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bs["season_year"], y=bs["runs"], name="runs", marker_color=ui.VIOLET))
            fig.add_trace(go.Scatter(x=bs["season_year"], y=bs["balls"], name="balls",
                                     mode="lines+markers", line=dict(color=ui.CYAN, width=3)))
            st.plotly_chart(ui.style_fig(fig, height=330, legend=True), width="stretch")
        else:
            st.info("Not enough data for a seasonal split.")


# ==========================================================================
# PAGE -- Bowler Analytics
# ==========================================================================
def bowler_page() -> None:
    ui.hero("BOWLER ANALYTICS", "bowling deep-dive · economy, wickets & favourite victims")
    bowlers = an.bowler_list()
    idx = bowlers.index("JJ Bumrah") if "JJ Bumrah" in bowlers else 0
    bowler = st.selectbox("Select a bowler", bowlers, index=idx)
    photo = an.player_photo_url(bowler)
    k = an.bowler_career(bowler)
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:16px;margin:2px 0 14px">
        {ui.player_face(bowler, photo, 76)}
        <div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#fff">{bowler}</div>
        <div style="color:{ui.MUTED};font-size:0.85rem">{k['wickets']} wickets · econ {k['economy']:.2f} · {k['balls'] // 6} overs</div></div></div>""",
        unsafe_allow_html=True,
    )
    ui.kpi_row([
        dict(label="Wickets", value=k["wickets"], sub=f"average {k['average']:.1f}", accent=ui.CYAN),
        dict(label="Economy", value=f"{k['economy']:.2f}", sub="runs per over", accent=ui.GREEN),
        dict(label="Strike rate", value=f"{k['sr']:.1f}", sub="balls per wicket", accent=ui.VIOLET),
        dict(label="Dot balls", value=f"{k['dot_pct']:.0f}%", sub="pressure", accent=ui.GOLD),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Economy by phase")
        ph = an.bowler_phase(bowler)
        fig = px.bar(ph, x="phase", y="econ")
        fig.update_traces(marker_color=ui.GREEN)
        st.plotly_chart(ui.style_fig(fig, height=330), width="stretch")
    with c2:
        st.markdown("#### Wickets by season")
        s = an.bowler_by_season(bowler)
        fig = px.bar(s, x="season_year", y="wickets")
        fig.update_traces(marker_color=ui.CYAN)
        st.plotly_chart(ui.style_fig(fig, height=330), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Favourite victims")
        v = an.bowler_victims(bowler).sort_values("dismissals")
        fig = px.bar(v, x="dismissals", y="victim", orientation="h")
        fig.update_traces(marker_color=ui.PINK)
        st.plotly_chart(ui.style_fig(fig, height=360), width="stretch")
    with c4:
        st.markdown("#### How they take wickets")
        dt = an.bowler_dismissal_types(bowler)
        fig = go.Figure(go.Pie(labels=dt["type"], values=dt["count"], hole=0.6, sort=False))
        fig.update_traces(marker_colors=ui.SEQ, textinfo="percent")
        st.plotly_chart(ui.style_fig(fig, height=360, legend=True), width="stretch")


# ==========================================================================
# PAGE -- Venue / Pitch Report
# ==========================================================================
def venue_page() -> None:
    ui.hero("VENUE · PITCH REPORT", "batting paradise or bowler's graveyard?")
    venues = an.venue_list()
    venue = st.selectbox("Select a venue", venues)
    k = an.venue_report(venue)
    ui.kpi_row([
        dict(label="Matches", value=k["matches"], sub="hosted", accent=ui.CYAN),
        dict(label="Avg 1st innings", value=f"{k['avg_first']:.0f}", sub=f"highest {k['highest']}", accent=ui.GOLD),
        dict(label="Chose to field", value=f"{k['pct_field']:.0f}%", sub="at the toss", accent=ui.VIOLET),
        dict(label="Chasing wins", value=f"{k['chase_win']:.0f}%", sub=f"batting first {k['bat_win']:.0f}%", accent=ui.GREEN),
    ])
    st.write("")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### Average 1st-innings score by season")
        s = an.venue_avg_by_season(venue)
        fig = px.line(s, x="season_year", y="avg_score", markers=True)
        fig.update_traces(line_color=ui.CYAN, line_width=3)
        st.plotly_chart(ui.style_fig(fig, height=360), width="stretch")
    with c2:
        st.markdown("#### Bat first vs chase")
        fig = go.Figure(go.Pie(labels=["Batting first", "Chasing"], values=[k["bat_win"], k["chase_win"]],
                               hole=0.6, marker_colors=[ui.GOLD, ui.GREEN], sort=False))
        fig.update_traces(textinfo="percent")
        st.plotly_chart(ui.style_fig(fig, height=360, legend=True), width="stretch")


# ==========================================================================
# PAGE -- Season Explorer
# ==========================================================================
def season_page() -> None:
    ui.hero("SEASON EXPLORER", "standings, caps & champions, season by season")
    seasons = an.season_list()
    season = st.selectbox("Select a season", seasons, index=len(seasons) - 1)
    k = an.season_summary(season)
    ui.kpi_row([
        dict(label="Champion", value=k["champion"], sub=f"{season}", accent=ui.GOLD),
        dict(label="Matches", value=k["matches"], sub="played", accent=ui.CYAN),
        dict(label="Orange Cap", value=k["orange"], sub=f"{k['orange_runs']} runs", accent=ui.VIOLET),
        dict(label="Purple Cap", value=k["purple"], sub=f"{k['purple_wkts']} wickets", accent=ui.PINK),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Wins by team")
        sd = an.season_standings(season).sort_values("wins")
        fig = px.bar(sd, x="wins", y="team", orientation="h")
        fig.update_traces(marker_color=ui.CYAN)
        st.plotly_chart(ui.style_fig(fig, height=380), width="stretch")
    with c2:
        st.markdown("#### Orange Cap race")
        oc = an.season_orange_cap(season).sort_values("runs")
        fig = px.bar(oc, x="runs", y="striker", orientation="h")
        fig.update_traces(marker_color=ui.GOLD)
        st.plotly_chart(ui.style_fig(fig, height=380), width="stretch")


# ==========================================================================
# PAGE -- Player Compare
# ==========================================================================
def compare_page() -> None:
    ui.hero("PLAYER COMPARE", "two batters, head to head")
    batters = an.batter_list(min_balls=500)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Batter A", batters, index=batters.index("V Kohli") if "V Kohli" in batters else 0)
    b = c2.selectbox("Batter B", batters, index=batters.index("RG Sharma") if "RG Sharma" in batters else 1)
    if a == b:
        st.warning("Pick two different batters.")
        return

    ca, cb = an.player_career(a), an.player_career(b)
    pa, pb = an.player_photo_url(a), an.player_photo_url(b)
    f1, f2 = st.columns(2)
    f1.markdown(
        f"""<div style="display:flex;align-items:center;gap:14px">{ui.player_face(a, pa, 70)}
        <div><div style="font-weight:700;font-size:1.2rem;color:{ui.CYAN}">{a}</div>
        <div style="color:{ui.MUTED};font-size:0.8rem">{ca['runs']:,} runs · SR {ca['sr']:.1f} · avg {ca['avg']:.1f}</div></div></div>""",
        unsafe_allow_html=True,
    )
    f2.markdown(
        f"""<div style="display:flex;align-items:center;gap:14px">{ui.player_face(b, pb, 70)}
        <div><div style="font-weight:700;font-size:1.2rem;color:{ui.PINK}">{b}</div>
        <div style="color:{ui.MUTED};font-size:0.8rem">{cb['runs']:,} runs · SR {cb['sr']:.1f} · avg {cb['avg']:.1f}</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    labels, na, nb, va, vb = an.compare_metrics(a, b)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=na + [na[0]], theta=labels + [labels[0]], fill="toself",
                                  name=a, line_color=ui.CYAN, fillcolor="rgba(34,211,238,0.18)"))
    fig.add_trace(go.Scatterpolar(r=nb + [nb[0]], theta=labels + [labels[0]], fill="toself",
                                  name=b, line_color=ui.PINK, fillcolor="rgba(244,113,181,0.18)"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(255,255,255,0.10)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.10)")),
    )
    st.plotly_chart(ui.style_fig(fig, height=460, legend=True), width="stretch")
    st.caption("Radar axes scaled 0–100 for a fair shape comparison; raw figures are shown above each player.")
