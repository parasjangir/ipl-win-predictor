"""
analytics.py -- the data/intelligence layer.

Cached loaders + all the aggregations the dashboard needs. Every function is
wrapped in st.cache_data so a query runs once and is instant thereafter.
Heavy lifting (groupby over 295k deliveries) happens here, not in the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cricket_utils import BOWLER_WICKETS, load_deliveries, load_matches


# --------------------------------------------------------------------------
# Cached loaders (loaded once per session)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_matches() -> pd.DataFrame:
    return load_matches()


@st.cache_data(show_spinner=False)
def get_deliveries() -> pd.DataFrame:
    return load_deliveries()


@st.cache_data(show_spinner=False)
def team_list() -> list[str]:
    m = get_matches()
    teams = pd.unique(m[["team1", "team2"]].values.ravel())
    return sorted(t for t in teams if pd.notna(t))


# --------------------------------------------------------------------------
# League-wide
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def league_kpis() -> dict:
    m, d = get_matches(), get_deliveries()
    totals = d.groupby(["match_id", "innings", "batting_team"])["total_runs"].sum().reset_index()
    top = totals.loc[totals["total_runs"].idxmax()]
    return {
        "matches": int(len(m)),
        "seasons": int(m["season_year"].nunique()),
        "balls": int(len(d)),
        "runs": int(d["total_runs"].sum()),
        "sixes": int((d["runs_off_bat"] == 6).sum()),
        "fours": int((d["runs_off_bat"] == 4).sum()),
        "teams": int(len(team_list())),
        "highest_total": int(top["total_runs"]),
        "highest_total_team": str(top["batting_team"]),
    }


@st.cache_data(show_spinner=False)
def matches_per_season() -> pd.DataFrame:
    return get_matches().groupby("season_year").size().rename_axis("season").reset_index(name="matches")


@st.cache_data(show_spinner=False)
def wins_by_team() -> pd.DataFrame:
    return get_matches()["winner"].value_counts().rename_axis("team").reset_index(name="wins")


@st.cache_data(show_spinner=False)
def avg_first_innings_by_season() -> pd.DataFrame:
    d = get_deliveries()
    per_match = (
        d[d["innings"] == 1].groupby(["season_year", "match_id"])["total_runs"].sum().reset_index()
    )
    return per_match.groupby("season_year")["total_runs"].mean().rename_axis("season").reset_index(name="avg_score")


@st.cache_data(show_spinner=False)
def toss_field_pct_by_season() -> pd.DataFrame:
    m = get_matches()
    g = m.assign(field=m["toss_decision"].eq("field")).groupby("season_year")["field"].mean().mul(100)
    return g.rename_axis("season").reset_index(name="field_pct")


# --------------------------------------------------------------------------
# Team deep-dive
# --------------------------------------------------------------------------
def _team_matches(team: str) -> pd.DataFrame:
    m = get_matches()
    return m[(m["team1"] == team) | (m["team2"] == team)].copy()


@st.cache_data(show_spinner=False)
def team_kpis(team: str) -> dict:
    tm = _team_matches(team)
    won_mask = tm["winner"] == team
    played = int(len(tm))
    won = int(won_mask.sum())
    nr = int(tm["winner"].isna().sum())
    lost = played - won - nr

    batted_first = ((tm["toss_winner"] == team) & (tm["toss_decision"] == "bat")) | (
        (tm["toss_winner"] != team) & (tm["toss_decision"] == "field")
    )
    decided = tm["winner"].notna()
    bf_n = int((batted_first & decided).sum())
    ch_n = int((~batted_first & decided).sum())
    bf_w = int((won_mask & batted_first).sum())
    ch_w = int((won_mask & ~batted_first).sum())

    win_runs = pd.to_numeric(tm.loc[won_mask, "winner_runs"], errors="coerce")
    win_wkts = pd.to_numeric(tm.loc[won_mask, "winner_wickets"], errors="coerce")
    return {
        "played": played, "won": won, "lost": lost, "nr": nr,
        "winpct": (won / (won + lost) * 100) if (won + lost) else 0.0,
        "bf_wp": (bf_w / bf_n * 100) if bf_n else 0.0,
        "ch_wp": (ch_w / ch_n * 100) if ch_n else 0.0,
        "big_runs": int(win_runs.max()) if win_runs.notna().any() else 0,
        "big_wkts": int(win_wkts.max()) if win_wkts.notna().any() else 0,
        "first_season": int(tm["season_year"].min()),
        "last_season": int(tm["season_year"].max()),
    }


@st.cache_data(show_spinner=False)
def team_by_season(team: str) -> pd.DataFrame:
    tm = _team_matches(team)
    tm = tm.assign(is_win=(tm["winner"] == team).astype(int))
    g = tm.groupby("season_year").agg(played=("match_id", "size"), won=("is_win", "sum")).reset_index()
    g["win_pct"] = g["won"] / g["played"] * 100
    return g


@st.cache_data(show_spinner=False)
def team_h2h(team: str) -> pd.DataFrame:
    tm = _team_matches(team)
    opponent = np.where(tm["team1"] == team, tm["team2"], tm["team1"])
    tm = tm.assign(opponent=opponent, is_win=(tm["winner"] == team).astype(int))
    g = tm.groupby("opponent").agg(played=("match_id", "size"), won=("is_win", "sum")).reset_index()
    g["win_pct"] = g["won"] / g["played"] * 100
    return g.sort_values("played", ascending=False)


@st.cache_data(show_spinner=False)
def team_top_batters(team: str, n: int = 10) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[d["batting_team"] == team]
    runs = sub.groupby("striker")["runs_off_bat"].sum()
    balls = sub[sub["wides"].isna()].groupby("striker").size()
    df = pd.DataFrame({"runs": runs, "balls": balls}).fillna(0)
    df["sr"] = np.where(df["balls"] > 0, df["runs"] / df["balls"] * 100, 0)
    return df.sort_values("runs", ascending=False).head(n).reset_index()


@st.cache_data(show_spinner=False)
def team_top_bowlers(team: str, n: int = 10) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[(d["bowling_team"] == team) & (d["wicket_type"].isin(BOWLER_WICKETS))]
    return sub.groupby("bowler").size().rename_axis("bowler").reset_index(name="wickets") \
        .sort_values("wickets", ascending=False).head(n)


@st.cache_data(show_spinner=False)
def team_venues(team: str, n: int = 8) -> pd.DataFrame:
    tm = _team_matches(team)
    tm = tm.assign(is_win=(tm["winner"] == team).astype(int))
    g = tm.groupby("venue").agg(played=("match_id", "size"), won=("is_win", "sum")).reset_index()
    g["win_pct"] = g["won"] / g["played"] * 100
    g["venue_short"] = g["venue"].str.split(",").str[0].str.slice(0, 26)
    return g.sort_values("played", ascending=False).head(n)


# --------------------------------------------------------------------------
# Head-to-head
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def head_to_head(a: str, b: str):
    m = get_matches()
    sub = m[((m["team1"] == a) & (m["team2"] == b)) | ((m["team1"] == b) & (m["team2"] == a))]
    summary = {
        "played": int(len(sub)),
        "a_wins": int((sub["winner"] == a).sum()),
        "b_wins": int((sub["winner"] == b).sum()),
        "nr": int(sub["winner"].isna().sum()),
    }
    recent = sub.sort_values("date", ascending=False)[["date", "winner", "venue"]].head(10).copy()
    recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%d %b %Y")
    return summary, recent


# --------------------------------------------------------------------------
# Player analytics (batting)
# --------------------------------------------------------------------------
def _phase_of(ball_series):
    """Map the 'ball' value (over.delivery) to the T20 phase via the over index."""
    over_idx = np.floor(ball_series).astype(int)
    return np.where(over_idx <= 5, "Powerplay", np.where(over_idx <= 14, "Middle", "Death"))


@st.cache_data(show_spinner=False)
def batter_list(min_balls: int = 500) -> list[str]:
    d = get_deliveries()
    faced = d[d["wides"].isna()].groupby("striker").size()
    return sorted(faced[faced >= min_balls].index.tolist())


@st.cache_data(show_spinner=False)
def bowler_list(min_balls: int = 500) -> list[str]:
    d = get_deliveries()
    legal = d[d["wides"].isna() & d["noballs"].isna()].groupby("bowler").size()
    return sorted(legal[legal >= min_balls].index.tolist())


@st.cache_data(show_spinner=False)
def player_career(player: str) -> dict:
    d = get_deliveries()
    bat = d[d["striker"] == player]
    faced = bat[bat["wides"].isna()]          # balls faced exclude wides
    balls = int(len(faced))
    runs = int(bat["runs_off_bat"].sum())
    outs = int((d["player_dismissed"] == player).sum())
    per_inn = bat.groupby("match_id")["runs_off_bat"].sum()
    return {
        "runs": runs, "balls": balls, "innings": int(bat["match_id"].nunique()),
        "sr": runs / balls * 100 if balls else 0.0,
        "avg": runs / outs if outs else float(runs),
        "fours": int((bat["runs_off_bat"] == 4).sum()),
        "sixes": int((bat["runs_off_bat"] == 6).sum()),
        "fifties": int(((per_inn >= 50) & (per_inn < 100)).sum()),
        "hundreds": int((per_inn >= 100).sum()),
        "highest": int(per_inn.max()) if len(per_inn) else 0,
        "dot_pct": float((faced["runs_off_bat"] == 0).mean() * 100) if balls else 0.0,
    }


@st.cache_data(show_spinner=False)
def player_phase(player: str) -> pd.DataFrame:
    d = get_deliveries()
    faced = d[(d["striker"] == player) & (d["wides"].isna())].copy()
    faced["phase"] = _phase_of(faced["ball"])
    g = faced.groupby("phase").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "size")).reset_index()
    g["sr"] = g["runs"] / g["balls"] * 100
    order = {"Powerplay": 0, "Middle": 1, "Death": 2}
    return g.sort_values("phase", key=lambda s: s.map(order)).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def player_over_profile(player: str) -> pd.DataFrame:
    d = get_deliveries()
    faced = d[(d["striker"] == player) & (d["wides"].isna())].copy()
    faced["over"] = np.floor(faced["ball"]).astype(int) + 1
    g = faced.groupby("over").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "size")).reset_index()
    g["sr"] = g["runs"] / g["balls"] * 100
    return g


@st.cache_data(show_spinner=False)
def player_scoring_dna(player: str) -> dict:
    d = get_deliveries()
    faced = d[(d["striker"] == player) & (d["wides"].isna())]
    vc = faced["runs_off_bat"].value_counts()
    return {"Dot": int(vc.get(0, 0)), "1s": int(vc.get(1, 0)), "2s": int(vc.get(2, 0)),
            "3s": int(vc.get(3, 0)), "4s": int(vc.get(4, 0)), "6s": int(vc.get(6, 0))}


@st.cache_data(show_spinner=False)
def player_dismissals(player: str) -> pd.DataFrame:
    d = get_deliveries()
    outs = d[d["player_dismissed"] == player]
    return outs.groupby("wicket_type").size().rename_axis("type").reset_index(name="count") \
        .sort_values("count", ascending=False)


@st.cache_data(show_spinner=False)
def player_nemeses(player: str, n: int = 8) -> pd.DataFrame:
    d = get_deliveries()
    outs = d[(d["player_dismissed"] == player) & (d["wicket_type"].isin(BOWLER_WICKETS))]
    return outs.groupby("bowler").size().rename_axis("bowler").reset_index(name="dismissals") \
        .sort_values("dismissals", ascending=False).head(n)


@st.cache_data(show_spinner=False)
def player_by_season(player: str) -> pd.DataFrame:
    d = get_deliveries()
    bat = d[d["striker"] == player]
    runs = bat.groupby("season_year")["runs_off_bat"].sum().rename("runs")
    balls = bat[bat["wides"].isna()].groupby("season_year").size().rename("balls")
    g = pd.concat([runs, balls], axis=1).fillna(0).reset_index()
    g["sr"] = np.where(g["balls"] > 0, g["runs"] / g["balls"] * 100, 0)
    return g


# --------------------------------------------------------------------------
# Player battles (batter vs bowler)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def battle(batter: str, bowler: str) -> dict:
    d = get_deliveries()
    sub = d[(d["striker"] == batter) & (d["bowler"] == bowler)]
    faced = sub[sub["wides"].isna()]
    balls = int(len(faced))
    runs = int(sub["runs_off_bat"].sum())
    dismissals = int(((sub["player_dismissed"] == batter) & (sub["wicket_type"].isin(BOWLER_WICKETS))).sum())
    return {
        "balls": balls, "runs": runs,
        "sr": runs / balls * 100 if balls else 0.0,
        "dismissals": dismissals,
        "avg": runs / dismissals if dismissals else float(runs),
        "fours": int((sub["runs_off_bat"] == 4).sum()),
        "sixes": int((sub["runs_off_bat"] == 6).sum()),
        "dots": int((faced["runs_off_bat"] == 0).sum()),
    }


@st.cache_data(show_spinner=False)
def battle_outcomes(batter: str, bowler: str) -> dict:
    d = get_deliveries()
    faced = d[(d["striker"] == batter) & (d["bowler"] == bowler) & (d["wides"].isna())]
    vc = faced["runs_off_bat"].value_counts()
    return {"Dot": int(vc.get(0, 0)), "1": int(vc.get(1, 0)), "2": int(vc.get(2, 0)),
            "3": int(vc.get(3, 0)), "4": int(vc.get(4, 0)), "6": int(vc.get(6, 0))}


@st.cache_data(show_spinner=False)
def battle_by_season(batter: str, bowler: str) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[(d["striker"] == batter) & (d["bowler"] == bowler)]
    if sub.empty:
        return pd.DataFrame(columns=["season_year", "runs", "balls"])
    runs = sub.groupby("season_year")["runs_off_bat"].sum().rename("runs")
    balls = sub[sub["wides"].isna()].groupby("season_year").size().rename("balls")
    return pd.concat([runs, balls], axis=1).fillna(0).reset_index()
