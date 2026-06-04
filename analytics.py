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


# ==========================================================================
# Player photos -- curated Wikipedia map + on-demand fetch (avatar fallback)
# ==========================================================================
import json as _json
import ssl as _ssl
import urllib.parse as _uparse
import urllib.request as _urequest

# Cricsheet name -> exact Wikipedia title (so we never grab the wrong page).
PLAYER_WIKI = {
    "V Kohli": "Virat Kohli", "RG Sharma": "Rohit Sharma", "MS Dhoni": "MS Dhoni",
    "AB de Villiers": "AB de Villiers", "CH Gayle": "Chris Gayle",
    "DA Warner": "David Warner (cricketer)", "S Dhawan": "Shikhar Dhawan",
    "SK Raina": "Suresh Raina", "JJ Bumrah": "Jasprit Bumrah", "YS Chahal": "Yuzvendra Chahal",
    "Rashid Khan": "Rashid Khan (cricketer)", "AD Russell": "Andre Russell",
    "KA Pollard": "Kieron Pollard", "SL Malinga": "Lasith Malinga", "B Kumar": "Bhuvneshwar Kumar",
    "HH Pandya": "Hardik Pandya", "RA Jadeja": "Ravindra Jadeja", "R Ashwin": "Ravichandran Ashwin",
    "KL Rahul": "KL Rahul", "DJ Bravo": "Dwayne Bravo", "AT Rayudu": "Ambati Rayudu",
    "SR Watson": "Shane Watson", "G Gambhir": "Gautam Gambhir", "V Sehwag": "Virender Sehwag",
    "SR Tendulkar": "Sachin Tendulkar", "JC Buttler": "Jos Buttler", "F du Plessis": "Faf du Plessis",
    "Q de Kock": "Quinton de Kock", "GJ Maxwell": "Glenn Maxwell", "MM Ali": "Moeen Ali",
    "SP Narine": "Sunil Narine", "Mohammed Shami": "Mohammed Shami", "Mohammed Siraj": "Mohammed Siraj",
    "K Rabada": "Kagiso Rabada", "TA Boult": "Trent Boult", "PJ Cummins": "Pat Cummins",
    "MA Starc": "Mitchell Starc", "A Nehra": "Ashish Nehra", "Harbhajan Singh": "Harbhajan Singh",
    "DW Steyn": "Dale Steyn", "RR Pant": "Rishabh Pant", "Shubman Gill": "Shubman Gill",
    "SV Samson": "Sanju Samson", "SA Yadav": "Suryakumar Yadav", "KD Karthik": "Dinesh Karthik",
    "RV Uthappa": "Robin Uthappa", "Yuvraj Singh": "Yuvraj Singh", "BB McCullum": "Brendon McCullum",
    "AC Gilchrist": "Adam Gilchrist", "EJG Morgan": "Eoin Morgan", "Z Khan": "Zaheer Khan",
    "MK Pandey": "Manish Pandey", "WP Saha": "Wriddhiman Saha", "PP Chawla": "Piyush Chawla",
    "JD Unadkat": "Jaydev Unadkat", "N Rana": "Nitish Rana", "Sandeep Sharma": "Sandeep Sharma (cricketer)",
}


@st.cache_data(show_spinner=False)
def player_photo_url(name: str) -> str:
    """Best-effort headshot URL from Wikipedia for known stars, else "" (the UI
    then shows a clean avatar). Cached, one network call per player, never raises.
    """
    title = PLAYER_WIKI.get(name)
    if not title:
        return ""
    try:
        import certifi
        ctx = _ssl.create_default_context(cafile=certifi.where())
        q = _uparse.quote(title)
        url = (f"https://en.wikipedia.org/w/api.php?action=query&titles={q}"
               f"&prop=pageimages&piprop=thumbnail&pithumbsize=240&format=json&redirects=1")
        req = _urequest.Request(url, headers={"User-Agent": "PitchIQ/1.0 (portfolio project)"})
        with _urequest.urlopen(req, timeout=5, context=ctx) as resp:
            data = _json.load(resp)
        for page in data.get("query", {}).get("pages", {}).values():
            src = (page.get("thumbnail") or {}).get("source")
            if src:
                return src
    except Exception:
        pass
    return ""


# ==========================================================================
# Bowler analytics
# ==========================================================================
@st.cache_data(show_spinner=False)
def bowler_career(bowler: str) -> dict:
    d = get_deliveries()
    sub = d[d["bowler"] == bowler]
    legal = sub[sub["wides"].isna() & sub["noballs"].isna()]
    balls = int(len(legal))
    conceded = int((sub["runs_off_bat"] + sub["wides"].fillna(0) + sub["noballs"].fillna(0)).sum())
    wickets = int((sub["player_dismissed"].notna() & sub["wicket_type"].isin(BOWLER_WICKETS)).sum())
    dots = int(((legal["runs_off_bat"] == 0) & (legal["extras"] == 0)).sum())
    return {
        "wickets": wickets, "balls": balls, "conceded": conceded,
        "economy": conceded / (balls / 6) if balls else 0.0,
        "average": conceded / wickets if wickets else float(conceded),
        "sr": balls / wickets if wickets else 0.0,
        "dot_pct": dots / balls * 100 if balls else 0.0,
    }


@st.cache_data(show_spinner=False)
def bowler_phase(bowler: str) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[d["bowler"] == bowler].copy()
    sub["phase"] = _phase_of(sub["ball"])
    sub["conc"] = sub["runs_off_bat"] + sub["wides"].fillna(0) + sub["noballs"].fillna(0)
    sub["legal"] = (sub["wides"].isna() & sub["noballs"].isna()).astype(int)
    g = sub.groupby("phase").agg(conc=("conc", "sum"), balls=("legal", "sum")).reset_index()
    g["econ"] = g["conc"] / (g["balls"] / 6)
    order = {"Powerplay": 0, "Middle": 1, "Death": 2}
    return g.sort_values("phase", key=lambda s: s.map(order)).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def bowler_by_season(bowler: str) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[(d["bowler"] == bowler) & d["player_dismissed"].notna() & d["wicket_type"].isin(BOWLER_WICKETS)]
    return sub.groupby("season_year").size().rename_axis("season_year").reset_index(name="wickets")


@st.cache_data(show_spinner=False)
def bowler_victims(bowler: str, n: int = 8) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[(d["bowler"] == bowler) & d["player_dismissed"].notna() & d["wicket_type"].isin(BOWLER_WICKETS)]
    return sub.groupby("player_dismissed").size().rename_axis("victim").reset_index(name="dismissals") \
        .sort_values("dismissals", ascending=False).head(n)


@st.cache_data(show_spinner=False)
def bowler_dismissal_types(bowler: str) -> pd.DataFrame:
    d = get_deliveries()
    sub = d[(d["bowler"] == bowler) & d["player_dismissed"].notna() & d["wicket_type"].isin(BOWLER_WICKETS)]
    return sub.groupby("wicket_type").size().rename_axis("type").reset_index(name="count") \
        .sort_values("count", ascending=False)


# ==========================================================================
# Venue / pitch report
# ==========================================================================
@st.cache_data(show_spinner=False)
def venue_list(min_matches: int = 5) -> list[str]:
    vc = get_matches()["venue"].value_counts()
    return sorted(vc[vc >= min_matches].index.tolist())


@st.cache_data(show_spinner=False)
def venue_report(venue: str) -> dict:
    m = get_matches()
    mv = m[m["venue"] == venue]
    d = get_deliveries()
    first = d[(d["venue"] == venue) & (d["innings"] == 1)].groupby("match_id")["total_runs"].sum()
    bat_wins = int(pd.to_numeric(mv["winner_runs"], errors="coerce").notna().sum())
    chase_wins = int(pd.to_numeric(mv["winner_wickets"], errors="coerce").notna().sum())
    decided = bat_wins + chase_wins
    return {
        "matches": int(len(mv)),
        "avg_first": float(first.mean()) if len(first) else 0.0,
        "highest": int(first.max()) if len(first) else 0,
        "pct_field": float(mv["toss_decision"].eq("field").mean() * 100),
        "bat_win": bat_wins / decided * 100 if decided else 0.0,
        "chase_win": chase_wins / decided * 100 if decided else 0.0,
    }


@st.cache_data(show_spinner=False)
def venue_avg_by_season(venue: str) -> pd.DataFrame:
    d = get_deliveries()
    first = d[(d["venue"] == venue) & (d["innings"] == 1)].groupby(["season_year", "match_id"])["total_runs"].sum().reset_index()
    return first.groupby("season_year")["total_runs"].mean().rename_axis("season_year").reset_index(name="avg_score")


# ==========================================================================
# Season explorer
# ==========================================================================
@st.cache_data(show_spinner=False)
def season_list() -> list[int]:
    return sorted(int(s) for s in get_matches()["season_year"].dropna().unique())


@st.cache_data(show_spinner=False)
def season_standings(season: int) -> pd.DataFrame:
    m = get_matches()
    ms = m[m["season_year"] == season]
    return ms["winner"].value_counts().rename_axis("team").reset_index(name="wins")


@st.cache_data(show_spinner=False)
def season_orange_cap(season: int, n: int = 8) -> pd.DataFrame:
    d = get_deliveries()
    ds = d[d["season_year"] == season]
    return ds.groupby("striker")["runs_off_bat"].sum().nlargest(n).rename_axis("striker").reset_index(name="runs")


@st.cache_data(show_spinner=False)
def season_purple_cap(season: int, n: int = 8) -> pd.DataFrame:
    d = get_deliveries()
    ds = d[(d["season_year"] == season) & d["wicket_type"].isin(BOWLER_WICKETS)]
    return ds.groupby("bowler").size().nlargest(n).rename_axis("bowler").reset_index(name="wickets")


@st.cache_data(show_spinner=False)
def season_summary(season: int) -> dict:
    m = get_matches()
    ms = m[m["season_year"] == season].sort_values("date")
    champion = ms["winner"].iloc[-1] if len(ms) and pd.notna(ms["winner"].iloc[-1]) else "—"
    oc, pc = season_orange_cap(season, 1), season_purple_cap(season, 1)
    return {
        "matches": int(len(ms)), "champion": champion,
        "orange": oc["striker"].iloc[0] if len(oc) else "—",
        "orange_runs": int(oc["runs"].iloc[0]) if len(oc) else 0,
        "purple": pc["bowler"].iloc[0] if len(pc) else "—",
        "purple_wkts": int(pc["wickets"].iloc[0]) if len(pc) else 0,
    }


# ==========================================================================
# Player comparison (radar)
# ==========================================================================
def _radar_vec(player: str) -> dict:
    c = player_career(player)
    ph = player_phase(player)

    def phase_sr(name):
        row = ph[ph["phase"] == name]
        return float(row["sr"].iloc[0]) if len(row) else 0.0

    boundary = (c["fours"] + c["sixes"]) / c["balls"] * 100 if c["balls"] else 0.0
    return {
        "Strike rate": c["sr"], "Average": c["avg"], "Boundary %": boundary,
        "Powerplay SR": phase_sr("Powerplay"), "Death SR": phase_sr("Death"),
    }


@st.cache_data(show_spinner=False)
def compare_metrics(a: str, b: str):
    va, vb = _radar_vec(a), _radar_vec(b)
    caps = {"Strike rate": 200, "Average": 55, "Boundary %": 25, "Powerplay SR": 200, "Death SR": 260}
    labels = list(caps)
    na = [min(va[l] / caps[l] * 100, 100) for l in labels]
    nb = [min(vb[l] / caps[l] * 100, 100) for l in labels]
    return labels, na, nb, va, vb
