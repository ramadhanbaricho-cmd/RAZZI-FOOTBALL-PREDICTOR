import json
import io
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/football.json/master"
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "English Premier League": {"current": "2026-27/en.1.json", "previous": "2025-26/en.1.json", "code": "E0"},
    "Spanish La Liga": {"current": "2026-27/es.1.json", "previous": "2025-26/es.1.json", "code": "SP1"},
    "Italian Serie A": {"current": "2026-27/it.1.json", "previous": "2025-26/it.1.json", "code": "I1"},
    "German Bundesliga": {"current": "2026-27/de.1.json", "previous": "2025-26/de.1.json", "code": "D1"},
    "French Ligue 1": {"current": "2026-27/fr.1.json", "previous": "2025-26/fr.1.json", "code": "F1"},
}

HEADERS = {"User-Agent": "RAZZI-Football-Predictor/2.0"}


def _get_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()


def _get_csv(url: str):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), on_bad_lines="skip")


def _parse_openfootball(payload: dict) -> pd.DataFrame:
    rows=[]
    for m in payload.get("matches", []):
        d = m.get("date")
        if not d:
            continue
        try:
            md = pd.to_datetime(d).date()
        except Exception:
            continue
        score = m.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else None
        home_goals = away_goals = None
        if isinstance(ft, list) and len(ft) >= 2 and all(x is not None for x in ft[:2]):
            try:
                home_goals, away_goals = int(ft[0]), int(ft[1])
            except Exception:
                pass
        rows.append({
            "date": md,
            "team1": str(m.get("team1", "")).strip(),
            "team2": str(m.get("team2", "")).strip(),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "round": m.get("round", ""),
            "source": "OpenFootball",
        })
    return pd.DataFrame(rows)


def _parse_football_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
        return pd.DataFrame()
    rows=[]
    for _, r in df.iterrows():
        try: d = pd.to_datetime(r["Date"], dayfirst=True).date()
        except Exception: continue
        hg = r.get("FTHG"); ag = r.get("FTAG")
        hg = int(hg) if pd.notna(hg) else None
        ag = int(ag) if pd.notna(ag) else None
        rows.append({"date":d,"team1":str(r["HomeTeam"]).strip(),"team2":str(r["AwayTeam"]).strip(),"home_goals":hg,"away_goals":ag,"round":"","source":"Football-Data"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_league(league_name: str) -> Tuple[pd.DataFrame, str]:
    cfg = LEAGUES[league_name]
    frames=[]
    errors=[]
    for season_key in ("previous", "current"):
        path=cfg[season_key]
        try:
            frames.append(_parse_openfootball(_get_json(f"{OPENFOOTBALL}/{path}")))
        except Exception as e:
            errors.append(f"OpenFootball {season_key}: {e}")
    data = pd.concat([x for x in frames if not x.empty], ignore_index=True) if frames else pd.DataFrame()
    # Football-Data fallback/current supplement. Its 2026/27 files can contain fixtures not yet present in OpenFootball.
    try:
        fb = _parse_football_data(_get_csv(FOOTBALL_DATA.format(season="2627", code=cfg["code"])))
        if not fb.empty:
            data = pd.concat([data, fb], ignore_index=True)
    except Exception as e:
        errors.append(f"Football-Data fallback: {e}")
    if data.empty:
        return data, " | ".join(errors)
    data["team1"] = data.team1.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data["team2"] = data.team2.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data.drop_duplicates(subset=["date","team1","team2"], keep="first").sort_values("date").reset_index(drop=True)
    return data, ""


def split_matches(data: pd.DataFrame, target_date: date):
    if data.empty:
        return data.copy(), data.copy()
    completed = data[data.home_goals.notna() & data.away_goals.notna() & (data.date < target_date)].copy()
    upcoming = data[data.home_goals.isna() | data.away_goals.isna()].copy()
    upcoming = upcoming[upcoming.date >= target_date].sort_values(["date","team1","team2"])
    return completed, upcoming


def next_fixtures(data: pd.DataFrame, start_date: date, limit: int = 15):
    _, up = split_matches(data, start_date)
    return up.head(limit).copy()


def available_dates(data: pd.DataFrame, start_date: date, n: int = 8):
    if data.empty: return []
    _, up = split_matches(data, start_date)
    return sorted(up.date.unique())[:n]
