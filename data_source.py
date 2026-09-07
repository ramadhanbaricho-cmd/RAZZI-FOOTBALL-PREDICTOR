import requests
import streamlit as st

BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES = {
    "English Premier League": {"code": "en.1", "season": "2026-27", "previous": "2025-26"},
    "La Liga": {"code": "es.1", "season": "2026-27", "previous": "2025-26"},
    "Serie A": {"code": "it.1", "season": "2026-27", "previous": "2025-26"},
    "Bundesliga": {"code": "de.1", "season": "2026-27", "previous": "2025-26"},
    "Ligue 1": {"code": "fr.1", "season": "2026-27", "previous": "2025-26"},
}

class DataError(RuntimeError):
    pass

@st.cache_data(ttl=3600, show_spinner=False)
def load_json(season, code):
    url = f"{BASE}/{season}/{code}.json"
    try:
        r = requests.get(url, timeout=20)
        if not r.ok:
            raise DataError(f"Football data returned HTTP {r.status_code}: {url}")
        return r.json()
    except requests.RequestException as e:
        raise DataError(f"Could not download football data: {e}") from e
    except ValueError as e:
        raise DataError("Football data was not valid JSON.") from e


def league_data(league_name):
    cfg = LEAGUES[league_name]
    current = load_json(cfg["season"], cfg["code"])
    previous = load_json(cfg["previous"], cfg["code"])
    return current, previous
