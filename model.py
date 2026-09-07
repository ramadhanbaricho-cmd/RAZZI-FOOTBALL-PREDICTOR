import math
from collections import defaultdict
import numpy as np
import pandas as pd


def _score_pair(m):
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return int(i), int(j)


def _poisson(lam, n):
    return np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(n + 1)])


def make_training_df(payloads):
    rows = []
    for payload in payloads:
        for m in payload.get("matches", []):
            score = m.get("score")
            if isinstance(score, dict):
                ft = score.get("ft")
            else:
                ft = score
            if not isinstance(ft, list) or len(ft) < 2:
                continue
            if ft[0] is None or ft[1] is None:
                continue
            try:
                hg, ag = int(ft[0]), int(ft[1])
            except (TypeError, ValueError):
                continue
            rows.append({
                "date": m.get("date", ""),
                "home_team": m.get("team1", ""),
                "away_team": m.get("team2", ""),
                "home_goals": hg,
                "away_goals": ag,
            })
    return pd.DataFrame(rows)


def upcoming_matches(payload, selected_date):
    target = selected_date.isoformat()
    out = []
    for m in payload.get("matches", []):
        if m.get("date") != target:
            continue
        score = m.get("score")
        # Future fixtures have no full-time score. A list score such as [0,0] is treated as played.
        played = False
        if isinstance(score, dict) and isinstance(score.get("ft"), list):
            played = len(score["ft"]) >= 2
        elif isinstance(score, list):
            played = len(score) >= 2
        if not played:
            out.append(m)
    return out


def fit_ratings(df, recent_n=10, shrink=0.25):
    if df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    league_home = max(df.home_goals.mean(), 0.2)
    league_away = max(df.away_goals.mean(), 0.2)
    teams = sorted(set(df.home_team) | set(df.away_team))
    recent = df.groupby("home_team", group_keys=False).tail(recent_n)
    # Team strengths use all available history, with recent form blended in.
    home_att, home_def, away_att, away_def = {}, {}, {}, {}
    for t in teams:
        h = df[df.home_team == t]
        a = df[df.away_team == t]
        rh = recent[recent.home_team == t]
        ra = recent[recent.away_team == t]
        h_for = h.home_goals.mean() if len(h) else league_home
        h_against = h.away_goals.mean() if len(h) else league_away
        a_for = a.away_goals.mean() if len(a) else league_away
        a_against = a.home_goals.mean() if len(a) else league_home
        form_for = np.mean(list(rh.home_goals) + list(ra.away_goals)) if len(rh) + len(ra) else (h_for + a_for) / 2
        form_against = np.mean(list(rh.away_goals) + list(ra.home_goals)) if len(rh) + len(ra) else (h_against + a_against) / 2
        att_h = (h_for / league_home) * 0.7 + (form_for / ((league_home + league_away) / 2)) * 0.3
        def_h = (h_against / league_away) * 0.7 + (form_against / ((league_home + league_away) / 2)) * 0.3
        att_a = (a_for / league_away) * 0.7 + (form_for / ((league_home + league_away) / 2)) * 0.3
        def_a = (a_against / league_home) * 0.7 + (form_against / ((league_home + league_away) / 2)) * 0.3
        home_att[t] = 1 + (att_h - 1) * (1 - shrink)
        home_def[t] = 1 + (def_h - 1) * (1 - shrink)
        away_att[t] = 1 + (att_a - 1) * (1 - shrink)
        away_def[t] = 1 + (def_a - 1) * (1 - shrink)
    return {
        "league_home": league_home, "league_away": league_away, "teams": set(teams),
        "home_att": home_att, "home_def": home_def,
        "away_att": away_att, "away_def": away_def,
    }


def predict(ratings, home, away, max_goals=7):
    if ratings is None or home not in ratings["teams"] or away not in ratings["teams"]:
        return None
    r = ratings
    lh = r["league_home"] * r["home_att"][home] * r["away_def"][away]
    la = r["league_away"] * r["away_att"][away] * r["home_def"][home]
    lh = float(np.clip(lh, 0.15, 4.5))
    la = float(np.clip(la, 0.15, 4.0))
    ph, pa = _poisson(lh, max_goals), _poisson(la, max_goals)
    matrix = np.outer(ph, pa)
    # Renormalize the truncated tail.
    matrix = matrix / matrix.sum()
    home_p = float(np.tril(matrix, -1).sum())
    draw_p = float(np.trace(matrix))
    away_p = float(np.triu(matrix, 1).sum())
    over25 = float(sum(matrix[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j >= 3))
    btts = float(sum(matrix[i, j] for i in range(1, max_goals + 1) for j in range(1, max_goals + 1)))
    si, sj = _score_pair(matrix)
    probs = {"Home Win": home_p, "Draw": draw_p, "Away Win": away_p}
    pick, conf = max(probs.items(), key=lambda x: x[1])
    return {
        "home_pct": home_p * 100, "draw_pct": draw_p * 100, "away_pct": away_p * 100,
        "pick": pick, "confidence": conf * 100, "score": f"{si}-{sj}",
        "over25": over25 * 100, "under25": (1 - over25) * 100,
        "btts": btts * 100, "xg_home": lh, "xg_away": la,
    }
