import math
from collections import defaultdict
import numpy as np
import pandas as pd


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _elo_table(matches, k=20.0, home_adv=55.0):
    ratings = defaultdict(lambda: 1500.0)
    if matches.empty:
        return ratings
    for _, r in matches.sort_values("date").iterrows():
        h, a = r.team1, r.team2
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue
        rh, ra = ratings[h], ratings[a]
        exp = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        hg, ag = int(r.home_goals), int(r.away_goals)
        s = 1 if hg > ag else 0.5 if hg == ag else 0
        margin = max(1, abs(hg - ag))
        mult = math.log1p(margin) + 1
        ratings[h] = rh + k * mult * (s - exp)
        ratings[a] = ra - k * mult * ((1 - s) - (1 - exp))
    return ratings


def _team_stats(matches, decay=0.985):
    rows = []
    if matches.empty:
        return {}, {}, 1.35
    maxd = max(matches.date)
    for _, r in matches.iterrows():
        age = max(0, (maxd - r.date).days)
        w = decay ** (age / 7.0)
        rows.append((r.team1, r.team2, float(r.home_goals), float(r.away_goals), w))
    teams = sorted(set([x[0] for x in rows] + [x[1] for x in rows]))
    league_home = sum(x[2] * x[4] for x in rows) / max(1e-9, sum(x[4] for x in rows))
    league_away = sum(x[3] * x[4] for x in rows) / max(1e-9, sum(x[4] for x in rows))
    attack = {t: {"home": 1.0, "away": 1.0} for t in teams}
    defense = {t: {"home": 1.0, "away": 1.0} for t in teams}
    for t in teams:
        home = [x for x in rows if x[0] == t]
        away = [x for x in rows if x[1] == t]
        hw = sum(x[4] for x in home)
        aw = sum(x[4] for x in away)
        if hw:
            attack[t]["home"] = (sum(x[2] * x[4] for x in home) / hw) / max(league_home, 0.6)
            defense[t]["home"] = (sum(x[3] * x[4] for x in home) / hw) / max(league_away, 0.5)
        if aw:
            attack[t]["away"] = (sum(x[3] * x[4] for x in away) / aw) / max(league_away, 0.5)
            defense[t]["away"] = (sum(x[2] * x[4] for x in away) / aw) / max(league_home, 0.6)
        for side in ("home", "away"):
            attack[t][side] = float(np.clip(attack[t][side], 0.35, 2.6))
            defense[t][side] = float(np.clip(defense[t][side], 0.35, 2.6))
    return attack, defense, max(1.05, min(1.65, league_home + league_away))


def _form(matches, team, n=5):
    m = matches[(matches.team1 == team) | (matches.team2 == team)].sort_values("date", ascending=False).head(n)
    pts, gf, ga = [], [], []
    for _, r in m.iterrows():
        if r.team1 == team:
            g, con = int(r.home_goals), int(r.away_goals)
        else:
            g, con = int(r.away_goals), int(r.home_goals)
        gf.append(g)
        ga.append(con)
        pts.append(3 if g > con else 1 if g == con else 0)
    return {"points": sum(pts), "gf": sum(gf), "ga": sum(ga), "matches": len(m)}


def _confidence(raw, margin, sample_size):
    """A transparent heuristic confidence score, not a probability guarantee."""
    sample_factor = min(1.0, sample_size / 180.0)
    score = 100 * (0.72 * raw + 0.90 * margin)
    score = score * (0.88 + 0.12 * sample_factor)
    return float(np.clip(score, 38, 92))


def predict_match(history, home, away):
    if history.empty:
        return None
    attack, defense, league_goals = _team_stats(history)
    elo = _elo_table(history)

    ha = attack.get(home, {"home": 1, "away": 1})["home"]
    aa = attack.get(away, {"home": 1, "away": 1})["away"]
    hd = defense.get(home, {"home": 1, "away": 1})["home"]
    ad = defense.get(away, {"home": 1, "away": 1})["away"]

    base_home = max(0.35, min(3.4, league_goals * 0.53 * ha * ad))
    base_away = max(0.25, min(3.0, league_goals * 0.47 * aa * hd))

    er = elo.get(home, 1500) + 45 - elo.get(away, 1500)
    elo_home = 1 / (1 + 10 ** (-er / 400))
    target_home_share = 0.50 + 0.72 * (elo_home - 0.5)
    current = base_home / max(base_home + base_away, 0.05)
    factor = np.clip(target_home_share / max(current, 0.05), 0.82, 1.18)
    lam_h = base_home * factor
    lam_a = base_away * (2 - factor)

    maxg = 8
    matrix = np.array([[poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
                         for j in range(maxg + 1)] for i in range(maxg + 1)])
    matrix /= matrix.sum()

    home_p = float(np.tril(matrix, -1).sum())
    draw_p = float(np.trace(matrix))
    away_p = float(np.triu(matrix, 1).sum())
    over25 = float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j >= 3))
    btts = float(sum(matrix[i, j] for i in range(1, maxg + 1) for j in range(1, maxg + 1)))

    score_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    score = f"{score_idx[0]} - {score_idx[1]}"
    probs = {"Home Win": home_p, "Draw": draw_p, "Away Win": away_p}
    pick = max(probs, key=probs.get)
    raw = probs[pick]
    sortedp = sorted(probs.values(), reverse=True)
    margin = sortedp[0] - sortedp[1]
    confidence = _confidence(raw, margin, len(history))

    fh = _form(history, home)
    fa = _form(history, away)

    over15 = float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j >= 2))
    markets = {
        "Home Win": home_p, "Draw": draw_p, "Away Win": away_p,
        "Double Chance 1X": home_p + draw_p,
        "Double Chance X2": draw_p + away_p,
        "Double Chance 12": home_p + away_p,
        "Over 1.5 Goals": over15, "Under 1.5 Goals": 1 - over15,
        "Over 2.5 Goals": over25, "Under 2.5 Goals": 1 - over25,
        "BTTS": btts, "No BTTS": 1 - btts,
    }
    # Only recommend a secondary market when its model probability is clearly stronger.
    secondary = max((k for k in markets if k not in probs), key=markets.get)
    secondary_prob = markets[secondary]
    if secondary_prob >= 0.60 and secondary_prob > raw + 0.04:
        best_market, best_market_pct = secondary, secondary_prob * 100
    else:
        best_market, best_market_pct = pick, raw * 100

    if confidence >= 65:
        tier = "STRONG"
    elif confidence >= 58:
        tier = "GOOD"
    elif confidence >= 50:
        tier = "MODERATE"
    else:
        tier = "WATCH"

    return {
        "home_pct": home_p * 100, "draw_pct": draw_p * 100, "away_pct": away_p * 100,
        "pick": pick, "confidence": confidence, "tier": tier,
        "score": score, "xg_home": lam_h, "xg_away": lam_a,
        "over25": over25 * 100, "under25": (1 - over25) * 100,
        "btts": btts * 100, "no_btts": (1 - btts) * 100,
        "over15": float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j >= 2) * 100),
        "under15": float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j < 2) * 100),
        "best_market": best_market, "best_market_pct": best_market_pct,
        "home_form": fh, "away_form": fa,
        "elo_home": elo.get(home, 1500), "elo_away": elo.get(away, 1500),
        "history_count": len(history),
    }


def _market_probabilities(matrix):
    maxg = matrix.shape[0] - 1
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    over15 = float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j >= 2))
    over25 = float(sum(matrix[i, j] for i in range(maxg + 1) for j in range(maxg + 1) if i + j >= 3))
    btts = float(matrix[1:, 1:].sum())
    return {
        "Home Win": home, "Draw": draw, "Away Win": away,
        "Double Chance 1X": home + draw,
        "Double Chance X2": draw + away,
        "Double Chance 12": home + away,
        "Over 1.5 Goals": over15,
        "Under 1.5 Goals": 1 - over15,
        "Over 2.5 Goals": over25,
        "Under 2.5 Goals": 1 - over25,
        "BTTS": btts,
        "No BTTS": 1 - btts,
    }


def backtest(history, max_matches=80, min_history=80):
    """Rolling historical evaluation. Predictions only use matches before each test match."""
    completed = history[history.home_goals.notna() & history.away_goals.notna()].sort_values("date").reset_index(drop=True)
    if len(completed) <= min_history:
        return {"tested": 0, "reason": f"Need more than {min_history} completed matches."}
    test = completed.tail(min(max_matches, len(completed) - min_history))
    records = []
    for idx in test.index:
        row = completed.loc[idx]
        train = completed.iloc[:idx]
        pred = predict_match(train, row.team1, row.team2)
        if not pred:
            continue
        hg, ag = int(row.home_goals), int(row.away_goals)
        actual_1x2 = "Home Win" if hg > ag else "Draw" if hg == ag else "Away Win"
        records.append({
            "pred": pred["pick"], "actual": actual_1x2,
            "over15_pred": pred["over15"] >= 50, "over15_actual": hg + ag >= 2,
            "over25_pred": pred["over25"] >= 50, "over25_actual": hg + ag >= 3,
            "btts_pred": pred["btts"] >= 50, "btts_actual": hg > 0 and ag > 0,
            "score_pred": pred["score"], "score_actual": f"{hg} - {ag}",
        })
    if not records:
        return {"tested": 0, "reason": "No historical predictions could be calculated."}
    r = pd.DataFrame(records)
    return {
        "tested": len(r),
        "one_x_two": float((r.pred == r.actual).mean() * 100),
        "over15": float((r.over15_pred == r.over15_actual).mean() * 100),
        "over25": float((r.over25_pred == r.over25_actual).mean() * 100),
        "btts": float((r.btts_pred == r.btts_actual).mean() * 100),
        "correct_score": float((r.score_pred == r.score_actual).mean() * 100),
    }
