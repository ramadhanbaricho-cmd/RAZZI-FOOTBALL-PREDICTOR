import math
from collections import defaultdict
import numpy as np
import pandas as pd


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _elo_table(matches, k=20.0, home_adv=55.0):
    ratings=defaultdict(lambda:1500.0)
    if matches.empty: return ratings
    for _, r in matches.sort_values("date").iterrows():
        h,a=r.team1,r.team2
        if pd.isna(r.home_goals) or pd.isna(r.away_goals): continue
        rh,ra=ratings[h],ratings[a]
        exp=1/(1+10**(-(rh+home_adv-ra)/400))
        hg,ag=int(r.home_goals),int(r.away_goals)
        s=1 if hg>ag else 0.5 if hg==ag else 0
        margin=max(1,abs(hg-ag))
        mult=math.log1p(margin)+1
        ratings[h]=rh+k*mult*(s-exp)
        ratings[a]=ra-k*mult*((1-s)-(1-exp))
    return ratings


def _team_stats(matches, decay=0.985):
    # Recency-weighted attack/defense rates with home/away splits.
    rows=[]
    if matches.empty: return {}, {}, 1.35
    maxd=max(matches.date)
    for _,r in matches.iterrows():
        age=max(0,(maxd-r.date).days)
        w=decay**(age/7.0)
        rows.append((r.team1,r.team2,float(r.home_goals),float(r.away_goals),w))
    teams=sorted(set([x[0] for x in rows]+[x[1] for x in rows]))
    league_home=sum(x[2]*x[4] for x in rows)/max(1e-9,sum(x[4] for x in rows))
    league_away=sum(x[3]*x[4] for x in rows)/max(1e-9,sum(x[4] for x in rows))
    attack={t:{"home":1.0,"away":1.0} for t in teams}; defense={t:{"home":1.0,"away":1.0} for t in teams}
    for t in teams:
        home=[x for x in rows if x[0]==t]; away=[x for x in rows if x[1]==t]
        hw=sum(x[4] for x in home); aw=sum(x[4] for x in away)
        if hw:
            attack[t]["home"]=(sum(x[2]*x[4] for x in home)/hw)/max(league_home,0.6)
            defense[t]["home"]=(sum(x[3]*x[4] for x in home)/hw)/max(league_away,0.5)
        if aw:
            attack[t]["away"]=(sum(x[3]*x[4] for x in away)/aw)/max(league_away,0.5)
            defense[t]["away"]=(sum(x[2]*x[4] for x in away)/aw)/max(league_home,0.6)
        for side in ("home","away"):
            attack[t][side]=float(np.clip(attack[t][side],0.35,2.6))
            defense[t][side]=float(np.clip(defense[t][side],0.35,2.6))
    return attack, defense, max(1.05, min(1.65, league_home+league_away))


def _form(matches, team, n=5):
    m=matches[(matches.team1==team)|(matches.team2==team)].sort_values("date",ascending=False).head(n)
    pts=[]; gf=ga=[]
    gf=[];ga=[]
    for _,r in m.iterrows():
        if r.team1==team: g,con=int(r.home_goals),int(r.away_goals)
        else: g,con=int(r.away_goals),int(r.home_goals)
        gf.append(g);ga.append(con);pts.append(3 if g>con else 1 if g==con else 0)
    return {"points":sum(pts),"gf":sum(gf),"ga":sum(ga),"matches":len(m)}


def predict_match(history, home, away):
    if history.empty:
        return None
    attack, defense, league_goals = _team_stats(history)
    elo=_elo_table(history)
    # Baseline home advantage plus team attack/defence interactions.
    ha=attack.get(home,{"home":1,"away":1})["home"]
    aa=attack.get(away,{"home":1,"away":1})["away"]
    hd=defense.get(home,{"home":1,"away":1})["home"]
    ad=defense.get(away,{"home":1,"away":1})["away"]
    base_home=max(0.35, min(3.4, league_goals*0.53*ha*ad))
    base_away=max(0.25, min(3.0, league_goals*0.47*aa*hd))
    # Elo correction, deliberately modest so goals remain data-driven.
    er=elo.get(home,1500)+45-elo.get(away,1500)
    elo_home=1/(1+10**(-er/400))
    target_home_share=0.50+0.72*(elo_home-0.5)
    current=base_home/(base_home+base_away)
    factor=np.clip(target_home_share/max(current,0.05),0.82,1.18)
    lam_h=base_home*factor; lam_a=base_away*(2-factor)

    maxg=8
    matrix=np.array([[poisson_pmf(i,lam_h)*poisson_pmf(j,lam_a) for j in range(maxg+1)] for i in range(maxg+1)])
    matrix=matrix/matrix.sum()
    home_p=float(np.tril(matrix,-1).sum())
    draw_p=float(np.trace(matrix))
    away_p=float(np.triu(matrix,1).sum())
    over25=float(sum(matrix[i,j] for i in range(maxg+1) for j in range(maxg+1) if i+j>=3))
    btts=float(sum(matrix[i,j] for i in range(1,maxg+1) for j in range(1,maxg+1)))
    score_idx=np.unravel_index(np.argmax(matrix),matrix.shape)
    score=f"{score_idx[0]} - {score_idx[1]}"
    probs={"Home Win":home_p,"Draw":draw_p,"Away Win":away_p}
    pick=max(probs,key=probs.get); raw=probs[pick]
    # Confidence combines margin over second-best with base probability.
    sortedp=sorted(probs.values(),reverse=True)
    margin=sortedp[0]-sortedp[1]
    confidence=float(np.clip(100*(0.65*raw+0.75*margin),30,96))
    fh=_form(history,home); fa=_form(history,away)
    return {
        "home_pct":home_p*100,"draw_pct":draw_p*100,"away_pct":away_p*100,
        "pick":pick,"confidence":confidence,"score":score,
        "xg_home":lam_h,"xg_away":lam_a,"over25":over25*100,"under25":(1-over25)*100,
        "btts":btts*100,"no_btts":(1-btts)*100,"home_form":fh,"away_form":fa,
        "elo_home":elo.get(home,1500),"elo_away":elo.get(away,1500)
    }
