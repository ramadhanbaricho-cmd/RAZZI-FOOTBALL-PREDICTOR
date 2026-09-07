import datetime as dt
import pandas as pd
import streamlit as st
from data_source import LEAGUES, DataError, league_data
from model import make_training_df, upcoming_matches, fit_ratings, predict

st.set_page_config(page_title="RAZZI FOOTBALL PREDICTOR", page_icon="⚽", layout="wide")
st.markdown("""<style>
.block-container{max-width:1450px;padding-top:1.2rem}
.hero{padding:22px 24px;border-radius:18px;background:linear-gradient(135deg,#101827,#24344d);color:white;margin-bottom:18px}
.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.4rem 0 0;opacity:.82}
.match{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:16px;margin:10px 0;background:rgba(128,128,128,.04)}
.pick{font-size:1.25rem;font-weight:800}.muted{opacity:.7;font-size:.9rem}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚽ RAZZI FOOTBALL PREDICTOR</h1><p>No API key • free public football data • statistical predictions</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    league = st.selectbox("League", list(LEAGUES))
    date = st.date_input("Match date", dt.date.today())
    minimum = st.slider("Minimum confidence", 0, 100, 50, 5)
    max_matches = st.slider("Maximum matches", 1, 30, 15, 1)
    st.divider()
    st.caption("Data source: OpenFootball public-domain datasets. Updated periodically; this is not a live-score feed.")

try:
    with st.spinner("Loading free football data..."):
        current, previous = league_data(league)
except DataError as e:
    st.error(str(e))
    st.stop()

train = make_training_df([previous, current])
ratings = fit_ratings(train)
fixtures = upcoming_matches(current, date)

st.subheader(f"📅 {league.upper()} — {date:%d %B %Y}")
st.caption(f"Historical matches used for the model: {len(train):,} • Fixtures found: {len(fixtures)}")

if not fixtures:
    st.info("No upcoming fixtures were found for this league/date in the public dataset. Try another date or league.")
    st.stop()

rows = []
for m in fixtures[:max_matches]:
    home = m.get("team1", "Home")
    away = m.get("team2", "Away")
    p = predict(ratings, home, away)
    if not p:
        continue
    rows.append({"home":home, "away":away, **p, "time":m.get("time", "")})

if not rows:
    st.warning("The fixture data loaded, but the model could not calculate predictions for these teams yet.")
    st.stop()

df = pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)

st.markdown("### 🔥 TOP RAZZI PICKS")
top = df[df.confidence >= minimum].head(10)
if top.empty:
    st.info("No match meets the selected confidence threshold. Lower the threshold to see more picks.")
else:
    for _, r in top.iterrows():
        st.markdown(f'''<div class="match">
        <div><b>{r.home}</b> <span class="muted">vs</span> <b>{r.away}</b> <span class="muted">{r.time}</span></div>
        <div class="pick">{r.pick} — {r.confidence:.0f}%</div>
        <div>Score: <b>{r.score}</b> &nbsp; | &nbsp; O2.5: <b>{r.over25:.0f}%</b> &nbsp; | &nbsp; BTTS: <b>{r.btts:.0f}%</b></div>
        <div class="muted">xG {r.xg_home:.2f} — {r.xg_away:.2f}</div>
        </div>''', unsafe_allow_html=True)

st.markdown("### 📋 ALL ANALYZED MATCHES")
display = pd.DataFrame({
    "Time": df.time,
    "Home": df.home,
    "Away": df.away,
    "Prediction": df.pick,
    "Confidence": df.confidence.map(lambda x:f"{x:.0f}%"),
    "Home %": df.home_pct.map(lambda x:f"{x:.0f}%"),
    "Draw %": df.draw_pct.map(lambda x:f"{x:.0f}%"),
    "Away %": df.away_pct.map(lambda x:f"{x:.0f}%"),
    "Score": df.score,
    "O2.5": df.over25.map(lambda x:f"{x:.0f}%"),
    "BTTS": df.btts.map(lambda x:f"{x:.0f}%"),
})
st.dataframe(display, hide_index=True, use_container_width=True)

with st.expander("How RAZZI calculates predictions"):
    st.write("RAZZI estimates expected goals from league averages, team attacking/defensive strength and recent form, then uses a Poisson goal model to calculate 1X2, correct score, Over/Under 2.5 and BTTS probabilities.")
    st.warning("These are statistical estimates, not guaranteed results. Do not treat them as certain betting outcomes.")

st.divider()
st.caption("RAZZI Football Predictor — no API-Football key required. Public data source: OpenFootball football.json.")
