import datetime as dt
import pandas as pd
import streamlit as st
from live_api import key, fixtures, prediction, APIError

st.set_page_config(page_title="RAZZI FOOTBALL PREDICTOR", page_icon="⚽", layout="wide")
st.markdown("""<style>.block-container{max-width:1450px;padding-top:1.2rem}.hero{padding:22px 24px;border-radius:18px;background:linear-gradient(135deg,#101827,#24344d);color:white;margin-bottom:18px}.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.4rem 0 0;opacity:.82}.match{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:16px;margin:10px 0;background:rgba(128,128,128,.04)}.pick{font-size:1.25rem;font-weight:800}.muted{opacity:.7;font-size:.9rem}</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚽ RAZZI FOOTBALL PREDICTOR</h1><p>Real fixtures • statistical probabilities • ranked predictions</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Mode", ["Live API", "Offline Demo"])
    date = st.date_input("Match date", dt.date.today())
    minimum = st.slider("Minimum confidence", 0, 100, 50, 5)
    max_matches = st.slider("Maximum matches to analyze", 5, 30, 15, 5)
    st.checkbox("Show all analyzed matches", True, key="show_all")
    st.divider()
    st.caption("Predictions are statistical estimates, not guaranteed results.")

if mode == "Offline Demo":
    st.info("Offline mode lets you test the prediction display without an API key.")
    teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Everton", "Tottenham"]
    hcol, acol = st.columns(2)
    h = hcol.selectbox("Home team", teams)
    a = acol.selectbox("Away team", [x for x in teams if x != h])
    if st.button("ANALYZE MATCH", type="primary", use_container_width=True):
        base = {"Arsenal": .58, "Chelsea": .46, "Liverpool": .62, "Manchester City": .68, "Everton": .38, "Tottenham": .44}
        hp, ap = base[h], base[a]
        home_p = max(.10, min(.78, .50 + (hp-ap)*.45))
        away_p = max(.10, min(.65, .28 + (ap-hp)*.35))
        draw_p = max(.08, 1-home_p-away_p)
        total = home_p + draw_p + away_p
        home_p, draw_p, away_p = home_p/total, draw_p/total, away_p/total
        vals = {"Home Win": home_p, "Draw": draw_p, "Away Win": away_p}
        pick, conf = max(vals.items(), key=lambda x: x[1])
        c = st.columns(4)
        c[0].metric("Home Win", f"{home_p:.1%}"); c[1].metric("Draw", f"{draw_p:.1%}"); c[2].metric("Away Win", f"{away_p:.1%}"); c[3].metric("Confidence", f"{conf:.1%}")
        st.success(f"Prediction: **{pick}**")
        st.info("Most likely score: **2 - 1**")
    st.stop()

if not key():
    st.error("API key is not configured.")
    st.code('API_FOOTBALL_KEY = "YOUR_API_KEY_HERE"', language="toml")
    st.stop()

@st.cache_data(ttl=300, show_spinner=False)
def get_fixtures(selected_date):
    return fixtures(selected_date)

@st.cache_data(ttl=1800, show_spinner=False)
def get_prediction(fid):
    return prediction(fid)

try:
    with st.spinner("Loading fixtures..."):
        fs = get_fixtures(date.isoformat())
except APIError as e:
    st.error(f"API error: {e}")
    st.warning("The live prediction display needs an active API-Football account/key. Switch to Offline Demo to test the interface while the account is being restored.")
    st.stop()
except Exception as e:
    st.error(f"Could not load fixtures: {e}")
    st.stop()

if not fs:
    st.info(f"No fixtures found for {date:%d %B %Y}.")
    st.stop()

st.subheader(f"📅 MATCH PREDICTIONS — {date:%d %B %Y}")
st.caption(f"Found {len(fs)} fixtures. Analyzing up to {max_matches} matches to protect API quota.")
rows = []
for f in fs[:max_matches]:
    fx, teams = f.get("fixture", {}), f.get("teams", {})
    fid = fx.get("id")
    home = teams.get("home", {}).get("name", "Home")
    away = teams.get("away", {}).get("name", "Away")
    if not fid: continue
    try:
        p = get_prediction(fid)
    except APIError as e:
        st.warning(f"{home} vs {away}: {e}")
        continue
    except Exception:
        continue
    if not p: continue
    pr = p.get("predictions", {}) or {}
    pct = pr.get("percent", {}) or {}
    def num(v):
        try: return float(str(v).replace("%", ""))
        except Exception: return 0.0
    probs = {"Home Win": num(pct.get("home")), "Draw": num(pct.get("draw")), "Away Win": num(pct.get("away"))}
    pick, conf = max(probs.items(), key=lambda x: x[1])
    rows.append({"home":home,"away":away,"home_pct":probs["Home Win"],"draw_pct":probs["Draw"],"away_pct":probs["Away Win"],"pick":pick,"conf":conf,"score":(pr.get("score",{}) or {}).get("full","—"),"winner":(pr.get("winner",{}) or {}).get("name","—"),"advice":pr.get("advice","—")})

if not rows:
    st.warning("Fixtures loaded, but no prediction records were returned. This can happen when prediction coverage is unavailable or the API account is restricted.")
    st.stop()

df = pd.DataFrame(rows).sort_values("conf", ascending=False)
st.markdown("### 🔥 TOP RAZZI PICKS")
top = df[df.conf >= minimum].head(10)
if top.empty:
    st.info("No match meets the selected confidence threshold.")
else:
    for _, r in top.iterrows():
        st.markdown(f'''<div class="match"><div><b>{r.home}</b> <span class="muted">vs</span> <b>{r.away}</b></div><div class="pick">{r.pick} — {r.conf:.0f}%</div><div>Most likely score: <b>{r.score}</b> &nbsp; | &nbsp; API winner: <b>{r.winner}</b></div><div class="muted">{r.advice}</div></div>''', unsafe_allow_html=True)

if st.session_state.get("show_all", True):
    st.markdown("### 📋 ALL ANALYZED MATCHES")
    display = pd.DataFrame({"Home":df.home,"Away":df.away,"Prediction":df.pick,"Confidence":df.conf.map(lambda x:f"{x:.0f}%"),"Home %":df.home_pct.map(lambda x:f"{x:.0f}%"),"Draw %":df.draw_pct.map(lambda x:f"{x:.0f}%"),"Away %":df.away_pct.map(lambda x:f"{x:.0f}%"),"Score":df.score,"Advice":df.advice})
    st.dataframe(display, hide_index=True, use_container_width=True)

st.divider(); st.caption("RAZZI Football Predictor uses API-Football prediction data. Probabilities are estimates and should not be treated as guaranteed outcomes.")
