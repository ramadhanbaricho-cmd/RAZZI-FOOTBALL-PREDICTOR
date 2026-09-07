import datetime as dt
import pandas as pd
import streamlit as st
from data_source import LEAGUES, load_league, next_fixtures, available_dates
from model import predict_match

st.set_page_config(page_title="RAZZI FOOTBALL PREDICTOR", page_icon="⚽", layout="wide")
st.markdown("""<style>
.block-container{max-width:1450px;padding-top:1rem}.hero{padding:24px;border-radius:20px;background:linear-gradient(135deg,#07111f,#17365d);color:#fff;margin-bottom:18px}.hero h1{margin:0;font-size:2.35rem}.hero p{margin:.5rem 0 0;opacity:.82}.card{border:1px solid rgba(120,140,170,.28);border-radius:18px;padding:17px;margin:10px 0;background:rgba(120,140,170,.06)}.pick{font-size:1.35rem;font-weight:850}.muted{opacity:.68;font-size:.88rem}.pill{display:inline-block;padding:4px 9px;border-radius:20px;background:#16395c;color:white;margin-right:5px;font-size:.8rem}.small{font-size:.86rem}.metricbox{padding:10px;border-radius:12px;background:rgba(120,140,170,.08);text-align:center}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚽ RAZZI FOOTBALL PREDICTOR</h1><p>No API key • public football data • Elo + Poisson prediction engine • ranked next matches</p></div>', unsafe_allow_html=True)

if "selected_target" not in st.session_state: st.session_state.selected_target = dt.date.today()
if "prediction_log" not in st.session_state: st.session_state.prediction_log=[]

with st.sidebar:
    st.header("⚙️ RAZZI CONTROLS")
    league=st.selectbox("League", list(LEAGUES.keys()))
    date=st.date_input("Match date", st.session_state.selected_target)
    min_conf=st.slider("Minimum confidence",30,90,50,5)
    max_matches=st.slider("Maximum matches",5,30,15,5)
    if st.button("🔄 Refresh public data", use_container_width=True):
        load_league.clear(); st.rerun()
    st.divider()
    st.caption("No API key or paid football API is used. Data availability depends on the public datasets.")

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(league_name):
    return load_league(league_name)

with st.spinner("Loading public football data..."):
    data, source_error=get_data(league)

if data.empty:
    st.error("Could not load the public football dataset right now.")
    if source_error: st.caption(source_error)
    st.stop()

completed_all=data[data.home_goals.notna() & data.away_goals.notna()]
upcoming_all=data[(data.home_goals.isna()) | (data.away_goals.isna())]

c1,c2,c3,c4=st.columns(4)
c1.metric("Historical matches", f"{len(completed_all):,}")
c2.metric("Upcoming fixtures", f"{len(upcoming_all):,}")
c3.metric("Teams", f"{len(set(data.team1)|set(data.team2))}")
c4.metric("Data source", "Public / no key")

st.subheader(f"📅 {league} — {date:%d %B %Y}")
today_fixtures=next_fixtures(data,date,max_matches)
today_fixtures=today_fixtures[today_fixtures.date==date]

# Prominent next-match action requested by user.
col1,col2=st.columns([2,1])
with col1:
    if st.button("🔮 PREDICT NEXT MATCHES", type="primary", use_container_width=True):
        st.session_state.selected_target=date
        st.session_state.show_next=True
with col2:
    if st.button("📆 FIND NEXT AVAILABLE DATE", use_container_width=True):
        st.session_state.show_next=True
        st.session_state.force_next=True

show_next=st.session_state.get("show_next",False) or not bool(today_fixtures)
if st.session_state.get("force_next",False): show_next=True

if not today_fixtures:
    dates=available_dates(data,date,10)
    if dates:
        nxt=dates[0]
        st.info(f"No fixtures found on {date:%d %B %Y}. The next public-data fixtures are on **{pd.Timestamp(nxt):%d %B %Y}**.")
        if st.button(f"➡️ USE {pd.Timestamp(nxt):%d %B %Y}", use_container_width=True):
            st.session_state.selected_target=pd.Timestamp(nxt).date(); st.session_state.force_next=False; st.rerun()
    else:
        st.warning("No future fixtures are currently available in the public dataset. Try Refresh public data later.")

if show_next:
    fixtures=next_fixtures(data,date,max_matches)
else:
    fixtures=today_fixtures

if fixtures.empty:
    st.stop()

# If no date matches, automatically show the next available block.
if not today_fixtures:
    first_date=fixtures.iloc[0].date
    fixtures=fixtures[fixtures.date==first_date]
    active_date=first_date
else:
    active_date=date

st.caption(f"Showing {len(fixtures)} fixture(s) for {pd.Timestamp(active_date):%d %B %Y}. Model uses completed matches before each fixture date.")

rows=[]
for _,fx in fixtures.iterrows():
    hist=data[(data.home_goals.notna()) & (data.away_goals.notna()) & (data.date < fx.date)]
    pred=predict_match(hist,fx.team1,fx.team2)
    if pred:
        rows.append({"date":fx.date,"home":fx.team1,"away":fx.team2,**pred})

if not rows:
    st.warning("Fixtures exist, but there is not enough historical data to calculate predictions yet.")
    st.stop()

df=pd.DataFrame(rows).sort_values("confidence",ascending=False).reset_index(drop=True)

st.markdown("### 🔥 TOP RAZZI PICKS")
top=df[df.confidence>=min_conf].head(10)
if top.empty: st.info("No fixture meets your confidence threshold. Lower the Minimum confidence slider.")
for i,r in top.iterrows():
    st.markdown(f"""<div class="card"><div><span class="pill">#{i+1} TOP PICK</span> <span class="muted">{pd.Timestamp(r.date):%d %b %Y}</span></div><h3 style="margin:.45rem 0">{r.home} <span class="muted">vs</span> {r.away}</h3><div class="pick">{r.pick} — {r.confidence:.0f}% confidence</div><div class="small">Score <b>{r.score}</b> &nbsp; • &nbsp; xG <b>{r.xg_home:.2f} - {r.xg_away:.2f}</b> &nbsp; • &nbsp; Over 2.5 <b>{r.over25:.0f}%</b> &nbsp; • &nbsp; BTTS <b>{r.btts:.0f}%</b></div></div>""",unsafe_allow_html=True)

st.markdown("### 📊 MATCH-BY-MATCH ANALYSIS")
for _,r in df.iterrows():
    with st.expander(f"{r.home} vs {r.away} — {r.pick} ({r.confidence:.0f}%)"):
        a,b,c,d=st.columns(4)
        a.metric("Home",f"{r.home_pct:.0f}%")
        b.metric("Draw",f"{r.draw_pct:.0f}%")
        c.metric("Away",f"{r.away_pct:.0f}%")
        d.metric("Confidence",f"{r.confidence:.0f}%")
        e,f,g,h=st.columns(4)
        e.metric("Most likely score",r.score)
        f.metric("Over 2.5",f"{r.over25:.0f}%")
        g.metric("BTTS",f"{r.btts:.0f}%")
        h.metric("Expected goals",f"{r.xg_home+r.xg_away:.2f}")
        st.write(f"**Recent form:** {r.home}: {r.home_form['points']}/{r.home_form['matches']*3 if r.home_form['matches'] else 0} pts • {r.home_form['gf']} GF / {r.home_form['ga']} GA | {r.away}: {r.away_form['points']}/{r.away_form['matches']*3 if r.away_form['matches'] else 0} pts • {r.away_form['gf']} GF / {r.away_form['ga']} GA")
        if st.button("Save prediction", key=f"save_{r.date}_{r.home}_{r.away}"):
            st.session_state.prediction_log.append(dict(r))
            st.success("Prediction saved for this session.")

st.markdown("### 📋 RAZZI PREDICTION TABLE")
table=df[["date","home","away","pick","confidence","home_pct","draw_pct","away_pct","score","over25","btts","xg_home","xg_away"]].copy()
table.columns=["Date","Home","Away","Prediction","Confidence","Home %","Draw %","Away %","Score","Over 2.5 %","BTTS %","Home xG","Away xG"]
table["Date"]=pd.to_datetime(table.Date).dt.strftime("%d %b")
for col in ["Confidence","Home %","Draw %","Away %","Over 2.5 %","BTTS %"]: table[col]=table[col].map(lambda x:f"{x:.0f}%")
for col in ["Home xG","Away xG"]: table[col]=table[col].map(lambda x:f"{x:.2f}")
st.dataframe(table,hide_index=True,use_container_width=True)

if st.session_state.prediction_log:
    st.markdown("### 🧾 SAVED PREDICTIONS (THIS SESSION)")
    saved=pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(saved[["date","home","away","pick","confidence","score"]],hide_index=True,use_container_width=True)

st.divider()
st.caption("RAZZI uses public OpenFootball and Football-Data datasets. Predictions are statistical estimates, not guaranteed outcomes. Public data is updated periodically, so it is not a live-score feed.")
