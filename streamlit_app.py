import streamlit as st
import requests
from datetime import date, timedelta

st.set_page_config(
    page_title="Razzi Football Predictor",
    page_icon="⚽",
    layout="wide"
)

# Get API key securely from Streamlit Secrets
API_KEY = st.secrets["API_FOOTBALL_KEY"]

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}


def get_fixtures(selected_date):
    url = f"{BASE_URL}/fixtures"
    params = {
        "date": selected_date,
        "timezone": "Africa/Nairobi"
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=20
    )

    return response.json()


def get_prediction(fixture_id):
    url = f"{BASE_URL}/predictions"
    params = {
        "fixture": fixture_id
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=20
    )

    return response.json()


# -------------------------
# APP
# -------------------------

st.title("⚽ RAZZI FOOTBALL PREDICTOR")
st.caption("Real football fixtures and statistical predictions")

st.divider()

selected_date = st.date_input(
    "📅 Select match date",
    value=date.today()
)

if st.button("🔍 Find Matches", type="primary"):

    with st.spinner("Loading football matches..."):

        try:
            data = get_fixtures(str(selected_date))

            if data.get("errors"):
                st.error(f"API Error: {data['errors']}")
            elif not data.get("response"):
                st.warning("No matches found for this date.")
            else:

                matches = data["response"]

                st.success(f"Found {len(matches)} matches.")

                for match in matches:

                    fixture_id = match["fixture"]["id"]

                    home = match["teams"]["home"]["name"]
                    away = match["teams"]["away"]["name"]

                    league = match["league"]["name"]
                    country = match["league"]["country"]

                    kickoff = match["fixture"]["date"]

                    st.markdown("---")

                    st.subheader(
                        f"⚽ {home}  vs  {away}"
                    )

                    st.write(
                        f"🏆 {league} — {country}"
                    )

                    st.write(
                        f"🕐 Kickoff: {kickoff}"
                    )

                    if st.button(
                        f"🔮 Predict {home} vs {away}",
                        key=f"predict_{fixture_id}"
                    ):

                        with st.spinner("Analyzing match..."):

                            prediction_data = get_prediction(
                                fixture_id
                            )

                        if prediction_data.get("errors"):
                            st.error(
                                f"Prediction error: "
                                f"{prediction_data['errors']}"
                            )

                        elif prediction_data.get("response"):

                            prediction = prediction_data[
                                "response"
                            ][0]

                            predictions = prediction[
                                "predictions"
                            ]

                            winner = predictions[
                                "winner"
                            ]

                            percent = predictions[
                                "percent"
                            ]

                            advice = predictions.get(
                                "advice",
                                "No advice available"
                            )

                            under_over = predictions.get(
                                "under_over",
                                "N/A"
                            )

                            goals = predictions.get(
                                "goals",
                                {}
                            )

                            st.markdown(
                                "### 🔮 Prediction"
                            )

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "🏠 Home Win",
                                    percent.get(
                                        "home",
                                        "N/A"
                                    )
                                )

                            with col2:
                                st.metric(
                                    "🤝 Draw",
                                    percent.get(
                                        "draw",
                                        "N/A"
                                    )
                                )

                            with col3:
                                st.metric(
                                    "✈️ Away Win",
                                    percent.get(
                                        "away",
                                        "N/A"
                                    )
                                )

                            st.success(
                                f"🏆 Predicted Winner: "
                                f"{winner.get('name', 'N/A')}"
                            )

                            st.info(
                                f"💡 Advice: {advice}"
                            )

                            st.write(
                                f"⚽ Over/Under: "
                                f"{under_over}"
                            )

                            st.write(
                                f"🎯 Expected Goals: "
                                f"{goals.get('home', 'N/A')} - "
                                f"{goals.get('away', 'N/A')}"
                            )

                        else:
                            st.warning(
                                "No prediction available "
                                "for this match."
                            )

        except Exception as e:
            st.error(
                f"Something went wrong: {e}"
            )
