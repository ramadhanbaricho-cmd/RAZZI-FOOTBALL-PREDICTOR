import streamlit as st

st.set_page_config(
    page_title="Razzi Football Predictor",
    page_icon="⚽"
)

st.title("⚽ RAZZI FOOTBALL PREDICTOR")

st.write("Welcome to the football prediction app!")

st.subheader("Match Prediction")

team1 = st.text_input("Home Team")
team2 = st.text_input("Away Team")

if st.button("Predict"):
    st.success("Prediction system is ready!")
