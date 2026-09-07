# RAZZI FOOTBALL PREDICTOR — NO API KEY

A Streamlit football prediction dashboard that does **not** use API-Football and does not require an API key.

## Data source

The app downloads public-domain football match data from OpenFootball's `football.json` repository using raw GitHub files. The repository includes major leagues such as the English Premier League, Bundesliga, La Liga, Serie A and Ligue 1 and states that no API key is required.

The current and previous seasons are used for the statistical model. The dataset is updated periodically, so this is **not** a live-score service.

## Predictions

RAZZI calculates:
- Home / Draw / Away probabilities
- Confidence
- Most likely correct score
- Over 2.5 probability
- BTTS probability
- Expected goals (xG-style estimate)
- Ranked TOP RAZZI PICKS

The model combines league averages, team attacking/defensive strength and recent form with a Poisson goal model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload the project files to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select `app.py` as the main file.
4. Deploy.
5. **No Streamlit Secret/API key is required.**

## Important

Public data availability can change. Predictions are estimates and are not guaranteed results.
