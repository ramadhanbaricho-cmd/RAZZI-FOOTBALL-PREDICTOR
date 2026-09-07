# RAZZI FOOTBALL PREDICTOR — NO API KEY

A Streamlit football prediction dashboard that does not use API-Football or any API key.

## Features
- English Premier League, La Liga, Serie A, Bundesliga and Ligue 1
- Automatic next-fixture discovery
- **PREDICT NEXT MATCHES** button
- Elo + recency-weighted Poisson model
- Home / Draw / Away probabilities
- Most likely score
- Expected goals
- Over/Under 2.5
- BTTS
- Confidence ranking and TOP RAZZI PICKS
- Recent form
- Session prediction saving
- Public-data refresh

## Deploy on Streamlit Community Cloud
1. Upload all files in this folder to GitHub.
2. Choose `app.py` as the Main file path.
3. Deploy.
4. No Streamlit Secrets and no API key are required.

## Data
The app reads public football datasets from OpenFootball and uses Football-Data.co.uk as a supplementary/fallback source for current fixtures/results. The app requires internet access from the deployed server to refresh public data.

## Important
This is a statistical prediction tool, not a guarantee of match outcomes and not financial/gambling advice.
