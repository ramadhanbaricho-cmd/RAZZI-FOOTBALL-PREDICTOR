# RAZZI FOOTBALL PREDICTOR — FIXED

This update changes the Stage 4 dashboard so match predictions are fetched automatically and displayed as ranked cards and a table.

### Improvements
- Automatic `/predictions?fixture=...` calls after fixtures load.
- Home/Draw/Away percentages, confidence, predicted score, winner and advice.
- Top RAZZI Picks ranking.
- 5-minute fixture cache and 30-minute prediction cache.
- Maximum-match control to reduce API usage.
- Clear handling of API-Football account suspension.
- Offline Demo mode to test the UI without the API.

### Streamlit Secrets
```toml
API_FOOTBALL_KEY = "YOUR_API_FOOTBALL_KEY"
```

Your screenshot shows an API-Football account-suspended response. Live predictions cannot be retrieved until that account/key is active again. API-Football documents `/predictions` as the endpoint for fixture-level forecast data.
