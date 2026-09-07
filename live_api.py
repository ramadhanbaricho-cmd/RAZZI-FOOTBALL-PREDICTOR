import os, requests, streamlit as st
BASE="https://v3.football.api-sports.io"
class APIError(RuntimeError): pass
def key():
    try: return st.secrets.get("API_FOOTBALL_KEY") or os.getenv("API_FOOTBALL_KEY")
    except Exception: return os.getenv("API_FOOTBALL_KEY")
def format_errors(errors):
    text = "; ".join(f"{k}: {v}" for k,v in errors.items()) if isinstance(errors,dict) else str(errors)
    if "suspended" in text.lower(): return "Your API-Football account is suspended. Reactivate/check the account in the API-Football dashboard, then replace the API key in Streamlit Secrets if necessary."
    return text
def get(endpoint, params=None):
    k=key()
    if not k: raise APIError("API_FOOTBALL_KEY is not configured.")
    try: r=requests.get(BASE+endpoint, params=params or {}, headers={"x-apisports-key":k}, timeout=30)
    except requests.RequestException as e: raise APIError(f"Network error: {e}") from e
    try: data=r.json()
    except ValueError: raise APIError(f"API returned HTTP {r.status_code} with an invalid response.")
    if data.get("errors"): raise APIError(format_errors(data["errors"]))
    if not r.ok: raise APIError(f"HTTP {r.status_code}")
    return data.get("response",[])
def fixtures(date): return get("/fixtures",{"date":date})
def prediction(fid):
    x=get("/predictions",{"fixture":fid}); return x[0] if x else None
