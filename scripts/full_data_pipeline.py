import pandas as pd
from datetime import date
import os
import requests
import json
from datetime import datetime

def load_from_cache_with_scraping():
    if os.path.exists("cache/historical_data.parquet"):
        hist = pd.read_parquet("cache/historical_data.parquet")
    else:
        hist = None
    if os.path.exists("cache/jeff_data.json"):
        with open("cache/jeff_data.json", "r") as f:
            jeff_data = json.load(f)
    else:
        jeff_data = None
    if os.path.exists("cache/defaults.json"):
        with open("cache/defaults.json", "r") as f:
            defaults = json.load(f)
    else:
        defaults = None

    if hist is not None:
        if 'date' in hist.columns:
            # Coerce to datetime, then extract date, handling invalid entries
            dates = pd.to_datetime(hist['date'], errors='coerce')
            date_vals = dates.dt.date
            # Determine latest valid date or fallback default
            if date_vals.notna().any():
                latest_date = max(d for d in date_vals if d is not None)
            else:
                latest_date = date(2025, 6, 10)
        else:
            latest_date = date(2025, 6, 10)
    else:
        latest_date = date(2025, 6, 10)

    return hist, jeff_data, defaults, latest_date

# Additional functions and classes in tennis_updated.py would follow...
