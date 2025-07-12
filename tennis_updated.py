#%%
import hashlib
from bs4 import BeautifulSoup
#%%
# ============================================================================
# TENNIS DATA PIPELINE - COMPREHENSIVE TENNIS PREDICTION SYSTEM
# ============================================================================

# ============================================================================
# TENNIS DATA PIPELINE
# 1. IMPORTS AND CONSTANTS
# ============================================================================
import numpy as np
import pandas as pd
import os
import requests
import json
import pickle
import shutil
import time
import hashlib
import html
import re
import sys
import functools
import io
import collections
from datetime import datetime, date, timedelta
from pathlib import Path
import re
from unidecode import unidecode
import time
from bs4 import BeautifulSoup, FeatureNotFound
from urllib.parse import urlparse

# ============================================================================
# API CONFIGURATION AND CORE FUNCTIONS
# ============================================================================

# API-Tennis configuration
# API Configuration
API_KEY = "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb"
BASE = "https://api.api-tennis.com/tennis/"
CACHE_API = Path.home() / ".api_tennis_cache"
CACHE_API.mkdir(exist_ok=True)

# Cache Configuration
CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

# Data Configuration
DATA_DIR = "data"
BASE_CUTOFF_DATE = datetime.date(2025, 6, 10)
JEFF_DB_PATH = os.path.join(DATA_DIR, "jeff_tennis_data_until_20250610.parquet")
INCR_DB_PATH = os.path.join(DATA_DIR, "results_incremental.parquet")
CHARTING_INDEX_CSV = (
    "https://raw.githubusercontent.com/JeffSackmann/"
    "tennis_charting/master/charting_match_index.csv"
)


# ============================================================================
# 2. DATA INTEGRATION
# ============================================================================

# 2.1 Core API Functions
def api_call(method: str, **params):
    """Unified API call function with proper error handling"""
    try:
        response = requests.get(BASE, params={"method": method, "APIkey": API_KEY, **params}, timeout=10)
        response.raise_for_status()
        data = response.json()

        error_code = str(data.get("error", "0"))
        if error_code != "0":
            return []

        return data.get("result", [])
    except Exception as e:
        print(f"API call failed for {method}: {e}")
        return []

# ============================================================================
# NAME NORMALIZATION FUNCTIONS
# ============================================================================

def safe_int_convert(value, default=None):
    """Safely convert string/float to int, handling decimals and None values"""
    if value is None or value == "":
        return default
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default


# 2.2 Name Normalization Functions
def normalize_name(name):
    """Normalize tennis player names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).replace('.', '').lower()
    parts = name.split()
    if len(parts) < 2:
        return name.replace(' ', '_')
    if len(parts[-1]) == 1:  # Last part is single letter (first initial)
        last_name = parts[-2]
        first_initial = parts[-1]
    else:  # Handle "First Lastname" format
        last_name = parts[-1]
        first_initial = parts[0][0] if parts[0] else ''
    return f"{last_name}_{first_initial}"


def normalize_jeff_name(name):
    """Normalize Jeff's player names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    parts = name.split()
    if len(parts) < 2:
        return name.replace(' ', '_')
    last_name = parts[-1]
    first_initial = parts[0][0] if parts[0] else ''
    return f"{last_name}_{first_initial}"


def normalize_name_canonical(name):
    """Canonical name normalization for simulation"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    return ' '.join(name.lower().split())


def normalize_tournament_name(name):
    """Normalize tournament names"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = name.replace('masters cup', 'masters')
    name = name.replace('atp finals', 'masters')
    name = name.replace('wta finals', 'masters')
    return name.strip()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """YYYYMMDD-tournament-player1-player2 (all lower-snake)"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"


# 2.3 Data Loading Functions
def load_excel_data(file_path):
    """Load data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        if 'Date' not in df.columns:
            print(f"Warning: No Date column in {file_path}")
            return pd.DataFrame()
        print(f"Loaded {len(df)} matches from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_jeff_comprehensive_data():
    """Load all of Jeff's comprehensive tennis data"""
    base_path = os.path.expanduser("~/Desktop/data/Jeff 6.14.25")
    data = {'men': {}, 'women': {}}
    files = {
        'matches': 'charting-{}-matches.csv',
        'points_2020s': 'charting-{}-points-2020s.csv',
        'overview': 'charting-{}-stats-Overview.csv',
        'serve_basics': 'charting-{}-stats-ServeBasics.csv',
        'return_outcomes': 'charting-{}-stats-ReturnOutcomes.csv',
        'return_depth': 'charting-{}-stats-ReturnDepth.csv',
        'key_points_serve': 'charting-{}-stats-KeyPointsServe.csv',
        'key_points_return': 'charting-{}-stats-KeyPointsReturn.csv',
        'net_points': 'charting-{}-stats-NetPoints.csv',
        'rally': 'charting-{}-stats-Rally.csv',
        'serve_direction': 'charting-{}-stats-ServeDirection.csv',
        'serve_influence': 'charting-{}-stats-ServeInfluence.csv',
        'shot_direction': 'charting-{}-stats-ShotDirection.csv',
        'shot_dir_outcomes': 'charting-{}-stats-ShotDirOutcomes.csv',
        'shot_types': 'charting-{}-stats-ShotTypes.csv',
        'snv': 'charting-{}-stats-SnV.csv',
        'sv_break_split': 'charting-{}-stats-SvBreakSplit.csv',
        'sv_break_total': 'charting-{}-stats-SvBreakTotal.csv'
    }

    for gender in ['men', 'women']:
        gender_path = os.path.join(base_path, gender)
        if os.path.exists(gender_path):
            for key, filename_template in files.items():
                filename = filename_template.format('m' if gender == 'men' else 'w')
                file_path = os.path.join(gender_path, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, low_memory=False)
                    if 'player' in df.columns:
                        df['Player_canonical'] = df['player'].apply(normalize_jeff_name)
                    data[gender][key] = df
                    print(f"Loaded {gender}/{filename}: {len(df)} records")
    return data


def load_all_tennis_data():
    """Load tennis data from all years"""
    base_path = os.path.expanduser("~/Desktop/data")
    all_data = []

    for gender_name, gender_code in [("tennisdata_men", "M"), ("tennisdata_women", "W")]:
        gender_path = os.path.join(base_path, gender_name)
        if os.path.exists(gender_path):
            for year in range(2020, 2026):
                file_path = os.path.join(gender_path, f"{year}_{gender_code.lower()}.xlsx")
                if os.path.exists(file_path):
                    df = load_excel_data(file_path)
                    if not df.empty and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df['gender'] = gender_code
                        df['year'] = df['Date'].dt.year
                        all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# ============================================================================
# FEATURE EXTRACTION AND DEFAULTS
# ============================================================================

# 2.4 Feature Extraction Functions
def get_fallback_defaults(gender_key):
    """Fallback defaults when no Jeff data available"""
    base_defaults = {
        'serve_pts': 80, 'aces': 6, 'double_faults': 3, 'first_serve_pct': 0.62,
        'first_serve_won': 35, 'second_serve_won': 16, 'break_points_saved': 4,
        'return_pts_won': 30, 'winners_total': 28, 'winners_fh': 16, 'winners_bh': 12,
        'unforced_errors': 28, 'unforced_fh': 16, 'unforced_bh': 12,
        'serve_wide_pct': 0.3, 'serve_t_pct': 0.4, 'serve_body_pct': 0.3,
        'return_deep_pct': 0.4, 'return_shallow_pct': 0.3, 'return_very_deep_pct': 0.2,
        'key_points_serve_won_pct': 0.6, 'key_points_aces_pct': 0.05, 'key_points_first_in_pct': 0.55,
        'key_points_return_won_pct': 0.35, 'key_points_return_winners': 0.02,
        'net_points_won_pct': 0.65, 'net_winners_pct': 0.3, 'passed_at_net_pct': 0.3,
        'rally_server_winners_pct': 0.15, 'rally_server_unforced_pct': 0.2,
        'rally_returner_winners_pct': 0.1, 'rally_returner_unforced_pct': 0.25,
        'shot_crosscourt_pct': 0.5, 'shot_down_line_pct': 0.25, 'shot_inside_out_pct': 0.15,
        'serve_volley_frequency': 0.02, 'serve_volley_success_pct': 0.6,
        'return_error_net_pct': 0.1, 'return_error_wide_pct': 0.05,
        'aggression_index': 0.5, 'consistency_index': 0.5, 'pressure_performance': 0.5, 'net_game_strength': 0.5
    }

    if gender_key == 'women':
        base_defaults.update({
            'serve_pts': 75, 'aces': 4, 'first_serve_pct': 0.60,
            'first_serve_won': 32, 'second_serve_won': 15,
            'serve_volley_frequency': 0.01, 'net_points_won_pct': 0.60
        })

    return base_defaults

import collections, pandas as pd          # add near other imports

def calculate_comprehensive_weighted_defaults(jeff_data: dict) -> dict:
    defaults = {"men": {}, "women": {}}
    skip = {"matches", "points_2020s"}

    for sex in ("men", "women"):
        sums, counts = collections.defaultdict(float), collections.defaultdict(int)

        for name, df in jeff_data.get(sex, {}).items():
            if name in skip or df is None or df.empty:
                continue
            num = df.select_dtypes(include=["number"])
            for col in num.columns:
                vals = pd.to_numeric(num[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                sums[col]   += vals.sum()
                sums[col] += vals.sum()
                counts[col] += len(vals)

        defaults[sex] = {c: sums[c] / counts[c] for c in sums}

    return defaults


def extract_comprehensive_jeff_features(player_canonical, gender, jeff_data, weighted_defaults=None):
    """Extract features from all Jeff datasets with Player_canonical checks"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data:
        return get_fallback_defaults(gender_key)

    if weighted_defaults and gender_key in weighted_defaults:
        features = weighted_defaults[gender_key].copy()
    else:
        features = get_fallback_defaults(gender_key)

    # Overview stats
    if 'overview' in jeff_data[gender_key]:
        overview_df = jeff_data[gender_key]['overview']
        if 'Player_canonical' in overview_df.columns:
            player_overview = overview_df[
                (overview_df['Player_canonical'] == player_canonical) &
                (overview_df['set'] == 'Total')
            ]
                ]

            if len(player_overview) > 0:
                latest = player_overview.iloc[-1]
                serve_pts = latest.get('serve_pts', 80)
                if serve_pts > 0:
                    features.update({
                        'serve_pts': float(serve_pts),
                        'aces': float(latest.get('aces', 0)),
                        'double_faults': float(latest.get('dfs', 0)),
                        'first_serve_pct': float(latest.get('first_in', 0)) / float(serve_pts) if serve_pts > 0 else 0.62,
                        'first_serve_pct': float(latest.get('first_in', 0)) / float(
                            serve_pts) if serve_pts > 0 else 0.62,
                        'first_serve_won': float(latest.get('first_won', 0)),
                        'second_serve_won': float(latest.get('second_won', 0)),
                        'break_points_saved': float(latest.get('bp_saved', 0)),
                        'return_pts_won': float(latest.get('return_pts_won', 0)),
                        'winners_total': float(latest.get('winners', 0)),
                        'winners_fh': float(latest.get('winners_fh', 0)),
                        'winners_bh': float(latest.get('winners_bh', 0)),
                        'unforced_errors': float(latest.get('unforced', 0)),
                        'unforced_fh': float(latest.get('unforced_fh', 0)),
                        'unforced_bh': float(latest.get('unforced_bh', 0))
                    })

    return features

# ============================================================================
# API-TENNIS HELPER FUNCTIONS
# ============================================================================

def extract_jeff_features(player_canonical, gender, jeff_data):
    """Extract actual features from Jeff Sackmann data - simplified version for simulation"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or player_canonical not in jeff_data[gender_key]:
        return {
            'serve_pts': 60,
            'first_won': 0,
            'second_won': 0,
            'return_pts_won': 20
        }

    player_data = jeff_data[gender_key][player_canonical]

    first_in = player_data.get('1stIn', 0)
    first_won = player_data.get('1stWon', 0)
    second_won = player_data.get('2ndWon', 0)
    double_faults = player_data.get('df', 0)

    total_serve_pts = first_in + double_faults + (
                first_won - first_in) if first_won >= first_in else first_in + second_won + double_faults

    break_points_saved = player_data.get('bpSaved', 0)
    break_points_faced = player_data.get('bpFaced', 0)
    return_pts_won = break_points_faced - break_points_saved

    return {
        'serve_pts': max(1, total_serve_pts),
        'first_won': first_won,
        'second_won': second_won,
        'return_pts_won': max(0, return_pts_won)
    }


# 2.5 Tennis Abstract Scraper
class TennisAbstractScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def _normalize_player_name(self, name: str) -> str:
        import re, unicodedata
        name = unicodedata.normalize("NFKD", name).strip()
        name = re.sub(r"[\s\-]+", "_", name)
        return name.lower()

    def _to_int(self, text):
        """Extract leading integer from a value"""
        if text is None or pd.isna(text):
            return 0
        if isinstance(text, (int, float)):
            return int(text)
        m = re.search(r"\d+", str(text))
        return int(m.group(0)) if m else 0

    def _pct(self, text: str) -> float:
        """Convert percentage inputs to decimal"""
        import pandas as pd, math, re

        if text is None or (isinstance(text, float) and math.isnan(text)) or (
                isinstance(text, (int, float)) and pd.isna(text)
        ):
            return 0.0

        if isinstance(text, (int, float)):
            val = float(text)
            return val if 0 <= val <= 1 else val / 100.0

        s = str(text).strip()
        if s.endswith("%"):
            s = s[:-1].strip()

        m = re.search(r"[\d.]+", s)
        if not m:
            return 0.0

        val = float(m.group(0))
        return val if 0 <= val <= 1 else val / 100.0

    def scrape_shot_types(self, url, debug: bool = False):
        """Parse shot-type distribution tables"""
        import re, html, requests
        import numpy as np

        def _parse_count_pct(txt: str) -> tuple[int, float | None]:
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", txt or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        matches = re.findall(r"var\s+(shots\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta = self._parse_match_url(url)
        player_map = {"shots1": meta.get("player1"), "shots2": meta.get("player2")}
        out = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            table = BeautifulSoup(html_blob, "html.parser").find("table")
            if table is None:
                continue

            headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
            hdr_norm = [re.sub(r"[%\s]+", "_", h.lower()).strip("_") for h in headers]

            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not cells or not cells[0]:
                    continue

                rec = {
                    "player": player,
                    "Player_canonical": canon,
                    "category": cells[0],
                }

                for h, v in zip(hdr_norm[1:], cells[1:]):
                    cnt, pct = _parse_count_pct(v)
                    rec[h] = cnt
                    rec[f"{h}_pct"] = pct if pct is not None else np.nan

                if cells[0].strip().lower() == "total":
                    rec["total_pct"] = 1.0

                out.append(rec)

        unique = {}
        for rec in out:
            key = (rec["player"], rec["category"])
            unique[key] = rec

        return list(unique.values())

    def _parse_match_url(self, url: str) -> dict:
        """Parse Tennis Abstract charting URL for metadata"""
        fname = os.path.basename(urlparse(url).path)
        if not fname.endswith(".html"):
            return {}

        parts = fname[:-5].split("-")
        if len(parts) < 6 or not parts[0].isdigit():
            return {}

        date_str = parts[0]
        gender = parts[1]
        round_tok = parts[-3]
        player1 = parts[-2].replace("_", " ")
        player2 = parts[-1].replace("_", " ")
        tournament = "-".join(parts[2:-3]).replace("_", " ")

        return {
            "Date": date_str,
            "gender": gender,
            "tournament": tournament,
            "round": round_tok,
            "player1": player1,
            "player2": player2,
        }


# 2.6 API Integration Functions
def get_fixtures_for_date(target_date):
    """Get all fixtures for a specific date - includes embedded statistics"""
    """Get all fixtures for a specific date"""
    try:
        fixtures = api_call("get_fixtures",
                           date_start=target_date.isoformat(),
                           date_stop=target_date.isoformat(),
                           timezone="UTC")
                            date_start=target_date.isoformat(),
                            date_stop=target_date.isoformat(),
                            timezone="UTC")

        finished_fixtures = [ev for ev in fixtures if ev.get("event_status") == "Finished"]
        return finished_fixtures
    except Exception as e:
        print(f"Error getting fixtures for {target_date}: {e}")
        return []


def extract_embedded_statistics(fixture):
    """Extract statistics from fixture data (no separate API call needed)"""
    """Extract statistics from fixture data"""
    stats = {}

    # Extract from scores data
    scores = fixture.get("scores", [])
    if scores:
        try:
            p1_sets = 0
            p2_sets = 0
            for s in scores:
                score_first = safe_int_convert(s.get("score_first", 0), 0)
                score_second = safe_int_convert(s.get("score_second", 0), 0)
                if score_first > score_second:
                    p1_sets += 1
                elif score_second > score_first:
                    p2_sets += 1

            stats["sets_won_p1"] = p1_sets
            stats["sets_won_p2"] = p2_sets
            stats["total_sets"] = len(scores)
        except Exception as e:
            # Don't fail the entire match for score parsing errors
            pass

    return stats


# ----------------------------------------------------------------------------
# API-TENNIS: Statistics parsing helper
# ----------------------------------------------------------------------------
def parse_match_statistics(fixture):
    """
    Parse the raw ``fixture["statistics"]`` list into per–player dictionaries.

    Parameters
    ----------
    fixture : dict
        A single fixture dictionary returned by the API.  We expect a key
        ``"statistics"`` that is a list of dictionaries, each containing at
        least ``"player_key"``, ``"type"``, and ``"value"``.

    Returns
    -------
    dict
        ``{player_key: {stat_name: value, ...}, ...}``
        where *stat_name* is the lower‑snake‑case version of the original
        ``type`` string and *value* is converted to ``float`` when possible.
    """
    stats_pair = {}

    for entry in fixture.get("statistics", []):
        p_key = safe_int_convert(entry.get("player_key"))
        if p_key is None:
            continue

        # Normalise statistic name: "First Serves In" -> "first_serves_in"
        stat_name = str(entry.get("type", "")).strip().lower().replace(" ", "_")
        if not stat_name:
            continue

        raw_val = entry.get("value")
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = raw_val  # leave as‑is if non‑numeric

        if p_key not in stats_pair:
            stats_pair[p_key] = {}
        stats_pair[p_key][stat_name] = val

    return stats_pair

def get_match_odds(match_key, date_check=None):
    """Get odds with proper error handling - only for dates >= 2025-06-23"""
    """Get odds with proper error handling"""
    if date_check and date_check < date(2025, 6, 23):
        return (None, None)

    try:
        # Convert match_key to int for API call, but keep as string for lookup
        match_key_int = safe_int_convert(match_key)
        if match_key_int is None:
            return (None, None)

        odds_data = api_call("get_odds", match_key=match_key_int)
        if not odds_data or str(match_key_int) not in odds_data:
            return (None, None)

        match_odds = odds_data[str(match_key_int)]
        home_away = match_odds.get("Home/Away", {})

        # Get average odds across bookmakers
        home_odds = home_away.get("Home", {})
        away_odds = home_away.get("Away", {})

        if home_odds and away_odds:
            # Take first bookmaker's odds or average if multiple
            home_val = next(iter(home_odds.values())) if home_odds else None
            away_val = next(iter(away_odds.values())) if away_odds else None

            return (float(home_val) if home_val else None,
                   float(away_val) if away_val else None)
                    float(away_val) if away_val else None)

        return (None, None)
    except Exception as e:
        print(f"Error getting odds for match {match_key}: {e}")
        return (None, None)

def safe_int_convert(value, default=None):
    """Safely convert string/float to int, handling decimals and None values"""
    if value is None or value == "":
        return default
    try:
        # Convert to float first to handle decimal strings like "7.7", then to int
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default

def get_player_rankings(day, league="ATP"):
    """Get standings with proper caching and error handling"""
    tag = f"{league}_{day.isocalendar()[0]}_{day.isocalendar()[1]:02d}.pkl"
    cache_file = CACHE_API / tag

    if cache_file.exists():
        try:
            standings = pickle.loads(cache_file.read_bytes())
            if standings:  # Non-empty cache
            if standings:
                rankings = {}
                for r in standings:
                    player_key = safe_int_convert(r.get("player_key"))
                    place = safe_int_convert(r.get("place"))
                    if player_key is not None and place is not None:
                        rankings[player_key] = place
                return rankings
        except Exception as e:
            print(f"Cache read error for {tag}: {e}")

    # Correct parameter for API
    standings = api_call("get_standings", event_type=league.upper())

    try:
        cache_file.write_bytes(pickle.dumps(standings, 4))
    except Exception as e:
        print(f"Cache write error for {tag}: {e}")

    # Process standings with safe conversion
    rankings = {}
    for r in standings:
        player_key = safe_int_convert(r.get("player_key"))
        place = safe_int_convert(r.get("place"))
        if player_key is not None and place is not None:
            rankings[player_key] = place

    return rankings

def get_h2h_data(p1_key, p2_key):
    """Get head-to-head data with caching"""
    cache_file = CACHE_API / f"h2h_{p1_key}_{p2_key}.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        h2h_data = api_call("get_H2H", first_player_key=p1_key, second_player_key=p2_key)

        # Handle empty response or error
        if not h2h_data or not isinstance(h2h_data, list) or len(h2h_data) == 0:
            result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
        else:
            h2h_matches = h2h_data[0].get("H2H", []) if isinstance(h2h_data[0], dict) else []
            if not h2h_matches:
                result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
            else:
                p1_wins = sum(1 for m in h2h_matches if m.get("event_winner") == "First Player")
                p2_wins = len(h2h_matches) - p1_wins

                result = {
                    "h2h_matches": len(h2h_matches),
                    "p1_wins": p1_wins,
                    "p2_wins": p2_wins,
                    "p1_win_pct": p1_wins / len(h2h_matches) if h2h_matches else 0.5
                }

        cache_file.write_bytes(pickle.dumps(result, 4))
        return result

    except Exception as e:
        result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
        cache_file.write_bytes(pickle.dumps(result, 4))
        return result

def get_tournaments_metadata():
    """Get tournament metadata (surface, level, location) - cached statically"""
    cache_file = CACHE_API / "tournaments.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        tournaments = api_call("get_tournaments")
        # Use safe conversion for tournament_key
        tournament_dict = {}
        for t in tournaments:
            tournament_key = safe_int_convert(t.get("tournament_key"))
            if tournament_key is not None:
                tournament_dict[str(tournament_key)] = t  # Keep as string for lookup
        cache_file.write_bytes(pickle.dumps(tournament_dict, 4))
        return tournament_dict
    except Exception as e:
        print(f"Error getting tournaments: {e}")
        return {}

def get_event_types():
    """Get event types metadata - cached statically"""
    cache_file = CACHE_API / "event_types.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        events = api_call("get_events")
        # Use safe conversion for event_type_key
        event_dict = {}
        for e in events:
            event_type_key = safe_int_convert(e.get("event_type_key"))
            if event_type_key is not None:
                event_dict[str(event_type_key)] = e  # Keep as string for lookup
        cache_file.write_bytes(pickle.dumps(event_dict, 4))
        return event_dict
    except Exception as e:
        print(f"Error getting event types: {e}")
        return {}

def get_player_profile(player_key):
    """Get player profile with career stats"""
    cache_file = CACHE_API / f"player_{player_key}.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        player_data = api_call("get_players", player_key=player_key)
        result = player_data[0] if player_data else {}
        cache_file.write_bytes(pickle.dumps(result, 4))
        return result
    except Exception as e:
        print(f"Error getting player {player_key}: {e}")
        return {}

# ------------------------------------------------------------------
# composite-key helper
# ------------------------------------------------------------------
def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """YYYYMMDD-tournament-player1-player2 (all lower-snake)"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"

# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

def generate_comprehensive_historical_data(fast=True, n_sample=500):
    """Generate comprehensive historical data with API integration"""
    print("=== STARTING DATA GENERATION ===")

    # Step 1: Load Jeff's data
    print("Step 1: Loading Jeff's comprehensive data...")
    try:
        jeff_data = load_jeff_comprehensive_data()
        if not jeff_data or ('men' not in jeff_data and 'women' not in jeff_data):
            print("ERROR: Jeff data loading failed")
            return pd.DataFrame(), {}, {}

        print(f"✓ Jeff data loaded successfully")
        print(f"  - Men's datasets: {len(jeff_data.get('men', {}))}")
        print(f"  - Women's datasets: {len(jeff_data.get('women', {}))}")

    except Exception as e:
        print(f"ERROR loading Jeff data: {e}")
        return pd.DataFrame(), {}, {}

    # Step 2: Calculate weighted defaults
    print("Step 2: Calculating weighted defaults...")
    try:
        weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)
        if not weighted_defaults:
            print("ERROR: Weighted defaults calculation failed")
            return pd.DataFrame(), jeff_data, {}

        print(f"✓ Weighted defaults calculated")
        print(f"  - Men's features: {len(weighted_defaults.get('men', {}))}")
        print(f"  - Women's features: {len(weighted_defaults.get('women', {}))}")

    except Exception as e:
        print(f"ERROR calculating weighted defaults: {e}")
        return pd.DataFrame(), jeff_data, {}

    # Step 3: Load tennis match data
    print("Step 3: Loading tennis match data...")
    try:
        tennis_data = load_all_tennis_data()
        if tennis_data.empty:
            print("ERROR: No tennis data loaded")
            return pd.DataFrame(), jeff_data, weighted_defaults

        print(f"✓ Tennis data loaded: {len(tennis_data)} matches")

        # Fast mode for testing
        if fast:
            total_rows = len(tennis_data)
            take = min(n_sample, total_rows)
            tennis_data = tennis_data.sample(take, random_state=1).reset_index(drop=True)
            print(f"[FAST MODE] Using sample of {take}/{total_rows} rows")

    except Exception as e:
        print(f"ERROR loading tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 4: Process tennis data
    print("Step 4: Processing tennis data...")
    try:
        # Normalize player names
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)
        tennis_data['tournament_canonical'] = tennis_data['Tournament'].apply(normalize_tournament_name)
        # --- fix: ensure Date columns exist before building composite_id ---
        tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
        tennis_data['date'] = tennis_data['Date'].dt.date

        # ------------------------------------------------------------------
        # Composite match identifier shared across data sources
        # ------------------------------------------------------------------
        tennis_data['composite_id'] = tennis_data.apply(
            lambda r: build_composite_id(
                r['date'],
                r['tournament_canonical'],
                r['winner_canonical'],
                r['loser_canonical']
            ),
            axis=1
        )

        # Add odds data
        tennis_data['tennis_data_odds1'] = pd.to_numeric(tennis_data.get('PSW', 0), errors='coerce')
        tennis_data['tennis_data_odds2'] = pd.to_numeric(tennis_data.get('PSL', 0), errors='coerce')

        # Add ranking difference
        if 'WRank' in tennis_data.columns and 'LRank' in tennis_data.columns:
            tennis_data['rank_difference'] = abs(pd.to_numeric(tennis_data['WRank'], errors='coerce') -
                                                 pd.to_numeric(tennis_data['LRank'], errors='coerce'))

        print(f"✓ Tennis data processed")

    except Exception as e:
        print(f"ERROR processing tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 5: Adding Jeff feature columns...
    print("Step 5: Adding Jeff feature columns...")
    try:
        # Build feature list dynamically from the weighted defaults
        men_feats   = set(weighted_defaults.get('men', {}).keys())
        women_feats = set(weighted_defaults.get('women', {}).keys())
        all_jeff_features = sorted(men_feats.union(women_feats))

        if not all_jeff_features:
            raise ValueError("No features available in weighted_defaults")

        # ------------------------------------------------------------------
        #  vectorised column creation – avoid fragmentation warnings
        # ------------------------------------------------------------------
        missing_cols_dict = {}

        for feat in all_jeff_features:
            w_col = f"winner_{feat}"
            l_col = f"loser_{feat}"

            if w_col not in tennis_data.columns:
                missing_cols_dict[w_col] = np.full(len(tennis_data), np.nan, dtype="float64")
            if l_col not in tennis_data.columns:
                missing_cols_dict[l_col] = np.full(len(tennis_data), np.nan, dtype="float64")

        if missing_cols_dict:
            tennis_data = pd.concat(
                [tennis_data, pd.DataFrame(missing_cols_dict, index=tennis_data.index)],
                axis=1
            )

        print(f"✓ Added/verified {len(all_jeff_features) * 2} feature columns")

    except Exception as e:
        print(f"ERROR adding feature columns: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 6: Extract Jeff features
    print("Step 6: Extracting Jeff features...")
    try:
        total_matches = len(tennis_data)
        matches_with_jeff_features = 0

        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                print(f"  Processing match {idx}/{total_matches}")

            try:
                gender = row['gender']

                # Only extract Jeff features for matches before cutoff
                if row['date'] <= date(2025, 6, 10):
                    winner_features = extract_comprehensive_jeff_features(
                        row['winner_canonical'], gender, jeff_data, weighted_defaults
                    )
                    loser_features = extract_comprehensive_jeff_features(
                        row['loser_canonical'], gender, jeff_data, weighted_defaults
                    )

                    # Assign features
                    for feature_name, feature_value in winner_features.items():
                        col_name = f'winner_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    for feature_name, feature_value in loser_features.items():
                        col_name = f'loser_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    if winner_features and loser_features:
                        matches_with_jeff_features += 1

            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"  Warning: Error processing match {idx}: {e}")
                continue

        print(f"✓ Jeff features extracted for {matches_with_jeff_features}/{total_matches} matches")

    except Exception as e:
        print(f"ERROR extracting Jeff features: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print(f"=== DATA GENERATION COMPLETE ===")
    print(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

def save_to_cache(historical_data, jeff_data, weighted_defaults):
    """Save data to cache"""
    print("\n=== SAVING TO CACHE ===")
    # --- sanitize numeric columns that sometimes contain stray text ---
    numeric_cols = ["MaxW", "MaxL", "AvgW", "AvgL", "PSW", "PSL"]
    for col in numeric_cols:
        if col in historical_data.columns:
            historical_data[col] = pd.to_numeric(historical_data[col], errors="coerce")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # Save historical data
        historical_data.to_parquet(HD_PATH, index=False)
        print("✓ Historical data saved")

        # Save Jeff data
        with open(JEFF_PATH, "wb") as f:
            pickle.dump(jeff_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Jeff data saved")

        # Save weighted defaults
        with open(DEF_PATH, "wb") as f:
            pickle.dump(weighted_defaults, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Weighted defaults saved")

        return True
    except Exception as e:
        print(f"ERROR saving cache: {e}")
        return False

def load_from_cache():
    """Load data from cache if available"""
    if (os.path.exists(HD_PATH) and
        os.path.exists(JEFF_PATH) and
        os.path.exists(DEF_PATH)):

        print("Loading from cache...")
        historical_data = pd.read_parquet(HD_PATH)

        with open(JEFF_PATH, "rb") as f:
            jeff_data = pickle.load(f)

        with open(DEF_PATH, "rb") as f:
            weighted_defaults = pickle.load(f)

        return historical_data, jeff_data, weighted_defaults
    else:
        return None, None, None

# %%
class TennisAbstractScraper:
    # ------------------------------------------------------------------
    # helper: the "serve" JS variable actually contains TWO tables
    # (Serve Basics + Direction).  Split them cleanly.
    # ------------------------------------------------------------------
    def _split_serve_tables(self, raw_html: str) -> tuple[str, str]:
        """Return `(basics_html, direction_html)` from the combined blob."""
        tables = BeautifulSoup(raw_html, "html.parser").find_all("table")
        if not tables:
            return "", ""
        basics_html = str(tables[0])
        direction_html = str(tables[1]) if len(tables) > 1 else ""
        return basics_html, direction_html
    # Column headers → Jeff keys for the “Serve Influence” table
    MAP_SERVE_INFL = {
        "1+ shots": "srv_neut_len1_pct",
        "2+": "srv_neut_len2_pct",
        "3+": "srv_neut_len3_pct",
        "4+": "srv_neut_len4_pct",
        "5+": "srv_neut_len5_pct",
        "6+": "srv_neut_len6_pct",
        "7+": "srv_neut_len7_pct",
        "8+": "srv_neut_len8_pct",
        "9+": "srv_neut_len9_pct",
        "10+": "srv_neut_len10_pct",
    }
    def __init__(self, jeff_headers: set[str] | None = None):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.jeff_headers = jeff_headers if jeff_headers is not None else JEFF_HEADERS

    # --- generic cell parser ------------------------------------------
    def _to_int(self, text):
        """
        Extract leading integer from a value such as '51  (52%)', '0', or a numeric/NaN.
        Returns 0 when the value is missing or no digit is found.
        """
        import pandas as pd
        # Handle None / NaN (including numpy.nan) early
        if text is None or pd.isna(text):
            return 0

        # Direct numeric input
        if isinstance(text, (int, float)):
            return int(text)

        # Fallback for string input
        m = re.search(r"\d+", str(text))
        return int(m.group(0)) if m else 0

    def _filter_jeff(self, rec: dict) -> dict:
        """
        Pass‑through filter: return the record unchanged.

        The caller now receives *all* scraped fields, not just the subset
        present in Jeff Sackmann’s original column set.  Down‑stream code
        should handle any extra keys as needed.
        """
        return rec

    # --- helper: canonicalise a player name ----------------------------
    def _normalize_player_name(self, name: str) -> str:
        import re, unicodedata
        name = unicodedata.normalize("NFKD", name).strip()
        name = re.sub(r"[\s\-]+", "_", name)
        return name.lower()

    # --- Jeff conversion placeholder ---------------------------------
    def _convert_to_jeff_format(self, raw_records: list[dict], match_meta: dict) -> list[dict]:
        """
        Translate raw parsed records into Jeff‑style keys.

        Current placeholder just merges match metadata onto every
        raw record and returns the list unchanged.

        Parameters
        ----------
        raw_records : list[dict]
            Data extracted from a stats table.
        match_meta : dict
            Output of `_parse_match_url` with Date, tournament, etc.

        Returns
        -------
        list[dict]
            Records augmented with basic match metadata.
        """
        if not raw_records:
            return []

        shared = match_meta.copy()
        out = []
        for rec in raw_records:
            merged = rec.copy()
            for k, v in shared.items():
                if k not in merged:
                    merged[k] = v
            out.append(merged)
        return out

    def _parse_match_url(self, url: str) -> dict:
        """
        Parse a Tennis‑Abstract charting URL for basic metadata.

        URL pattern (examples)
        ----------------------
        https://…/20250701-W-Wimbledon-R128-Mayar_Sherif-Mirra_Andreeva.html
        https://…/20240611-M-French_Open-SF-Nadal_R-Djokovic_N.html

        Returns
        -------
        dict with keys
          Date, gender, tournament, round, player1, player2
        """
        from urllib.parse import urlparse
        import os

        fname = os.path.basename(urlparse(url).path)
        if not fname.endswith(".html"):
            return {}

        parts = fname[:-5].split("-")              # drop ".html"
        if len(parts) < 6 or not parts[0].isdigit():
            return {}

        date_str = parts[0]
        gender   = parts[1]                        # "M" or "W"
        # The last three tokens are <round> <player1> <player2>
        round_tok = parts[-3]
        player1   = parts[-2].replace("_", " ")
        player2   = parts[-1].replace("_", " ")
        tournament = "-".join(parts[2:-3]).replace("_", " ")

        return {
            "Date"     : date_str,
            "gender"   : gender,
            "tournament": tournament,
            "round"    : round_tok,
            "player1"  : player1,
            "player2"  : player2,
        }

    # ------------------------------------------------------------------
    #  header scraper
    # ------------------------------------------------------------------
    def _extract_match_header(self, soup: BeautifulSoup) -> dict:
        """
        Extract header metadata displayed above the stats tables.

        Expected HTML structure:

          <h2>2025 Wimbledon R128: Mayar Sherif vs Mirra Andreeva</h2>
          <b>Mirra Andreeva d. Mayar Sherif 6-3 6-3</b>
        """
        head_tag = soup.find("h2")
        bold_tag = soup.find("b")
        if not head_tag:
            return {}

        # ---------- line 1 -------------------------------------------------
        head_text = head_tag.get_text(" ", strip=True)
        if ":" not in head_text or " vs " not in head_text.lower():
            return {}

        left, right = head_text.split(":", 1)
        year_tour_round = left.strip().split()
        if len(year_tour_round) < 3 or not year_tour_round[0].isdigit():
            return {}

        year   = year_tour_round[0]
        round_ = year_tour_round[-1]
        tournament = " ".join(year_tour_round[1:-1])

        p1, p2 = [p.strip() for p in re.split(r"\s+vs\s+", right, flags=re.I)]

        # ---------- line 2 -------------------------------------------------
        winner = loser = score = ""
        if bold_tag:
            bold_text = bold_tag.get_text(" ", strip=True)
            m = re.match(r"(.+?)\s+d\.\s+(.+?)\s+([\d\s\-–]+)$", bold_text, flags=re.I)
            if m:
                winner, loser, score = m.groups()

        return {
            "year"     : year,
            "tournament": tournament,
            "round"    : round_,
            "player1"  : p1,
            "player2"  : p2,
            "winner"   : winner.strip(),
            "loser"    : loser.strip(),
            "score"    : score.strip()
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Stats Overview entry point – now merges header metadata
    # ──────────────────────────────────────────────────────────────────────────
    def scrape_stats_overview(self, url):
        resp = requests.get(url, headers=self.headers)
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(resp.text, "html.parser")

        # header block
        header_meta = self._extract_match_header(soup)

        # embedded JS blocks
        scripts = [t.string for t in soup.find_all("script") if t.string]
        blocks  = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';",
                                  "\n".join(scripts), re.S))
        labels  = {sp.get_text(strip=True): sp["id"]
                   for sp in soup.select("span.rounds")}
        stats_html = html.unescape(blocks.get(labels.get("Stats Overview", ""), ""))

        records = self._extract_stats_overview_table(stats_html)

        # attach meta
        for r in records:
            r.update(header_meta)

        return records


    def scrape_serve_influence(self, url):
        resp  = requests.get(url, headers=self.headers)
        soup  = BeautifulSoup(resp.text, "lxml")

        js    = "\n".join(t.string for t in soup.find_all("script") if t.string)
        html_ = html.unescape(
                    dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))
                    .get("serveNeut", "")
                )
        if not html_:
            return []

        tbl   = BeautifulSoup(html_, "html.parser").table
        heads = [h.get_text(strip=True).rstrip('%').strip()
                 for h in tbl.tr.find_all(['th', 'td'])]

        out = []
        for tr in tbl.find_all("tr")[1:]:
            cells  = [c.get_text(strip=True) for c in tr.find_all("td")]
            if not cells or not cells[0]:
                continue
            rec = {"Player_canonical": self._normalize_player_name(cells[0])}
            for h, v in zip(heads[1:], cells[1:]):
                key = self.MAP_SERVE_INFL.get(h)         # now matches
                if key:
                    # handle missing or non‑numeric percentages
                    txt = v.strip()
                    if txt in {"", "-", "—"}:
                        rec[key] = np.nan
                    else:
                        try:
                            rec[key] = float(txt.rstrip("%")) / 100
                        except ValueError:
                            rec[key] = np.nan
            out.append(self._filter_jeff(rec))
        return out

    # ------------------------------------------------------------------
    # SERVE BASICS  +  SERVE DIRECTION
    # ------------------------------------------------------------------
    def scrape_serve_statistics_overview(self, url, debug=False):
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        js = "\n".join(t.string for t in soup.find_all("script") if t.string)
        blob = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))

        raw_html = html.unescape(blob.get("serve", ""))
        basics_html, direct_html = self._split_serve_tables(raw_html)
        if not basics_html:
            return []

        match = self._parse_match_url(url)
        init = {"".join(w[0] for w in p.split()).upper(): p
                for p in (match["player1"], match["player2"])}

        # ───── BASICS ─────────────────────────────────────────────────────
        tbl = BeautifulSoup(basics_html, "html.parser").table
        rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")][1:]

        if debug:
            print("=== BASICS RAW ROWS ===")
            for r in rows[:6]:
                print(r)

        out = []
        for r in rows:
            if len(r) < 10 or not r[0]:
                continue
            abbr, serve_type = r[0].split()  # 'MS', 'Total'
            player = init.get(abbr.upper())
            if player is None:
                continue
            pts, won, ace, unret, fcde, le3w, wide, body, t = r[1:10]
            out.append({
                "match_id": f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date": match["Date"],
                "Tournament": match["tournament"],
                "player": player,
                "Player_canonical": self._normalize_player_name(player),
                "set": "Total",
                "serve_type": serve_type,
                "serve_pts": int(pts),
                "serve_won": self._to_int(won),
                "aces": self._to_int(ace),
                "unret": self._to_int(unret),
                "fcdE": self._to_int(fcde),
                "le3w": self._to_int(le3w),
                "wide_pct": self._pct(wide),
                "body_pct": self._pct(body),
                "t_pct": self._pct(t),
            })

        # ───── DIRECTION ─────────────────────────────────────────────────
        if direct_html:
            tbl2 = BeautifulSoup(direct_html, "html.parser").table
            rows2 = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                     for tr in tbl2.find_all("tr")][1:]

            if debug:
                print("=== DIRECTION RAW ROWS ===")
                for r in rows2[:6]:
                    print(r)

            for r in rows2:
                if len(r) < 13 or not r[0]:
                    continue
                abbr, serve_type = r[0].split()
                player = init.get(abbr.upper())
                if player is None:
                    continue
                dcw, dcb, dct, adw, adb, adt, net, werr, derr, wd, foot, unk = r[1:13]
                rec = next(i for i in out
                           if i["player"] == player and i["serve_type"] == serve_type)
                rec.update({
                    "dc_wide_pct": self._pct(dcw),
                    "dc_body_pct": self._pct(dcb),
                    "dc_t_pct": self._pct(dct),
                    "ad_wide_pct": self._pct(adw),
                    "ad_body_pct": self._pct(adb),
                    "ad_t_pct": self._pct(adt),
                    "net_pct": self._pct(net),
                    "wide_err_pct": self._pct(werr),
                    "deep_err_pct": self._pct(derr),
                    "w_d_pct": self._pct(wd),
                    "footfault_pct": self._pct(foot),
                    "unk_pct": self._pct(unk),
                })

        return out

    def scrape_serve_breakdown(self, url, debug: bool = False):
        """
        Extract per‑player serve and return breakdown tables embedded in the
        inline‑JS variables “serve1”, “serve2”, “return1”, “return2”.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print the first few parsed rows.

        Returns
        -------
        list[dict]
            One record per player × category row.
        """
        import re, html, pandas as pd

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # gather all inline‑JS blocks
        scripts = [t.string for t in soup.find_all("script") if t.string]
        js_blob = "\n".join(scripts)

        # match serve1/serve2/return1/return2 blobs
        pattern = re.compile(r"var\s+((?:serve|return)[12])\s*=\s*'([\s\S]*?)';", re.S)
        matches = pattern.findall(js_blob)
        if not matches:
            return []

        meta = self._parse_match_url(url)
        player_map = {"1": meta["player1"], "2": meta["player2"]}

        records: list[dict] = []
        for var_name, raw_html in matches:
            side   = "serve" if var_name.startswith("serve") else "return"
            player = player_map[var_name[-1]]
            canon  = self._normalize_player_name(player)

            tables = pd.read_html(html.unescape(raw_html))
            # tables[0] → Totals; tables[1] → 1st/2nd breakdown
            for idx, tbl in enumerate(tables):
                is_total = idx == 0
                for _, row in tbl.iterrows():
                    cat = row.iloc[0]
                    import pandas as pd
                    if pd.isna(cat) or str(cat).strip() == "":
                        continue
                    if is_total:
                        total_pts   = self._to_int(row.iloc[1])
                        won         = self._to_int(row.iloc[2])
                        aces        = self._to_int(row.iloc[3])
                        unret       = self._to_int(row.iloc[4])
                        fcdE        = self._to_int(row.iloc[5])
                        le3w        = self._to_int(row.iloc[6])
                        canon       = self._normalize_player_name(player)

                        rec = {
                            "player": player,
                            "Player_canonical": canon,
                            "side": side,
                            "category": cat,
                            "total_pts": total_pts,
                            "won": won,
                            "won_pct": won / total_pts if total_pts else 0.0,
                            "aces": aces,
                            "aces_pct": aces / total_pts if total_pts else 0.0,
                            "unret": unret,
                            "unret_pct": unret / total_pts if total_pts else 0.0,
                            "fcdE": fcdE,
                            "fcdE_pct": fcdE / total_pts if total_pts else 0.0,
                            "le3w": le3w,
                            "le3w_pct": le3w / total_pts if total_pts else 0.0,
                        }
                        if side == "serve":
                            rec["first_in"] = self._to_int(row.iloc[7])
                            rec["double_faults"] = self._to_int(row.iloc[8])

                        records.append(rec)
                    else:
                        first_pts   = self._to_int(row.iloc[1])
                        first_won   = self._to_int(row.iloc[2])
                        aces_first  = self._to_int(row.iloc[3])
                        unret_first = self._to_int(row.iloc[4])
                        fcdE_first  = self._to_int(row.iloc[5])
                        le3w_first  = self._to_int(row.iloc[6])
                        second_pts  = self._to_int(row.iloc[7])
                        second_won  = self._to_int(row.iloc[8])
                        canon       = self._normalize_player_name(player)

                        rec = {
                            "player": player,
                            "Player_canonical": canon,
                            "side": side,
                            "category": cat,
                            "first_pts": first_pts,
                            "first_won": first_won,
                            "first_won_pct": first_won / first_pts if first_pts else 0.0,
                            "aces_first": aces_first,
                            "aces_first_pct": aces_first / first_pts if first_pts else 0.0,
                            "unret_first": unret_first,
                            "unret_first_pct": unret_first / first_pts if first_pts else 0.0,
                            "fcdE_first": fcdE_first,
                            "fcdE_first_pct": fcdE_first / first_pts if first_pts else 0.0,
                            "le3w_first": le3w_first,
                            "le3w_first_pct": le3w_first / first_pts if first_pts else 0.0,
                            "second_pts": second_pts,
                            "second_won": second_won,
                            "second_won_pct": second_won / second_pts if second_pts else 0.0,
                        }

                        records.append(rec)

        if debug:
            for rec in records[:8]:
                print(rec)

        return records

    # ------------------------------------------------------------------
    # KEY‑POINT OUTCOMES  (serve‑side **and** return‑side)
    # ------------------------------------------------------------------
    def scrape_key_point_outcomes(self, url, debug: bool = False):
        """
        Parse “Key point outcomes” tables.
        TA now embeds SERVES and RETURNS tables in one blob (JS var *keypoints*).
        We split them, tag rows with side = “serve” / “return”, and return a
        unified list.
        """
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # decode inline-JS variables
        js_blob = "\n".join(s.string for s in soup.find_all("script") if s.string)
        blocks = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js_blob, re.S))

        html_blob = html.unescape(blocks.get("keypoints", ""))
        if not html_blob:  # fallback for very old pages
            out = []
            for side, var in (("serve", "keypoints"), ("return", "keypointsR")):
                h = html.unescape(blocks.get(var, ""))
                if h:
                    out.extend(self._parse_key_points_table(h, url, side, debug))
            return out

        # new-format: one blob, two <table> blocks
        soup_blob = BeautifulSoup(html_blob, "html.parser")
        out = []
        for tbl in soup_blob.find_all("table"):
            heading = tbl.find("th").get_text(strip=True).lower()
            side = "serve" if "serves" in heading else "return"
            out.extend(self._parse_key_points_table(str(tbl), url, side, debug))
        return out

    # ------------------------------------------------------------------
    # helper: parse one key‑points table (serve OR return)
    # ------------------------------------------------------------------
    def _parse_key_points_table(self, html_: str, url: str, side: str, debug: bool = False) -> list[dict]:
        """Internal helper used by *scrape_key_point_outcomes*."""
        match = self._parse_match_url(url)

        # map initials → full names, e.g. "MS" ➝ "Mayar Sherif"
        init = {"".join(w[0] for w in p.split()).upper(): p
                for p in (match["player1"], match["player2"])}

        tbl   = BeautifulSoup(html_, "html.parser").table
        rows  = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                 for tr in tbl.find_all("tr")]

        header = rows[0]
        data   = rows[1:]
        current_set = "Total"
        out: list[dict] = []

        for r in data:
            if not r or all(not x for x in r):
                continue

            # set‑separator rows such as "SET 1"
            if re.match(r"set\s+\d+", r[0].lower()):
                current_set = r[0].title()
                continue

            parts = r[0].split(None, 1)
            if len(parts) < 2:
                continue
            abbr, cat_raw = parts[0], parts[1]
            player = init.get(abbr.upper())
            if player is None:
                continue

            rec = {
                "match_id"        : f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date"            : match["Date"],
                "Tournament"      : match["tournament"],
                "player"          : player,
                "Player_canonical": self._normalize_player_name(player),
                "set"             : current_set,
                "context"         : re.sub(r"[^A-Za-z0-9]+", "_", cat_raw.lower()).strip("_"),
                "side"            : side  # serve | return
            }

            # parse remaining cells
            for h, v in zip(header[1:], r[1:]):
                key = re.sub(r"[%\s+/\\-]", "_", h.lower())
                key = re.sub(r"_+", "_", key).strip("_")
                if v.endswith("%"):
                    rec[key] = self._pct(v)
                elif "/" in v:
                    w, t = v.split("/")
                    rec[f"{key}_won"]   = self._to_int(w)
                    rec[f"{key}_total"] = self._to_int(t)
                else:
                    rec[key] = self._to_int(v)

            out.append(rec)

        if debug:
            print(f"=== KEY POINTS ({side}) ===")
            for row in out[:4]:
                print(row)

        return out

    # ------------------------------------------------------------------
    #  POINT OUTCOMES BY RALLY LENGTH
    # ------------------------------------------------------------------
    def scrape_rally_outcomes(self, url, debug: bool = False):
        """
        Extract “Point outcomes by rally length” table(s).
        Returns one record per player × rally-length bin.
        """
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # --- pull JS variable(s) that hold the rally‑outcome tables -----------
        js     = "\n".join(t.string for t in soup.find_all("script") if t.string)
        blocks = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))

        # Tennis‑Abstract sometimes names the blob “rallyoutcomes”, “rallyoutcomes1”, etc.
        raw = ""
        for k, v in blocks.items():
            key = k.lower()
            if key.startswith("rallyoutcomes") or key.startswith("ralloutcomes"):
                raw = v
                break

        html_ = html.unescape(raw)
        if not html_:
            return []

        match = self._parse_match_url(url)

        tbl = BeautifulSoup(html_, "html.parser").table
        rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")]

        header = [h.lower() for h in rows[0]]
        out = []

        for r in rows[1:]:
            if len(r) < 2 or not r[0]:
                continue
            # ------------------------------------------------------------------
            # First cell looks like e.g.  "Sv: 1-3 Shots"  or  "Rt: 4-6 Shots".
            # Capture rally‑side (Sv | Rt) and the numeric bin, drop the word 'Shots'.
            # ------------------------------------------------------------------
            label = r[0].strip()
            m = re.match(r'^(?:[A-Za-z]{2}\s+)?(Sv|Rt)\s*[:\s]\s*([0-9+<=\-]+)', label, flags=re.I)
            if not m:
                continue
            side_tag, bin_lbl = m.groups()
            side_tag = side_tag.lower()
            rally_side = "serve" if side_tag == "sv" else "return"
            bin_lbl = bin_lbl.replace("<=3", "<=3")  # preserve original label

            # Skip aggregate rows such as "All: 1‑3 Shots"
            if label.lower().startswith("all:"):
                continue

            # Identify the player via the two‑letter initials at the start
            abbr_match = re.match(r'^([A-Za-z]{2})', label)
            init = {"".join(w[0] for w in p.split()).upper(): p
                    for p in (match["player1"], match["player2"])}
            player = init.get(abbr_match.group(1).upper()) if abbr_match else None
            if player is None:
                continue  # ignore unrecognised rows

            rec = {
                "match_id": f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date": match["Date"],
                "Tournament": match["tournament"],
                "player": player,
                "Player_canonical": self._normalize_player_name(player),
                "rally_bin": bin_lbl,       # '<=3', '4-6', '7-9', '10+'
                "side": rally_side          # 'serve' | 'return'
            }

            for h, v in zip(header[1:], r[1:]):
                key = re.sub(r"[%\s+/\\:-]", "_", h.lower())
                key = re.sub(r"_+", "_", key).strip("_")
                rec[key] = self._pct(v) if v.endswith('%') else self._to_int(v)

            out.append(rec)

        if debug:
            print("=== RALLY OUTCOMES ===")
            for row in out[:6]:
                print(row)

        return out

    # ------------------------------------------------------------------
    #  POINT‑BY‑POINT DESCRIPTION (“pointlog” JS variable)
    # ------------------------------------------------------------------
    def scrape_pointlog(self, url, debug: bool = False):
        """
        Return the full point‑by‑point log.

        Each record contains:
          match_id, Date, Tournament, server, Player_canonical,
          set_score, game_score, point_score, description
        """
        import html, re, requests
        from bs4 import FeatureNotFound

        resp = requests.get(url, headers=self.headers)
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Inline‑JS variables
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)
        blocks  = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js_blob, re.S))
        # Tennis‑Abstract may use pointlog, pointlog1, pointlog2, …
        html_blob = ""
        for k, v in blocks.items():
            if k.lower().startswith("pointlog"):
                html_blob = html.unescape(v)
                break
        if not html_blob:
            return []

        table = BeautifulSoup(html_blob, "html.parser").find("table")
        if table is None:
            return []

        match = self._parse_match_url(url)
        out   = []

        # Skip header row
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 5 or not cells[0]:
                continue

            server, sets_sc, games_sc, pts_sc, desc = cells[:5]
            out.append({
                "match_id"        : f"{match['Date']}-{self._normalize_player_name(server)}",
                "Date"            : match["Date"],
                "Tournament"      : match["tournament"],
                "server"          : server,
                "Player_canonical": self._normalize_player_name(server),
                "set_score"       : sets_sc,
                "game_score"      : games_sc,
                "point_score"     : pts_sc,
                "description"     : desc,
            })

        if debug:
            print("=== POINT LOG ===")
            for row in out[:10]:
                print(row)

        return out

    def scrape_return_breakdown(self, url, debug: bool = False):
        """
        Extract return‑side breakdown categories.

        This is a thin wrapper around *scrape_serve_breakdown* which
        already parses both serve‑ and return‑side tables.  It filters
        the combined list and keeps only records where ``side == 'return'``.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print the first few parsed records.
        """
        records = self.scrape_serve_breakdown(url, debug=debug)
        # normalise guard ensures strings like 'Return' or stray spaces still match
        return [rec for rec in records if str(rec.get("side", "")).lower().strip() == "return"]

    def scrape_net_points(self, url, debug: bool = False):
        """
        Parse Net‑Points and Serve‑and‑Volley tables embedded in the inline JavaScript variables ``netpts1`` and ``netpts2``.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print a preview of the parsed records.

        Returns
        -------
        list[dict]
            One record per player × category row.
        """
        import re, html, requests
        # ───────── helpers ────────────────────────────────────────────────
        def _parse_count_pct(text: str) -> tuple[int, float | None]:
            """'11  (48%)' → (11, 0.48); returns (cnt, None) if % missing."""
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", text or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct

        def _to_float(text: str) -> float:
            """Extract leading numeric token as float; fallback 0.0."""
            m = re.search(r"[\d.]+", text or "")
            return float(m.group(0)) if m else 0.0
        # ──────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        # Pull net‑points blobs:  var netpts1 = '…';  var netpts2 = '…';
        matches = re.findall(r"var\s+(netpts\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta = self._parse_match_url(url)
        player_map = {"netpts1": meta.get("player1"), "netpts2": meta.get("player2")}

        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            bs = BeautifulSoup(html_blob, "html.parser")
            tables = bs.find_all("table")
            if not tables:
                continue
            # iterate through every embedded <table>
            for tbl in tables:
                hdr_cell = tbl.find("th")
                if not hdr_cell:
                    continue
                heading = hdr_cell.get_text(" ", strip=True).lower()

                # ─── NET POINTS ──────────────────────────────────────────
                if "net points" in heading:
                    for tr in tbl.find_all("tr")[1:]:
                        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                        if len(cells) < 9 or not cells[0]:
                            continue

                        cat = cells[0]
                        pts = self._to_int(cells[1])
                        won, won_pct = _parse_count_pct(cells[2])
                        if won_pct is None:
                            won_pct = won / pts if pts else 0.0
                        wnr           = self._to_int(cells[3])
                        ind_forced    = self._to_int(cells[4])
                        ufe           = self._to_int(cells[5])
                        passed        = self._to_int(cells[6])
                        psg_indfcd    = self._to_int(cells[7])
                        rally_len     = _to_float(cells[8])

                        out.append({
                            "player"             : player,
                            "Player_canonical"   : canon,
                            "category_group"     : "net_points",
                            "category"           : cat,
                            "pts"                : pts,
                            "won"                : won,
                            "won_pct"            : won_pct,
                            "net_winners"        : wnr,
                            "induced_forced_net" : ind_forced,
                            "ufe_net"            : ufe,
                            "passed_net"         : passed,
                            "pass_shot_induced"  : psg_indfcd,
                            "rally_len_avg"      : rally_len,
                        })
                # ─── SERVE‑AND‑VOLLEY ────────────────────────────────────
                elif "serve-and-volley" in heading:
                    for tr in tbl.find_all("tr")[1:]:
                        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                        if len(cells) < 12 or not cells[0]:
                            continue

                        cat = cells[0]
                        pts = self._to_int(cells[1])
                        won,  won_pct  = _parse_count_pct(cells[2])
                        if won_pct is None:
                            won_pct = won / pts if pts else 0.0
                        aces, ace_pct  = _parse_count_pct(cells[3])
                        if ace_pct is None:
                            ace_pct = aces / pts if pts else 0.0
                        unret, unret_pct = _parse_count_pct(cells[4])
                        if unret_pct is None:
                            unret_pct = unret / pts if pts else 0.0
                        retfcd, retfcd_pct = _parse_count_pct(cells[5])
                        if retfcd_pct is None:
                            retfcd_pct = retfcd / pts if pts else 0.0
                        wnr   = self._to_int(cells[6])
                        indf  = self._to_int(cells[7])
                        ufe   = self._to_int(cells[8])
                        passed= self._to_int(cells[9])
                        psg   = self._to_int(cells[10])
                        rally = _to_float(cells[11])

                        out.append({
                            "player"           : player,
                            "Player_canonical" : canon,
                            "category_group"   : "serve_and_volley",
                            "category"         : cat,
                            "pts"              : pts,
                            "won"              : won,
                            "won_pct"          : won_pct,
                            "aces"             : aces,
                            "aces_pct"         : ace_pct,
                            "unret"            : unret,
                            "unret_pct"        : unret_pct,
                            "retFcd"           : retfcd,
                            "retFcd_pct"       : retfcd_pct,
                            "net_winners"      : wnr,
                            "induced_forced_net": indf,
                            "ufe_net"          : ufe,
                            "passed_net"       : passed,
                            "pass_shot_induced": psg,
                            "rally_len_avg"    : rally,
                        })

        # ─── de‑duplicate --------------------------------------------------
        unique = {}
        for rec in out:
            key = (rec["player"], rec["category_group"], rec["category"])
            if key not in unique:
                unique[key] = rec
        # -------------------------------------------------------------------

        if debug:
            print("=== NET POINTS ===")
            for row in unique.values():
                print(row)

        return list(unique.values())

    def scrape_shot_types(self, url, debug: bool = False):
        """
        Parse shot‑type distribution tables embedded in inline‑JS variables
        “shots1” and “shots2”.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print a preview of the parsed rows.

        Returns
        -------
        list[dict]
            One record per player × shot‑type row.
        """
        import re, html, requests
        import numpy as np

        # ───────── helper ───────────────────────────────────────────────
        def _parse_count_pct(txt: str) -> tuple[int, float | None]:
            """
            Convert strings like '23  (48%)' to a tuple ``(23, 0.48)``.
            If the percentage is missing, returns ``(cnt, None)``.
            """
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", txt or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct
        # ────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        # grab JS blobs   var shots1 = '…';   var shots2 = '…';
        matches = re.findall(r"var\s+(shots\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta        = self._parse_match_url(url)
        player_map  = {"shots1": meta.get("player1"), "shots2": meta.get("player2")}
        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon  = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            table = BeautifulSoup(html_blob, "html.parser").find("table")
            if table is None:
                continue

            # header row
            headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
            hdr_norm = [re.sub(r"[%\s]+", "_", h.lower()).strip("_") for h in headers]

            # data rows
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not cells or not cells[0]:
                    continue

                rec = {
                    "player"           : player,
                    "Player_canonical" : canon,
                    "category"         : cells[0],
                }

                for h, v in zip(hdr_norm[1:], cells[1:]):
                    cnt, pct = _parse_count_pct(v)
                    rec[h] = cnt
                    rec[f"{h}_pct"] = pct if pct is not None else np.nan

                if cells[0].strip().lower() == "total":
                    rec["total_pct"] = 1.0

                out.append(rec)

        # de‑duplicate possible repeats
        unique = {}
        for rec in out:
            key = (rec["player"], rec["category"])
            unique[key] = rec

        if debug:
            for row in unique.values():
                print(row)

        return list(unique.values())

    def scrape_shot_direction(self, url, debug: bool = False):
        """
        Parse shot‑direction distribution tables embedded in inline‑JS variables
        ``shotdir1`` and ``shotdir2``.

        Each blob contains *two* tables:
          • a high‑level directional mix (Total / Forehand / Backhand / …)
          • a granular table with outcome columns for every shot‑direction string
            (e.g. “FH crosscourt”, “BH down middle”, …).

        Returns
        -------
        list[dict]
            One record per player × row.  ``category_group`` is either
            ``'direction_summary'`` (first table) or ``'direction_outcome'``
            (second table).  Percentages are returned as decimals (0 – 1).
        """
        import re, html, requests, numpy as np

        # ───────── helper ────────────────────────────────────────────────
        def _cnt_pct(txt: str) -> tuple[int, float | None]:
            """Convert strings like '48  (34%)' → (48, 0.34)."""
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", txt or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct
        # ─────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        #   var shotdir1 = '…';   var shotdir2 = '…';
        matches = re.findall(r"var\s+(shotdir\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta        = self._parse_match_url(url)
        player_map  = {"shotdir1": meta.get("player1"), "shotdir2": meta.get("player2")}

        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon   = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            bs        = BeautifulSoup(html_blob, "html.parser")
            tables    = bs.find_all("table")
            if not tables:
                continue

            # ─── 1) directional mix (counts + % of total) ───────────────────
            tbl1 = tables[0]
            hdr1 = [th.get_text(" ", strip=True) for th in tbl1.find_all("th")][1:]
            hdr1_norm = [
                re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
                for h in hdr1
            ]

            for tr in tbl1.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not cells or not cells[0]:
                    continue
                rec = {
                    "player"           : player,
                    "Player_canonical" : canon,
                    "category_group"   : "direction_summary",
                    "category"         : cells[0],
                }
                for h, val in zip(hdr1_norm, cells[1:]):
                    cnt, pct = _cnt_pct(val)
                    rec[h]        = cnt
                    rec[f"{h}_pct"] = pct if pct is not None else np.nan
                out.append(rec)

            # ─── 2) outcome table per shot‑direction string ─────────────────
            if len(tables) > 1:
                tbl2 = tables[1]
                hdr2 = [th.get_text(" ", strip=True) for th in tbl2.find_all("th")][1:]
                hdr2_norm = [
                    re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
                    for h in hdr2
                ]

                for tr in tbl2.find_all("tr")[1:]:
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                    if not cells or not cells[0]:
                        continue
                    # skip slice‑direction rows – not part of the 12 primary outcomes
                    if cells[0].strip().lower().startswith("slice"):
                        continue
                    rec = {
                        "player"           : player,
                        "Player_canonical" : canon,
                        "category_group"   : "direction_outcome",
                        "category"         : cells[0],
                    }

                    for h, val in zip(hdr2_norm, cells[1:]):
                        if "(" in val and "%" in val:          # count + %
                            cnt, pct = _cnt_pct(val)
                            rec[h]        = cnt
                            rec[f"{h}_pct"] = pct if pct is not None else np.nan
                        else:                                  # plain count
                            m = re.search(r"\d+", val)
                            rec[h] = int(m.group(0)) if m else 0
                    out.append(rec)

        # de‑duplicate (edge‑case pages sometimes repeat rows)
        unique = {}
        for r in out:
            key = (r["player"], r["category_group"], r["category"])
            unique[key] = r

        if debug:
            print("=== SHOT DIRECTION ===")
            for row in list(unique.values())[:8]:
                print(row)

        return list(unique.values())

    def test_extraction_completeness(self, url):
        """Test all available sections and validate data structure"""
        sections = self.debug_available_sections(url)

        results = {}
        for section_name in sections.keys():
            try:
                # Test each section extraction
                extracted_data = self._test_section_extraction(url, section_name)
                results[section_name] = len(extracted_data) > 0
            except Exception as e:
                results[section_name] = f"Error: {e}"

        return results

    # ---- helpers used by Stats‑Overview parser ------------------------
    def _pct(self, text: str) -> float:
        """
        Convert percentage‑like inputs to a decimal in the range [0, 1].

        Accepts
        -------
        • strings such as '54%' or '54.0 %'
        • numeric values (≤ 1 already decimal, > 1 treated as raw percent)
        • None / NaN → 0.0
        """
        import pandas as pd, math, re

        # Null / NaN
        if text is None or (isinstance(text, float) and math.isnan(text)) or (
            isinstance(text, (int, float)) and pd.isna(text)
        ):
            return 0.0

        # Numeric literal
        if isinstance(text, (int, float)):
            val = float(text)
            return val if 0 <= val <= 1 else val / 100.0

        # String input
        s = str(text).strip()
        if s.endswith("%"):
            s = s[:-1].strip()

        m = re.search(r"[\d.]+", s)
        if not m:
            return 0.0

        val = float(m.group(0))
        return val if 0 <= val <= 1 else val / 100.0

    def _int_before_slash(self, text: str) -> int:
        """Take '7/12' → 7   or '0/0' → 0."""
        m = re.match(r"(\d+)", text or "")
        return int(m.group(1)) if m else 0

    def _split_parenthesised(self, text: str) -> tuple[int, int, int]:
        """
        Parse cells like '15 (7/5)'.
        Returns (total, fh, bh).  Gracefully degrades to (tot, 0, 0)
        when the breakdown is missing.
        """
        m = re.match(r"(\d+)\s*\((\d+)/(\d+)\)", text or "")
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        n = re.match(r"(\d+)", text or "")
        total = int(n.group(1)) if n else 0
        return total, 0, 0

    def _extract_stats_overview_table(self, html: str) -> list[dict]:
        """
        Parse the 'Stats Overview' section and return one record per
        player × set.

        Expected columns:
          A%, DF%, 1stIn, 1st%, 2nd%, BPSaved, RPW%, Winners (FH/BH),
          UFE (FH/BH)

        Percentage fields are returned as decimals (e.g. 54 % → 0.54);
        BPSaved keeps only the *saved* count (numerator).
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return []

        rows = [
            [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]

        out = []
        current_set = "Total"
        # Skip header row (rows[0])
        for r in rows[1:]:
            if not r or all(not cell for cell in r):
                continue

            label = r[0]

            # Detect set separator rows such as 'SET 1', 'Set 2', …
            if re.match(r"set\s+\d+", label.lower()):
                current_set = label.title()
                continue

            # Data rows must have at least 10 columns
            if len(r) < 10:
                continue

            player = label
            winners_tot, winners_fh, winners_bh = self._split_parenthesised(r[8])
            ufe_tot, ufe_fh, ufe_bh = self._split_parenthesised(r[9])

            rec = {
                "player"           : player,
                "Player_canonical" : self._normalize_player_name(player),
                "set"              : current_set,
                "ace_pct"          : self._pct(r[1]),
                "df_pct"           : self._pct(r[2]),
                "first_in_pct"     : self._pct(r[3]),
                "first_won_pct"    : self._pct(r[4]),
                "second_won_pct"   : self._pct(r[5]),
                "bp_saved"         : self._int_before_slash(r[6]),
                "return_pts_won_pct": self._pct(r[7]),
                "winners"          : winners_tot,
                "winners_fh"       : winners_fh,
                "winners_bh"       : winners_bh,
                "unforced"         : ufe_tot,
                "unforced_fh"      : ufe_fh,
                "unforced_bh"      : ufe_bh,
            }

            out.append(rec)
        return out

# %%
# ──────────────────────────────────────────────────────────────────────────────
# INCREMENTAL UPDATE PIPELINE
#
# • Base dataset (Jeff / tennis‑data) is assumed frozen up to 2025‑06‑10.
# • This pipeline appends any matches whose actual match‑date is strictly after
#   2025‑06‑10 by pulling:
#     1.  daily match metadata from an external REST API
#     2.  detailed charting pages via Tennis‑Abstract scraping
# • Results are persisted in one Parquet file that grows monotonically.
# ──────────────────────────────────────────────────────────────────────────────
import os, datetime, requests, pandas as pd

DATA_DIR = "data"
BASE_CUTOFF_DATE = datetime.date(2025, 6, 10)   # last date in frozen dataset
JEFF_DB_PATH     = os.path.join(DATA_DIR, "jeff_tennis_data_until_20250610.parquet")
INCR_DB_PATH     = os.path.join(DATA_DIR, "results_incremental.parquet")


def _load_base_dataset() -> pd.DataFrame:
    """Return the immutable baseline dataset (≤ 2025‑06‑10)."""
    return pd.read_parquet(JEFF_DB_PATH)


def _load_incremental() -> pd.DataFrame:
    """Return the already‑scraped incremental dataset (if any)."""
    if os.path.exists(INCR_DB_PATH):
        return pd.read_parquet(INCR_DB_PATH)
    return pd.DataFrame()


def _save_incremental(df: pd.DataFrame) -> None:
    df.to_parquet(INCR_DB_PATH, index=False)


def _latest_recorded_date() -> datetime.date:
    """Last match‑date recorded across base + incremental parts."""
    incr = _load_incremental()
    if incr.empty or "Date" not in incr.columns:
        return BASE_CUTOFF_DATE
    return max(
        pd.to_datetime(incr["Date"], errors="coerce").dropna().dt.date.max(),
        BASE_CUTOFF_DATE,
    )


# ─── API helpers ──────────────────────────────────────────────────────────────
def fetch_api_matches(day: datetime.date) -> list[dict]:
    """Finished fixtures for given day."""
    return api_call(
        "get_fixtures",
        date_start=day.isoformat(),
        date_stop=day.isoformat(),
        timezone="UTC",
    )

def build_charting_url(api_row: dict) -> str:
    """
    Derive Tennis‑Abstract charting URL from one API row.
    The exact rule depends on the API payload; placeholder below.
    """
    return api_row.get("charting_url", "").strip()


# ─── fallback: pull Tennis‑Abstract charting index directly ───────────────
#
# Jeff Sackmann maintains a public CSV containing every charting match with a
# relative URL column “url” and a “date” column in YYYY‑MM‑DD format:
#
#   https://raw.githubusercontent.com/JeffSackmann/tennis_charting/master/charting_match_index.csv
#
# We hit that file once per run, cache it in‑memory, and expose a helper that
# returns the full “https://www.tennisabstract.com/charting/…html” URL list
# for a given *day*.
# -------------------------------------------------------------------------

import functools, io

CHARTING_INDEX_CSV = (
    "https://raw.githubusercontent.com/JeffSackmann/"
    "tennis_charting/master/charting_match_index.csv"
)

@functools.lru_cache(maxsize=1)
def _load_charting_index() -> pd.DataFrame:
    csv_bytes = requests.get(CHARTING_INDEX_CSV, timeout=30).content
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["full_url"] = (
        "https://www.tennisabstract.com/charting/" + df["url"].str.strip("/")
    )
    return df[["date", "full_url"]]

def charting_urls_for_day(day: datetime.date) -> list[str]:
    """Return list of TA charting URLs where match‑date == *day*."""
    df = _load_charting_index()
    return df.loc[df["date"] == day, "full_url"].tolist()


# ─── orchestrator ─────────────────────────────────────────────────────────────
def sync(date_from: datetime.date | None = None,
         date_to  : datetime.date | None = None) -> None:
    """
    Incrementally append all matches with match‑date ∈ [date_from, date_to].
    When *date_from* is omitted, starts with the first day **after** the latest
    recorded match; *date_to* defaults to today().
    """
    if date_from is None:
        date_from = _latest_recorded_date() + datetime.timedelta(days=1)
    if date_to is None:
        date_to = datetime.date.today()

    if date_from > date_to:
        print("No new days to process.")
        return

    scraper = TennisAbstractScraper()
    frames: list[pd.DataFrame] = []

    current = date_from
    while current <= date_to:
        # 1) API ingest -------------------------------------------------------
        try:
            api_rows = fetch_api_matches(current)
            if api_rows:
                frames.append(pd.DataFrame(api_rows))
        except Exception as exc:
            print(f"[WARN] API fetch failed {current}: {exc}")
            api_rows = []

        # 1b) direct Tennis‑Abstract index fallback -------------------------
        if not api_rows:
            for ta_url in charting_urls_for_day(current):
                api_rows.append({"charting_url": ta_url})

        # 2) Tennis‑Abstract scraping ----------------------------------------
        for row in api_rows:
            ta_url = build_charting_url(row)
            if not ta_url:
                continue
            try:
                frames.append(pd.DataFrame(scraper.scrape_shot_types(ta_url)))
                frames.append(pd.DataFrame(scraper.scrape_shot_direction(ta_url)))
                # ── add further scraper calls here as needed ──
            except Exception as exc:
                print(f"[WARN] TA scrape failed {ta_url}: {exc}")

        current += datetime.timedelta(days=1)

    if frames:
        increment = pd.concat(frames, ignore_index=True)
        existing  = _load_incremental()
        combined  = pd.concat([existing, increment], ignore_index=True)
        _save_incremental(combined)
        print(f"[OK] Appended {len(increment)} new rows (through {date_to}).")
    else:
        print("[INFO] Nothing new to append.")


# ─── optional CLI entry‑point ────────────────────────────────────────────────
# Run `python t3n11s.py --sync` to execute the incremental updater.
if __name__ == "__main__":
    import sys
    if "--sync" in sys.argv:
        sync()
# %%
# ------------------------------------------------------------------
# JEFF column whitelist
# ------------------------------------------------------------------
def _load_jeff_header_set(root: str | None = None) -> set[str]:
    """
    Build a superset of all column names found in Jeff Sackmann
    charting statistics CSVs.

    If *root* is omitted, the default path
        '~/Desktop/data/Jeff 6.14.25'
    is used.  Every file that matches the glob pattern
        'charting-*stats-*.csv'
    (men, women, any subtype) is read and its headers merged into
    a single set.  Whitespace is stripped from each column label.
    """
    import os, glob, pandas as pd

    if root is None:
        root = os.path.expanduser("~/Desktop/data/Jeff 6.14.25")

    headers: set[str] = set()
    patterns = (
        "charting-*stats-*.csv",
        "charting-*matches*.csv",
        "charting-*points-*.csv",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(root, pattern)):
            try:
                headers |= {h.strip() for h in pd.read_csv(path, nrows=0).columns}
            except Exception:
                # Skip unreadable or malformed files
                continue
    return headers

JEFF_HEADERS = _load_jeff_header_set()

# --------------------------------------------------------------
# Extra columns generated by Serve Basics/Direction parsing that
# are not present in the original Jeff CSV header set.
# --------------------------------------------------------------
SERVE_COLS = {
    "serve_type", "serve_pts", "serve_won", "aces", "unret", "fcdE", "le3w",
    "wide_pct", "body_pct", "t_pct",
    "dc_wide_pct", "dc_body_pct", "dc_t_pct",
    "ad_wide_pct", "ad_body_pct", "ad_t_pct",
    "net_pct", "wide_err_pct", "deep_err_pct", "w_d_pct",
    "footfault_pct", "unk_pct"
}

# Promote the new serve columns to first‑class status so the Jeff
# whitelist filter keeps them.
JEFF_HEADERS |= SERVE_COLS

# ------------------------------------------------------------------
#  Abbreviation glossary (site → canonical Jeff‑style field names)
#  This is a one‑time extension so downstream pipelines recognise
#  short labels such as “FcdE” without additional scrapers.
# ------------------------------------------------------------------
ABBREV_MAP = {
    # serve / key‑point abbreviations
    "FcdE": "forced_errors_induced",     # points where server induced a forced return error
    "RlyFcd": "rally_forced_errors",     # rally ended on forced error
    "RlyWnr": "rally_winners",
    "SvWnr": "serve_winners",            # unreturnables + forced return errors
    "UFE": "unforced_errors",
    "DF": "double_faults",
    # generic counters
    "PtsW": "points_won",
    "Pts": "points_total",
    "1stIn": "first_serves_in",
}

# Merge the canonical names into the Jeff header whitelist so every
# new column passes the filter stage exactly once.
JEFF_HEADERS |= set(ABBREV_MAP.values())

# ============================================================================
# FIXTURE‑FLATTENING HELPERS  (required by integrate_api_tennis_data)
# ============================================================================

def _fx_canonical_name(name: str) -> str:
    return normalize_name(name)

def _fx_winner(fx: dict) -> str:
    return fx["event_first_player"] if fx["event_winner"].startswith("First") else fx["event_second_player"]

def _fx_loser(fx: dict) -> str:
    return fx["event_second_player"] if _fx_winner(fx) == fx["event_first_player"] else fx["event_first_player"]

def _fx_parse_scores(scores: list[dict]) -> list[tuple[int, int]]:
    sets = []
    for s in scores:
        try:
            f, l = int(float(s["score_first"])), int(float(s["score_second"]))
        except Exception:
            f, l = np.nan, np.nan
        sets.append((f, l))
    return sets

def flatten_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    """
    Convert the nested API‑Tennis fixture payload into one wide row per match
    with basic score‑line stats and wide player‑statistics columns.
    """
    records = []
    for fx in fixtures:
        # ---  core match metadata  ---
        rec = {
            "event_key": int(fx["event_key"]),
            "date": pd.to_datetime(fx["event_date"]),
            "tournament": fx.get("tournament_name"),
            "round": fx.get("tournament_round", ""),
            "surface": fx.get("court_surface", "Hard"),
            "event_type": fx.get("event_type_type"),
            "season": fx.get("tournament_season"),
            "winner": _fx_winner(fx),
            "loser":  _fx_loser(fx),
        }
        rec["winner_id"] = int(fx["first_player_key"]
                               if rec["winner"] == fx["event_first_player"]
                               else fx["second_player_key"])
        rec["loser_id"]  = int(fx["second_player_key"]
                               if rec["winner"] == fx["event_first_player"]
                               else fx["first_player_key"])
        rec["winner_canonical"] = _fx_canonical_name(rec["winner"])
        rec["loser_canonical"]  = _fx_canonical_name(rec["loser"])
        rec["api_scores_raw"]   = json.dumps(fx.get("scores", []))
        rec["api_pointbypoint"] = len(fx.get("pointbypoint", [])) > 0
        rec["composite_id"] = build_composite_id(
            rec["date"].date(),
            normalize_tournament_name(rec["tournament"]),
            rec["winner_canonical"],
            rec["loser_canonical"],
        )

        # ---  set‑level metrics  ---
        sets = _fx_parse_scores(fx.get("scores", []))
        rec["sets_played"]      = len(sets)
        rec["sets_won_winner"]  = sum(f > l for f, l in sets)
        rec["games_winner"]     = sum(f for f, _ in sets)
        rec["games_loser"]      = sum(l for _, l in sets)
        rec["tb_played"]        = any(max(f, l) >= 7 for f, l in sets)

        # ---  wide statistics  ---
        stat_df = pd.DataFrame(fx.get("statistics", []))
        if not stat_df.empty:
            stat_df["stat_col"] = (
                stat_df["stat_name"]
                  .str.lower()
                  .str.replace("%", "pct")
                  .str.replace(" ", "_")
                + "_" + stat_df["stat_period"]
            )
            for pid, tag in [(fx["first_player_key"], "p1"),
                             (fx["second_player_key"], "p2")]:
                sub = (stat_df[stat_df["player_key"] == pid]
                       .drop_duplicates(subset=["stat_col"], keep="last"))
                wide = (sub
                        .pivot_table(index="player_key",
                                     columns="stat_col",
                                     values="stat_value",
                                     aggfunc="first"))
                if not wide.empty:
                    for col, val in wide.iloc[0].items():
                        rec[f"{tag}_{col}"] = val

        records.append(rec)

    df = pd.DataFrame(records)
    num_cols = [c for c in df.columns if c.startswith(("p1_", "p2_"))]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="ignore")
    return df

# ============================================================================
# API INTEGRATION FUNCTIONS
# ============================================================================

from datetime import date, timedelta

def integrate_api_tennis_data(historical_data, days_back: int = 3):
    """Integrate API-Tennis data for recent matches.

    Parameters
    ----------
    historical_data : DataFrame
        Existing combined dataset.
    days_back : int, default 3
        How many days before *today* to start pulling finished matches.
    """
    """Integrate API-Tennis data for recent matches"""
    start_date = date.today() - timedelta(days=days_back)
    print("Step 7: Integrating comprehensive API-Tennis data...")
    print("Integrating comprehensive API-Tennis data...")

    # Load static metadata once
    print("  Loading tournament and event type metadata...")
    tournaments_meta = get_tournaments_metadata()
    event_types_meta = get_event_types()
    print(f"  Loaded {len(tournaments_meta)} tournaments, {len(event_types_meta)} event types")

    # Ensure event_key column exists
    if "event_key" not in historical_data.columns:
        historical_data["event_key"] = pd.NA

    # Get existing API keys to avoid duplicates
    if "composite_id" not in historical_data.columns:
        historical_data["composite_id"] = pd.NA

    existing_keys = set()
    if len(historical_data) > 0:
        cutoff_data = historical_data[historical_data["date"] >= start_date]
        if len(cutoff_data) > 0:
            # Safely convert existing event_keys to integers
            for key in cutoff_data["event_key"].dropna():
                converted_key = safe_int_convert(key)
                if converted_key is not None:
                    existing_keys.add(converted_key)
    # ------------------------------------------------------------------
    # Ensure composite_id column and back-fill where missing
    # ------------------------------------------------------------------
    if "composite_id" not in historical_data.columns:
        historical_data["composite_id"] = pd.NA

    if historical_data['composite_id'].isna().any():
        historical_data.loc[historical_data['composite_id'].isna(), 'composite_id'] = (
            historical_data[historical_data['composite_id'].isna()]
            .apply(lambda r: build_composite_id(
                r['date'],
                r.get('tournament_canonical', normalize_tournament_name(r['Tournament'])),
                r.get('winner_canonical',  normalize_name(r['Winner'])),
                r.get('loser_canonical',   normalize_name(r['Loser']))
            ), axis=1)
        )

    print(f"Found {len(existing_keys)} existing API matches")

    # initialise Tennis‑Abstract scraper once per run
    scraper = TennisAbstractScraper()
    api_matches = []
    date_range = list(pd.date_range(start_date, date.today()))

    for d in date_range:
        day = d.date()
        try:
            fixtures = get_fixtures_for_date(day)
            if fixtures:
                print(f"  {day}: {len(fixtures)} finished matches")

            # Build a per‑day dataframe with rich stats for quick lookup
            fixtures_df = flatten_fixtures(fixtures)
            fixtures_df = fixtures_df.set_index("event_key")

            for fixture in fixtures:
                try:
                    event_key = safe_int_convert(fixture.get("event_key"))
                    if event_key is None:
                        continue

                    if event_key in existing_keys:
                    if event_key is None or event_key in existing_keys:
                        continue

                    # Basic match info
                    p1_name = fixture["event_first_player"]
                    p2_name = fixture["event_second_player"]
                    winner = p1_name if fixture.get("event_winner", "").startswith("First") else p2_name
                    loser = p2_name if winner == p1_name else p1_name

                    # Tournament metadata
                    tournament_key = fixture.get("tournament_key")
                    tournament_meta = tournaments_meta.get(tournament_key, {})

                    # Event type metadata
                    event_type_key = tournament_meta.get("event_type_key")
                    event_type_meta = event_types_meta.get(event_type_key, {})

                    # --- build winner / loser names first -----------------------------
                    winner = p1_name if fixture.get("event_winner", "").startswith("First") else p2_name
                    loser  = p2_name if winner == p1_name else p1_name

                    win_c  = normalize_name(winner)
                    los_c  = normalize_name(loser)
                    win_c = normalize_name(winner)
                    los_c = normalize_name(loser)
                    tour_c = normalize_tournament_name(fixture.get("tournament_name", "Unknown"))

                    match_record = {
                        "event_key":        event_key,
                        "Date":             pd.to_datetime(fixture["event_date"]),
                        "date":             pd.to_datetime(fixture["event_date"]).date(),
                        "Tournament":       fixture.get("tournament_name", "Unknown"),
                        "round":            fixture.get("tournament_round", ""),
                        "Surface":          fixture.get("court_surface", "Hard"),
                        "Winner":           winner,
                        "Loser":            loser,
                        "source_rank":      1,   # API source
                        "gender":           "W" if "wta" in fixture.get("event_type_type", "").lower() else "M",
                        "event_key": event_key,
                        "Date": pd.to_datetime(fixture["event_date"]),
                        "date": pd.to_datetime(fixture["event_date"]).date(),
                        "Tournament": fixture.get("tournament_name", "Unknown"),
                        "round": fixture.get("tournament_round", ""),
                        "Surface": fixture.get("court_surface", "Hard"),
                        "Winner": winner,
                        "Loser": loser,
                        "source_rank": 1,
                        "gender": "W" if "wta" in fixture.get("event_type_type", "").lower() else "M",
                        "winner_canonical": win_c,
                        "loser_canonical":  los_c,
                        "loser_canonical": los_c,
                        "tournament_canonical": tour_c,
                    }

                    # add composite id once dict exists
                    match_record["composite_id"] = build_composite_id(
                        match_record["date"],
                        tour_c,
                        win_c,
                        los_c,
                        match_record["date"], tour_c, win_c, los_c
                    )

                    # Extract embedded statistics from fixture
                    embedded_stats = extract_embedded_statistics(fixture)
                    match_record.update(embedded_stats)

                    # Merge the pre‑flattened wide statistics row for this match
                    if event_key in fixtures_df.index:
                        rich = fixtures_df.loc[event_key].to_dict()
                        for col, val in rich.items():
                            if col not in match_record:      # keep existing core fields
                                match_record[col] = val
                    # ─── Tennis‑Abstract scraping (11 sections) ───────────────
                    ta_url = None
                    for url in charting_urls_for_day(day):
                        url_lc = url.lower()
                        if win_c in url_lc and los_c in url_lc:
                            ta_url = url
                            break
                    if ta_url:
                        try:
                            ta_stats = {
                                "stats_overview":           scraper.scrape_stats_overview(ta_url),
                                "serve_influence":          scraper.scrape_serve_influence(ta_url),
                                "serve_breakdown":          scraper.scrape_serve_breakdown(ta_url),
                                "return_breakdown":         scraper.scrape_return_breakdown(ta_url),
                                "key_points":               scraper.scrape_key_point_outcomes(ta_url),
                                "rally_outcomes":           scraper.scrape_rally_outcomes(ta_url),
                                "net_points":               scraper.scrape_net_points(ta_url),
                                "shot_types":               scraper.scrape_shot_types(ta_url),
                                "shot_direction":           scraper.scrape_shot_direction(ta_url),
                                "serve_stats":              scraper.scrape_serve_statistics_overview(ta_url),
                                "pointlog":                 scraper.scrape_pointlog(ta_url),
                            }
                            # persist raw TA data (JSON) for downstream parsing
                            match_record["ta_url"]        = ta_url
                            match_record["ta_stats_json"] = json.dumps(ta_stats)
                        except Exception as exc:
                            print(f"      TA scrape failed {ta_url}: {exc}")

                    # Get odds (only for dates >= 2025-06-23)
                    odds1, odds2 = get_match_odds(event_key, day)

                    # Add normalized names
                    match_record["winner_canonical"] = normalize_name(winner)
                    match_record["loser_canonical"] = normalize_name(loser)
                    match_record["tournament_canonical"] = normalize_tournament_name(match_record["Tournament"])

                    # Extract embedded statistics from fixture
                    embedded_stats = extract_embedded_statistics(fixture)
                    match_record.update(embedded_stats)

                    # Add raw fixture data for later analysis
                    match_record["api_scores"] = json.dumps(fixture.get("scores", []))
                    match_record["api_pointbypoint_available"] = len(fixture.get("pointbypoint", [])) > 0

                    # Get odds (only for dates >= 2025-06-23)
                    odds1, odds2 = get_match_odds(event_key, day)
                    match_record["api_odds_home"] = odds1
                    match_record["api_odds_away"] = odds2

                    # Get rankings
                    league = "WTA" if match_record["gender"] == "W" else "ATP"
                    rankings = get_player_rankings(day, league)
                    p1_key = int(fixture.get("first_player_key", 0))
                    p2_key = int(fixture.get("second_player_key", 0))

                    if winner == p1_name:
                        match_record["WRank"] = rankings.get(p1_key, pd.NA)
                        match_record["LRank"] = rankings.get(p2_key, pd.NA)
                        match_record["winner_player_key"] = p1_key
                        match_record["loser_player_key"] = p2_key
                    else:
                        match_record["WRank"] = rankings.get(p2_key, pd.NA)
                        match_record["LRank"] = rankings.get(p1_key, pd.NA)
                        match_record["winner_player_key"] = p2_key
                        match_record["loser_player_key"] = p1_key

                    # Get H2H data
                    h2h_data = get_h2h_data(p1_key, p2_key)
                    match_record.update({f"h2h_{k}": v for k, v in h2h_data.items()})

                    # Add tournament metadata
                    match_record["tournament_key"] = tournament_key
                    match_record["event_type_key"] = event_type_key
                    match_record["event_type_type"] = fixture.get("event_type_type", "")
                    match_record["tournament_season"] = fixture.get("tournament_season", "")

                    api_matches.append(match_record)
                    existing_keys.add(event_key)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"    Error processing match {fixture.get('event_key', 'unknown')}: {e}")
                    continue

        except Exception as e:
            print(f"  Error processing day {day}: {e}")
            continue

    print(f"Collected {len(api_matches)} new API matches with full metadata")

    # Merge with historical data
    if api_matches:
        try:
            api_df = pd.DataFrame(api_matches)

            # Align columns
            for col in historical_data.columns:
                if col not in api_df.columns:
                    api_df[col] = pd.NA

            for col in api_df.columns:
                if col not in historical_data.columns:
                    historical_data[col] = pd.NA

            # Ensure column order matches
            api_df = api_df.reindex(columns=historical_data.columns, fill_value=pd.NA)

            # Concatenate
            combined_data = pd.concat([historical_data, api_df], ignore_index=True)

            # Add source_rank if missing
            if "source_rank" not in combined_data.columns:
                combined_data["source_rank"] = 2  # Default to tennis-data
                combined_data["source_rank"] = 2
            combined_data["source_rank"] = combined_data["source_rank"].fillna(2)

            # Deduplicate (keep API data over tennis-data when available)
            dedup_keys = ["event_key", "composite_id"]
            final_data = (
                combined_data
                .sort_values("source_rank")  # API=1, tennis-data=2
                .sort_values("source_rank")
                .drop_duplicates(subset=dedup_keys, keep="first")
                .reset_index(drop=True)
            )

            print(f"✓ Successfully integrated {len(api_df)} API matches")
            print(f"Final dataset: {len(final_data)} matches")

            # Show what we got
            if len(api_df) > 0:
                odds_count = api_df["api_odds_home"].notna().sum()
                h2h_count = api_df["h2h_matches"].notna().sum()
                rankings_count = api_df["WRank"].notna().sum()
                print(f"  - Matches with odds: {odds_count}")
                print(f"  - Matches with H2H data: {h2h_count}")
                print(f"  - Matches with rankings: {rankings_count}")

            return final_data

        except Exception as e:
            print(f"Error merging API data: {e}")
            return historical_data
    else:
        print("No new API data to merge")
        return historical_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ------------------------------------------------------------------
# Cache logic — flip REFRESH_CACHE to force regeneration
# ------------------------------------------------------------------
REFRESH_CACHE = True          # set False to reuse cached data

if REFRESH_CACHE:
    print("Refreshing cache …")
    historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(
        fast=False                         # use full dataset so API rows can merge
    )
    save_to_cache(historical_data, jeff_data, weighted_defaults)
    # Pull finished API‑Tennis matches from 10 June 2025 onward
    from datetime import date
    days_back = (date.today() - date(2025, 6, 10)).days
    historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
    save_to_cache(historical_data, jeff_data, weighted_defaults)
else:
    historical_data, jeff_data, weighted_defaults = load_from_cache()
    if historical_data is None:
        print("Cache miss – generating data …")
        historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(
            fast=False
        )
        save_to_cache(historical_data, jeff_data, weighted_defaults)
    else:
        print("✓ Data loaded from cache")

    # pull finished API‑Tennis matches from 10 June 2025 onward
    days_back = (date.today() - date(2025, 6, 10)).days
    historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
    save_to_cache(historical_data, jeff_data, weighted_defaults)

#%% md
# SIMULATION
#%%
# LAYER 1 ##
def extract_data_samples():
    # Jeff Sackmann data samples
    jeff_samples = {
        'matches': jeff_data['men']['matches'].head(3),
        'serve_basics': jeff_data['men']['serve_basics'].head(3),
        'overview': jeff_data['men']['overview'].head(3)
    }

    # Tennis-data samples
    tennis_samples = historical_data[
        ['Winner', 'Loser', 'WRank', 'LRank', 'PSW', 'PSL', 'Surface']
    ].head(3)

    return jeff_samples, tennis_samples

# Hold/break computation method verification
hold_break_computation = {
    'current_method': 'Jeff aggregated stats from overview dataset',
    'available_columns': ['serve_pts', 'first_in', 'first_won', 'second_won'],
    'computation_level': 'Per-player aggregate from charting data'
}

# Bayesian
def extract_priors_from_current_data(player_canonical, gender, surface):
    priors = {}

    # Layer 1: Elo approximation from rankings
    player_matches = historical_data[
        (historical_data['winner_canonical'] == player_canonical) |
        (historical_data['loser_canonical'] == player_canonical)
    ]

    if len(player_matches) > 0:
        # Ranking-based Elo estimation
        recent_rank = get_recent_rank(player_canonical, player_matches)
        elo_estimate = 2000 - (recent_rank * 5) if recent_rank else 1500

        # Jeff feature extraction
        jeff_features = extract_jeff_features(player_canonical, gender, jeff_data)

        priors = {
            'elo_estimate': elo_estimate,
            'serve_effectiveness': jeff_features.get('serve_pts', 0.6),
            'return_strength': jeff_features.get('return_pts_won', 0.3),
            'surface_factor': calculate_surface_adjustment(player_matches, surface)
        }

    return priors

# Time decay for recent form
def calculate_time_decayed_performance(player_matches, reference_date):
    player_matches['days_ago'] = (reference_date - player_matches['date']).dt.days

    # Exponential decay: recent matches weighted heavier
    weights = np.exp(-0.01 * player_matches['days_ago'])  # 1% daily decay

    weighted_performance = {
        'win_rate': np.average(player_matches['is_winner'], weights=weights),
        'games_won_rate': np.average(player_matches['games_won_pct'], weights=weights)
    }

    return weighted_performance
#%%
## TEST ##
import os, pickle, pandas as pd

CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
os.makedirs(CACHE_DIR, exist_ok=True)
HD_PATH   = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH  = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

if (os.path.exists(HD_PATH) and
    os.path.exists(JEFF_PATH) and
    os.path.exists(DEF_PATH)):
    print("Loading cached data …")
    historical_data = pd.read_parquet(HD_PATH)
    with open(JEFF_PATH, "rb") as fh:
        jeff_data = pickle.load(fh)
    with open(DEF_PATH, "rb") as fh:
        weighted_defaults = pickle.load(fh)
else:
    print("Cache miss – regenerating (one-time slow run).")
    combined_data, jeff_data, weighted_defaults = generate_comprehensive_historical_all_years()
    historical_data = combined_data
    historical_data.to_parquet(HD_PATH, index=False)
    with open(JEFF_PATH, "wb") as fh:
        pickle.dump(jeff_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(DEF_PATH, "wb") as fh:
        pickle.dump(weighted_defaults, fh, protocol=pickle.HIGHEST_PROTOCOL)

# "SIMULATION"
#%%

#%%
import pandas as pd
import numpy as np
from collections import defaultdict

def normalize_name_canonical(name):
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    return ' '.join(name.lower().split())

class BayesianTennisModel:
    def __init__(self):
        self.simulation_count = 10000
        self.jeff_data = jeff_data
        self.historical_data = historical_data

    def default_priors(self):
        return {
            'elo_mean': 1500,
            'elo_std': 200,
            'hold_prob': 0.65,
            'break_prob': 0.35,
            'surface': 'Hard',
            'form_factor': 1.0,
            'confidence': 0.1
        }

    def extract_refined_priors(self, player_canonical, gender, surface, reference_date):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].copy()

        if len(player_matches) == 0:
            return self.default_priors()

        surface_matches = player_matches[player_matches['Surface'] == surface]
        if len(surface_matches) < 5:
            surface_matches = player_matches

        recent_matches = surface_matches.tail(20).copy()
        recent_matches['days_ago'] = (pd.to_datetime(reference_date) - pd.to_datetime(recent_matches['Date'])).dt.days
        weights = np.exp(-0.05 * recent_matches['days_ago'])

        base_elo = self.get_player_weighted_elo(player_canonical, surface, reference_date)
        surface_factor = self.calculate_surface_adaptation(player_canonical, surface)
        elo_prior = base_elo * surface_factor

        jeff_features = extract_jeff_features(player_canonical, gender, self.jeff_data)

        serve_pts = jeff_features['serve_pts']
        serve_won = jeff_features['first_won'] + jeff_features['second_won']
        hold_prob = serve_won / serve_pts if serve_pts > 0 else 0.65

        return_pts = jeff_features['return_pts_won']
        total_return_pts = serve_pts
        break_prob = (1 - return_pts / total_return_pts) if total_return_pts > 0 else 0.35

        return {
            'elo_mean': elo_prior,
            'elo_std': 150,
            'hold_prob': min(0.95, max(0.3, hold_prob)),
            'break_prob': max(0.05, min(0.7, break_prob)),
            'surface': surface,
            'form_factor': self.calculate_form_spike(recent_matches, weights, player_canonical),
            'confidence': max(0.05, min(1.0, len(recent_matches) / 15))
        }

    def calculate_ranking_differential_odds(self, p1_ranking, p2_ranking):
        """Convert ranking differential to implied probability"""
        if p1_ranking == 0 or p2_ranking == 0:
            return 0.5

        ranking_diff = p2_ranking - p1_ranking

        if ranking_diff > 50:
            return 0.85
        elif ranking_diff > 20:
            return 0.75
        elif ranking_diff > 10:
            return 0.65
        elif ranking_diff > 0:
            return 0.55
        elif ranking_diff > -10:
            return 0.45
        elif ranking_diff > -20:
            return 0.35
        elif ranking_diff > -50:
            return 0.25
        else:
            return 0.15

    def calculate_upset_frequency(self, ranking_diff, surface, historical_data):
        """Calculate upset frequency by ranking differential and surface"""
        upset_matches = historical_data[
            ((historical_data['WRank'] - historical_data['LRank']) > ranking_diff) &
            (historical_data['Surface'] == surface)
        ]

        total_matches = historical_data[
            (abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)) &
            (historical_data['Surface'] == surface)
        ]

        if len(total_matches) < 10 and surface != 'fallback':
            return self.calculate_upset_frequency(ranking_diff, 'fallback', historical_data)

        if surface == 'fallback':
            upset_matches = historical_data[
                (historical_data['WRank'] - historical_data['LRank']) > ranking_diff
            ]
            total_matches = historical_data[
                abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)
            ]

        if len(total_matches) == 0:
            return 0.1

        upset_rate = len(upset_matches) / len(total_matches)
        return min(0.45, max(0.05, upset_rate))

    def calculate_surface_performance_ratio(self, player_canonical, surface, opponent_canonical, reference_date):
        """Calculate player's surface-specific performance vs opponent's baseline"""
        player_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
        ].tail(20)

        opponent_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == opponent_canonical) |
             (self.historical_data['loser_canonical'] == opponent_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
        ].tail(20)

        if len(player_surface_matches) < 3 or len(opponent_surface_matches) < 3:
            return 1.0

        player_wins = len(player_surface_matches[player_surface_matches['winner_canonical'] == player_canonical])
        opponent_wins = len(opponent_surface_matches[opponent_surface_matches['winner_canonical'] == opponent_canonical])

        player_ratio = player_wins / len(player_surface_matches)
        opponent_ratio = opponent_wins / len(opponent_surface_matches)

        return player_ratio / opponent_ratio if opponent_ratio > 0 else 1.0

    def run_simulation(self, p1_priors, p2_priors, iterations):
        return [self.simulate_match(p1_priors, p2_priors)]

    def predict_match_outcome(self, player1_canonical, player2_canonical, surface, gender, date):
        p1_priors = self.extract_refined_priors(player1_canonical, gender, surface, date)
        p2_priors = self.extract_refined_priors(player2_canonical, gender, surface, date)

        base_prob = self.run_simulation(p1_priors, p2_priors, 1000)[0]

        p1_rank = self.get_player_ranking(player1_canonical, date)
        p2_rank = self.get_player_ranking(player2_canonical, date)
        ranking_prob = self.calculate_ranking_differential_odds(p1_rank, p2_rank)

        ranking_diff = p1_rank - p2_rank
        upset_adjustment = self.calculate_upset_frequency(ranking_diff, surface, self.historical_data)

        surface_ratio = self.calculate_surface_performance_ratio(player1_canonical, surface, player2_canonical, date)

        calibrated_prob = (0.6 * base_prob + 0.25 * ranking_prob + 0.15 * surface_ratio) * (1 - upset_adjustment * 0.1)

        return max(0.05, min(0.95, calibrated_prob))

    def get_player_ranking(self, player_canonical, date):
        """Get player ranking at specific date"""
        date_obj = pd.to_datetime(date)

        player_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (pd.to_datetime(self.historical_data['Date']) <= date_obj)
        ].sort_values('Date', ascending=False)

        if len(player_matches) == 0:
            return 999

        latest_match = player_matches.iloc[0]

        if latest_match['winner_canonical'] == player_canonical:
            return latest_match.get('WRank', 999)
        else:
            return latest_match.get('LRank', 999)

    def calculate_match_probability(self, player1_canonical, player2_canonical, gender, surface, reference_date, best_of=3):
        player1_priors = self.extract_refined_priors(player1_canonical, gender, surface, reference_date)
        player2_priors = self.extract_refined_priors(player2_canonical, gender, surface, reference_date)

        probability = self.simulate_match(player1_priors, player2_priors, best_of)
        confidence = min(player1_priors['confidence'], player2_priors['confidence'])

        return {
            'player1_win_probability': probability,
            'player2_win_probability': 1 - probability,
            'confidence': confidence,
            'player1_priors': player1_priors,
            'player2_priors': player2_priors
        }

    def calculate_form_spike(self, recent_matches, weights, player_canonical):
        if len(recent_matches) == 0:
            return 1.0

        wins = (recent_matches['winner_canonical'] == player_canonical).astype(int)
        weighted_win_rate = np.average(wins, weights=weights)

        avg_opponent_rank = recent_matches['LRank'].fillna(recent_matches['WRank']).mean()
        player_rank = recent_matches['WRank'].fillna(recent_matches['LRank']).iloc[-1]

        if pd.notna(avg_opponent_rank) and pd.notna(player_rank):
            rank_diff = player_rank - avg_opponent_rank
            expected_win_rate = 1 / (1 + 10**(rank_diff/400))
            form_spike = min(2.0, weighted_win_rate / max(0.1, expected_win_rate))
        else:
            form_spike = 1.0

        return form_spike

    def get_player_weighted_elo(self, player_canonical, surface, reference_date):
        recent_match = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface)
        ].tail(1)

        if len(recent_match) > 0 and 'BlendScore' in recent_match.columns:
            blend_score = recent_match['BlendScore'].iloc[0]
            return 1500 + blend_score * 50

        any_surface_match = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].tail(1)

        if len(any_surface_match) > 0 and 'BlendScore' in any_surface_match.columns:
            return 1500 + any_surface_match['BlendScore'].iloc[0] * 200

        return 1500

    def calculate_surface_adaptation(self, player_canonical, target_surface):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].copy()

        if len(player_matches) < 10:
            return 1.0

        surface_matches = player_matches[player_matches['Surface'] == target_surface]
        if len(surface_matches) < 3:
            return 1.0

        surface_wins = (surface_matches['winner_canonical'] == player_canonical).sum()
        surface_win_rate = surface_wins / len(surface_matches)

        total_wins = (player_matches['winner_canonical'] == player_canonical).sum()
        baseline_win_rate = total_wins / len(player_matches)

        if baseline_win_rate == 0:
            return 1.0

        adaptation_ratio = surface_win_rate / baseline_win_rate
        return max(0.7, min(1.5, adaptation_ratio))

    def evaluate_predictions(self, test_data):
        """Evaluate model accuracy on test dataset"""
        correct = 0
        total = 0

        for _, match in test_data.iterrows():
            prob = self.predict_match_outcome(
                match['winner_canonical'],
                match['loser_canonical'],
                match['Surface'],
                match['gender'],
                match['Date']
            )

            predicted_winner = match['winner_canonical'] if prob > 0.5 else match['loser_canonical']
            actual_winner = match['winner_canonical']

            if predicted_winner == actual_winner:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

def convert_to_canonical(name):
    return normalize_name_canonical(name)

model = BayesianTennisModel()
#%%
## LAYER 2 ##
def apply_contextual_adjustments(self, priors, player_canonical, opponent_canonical, match_context):
    """Layer 2: Contextual Bayesian adjustments for fatigue, injury, motivation"""

    adjusted_priors = priors.copy()

    # Fatigue Index
    fatigue_penalty = self.calculate_fatigue_index(player_canonical, match_context['reference_date'])
    adjusted_priors['hold_prob'] *= (1 - fatigue_penalty * 0.15)  # Max 15% hold penalty
    adjusted_priors['elo_std'] *= (1 + fatigue_penalty * 0.3)    # Increase uncertainty

    # Injury Flag Adjustment
    injury_factor = self.get_injury_factor(player_canonical, match_context['reference_date'])
    adjusted_priors['hold_prob'] *= injury_factor
    adjusted_priors['break_prob'] *= (2 - injury_factor)  # Inverse relationship

    # Form Spike Sustainability
    form_sustainability = self.calculate_form_sustainability(player_canonical, match_context)
    if adjusted_priors['form_factor'] > 1.2:  # Hot streak detection
        sustainability_discount = 1 - ((adjusted_priors['form_factor'] - 1) * (1 - form_sustainability))
        adjusted_priors['hold_prob'] *= sustainability_discount
        adjusted_priors['elo_mean'] *= sustainability_discount

    # Opponent Quality Weighting
    opponent_elo = self.estimate_opponent_elo(opponent_canonical, match_context)
    elo_diff = adjusted_priors['elo_mean'] - opponent_elo
    quality_adjustment = 1 / (1 + np.exp(-elo_diff / 200))  # Sigmoid scaling
    adjusted_priors['break_prob'] *= quality_adjustment

    return adjusted_priors

def calculate_fatigue_index(self, player_canonical, reference_date):
    """Fatigue based on recent match load and recovery time"""
    recent_matches = self.get_recent_matches(player_canonical, reference_date, days=14)

    if len(recent_matches) == 0:
        return 0.0

    # Calculate cumulative fatigue
    fatigue_score = 0
    for _, match in recent_matches.iterrows():
        days_ago = (pd.to_datetime(reference_date) - pd.to_datetime(match['Date'])).days
        match_duration = match.get('minutes', 120)  # Default 2 hours

        # Exponential decay with match duration weighting
        fatigue_contribution = (match_duration / 60) * np.exp(-0.1 * days_ago)
        fatigue_score += fatigue_contribution

    return min(1.0, fatigue_score / 10)  # Normalize to 0-1

def get_injury_factor(self, player_canonical, reference_date):
    """Player-specific injury fragility scoring"""
    # Injury memory bank - replace with actual injury tracking
    injury_prone_players = {
        'nadal_r': 0.85,
        'murray_a': 0.80,
        'thiem_d': 0.75,
        'badosa_p': 0.70
    }

    base_factor = injury_prone_players.get(player_canonical, 0.95)

    # Check for recent retirement/walkover flags
    recent_retirements = self.check_recent_retirements(player_canonical, reference_date)
    if recent_retirements > 0:
        base_factor *= (0.8 ** recent_retirements)

    return max(0.5, base_factor)

def calculate_form_sustainability(self, player_canonical, match_context):
    """Form spike sustainability based on opponent quality and win quality"""
    recent_matches = self.get_recent_matches(player_canonical, match_context['reference_date'], days=21)

    if len(recent_matches) < 3:
        return 0.5

    # Quality-weighted recent performance
    quality_scores = []
    for _, match in recent_matches.iterrows():
        opponent_rank = match['LRank'] if match['winner_canonical'] == player_canonical else match['WRank']
        win_quality = 1 / (1 + opponent_rank / 100) if pd.notna(opponent_rank) else 0.5
        quality_scores.append(win_quality)

    avg_opponent_quality = np.mean(quality_scores)
    consistency = 1 - np.std(quality_scores)

    return min(1.0, avg_opponent_quality * consistency)

def estimate_opponent_elo(self, opponent_canonical, match_context):
    """Quick opponent Elo estimation for quality weighting"""
    opponent_priors = self.extract_refined_priors(
        opponent_canonical,
        match_context['gender'],
        match_context['surface'],
        match_context['reference_date']
    )
    return opponent_priors['elo_mean']

def get_recent_matches(self, player_canonical, reference_date, days=14):
    try:
        cutoff_date = pd.to_datetime(reference_date) - pd.Timedelta(days=days)

        player_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical))
        ].copy()

        if len(player_matches) == 0:
            return player_matches

        # Force string conversion then datetime to avoid mixed types
        player_matches['Date'] = pd.to_datetime(player_matches['Date'].astype(str), errors='coerce')
        player_matches = player_matches.dropna(subset=['Date'])
        player_matches = player_matches[player_matches['Date'] >= cutoff_date]

        return player_matches.sort_values('Date')
    except:
        # Return empty DataFrame on any error
        return pd.DataFrame()

def check_recent_retirements(self, player_canonical, reference_date):
    """Count recent retirements/walkovers - placeholder for actual retirement tracking"""
    # Implementation depends on your data structure for retirement flags
    return 0
#%%
# Tomorrow's slate
from datetime import date, timedelta
import numpy as np
#%%

#%%
import hashlib
from bs4 import BeautifulSoup
# ============================================================================
# TENNIS DATA PIPELINE
# ============================================================================
import numpy as np
import pandas as pd
import os
import requests
import json
import pickle
import shutil
import time
from datetime import datetime, date, timedelta
from pathlib import Path
import re
from unidecode import unidecode

# ============================================================================
# API CONFIGURATION AND CORE FUNCTIONS
# ============================================================================

# API-Tennis configuration
def api_call(method: str, **params):
    """Unified API call function with proper error handling"""
    try:
        response = requests.get(BASE, params={"method": method, "APIkey": API_KEY, **params}, timeout=10)
        response.raise_for_status()
        data = response.json()

        error_code = str(data.get("error", "0"))
        if error_code != "0":
            return []

        return data.get("result", [])
    except Exception as e:
        print(f"API call failed for {method}: {e}")
        return []

# ============================================================================
# NAME NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_name(name):
    """Normalize tennis player names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).replace('.', '').lower()
    parts = name.split()
    if len(parts) < 2:
        return name.replace(' ', '_')
    if len(parts[-1]) == 1:  # Last part is single letter (first initial)
        last_name = parts[-2]
        first_initial = parts[-1]
    else:  # Handle "First Lastname" format
        last_name = parts[-1]
        first_initial = parts[0][0] if parts[0] else ''
    return f"{last_name}_{first_initial}"

def normalize_jeff_name(name):
    """Normalize Jeff's player names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    parts = name.split()
    if len(parts) < 2:
        return name.replace(' ', '_')
    last_name = parts[-1]
    first_initial = parts[0][0] if parts[0] else ''
    return f"{last_name}_{first_initial}"

def normalize_tournament_name(name):
    """Normalize tournament names"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = name.replace('masters cup', 'masters')
    name = name.replace('atp finals', 'masters')
    name = name.replace('wta finals', 'masters')
    return name.strip()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_excel_data(file_path):
    """Load data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        if 'Date' not in df.columns:
            print(f"Warning: No Date column in {file_path}")
            return pd.DataFrame()
        print(f"Loaded {len(df)} matches from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def load_jeff_comprehensive_data():
    """Load all of Jeff's comprehensive tennis data"""
    base_path = os.path.expanduser("~/Desktop/data/Jeff 6.14.25")
    data = {'men': {}, 'women': {}}
    files = {
        'matches': 'charting-{}-matches.csv',
        'points_2020s': 'charting-{}-points-2020s.csv',
        'overview': 'charting-{}-stats-Overview.csv',
        'serve_basics': 'charting-{}-stats-ServeBasics.csv',
        'return_outcomes': 'charting-{}-stats-ReturnOutcomes.csv',
        'return_depth': 'charting-{}-stats-ReturnDepth.csv',
        'key_points_serve': 'charting-{}-stats-KeyPointsServe.csv',
        'key_points_return': 'charting-{}-stats-KeyPointsReturn.csv',
        'net_points': 'charting-{}-stats-NetPoints.csv',
        'rally': 'charting-{}-stats-Rally.csv',
        'serve_direction': 'charting-{}-stats-ServeDirection.csv',
        'serve_influence': 'charting-{}-stats-ServeInfluence.csv',
        'shot_direction': 'charting-{}-stats-ShotDirection.csv',
        'shot_dir_outcomes': 'charting-{}-stats-ShotDirOutcomes.csv',
        'shot_types': 'charting-{}-stats-ShotTypes.csv',
        'snv': 'charting-{}-stats-SnV.csv',
        'sv_break_split': 'charting-{}-stats-SvBreakSplit.csv',
        'sv_break_total': 'charting-{}-stats-SvBreakTotal.csv'
    }

    for gender in ['men', 'women']:
        gender_path = os.path.join(base_path, gender)
        if os.path.exists(gender_path):
            for key, filename_template in files.items():
                filename = filename_template.format('m' if gender == 'men' else 'w')
                file_path = os.path.join(gender_path, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, low_memory=False)
                    if 'player' in df.columns:
                        df['Player_canonical'] = df['player'].apply(normalize_jeff_name)
                    data[gender][key] = df
                    print(f"Loaded {gender}/{filename}: {len(df)} records")
    return data

def load_all_tennis_data():
    """Load tennis data from all years"""
    base_path = os.path.expanduser("~/Desktop/data")
    all_data = []

    for gender_name, gender_code in [("tennisdata_men", "M"), ("tennisdata_women", "W")]:
        gender_path = os.path.join(base_path, gender_name)
        if os.path.exists(gender_path):
            for year in range(2020, 2026):
                file_path = os.path.join(gender_path, f"{year}_{gender_code.lower()}.xlsx")
                if os.path.exists(file_path):
                    df = load_excel_data(file_path)
                    if not df.empty and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df['gender'] = gender_code
                        df['year'] = df['Date'].dt.year
                        all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# ============================================================================
# FEATURE EXTRACTION AND DEFAULTS
# ============================================================================

def get_fallback_defaults(gender_key):
    """Fallback defaults when no Jeff data available"""
    base_defaults = {
        'serve_pts': 80, 'aces': 6, 'double_faults': 3, 'first_serve_pct': 0.62,
        'first_serve_won': 35, 'second_serve_won': 16, 'break_points_saved': 4,
        'return_pts_won': 30, 'winners_total': 28, 'winners_fh': 16, 'winners_bh': 12,
        'unforced_errors': 28, 'unforced_fh': 16, 'unforced_bh': 12,
        'serve_wide_pct': 0.3, 'serve_t_pct': 0.4, 'serve_body_pct': 0.3,
        'return_deep_pct': 0.4, 'return_shallow_pct': 0.3, 'return_very_deep_pct': 0.2,
        'key_points_serve_won_pct': 0.6, 'key_points_aces_pct': 0.05, 'key_points_first_in_pct': 0.55,
        'key_points_return_won_pct': 0.35, 'key_points_return_winners': 0.02,
        'net_points_won_pct': 0.65, 'net_winners_pct': 0.3, 'passed_at_net_pct': 0.3,
        'rally_server_winners_pct': 0.15, 'rally_server_unforced_pct': 0.2,
        'rally_returner_winners_pct': 0.1, 'rally_returner_unforced_pct': 0.25,
        'shot_crosscourt_pct': 0.5, 'shot_down_line_pct': 0.25, 'shot_inside_out_pct': 0.15,
        'serve_volley_frequency': 0.02, 'serve_volley_success_pct': 0.6,
        'return_error_net_pct': 0.1, 'return_error_wide_pct': 0.05,
        'aggression_index': 0.5, 'consistency_index': 0.5, 'pressure_performance': 0.5, 'net_game_strength': 0.5
    }

    if gender_key == 'women':
        base_defaults.update({
            'serve_pts': 75, 'aces': 4, 'first_serve_pct': 0.60,
            'first_serve_won': 32, 'second_serve_won': 15,
            'serve_volley_frequency': 0.01, 'net_points_won_pct': 0.60
        })

    return base_defaults

import collections, pandas as pd          # add near other imports

def calculate_comprehensive_weighted_defaults(jeff_data: dict) -> dict:
    defaults = {"men": {}, "women": {}}
    skip = {"matches", "points_2020s"}

    for sex in ("men", "women"):
        sums, counts = collections.defaultdict(float), collections.defaultdict(int)

        for name, df in jeff_data.get(sex, {}).items():
            if name in skip or df is None or df.empty:
                continue
            num = df.select_dtypes(include=["number"])
            for col in num.columns:
                vals = pd.to_numeric(num[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                sums[col]   += vals.sum()
                counts[col] += len(vals)

        defaults[sex] = {c: sums[c] / counts[c] for c in sums}

    return defaults

def extract_comprehensive_jeff_features(player_canonical, gender, jeff_data, weighted_defaults=None):
    """Extract features from all Jeff datasets with Player_canonical checks"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data:
        return get_fallback_defaults(gender_key)

    if weighted_defaults and gender_key in weighted_defaults:
        features = weighted_defaults[gender_key].copy()
    else:
        features = get_fallback_defaults(gender_key)

    # Overview stats
    if 'overview' in jeff_data[gender_key]:
        overview_df = jeff_data[gender_key]['overview']
        if 'Player_canonical' in overview_df.columns:
            player_overview = overview_df[
                (overview_df['Player_canonical'] == player_canonical) &
                (overview_df['set'] == 'Total')
            ]

            if len(player_overview) > 0:
                latest = player_overview.iloc[-1]
                serve_pts = latest.get('serve_pts', 80)
                if serve_pts > 0:
                    features.update({
                        'serve_pts': float(serve_pts),
                        'aces': float(latest.get('aces', 0)),
                        'double_faults': float(latest.get('dfs', 0)),
                        'first_serve_pct': float(latest.get('first_in', 0)) / float(serve_pts) if serve_pts > 0 else 0.62,
                        'first_serve_won': float(latest.get('first_won', 0)),
                        'second_serve_won': float(latest.get('second_won', 0)),
                        'break_points_saved': float(latest.get('bp_saved', 0)),
                        'return_pts_won': float(latest.get('return_pts_won', 0)),
                        'winners_total': float(latest.get('winners', 0)),
                        'winners_fh': float(latest.get('winners_fh', 0)),
                        'winners_bh': float(latest.get('winners_bh', 0)),
                        'unforced_errors': float(latest.get('unforced', 0)),
                        'unforced_fh': float(latest.get('unforced_fh', 0)),
                        'unforced_bh': float(latest.get('unforced_bh', 0))
                    })

    return features

# ============================================================================
# API-TENNIS HELPER FUNCTIONS
# ============================================================================

def get_fixtures_for_date(target_date):
    """Get all fixtures for a specific date - includes embedded statistics"""
    try:
        fixtures = api_call("get_fixtures",
                           date_start=target_date.isoformat(),
                           date_stop=target_date.isoformat(),
                           timezone="UTC")

        finished_fixtures = [ev for ev in fixtures if ev.get("event_status") == "Finished"]
        return finished_fixtures
    except Exception as e:
        print(f"Error getting fixtures for {target_date}: {e}")
        return []


def extract_embedded_statistics(fixture):
    """Extract statistics from fixture data (no separate API call needed)"""
    stats = {}

    # Extract from scores data
    scores = fixture.get("scores", [])
    if scores:
        try:
            p1_sets = 0
            p2_sets = 0
            for s in scores:
                score_first = safe_int_convert(s.get("score_first", 0), 0)
                score_second = safe_int_convert(s.get("score_second", 0), 0)
                if score_first > score_second:
                    p1_sets += 1
                elif score_second > score_first:
                    p2_sets += 1

            stats["sets_won_p1"] = p1_sets
            stats["sets_won_p2"] = p2_sets
            stats["total_sets"] = len(scores)
        except Exception as e:
            # Don't fail the entire match for score parsing errors
            pass

    return stats


# ----------------------------------------------------------------------------
# API-TENNIS: Statistics parsing helper
# ----------------------------------------------------------------------------
def parse_match_statistics(fixture):
    """
    Parse the raw ``fixture["statistics"]`` list into per–player dictionaries.

    Parameters
    ----------
    fixture : dict
        A single fixture dictionary returned by the API.  We expect a key
        ``"statistics"`` that is a list of dictionaries, each containing at
        least ``"player_key"``, ``"type"``, and ``"value"``.

    Returns
    -------
    dict
        ``{player_key: {stat_name: value, ...}, ...}``
        where *stat_name* is the lower‑snake‑case version of the original
        ``type`` string and *value* is converted to ``float`` when possible.
    """
    stats_pair = {}

    for entry in fixture.get("statistics", []):
        p_key = safe_int_convert(entry.get("player_key"))
        if p_key is None:
            continue

        # Normalise statistic name: "First Serves In" -> "first_serves_in"
        stat_name = str(entry.get("type", "")).strip().lower().replace(" ", "_")
        if not stat_name:
            continue

        raw_val = entry.get("value")
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = raw_val  # leave as‑is if non‑numeric

        if p_key not in stats_pair:
            stats_pair[p_key] = {}
        stats_pair[p_key][stat_name] = val

    return stats_pair

def get_match_odds(match_key, date_check=None):
    """Get odds with proper error handling - only for dates >= 2025-06-23"""
    if date_check and date_check < date(2025, 6, 23):
        return (None, None)

    try:
        # Convert match_key to int for API call, but keep as string for lookup
        match_key_int = safe_int_convert(match_key)
        if match_key_int is None:
            return (None, None)

        odds_data = api_call("get_odds", match_key=match_key_int)
        if not odds_data or str(match_key_int) not in odds_data:
            return (None, None)

        match_odds = odds_data[str(match_key_int)]
        home_away = match_odds.get("Home/Away", {})

        # Get average odds across bookmakers
        home_odds = home_away.get("Home", {})
        away_odds = home_away.get("Away", {})

        if home_odds and away_odds:
            # Take first bookmaker's odds or average if multiple
            home_val = next(iter(home_odds.values())) if home_odds else None
            away_val = next(iter(away_odds.values())) if away_odds else None

            return (float(home_val) if home_val else None,
                   float(away_val) if away_val else None)

        return (None, None)
    except Exception as e:
        print(f"Error getting odds for match {match_key}: {e}")
        return (None, None)

def safe_int_convert(value, default=None):
    """Safely convert string/float to int, handling decimals and None values"""
    if value is None or value == "":
        return default
    try:
        # Convert to float first to handle decimal strings like "7.7", then to int
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default

def get_player_rankings(day, league="ATP"):
    """Get standings with proper caching and error handling"""
    tag = f"{league}_{day.isocalendar()[0]}_{day.isocalendar()[1]:02d}.pkl"
    cache_file = CACHE_API / tag

    if cache_file.exists():
        try:
            standings = pickle.loads(cache_file.read_bytes())
            if standings:  # Non-empty cache
                rankings = {}
                for r in standings:
                    player_key = safe_int_convert(r.get("player_key"))
                    place = safe_int_convert(r.get("place"))
                    if player_key is not None and place is not None:
                        rankings[player_key] = place
                return rankings
        except Exception as e:
            print(f"Cache read error for {tag}: {e}")

    # Correct parameter for API
    standings = api_call("get_standings", event_type=league.upper())

    try:
        cache_file.write_bytes(pickle.dumps(standings, 4))
    except Exception as e:
        print(f"Cache write error for {tag}: {e}")

    # Process standings with safe conversion
    rankings = {}
    for r in standings:
        player_key = safe_int_convert(r.get("player_key"))
        place = safe_int_convert(r.get("place"))
        if player_key is not None and place is not None:
            rankings[player_key] = place

    return rankings

def get_h2h_data(p1_key, p2_key):
    """Get head-to-head data with caching"""
    cache_file = CACHE_API / f"h2h_{p1_key}_{p2_key}.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        h2h_data = api_call("get_H2H", first_player_key=p1_key, second_player_key=p2_key)

        # Handle empty response or error
        if not h2h_data or not isinstance(h2h_data, list) or len(h2h_data) == 0:
            result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
        else:
            h2h_matches = h2h_data[0].get("H2H", []) if isinstance(h2h_data[0], dict) else []
            if not h2h_matches:
                result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
            else:
                p1_wins = sum(1 for m in h2h_matches if m.get("event_winner") == "First Player")
                p2_wins = len(h2h_matches) - p1_wins

                result = {
                    "h2h_matches": len(h2h_matches),
                    "p1_wins": p1_wins,
                    "p2_wins": p2_wins,
                    "p1_win_pct": p1_wins / len(h2h_matches) if h2h_matches else 0.5
                }

        cache_file.write_bytes(pickle.dumps(result, 4))
        return result

    except Exception as e:
        result = {"h2h_matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_pct": 0.5}
        cache_file.write_bytes(pickle.dumps(result, 4))
        return result

def get_tournaments_metadata():
    """Get tournament metadata (surface, level, location) - cached statically"""
    cache_file = CACHE_API / "tournaments.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        tournaments = api_call("get_tournaments")
        # Use safe conversion for tournament_key
        tournament_dict = {}
        for t in tournaments:
            tournament_key = safe_int_convert(t.get("tournament_key"))
            if tournament_key is not None:
                tournament_dict[str(tournament_key)] = t  # Keep as string for lookup
        cache_file.write_bytes(pickle.dumps(tournament_dict, 4))
        return tournament_dict
    except Exception as e:
        print(f"Error getting tournaments: {e}")
        return {}

def get_event_types():
    """Get event types metadata - cached statically"""
    cache_file = CACHE_API / "event_types.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        events = api_call("get_events")
        # Use safe conversion for event_type_key
        event_dict = {}
        for e in events:
            event_type_key = safe_int_convert(e.get("event_type_key"))
            if event_type_key is not None:
                event_dict[str(event_type_key)] = e  # Keep as string for lookup
        cache_file.write_bytes(pickle.dumps(event_dict, 4))
        return event_dict
    except Exception as e:
        print(f"Error getting event types: {e}")
        return {}

def get_player_profile(player_key):
    """Get player profile with career stats"""
    cache_file = CACHE_API / f"player_{player_key}.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    try:
        player_data = api_call("get_players", player_key=player_key)
        result = player_data[0] if player_data else {}
        cache_file.write_bytes(pickle.dumps(result, 4))
        return result
    except Exception as e:
        print(f"Error getting player {player_key}: {e}")
        return {}

# ------------------------------------------------------------------
# composite-key helper
# ------------------------------------------------------------------
def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """YYYYMMDD-tournament-player1-player2 (all lower-snake)"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"

# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

# 2.7 Data Generation and Cache Management
def generate_comprehensive_historical_data(fast=True, n_sample=500):
    """Generate comprehensive historical data with API integration"""
    print("=== STARTING DATA GENERATION ===")

    # Step 1: Load Jeff's data
    print("Step 1: Loading Jeff's comprehensive data...")
    try:
        jeff_data = load_jeff_comprehensive_data()
        if not jeff_data or ('men' not in jeff_data and 'women' not in jeff_data):
            print("ERROR: Jeff data loading failed")
            return pd.DataFrame(), {}, {}

        print(f"✓ Jeff data loaded successfully")
        print(f"  - Men's datasets: {len(jeff_data.get('men', {}))}")
        print(f"  - Women's datasets: {len(jeff_data.get('women', {}))}")

    except Exception as e:
        print(f"ERROR loading Jeff data: {e}")
        return pd.DataFrame(), {}, {}

    # Step 2: Calculate weighted defaults
    print("Step 2: Calculating weighted defaults...")
    try:
        weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)
        if not weighted_defaults:
            print("ERROR: Weighted defaults calculation failed")
            return pd.DataFrame(), jeff_data, {}

        print(f"✓ Weighted defaults calculated")
        print(f"  - Men's features: {len(weighted_defaults.get('men', {}))}")
        print(f"  - Women's features: {len(weighted_defaults.get('women', {}))}")

    except Exception as e:
        print(f"ERROR calculating weighted defaults: {e}")
        return pd.DataFrame(), jeff_data, {}

    # Step 3: Load tennis match data
    print("Step 3: Loading tennis match data...")
    try:
        tennis_data = load_all_tennis_data()
        if tennis_data.empty:
            print("ERROR: No tennis data loaded")
            return pd.DataFrame(), jeff_data, weighted_defaults

        print(f"✓ Tennis data loaded: {len(tennis_data)} matches")

        # Fast mode for testing
        if fast:
            total_rows = len(tennis_data)
            take = min(n_sample, total_rows)
            tennis_data = tennis_data.sample(take, random_state=1).reset_index(drop=True)
            print(f"[FAST MODE] Using sample of {take}/{total_rows} rows")

    except Exception as e:
        print(f"ERROR loading tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 4: Process tennis data
    print("Step 4: Processing tennis data...")
    try:
        # Normalize player names
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)
        tennis_data['tournament_canonical'] = tennis_data['Tournament'].apply(normalize_tournament_name)
        # --- fix: ensure Date columns exist before building composite_id ---
        tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
        tennis_data['date'] = tennis_data['Date'].dt.date

        # ------------------------------------------------------------------
        # Composite match identifier shared across data sources
        # ------------------------------------------------------------------
        tennis_data['composite_id'] = tennis_data.apply(
            lambda r: build_composite_id(
                r['date'],
                r['tournament_canonical'],
                r['winner_canonical'],
                r['loser_canonical']
            ),
            axis=1
                r['date'], r['tournament_canonical'], r['winner_canonical'], r['loser_canonical']
            ), axis=1
        )

        # Add odds data
        tennis_data['tennis_data_odds1'] = pd.to_numeric(tennis_data.get('PSW', 0), errors='coerce')
        tennis_data['tennis_data_odds2'] = pd.to_numeric(tennis_data.get('PSL', 0), errors='coerce')

        # Add ranking difference
        if 'WRank' in tennis_data.columns and 'LRank' in tennis_data.columns:
            tennis_data['rank_difference'] = abs(pd.to_numeric(tennis_data['WRank'], errors='coerce') -
                                                 pd.to_numeric(tennis_data['LRank'], errors='coerce'))

        print(f"✓ Tennis data processed")

    except Exception as e:
        print(f"ERROR processing tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 5: Adding Jeff feature columns...
    print("Step 5: Adding Jeff feature columns...")
    try:
        # Build feature list dynamically from the weighted defaults
        men_feats   = set(weighted_defaults.get('men', {}).keys())
        men_feats = set(weighted_defaults.get('men', {}).keys())
        women_feats = set(weighted_defaults.get('women', {}).keys())
        all_jeff_features = sorted(men_feats.union(women_feats))

        if not all_jeff_features:
            raise ValueError("No features available in weighted_defaults")

        # ------------------------------------------------------------------
        #  vectorised column creation – avoid fragmentation warnings
        # ------------------------------------------------------------------
        missing_cols_dict = {}

        for feat in all_jeff_features:
            w_col = f"winner_{feat}"
            l_col = f"loser_{feat}"

            if w_col not in tennis_data.columns:
                missing_cols_dict[w_col] = np.full(len(tennis_data), np.nan, dtype="float64")
            if l_col not in tennis_data.columns:
                missing_cols_dict[l_col] = np.full(len(tennis_data), np.nan, dtype="float64")

        if missing_cols_dict:
            tennis_data = pd.concat(
                [tennis_data, pd.DataFrame(missing_cols_dict, index=tennis_data.index)],
                axis=1
            )

        print(f"✓ Added/verified {len(all_jeff_features) * 2} feature columns")

    except Exception as e:
        print(f"ERROR adding feature columns: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 6: Extract Jeff features
    print("Step 6: Extracting Jeff features...")
    try:
        total_matches = len(tennis_data)
        matches_with_jeff_features = 0

        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                print(f"  Processing match {idx}/{total_matches}")

            try:
                gender = row['gender']

                # Only extract Jeff features for matches before cutoff
                if row['date'] <= date(2025, 6, 10):
                    winner_features = extract_comprehensive_jeff_features(
                        row['winner_canonical'], gender, jeff_data, weighted_defaults
                    )
                    loser_features = extract_comprehensive_jeff_features(
                        row['loser_canonical'], gender, jeff_data, weighted_defaults
                    )

                    # Assign features
                    for feature_name, feature_value in winner_features.items():
                        col_name = f'winner_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    for feature_name, feature_value in loser_features.items():
                        col_name = f'loser_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    if winner_features and loser_features:
                        matches_with_jeff_features += 1

            except Exception as e:
                if idx < 5:  # Only print first few errors
                if idx < 5:
                    print(f"  Warning: Error processing match {idx}: {e}")
                continue

        print(f"✓ Jeff features extracted for {matches_with_jeff_features}/{total_matches} matches")

    except Exception as e:
        print(f"ERROR extracting Jeff features: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print(f"=== DATA GENERATION COMPLETE ===")
    print(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

def save_to_cache(historical_data, jeff_data, weighted_defaults):
    """Save data to cache"""
    print("\n=== SAVING TO CACHE ===")
    # --- sanitize numeric columns that sometimes contain stray text ---
    numeric_cols = ["MaxW", "MaxL", "AvgW", "AvgL", "PSW", "PSL"]
    for col in numeric_cols:
        if col in historical_data.columns:
            historical_data[col] = pd.to_numeric(historical_data[col], errors="coerce")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # Save historical data
        historical_data.to_parquet(HD_PATH, index=False)
        print("✓ Historical data saved")

        # Save Jeff data
        with open(JEFF_PATH, "wb") as f:
            pickle.dump(jeff_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Jeff data saved")

        # Save weighted defaults
        with open(DEF_PATH, "wb") as f:
            pickle.dump(weighted_defaults, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Weighted defaults saved")

        return True
    except Exception as e:
        print(f"ERROR saving cache: {e}")
        return False


def load_from_cache():
    """Load data from cache if available"""
    if (os.path.exists(HD_PATH) and
        os.path.exists(JEFF_PATH) and
        os.path.exists(DEF_PATH)):
            os.path.exists(JEFF_PATH) and
            os.path.exists(DEF_PATH)):

        print("Loading from cache...")
        historical_data = pd.read_parquet(HD_PATH)

        with open(JEFF_PATH, "rb") as f:
            jeff_data = pickle.load(f)

        with open(DEF_PATH, "rb") as f:
            weighted_defaults = pickle.load(f)

        return historical_data, jeff_data, weighted_defaults
    else:
        return None, None, None

# %%
class TennisAbstractScraper:
    # ------------------------------------------------------------------
    # helper: the "serve" JS variable actually contains TWO tables
    # (Serve Basics + Direction).  Split them cleanly.
    # ------------------------------------------------------------------
    def _split_serve_tables(self, raw_html: str) -> tuple[str, str]:
        """Return `(basics_html, direction_html)` from the combined blob."""
        tables = BeautifulSoup(raw_html, "html.parser").find_all("table")
        if not tables:
            return "", ""
        basics_html = str(tables[0])
        direction_html = str(tables[1]) if len(tables) > 1 else ""
        return basics_html, direction_html
    # Column headers → Jeff keys for the “Serve Influence” table
    MAP_SERVE_INFL = {
        "1+ shots": "srv_neut_len1_pct",
        "2+": "srv_neut_len2_pct",
        "3+": "srv_neut_len3_pct",
        "4+": "srv_neut_len4_pct",
        "5+": "srv_neut_len5_pct",
        "6+": "srv_neut_len6_pct",
        "7+": "srv_neut_len7_pct",
        "8+": "srv_neut_len8_pct",
        "9+": "srv_neut_len9_pct",
        "10+": "srv_neut_len10_pct",
    }
    def __init__(self, jeff_headers: set[str] | None = None):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.jeff_headers = jeff_headers if jeff_headers is not None else JEFF_HEADERS

    # --- generic cell parser ------------------------------------------
    def _to_int(self, text):
        """
        Extract leading integer from a value such as '51  (52%)', '0', or a numeric/NaN.
        Returns 0 when the value is missing or no digit is found.
        """
        import pandas as pd
        # Handle None / NaN (including numpy.nan) early
        if text is None or pd.isna(text):
            return 0

        # Direct numeric input
        if isinstance(text, (int, float)):
            return int(text)

        # Fallback for string input
        m = re.search(r"\d+", str(text))
        return int(m.group(0)) if m else 0

    def _filter_jeff(self, rec: dict) -> dict:
        """
        Pass‑through filter: return the record unchanged.

        The caller now receives *all* scraped fields, not just the subset
        present in Jeff Sackmann’s original column set.  Down‑stream code
        should handle any extra keys as needed.
        """
        return rec

    # --- helper: canonicalise a player name ----------------------------
    def _normalize_player_name(self, name: str) -> str:
        import re, unicodedata
        name = unicodedata.normalize("NFKD", name).strip()
        name = re.sub(r"[\s\-]+", "_", name)
        return name.lower()

    # --- Jeff conversion placeholder ---------------------------------
    def _convert_to_jeff_format(self, raw_records: list[dict], match_meta: dict) -> list[dict]:
        """
        Translate raw parsed records into Jeff‑style keys.

        Current placeholder just merges match metadata onto every
        raw record and returns the list unchanged.

        Parameters
        ----------
        raw_records : list[dict]
            Data extracted from a stats table.
        match_meta : dict
            Output of `_parse_match_url` with Date, tournament, etc.

        Returns
        -------
        list[dict]
            Records augmented with basic match metadata.
        """
        if not raw_records:
            return []

        shared = match_meta.copy()
        out = []
        for rec in raw_records:
            merged = rec.copy()
            for k, v in shared.items():
                if k not in merged:
                    merged[k] = v
            out.append(merged)
        return out

    def _parse_match_url(self, url: str) -> dict:
        """
        Parse a Tennis‑Abstract charting URL for basic metadata.

        URL pattern (examples)
        ----------------------
        https://…/20250701-W-Wimbledon-R128-Mayar_Sherif-Mirra_Andreeva.html
        https://…/20240611-M-French_Open-SF-Nadal_R-Djokovic_N.html

        Returns
        -------
        dict with keys
          Date, gender, tournament, round, player1, player2
        """
        from urllib.parse import urlparse
        import os

        fname = os.path.basename(urlparse(url).path)
        if not fname.endswith(".html"):
            return {}

        parts = fname[:-5].split("-")              # drop ".html"
        if len(parts) < 6 or not parts[0].isdigit():
            return {}

        date_str = parts[0]
        gender   = parts[1]                        # "M" or "W"
        # The last three tokens are <round> <player1> <player2>
        round_tok = parts[-3]
        player1   = parts[-2].replace("_", " ")
        player2   = parts[-1].replace("_", " ")
        tournament = "-".join(parts[2:-3]).replace("_", " ")

        return {
            "Date"     : date_str,
            "gender"   : gender,
            "tournament": tournament,
            "round"    : round_tok,
            "player1"  : player1,
            "player2"  : player2,
        }

    # ------------------------------------------------------------------
    #  header scraper
    # ------------------------------------------------------------------
    def _extract_match_header(self, soup: BeautifulSoup) -> dict:
        """
        Extract header metadata displayed above the stats tables.

        Expected HTML structure:

          <h2>2025 Wimbledon R128: Mayar Sherif vs Mirra Andreeva</h2>
          <b>Mirra Andreeva d. Mayar Sherif 6-3 6-3</b>
        """
        head_tag = soup.find("h2")
        bold_tag = soup.find("b")
        if not head_tag:
            return {}

        # ---------- line 1 -------------------------------------------------
        head_text = head_tag.get_text(" ", strip=True)
        if ":" not in head_text or " vs " not in head_text.lower():
            return {}

        left, right = head_text.split(":", 1)
        year_tour_round = left.strip().split()
        if len(year_tour_round) < 3 or not year_tour_round[0].isdigit():
            return {}

        year   = year_tour_round[0]
        round_ = year_tour_round[-1]
        tournament = " ".join(year_tour_round[1:-1])

        p1, p2 = [p.strip() for p in re.split(r"\s+vs\s+", right, flags=re.I)]

        # ---------- line 2 -------------------------------------------------
        winner = loser = score = ""
        if bold_tag:
            bold_text = bold_tag.get_text(" ", strip=True)
            m = re.match(r"(.+?)\s+d\.\s+(.+?)\s+([\d\s\-–]+)$", bold_text, flags=re.I)
            if m:
                winner, loser, score = m.groups()

        return {
            "year"     : year,
            "tournament": tournament,
            "round"    : round_,
            "player1"  : p1,
            "player2"  : p2,
            "winner"   : winner.strip(),
            "loser"    : loser.strip(),
            "score"    : score.strip()
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Stats Overview entry point – now merges header metadata
    # ──────────────────────────────────────────────────────────────────────────
    def scrape_stats_overview(self, url):
        resp = requests.get(url, headers=self.headers)
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(resp.text, "html.parser")

        # header block
        header_meta = self._extract_match_header(soup)

        # embedded JS blocks
        scripts = [t.string for t in soup.find_all("script") if t.string]
        blocks  = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';",
                                  "\n".join(scripts), re.S))
        labels  = {sp.get_text(strip=True): sp["id"]
                   for sp in soup.select("span.rounds")}
        stats_html = html.unescape(blocks.get(labels.get("Stats Overview", ""), ""))

        records = self._extract_stats_overview_table(stats_html)

        # attach meta
        for r in records:
            r.update(header_meta)

        return records


    def scrape_serve_influence(self, url):
        resp  = requests.get(url, headers=self.headers)
        soup  = BeautifulSoup(resp.text, "lxml")

        js    = "\n".join(t.string for t in soup.find_all("script") if t.string)
        html_ = html.unescape(
                    dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))
                    .get("serveNeut", "")
                )
        if not html_:
            return []

        tbl   = BeautifulSoup(html_, "html.parser").table
        heads = [h.get_text(strip=True).rstrip('%').strip()
                 for h in tbl.tr.find_all(['th', 'td'])]

        out = []
        for tr in tbl.find_all("tr")[1:]:
            cells  = [c.get_text(strip=True) for c in tr.find_all("td")]
            if not cells or not cells[0]:
                continue
            rec = {"Player_canonical": self._normalize_player_name(cells[0])}
            for h, v in zip(heads[1:], cells[1:]):
                key = self.MAP_SERVE_INFL.get(h)         # now matches
                if key:
                    # handle missing or non‑numeric percentages
                    txt = v.strip()
                    if txt in {"", "-", "—"}:
                        rec[key] = np.nan
                    else:
                        try:
                            rec[key] = float(txt.rstrip("%")) / 100
                        except ValueError:
                            rec[key] = np.nan
            out.append(self._filter_jeff(rec))
        return out

    # ------------------------------------------------------------------
    # SERVE BASICS  +  SERVE DIRECTION
    # ------------------------------------------------------------------
    def scrape_serve_statistics_overview(self, url, debug=False):
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        js = "\n".join(t.string for t in soup.find_all("script") if t.string)
        blob = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))

        raw_html = html.unescape(blob.get("serve", ""))
        basics_html, direct_html = self._split_serve_tables(raw_html)
        if not basics_html:
            return []

        match = self._parse_match_url(url)
        init = {"".join(w[0] for w in p.split()).upper(): p
                for p in (match["player1"], match["player2"])}

        # ───── BASICS ─────────────────────────────────────────────────────
        tbl = BeautifulSoup(basics_html, "html.parser").table
        rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")][1:]

        if debug:
            print("=== BASICS RAW ROWS ===")
            for r in rows[:6]:
                print(r)

        out = []
        for r in rows:
            if len(r) < 10 or not r[0]:
                continue
            abbr, serve_type = r[0].split()  # 'MS', 'Total'
            player = init.get(abbr.upper())
            if player is None:
                continue
            pts, won, ace, unret, fcde, le3w, wide, body, t = r[1:10]
            out.append({
                "match_id": f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date": match["Date"],
                "Tournament": match["tournament"],
                "player": player,
                "Player_canonical": self._normalize_player_name(player),
                "set": "Total",
                "serve_type": serve_type,
                "serve_pts": int(pts),
                "serve_won": self._to_int(won),
                "aces": self._to_int(ace),
                "unret": self._to_int(unret),
                "fcdE": self._to_int(fcde),
                "le3w": self._to_int(le3w),
                "wide_pct": self._pct(wide),
                "body_pct": self._pct(body),
                "t_pct": self._pct(t),
            })

        # ───── DIRECTION ─────────────────────────────────────────────────
        if direct_html:
            tbl2 = BeautifulSoup(direct_html, "html.parser").table
            rows2 = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                     for tr in tbl2.find_all("tr")][1:]

            if debug:
                print("=== DIRECTION RAW ROWS ===")
                for r in rows2[:6]:
                    print(r)

            for r in rows2:
                if len(r) < 13 or not r[0]:
                    continue
                abbr, serve_type = r[0].split()
                player = init.get(abbr.upper())
                if player is None:
                    continue
                dcw, dcb, dct, adw, adb, adt, net, werr, derr, wd, foot, unk = r[1:13]
                rec = next(i for i in out
                           if i["player"] == player and i["serve_type"] == serve_type)
                rec.update({
                    "dc_wide_pct": self._pct(dcw),
                    "dc_body_pct": self._pct(dcb),
                    "dc_t_pct": self._pct(dct),
                    "ad_wide_pct": self._pct(adw),
                    "ad_body_pct": self._pct(adb),
                    "ad_t_pct": self._pct(adt),
                    "net_pct": self._pct(net),
                    "wide_err_pct": self._pct(werr),
                    "deep_err_pct": self._pct(derr),
                    "w_d_pct": self._pct(wd),
                    "footfault_pct": self._pct(foot),
                    "unk_pct": self._pct(unk),
                })

        return out

    def scrape_serve_breakdown(self, url, debug: bool = False):
        """
        Extract per‑player serve and return breakdown tables embedded in the
        inline‑JS variables “serve1”, “serve2”, “return1”, “return2”.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print the first few parsed rows.

        Returns
        -------
        list[dict]
            One record per player × category row.
        """
        import re, html, pandas as pd

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # gather all inline‑JS blocks
        scripts = [t.string for t in soup.find_all("script") if t.string]
        js_blob = "\n".join(scripts)

        # match serve1/serve2/return1/return2 blobs
        pattern = re.compile(r"var\s+((?:serve|return)[12])\s*=\s*'([\s\S]*?)';", re.S)
        matches = pattern.findall(js_blob)
        if not matches:
            return []

        meta = self._parse_match_url(url)
        player_map = {"1": meta["player1"], "2": meta["player2"]}

        records: list[dict] = []
        for var_name, raw_html in matches:
            side   = "serve" if var_name.startswith("serve") else "return"
            player = player_map[var_name[-1]]
            canon  = self._normalize_player_name(player)

            tables = pd.read_html(html.unescape(raw_html))
            # tables[0] → Totals; tables[1] → 1st/2nd breakdown
            for idx, tbl in enumerate(tables):
                is_total = idx == 0
                for _, row in tbl.iterrows():
                    cat = row.iloc[0]
                    import pandas as pd
                    if pd.isna(cat) or str(cat).strip() == "":
                        continue
                    if is_total:
                        total_pts   = self._to_int(row.iloc[1])
                        won         = self._to_int(row.iloc[2])
                        aces        = self._to_int(row.iloc[3])
                        unret       = self._to_int(row.iloc[4])
                        fcdE        = self._to_int(row.iloc[5])
                        le3w        = self._to_int(row.iloc[6])
                        canon       = self._normalize_player_name(player)

                        rec = {
                            "player": player,
                            "Player_canonical": canon,
                            "side": side,
                            "category": cat,
                            "total_pts": total_pts,
                            "won": won,
                            "won_pct": won / total_pts if total_pts else 0.0,
                            "aces": aces,
                            "aces_pct": aces / total_pts if total_pts else 0.0,
                            "unret": unret,
                            "unret_pct": unret / total_pts if total_pts else 0.0,
                            "fcdE": fcdE,
                            "fcdE_pct": fcdE / total_pts if total_pts else 0.0,
                            "le3w": le3w,
                            "le3w_pct": le3w / total_pts if total_pts else 0.0,
                        }
                        if side == "serve":
                            rec["first_in"] = self._to_int(row.iloc[7])
                            rec["double_faults"] = self._to_int(row.iloc[8])

                        records.append(rec)
                    else:
                        first_pts   = self._to_int(row.iloc[1])
                        first_won   = self._to_int(row.iloc[2])
                        aces_first  = self._to_int(row.iloc[3])
                        unret_first = self._to_int(row.iloc[4])
                        fcdE_first  = self._to_int(row.iloc[5])
                        le3w_first  = self._to_int(row.iloc[6])
                        second_pts  = self._to_int(row.iloc[7])
                        second_won  = self._to_int(row.iloc[8])
                        canon       = self._normalize_player_name(player)

                        rec = {
                            "player": player,
                            "Player_canonical": canon,
                            "side": side,
                            "category": cat,
                            "first_pts": first_pts,
                            "first_won": first_won,
                            "first_won_pct": first_won / first_pts if first_pts else 0.0,
                            "aces_first": aces_first,
                            "aces_first_pct": aces_first / first_pts if first_pts else 0.0,
                            "unret_first": unret_first,
                            "unret_first_pct": unret_first / first_pts if first_pts else 0.0,
                            "fcdE_first": fcdE_first,
                            "fcdE_first_pct": fcdE_first / first_pts if first_pts else 0.0,
                            "le3w_first": le3w_first,
                            "le3w_first_pct": le3w_first / first_pts if first_pts else 0.0,
                            "second_pts": second_pts,
                            "second_won": second_won,
                            "second_won_pct": second_won / second_pts if second_pts else 0.0,
                        }

                        records.append(rec)

        if debug:
            for rec in records[:8]:
                print(rec)

        return records

    # ------------------------------------------------------------------
    # KEY‑POINT OUTCOMES  (serve‑side **and** return‑side)
    # ------------------------------------------------------------------
    def scrape_key_point_outcomes(self, url, debug: bool = False):
        """
        Parse “Key point outcomes” tables.
        TA now embeds SERVES and RETURNS tables in one blob (JS var *keypoints*).
        We split them, tag rows with side = “serve” / “return”, and return a
        unified list.
        """
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # decode inline-JS variables
        js_blob = "\n".join(s.string for s in soup.find_all("script") if s.string)
        blocks = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js_blob, re.S))

        html_blob = html.unescape(blocks.get("keypoints", ""))
        if not html_blob:  # fallback for very old pages
            out = []
            for side, var in (("serve", "keypoints"), ("return", "keypointsR")):
                h = html.unescape(blocks.get(var, ""))
                if h:
                    out.extend(self._parse_key_points_table(h, url, side, debug))
            return out

        # new-format: one blob, two <table> blocks
        soup_blob = BeautifulSoup(html_blob, "html.parser")
        out = []
        for tbl in soup_blob.find_all("table"):
            heading = tbl.find("th").get_text(strip=True).lower()
            side = "serve" if "serves" in heading else "return"
            out.extend(self._parse_key_points_table(str(tbl), url, side, debug))
        return out

    # ------------------------------------------------------------------
    # helper: parse one key‑points table (serve OR return)
    # ------------------------------------------------------------------
    def _parse_key_points_table(self, html_: str, url: str, side: str, debug: bool = False) -> list[dict]:
        """Internal helper used by *scrape_key_point_outcomes*."""
        match = self._parse_match_url(url)

        # map initials → full names, e.g. "MS" ➝ "Mayar Sherif"
        init = {"".join(w[0] for w in p.split()).upper(): p
                for p in (match["player1"], match["player2"])}

        tbl   = BeautifulSoup(html_, "html.parser").table
        rows  = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                 for tr in tbl.find_all("tr")]

        header = rows[0]
        data   = rows[1:]
        current_set = "Total"
        out: list[dict] = []

        for r in data:
            if not r or all(not x for x in r):
                continue

            # set‑separator rows such as "SET 1"
            if re.match(r"set\s+\d+", r[0].lower()):
                current_set = r[0].title()
                continue

            parts = r[0].split(None, 1)
            if len(parts) < 2:
                continue
            abbr, cat_raw = parts[0], parts[1]
            player = init.get(abbr.upper())
            if player is None:
                continue

            rec = {
                "match_id"        : f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date"            : match["Date"],
                "Tournament"      : match["tournament"],
                "player"          : player,
                "Player_canonical": self._normalize_player_name(player),
                "set"             : current_set,
                "context"         : re.sub(r"[^A-Za-z0-9]+", "_", cat_raw.lower()).strip("_"),
                "side"            : side  # serve | return
            }

            # parse remaining cells
            for h, v in zip(header[1:], r[1:]):
                key = re.sub(r"[%\s+/\\-]", "_", h.lower())
                key = re.sub(r"_+", "_", key).strip("_")
                if v.endswith("%"):
                    rec[key] = self._pct(v)
                elif "/" in v:
                    w, t = v.split("/")
                    rec[f"{key}_won"]   = self._to_int(w)
                    rec[f"{key}_total"] = self._to_int(t)
                else:
                    rec[key] = self._to_int(v)

            out.append(rec)

        if debug:
            print(f"=== KEY POINTS ({side}) ===")
            for row in out[:4]:
                print(row)

        return out

    # ------------------------------------------------------------------
    #  POINT OUTCOMES BY RALLY LENGTH
    # ------------------------------------------------------------------
    def scrape_rally_outcomes(self, url, debug: bool = False):
        """
        Extract “Point outcomes by rally length” table(s).
        Returns one record per player × rally-length bin.
        """
        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")

        # --- pull JS variable(s) that hold the rally‑outcome tables -----------
        js     = "\n".join(t.string for t in soup.find_all("script") if t.string)
        blocks = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js, re.S))

        # Tennis‑Abstract sometimes names the blob “rallyoutcomes”, “rallyoutcomes1”, etc.
        raw = ""
        for k, v in blocks.items():
            key = k.lower()
            if key.startswith("rallyoutcomes") or key.startswith("ralloutcomes"):
                raw = v
                break

        html_ = html.unescape(raw)
        if not html_:
            return []

        match = self._parse_match_url(url)

        tbl = BeautifulSoup(html_, "html.parser").table
        rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")]

        header = [h.lower() for h in rows[0]]
        out = []

        for r in rows[1:]:
            if len(r) < 2 or not r[0]:
                continue
            # ------------------------------------------------------------------
            # First cell looks like e.g.  "Sv: 1-3 Shots"  or  "Rt: 4-6 Shots".
            # Capture rally‑side (Sv | Rt) and the numeric bin, drop the word 'Shots'.
            # ------------------------------------------------------------------
            label = r[0].strip()
            m = re.match(r'^(?:[A-Za-z]{2}\s+)?(Sv|Rt)\s*[:\s]\s*([0-9+<=\-]+)', label, flags=re.I)
            if not m:
                continue
            side_tag, bin_lbl = m.groups()
            side_tag = side_tag.lower()
            rally_side = "serve" if side_tag == "sv" else "return"
            bin_lbl = bin_lbl.replace("<=3", "<=3")  # preserve original label

            # Skip aggregate rows such as "All: 1‑3 Shots"
            if label.lower().startswith("all:"):
                continue

            # Identify the player via the two‑letter initials at the start
            abbr_match = re.match(r'^([A-Za-z]{2})', label)
            init = {"".join(w[0] for w in p.split()).upper(): p
                    for p in (match["player1"], match["player2"])}
            player = init.get(abbr_match.group(1).upper()) if abbr_match else None
            if player is None:
                continue  # ignore unrecognised rows

            rec = {
                "match_id": f"{match['Date']}-{self._normalize_player_name(player)}",
                "Date": match["Date"],
                "Tournament": match["tournament"],
                "player": player,
                "Player_canonical": self._normalize_player_name(player),
                "rally_bin": bin_lbl,       # '<=3', '4-6', '7-9', '10+'
                "side": rally_side          # 'serve' | 'return'
            }

            for h, v in zip(header[1:], r[1:]):
                key = re.sub(r"[%\s+/\\:-]", "_", h.lower())
                key = re.sub(r"_+", "_", key).strip("_")
                rec[key] = self._pct(v) if v.endswith('%') else self._to_int(v)

            out.append(rec)

        if debug:
            print("=== RALLY OUTCOMES ===")
            for row in out[:6]:
                print(row)

        return out

    # ------------------------------------------------------------------
    #  POINT‑BY‑POINT DESCRIPTION (“pointlog” JS variable)
    # ------------------------------------------------------------------
    def scrape_pointlog(self, url, debug: bool = False):
        """
        Return the full point‑by‑point log.

        Each record contains:
          match_id, Date, Tournament, server, Player_canonical,
          set_score, game_score, point_score, description
        """
        import html, re, requests
        from bs4 import FeatureNotFound

        resp = requests.get(url, headers=self.headers)
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Inline‑JS variables
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)
        blocks  = dict(re.findall(r"var\s+(\w+)\s*=\s*'([\s\S]*?)';", js_blob, re.S))
        # Tennis‑Abstract may use pointlog, pointlog1, pointlog2, …
        html_blob = ""
        for k, v in blocks.items():
            if k.lower().startswith("pointlog"):
                html_blob = html.unescape(v)
                break
        if not html_blob:
            return []

        table = BeautifulSoup(html_blob, "html.parser").find("table")
        if table is None:
            return []

        match = self._parse_match_url(url)
        out   = []

        # Skip header row
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 5 or not cells[0]:
                continue

            server, sets_sc, games_sc, pts_sc, desc = cells[:5]
            out.append({
                "match_id"        : f"{match['Date']}-{self._normalize_player_name(server)}",
                "Date"            : match["Date"],
                "Tournament"      : match["tournament"],
                "server"          : server,
                "Player_canonical": self._normalize_player_name(server),
                "set_score"       : sets_sc,
                "game_score"      : games_sc,
                "point_score"     : pts_sc,
                "description"     : desc,
            })

        if debug:
            print("=== POINT LOG ===")
            for row in out[:10]:
                print(row)

        return out

    def scrape_return_breakdown(self, url, debug: bool = False):
        """
        Extract return‑side breakdown categories.

        This is a thin wrapper around *scrape_serve_breakdown* which
        already parses both serve‑ and return‑side tables.  It filters
        the combined list and keeps only records where ``side == 'return'``.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print the first few parsed records.
        """
        records = self.scrape_serve_breakdown(url, debug=debug)
        # normalise guard ensures strings like 'Return' or stray spaces still match
        return [rec for rec in records if str(rec.get("side", "")).lower().strip() == "return"]

    def scrape_net_points(self, url, debug: bool = False):
        """
        Parse Net‑Points and Serve‑and‑Volley tables embedded in the inline JavaScript variables ``netpts1`` and ``netpts2``.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print a preview of the parsed records.

        Returns
        -------
        list[dict]
            One record per player × category row.
        """
        import re, html, requests
        # ───────── helpers ────────────────────────────────────────────────
        def _parse_count_pct(text: str) -> tuple[int, float | None]:
            """'11  (48%)' → (11, 0.48); returns (cnt, None) if % missing."""
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", text or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct

        def _to_float(text: str) -> float:
            """Extract leading numeric token as float; fallback 0.0."""
            m = re.search(r"[\d.]+", text or "")
            return float(m.group(0)) if m else 0.0
        # ──────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        # Pull net‑points blobs:  var netpts1 = '…';  var netpts2 = '…';
        matches = re.findall(r"var\s+(netpts\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta = self._parse_match_url(url)
        player_map = {"netpts1": meta.get("player1"), "netpts2": meta.get("player2")}

        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            bs = BeautifulSoup(html_blob, "html.parser")
            tables = bs.find_all("table")
            if not tables:
                continue
            # iterate through every embedded <table>
            for tbl in tables:
                hdr_cell = tbl.find("th")
                if not hdr_cell:
                    continue
                heading = hdr_cell.get_text(" ", strip=True).lower()

                # ─── NET POINTS ──────────────────────────────────────────
                if "net points" in heading:
                    for tr in tbl.find_all("tr")[1:]:
                        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                        if len(cells) < 9 or not cells[0]:
                            continue

                        cat = cells[0]
                        pts = self._to_int(cells[1])
                        won, won_pct = _parse_count_pct(cells[2])
                        if won_pct is None:
                            won_pct = won / pts if pts else 0.0
                        wnr           = self._to_int(cells[3])
                        ind_forced    = self._to_int(cells[4])
                        ufe           = self._to_int(cells[5])
                        passed        = self._to_int(cells[6])
                        psg_indfcd    = self._to_int(cells[7])
                        rally_len     = _to_float(cells[8])

                        out.append({
                            "player"             : player,
                            "Player_canonical"   : canon,
                            "category_group"     : "net_points",
                            "category"           : cat,
                            "pts"                : pts,
                            "won"                : won,
                            "won_pct"            : won_pct,
                            "net_winners"        : wnr,
                            "induced_forced_net" : ind_forced,
                            "ufe_net"            : ufe,
                            "passed_net"         : passed,
                            "pass_shot_induced"  : psg_indfcd,
                            "rally_len_avg"      : rally_len,
                        })
                # ─── SERVE‑AND‑VOLLEY ────────────────────────────────────
                elif "serve-and-volley" in heading:
                    for tr in tbl.find_all("tr")[1:]:
                        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                        if len(cells) < 12 or not cells[0]:
                            continue

                        cat = cells[0]
                        pts = self._to_int(cells[1])
                        won,  won_pct  = _parse_count_pct(cells[2])
                        if won_pct is None:
                            won_pct = won / pts if pts else 0.0
                        aces, ace_pct  = _parse_count_pct(cells[3])
                        if ace_pct is None:
                            ace_pct = aces / pts if pts else 0.0
                        unret, unret_pct = _parse_count_pct(cells[4])
                        if unret_pct is None:
                            unret_pct = unret / pts if pts else 0.0
                        retfcd, retfcd_pct = _parse_count_pct(cells[5])
                        if retfcd_pct is None:
                            retfcd_pct = retfcd / pts if pts else 0.0
                        wnr   = self._to_int(cells[6])
                        indf  = self._to_int(cells[7])
                        ufe   = self._to_int(cells[8])
                        passed= self._to_int(cells[9])
                        psg   = self._to_int(cells[10])
                        rally = _to_float(cells[11])

                        out.append({
                            "player"           : player,
                            "Player_canonical" : canon,
                            "category_group"   : "serve_and_volley",
                            "category"         : cat,
                            "pts"              : pts,
                            "won"              : won,
                            "won_pct"          : won_pct,
                            "aces"             : aces,
                            "aces_pct"         : ace_pct,
                            "unret"            : unret,
                            "unret_pct"        : unret_pct,
                            "retFcd"           : retfcd,
                            "retFcd_pct"       : retfcd_pct,
                            "net_winners"      : wnr,
                            "induced_forced_net": indf,
                            "ufe_net"          : ufe,
                            "passed_net"       : passed,
                            "pass_shot_induced": psg,
                            "rally_len_avg"    : rally,
                        })

        # ─── de‑duplicate --------------------------------------------------
        unique = {}
        for rec in out:
            key = (rec["player"], rec["category_group"], rec["category"])
            if key not in unique:
                unique[key] = rec
        # -------------------------------------------------------------------

        if debug:
            print("=== NET POINTS ===")
            for row in unique.values():
                print(row)

        return list(unique.values())

    def scrape_shot_types(self, url, debug: bool = False):
        """
        Parse shot‑type distribution tables embedded in inline‑JS variables
        “shots1” and “shots2”.

        Parameters
        ----------
        url : str
            Tennis‑Abstract charting URL.
        debug : bool, optional
            When True, print a preview of the parsed rows.

        Returns
        -------
        list[dict]
            One record per player × shot‑type row.
        """
        import re, html, requests
        import numpy as np

        # ───────── helper ───────────────────────────────────────────────
        def _parse_count_pct(txt: str) -> tuple[int, float | None]:
            """
            Convert strings like '23  (48%)' to a tuple ``(23, 0.48)``.
            If the percentage is missing, returns ``(cnt, None)``.
            """
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", txt or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct
        # ────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        # grab JS blobs   var shots1 = '…';   var shots2 = '…';
        matches = re.findall(r"var\s+(shots\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta        = self._parse_match_url(url)
        player_map  = {"shots1": meta.get("player1"), "shots2": meta.get("player2")}
        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon  = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            table = BeautifulSoup(html_blob, "html.parser").find("table")
            if table is None:
                continue

            # header row
            headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
            hdr_norm = [re.sub(r"[%\s]+", "_", h.lower()).strip("_") for h in headers]

            # data rows
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not cells or not cells[0]:
                    continue

                rec = {
                    "player"           : player,
                    "Player_canonical" : canon,
                    "category"         : cells[0],
                }

                for h, v in zip(hdr_norm[1:], cells[1:]):
                    cnt, pct = _parse_count_pct(v)
                    rec[h] = cnt
                    rec[f"{h}_pct"] = pct if pct is not None else np.nan

                if cells[0].strip().lower() == "total":
                    rec["total_pct"] = 1.0

                out.append(rec)

        # de‑duplicate possible repeats
        unique = {}
        for rec in out:
            key = (rec["player"], rec["category"])
            unique[key] = rec

        if debug:
            for row in unique.values():
                print(row)

        return list(unique.values())

    def scrape_shot_direction(self, url, debug: bool = False):
        """
        Parse shot‑direction distribution tables embedded in inline‑JS variables
        ``shotdir1`` and ``shotdir2``.

        Each blob contains *two* tables:
          • a high‑level directional mix (Total / Forehand / Backhand / …)
          • a granular table with outcome columns for every shot‑direction string
            (e.g. “FH crosscourt”, “BH down middle”, …).

        Returns
        -------
        list[dict]
            One record per player × row.  ``category_group`` is either
            ``'direction_summary'`` (first table) or ``'direction_outcome'``
            (second table).  Percentages are returned as decimals (0 – 1).
        """
        import re, html, requests, numpy as np

        # ───────── helper ────────────────────────────────────────────────
        def _cnt_pct(txt: str) -> tuple[int, float | None]:
            """Convert strings like '48  (34%)' → (48, 0.34)."""
            m = re.match(r"\s*(\d+)(?:\s*\(([\d.]+)%\))?", txt or "")
            cnt = int(m.group(1)) if m else 0
            pct = float(m.group(2)) / 100 if m and m.group(2) else None
            return cnt, pct
        # ─────────────────────────────────────────────────────────────────

        resp = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(resp.text, "lxml")
        js_blob = "\n".join(tag.string for tag in soup.find_all("script") if tag.string)

        #   var shotdir1 = '…';   var shotdir2 = '…';
        matches = re.findall(r"var\s+(shotdir\d+)\s*=\s*'([\s\S]*?)';", js_blob, re.S)
        if not matches:
            return []

        meta        = self._parse_match_url(url)
        player_map  = {"shotdir1": meta.get("player1"), "shotdir2": meta.get("player2")}

        out: list[dict] = []

        for var_name, raw_html in matches:
            player = player_map.get(var_name)
            if not player:
                continue
            canon   = self._normalize_player_name(player)

            html_blob = html.unescape(raw_html)
            bs        = BeautifulSoup(html_blob, "html.parser")
            tables    = bs.find_all("table")
            if not tables:
                continue

            # ─── 1) directional mix (counts + % of total) ───────────────────
            tbl1 = tables[0]
            hdr1 = [th.get_text(" ", strip=True) for th in tbl1.find_all("th")][1:]
            hdr1_norm = [
                re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
                for h in hdr1
            ]

            for tr in tbl1.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not cells or not cells[0]:
                    continue
                rec = {
                    "player"           : player,
                    "Player_canonical" : canon,
                    "category_group"   : "direction_summary",
                    "category"         : cells[0],
                }
                for h, val in zip(hdr1_norm, cells[1:]):
                    cnt, pct = _cnt_pct(val)
                    rec[h]        = cnt
                    rec[f"{h}_pct"] = pct if pct is not None else np.nan
                out.append(rec)

            # ─── 2) outcome table per shot‑direction string ─────────────────
            if len(tables) > 1:
                tbl2 = tables[1]
                hdr2 = [th.get_text(" ", strip=True) for th in tbl2.find_all("th")][1:]
                hdr2_norm = [
                    re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
                    for h in hdr2
                ]

                for tr in tbl2.find_all("tr")[1:]:
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                    if not cells or not cells[0]:
                        continue
                    # skip slice‑direction rows – not part of the 12 primary outcomes
                    if cells[0].strip().lower().startswith("slice"):
                        continue
                    rec = {
                        "player"           : player,
                        "Player_canonical" : canon,
                        "category_group"   : "direction_outcome",
                        "category"         : cells[0],
                    }

                    for h, val in zip(hdr2_norm, cells[1:]):
                        if "(" in val and "%" in val:          # count + %
                            cnt, pct = _cnt_pct(val)
                            rec[h]        = cnt
                            rec[f"{h}_pct"] = pct if pct is not None else np.nan
                        else:                                  # plain count
                            m = re.search(r"\d+", val)
                            rec[h] = int(m.group(0)) if m else 0
                    out.append(rec)

        # de‑duplicate (edge‑case pages sometimes repeat rows)
        unique = {}
        for r in out:
            key = (r["player"], r["category_group"], r["category"])
            unique[key] = r

        if debug:
            print("=== SHOT DIRECTION ===")
            for row in list(unique.values())[:8]:
                print(row)

        return list(unique.values())

    def test_extraction_completeness(self, url):
        """Test all available sections and validate data structure"""
        sections = self.debug_available_sections(url)

        results = {}
        for section_name in sections.keys():
            try:
                # Test each section extraction
                extracted_data = self._test_section_extraction(url, section_name)
                results[section_name] = len(extracted_data) > 0
            except Exception as e:
                results[section_name] = f"Error: {e}"

        return results

    # ---- helpers used by Stats‑Overview parser ------------------------
    def _pct(self, text: str) -> float:
        """
        Convert percentage‑like inputs to a decimal in the range [0, 1].

        Accepts
        -------
        • strings such as '54%' or '54.0 %'
        • numeric values (≤ 1 already decimal, > 1 treated as raw percent)
        • None / NaN → 0.0
        """
        import pandas as pd, math, re

        # Null / NaN
        if text is None or (isinstance(text, float) and math.isnan(text)) or (
            isinstance(text, (int, float)) and pd.isna(text)
        ):
            return 0.0

        # Numeric literal
        if isinstance(text, (int, float)):
            val = float(text)
            return val if 0 <= val <= 1 else val / 100.0

        # String input
        s = str(text).strip()
        if s.endswith("%"):
            s = s[:-1].strip()

        m = re.search(r"[\d.]+", s)
        if not m:
            return 0.0

        val = float(m.group(0))
        return val if 0 <= val <= 1 else val / 100.0

    def _int_before_slash(self, text: str) -> int:
        """Take '7/12' → 7   or '0/0' → 0."""
        m = re.match(r"(\d+)", text or "")
        return int(m.group(1)) if m else 0

    def _split_parenthesised(self, text: str) -> tuple[int, int, int]:
        """
        Parse cells like '15 (7/5)'.
        Returns (total, fh, bh).  Gracefully degrades to (tot, 0, 0)
        when the breakdown is missing.
        """
        m = re.match(r"(\d+)\s*\((\d+)/(\d+)\)", text or "")
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        n = re.match(r"(\d+)", text or "")
        total = int(n.group(1)) if n else 0
        return total, 0, 0

    def _extract_stats_overview_table(self, html: str) -> list[dict]:
        """
        Parse the 'Stats Overview' section and return one record per
        player × set.

        Expected columns:
          A%, DF%, 1stIn, 1st%, 2nd%, BPSaved, RPW%, Winners (FH/BH),
          UFE (FH/BH)

        Percentage fields are returned as decimals (e.g. 54 % → 0.54);
        BPSaved keeps only the *saved* count (numerator).
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return []

        rows = [
            [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]

        out = []
        current_set = "Total"
        # Skip header row (rows[0])
        for r in rows[1:]:
            if not r or all(not cell for cell in r):
                continue

            label = r[0]

            # Detect set separator rows such as 'SET 1', 'Set 2', …
            if re.match(r"set\s+\d+", label.lower()):
                current_set = label.title()
                continue

            # Data rows must have at least 10 columns
            if len(r) < 10:
                continue

            player = label
            winners_tot, winners_fh, winners_bh = self._split_parenthesised(r[8])
            ufe_tot, ufe_fh, ufe_bh = self._split_parenthesised(r[9])

            rec = {
                "player"           : player,
                "Player_canonical" : self._normalize_player_name(player),
                "set"              : current_set,
                "ace_pct"          : self._pct(r[1]),
                "df_pct"           : self._pct(r[2]),
                "first_in_pct"     : self._pct(r[3]),
                "first_won_pct"    : self._pct(r[4]),
                "second_won_pct"   : self._pct(r[5]),
                "bp_saved"         : self._int_before_slash(r[6]),
                "return_pts_won_pct": self._pct(r[7]),
                "winners"          : winners_tot,
                "winners_fh"       : winners_fh,
                "winners_bh"       : winners_bh,
                "unforced"         : ufe_tot,
                "unforced_fh"      : ufe_fh,
                "unforced_bh"      : ufe_bh,
            }

            out.append(rec)
        return out

# %%
# ──────────────────────────────────────────────────────────────────────────────
# INCREMENTAL UPDATE PIPELINE
#
# • Base dataset (Jeff / tennis‑data) is assumed frozen up to 2025‑06‑10.
# • This pipeline appends any matches whose actual match‑date is strictly after
#   2025‑06‑10 by pulling:
#     1.  daily match metadata from an external REST API
#     2.  detailed charting pages via Tennis‑Abstract scraping
# • Results are persisted in one Parquet file that grows monotonically.
# ──────────────────────────────────────────────────────────────────────────────
import os, datetime, requests, pandas as pd

DATA_DIR = "data"
BASE_CUTOFF_DATE = datetime.date(2025, 6, 10)   # last date in frozen dataset
JEFF_DB_PATH     = os.path.join(DATA_DIR, "jeff_tennis_data_until_20250610.parquet")
INCR_DB_PATH     = os.path.join(DATA_DIR, "results_incremental.parquet")


def _load_base_dataset() -> pd.DataFrame:
    """Return the immutable baseline dataset (≤ 2025‑06‑10)."""
    return pd.read_parquet(JEFF_DB_PATH)


def _load_incremental() -> pd.DataFrame:
    """Return the already‑scraped incremental dataset (if any)."""
    if os.path.exists(INCR_DB_PATH):
        return pd.read_parquet(INCR_DB_PATH)
    return pd.DataFrame()


def _save_incremental(df: pd.DataFrame) -> None:
    df.to_parquet(INCR_DB_PATH, index=False)


def _latest_recorded_date() -> datetime.date:
    """Last match‑date recorded across base + incremental parts."""
    incr = _load_incremental()
    if incr.empty or "Date" not in incr.columns:
        return BASE_CUTOFF_DATE
    return max(
        pd.to_datetime(incr["Date"], errors="coerce").dropna().dt.date.max(),
        BASE_CUTOFF_DATE,
    )


# ─── API helpers ─────────────────────────────────────────────────────────────
def fetch_api_matches(day: datetime.date) -> list[dict]:
    """Return a list of match dictionaries for *day* via external API."""
    url = f"{API_ENDPOINT}?date={day.isoformat()}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json() or []


def build_charting_url(api_row: dict) -> str:
    """
    Derive Tennis‑Abstract charting URL from one API row.
    The exact rule depends on the API payload; placeholder below.
    """
    return api_row.get("charting_url", "").strip()


# ─── fallback: pull Tennis‑Abstract charting index directly ───────────────
#
# Jeff Sackmann maintains a public CSV containing every charting match with a
# relative URL column “url” and a “date” column in YYYY‑MM‑DD format:
#
#   https://raw.githubusercontent.com/JeffSackmann/tennis_charting/master/charting_match_index.csv
#
# We hit that file once per run, cache it in‑memory, and expose a helper that
# returns the full “https://www.tennisabstract.com/charting/…html” URL list
# for a given *day*.
# -------------------------------------------------------------------------

import functools, io

CHARTING_INDEX_CSV = (
    "https://raw.githubusercontent.com/JeffSackmann/"
    "tennis_charting/master/charting_match_index.csv"
)

@functools.lru_cache(maxsize=1)
def _load_charting_index() -> pd.DataFrame:
    csv_bytes = requests.get(CHARTING_INDEX_CSV, timeout=30).content
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["full_url"] = (
        "https://www.tennisabstract.com/charting/" + df["url"].str.strip("/")
    )
    return df[["date", "full_url"]]

def charting_urls_for_day(day: datetime.date) -> list[str]:
    """Return list of TA charting URLs where match‑date == *day*."""
    df = _load_charting_index()
    return df.loc[df["date"] == day, "full_url"].tolist()


# ─── orchestrator ─────────────────────────────────────────────────────────────
def sync(date_from: datetime.date | None = None,
         date_to  : datetime.date | None = None) -> None:
    """
    Incrementally append all matches with match‑date ∈ [date_from, date_to].
    When *date_from* is omitted, starts with the first day **after** the latest
    recorded match; *date_to* defaults to today().
    """
    if date_from is None:
        date_from = _latest_recorded_date() + datetime.timedelta(days=1)
    if date_to is None:
        date_to = datetime.date.today()

    if date_from > date_to:
        print("No new days to process.")
        return

    scraper = TennisAbstractScraper()
    frames: list[pd.DataFrame] = []

    current = date_from
    while current <= date_to:
        # 1) API ingest -------------------------------------------------------
        try:
            api_rows = fetch_api_matches(current)
            if api_rows:
                frames.append(pd.DataFrame(api_rows))
        except Exception as exc:
            print(f"[WARN] API fetch failed {current}: {exc}")
            api_rows = []

        # 1b) direct Tennis‑Abstract index fallback -------------------------
        if not api_rows:
            for ta_url in charting_urls_for_day(current):
                api_rows.append({"charting_url": ta_url})

        # 2) Tennis‑Abstract scraping ----------------------------------------
        for row in api_rows:
            ta_url = build_charting_url(row)
            if not ta_url:
                continue
            try:
                frames.append(pd.DataFrame(scraper.scrape_shot_types(ta_url)))
                frames.append(pd.DataFrame(scraper.scrape_shot_direction(ta_url)))
                # ── add further scraper calls here as needed ──
            except Exception as exc:
                print(f"[WARN] TA scrape failed {ta_url}: {exc}")

        current += datetime.timedelta(days=1)

    if frames:
        increment = pd.concat(frames, ignore_index=True)
        existing  = _load_incremental()
        combined  = pd.concat([existing, increment], ignore_index=True)
        _save_incremental(combined)
        print(f"[OK] Appended {len(increment)} new rows (through {date_to}).")
    else:
        print("[INFO] Nothing new to append.")


# ─── optional CLI entry‑point ────────────────────────────────────────────────
# Run `python t3n11s.py --sync` to execute the incremental updater.
if __name__ == "__main__":
    import sys
    if "--sync" in sys.argv:
        sync()
# %%
# ------------------------------------------------------------------
# JEFF column whitelist
# ------------------------------------------------------------------
def _load_jeff_header_set(root: str | None = None) -> set[str]:
    """
    Build a superset of all column names found in Jeff Sackmann
    charting statistics CSVs.

    If *root* is omitted, the default path
        '~/Desktop/data/Jeff 6.14.25'
    is used.  Every file that matches the glob pattern
        'charting-*stats-*.csv'
    (men, women, any subtype) is read and its headers merged into
    a single set.  Whitespace is stripped from each column label.
    """
    import os, glob, pandas as pd

    if root is None:
        root = os.path.expanduser("~/Desktop/data/Jeff 6.14.25")

    headers: set[str] = set()
    patterns = (
        "charting-*stats-*.csv",
        "charting-*matches*.csv",
        "charting-*points-*.csv",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(root, pattern)):
            try:
                headers |= {h.strip() for h in pd.read_csv(path, nrows=0).columns}
            except Exception:
                # Skip unreadable or malformed files
                continue
    return headers

JEFF_HEADERS = _load_jeff_header_set()

# --------------------------------------------------------------
# Extra columns generated by Serve Basics/Direction parsing that
# are not present in the original Jeff CSV header set.
# --------------------------------------------------------------
SERVE_COLS = {
    "serve_type", "serve_pts", "serve_won", "aces", "unret", "fcdE", "le3w",
    "wide_pct", "body_pct", "t_pct",
    "dc_wide_pct", "dc_body_pct", "dc_t_pct",
    "ad_wide_pct", "ad_body_pct", "ad_t_pct",
    "net_pct", "wide_err_pct", "deep_err_pct", "w_d_pct",
    "footfault_pct", "unk_pct"
}

# Promote the new serve columns to first‑class status so the Jeff
# whitelist filter keeps them.
JEFF_HEADERS |= SERVE_COLS

# ------------------------------------------------------------------
#  Abbreviation glossary (site → canonical Jeff‑style field names)
#  This is a one‑time extension so downstream pipelines recognise
#  short labels such as “FcdE” without additional scrapers.
# ------------------------------------------------------------------
ABBREV_MAP = {
    # serve / key‑point abbreviations
    "FcdE": "forced_errors_induced",     # points where server induced a forced return error
    "RlyFcd": "rally_forced_errors",     # rally ended on forced error
    "RlyWnr": "rally_winners",
    "SvWnr": "serve_winners",            # unreturnables + forced return errors
    "UFE": "unforced_errors",
    "DF": "double_faults",
    # generic counters
    "PtsW": "points_won",
    "Pts": "points_total",
    "1stIn": "first_serves_in",
}

# Merge the canonical names into the Jeff header whitelist so every
# new column passes the filter stage exactly once.
JEFF_HEADERS |= set(ABBREV_MAP.values())

# ============================================================================
# FIXTURE‑FLATTENING HELPERS  (required by integrate_api_tennis_data)
# ============================================================================

def _fx_canonical_name(name: str) -> str:
    return normalize_name(name)

def _fx_winner(fx: dict) -> str:
    return fx["event_first_player"] if fx["event_winner"].startswith("First") else fx["event_second_player"]

def _fx_loser(fx: dict) -> str:
    return fx["event_second_player"] if _fx_winner(fx) == fx["event_first_player"] else fx["event_first_player"]

def _fx_parse_scores(scores: list[dict]) -> list[tuple[int, int]]:
    sets = []
    for s in scores:
        try:
            f, l = int(float(s["score_first"])), int(float(s["score_second"]))
        except Exception:
            f, l = np.nan, np.nan
        sets.append((f, l))
    return sets

def flatten_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    """
    Convert the nested API‑Tennis fixture payload into one wide row per match
    with basic score‑line stats and wide player‑statistics columns.
    """
    records = []
    for fx in fixtures:
        # ---  core match metadata  ---
        rec = {
            "event_key": int(fx["event_key"]),
            "date": pd.to_datetime(fx["event_date"]),
            "tournament": fx.get("tournament_name"),
            "round": fx.get("tournament_round", ""),
            "surface": fx.get("court_surface", "Hard"),
            "event_type": fx.get("event_type_type"),
            "season": fx.get("tournament_season"),
            "winner": _fx_winner(fx),
            "loser":  _fx_loser(fx),
        }
        rec["winner_id"] = int(fx["first_player_key"]
                               if rec["winner"] == fx["event_first_player"]
                               else fx["second_player_key"])
        rec["loser_id"]  = int(fx["second_player_key"]
                               if rec["winner"] == fx["event_first_player"]
                               else fx["first_player_key"])
        rec["winner_canonical"] = _fx_canonical_name(rec["winner"])
        rec["loser_canonical"]  = _fx_canonical_name(rec["loser"])
        rec["api_scores_raw"]   = json.dumps(fx.get("scores", []))
        rec["api_pointbypoint"] = len(fx.get("pointbypoint", [])) > 0
        rec["composite_id"] = build_composite_id(
            rec["date"].date(),
            normalize_tournament_name(rec["tournament"]),
            rec["winner_canonical"],
            rec["loser_canonical"],
        )

        # ---  set‑level metrics  ---
        sets = _fx_parse_scores(fx.get("scores", []))
        rec["sets_played"]      = len(sets)
        rec["sets_won_winner"]  = sum(f > l for f, l in sets)
        rec["games_winner"]     = sum(f for f, _ in sets)
        rec["games_loser"]      = sum(l for _, l in sets)
        rec["tb_played"]        = any(max(f, l) >= 7 for f, l in sets)

        # ---  wide statistics  ---
        stat_df = pd.DataFrame(fx.get("statistics", []))
        if not stat_df.empty:
            stat_df["stat_col"] = (
                stat_df["stat_name"]
                  .str.lower()
                  .str.replace("%", "pct")
                  .str.replace(" ", "_")
                + "_" + stat_df["stat_period"]
            )
            for pid, tag in [(fx["first_player_key"], "p1"),
                             (fx["second_player_key"], "p2")]:
                sub = (stat_df[stat_df["player_key"] == pid]
                       .drop_duplicates(subset=["stat_col"], keep="last"))
                wide = (sub
                        .pivot_table(index="player_key",
                                     columns="stat_col",
                                     values="stat_value",
                                     aggfunc="first"))
                if not wide.empty:
                    for col, val in wide.iloc[0].items():
                        rec[f"{tag}_{col}"] = val

        records.append(rec)

    df = pd.DataFrame(records)
    num_cols = [c for c in df.columns if c.startswith(("p1_", "p2_"))]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="ignore")
    return df

# ============================================================================
# API INTEGRATION FUNCTIONS
# ============================================================================

from datetime import date, timedelta

def integrate_api_tennis_data(historical_data, days_back: int = 3):
    """Integrate API-Tennis data for recent matches.

    Parameters
    ----------
    historical_data : DataFrame
        Existing combined dataset.
    days_back : int, default 3
        How many days before *today* to start pulling finished matches.
    """
    start_date = date.today() - timedelta(days=days_back)
    print("Step 7: Integrating comprehensive API-Tennis data...")

    # Load static metadata once
    print("  Loading tournament and event type metadata...")
    tournaments_meta = get_tournaments_metadata()
    event_types_meta = get_event_types()
    print(f"  Loaded {len(tournaments_meta)} tournaments, {len(event_types_meta)} event types")

    # Ensure event_key column exists
    if "event_key" not in historical_data.columns:
        historical_data["event_key"] = pd.NA

    # Get existing API keys to avoid duplicates
    existing_keys = set()
    if len(historical_data) > 0:
        cutoff_data = historical_data[historical_data["date"] >= start_date]
        if len(cutoff_data) > 0:
            # Safely convert existing event_keys to integers
            for key in cutoff_data["event_key"].dropna():
                converted_key = safe_int_convert(key)
                if converted_key is not None:
                    existing_keys.add(converted_key)
    # ------------------------------------------------------------------
    # Ensure composite_id column and back-fill where missing
    # ------------------------------------------------------------------
    if "composite_id" not in historical_data.columns:
        historical_data["composite_id"] = pd.NA

    if historical_data['composite_id'].isna().any():
        historical_data.loc[historical_data['composite_id'].isna(), 'composite_id'] = (
            historical_data[historical_data['composite_id'].isna()]
            .apply(lambda r: build_composite_id(
                r['date'],
                r.get('tournament_canonical', normalize_tournament_name(r['Tournament'])),
                r.get('winner_canonical',  normalize_name(r['Winner'])),
                r.get('loser_canonical',   normalize_name(r['Loser']))
            ), axis=1)
        )

    print(f"Found {len(existing_keys)} existing API matches")

    api_matches = []
    date_range = list(pd.date_range(start_date, date.today()))

    for d in date_range:
        day = d.date()
        try:
            fixtures = get_fixtures_for_date(day)
            if fixtures:
                print(f"  {day}: {len(fixtures)} finished matches")

            # Build a per‑day dataframe with rich stats for quick lookup
            fixtures_df = flatten_fixtures(fixtures)
            fixtures_df = fixtures_df.set_index("event_key")

            for fixture in fixtures:
                try:
                    event_key = safe_int_convert(fixture.get("event_key"))
                    if event_key is None:
                        continue

                    if event_key in existing_keys:
                        continue

                    # Basic match info
                    p1_name = fixture["event_first_player"]
                    p2_name = fixture["event_second_player"]
                    winner = p1_name if fixture.get("event_winner", "").startswith("First") else p2_name
                    loser = p2_name if winner == p1_name else p1_name

                    # Tournament metadata
                    tournament_key = fixture.get("tournament_key")
                    tournament_meta = tournaments_meta.get(tournament_key, {})

                    # Event type metadata
                    event_type_key = tournament_meta.get("event_type_key")
                    event_type_meta = event_types_meta.get(event_type_key, {})

                    # --- build winner / loser names first -----------------------------
                    winner = p1_name if fixture.get("event_winner", "").startswith("First") else p2_name
                    loser  = p2_name if winner == p1_name else p1_name

                    win_c  = normalize_name(winner)
                    los_c  = normalize_name(loser)
                    tour_c = normalize_tournament_name(fixture.get("tournament_name", "Unknown"))

                    match_record = {
                        "event_key":        event_key,
                        "Date":             pd.to_datetime(fixture["event_date"]),
                        "date":             pd.to_datetime(fixture["event_date"]).date(),
                        "Tournament":       fixture.get("tournament_name", "Unknown"),
                        "round":            fixture.get("tournament_round", ""),
                        "Surface":          fixture.get("court_surface", "Hard"),
                        "Winner":           winner,
                        "Loser":            loser,
                        "source_rank":      1,   # API source
                        "gender":           "W" if "wta" in fixture.get("event_type_type", "").lower() else "M",
                        "winner_canonical": win_c,
                        "loser_canonical":  los_c,
                        "tournament_canonical": tour_c,
                    }

                    # add composite id once dict exists
                    match_record["composite_id"] = build_composite_id(
                        match_record["date"],
                        tour_c,
                        win_c,
                        los_c,
                    )

                    # Extract embedded statistics from fixture
                    embedded_stats = extract_embedded_statistics(fixture)
                    match_record.update(embedded_stats)

                    # Merge the pre‑flattened wide statistics row for this match
                    if event_key in fixtures_df.index:
                        rich = fixtures_df.loc[event_key].to_dict()
                        for col, val in rich.items():
                            if col not in match_record:      # keep existing core fields
                                match_record[col] = val

                    # Get odds (only for dates >= 2025-06-23)
                    odds1, odds2 = get_match_odds(event_key, day)

                    # Add normalized names
                    match_record["winner_canonical"] = normalize_name(winner)
                    match_record["loser_canonical"] = normalize_name(loser)
                    match_record["tournament_canonical"] = normalize_tournament_name(match_record["Tournament"])

                    # Extract embedded statistics from fixture
                    embedded_stats = extract_embedded_statistics(fixture)
                    match_record.update(embedded_stats)

                    # Add raw fixture data for later analysis
                    match_record["api_scores"] = json.dumps(fixture.get("scores", []))
                    match_record["api_pointbypoint_available"] = len(fixture.get("pointbypoint", [])) > 0

                    # Get odds (only for dates >= 2025-06-23)
                    odds1, odds2 = get_match_odds(event_key, day)
                    match_record["api_odds_home"] = odds1
                    match_record["api_odds_away"] = odds2

                    # Get rankings
                    league = "WTA" if match_record["gender"] == "W" else "ATP"
                    rankings = get_player_rankings(day, league)
                    p1_key = int(fixture.get("first_player_key", 0))
                    p2_key = int(fixture.get("second_player_key", 0))

                    if winner == p1_name:
                        match_record["WRank"] = rankings.get(p1_key, pd.NA)
                        match_record["LRank"] = rankings.get(p2_key, pd.NA)
                        match_record["winner_player_key"] = p1_key
                        match_record["loser_player_key"] = p2_key
                    else:
                        match_record["WRank"] = rankings.get(p2_key, pd.NA)
                        match_record["LRank"] = rankings.get(p1_key, pd.NA)
                        match_record["winner_player_key"] = p2_key
                        match_record["loser_player_key"] = p1_key

                    # Get H2H data
                    h2h_data = get_h2h_data(p1_key, p2_key)
                    match_record.update({f"h2h_{k}": v for k, v in h2h_data.items()})

                    # Add tournament metadata
                    match_record["tournament_key"] = tournament_key
                    match_record["event_type_key"] = event_type_key
                    match_record["event_type_type"] = fixture.get("event_type_type", "")
                    match_record["tournament_season"] = fixture.get("tournament_season", "")

                    api_matches.append(match_record)
                    existing_keys.add(event_key)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"    Error processing match {fixture.get('event_key', 'unknown')}: {e}")
                    continue

        except Exception as e:
            print(f"  Error processing day {day}: {e}")
            continue

    print(f"Collected {len(api_matches)} new API matches with full metadata")

    # Merge with historical data
    if api_matches:
        try:
            api_df = pd.DataFrame(api_matches)

            # Align columns
            for col in historical_data.columns:
                if col not in api_df.columns:
                    api_df[col] = pd.NA

            for col in api_df.columns:
                if col not in historical_data.columns:
                    historical_data[col] = pd.NA

            # Ensure column order matches
            api_df = api_df.reindex(columns=historical_data.columns, fill_value=pd.NA)

            # Concatenate
            combined_data = pd.concat([historical_data, api_df], ignore_index=True)

            # Add source_rank if missing
            if "source_rank" not in combined_data.columns:
                combined_data["source_rank"] = 2  # Default to tennis-data
            combined_data["source_rank"] = combined_data["source_rank"].fillna(2)

            # Deduplicate (keep API data over tennis-data when available)
            dedup_keys = ["event_key", "composite_id"]
            final_data = (
                combined_data
                .sort_values("source_rank")  # API=1, tennis-data=2
                .drop_duplicates(subset=dedup_keys, keep="first")
                .reset_index(drop=True)
            )

            print(f"✓ Successfully integrated {len(api_df)} API matches")
            print(f"Final dataset: {len(final_data)} matches")

            # Show what we got
            if len(api_df) > 0:
                odds_count = api_df["api_odds_home"].notna().sum()
                h2h_count = api_df["h2h_matches"].notna().sum()
                rankings_count = api_df["WRank"].notna().sum()
                print(f"  - Matches with odds: {odds_count}")
                print(f"  - Matches with H2H data: {h2h_count}")
                print(f"  - Matches with rankings: {rankings_count}")

            return final_data

        except Exception as e:
            print(f"Error merging API data: {e}")
            return historical_data
    else:
        print("No new API data to merge")
        return historical_data

# ============================================================================
# MAIN EXECUTION
# 3. SIMULATION ENGINE
# ============================================================================

# ------------------------------------------------------------------
# Cache logic — flip REFRESH_CACHE to force regeneration
# ------------------------------------------------------------------
REFRESH_CACHE = True          # set False to reuse cached data

if REFRESH_CACHE:
    print("Refreshing cache …")
    historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(
        fast=False                         # use full dataset so API rows can merge
    )
    save_to_cache(historical_data, jeff_data, weighted_defaults)
    # Pull finished API‑Tennis matches from 10 June 2025 onward
    from datetime import date
    days_back = (date.today() - date(2025, 6, 10)).days
    historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
    save_to_cache(historical_data, jeff_data, weighted_defaults)
else:
    historical_data, jeff_data, weighted_defaults = load_from_cache()
    if historical_data is None:
        print("Cache miss – generating data …")
        historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(
            fast=False
        )
        save_to_cache(historical_data, jeff_data, weighted_defaults)
    else:
        print("✓ Data loaded from cache")

    # pull finished API‑Tennis matches from 10 June 2025 onward
    days_back = (date.today() - date(2025, 6, 10)).days
    historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
    save_to_cache(historical_data, jeff_data, weighted_defaults)

#%% md
# SIMULATION
#%%
# LAYER 1 ##
def extract_data_samples():
    # Jeff Sackmann data samples
    jeff_samples = {
        'matches': jeff_data['men']['matches'].head(3),
        'serve_basics': jeff_data['men']['serve_basics'].head(3),
        'overview': jeff_data['men']['overview'].head(3)
    }

    # Tennis-data samples
    tennis_samples = historical_data[
        ['Winner', 'Loser', 'WRank', 'LRank', 'PSW', 'PSL', 'Surface']
    ].head(3)

    return jeff_samples, tennis_samples

# Hold/break computation method verification
hold_break_computation = {
    'current_method': 'Jeff aggregated stats from overview dataset',
    'available_columns': ['serve_pts', 'first_in', 'first_won', 'second_won'],
    'computation_level': 'Per-player aggregate from charting data'
}

# Bayesian
def extract_priors_from_current_data(player_canonical, gender, surface):
    priors = {}

    # Layer 1: Elo approximation from rankings
    player_matches = historical_data[
        (historical_data['winner_canonical'] == player_canonical) |
        (historical_data['loser_canonical'] == player_canonical)
    ]

    if len(player_matches) > 0:
        # Ranking-based Elo estimation
        recent_rank = get_recent_rank(player_canonical, player_matches)
        elo_estimate = 2000 - (recent_rank * 5) if recent_rank else 1500

        # Jeff feature extraction
        jeff_features = extract_jeff_features(player_canonical, gender, jeff_data)

        priors = {
            'elo_estimate': elo_estimate,
            'serve_effectiveness': jeff_features.get('serve_pts', 0.6),
            'return_strength': jeff_features.get('return_pts_won', 0.3),
            'surface_factor': calculate_surface_adjustment(player_matches, surface)
        }

    return priors

# Time decay for recent form
def calculate_time_decayed_performance(player_matches, reference_date):
    player_matches['days_ago'] = (reference_date - player_matches['date']).dt.days

    # Exponential decay: recent matches weighted heavier
    weights = np.exp(-0.01 * player_matches['days_ago'])  # 1% daily decay

    weighted_performance = {
        'win_rate': np.average(player_matches['is_winner'], weights=weights),
        'games_won_rate': np.average(player_matches['games_won_pct'], weights=weights)
    }

    return weighted_performance
#%%
## TEST ##
import os, pickle, pandas as pd

CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
os.makedirs(CACHE_DIR, exist_ok=True)
HD_PATH   = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH  = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

if (os.path.exists(HD_PATH) and
    os.path.exists(JEFF_PATH) and
    os.path.exists(DEF_PATH)):
    print("Loading cached data …")
    historical_data = pd.read_parquet(HD_PATH)
    with open(JEFF_PATH, "rb") as fh:
        jeff_data = pickle.load(fh)
    with open(DEF_PATH, "rb") as fh:
        weighted_defaults = pickle.load(fh)
else:
    print("Cache miss – regenerating (one-time slow run).")
    combined_data, jeff_data, weighted_defaults = generate_comprehensive_historical_all_years()
    historical_data = combined_data
    historical_data.to_parquet(HD_PATH, index=False)
    with open(JEFF_PATH, "wb") as fh:
        pickle.dump(jeff_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(DEF_PATH, "wb") as fh:
        pickle.dump(weighted_defaults, fh, protocol=pickle.HIGHEST_PROTOCOL)

# "SIMULATION"
#%%

#%%
import pandas as pd
import numpy as np
from collections import defaultdict

def normalize_name_canonical(name):
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    return ' '.join(name.lower().split())

def extract_jeff_features(player_canonical, gender, jeff_data):
    """Extract actual features from Jeff Sackmann data"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or player_canonical not in jeff_data[gender_key]:
        return {
            'serve_pts': 60,
            'first_won': 0,
            'second_won': 0,
            'return_pts_won': 20
        }

    player_data = jeff_data[gender_key][player_canonical]

    first_in = player_data.get('1stIn', 0)
    first_won = player_data.get('1stWon', 0)
    second_won = player_data.get('2ndWon', 0)
    double_faults = player_data.get('df', 0)

    total_serve_pts = first_in + double_faults + (first_won - first_in) if first_won >= first_in else first_in + second_won + double_faults

    break_points_saved = player_data.get('bpSaved', 0)
    break_points_faced = player_data.get('bpFaced', 0)
    return_pts_won = break_points_faced - break_points_saved

    return {
        'serve_pts': max(1, total_serve_pts),
        'first_won': first_won,
        'second_won': second_won,
        'return_pts_won': max(0, return_pts_won)
    }

# 3.1 Base Bayesian Tennis Model
class BayesianTennisModel:
    def __init__(self):
        self.simulation_count = 10000
        self.jeff_data = jeff_data
        self.historical_data = historical_data
        self.jeff_data = None
        self.historical_data = None

    def default_priors(self):
        return {
            'elo_mean': 1500,
            'elo_std': 200,
            'hold_prob': 0.65,
            'break_prob': 0.35,
            'surface': 'Hard',
            'form_factor': 1.0,
            'confidence': 0.1
        }

    def extract_refined_priors(self, player_canonical, gender, surface, reference_date):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].copy()
            ].copy()

        if len(player_matches) == 0:
            return self.default_priors()

        surface_matches = player_matches[player_matches['Surface'] == surface]
        if len(surface_matches) < 5:
            surface_matches = player_matches

        recent_matches = surface_matches.tail(20).copy()
        recent_matches['days_ago'] = (pd.to_datetime(reference_date) - pd.to_datetime(recent_matches['Date'])).dt.days
        weights = np.exp(-0.05 * recent_matches['days_ago'])

        base_elo = self.get_player_weighted_elo(player_canonical, surface, reference_date)
        surface_factor = self.calculate_surface_adaptation(player_canonical, surface)
        elo_prior = base_elo * surface_factor

        jeff_features = extract_jeff_features(player_canonical, gender, self.jeff_data)

        serve_pts = jeff_features['serve_pts']
        serve_won = jeff_features['first_won'] + jeff_features['second_won']
        hold_prob = serve_won / serve_pts if serve_pts > 0 else 0.65

        return_pts = jeff_features['return_pts_won']
        total_return_pts = serve_pts
        break_prob = (1 - return_pts / total_return_pts) if total_return_pts > 0 else 0.35

        return {
            'elo_mean': elo_prior,
            'elo_std': 150,
            'hold_prob': min(0.95, max(0.3, hold_prob)),
            'break_prob': max(0.05, min(0.7, break_prob)),
            'surface': surface,
            'form_factor': self.calculate_form_spike(recent_matches, weights, player_canonical),
            'confidence': max(0.05, min(1.0, len(recent_matches) / 15))
        }

    def calculate_ranking_differential_odds(self, p1_ranking, p2_ranking):
        """Convert ranking differential to implied probability"""
        if p1_ranking == 0 or p2_ranking == 0:
            return 0.5

        ranking_diff = p2_ranking - p1_ranking

        if ranking_diff > 50:
            return 0.85
        elif ranking_diff > 20:
            return 0.75
        elif ranking_diff > 10:
            return 0.65
        elif ranking_diff > 0:
            return 0.55
        elif ranking_diff > -10:
            return 0.45
        elif ranking_diff > -20:
            return 0.35
        elif ranking_diff > -50:
            return 0.25
        else:
            return 0.15

    def calculate_upset_frequency(self, ranking_diff, surface, historical_data):
        """Calculate upset frequency by ranking differential and surface"""
        upset_matches = historical_data[
            ((historical_data['WRank'] - historical_data['LRank']) > ranking_diff) &
            (historical_data['Surface'] == surface)
        ]

        total_matches = historical_data[
            (abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)) &
            (historical_data['Surface'] == surface)
        ]

        if len(total_matches) < 10 and surface != 'fallback':
            return self.calculate_upset_frequency(ranking_diff, 'fallback', historical_data)

        if surface == 'fallback':
            upset_matches = historical_data[
                (historical_data['WRank'] - historical_data['LRank']) > ranking_diff
            ]
            total_matches = historical_data[
                abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)
            ]

        if len(total_matches) == 0:
            return 0.1

        upset_rate = len(upset_matches) / len(total_matches)
        return min(0.45, max(0.05, upset_rate))

    def calculate_surface_performance_ratio(self, player_canonical, surface, opponent_canonical, reference_date):
        """Calculate player's surface-specific performance vs opponent's baseline"""
        player_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
        ].tail(20)

        opponent_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == opponent_canonical) |
             (self.historical_data['loser_canonical'] == opponent_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
        ].tail(20)

        if len(player_surface_matches) < 3 or len(opponent_surface_matches) < 3:
            return 1.0

        player_wins = len(player_surface_matches[player_surface_matches['winner_canonical'] == player_canonical])
        opponent_wins = len(opponent_surface_matches[opponent_surface_matches['winner_canonical'] == opponent_canonical])

        player_ratio = player_wins / len(player_surface_matches)
        opponent_ratio = opponent_wins / len(opponent_surface_matches)

        return player_ratio / opponent_ratio if opponent_ratio > 0 else 1.0

    def run_simulation(self, p1_priors, p2_priors, iterations):
        return [self.simulate_match(p1_priors, p2_priors)]

    def predict_match_outcome(self, player1_canonical, player2_canonical, surface, gender, date):
        p1_priors = self.extract_refined_priors(player1_canonical, gender, surface, date)
        p2_priors = self.extract_refined_priors(player2_canonical, gender, surface, date)

        base_prob = self.run_simulation(p1_priors, p2_priors, 1000)[0]

        p1_rank = self.get_player_ranking(player1_canonical, date)
        p2_rank = self.get_player_ranking(player2_canonical, date)
        ranking_prob = self.calculate_ranking_differential_odds(p1_rank, p2_rank)

        ranking_diff = p1_rank - p2_rank
        upset_adjustment = self.calculate_upset_frequency(ranking_diff, surface, self.historical_data)

        surface_ratio = self.calculate_surface_performance_ratio(player1_canonical, surface, player2_canonical, date)

        calibrated_prob = (0.6 * base_prob + 0.25 * ranking_prob + 0.15 * surface_ratio) * (1 - upset_adjustment * 0.1)

        return max(0.05, min(0.95, calibrated_prob))

    def get_player_ranking(self, player_canonical, date):
        """Get player ranking at specific date"""
        date_obj = pd.to_datetime(date)

        player_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (pd.to_datetime(self.historical_data['Date']) <= date_obj)
        ].sort_values('Date', ascending=False)
            ].sort_values('Date', ascending=False)

        if len(player_matches) == 0:
            return 999

        latest_match = player_matches.iloc[0]

        if latest_match['winner_canonical'] == player_canonical:
            return latest_match.get('WRank', 999)
        else:
            return latest_match.get('LRank', 999)

    def calculate_match_probability(self, player1_canonical, player2_canonical, gender, surface, reference_date, best_of=3):
        player1_priors = self.extract_refined_priors(player1_canonical, gender, surface, reference_date)
        player2_priors = self.extract_refined_priors(player2_canonical, gender, surface, reference_date)

        probability = self.simulate_match(player1_priors, player2_priors, best_of)
        confidence = min(player1_priors['confidence'], player2_priors['confidence'])

        return {
            'player1_win_probability': probability,
            'player2_win_probability': 1 - probability,
            'confidence': confidence,
            'player1_priors': player1_priors,
            'player2_priors': player2_priors
        }

    def calculate_form_spike(self, recent_matches, weights, player_canonical):
        if len(recent_matches) == 0:
            return 1.0

        wins = (recent_matches['winner_canonical'] == player_canonical).astype(int)
        weighted_win_rate = np.average(wins, weights=weights)

        avg_opponent_rank = recent_matches['LRank'].fillna(recent_matches['WRank']).mean()
        player_rank = recent_matches['WRank'].fillna(recent_matches['LRank']).iloc[-1]

        if pd.notna(avg_opponent_rank) and pd.notna(player_rank):
            rank_diff = player_rank - avg_opponent_rank
            expected_win_rate = 1 / (1 + 10**(rank_diff/400))
            expected_win_rate = 1 / (1 + 10 ** (rank_diff / 400))
            form_spike = min(2.0, weighted_win_rate / max(0.1, expected_win_rate))
        else:
            form_spike = 1.0

        return form_spike

    def get_player_weighted_elo(self, player_canonical, surface, reference_date):
        recent_match = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface)
        ].tail(1)
            ].tail(1)

        if len(recent_match) > 0 and 'BlendScore' in recent_match.columns:
            blend_score = recent_match['BlendScore'].iloc[0]
            return 1500 + blend_score * 50

        any_surface_match = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].tail(1)
            ].tail(1)

        if len(any_surface_match) > 0 and 'BlendScore' in any_surface_match.columns:
            return 1500 + any_surface_match['BlendScore'].iloc[0] * 200

        return 1500

    def calculate_surface_adaptation(self, player_canonical, target_surface):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
        ].copy()
            ].copy()

        if len(player_matches) < 10:
            return 1.0

        surface_matches = player_matches[player_matches['Surface'] == target_surface]
        if len(surface_matches) < 3:
            return 1.0

        surface_wins = (surface_matches['winner_canonical'] == player_canonical).sum()
        surface_win_rate = surface_wins / len(surface_matches)

        total_wins = (player_matches['winner_canonical'] == player_canonical).sum()
        baseline_win_rate = total_wins / len(player_matches)

        if baseline_win_rate == 0:
            return 1.0

        adaptation_ratio = surface_win_rate / baseline_win_rate
        return max(0.7, min(1.5, adaptation_ratio))

    def evaluate_predictions(self, test_data):
        """Evaluate model accuracy on test dataset"""
        correct = 0
        total = 0

        for _, match in test_data.iterrows():
            prob = self.predict_match_outcome(
                match['winner_canonical'],
                match['loser_canonical'],
                match['Surface'],
                match['gender'],
                match['Date']
            )

            predicted_winner = match['winner_canonical'] if prob > 0.5 else match['loser_canonical']
            actual_winner = match['winner_canonical']

            if predicted_winner == actual_winner:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def convert_to_canonical(name):
        return normalize_name_canonical(name)

    model = BayesianTennisModel()
    #%%
    ## LAYER 2 ##
    # 3.2 Layer 2: Contextual Adjustments
    def apply_contextual_adjustments(self, priors, player_canonical, opponent_canonical, match_context):
        """Layer 2: Contextual Bayesian adjustments for fatigue, injury, motivation"""

        adjusted_priors = priors.copy()

        # Fatigue Index
        fatigue_penalty = self.calculate_fatigue_index(player_canonical, match_context['reference_date'])
        adjusted_priors['hold_prob'] *= (1 - fatigue_penalty * 0.15)  # Max 15% hold penalty
        adjusted_priors['elo_std'] *= (1 + fatigue_penalty * 0.3)    # Increase uncertainty
        adjusted_priors['hold_prob'] *= (1 - fatigue_penalty * 0.15)
        adjusted_priors['elo_std'] *= (1 + fatigue_penalty * 0.3)

        # Injury Flag Adjustment
        injury_factor = self.get_injury_factor(player_canonical, match_context['reference_date'])
        adjusted_priors['hold_prob'] *= injury_factor
        adjusted_priors['break_prob'] *= (2 - injury_factor)  # Inverse relationship
        adjusted_priors['break_prob'] *= (2 - injury_factor)

        # Form Spike Sustainability
        form_sustainability = self.calculate_form_sustainability(player_canonical, match_context)
        if adjusted_priors['form_factor'] > 1.2:  # Hot streak detection
        if adjusted_priors['form_factor'] > 1.2:
            sustainability_discount = 1 - ((adjusted_priors['form_factor'] - 1) * (1 - form_sustainability))
            adjusted_priors['hold_prob'] *= sustainability_discount
            adjusted_priors['elo_mean'] *= sustainability_discount

        # Opponent Quality Weighting
        opponent_elo = self.estimate_opponent_elo(opponent_canonical, match_context)
        elo_diff = adjusted_priors['elo_mean'] - opponent_elo
        quality_adjustment = 1 / (1 + np.exp(-elo_diff / 200))  # Sigmoid scaling
        quality_adjustment = 1 / (1 + np.exp(-elo_diff / 200))
        adjusted_priors['break_prob'] *= quality_adjustment

        return adjusted_priors

    def calculate_fatigue_index(self, player_canonical, reference_date):
        """Fatigue based on recent match load and recovery time"""
        recent_matches = self.get_recent_matches(player_canonical, reference_date, days=14)

        if len(recent_matches) == 0:
            return 0.0

        # Calculate cumulative fatigue
        fatigue_score = 0
        for _, match in recent_matches.iterrows():
            days_ago = (pd.to_datetime(reference_date) - pd.to_datetime(match['Date'])).days
            match_duration = match.get('minutes', 120)  # Default 2 hours
            match_duration = match.get('minutes', 120)

            # Exponential decay with match duration weighting
            fatigue_contribution = (match_duration / 60) * np.exp(-0.1 * days_ago)
            fatigue_score += fatigue_contribution

        return min(1.0, fatigue_score / 10)  # Normalize to 0-1
        return min(1.0, fatigue_score / 10)

    def get_injury_factor(self, player_canonical, reference_date):
        """Player-specific injury fragility scoring"""
        # Injury memory bank - replace with actual injury tracking
        injury_prone_players = {
            'nadal_r': 0.85,
            'murray_a': 0.80,
            'thiem_d': 0.75,
            'badosa_p': 0.70
        }

        base_factor = injury_prone_players.get(player_canonical, 0.95)

        # Check for recent retirement/walkover flags
        recent_retirements = self.check_recent_retirements(player_canonical, reference_date)
        if recent_retirements > 0:
            base_factor *= (0.8 ** recent_retirements)

        return max(0.5, base_factor)

    def calculate_form_sustainability(self, player_canonical, match_context):
        """Form spike sustainability based on opponent quality and win quality"""
        recent_matches = self.get_recent_matches(player_canonical, match_context['reference_date'], days=21)

        if len(recent_matches) < 3:
            return 0.5

        # Quality-weighted recent performance
        quality_scores = []
        for _, match in recent_matches.iterrows():
            opponent_rank = match['LRank'] if match['winner_canonical'] == player_canonical else match['WRank']
            win_quality = 1 / (1 + opponent_rank / 100) if pd.notna(opponent_rank) else 0.5
            quality_scores.append(win_quality)

        avg_opponent_quality = np.mean(quality_scores)
        consistency = 1 - np.std(quality_scores)

        return min(1.0, avg_opponent_quality * consistency)

    def estimate_opponent_elo(self, opponent_canonical, match_context):
        """Quick opponent Elo estimation for quality weighting"""
        opponent_priors = self.extract_refined_priors(
            opponent_canonical,
            match_context['gender'],
            match_context['surface'],
            match_context['reference_date']
        )
        return opponent_priors['elo_mean']

    def get_recent_matches(self, player_canonical, reference_date, days=14):
        try:
            cutoff_date = pd.to_datetime(reference_date) - pd.Timedelta(days=days)

            player_matches = self.historical_data[
                ((self.historical_data['winner_canonical'] == player_canonical) |
                 (self.historical_data['loser_canonical'] == player_canonical))
            ].copy()

            if len(player_matches) == 0:
                return player_matches

            # Force string conversion then datetime to avoid mixed types
            player_matches['Date'] = pd.to_datetime(player_matches['Date'].astype(str), errors='coerce')
            player_matches = player_matches.dropna(subset=['Date'])
            player_matches = player_matches[player_matches['Date'] >= cutoff_date]

            return player_matches.sort_values('Date')
        except:
            # Return empty DataFrame on any error
            return pd.DataFrame()

    def check_recent_retirements(self, player_canonical, reference_date):
        """Count recent retirements/walkovers - placeholder for actual retirement tracking"""
        # Implementation depends on your data structure for retirement flags
        """Count recent retirements/walkovers"""
        return 0
    #%%
    ## LAYER 3 ##
    def simulate_match(self, player1_priors, player2_priors, best_of=3, tiebreak_sets=[1,2,3]):

    # 3.3 Layer 3: Monte Carlo Simulation
    def simulate_match(self, player1_priors, player2_priors, best_of=3, tiebreak_sets=[1, 2, 3]):
        """Layer 3: Monte Carlo match simulation with Bayesian priors"""

        wins = 0
        simulations = self.simulation_count

        for _ in range(simulations):
            sets_won = [0, 0]  # [player1, player2]
            sets_won = [0, 0]

            while max(sets_won) < (best_of + 1) // 2:
                set_winner = self.simulate_set(
                    player1_priors,
                    player2_priors,
                    tiebreak=len([s for s in sets_won if s > 0]) + 1 in tiebreak_sets
                )
                sets_won[set_winner] += 1

            if sets_won[0] > sets_won[1]:
                wins += 1

        return wins / simulations

    def simulate_set(self, p1_priors, p2_priors, tiebreak=True):
        """Simulate single set with service alternation"""
        games = [0, 0]
        server = 0  # 0 = player1 serves first
        server = 0

        while True:
            # Determine game winner based on server
            if server == 0:
                hold_prob = p1_priors['hold_prob']
                game_winner = 0 if np.random.random() < hold_prob else 1
            else:
                hold_prob = p2_priors['hold_prob']
                game_winner = 1 if np.random.random() < hold_prob else 0

            games[game_winner] += 1
            server = 1 - server  # Alternate serve
            server = 1 - server

            # Check set completion
            if games[0] >= 6 and games[0] - games[1] >= 2:
                return 0
            elif games[1] >= 6 and games[1] - games[0] >= 2:
                return 1
            elif games[0] == 6 and games[1] == 6 and tiebreak:
                return self.simulate_tiebreak(p1_priors, p2_priors)

    def simulate_tiebreak(self, p1_priors, p2_priors):
        """Simulate tiebreak with point-by-point serve alternation"""
        points = [0, 0]
        server = 0
        serve_count = 0

        while True:
            # Determine point winner
            if server == 0:
                hold_prob = p1_priors['hold_prob']
                point_winner = 0 if np.random.random() < hold_prob else 1
            else:
                hold_prob = p2_priors['hold_prob']
                point_winner = 1 if np.random.random() < hold_prob else 0

            points[point_winner] += 1
            serve_count += 1

            # Alternate server every 2 points (except first point)
            if serve_count == 1 or serve_count % 2 == 0:
                server = 1 - server

            # Check tiebreak completion
            if points[0] >= 7 and points[0] - points[1] >= 2:
                return 0
            elif points[1] >= 7 and points[1] - points[0] >= 2:
                return 1
#%%
# Tomorrow's slate
from datetime import date, timedelta
import numpy as np

    def evaluate_predictions(self, test_data):
        """Evaluate model accuracy on test dataset"""
        correct = 0
        total = 0

        for _, match in test_data.iterrows():
            prob = self.predict_match_outcome(
                match['winner_canonical'],
                match['loser_canonical'],
                match['Surface'],
                match['gender'],
                match['Date']
            )

            predicted_winner = match['winner_canonical'] if prob > 0.5 else match['loser_canonical']
            actual_winner = match['winner_canonical']

            if predicted_winner == actual_winner:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def predict_match_outcome(self, player1_canonical, player2_canonical, surface, gender, date):
        p1_priors = self.extract_refined_priors(player1_canonical, gender, surface, date)
        p2_priors = self.extract_refined_priors(player2_canonical, gender, surface, date)

        base_prob = self.simulate_match(p1_priors, p2_priors)

        p1_rank = self.get_player_ranking(player1_canonical, date)
        p2_rank = self.get_player_ranking(player2_canonical, date)
        ranking_prob = self.calculate_ranking_differential_odds(p1_rank, p2_rank)

        ranking_diff = p1_rank - p2_rank
        upset_adjustment = self.calculate_upset_frequency(ranking_diff, surface, self.historical_data)

        surface_ratio = self.calculate_surface_performance_ratio(player1_canonical, surface, player2_canonical, date)

        calibrated_prob = (0.6 * base_prob + 0.25 * ranking_prob + 0.15 * surface_ratio) * (1 - upset_adjustment * 0.1)

        return max(0.05, min(0.95, calibrated_prob))

    def calculate_ranking_differential_odds(self, p1_ranking, p2_ranking):
        """Convert ranking differential to implied probability"""
        if p1_ranking == 0 or p2_ranking == 0:
            return 0.5

        ranking_diff = p2_ranking - p1_ranking

        if ranking_diff > 50:
            return 0.85
        elif ranking_diff > 20:
            return 0.75
        elif ranking_diff > 10:
            return 0.65
        elif ranking_diff > 0:
            return 0.55
        elif ranking_diff > -10:
            return 0.45
        elif ranking_diff > -20:
            return 0.35
        elif ranking_diff > -50:
            return 0.25
        else:
            return 0.15

    def calculate_upset_frequency(self, ranking_diff, surface, historical_data):
        """Calculate upset frequency by ranking differential and surface"""
        upset_matches = historical_data[
            ((historical_data['WRank'] - historical_data['LRank']) > ranking_diff) &
            (historical_data['Surface'] == surface)
            ]

        total_matches = historical_data[
            (abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)) &
            (historical_data['Surface'] == surface)
            ]

        if len(total_matches) < 10 and surface != 'fallback':
            return self.calculate_upset_frequency(ranking_diff, 'fallback', historical_data)

        if surface == 'fallback':
            upset_matches = historical_data[
                (historical_data['WRank'] - historical_data['LRank']) > ranking_diff
                ]
            total_matches = historical_data[
                abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)
                ]

        if len(total_matches) == 0:
            return 0.1

        upset_rate = len(upset_matches) / len(total_matches)
        return min(0.45, max(0.05, upset_rate))

    def calculate_surface_performance_ratio(self, player_canonical, surface, opponent_canonical, reference_date):
        """Calculate player's surface-specific performance vs opponent's baseline"""
        player_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
            ].tail(20)

        opponent_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == opponent_canonical) |
             (self.historical_data['loser_canonical'] == opponent_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
            ].tail(20)

        if len(player_surface_matches) < 3 or len(opponent_surface_matches) < 3:
            return 1.0

        player_wins = len(player_surface_matches[player_surface_matches['winner_canonical'] == player_canonical])
        opponent_wins = len(
            opponent_surface_matches[opponent_surface_matches['winner_canonical'] == opponent_canonical])

        player_ratio = player_wins / len(player_surface_matches)
        opponent_ratio = opponent_wins / len(opponent_surface_matches)

        return player_ratio / opponent_ratio if opponent_ratio > 0 else 1.0


# 3.4 Prediction Interface Functions
def convert_to_canonical(name):
    return normalize_name_canonical(name)


def get_matches_for_date(target_date):
    params = {
        "method": "get_fixtures",
        "APIkey": API_KEY,
        "date_start": target_date,
        "date_stop": target_date
    }
    response = requests.get(BASE, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")

    # Surface mapping
    TOURNAMENT_SURFACES = {
        'ATP Wimbledon': 'Grass',
        'WTA Wimbledon': 'Grass',
        'ATP French Open': 'Clay',
        'WTA French Open': 'Clay',
        'ATP US Open': 'Hard',
        'WTA US Open': 'Hard',
        'ATP Australian Open': 'Hard',
        'WTA Australian Open': 'Hard'
    }

    data = response.json()
    matches = []

    for event in data.get("result", []):
        matches.append({
            'event_key': event.get('event_key'),
            'player1_name': event['event_first_player'],
            'player2_name': event['event_second_player'],
            'tournament_name': event.get('tournament_name', 'Unknown'),
            'tournament_round': event.get('tournament_round', ''),
            'event_status': event.get('event_status', ''),
            'event_type_type': event.get('event_type_type', ''),
            'surface': TOURNAMENT_SURFACES.get(event.get('tournament_name', ''), 'Unknown'),
            'time': event.get('event_time', ''),
            'date': event.get('event_date', '')
        })

    return matches

def get_high_confidence_matches(target_date, min_confidence=0.2):
    matches = get_matches_for_date(target_date)

    results = []
    for match in matches:
        p1_canonical = convert_to_canonical(match['player1_name'])
        p2_canonical = convert_to_canonical(match['player2_name'])

        p1_priors = model.extract_refined_priors(p1_canonical, 'men', match['surface'], target_date)
        p2_priors = model.extract_refined_priors(p2_canonical, 'men', match['surface'], target_date)

        p1_win_prob = model.simulate_match(p1_priors, p2_priors)
        confidence = abs(p1_win_prob - 0.5)

        if confidence >= min_confidence:
            favorite = match['player1_name'] if p1_win_prob > 0.5 else match['player2_name']
            win_prob = max(p1_win_prob, 1 - p1_win_prob)

            results.append({
                'match': f"{match['player1_name']} vs {match['player2_name']}",
                'favorite': favorite,
                'probability': win_prob,
                'confidence': confidence
            })

    return sorted(results, key=lambda x: x['confidence'], reverse=True)

# Usage
today = date.today().isoformat()
tomorrow = (date.today() + timedelta(days=1)).isoformat()

todays_matches = get_matches_for_date(today)
tomorrows_matches = get_matches_for_date(tomorrow)
#%%
# Todays_matches or tomorrows_matches
todays_matches
#%%
# Get top 5 picks
def get_top_confidence_matches(target_date, top_n=5, min_confidence=0.05):
    matches = get_matches_for_date(target_date)

    results = []
    for match in matches:
        p1_canonical = convert_to_canonical(match['player1_name'])
        p2_canonical = convert_to_canonical(match['player2_name'])

        p1_priors = model.extract_refined_priors(p1_canonical, 'men', match['surface'], target_date)
        p2_priors = model.extract_refined_priors(p2_canonical, 'men', match['surface'], target_date)
        event_type = str(match.get('event_type_type', '')).lower()
        gender = 'W' if 'wta' in event_type else 'M'
        p1_priors = model.extract_refined_priors(p1_canonical, gender, match['surface'], target_date)
        p2_priors = model.extract_refined_priors(p2_canonical, gender, match['surface'], target_date)

        p1_win_prob = model.simulate_match(p1_priors, p2_priors)
        confidence = abs(p1_win_prob - 0.5)

        if confidence >= min_confidence:
            favorite = match['player1_name'] if p1_win_prob > 0.5 else match['player2_name']
            win_prob = max(p1_win_prob, 1 - p1_win_prob)

            results.append({
                'match': f"{match['player1_name']} vs {match['player2_name']}",
                'favorite': favorite,
                'probability': win_prob,
                'confidence': confidence
            })

    return sorted(results, key=lambda x: x['confidence'], reverse=True)[:top_n]


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    global historical_data, jeff_data, weighted_defaults, model

    REFRESH_CACHE = True

    if REFRESH_CACHE:
        print("Refreshing cache...")
        historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(fast=False)
        save_to_cache(historical_data, jeff_data, weighted_defaults)

        days_back = (date.today() - date(2025, 6, 10)).days
        historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
        save_to_cache(historical_data, jeff_data, weighted_defaults)
    else:
        historical_data, jeff_data, weighted_defaults = load_from_cache()
        if historical_data is None:
            print("Cache miss - generating data...")
            historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(fast=False)
            save_to_cache(historical_data, jeff_data, weighted_defaults)
        else:
            print("✓ Data loaded from cache")

        days_back = (date.today() - date(2025, 6, 10)).days
        historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
        save_to_cache(historical_data, jeff_data, weighted_defaults)

    # Initialize model
    model = BayesianTennisModel()
    model.historical_data = historical_data
    model.jeff_data = jeff_data

    return historical_data, jeff_data, weighted_defaults, model


if __name__ == "__main__":
    target_date = date.today().isoformat()  # today's matches
    picks = get_top_confidence_matches(target_date, top_n=5, min_confidence=0.15)
    # Initialize global variables
    historical_data, jeff_data, weighted_defaults, model = main()

    # Get today's predictions
    target_date = date.today().isoformat()
    picks = get_top_confidence_matches(target_date, top_n=5, min_confidence=0.05)

    print("\n=== TODAY'S TOP PICKS ===")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {pick['match']}")
        print(f"   Favorite: {pick['favorite']}")
        print(f"   Win Prob: {pick['probability']:.2%}")
        print(f"   Confidence: {pick['confidence']:.5%}\n")
#%%
# See picks`
from datetime import date

# get today’s top-5 at 5% confidence
picks = get_top_confidence_matches(date.today().isoformat(), top_n=5, min_confidence=0.05)

# print them
for i, pick in enumerate(picks, 1):
    print(f"{i}. {pick['match']}")
    print(f"   Favorite: {pick['favorite']}")
    print(f"   Win Prob: {pick['probability']:.2%}")
    print(f"   Confidence: {pick['confidence']:.1%}\n")
#%%
import pandas as pd

pd.DataFrame(picks)
#%%
# Split data chronologically
split_date = '2023-01-01'
train_data = historical_data[pd.to_datetime(historical_data['Date']) < split_date]
test_data = historical_data[pd.to_datetime(historical_data['Date']) >= split_date]

# Initialize model with training data
model.historical_data = train_data

# Run evaluation
accuracy = model.evaluate_predictions(test_data.head(100))
print(f"Enhanced model accuracy: {accuracy:.3f}")

# Compare with baseline
model_baseline = BayesianTennisModel()
model_baseline.historical_data = train_data
baseline_accuracy = model_baseline.evaluate_predictions(test_data.head(100))
print(f"Baseline accuracy: {baseline_accuracy:.3f}")
print(f"Improvement: {accuracy - baseline_accuracy:.3f}")

# %%
# ─── optional scraper smoke-test ─────────────────────────────────────────
if __name__ == "__main__" and "--test" in sys.argv:

    # TEST - Tennis-Abstract scraper (shot-direction + shot-types)

    from pathlib import Path
    import json, sys

    scraper = TennisAbstractScraper()

    TEST_URLS = [
        # Wimbledon 2025 – Sherif v. Andreeva (already validated)
        "https://www.tennisabstract.com/charting/20250701-W-Wimbledon-R128-Mayar_Sherif-Mirra_Andreeva.html",
        # add more charting URLs dated > 2025-06-10 as they become available …
    ]

    def _quick_check_direction(records: list[dict]) -> None:
        """
        Sanity‑check for shot‑direction output.

        • summary table: exactly 4 rows per player
        • outcome table: TA pages now show 8–13 rows per player
          (12 historical baseline + optional Slice rows or missing rows
           on incomplete feeds).  Accept any count in that range.
        """
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in records:
            key = (r["Player_canonical"], r["category_group"])
            groups.setdefault(key, []).append(r)

        for (player, cat_grp), rows in groups.items():
            if cat_grp == "direction_summary":
                if len(rows) != 4:
                    raise AssertionError(
                        f"{player} {cat_grp}: {len(rows)} rows (expected 4)"
                    )
            elif cat_grp == "direction_outcome":
                if not (8 <= len(rows) <= 13):
                    raise AssertionError(
                        f"{player} {cat_grp}: {len(rows)} rows (expected 8–13)"
                    )

    def _quick_check_types(records: list[dict]) -> None:
        """basic presence check – just ensure at least one ‘Total’ row per player"""
        seen = set()
        for r in records:
            if r["category"] == "Total":
                seen.add(r["Player_canonical"])
        if not seen:
            raise AssertionError("no ‘Total’ row found in shot-types output")

    for url in TEST_URLS:
        print(f"→ {url}", file=sys.stderr)

        shot_dir  = scraper.scrape_shot_direction(url)
        shot_type = scraper.scrape_shot_types(url)

        _quick_check_direction(shot_dir)
        _quick_check_types(shot_type)
    print("SCRAPER OK")



        print(f"   Confidence: {pick['confidence']:.1%}\n")
