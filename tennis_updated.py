
# ============================================================================
# TENNIS DATA PIPELINE - COMPREHENSIVE TENNIS PREDICTION SYSTEM
# ============================================================================

# ============================================================================
# 1. IMPORTS AND CONSTANTS
# ============================================================================
import logging
import numpy as np
import pandas as pd
import random
from datetime import date, timedelta
import os
import requests
import pickle
import html
import re
from pathlib import Path
from unidecode import unidecode
from bs4 import BeautifulSoup, FeatureNotFound
import argparse
import collections

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

SESSION = requests.Session()

def flatten_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    """
    Utility to flatten a list of fixture dicts into a DataFrame.
    """
    return pd.json_normalize(fixtures)

def api_call(method: str, **params):
    """
    Wrapper for API-Tennis endpoints. Uses query param "method" rather than path.
    Returns the parsed JSON response.
    """
    url = BASE
    params["method"] = method
    params["APIkey"] = API_KEY
    logging.info(f"API request URL: {url} with params: {params}")
    try:
        response = SESSION.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Debug logging
        logging.info(f"API response keys: {list(data.keys())}")

        # Check if response indicates success
        if data.get("success") == 1:
            return data.get("result", [])
        else:
            logging.error(f"API returned unsuccessful response: {data}")
            return []

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error for method {method}: {e}")
        return []
    except ValueError as e:
        logging.error(f"JSON decode error for method {method}: {e}")
        return []
    except Exception as e:
        logging.error(f"API call error for method {method} with params {params}: {e}")
        return []

#%%
# ============================================================================
# API CONFIGURATION AND CORE FUNCTIONS
# ============================================================================

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
BASE_CUTOFF_DATE = date(2025, 6, 10)
JEFF_DB_PATH = os.path.join(DATA_DIR, "jeff_tennis_data_until_20250610.parquet")
INCR_DB_PATH = os.path.join(DATA_DIR, "results_incremental.parquet")
CHARTING_INDEX_CSV = (
    "https://raw.githubusercontent.com/JeffSackmann/"
    "tennis_charting/master/charting_match_index.csv"
)


# ============================================================================
# 2. DATA INTEGRATION
# ============================================================================




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


def normalize_tournament_name(name, gender=None):
    """Normalize tournament names with gender awareness"""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()

    # If already has ATP/WTA prefix, just normalize case
    if name.startswith('atp ') or name.startswith('wta '):
        return name

    # Major tournaments mapping
    major_tournaments = {
        'wimbledon': 'wimbledon',
        'french open': 'french open',
        'roland garros': 'french open',
        'us open': 'us open',
        'australian open': 'australian open'
    }

    if name in major_tournaments:
        base_name = major_tournaments[name]
        if gender == 'M':
            name = f'atp {base_name}'
        elif gender == 'W':
            name = f'wta {base_name}'
        else:
            # Fallback to ATP if gender unknown
            name = f'atp {base_name}'

    # Other normalizations
    name = name.replace('masters cup', 'masters')
    name = name.replace('atp finals', 'masters')
    name = name.replace('wta finals', 'masters')

    return name.strip()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


# 2.3 Data Loading Functions
def load_excel_data(file_path):
    """Load data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        if 'Date' not in df.columns:
            logging.warning(f"Warning: No Date column in {file_path}")
            return pd.DataFrame()
        logging.info(f"Loaded {len(df)} matches from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
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
                    logging.info(f"Loaded {gender}/{filename}: {len(df)} records")
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

# grab JS blobs   var shots1 = '…';   var shots2 = '…';
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
def get_fixtures_for_date(target_date, event_type_key=None):
    """Get all fixtures for a specific date"""
    # Determine ATP Singles event type key if not provided
    if event_type_key is None:
        events = api_call("get_events")
        event_type_key = next(
            (safe_int_convert(e.get("event_type_key")) for e in events if e.get("event_type_type") == "Atp Singles"),
            None
        )
    try:
        fixtures = api_call(
            "get_fixtures",
            date_start=target_date.isoformat(),
            date_stop=target_date.isoformat(),
            event_type_key=event_type_key,
            timezone="UTC"
        )
        # Return all fixtures regardless of status to capture completed matches
        return fixtures
    except Exception as e:
        logging.error(f"Error getting fixtures for {target_date}: {e}")
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
                score_first = safe_int_convert(s.get("score_first", 0)) or 0
                score_second = safe_int_convert(s.get("score_second", 0)) or 0
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

# (REMOVED: old parse_match_statistics)


# Inserted updated parse_match_statistics function after extract_embedded_statistics
def parse_api_tennis_statistics(fixture: dict) -> dict[int, dict]:
    """
    Parse API-Tennis fixture statistics format into per-player dicts.

    The API returns statistics as a list of dicts with format:
    {
        'player_key': 372,
        'stat_period': 'match',
        'stat_type': 'Service',
        'stat_name': 'Aces',
        'stat_value': '13',
        'stat_won': None,
        'stat_total': None
    }

    We convert this to the expected p1_/p2_ format.
    """
    try:
        statistics = fixture.get('statistics', [])
        if not statistics:
            return {}

        p1_key = safe_int_convert(fixture.get("first_player_key"))
        p2_key = safe_int_convert(fixture.get("second_player_key"))

        if not p1_key or not p2_key:
            return {}

        player_stats = {p1_key: {}, p2_key: {}}

        # Process only match-level statistics
        for stat in statistics:
            if stat.get('stat_period') != 'match':
                continue

            player_key = safe_int_convert(stat.get('player_key'))
            if player_key not in player_stats:
                continue

            stat_type = stat.get('stat_type', '').lower()
            stat_name = stat.get('stat_name', '').lower()
            stat_value = stat.get('stat_value', '')
            stat_won = stat.get('stat_won')
            stat_total = stat.get('stat_total')

            # Create normalized key name
            key_name = f"{stat_type}_{stat_name}".replace(' ', '_').replace('%', '_pct')

            # Parse the value
            if stat_won is not None and stat_total is not None:
                # Use actual counts when available
                player_stats[player_key][f"{key_name}_won"] = stat_won
                player_stats[player_key][f"{key_name}_total"] = stat_total
                if stat_total > 0:
                    player_stats[player_key][f"{key_name}_pct"] = stat_won / stat_total
            else:
                # Parse the stat_value
                if '%' in str(stat_value):
                    # Percentage value
                    pct_val = float(str(stat_value).replace('%', '')) / 100
                    player_stats[player_key][f"{key_name}_pct"] = pct_val
                else:
                    # Raw number
                    try:
                        # Handle values like "194 km/h" by extracting just the number
                        import re
                        numeric_match = re.search(r'(\d+)', str(stat_value))
                        if numeric_match:
                            player_stats[player_key][key_name] = int(numeric_match.group(1))
                        else:
                            player_stats[player_key][key_name] = stat_value
                    except:
                        player_stats[player_key][key_name] = stat_value

        return player_stats

    except Exception as e:
        logging.error(f"Error parsing API tennis statistics: {e}")
        return {}


# Replace the parse_match_statistics function in tennis_updated.py with this version:
def parse_match_statistics(fixture: dict) -> dict[int, dict]:
    """
    Parse match statistics from API-Tennis fixture format.
    Returns a mapping from player_key (int) to a dict of their stats.
    """
    # Try the new API format first
    api_stats = parse_api_tennis_statistics(fixture)
    if api_stats:
        return api_stats

    # Fallback to old flatten format (for compatibility)
    try:
        df = flatten_fixtures([fixture])
        if df.empty:
            return {}
        row = df.iloc[0]
        stats = {}
        p1_key = safe_int_convert(fixture.get("first_player_key"))
        p2_key = safe_int_convert(fixture.get("second_player_key"))
        p1_stats = {col[len("p1_"):]: row[col] for col in df.columns if col.startswith("p1_")}
        p2_stats = {col[len("p2_"):]: row[col] for col in df.columns if col.startswith("p2_")}
        if p1_key is not None:
            stats[p1_key] = p1_stats
        if p2_key is not None:
            stats[p2_key] = p2_stats
        return stats
    except Exception:
        return {}

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
        logging.error(f"Error getting odds for match {match_key}: {e}")
        return (None, None)


def get_player_rankings(day, league="ATP"):
    """Get standings with proper caching and error handling"""
    tag = f"{league}_{day.isocalendar()[0]}_{day.isocalendar()[1]:02d}.pkl"
    cache_file = CACHE_API / tag

    if cache_file.exists():
        try:
            standings = pickle.loads(cache_file.read_bytes())
            if standings:
                rankings = {}
                for r in standings:
                    player_key = safe_int_convert(r.get("player_key"))
                    place = safe_int_convert(r.get("place"))
                    if player_key is not None and place is not None:
                        rankings[player_key] = place
                return rankings
        except Exception as e:
            logging.error(f"Cache read error for {tag}: {e}")

    # Correct parameter for API
    standings = api_call("get_standings", event_type=league.upper())

    try:
        cache_file.write_bytes(pickle.dumps(standings, 4))
    except Exception as e:
        logging.error(f"Cache write error for {tag}: {e}")

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
    """Get tournament metadata from fixtures since API endpoint is restricted"""
    cache_file = CACHE_API / "tournaments.pkl"

    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass

    # Extract tournament data from recent fixtures instead of API
    tournament_dict = {}

    try:
        for delta in range(30):  # Check last 30 days
            day = date.today() - timedelta(days=delta)
            fixtures = get_fixtures_for_date(day)

            for fixture in fixtures:
                tournament_key = safe_int_convert(fixture.get('tournament_key'))
                if tournament_key is not None:
                    tournament_dict[str(tournament_key)] = {
                        'tournament_name': fixture.get('tournament_name'),
                        'tournament_key': tournament_key,
                        'tournament_round': fixture.get('tournament_round'),
                        'tournament_season': fixture.get('tournament_season'),
                        'event_type_type': fixture.get('event_type_type'),
                        # Add surface data if available in future
                        'surface': 'Unknown'  # Not in current fixture format
                    }

        cache_file.write_bytes(pickle.dumps(tournament_dict, 4))
        logging.info(f"Extracted tournament data for {len(tournament_dict)} tournaments from fixtures")
        return tournament_dict

    except Exception as e:
        logging.error(f"Error extracting tournament data from fixtures: {e}")
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
        logging.error(f"Error getting event types: {e}")
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
        logging.error(f"Error getting player {player_key}: {e}")
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
    logging.info("=== STARTING DATA GENERATION ===")

    # Step 1: Load Jeff's data
    logging.info("Step 1: Loading Jeff's comprehensive data...")
    try:
        jeff_data = load_jeff_comprehensive_data()
        if not jeff_data or ('men' not in jeff_data and 'women' not in jeff_data):
            logging.error("ERROR: Jeff data loading failed")
            return pd.DataFrame(), {}, {}

        logging.info(f"✓ Jeff data loaded successfully")
        logging.info(f"  - Men's datasets: {len(jeff_data.get('men', {}))}")
        logging.info(f"  - Women's datasets: {len(jeff_data.get('women', {}))}")

    except Exception as e:
        logging.error(f"ERROR loading Jeff data: {e}")
        return pd.DataFrame(), {}, {}

    # Step 2: Calculate weighted defaults
    logging.info("Step 2: Calculating weighted defaults...")
    try:
        weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)
        if not weighted_defaults:
            logging.error("ERROR: Weighted defaults calculation failed")
            return pd.DataFrame(), jeff_data, {}

        logging.info(f"✓ Weighted defaults calculated")
        logging.info(f"  - Men's features: {len(weighted_defaults.get('men', {}))}")
        logging.info(f"  - Women's features: {len(weighted_defaults.get('women', {}))}")

    except Exception as e:
        logging.error(f"ERROR calculating weighted defaults: {e}")
        return pd.DataFrame(), jeff_data, {}

    # Step 3: Load tennis match data
    logging.info("Step 3: Loading tennis match data...")
    try:
        tennis_data = load_all_tennis_data()
        if tennis_data.empty:
            logging.error("ERROR: No tennis data loaded")
            return pd.DataFrame(), jeff_data, weighted_defaults

        logging.info(f"✓ Tennis data loaded: {len(tennis_data)} matches")

        # Fast mode for testing
        if fast:
            total_rows = len(tennis_data)
            take = min(n_sample, total_rows)
            tennis_data = tennis_data.sample(take, random_state=1).reset_index(drop=True)
            logging.info(f"[FAST MODE] Using sample of {take}/{total_rows} rows")

    except Exception as e:
        logging.error(f"ERROR loading tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 4: Process tennis data
    logging.info("Step 4: Processing tennis data...")
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

        logging.info(f"✓ Tennis data processed")

    except Exception as e:
        logging.error(f"ERROR processing tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 5: Adding Jeff feature columns...
    logging.info("Step 5: Adding Jeff feature columns...")
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

        logging.info(f"✓ Added/verified {len(all_jeff_features) * 2} feature columns")

    except Exception as e:
        logging.error(f"ERROR adding feature columns: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 6: Extract Jeff features
    logging.info("Step 6: Extracting Jeff features...")
    try:
        total_matches = len(tennis_data)
        matches_with_jeff_features = 0

        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                logging.info(f"  Processing match {idx}/{total_matches}")

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
                    logging.warning(f"Warning: Error processing match {idx}: {e}")
                continue

        logging.info(f"✓ Jeff features extracted for {matches_with_jeff_features}/{total_matches} matches")

    except Exception as e:
        logging.error(f"ERROR extracting Jeff features: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    # Step 7: Integrate API-Tennis and Tennis-Abstract data
    logging.info("Step 7: Integrating API and TA data...")
    tennis_data = integrate_api_tennis_data(tennis_data, days_back=7)

    logging.info(f"=== DATA GENERATION COMPLETE ===")
    logging.info(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults

# ============================================================================
# API-TENNIS DATA INTEGRATION
# ============================================================================

def integrate_api_tennis_data_incremental(historical_data):
    """Fetch API data for dates not already in dataset with full feature integration"""
    df = historical_data.copy()
    if "source_rank" not in df.columns:
        df["source_rank"] = 3
    else:
        df["source_rank"] = df["source_rank"].fillna(3)

    # Find what dates we already have API data for (source_rank = 2)
    existing_api_dates = set(df[df["source_rank"] == 2]["date"].dropna())

    # Get all dates from June 10, 2025 to today
    start_date = date(2025, 6, 10)
    end_date = date.today()

    dates_to_fetch = []
    current = start_date
    while current <= end_date:
        if current not in existing_api_dates:
            dates_to_fetch.append(current)
        current += timedelta(days=1)

    if not dates_to_fetch:
        print("All API data already cached.")
        return df

    print(f"Fetching API data for {len(dates_to_fetch)} new dates...")

    # Get all available event types and tournament metadata
    events = api_call("get_events")
    event_type_keys = [e.get("event_type_key") for e in events if e.get("event_type_key")]
    tournament_metadata = get_tournaments_metadata()
    event_types = get_event_types()

    for day in dates_to_fetch:
        print(f"  Fetching {day}...")

        # Get current rankings for this date
        atp_rankings = get_player_rankings(day, "ATP")
        wta_rankings = get_player_rankings(day, "WTA")

        # Get fixtures for all event types
        all_fixtures = []
        for event_type_key in event_type_keys:
            fixtures = api_call(
                "get_fixtures",
                date_start=day.isoformat(),
                date_stop=day.isoformat(),
                event_type_key=event_type_key,
                timezone="UTC"
            )
            all_fixtures.extend(fixtures)

        for fixture in all_fixtures:
            # Extract player keys
            p1_key = safe_int_convert(fixture.get("first_player_key"))
            p2_key = safe_int_convert(fixture.get("second_player_key"))

            if not p1_key or not p2_key:
                continue

            # Infer gender from event type
            event_type = fixture.get("event_type_type", "").lower()
            if "wta" in event_type or "women" in event_type or "girls" in event_type:
                gender = "W"
                rankings = wta_rankings
            elif "atp" in event_type or "men" in event_type or "boys" in event_type:
                gender = "M"
                rankings = atp_rankings
            else:
                gender = None
                rankings = {}

            # Get tournament info
            tournament_key = fixture.get("tournament_key")
            tournament_info = tournament_metadata.get(str(tournament_key), {}) if tournament_key else {}
            surface = tournament_info.get("surface", "Unknown")

            # Get event type info
            event_type_key = fixture.get("event_type_key")
            event_info = event_types.get(str(event_type_key), {}) if event_type_key else {}

            comp_id = build_composite_id(
                day,
                normalize_tournament_name(fixture.get("tournament_name", ""), gender),
                normalize_name(fixture.get("event_first_player", "")),
                normalize_name(fixture.get("event_second_player", ""))
            )

            # Get all API data
            stats_map = parse_match_statistics(fixture)
            embed = extract_embedded_statistics(fixture)
            h2h_data = get_h2h_data(p1_key, p2_key)
            odds1, odds2 = get_match_odds(fixture.get("match_key"), day)

            # Get player profiles for additional stats
            p1_profile = get_player_profile(p1_key)
            p2_profile = get_player_profile(p2_key)

            if not stats_map:
                continue

            stats_dict = next(iter(stats_map.values()))
            record = {**stats_dict, **embed}

            # Add comprehensive API features
            record.update({
                "composite_id": comp_id,
                "source_rank": 2,
                "date": day,
                "gender": gender,
                "surface": surface,
                "tournament_tier": event_info.get("event_type_type", ""),

                # Player rankings
                "p1_ranking": rankings.get(p1_key),
                "p2_ranking": rankings.get(p2_key),
                "ranking_difference": abs(rankings.get(p1_key, 999) - rankings.get(p2_key, 999)),

                # Head-to-head data
                "h2h_matches": h2h_data.get("h2h_matches", 0),
                "p1_h2h_wins": h2h_data.get("p1_wins", 0),
                "p2_h2h_wins": h2h_data.get("p2_wins", 0),
                "p1_h2h_win_pct": h2h_data.get("p1_win_pct", 0.5),

                # Betting odds and implied probabilities
                "odds_p1": odds1,
                "odds_p2": odds2,
                "implied_prob_p1": 1 / odds1 if odds1 and odds1 > 0 else None,
                "implied_prob_p2": 1 / odds2 if odds2 and odds2 > 0 else None,

                # Player profile data
                "p1_age": p1_profile.get("player_age"),
                "p2_age": p2_profile.get("player_age"),
                "p1_country": p1_profile.get("player_country"),
                "p2_country": p2_profile.get("player_country"),

                # Tournament context
                "tournament_key": tournament_key,
                "tournament_round": fixture.get("tournament_round"),
                "tournament_season": fixture.get("tournament_season"),
            })

            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df = df.sort_values("source_rank").drop_duplicates(subset="composite_id", keep="first").reset_index(drop=True)
    print(f"Added {len(dates_to_fetch)} days of API data to cache.")
    return df

def verify_match_coverage(start_date, end_date, auto_fix=False):
    """Check if all matches are captured for date range"""
    hist, _, _ = load_from_cache()
    events = api_call("get_events")
    event_type_keys = [e.get("event_type_key") for e in events if e.get("event_type_key")]

    missing_data = []

    current = start_date
    while current <= end_date:
        print(f"Checking {current}...")

        # Count cached matches for this date
        cached_count = len(hist[(hist["date"] == current) & (hist["source_rank"] == 2)])

        # Count API matches for this date
        api_count = 0
        for event_type_key in event_type_keys:
            fixtures = api_call(
                "get_fixtures",
                date_start=current.isoformat(),
                date_stop=current.isoformat(),
                event_type_key=event_type_key,
                timezone="UTC"
            )
            api_count += len(fixtures)

        if cached_count != api_count:
            missing_data.append({
                'date': current,
                'cached': cached_count,
                'api': api_count,
                'missing': api_count - cached_count
            })
            print(f"  MISMATCH: Cached={cached_count}, API={api_count}")
        else:
            print(f"  OK: {cached_count} matches")

        current += timedelta(days=1)

    return missing_data


def repair_missing_matches(missing_data_list):
    """Fix dates with missing match data"""
    if not missing_data_list:
        print("No missing data to repair")
        return

    hist, jeff_data, defaults = load_from_cache()
    events = api_call("get_events")
    event_type_keys = [e.get("event_type_key") for e in events if e.get("event_type_key")]

    for issue in missing_data_list:
        day = issue['date']
        print(f"Repairing {day} (missing {issue['missing']} matches)...")

        # Remove existing API data for this date
        hist = hist[~((hist["date"] == day) & (hist["source_rank"] == 2))]

        # Re-fetch all matches for this date
        all_fixtures = []
        for event_type_key in event_type_keys:
            fixtures = api_call(
                "get_fixtures",
                date_start=day.isoformat(),
                date_stop=day.isoformat(),
                event_type_key=event_type_key,
                timezone="UTC"
            )
            all_fixtures.extend(fixtures)

        # Process fixtures (same logic as integrate_api_tennis_data_incremental)
        for fixture in all_fixtures:
    # [same processing logic as before]
    # Add to hist with source_rank=2

    # Save repaired cache
    save_to_cache(hist, jeff_data, defaults)
    print(f"Repaired {len(missing_data_list)} dates")
# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

def save_to_cache(historical_data, jeff_data, weighted_defaults):
    """Save data to cache"""
    logging.info("\n=== SAVING TO CACHE ===")
    # --- sanitize numeric columns that sometimes contain stray text ---
    numeric_cols = ["MaxW", "MaxL", "AvgW", "AvgL", "PSW", "PSL"]
    for col in numeric_cols:
        if col in historical_data.columns:
            historical_data[col] = pd.to_numeric(historical_data[col], errors="coerce")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # Save historical data
        historical_data.to_parquet(HD_PATH, index=False)
        logging.info("✓ Historical data saved")

        # Save Jeff data
        with open(JEFF_PATH, "wb") as f:
            pickle.dump(jeff_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("✓ Jeff data saved")

        # Save weighted defaults
        with open(DEF_PATH, "wb") as f:
            pickle.dump(weighted_defaults, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("✓ Weighted defaults saved")

        return True
    except Exception as e:
        logging.error(f"ERROR saving cache: {e}")
        return False

def load_from_cache():
    """Load data from cache if available, with graceful fallback."""
    if (os.path.exists(HD_PATH) and
        os.path.exists(JEFF_PATH) and
        os.path.exists(DEF_PATH)):
        try:
            logging.info("Loading from cache...")
            historical_data = pd.read_parquet(HD_PATH)
            with open(JEFF_PATH, "rb") as f:
                jeff_data = pickle.load(f)
            with open(DEF_PATH, "rb") as f:
                weighted_defaults = pickle.load(f)
            return historical_data, jeff_data, weighted_defaults
        except Exception as e:
            logging.warning(f"Cache load failed, regenerating data: {e}")
    return None, None, None

def safe_int_convert(value):
    """
    Safely convert a value to int. Returns None if conversion fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_match_statistics(fixture: dict) -> dict[int, dict]:
    """
    Flatten API-Tennis fixture statistics into per-player dicts.
    Returns a mapping from player_key (int) to a dict of their wide stats.
    """
    df = flatten_fixtures([fixture])
    stats = {}
    if df.empty:
        return stats
    row = df.iloc[0]
    # Player keys
    p1_key = safe_int_convert(fixture.get("first_player_key"))
    p2_key = safe_int_convert(fixture.get("second_player_key"))
    # Extract p1_ and p2_ prefixed columns
    p1_stats = {col[len("p1_"):]: row[col] for col in df.columns if col.startswith("p1_")}
    p2_stats = {col[len("p2_"):]: row[col] for col in df.columns if col.startswith("p2_")}
    if p1_key is not None:
        stats[p1_key] = p1_stats
    if p2_key is not None:
        stats[p2_key] = p2_stats
    return stats

# %%

import random


class BayesianTennisModel:
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

        # Surface adjustment factors
        self.surface_adjustments = {
            "Clay": {"serve_advantage": 0.95, "rally_importance": 1.1},
            "Grass": {"serve_advantage": 1.1, "rally_importance": 0.9},
            "Hard": {"serve_advantage": 1.0, "rally_importance": 1.0},
            "Unknown": {"serve_advantage": 1.0, "rally_importance": 1.0}
        }

    def _calculate_ranking_factor(self, p1_rank, p2_rank):
        """Calculate ranking-based probability adjustment"""
        if pd.isna(p1_rank) or pd.isna(p2_rank):
            return 0.5

        # Convert rankings to strength (lower rank = higher strength)
        p1_strength = 1000 / (p1_rank + 50)  # +50 to avoid division issues
        p2_strength = 1000 / (p2_rank + 50)

        # Convert to probability
        total_strength = p1_strength + p2_strength
        return p1_strength / total_strength if total_strength > 0 else 0.5

    def _calculate_h2h_factor(self, h2h_matches, p1_h2h_win_pct):
        """Calculate head-to-head probability adjustment with confidence weighting"""
        if h2h_matches == 0:
            return 0.5

        # Weight H2H by number of matches (more matches = more reliable)
        confidence = min(h2h_matches / 10.0, 1.0)  # Max confidence at 10+ matches
        base_prob = 0.5

        return base_prob + confidence * (p1_h2h_win_pct - base_prob)

    def _calculate_odds_factor(self, implied_prob_p1, implied_prob_p2):
        """Calculate market-implied probability with vig removal"""
        if pd.isna(implied_prob_p1) or pd.isna(implied_prob_p2):
            return 0.5

        # Remove bookmaker vig (overround)
        total_implied = implied_prob_p1 + implied_prob_p2
        if total_implied <= 0:
            return 0.5

        return implied_prob_p1 / total_implied

    def _estimate_point_win_prob(self, p1_stats: dict, p2_stats: dict, match_context: dict = None):
        """Enhanced point probability estimation with API features"""
        # Base probability from serve/return stats
        serve_pts = p1_stats.get("serve_pts", 1)
        first_won = p1_stats.get("first_serve_won", 0)
        return_pts = p1_stats.get("return_pts_won", 0)
        opp_serve_pts = p2_stats.get("serve_pts", 1)

        # Offensive component: player1's first serve win rate
        off = first_won / serve_pts if serve_pts else 0.5
        # Defensive component: player1's return success
        defense = return_pts / opp_serve_pts if opp_serve_pts else 0.5
        base_prob = min(max((off + defense) / 2, 0.01), 0.99)

        if not match_context:
            return base_prob

        # Apply surface adjustments
        surface = match_context.get("surface", "Unknown")
        surface_adj = self.surface_adjustments.get(surface, self.surface_adjustments["Unknown"])

        # Adjust for serve advantage on different surfaces
        if off > defense:  # Player 1 is more serve-oriented
            base_prob *= surface_adj["serve_advantage"]
        else:  # Player 1 is more defensive/return-oriented
            base_prob *= surface_adj["rally_importance"]

        # Incorporate ranking factor
        ranking_prob = self._calculate_ranking_factor(
            match_context.get("p1_ranking"),
            match_context.get("p2_ranking")
        )

        # Incorporate H2H factor
        h2h_prob = self._calculate_h2h_factor(
            match_context.get("h2h_matches", 0),
            match_context.get("p1_h2h_win_pct", 0.5)
        )

        # Incorporate odds factor
        odds_prob = self._calculate_odds_factor(
            match_context.get("implied_prob_p1"),
            match_context.get("implied_prob_p2")
        )

        # Weighted combination of factors
        weights = {
            "stats": 0.4,  # Match statistics
            "ranking": 0.25,  # Current form/ranking
            "h2h": 0.15,  # Historical matchup
            "odds": 0.2  # Market expectation
        }

        # Only use available factors
        available_factors = [("stats", base_prob)]
        if not pd.isna(ranking_prob):
            available_factors.append(("ranking", ranking_prob))
        if h2h_prob != 0.5:  # Only if H2H data exists
            available_factors.append(("h2h", h2h_prob))
        if odds_prob != 0.5:  # Only if odds data exists
            available_factors.append(("odds", odds_prob))

        # Normalize weights for available factors
        total_weight = sum(weights[name] for name, _ in available_factors)

        combined_prob = sum(
            weights[name] * prob for name, prob in available_factors
        ) / total_weight

        return min(max(combined_prob, 0.01), 0.99)

    def simulate_set(self, p1_stats, p2_stats, match_context=None):
        """Simulate one set with enhanced probability calculation"""
        p1_games = p2_games = 0
        p_point = self._estimate_point_win_prob(p1_stats, p2_stats, match_context)

        while True:
            # Simulate a game to 4 points, win by 2
            p1_points = p2_points = 0
            while True:
                if random.random() < p_point:
                    p1_points += 1
                else:
                    p2_points += 1
                if (p1_points >= 4 or p2_points >= 4) and abs(p1_points - p2_points) >= 2:
                    break

            winner = 1 if p1_points > p2_points else 2
            if winner == 1:
                p1_games += 1
            else:
                p2_games += 1

            # Check set win
            if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                return 1 if p1_games > p2_games else 2
            if p1_games == 6 and p2_games == 6:
                # Tiebreak at 6-6
                tb_winner = self.simulate_tiebreak(p1_stats, p2_stats, match_context)
                return tb_winner

    def simulate_tiebreak(self, p1_stats, p2_stats, match_context=None):
        """Simulate a tiebreak with enhanced probability calculation"""
        p1_score = p2_score = 0
        p_point = self._estimate_point_win_prob(p1_stats, p2_stats, match_context)

        while True:
            if random.random() < p_point:
                p1_score += 1
            else:
                p2_score += 1
            if (p1_score >= 7 or p2_score >= 7) and abs(p1_score - p2_score) >= 2:
                return 1 if p1_score > p2_score else 2

    def simulate_match(self, p1_stats, p2_stats, match_context=None, best_of: int = 3):
        """Simulate a match with full API feature integration"""
        results = []
        for _ in range(self.n_simulations):
            p1_sets = p2_sets = 0
            for _ in range(best_of):
                winner = self.simulate_set(p1_stats, p2_stats, match_context)
                if winner == 1:
                    p1_sets += 1
                else:
                    p2_sets += 1
                if p1_sets > best_of // 2 or p2_sets > best_of // 2:
                    break
            results.append(1 if p1_sets > p2_sets else 0)
        return sum(results) / len(results)


# Updated main prediction function
def predict_match_with_full_api_integration(args, hist, jeff_data, defaults):
    """Enhanced prediction function using all available API features"""

    # Build composite_id for lookup
    match_date = pd.to_datetime(args.date).date()
    comp_id1 = build_composite_id(match_date, normalize_tournament_name(args.tournament, args.gender),
                                  normalize_name(args.player1), normalize_name(args.player2))
    comp_id2 = build_composite_id(match_date, normalize_tournament_name(args.tournament, args.gender),
                                  normalize_name(args.player2), normalize_name(args.player1))

    print(f"Looking for match: {comp_id1}")
    print(f"Alternative order: {comp_id2}")

    row = hist[hist["composite_id"] == comp_id1]
    player_order_swapped = False

    if row.empty:
        print(f"First order not found, trying alternative...")
        row = hist[hist["composite_id"] == comp_id2]
        if not row.empty:
            print(f"Found match with swapped player order")
            player_order_swapped = True

    if row.empty:
        print(f"No data for either order:")
        print(f"  {comp_id1}")
        print(f"  {comp_id2}")
        return None

    # Extract all available features
    match_row = row.iloc[0]

    # Extract player stats (accounting for potential player order swap)
    if player_order_swapped:
        p1_stats = {col[len("loser_"):]: match_row[col] for col in hist.columns if col.startswith("loser_")}
        p2_stats = {col[len("winner_"):]: match_row[col] for col in hist.columns if col.startswith("winner_")}
        # Swap API features too
        match_context = {
            "surface": match_row.get("surface"),
            "p1_ranking": match_row.get("p2_ranking"),
            "p2_ranking": match_row.get("p1_ranking"),
            "h2h_matches": match_row.get("h2h_matches", 0),
            "p1_h2h_win_pct": 1 - match_row.get("p1_h2h_win_pct", 0.5),  # Flip H2H
            "implied_prob_p1": match_row.get("implied_prob_p2"),
            "implied_prob_p2": match_row.get("implied_prob_p1"),
        }
    else:
        p1_stats = {col[len("winner_"):]: match_row[col] for col in hist.columns if col.startswith("winner_")}
        p2_stats = {col[len("loser_"):]: match_row[col] for col in hist.columns if col.startswith("loser_")}
        match_context = {
            "surface": match_row.get("surface"),
            "p1_ranking": match_row.get("p1_ranking"),
            "p2_ranking": match_row.get("p2_ranking"),
            "h2h_matches": match_row.get("h2h_matches", 0),
            "p1_h2h_win_pct": match_row.get("p1_h2h_win_pct", 0.5),
            "implied_prob_p1": match_row.get("implied_prob_p1"),
            "implied_prob_p2": match_row.get("implied_prob_p2"),
        }

    # Run enhanced prediction
    model = BayesianTennisModel()
    prob = model.simulate_match(p1_stats, p2_stats, match_context, best_of=args.best_of)

    # Print feature breakdown
    print(f"\n=== PREDICTION FEATURES ===")
    print(f"Surface: {match_context.get('surface', 'Unknown')}")
    print(f"Rankings: P1={match_context.get('p1_ranking', 'N/A')}, P2={match_context.get('p2_ranking', 'N/A')}")
    print(
        f"H2H Record: {match_context.get('h2h_matches', 0)} matches, P1 win rate: {match_context.get('p1_h2h_win_pct', 0.5):.1%}")
    if match_context.get('implied_prob_p1'):
        print(
            f"Market Odds: P1={match_context.get('implied_prob_p1'):.1%}, P2={match_context.get('implied_prob_p2'):.1%}")

    return prob


# Update the main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis match win probability predictor")
    parser.add_argument("--player1", required=True, help="Name of player 1")
    parser.add_argument("--player2", required=True, help="Name of player 2")
    parser.add_argument("--date", required=True, help="Match date in YYYY-MM-DD")
    parser.add_argument("--tournament", required=True, help="Tournament name slug")
    parser.add_argument("--gender", choices=["M", "W"], required=True, help="Gender: M or W")
    parser.add_argument("--best_of", type=int, default=3, help="Sets in match, default 3")
    args = parser.parse_args()

    # Load or generate data
    hist, jeff_data, defaults = load_from_cache()
    if hist is None:
        print("No cache found. Generating full historical dataset...")
        hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=False)
        save_to_cache(hist, jeff_data, defaults)
        print("Historical data cached for future use.")
    else:
        print("Loaded historical data from cache.")

    # Integrate recent API data with full feature set
    print("Integrating recent API data with full feature extraction...")
    hist = integrate_api_tennis_data_incremental(hist)
    save_to_cache(hist, jeff_data, defaults)

    # Run enhanced prediction
    prob = predict_match_with_full_api_integration(args, hist, jeff_data, defaults)

    if prob is not None:
        print(f"\n=== FINAL PREDICTION ===")
        print(f"P({args.player1} wins) = {prob:.3f}")
        print(f"P({args.player2} wins) = {1 - prob:.3f}")
    else:
        print("Prediction failed - no match data found")

#%%
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
            # ... rest of the code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis match win probability predictor")
    parser.add_argument("--player1", required=True, help="Name of player 1")
    parser.add_argument("--player2", required=True, help="Name of player 2")
    parser.add_argument("--date", required=True, help="Match date in YYYY-MM-DD")
    parser.add_argument("--tournament", required=True, help="Tournament name slug")
    parser.add_argument("--gender", choices=["M","W"], required=True, help="Gender: M or W")
    parser.add_argument("--best_of", type=int, default=3, help="Sets in match, default 3")
    args = parser.parse_args()

    # Load or generate data (with smarter caching)
    hist, jeff_data, defaults = load_from_cache()
    if hist is None:
        print("No cache found. Generating full historical dataset (this will take a few minutes)...")
        hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=False)
        save_to_cache(hist, jeff_data, defaults)
        print("Historical data cached for future use.")
    else:
        print("Loaded historical data from cache.")

    # Only integrate recent API data (post 6/10/2025)
    print("Integrating recent API data...")

    # Get API data from June 10, 2025 through today
    days_since_june_10 = (date.today() - date(2025, 6, 10)).days
    hist = integrate_api_tennis_data_incremental(hist)
    # Re-save cache with new API data
    save_to_cache(hist, jeff_data, defaults)

    # Build composite_id for lookup
    match_date = pd.to_datetime(args.date).date()
    # Try both player orders since API might order by seeding
    comp_id1 = build_composite_id(match_date, normalize_tournament_name(args.tournament, args.gender),
                                  normalize_name(args.player1), normalize_name(args.player2))
    comp_id2 = build_composite_id(match_date, normalize_tournament_name(args.tournament, args.gender),
                                  normalize_name(args.player2), normalize_name(args.player1))

    print(f"Looking for match: {comp_id1}")
    print(f"Alternative order: {comp_id2}")

    row = hist[hist["composite_id"] == comp_id1]
    if row.empty:
        print(f"First order not found, trying alternative...")
        row = hist[hist["composite_id"] == comp_id2]
        if not row.empty:
            print(f"Found match with swapped player order")
            # Swap the players in args to match the found order
            args.player1, args.player2 = args.player2, args.player1

    if row.empty:
        print(f"No data for either order:")
        print(f"  {comp_id1}")
        print(f"  {comp_id2}")
        exit(1)

    # Extract stats
    p1_stats = {col[len("winner_"):]: row.iloc[0][col] for col in hist.columns if col.startswith("winner_")}
    p2_stats = {col[len("loser_"):]: row.iloc[0][col] for col in hist.columns if col.startswith("loser_")}

    model = BayesianTennisModel()
    prob = model.simulate_match(p1_stats, p2_stats, best_of=args.best_of)
    print(f"P(player1 wins) = {prob:.3f}")