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
from datetime import date, timedelta, datetime
import os
import requests
import pickle
import html
import re
import time
from pathlib import Path
from unidecode import unidecode
from bs4 import BeautifulSoup, FeatureNotFound
from urllib.parse import urljoin, urlparse
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
    """Enhanced feature extraction with actual Jeff data processing"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data:
        return get_fallback_defaults(gender_key)

    if weighted_defaults and gender_key in weighted_defaults:
        features = weighted_defaults[gender_key].copy()
    else:
        features = get_fallback_defaults(gender_key)

    # Overview stats - actual match-level aggregates
    if 'overview' in jeff_data[gender_key]:
        overview_df = jeff_data[gender_key]['overview']
        if 'Player_canonical' in overview_df.columns:
            player_overview = overview_df[
                (overview_df['Player_canonical'] == player_canonical) &
                (overview_df['set'] == 'Total')
                ]

            if len(player_overview) > 0:
                # Use most recent match data
                latest = player_overview.iloc[-1]
                serve_pts = latest.get('serve_pts', 80)
                if serve_pts > 0:
                    features.update({
                        'serve_pts': float(serve_pts),
                        'aces': float(latest.get('aces', 0)),
                        'double_faults': float(latest.get('dfs', 0)),
                        'first_serve_pct': float(latest.get('first_in', 0)) / float(serve_pts),
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

    # Serve basics - detailed serve performance
    if 'serve_basics' in jeff_data[gender_key]:
        serve_df = jeff_data[gender_key]['serve_basics']
        if 'Player_canonical' in serve_df.columns:
            player_serve = serve_df[serve_df['Player_canonical'] == player_canonical]

            if len(player_serve) > 0:
                serve_data = player_serve.iloc[-1]
                total_pts = serve_data.get('pts', 1)
                if total_pts > 0:
                    features.update({
                        'serve_ace_rate': float(serve_data.get('aces', 0)) / total_pts,
                        'serve_unreturnable_rate': float(serve_data.get('unret', 0)) / total_pts,
                        'serve_forced_error_rate': float(serve_data.get('forced_err', 0)) / total_pts,
                        'serve_quick_points_rate': float(serve_data.get('pts_won_lte_3_shots', 0)) / total_pts,
                        'serve_wide_pct': float(serve_data.get('wide', 0)) / total_pts,
                        'serve_body_pct': float(serve_data.get('body', 0)) / total_pts,
                        'serve_t_pct': float(serve_data.get('t', 0)) / total_pts
                    })

    # Return outcomes - return game performance
    if 'return_outcomes' in jeff_data[gender_key]:
        return_df = jeff_data[gender_key]['return_outcomes']
        if 'Player_canonical' in return_df.columns:
            player_return = return_df[return_df['Player_canonical'] == player_canonical]

            if len(player_return) > 0:
                return_data = player_return.iloc[-1]
                total_pts = return_data.get('pts', 1)
                returnable = return_data.get('returnable', 1)
                in_play = return_data.get('in_play', 1)

                if total_pts > 0:
                    features.update({
                        'return_points_won_pct': float(return_data.get('pts_won', 0)) / total_pts,
                        'return_in_play_pct': float(in_play) / returnable if returnable > 0 else 0,
                        'return_in_play_won_pct': float(
                            return_data.get('in_play_won', 0)) / in_play if in_play > 0 else 0,
                        'return_winners_rate': float(return_data.get('winners', 0)) / total_pts
                    })

    # Return depth - tactical return positioning
    if 'return_depth' in jeff_data[gender_key]:
        depth_df = jeff_data[gender_key]['return_depth']
        if 'Player_canonical' in depth_df.columns:
            player_depth = depth_df[depth_df['Player_canonical'] == player_canonical]

            if len(player_depth) > 0:
                depth_data = player_depth.iloc[-1]
                returnable = depth_data.get('returnable', 1)

                if returnable > 0:
                    features.update({
                        'return_shallow_pct': float(depth_data.get('shallow', 0)) / returnable,
                        'return_deep_pct': float(depth_data.get('deep', 0)) / returnable,
                        'return_very_deep_pct': float(depth_data.get('very_deep', 0)) / returnable,
                        'return_error_net_pct': float(depth_data.get('err_net', 0)) / returnable,
                        'return_error_wide_pct': float(depth_data.get('err_wide', 0)) / returnable,
                        'return_error_deep_pct': float(depth_data.get('err_deep', 0)) / returnable
                    })

    # Key points serve - clutch serving
    if 'key_points_serve' in jeff_data[gender_key]:
        kp_serve_df = jeff_data[gender_key]['key_points_serve']
        if 'Player_canonical' in kp_serve_df.columns:
            player_kp_serve = kp_serve_df[kp_serve_df['Player_canonical'] == player_canonical]

            if len(player_kp_serve) > 0:
                kp_data = player_kp_serve.iloc[-1]
                total_pts = kp_data.get('pts', 1)

                if total_pts > 0:
                    features.update({
                        'key_points_serve_won_pct': float(kp_data.get('pts_won', 0)) / total_pts,
                        'key_points_aces_pct': float(kp_data.get('aces', 0)) / total_pts,
                        'key_points_first_in_pct': float(kp_data.get('first_in', 0)) / total_pts,
                        'key_points_svc_winners_pct': float(kp_data.get('svc_winners', 0)) / total_pts,
                        'key_points_rally_winners_pct': float(kp_data.get('rally_winners', 0)) / total_pts,
                        'key_points_forced_errors_pct': float(kp_data.get('rally_forced', 0)) / total_pts,
                        'key_points_unforced_pct': float(kp_data.get('unforced', 0)) / total_pts
                    })

    # Key points return - clutch returning
    if 'key_points_return' in jeff_data[gender_key]:
        kp_return_df = jeff_data[gender_key]['key_points_return']
        if 'Player_canonical' in kp_return_df.columns:
            player_kp_return = kp_return_df[kp_return_df['Player_canonical'] == player_canonical]

            if len(player_kp_return) > 0:
                kp_return_data = player_kp_return.iloc[-1]
                total_pts = kp_return_data.get('pts', 1)

                if total_pts > 0:
                    features.update({
                        'key_points_return_won_pct': float(kp_return_data.get('pts_won', 0)) / total_pts,
                        'key_points_return_winners_pct': float(kp_return_data.get('rally_winners', 0)) / total_pts,
                        'key_points_return_forced_pct': float(kp_return_data.get('rally_forced', 0)) / total_pts,
                        'key_points_return_unforced_pct': float(kp_return_data.get('unforced', 0)) / total_pts
                    })

    # Net points - net game effectiveness
    if 'net_points' in jeff_data[gender_key]:
        net_df = jeff_data[gender_key]['net_points']
        if 'Player_canonical' in net_df.columns:
            player_net = net_df[net_df['Player_canonical'] == player_canonical]

            if len(player_net) > 0:
                net_data = player_net.iloc[-1]
                net_pts = net_data.get('net_pts', 1)

                if net_pts > 0:
                    features.update({
                        'net_points_won_pct': float(net_data.get('pts_won', 0)) / net_pts,
                        'net_winners_pct': float(net_data.get('net_winner', 0)) / net_pts,
                        'net_induced_forced_pct': float(net_data.get('induced_forced', 0)) / net_pts,
                        'net_unforced_pct': float(net_data.get('net_unforced', 0)) / net_pts,
                        'passed_at_net_pct': float(net_data.get('passed_at_net', 0)) / net_pts,
                        'passing_shot_forced_pct': float(net_data.get('passing_shot_induced_forced', 0)) / net_pts
                    })

    # Rally stats - rally performance by position
    if 'rally' in jeff_data[gender_key]:
        rally_df = jeff_data[gender_key]['rally']
        if 'server' in rally_df.columns and 'returner' in rally_df.columns:
            # Get rally stats when serving
            server_rallies = rally_df[rally_df['server'].apply(lambda x: normalize_jeff_name(x) == player_canonical)]
            # Get rally stats when returning
            returner_rallies = rally_df[
                rally_df['returner'].apply(lambda x: normalize_jeff_name(x) == player_canonical)]

            if len(server_rallies) > 0:
                server_data = server_rallies.iloc[-1]
                total_pts = server_data.get('pts', 1)
                if total_pts > 0:
                    features.update({
                        'rally_server_win_pct': float(server_data.get('pl1_won', 0)) / total_pts,
                        'rally_server_winners_pct': float(server_data.get('pl1_winners', 0)) / total_pts,
                        'rally_server_forced_pct': float(server_data.get('pl1_forced', 0)) / total_pts,
                        'rally_server_unforced_pct': float(server_data.get('pl1_unforced', 0)) / total_pts
                    })

            if len(returner_rallies) > 0:
                returner_data = returner_rallies.iloc[-1]
                total_pts = returner_data.get('pts', 1)
                if total_pts > 0:
                    features.update({
                        'rally_returner_win_pct': float(returner_data.get('pl2_won', 0)) / total_pts,
                        'rally_returner_winners_pct': float(returner_data.get('pl2_winners', 0)) / total_pts,
                        'rally_returner_forced_pct': float(returner_data.get('pl2_forced', 0)) / total_pts,
                        'rally_returner_unforced_pct': float(returner_data.get('pl2_unforced', 0)) / total_pts
                    })

    # Shot direction preferences
    if 'shot_direction' in jeff_data[gender_key]:
        shot_dir_df = jeff_data[gender_key]['shot_direction']
        if 'Player_canonical' in shot_dir_df.columns:
            player_shots = shot_dir_df[shot_dir_df['Player_canonical'] == player_canonical]

            if len(player_shots) > 0:
                shot_data = player_shots.iloc[-1]
                total_shots = (shot_data.get('crosscourt', 0) +
                               shot_data.get('down_middle', 0) +
                               shot_data.get('down_the_line', 0) +
                               shot_data.get('inside_out', 0) +
                               shot_data.get('inside_in', 0))

                if total_shots > 0:
                    features.update({
                        'shot_crosscourt_pct': float(shot_data.get('crosscourt', 0)) / total_shots,
                        'shot_down_middle_pct': float(shot_data.get('down_middle', 0)) / total_shots,
                        'shot_down_line_pct': float(shot_data.get('down_the_line', 0)) / total_shots,
                        'shot_inside_out_pct': float(shot_data.get('inside_out', 0)) / total_shots,
                        'shot_inside_in_pct': float(shot_data.get('inside_in', 0)) / total_shots
                    })

    # Serve and volley frequency and success
    if 'snv' in jeff_data[gender_key]:
        snv_df = jeff_data[gender_key]['snv']
        if 'Player_canonical' in snv_df.columns:
            player_snv = snv_df[snv_df['Player_canonical'] == player_canonical]

            if len(player_snv) > 0:
                snv_data = player_snv.iloc[-1]
                snv_pts = snv_data.get('snv_pts', 0)

                if snv_pts > 0:
                    features.update({
                        'serve_volley_frequency': float(snv_pts) / features.get('serve_pts', 80),
                        'serve_volley_success_pct': float(snv_data.get('pts_won', 0)) / snv_pts,
                        'snv_aces_pct': float(snv_data.get('aces', 0)) / snv_pts,
                        'snv_net_winners_pct': float(snv_data.get('net_winner', 0)) / snv_pts,
                        'snv_passed_pct': float(snv_data.get('passed_at_net', 0)) / snv_pts
                    })

    # Calculate composite performance indices
    features['aggression_index'] = (
            features.get('winners_total', 20) / (
                features.get('winners_total', 20) + features.get('unforced_errors', 20))
    )

    features['consistency_index'] = 1 - (
            features.get('unforced_errors', 20) / (features.get('serve_pts', 80) + features.get('return_pts_won', 30))
    )

    features['pressure_performance'] = (
                                               features.get('key_points_serve_won_pct', 0.5) + features.get(
                                           'key_points_return_won_pct', 0.5)
                                       ) / 2

    features['net_game_strength'] = features.get('net_points_won_pct', 0.5)

    return features


def process_tennis_abstract_scraped_data(scraped_records):
    """Convert scraped records into per-match player features"""
    matches = {}

    for record in scraped_records:
        comp_id = record.get('composite_id')
        if comp_id not in matches:
            matches[comp_id] = {}

        player_canonical = record.get('Player_canonical')
        if not player_canonical:
            continue

        if player_canonical not in matches[comp_id]:
            matches[comp_id][player_canonical] = {}

        # Extract key performance stats from Tennis Abstract
        data_type = record.get('data_type')
        stat_name = record.get('stat_name')
        stat_value = record.get('stat_value')

        if data_type and stat_name and stat_value is not None:
            # Map to standardized feature names
            feature_key = f"ta_{data_type}_{stat_name}"
            matches[comp_id][player_canonical][feature_key] = stat_value

    return matches


def integrate_scraped_data_hybrid(historical_data, scraped_records):
    """Hybrid: enhance existing matches, add new TA matches if not in API"""
    if not scraped_records:
        return historical_data

    print(f"Hybrid integration of {len(scraped_records)} Tennis Abstract records")

    # Process scraped records into features by match
    processed_records = process_tennis_abstract_scraped_data(scraped_records)

    enhanced_data = historical_data.copy()
    new_matches_added = 0
    existing_matches_enhanced = 0

    for comp_id, match_players in processed_records.items():
        # Find matching row in historical data
        match_rows = enhanced_data[enhanced_data['composite_id'] == comp_id]

        if not match_rows.empty:
            # ENHANCE existing match (preserve API context)
            row_idx = match_rows.index[0]
            current_row = enhanced_data.loc[row_idx]

            winner_canonical = current_row['winner_canonical']
            loser_canonical = current_row['loser_canonical']

            # Overlay Tennis Abstract features while preserving API context
            for player_canonical, ta_features in match_players.items():
                if player_canonical == winner_canonical:
                    prefix = 'winner_'
                elif player_canonical == loser_canonical:
                    prefix = 'loser_'
                else:
                    continue

                for feature_name, feature_value in ta_features.items():
                    col_name = f"{prefix}{feature_name}"
                    if feature_value is not None and not pd.isna(feature_value):
                        enhanced_data.loc[row_idx, col_name] = feature_value

            enhanced_data.loc[row_idx, 'ta_enhanced'] = True
            if enhanced_data.loc[row_idx, 'source_rank'] == 2:  # API data
                enhanced_data.loc[row_idx, 'data_quality_score'] = 0.85  # High quality hybrid

            existing_matches_enhanced += 1

        else:
            # ADD new Tennis Abstract match
            players = list(match_players.keys())
            if len(players) >= 2:
                # Extract metadata from scraped records for this match
                match_records = [r for r in scraped_records if r.get('composite_id') == comp_id]
                if match_records:
                    sample_record = match_records[0]

                    # Create new match record with TA as primary source
                    new_record = {
                        'composite_id': comp_id,
                        'source_rank': 1,  # Tennis Abstract = highest quality
                        'data_quality_score': 0.9,
                        'ta_enhanced': True,
                        'Date': pd.to_datetime(sample_record.get('Date', ''), format='%Y%m%d', errors='coerce'),
                        'date': pd.to_datetime(sample_record.get('Date', ''), format='%Y%m%d', errors='coerce').date(),
                        'gender': sample_record.get('gender', 'M'),
                        'tournament': sample_record.get('tournament', 'Unknown'),
                        'tournament_canonical': sample_record.get('tournament', 'unknown').lower(),
                        'surface': 'Grass',  # Default for Wimbledon
                        'winner_canonical': players[0],
                        'loser_canonical': players[1],
                        'Winner': players[0].replace('_', ' ').title(),
                        'Loser': players[1].replace('_', ' ').title(),
                    }

                    # Add all Tennis Abstract features
                    for i, (player_canonical, ta_features) in enumerate(match_players.items()):
                        prefix = 'winner_' if i == 0 else 'loser_'
                        for feature_name, feature_value in ta_features.items():
                            col_name = f"{prefix}{feature_name}"
                            if feature_value is not None:
                                new_record[col_name] = feature_value

                    # Add to dataset
                    enhanced_data = pd.concat([enhanced_data, pd.DataFrame([new_record])], ignore_index=True)
                    new_matches_added += 1

    print(f"Enhanced {existing_matches_enhanced} existing matches")
    print(f"Added {new_matches_added} new Tennis Abstract matches")

    return enhanced_data


def process_tennis_abstract_scraped_data(scraped_records):
    """Convert scraped records into per-match player features"""
    matches = {}

    for record in scraped_records:
        comp_id = record.get('composite_id')
        if comp_id not in matches:
            matches[comp_id] = {}

        player_canonical = record.get('Player_canonical')
        if not player_canonical:
            continue

        if player_canonical not in matches[comp_id]:
            matches[comp_id][player_canonical] = {}

        # Extract key performance stats from Tennis Abstract
        data_type = record.get('data_type')
        stat_name = record.get('stat_name')
        stat_value = record.get('stat_value')

        if data_type and stat_name and stat_value is not None:
            # Map to standardized feature names
            feature_key = f"ta_{data_type}_{stat_name}"
            matches[comp_id][player_canonical][feature_key] = stat_value

    return matches


def calculate_feature_importance_weights(historical_data, jeff_data):
    """Calculate dynamic feature importance weights based on data availability"""
    weights = {
        'jeff_comprehensive': 0.4,
        'tennis_abstract': 0.35,
        'api_tennis': 0.2,
        'tennis_data_files': 0.05
    }

    # Adjust weights based on actual data coverage
    total_rows = len(historical_data)

    if total_rows > 0:
        # Calculate coverage percentages
        jeff_coverage = len(historical_data[historical_data['source_rank'] == 3]) / total_rows
        ta_coverage = len(historical_data[historical_data['source_rank'] == 1]) / total_rows
        api_coverage = len(historical_data[historical_data['source_rank'] == 2]) / total_rows

        # Rebalance weights based on coverage
        if ta_coverage > 0.1:  # If we have good TA coverage
            weights['tennis_abstract'] = 0.4
            weights['jeff_comprehensive'] = 0.35

        if api_coverage > 0.3:  # If we have good API coverage
            weights['api_tennis'] = 0.25
            weights['jeff_comprehensive'] = 0.35

    print(f"Feature importance weights: {weights}")
    return weights

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
# ============================================================================
# COMPLETE TENNIS ABSTRACT SCRAPER REPLACEMENT
# ============================================================================

import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from pathlib import Path
import os
from urllib.parse import urljoin, urlparse


class TennisAbstractScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def _normalize_player_name(self, name: str) -> str:
        import re, unicodedata
        name = unicodedata.normalize("NFKD", name).strip()
        name = re.sub(r"[\s\-]+", "_", name)
        return name.lower()

    def _parse_match_url(self, url: str) -> dict:
        """Parse Tennis Abstract charting URL for metadata"""
        fname = os.path.basename(urlparse(url).path)
        if not fname.endswith(".html"):
            return {}

        parts = fname[:-5].split("-")
        if len(parts) < 6 or not parts[0].isdigit():
            return {}

        return {
            "Date": parts[0],
            "gender": parts[1],
            "tournament": "-".join(parts[2:-3]).replace("_", " "),
            "round": parts[-3],
            "player1": parts[-2].replace("_", " "),
            "player2": parts[-1].replace("_", " "),
        }

    def _extract_all_js_tables(self, html_content: str) -> dict:
        """Extract all JavaScript table variables from Tennis Abstract page - FIXED VERSION"""

        # Known Tennis Abstract table variables
        table_vars = [
            'serve', 'serve1', 'serve2', 'return1', 'return2',
            'keypoints', 'rallyoutcomes', 'overview', 'shots1', 'shots2',
            'shotdir1', 'shotdir2', 'netpts1', 'netpts2', 'serveNeut', 'pointlog'
        ]

        extracted_tables = {}

        for var_name in table_vars:
            # FIXED: Use single quotes pattern that actually works
            pattern = fr'var\s+{re.escape(var_name)}\s*=\s*\'((?:[^\'\\]|\\.)*)\'\s*;?'

            matches = re.findall(pattern, html_content, re.DOTALL | re.MULTILINE)

            if matches:
                content = matches[0]

                # Clean up JavaScript escaping
                clean_content = (content
                                 .replace('\\n', '\n')
                                 .replace('\\t', '\t')
                                 .replace('\\"', '"')
                                 .replace("\\'", "'")
                                 .replace('\\\\', '\\')
                                 )

                if '<table' in clean_content:
                    extracted_tables[var_name] = clean_content

        return extracted_tables

    def _parse_tennis_abstract_table(self, table_html: str, table_type: str, match_meta: dict) -> list:
        """Parse a Tennis Abstract table into structured records"""

        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')

            if not table:
                return []

            rows = table.find_all('tr')
            if len(rows) < 2:
                return []

            records = []

            # Extract headers
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            # Map player codes to canonical names
            player_map = {
                'JS': self._normalize_player_name(match_meta.get('player1', '')),
                'PM': self._normalize_player_name(match_meta.get('player2', ''))
            }

            # Process data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                # First cell is usually player/stat identifier
                identifier = cells[0].get_text(strip=True)

                if not identifier:
                    continue

                # Parse identifier to get player and context
                player_code, stat_context = self._parse_row_identifier(identifier)

                if not player_code:
                    continue

                # Map to canonical player name
                player_canonical = player_map.get(player_code, player_code)

                # Extract stats from remaining cells
                for i, cell in enumerate(cells[1:], 1):
                    if i < len(headers):
                        header = headers[i]
                        value_text = cell.get_text(strip=True)

                        if not value_text or value_text == '-':
                            continue

                        # Parse Tennis Abstract format
                        parsed_value = self._parse_cell_value(value_text)

                        if parsed_value:
                            record = {
                                **match_meta,
                                'Player_canonical': player_canonical,
                                'stat_context': stat_context,
                                'stat_name': self._normalize_header(header),
                                'stat_value': parsed_value.get('value', parsed_value.get('count', 0)),
                                'stat_percentage': parsed_value.get('percentage'),
                                'raw_value': value_text,
                                'data_type': table_type,
                                'composite_id': self._build_composite_id(match_meta)
                            }
                            records.append(record)

            return records

        except Exception as e:
            print(f"Error parsing {table_type} table: {e}")
            return []

    def _parse_row_identifier(self, identifier: str) -> tuple:
        """Parse row identifier to extract player code and context"""
        if not identifier:
            return None, None

        # Handle different identifier formats
        if identifier.startswith('JS '):
            return 'JS', identifier[3:].strip() or 'Total'
        elif identifier.startswith('PM '):
            return 'PM', identifier[3:].strip() or 'Total'
        elif identifier in ['Jannik Sinner', 'JS']:
            return 'JS', 'Total'
        elif identifier in ['Pedro Martinez', 'PM']:
            return 'PM', 'Total'
        elif 'Sinner' in identifier:
            return 'JS', identifier.replace('Jannik Sinner', '').strip() or 'Total'
        elif 'Martinez' in identifier:
            return 'PM', identifier.replace('Pedro Martinez', '').strip() or 'Total'
        else:
            # Try to extract player code from beginning
            parts = identifier.split()
            if parts and len(parts[0]) <= 3:
                return parts[0], ' '.join(parts[1:]) or 'Total'

        return None, None

    def _parse_cell_value(self, cell_text: str) -> dict:
        """Parse Tennis Abstract cell values (e.g., '53  (72%)')"""
        if not cell_text or cell_text == '-':
            return None

        result = {}

        # Extract count before parentheses
        count_match = re.search(r'^(\d+)', cell_text.strip())
        if count_match:
            result['count'] = int(count_match.group(1))

        # Extract percentage in parentheses
        pct_match = re.search(r'\((\d+(?:\.\d+)?)%\)', cell_text)
        if pct_match:
            result['percentage'] = float(pct_match.group(1)) / 100.0

        # If no specific format, try to parse as number
        if not result:
            try:
                # Handle fractions
                if '/' in cell_text:
                    parts = cell_text.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        result['value'] = numerator / denominator if denominator > 0 else 0
                else:
                    # Simple number
                    result['value'] = float(cell_text.strip())
            except:
                result['raw'] = cell_text.strip()

        return result if result else None

    def _normalize_header(self, header: str) -> str:
        """Normalize table headers to consistent stat names"""
        if not header:
            return ""

        # Clean up header
        header = header.lower().strip()
        header = re.sub(r'---+%?$', '', header)  # Remove trailing dashes and %
        header = re.sub(r'[^\w\s]', '', header)  # Remove special chars
        header = header.replace(' ', '_')

        # Common mappings
        mappings = {
            'pts': 'points',
            'won': 'won_pct',
            'aces': 'aces_pct',
            'unret': 'unreturned_pct',
            'fcde': 'forced_errors_pct',
            '3w': 'quick_points_pct',
            'wide': 'wide_pct',
            'body': 'body_pct',
            't': 't_pct',
            'ptsw': 'points_won_pct',
            'returnable': 'returnable_serves',
            'rtblew': 'returnable_won_pct',
            'inplay': 'in_play_pct',
            'inplayw': 'in_play_won_pct',
            'wnr': 'winners_pct',
            'avgrally': 'avg_rally_length'
        }

        return mappings.get(header, header)

    def _build_composite_id(self, match_meta: dict) -> str:
        """Build composite match ID"""
        try:
            from datetime import datetime
            date_str = match_meta.get('Date', '')
            if len(date_str) == 8:  # YYYYMMDD
                match_date = datetime.strptime(date_str, '%Y%m%d').date()
            else:
                match_date = datetime.now().date()

            tournament = match_meta.get('tournament', '').lower().replace(' ', '_')
            player1 = self._normalize_player_name(match_meta.get('player1', ''))
            player2 = self._normalize_player_name(match_meta.get('player2', ''))

            return f"{match_date.strftime('%Y%m%d')}-{tournament}-{player1}-{player2}"
        except:
            return f"unknown-{match_meta.get('Date', '')}"

    def scrape_comprehensive_match_data(self, url: str) -> list:
        """Main method to scrape all Tennis Abstract data from a match URL"""

        try:
            print(f"Scraping comprehensive data from: {url}")

            # Get page content
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()

            # Parse match metadata
            match_meta = self._parse_match_url(url)
            if not match_meta:
                print(f"Could not parse match metadata from URL: {url}")
                return []

            # Extract all JavaScript tables
            js_tables = self._extract_all_js_tables(resp.text)

            print(f"Extracted {len(js_tables)} JavaScript tables: {list(js_tables.keys())}")

            all_records = []

            # Process each table
            for table_name, table_html in js_tables.items():
                records = self._parse_tennis_abstract_table(table_html, table_name, match_meta)
                print(f"  {table_name}: {len(records)} records")
                all_records.extend(records)

            print(f"Total records extracted: {len(all_records)}")
            return all_records

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    # Legacy method names for compatibility - redirect to comprehensive method
    def scrape_stats_overview(self, url: str) -> list:
        """Legacy method - now extracts overview from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') == 'overview']

    def scrape_serve_statistics_overview(self, url: str) -> list:
        """Legacy method - now extracts serve stats from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') in ['serve', 'serve1', 'serve2']]

    def scrape_return_breakdown(self, url: str) -> list:
        """Legacy method - now extracts return stats from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') in ['return1', 'return2']]

    def scrape_key_point_outcomes(self, url: str) -> list:
        """Legacy method - now extracts key points from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') == 'keypoints']

    def scrape_rally_outcomes(self, url: str) -> list:
        """Legacy method - now extracts rally data from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') == 'rallyoutcomes']

    def scrape_net_points(self, url: str) -> list:
        """Legacy method - now extracts net points from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') in ['netpts1', 'netpts2']]

    def scrape_shot_types(self, url: str) -> list:
        """Legacy method - now extracts shot types from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') in ['shots1', 'shots2']]

    def scrape_pointlog(self, url: str) -> list:
        """Legacy method - now extracts point log from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') == 'pointlog']

    def scrape_serve_influence(self, url: str) -> list:
        """Legacy method - now extracts serve influence from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') == 'serveNeut']

    def scrape_serve_breakdown(self, url: str) -> list:
        """Legacy method - now extracts detailed serve breakdown from comprehensive data"""
        all_records = self.scrape_comprehensive_match_data(url)
        return [r for r in all_records if r.get('data_type') in ['serve1', 'serve2']]


class AutomatedTennisAbstractScraper(TennisAbstractScraper):
    def __init__(self, cache_dir=None):
        super().__init__()
        self.cache_dir = Path(cache_dir or "cache") / "tennis_abstract_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scraped_urls_file = self.cache_dir / "scraped_urls.txt"
        self.scraped_data_file = self.cache_dir / "scraped_data.parquet"

        # Load previously scraped URLs to avoid duplicates
        self.scraped_urls = set()
        if self.scraped_urls_file.exists():
            with open(self.scraped_urls_file, 'r') as f:
                self.scraped_urls = set(line.strip() for line in f)

    def discover_match_urls(self, start_date=None, end_date=None, max_pages=50):
        """Discover Tennis Abstract charting URLs from the actual charting page"""
        from datetime import date, datetime, timedelta

        if start_date is None:
            start_date = date(2025, 6, 10)  # Default to our cutoff date
        if end_date is None:
            end_date = date.today()

        print(f"Discovering Tennis Abstract URLs from {start_date} to {end_date}")

        # Actual Tennis Abstract charting page
        charting_url = "https://www.tennisabstract.com/charting/"

        try:
            resp = requests.get(charting_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")

            match_urls = []

            # Look for all links that match Tennis Abstract charting pattern
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Match the actual URL pattern: YYYYMMDD-G-Tournament-Round-Player1-Player2.html
                if re.match(r'\d{8}-[MW]-.*\.html$', href):
                    # Convert relative URL to absolute
                    if href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(charting_url, href)

                    # Extract date from filename
                    filename = href.split('/')[-1]
                    date_str = filename[:8]

                    try:
                        url_date = datetime.strptime(date_str, '%Y%m%d').date()
                        if start_date <= url_date <= end_date:
                            match_urls.append(full_url)
                    except ValueError:
                        continue

            # Remove duplicates and already scraped URLs
            match_urls = list(set(match_urls))
            new_urls = [url for url in match_urls if url not in self.scraped_urls]

            # Sort by date (newest first)
            def extract_date_from_url(url):
                filename = url.split('/')[-1]
                if re.match(r'\d{8}', filename):
                    return filename[:8]
                return '00000000'

            new_urls.sort(key=extract_date_from_url, reverse=True)

            print(f"Found {len(match_urls)} total matches, {len(new_urls)} new to scrape")

            return new_urls[:max_pages]

        except Exception as e:
            print(f"Error discovering URLs from {charting_url}: {e}")
            return []

    def scrape_match_comprehensive(self, url):
        """Scrape all available data from a single Tennis Abstract match page using new method"""
        print(f"Scraping: {url}")

        try:
            # Use the new comprehensive scraping method
            scraped_records = self.scrape_comprehensive_match_data(url)

            if scraped_records:
                # Mark as successfully scraped
                self.scraped_urls.add(url)
                self._save_scraped_url(url)

                return {
                    'url': url,
                    'scrape_timestamp': datetime.now(),
                    'comprehensive_data': scraped_records,
                    'record_count': len(scraped_records)
                }
            else:
                print(f"No data extracted from {url}")
                return None

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def _save_scraped_url(self, url):
        """Save URL to scraped list to avoid re-scraping"""
        with open(self.scraped_urls_file, 'a') as f:
            f.write(f"{url}\n")

    def process_scraped_data_to_features(self, scraped_data):
        """Convert raw scraped data into standardized features"""
        if not scraped_data or 'comprehensive_data' not in scraped_data:
            return []

        records = scraped_data['comprehensive_data']

        # Add scraping metadata to each record
        for record in records:
            record.update({
                'source': 'tennis_abstract',
                'scrape_url': scraped_data['url'],
                'scrape_timestamp': scraped_data['scrape_timestamp']
            })

        return records

    def automated_scraping_session(self, days_back=None, max_matches=100):
        """Run automated scraping session for matches after 6/10/2025"""
        from datetime import date, timedelta

        # Default to scraping from our data cutoff date
        start_date = date(2025, 6, 10)  # Tennis Abstract data starts here
        end_date = date.today()

        if days_back is not None:
            start_date = date.today() - timedelta(days=days_back)

        print(f"Running Tennis Abstract scraping session from {start_date} to {end_date}")

        # Discover new URLs to scrape
        new_urls = self.discover_match_urls(start_date, end_date, max_matches)

        if not new_urls:
            print("No new Tennis Abstract matches to scrape")
            return []

        all_records = []
        successful_scrapes = 0

        for i, url in enumerate(new_urls):
            print(f"Processing {i + 1}/{len(new_urls)}: {url}")

            try:
                # Scrape the match
                scraped_data = self.scrape_match_comprehensive(url)

                if scraped_data:
                    # Convert to standardized features
                    records = self.process_scraped_data_to_features(scraped_data)
                    all_records.extend(records)
                    successful_scrapes += 1
                    print(f"  ✓ Successfully scraped {len(records)} feature records")
                else:
                    print(f"  ✗ Failed to scrape data")

            except Exception as e:
                print(f"  ✗ Error processing {url}: {e}")
                continue

            # Rate limiting - be respectful to Tennis Abstract
            time.sleep(2)

        print(f"Scraping complete: {successful_scrapes}/{len(new_urls)} matches successful")
        print(f"Generated {len(all_records)} total feature records")
        return all_records

def integrate_scraped_data_into_pipeline(historical_data, scraped_records):
    """Integrate Tennis Abstract scraped data into the main historical dataset"""
    if not scraped_records:
        return historical_data

    print(f"Integrating {len(scraped_records)} Tennis Abstract records")

    # Convert scraped records to DataFrame
    scraped_df = pd.DataFrame(scraped_records)

    # Group by composite_id and player to aggregate features
    player_features = {}

    for comp_id, group in scraped_df.groupby('composite_id'):
        # Initialize feature dict for this match
        if comp_id not in player_features:
            player_features[comp_id] = {}

        # Process each record type
        for _, record in group.iterrows():
            player_canonical = record.get('Player_canonical', '')
            if not player_canonical:
                continue

            # Initialize player features
            if player_canonical not in player_features[comp_id]:
                player_features[comp_id][player_canonical] = {
                    'ta_source_rank': 1,  # Tennis Abstract is highest quality
                    'ta_data_types': set()
                }

            player_dict = player_features[comp_id][player_canonical]
            player_dict['ta_data_types'].add(record.get('data_type'))

    # Merge with historical data
    enhanced_data = historical_data.copy()

    for comp_id, match_players in player_features.items():
        # Find matching row in historical data
        match_rows = enhanced_data[enhanced_data['composite_id'] == comp_id]

        if match_rows.empty:
            continue

        row_idx = match_rows.index[0]

        # Determine which player is winner/loser in historical data
        winner_canonical = enhanced_data.loc[row_idx, 'winner_canonical']
        loser_canonical = enhanced_data.loc[row_idx, 'loser_canonical']

        # Map Tennis Abstract players to winner/loser
        for player_canonical, features in match_players.items():
            if player_canonical == winner_canonical:
                prefix = 'winner_'
            elif player_canonical == loser_canonical:
                prefix = 'loser_'
            else:
                continue  # Player not found in historical data

            # Add features with appropriate prefix
            for feature_name, feature_value in features.items():
                if feature_name != 'ta_data_types':
                    col_name = f"{prefix}{feature_name}"
                    enhanced_data.loc[row_idx, col_name] = feature_value

    print(f"Enhanced {len(player_features)} matches with Tennis Abstract data")
    return enhanced_data


def run_automated_tennis_abstract_integration(historical_data, days_back=None):
    """Main function to run automated Tennis Abstract scraping and integration"""
    print("=== AUTOMATED TENNIS ABSTRACT INTEGRATION ===")

    # Initialize scraper
    scraper = AutomatedTennisAbstractScraper()

    # Run scraping session (defaults to 6/10/2025 if days_back not specified)
    scraped_records = scraper.automated_scraping_session(days_back=days_back)

    if not scraped_records:
        print("No new Tennis Abstract data scraped")
        return historical_data

    # Integrate into historical data with hybrid approach
    enhanced_data = integrate_scraped_data_hybrid(historical_data, scraped_records)  # ← NEW LINE

    print(f"Tennis Abstract integration complete. Enhanced dataset with detailed charting features.")
    return enhanced_data

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


# Inserted updated parse_match_statistics function after extract_embedded_statistics
def parse_api_tennis_statistics(fixture: dict) -> dict[int, dict]:
    try:
        statistics = fixture.get('statistics', [])
        if not statistics:
            return {}

        # Extract player keys from statistics instead of fixture metadata
        player_keys = set()
        for stat in statistics:
            if stat.get('stat_period') == 'match':
                player_key = safe_int_convert(stat.get('player_key'))
                if player_key:
                    player_keys.add(player_key)

        if len(player_keys) != 2:
            return {}

        player_stats = {key: {} for key in player_keys}

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
                player_stats[player_key][f"{key_name}_won"] = stat_won
                player_stats[player_key][f"{key_name}_total"] = stat_total
                if stat_total > 0:
                    player_stats[player_key][f"{key_name}_pct"] = stat_won / stat_total
            else:
                # Parse the stat_value
                if '%' in str(stat_value):
                    pct_val = float(str(stat_value).replace('%', '')) / 100
                    player_stats[player_key][f"{key_name}_pct"] = pct_val
                else:
                    try:
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
    tennis_data = integrate_api_tennis_data_incremental(tennis_data)

    logging.info(f"=== DATA GENERATION COMPLETE ===")
    logging.info(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults

# ============================================================================
# API-TENNIS DATA INTEGRATION
# ============================================================================

def integrate_api_tennis_data_incremental(historical_data):
    """FIXED: Fetch API data with proper statistics integration"""
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
            # Only process completed matches
            if fixture.get("event_status") != "Finished":
                continue

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

            # FIXED: Extract proper statistics from fixture
            stats_map = parse_match_statistics(fixture)
            embed = extract_embedded_statistics(fixture)
            h2h_data = get_h2h_data(p1_key, p2_key)
            odds1, odds2 = get_match_odds(fixture.get("match_key"), day)

            # Get player profiles for additional stats
            p1_profile = get_player_profile(p1_key)
            p2_profile = get_player_profile(p2_key)

            # FIXED: Build comprehensive record with proper player mapping
            record = {
                "composite_id": comp_id,
                "source_rank": 2,
                "date": day,
                "gender": gender,
                "surface": surface,
                "tournament_tier": event_info.get("event_type_type", "Unknown"),

                # Player info
                "Winner": fixture.get("event_first_player", ""),
                "Loser": fixture.get("event_second_player", ""),
                "winner_canonical": normalize_name(fixture.get("event_first_player", "")),
                "loser_canonical": normalize_name(fixture.get("event_second_player", "")),

                # Player rankings
                "WRank": rankings.get(p1_key),
                "LRank": rankings.get(p2_key),
                "p1_ranking": rankings.get(p1_key),
                "p2_ranking": rankings.get(p2_key),
                "ranking_difference": abs(rankings.get(p1_key, 999) - rankings.get(p2_key, 999)),

                # Head-to-head data
                "h2h_matches": h2h_data.get("h2h_matches", 0),
                "p1_h2h_wins": h2h_data.get("p1_wins", 0),
                "p2_h2h_wins": h2h_data.get("p2_wins", 0),
                "p1_h2h_win_pct": h2h_data.get("p1_win_pct", 0.5),

                # Betting odds and implied probabilities
                "PSW": odds1,
                "PSL": odds2,
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
            }

            # FIXED: Add statistics from API with proper winner/loser mapping
            if stats_map:
                p1_stats = stats_map.get(p1_key, {})
                p2_stats = stats_map.get(p2_key, {})

                # Map API statistics to winner/loser prefixes
                for stat_name, stat_value in p1_stats.items():
                    if pd.notna(stat_value):
                        record[f"winner_{stat_name}"] = stat_value

                for stat_name, stat_value in p2_stats.items():
                    if pd.notna(stat_value):
                        record[f"loser_{stat_name}"] = stat_value

            # Add embedded statistics
            record.update(embed)

            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df = df.sort_values("source_rank").drop_duplicates(subset="composite_id", keep="first").reset_index(drop=True)
    print(f"Added {len(dates_to_fetch)} days of API data to cache.")
    return df

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

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

def load_from_cache_with_scraping():
    """Load data from cache and optionally run incremental Tennis Abstract scraping"""
    hist, jeff_data, defaults = load_from_cache()

    if hist is not None:
        # Check if we need to update with recent Tennis Abstract data
        # Look for Tennis Abstract features to see if we've integrated before
        ta_columns = [col for col in hist.columns if col.startswith(('winner_ta_', 'loser_ta_'))]

        if not ta_columns:
            print("No Tennis Abstract data found in cache. Running initial integration...")
            hist = run_automated_tennis_abstract_integration(hist)
            save_to_cache(hist, jeff_data, defaults)
        else:
            # Check for recent updates
            latest_date = hist['date'].max() if 'date' in hist.columns else date(2025, 6, 10)
            days_since_update = (date.today() - latest_date).days

            if days_since_update > 2:  # Update every 2 days for Tennis Abstract
                print(f"Updating Tennis Abstract data (last update: {latest_date})")
                hist = run_automated_tennis_abstract_integration(hist, days_back=min(days_since_update + 1, 7))
                save_to_cache(hist, jeff_data, defaults)
            else:
                print(f"Tennis Abstract data is current (last update: {latest_date})")

    return hist, jeff_data, defaults

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

# ============================================================================
# UPDATED PREDICTION MODEL WITH UNIFIED FEATURE EXTRACTION
# ============================================================================

import pandas as pd
import numpy as np
import random
from datetime import date


class UnifiedBayesianTennisModel:
    """Enhanced model that handles features from any data source"""

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

        # Surface adjustment factors
        self.surface_adjustments = {
            "Clay": {"serve_advantage": 0.92, "rally_importance": 1.15, "consistency_weight": 1.2},
            "Grass": {"serve_advantage": 1.15, "rally_importance": 0.85, "consistency_weight": 0.9},
            "Hard": {"serve_advantage": 1.0, "rally_importance": 1.0, "consistency_weight": 1.0},
            "Indoor Hard": {"serve_advantage": 1.08, "rally_importance": 0.95, "consistency_weight": 0.95}
        }

    def _calculate_serve_probability(self, server_features: dict, returner_features: dict, surface: str):
        """Calculate serve game win probability using unified features"""
        base_prob = 0.65

        # Server effectiveness
        serve_eff = server_features.get('serve_effectiveness', 0.65)

        # Return resistance
        return_eff = returner_features.get('return_effectiveness', 0.35)

        # Pressure performance
        pressure = server_features.get('pressure_performance', 0.5)

        # Surface adjustment
        surface_adj = self.surface_adjustments.get(surface, self.surface_adjustments["Hard"])

        # Combined probability
        serve_prob = (
                base_prob * 0.3 +
                serve_eff * surface_adj["serve_advantage"] * 0.4 +
                (1 - return_eff) * 0.2 +
                pressure * 0.1
        )

        return min(max(serve_prob, 0.1), 0.9)

    def _calculate_rally_probability(self, p1_features: dict, p2_features: dict, surface: str):
        """Calculate rally win probability using unified features"""

        # Winners vs unforced errors ratio
        p1_aggression = p1_features.get('winners_rate', 0.2) / (p1_features.get('unforced_rate', 0.18) + 0.01)
        p2_aggression = p2_features.get('winners_rate', 0.2) / (p2_features.get('unforced_rate', 0.18) + 0.01)

        # Net game strength
        p1_net = p1_features.get('net_effectiveness', 0.65)
        p2_net = p2_features.get('net_effectiveness', 0.65)

        # Surface adjustments
        surface_adj = self.surface_adjustments.get(surface, self.surface_adjustments["Hard"])

        # Combined rally strength
        p1_rally_strength = (
                p1_aggression * surface_adj.get("consistency_weight", 1.0) * 0.4 +
                p1_net * 0.3 +
                p1_features.get('pressure_performance', 0.5) * 0.3
        )

        p2_rally_strength = (
                p2_aggression * surface_adj.get("consistency_weight", 1.0) * 0.4 +
                p2_net * 0.3 +
                p2_features.get('pressure_performance', 0.5) * 0.3
        )

        total_strength = p1_rally_strength + p2_rally_strength
        return p1_rally_strength / total_strength if total_strength > 0 else 0.5

    def _simulate_game(self, p1_features: dict, p2_features: dict, match_context: dict, server: int):
        """Simulate a single game"""
        surface = match_context.get('surface', 'Hard')

        if server == 1:
            serve_prob = self._calculate_serve_probability(p1_features, p2_features, surface)
        else:
            serve_prob = self._calculate_serve_probability(p2_features, p1_features, surface)
            serve_prob = 1 - serve_prob

        rally_prob = self._calculate_rally_probability(p1_features, p2_features, surface)

        # Simulate points
        p1_points = p2_points = 0

        while True:
            # Use serve probability for most points, blend with rally for long games
            if p1_points < 3 and p2_points < 3:
                point_prob = serve_prob
            else:
                point_prob = serve_prob * 0.7 + rally_prob * 0.3

            if random.random() < point_prob:
                p1_points += 1
            else:
                p2_points += 1

            # Check for game win
            if (p1_points >= 4 or p2_points >= 4) and abs(p1_points - p2_points) >= 2:
                return 1 if p1_points > p2_points else 2

    def _simulate_set(self, p1_features: dict, p2_features: dict, match_context: dict):
        """Simulate one set"""
        p1_games = p2_games = 0
        server = 1

        while True:
            game_winner = self._simulate_game(p1_features, p2_features, match_context, server)

            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1

            server = 2 if server == 1 else 1

            # Check for set win
            if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                return 1 if p1_games > p2_games else 2
            elif p1_games == 6 and p2_games == 6:
                # Simplified tiebreak
                tb_prob = self._calculate_rally_probability(p1_features, p2_features,
                                                            match_context.get('surface', 'Hard'))
                return 1 if random.random() < tb_prob else 2

    def simulate_match(self, p1_features: dict, p2_features: dict, match_context: dict, best_of: int = 3):
        """Simulate complete match"""
        results = []

        for _ in range(self.n_simulations):
            p1_sets = p2_sets = 0

            while p1_sets <= best_of // 2 and p2_sets <= best_of // 2:
                set_winner = self._simulate_set(p1_features, p2_features, match_context)

                if set_winner == 1:
                    p1_sets += 1
                else:
                    p2_sets += 1

                if p1_sets > best_of // 2 or p2_sets > best_of // 2:
                    break

            results.append(1 if p1_sets > p2_sets else 0)

        return sum(results) / len(results)


def extract_unified_features(match_data, player_prefix):
    """Extract features with fallbacks across all data sources"""
    features = {}

    # Serve effectiveness (Tennis Abstract > Jeff > API > defaults)
    if f'{player_prefix}_ta_serve_won_pct' in match_data:
        features['serve_effectiveness'] = match_data[f'{player_prefix}_ta_serve_won_pct']
    elif f'{player_prefix}_serve_pts' in match_data and f'{player_prefix}_serve_pts' != 0:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        first_won = match_data.get(f'{player_prefix}_first_won', 0)
        second_won = match_data.get(f'{player_prefix}_second_won', 0)
        features['serve_effectiveness'] = (first_won + second_won) / serve_pts if serve_pts > 0 else 0.65
    else:
        features['serve_effectiveness'] = 0.65

    # Return effectiveness
    if f'{player_prefix}_ta_return_won_pct' in match_data:
        features['return_effectiveness'] = match_data[f'{player_prefix}_ta_return_won_pct']
    elif f'{player_prefix}_return_pts_won' in match_data:
        return_pts = match_data.get(f'{player_prefix}_return_pts', 80)
        return_won = match_data.get(f'{player_prefix}_return_pts_won', 25)
        features['return_effectiveness'] = return_won / return_pts if return_pts > 0 else 0.35
    else:
        features['return_effectiveness'] = 0.35

    # Winners rate
    if f'{player_prefix}_ta_overview_winners' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70)
        total_pts = serve_pts + return_pts
        features['winners_rate'] = match_data[
                                       f'{player_prefix}_ta_overview_winners'] / total_pts if total_pts > 0 else 0.20
    elif f'{player_prefix}_winners_total' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['winners_rate'] = match_data[f'{player_prefix}_winners_total'] / serve_pts if serve_pts > 0 else 0.20
    else:
        features['winners_rate'] = 0.20

    # Unforced errors rate
    if f'{player_prefix}_ta_overview_unforced' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70)
        total_pts = serve_pts + return_pts
        features['unforced_rate'] = match_data[
                                        f'{player_prefix}_ta_overview_unforced'] / total_pts if total_pts > 0 else 0.18
    elif f'{player_prefix}_unforced_errors' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['unforced_rate'] = match_data[
                                        f'{player_prefix}_unforced_errors'] / serve_pts if serve_pts > 0 else 0.18
    else:
        features['unforced_rate'] = 0.18

    # Pressure performance
    if f'{player_prefix}_pressure_performance' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_pressure_performance']
    elif f'{player_prefix}_key_points_serve_won_pct' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_key_points_serve_won_pct']
    else:
        features['pressure_performance'] = 0.50

    # Net game effectiveness
    if f'{player_prefix}_ta_net_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_ta_net_won_pct']
    elif f'{player_prefix}_net_points_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_net_points_won_pct']
    else:
        features['net_effectiveness'] = 0.65

    return features


def extract_unified_match_context(match_data):
    """Extract match context from any data source"""
    context = {}

    # Surface
    context['surface'] = match_data.get('surface', match_data.get('Surface', 'Hard'))

    # Rankings
    context['p1_ranking'] = match_data.get('p1_ranking', match_data.get('WRank'))
    context['p2_ranking'] = match_data.get('p2_ranking', match_data.get('LRank'))

    # H2H data
    context['h2h_matches'] = match_data.get('h2h_matches', 0)
    context['p1_h2h_win_pct'] = match_data.get('p1_h2h_win_pct', 0.5)

    # Odds
    context['odds_p1'] = match_data.get('odds_p1', match_data.get('PSW'))
    context['odds_p2'] = match_data.get('odds_p2', match_data.get('PSL'))

    if context['odds_p1'] and context['odds_p2']:
        context['implied_prob_p1'] = 1 / context['odds_p1']
        context['implied_prob_p2'] = 1 / context['odds_p2']

    # Data quality
    context['source_rank'] = match_data.get('source_rank', 3)
    context['data_quality_score'] = calculate_data_quality_unified(match_data)

    return context


def calculate_data_quality_unified(match_data):
    """Calculate data quality score based on available features"""
    score = 0.3  # Base score

    # Source quality bonus
    source_rank = match_data.get('source_rank', 3)
    if source_rank == 1:  # Tennis Abstract
        score += 0.4
    elif source_rank == 2:  # API-Tennis
        score += 0.25
    elif source_rank == 3:  # Jeff/Tennis files
        score += 0.1

    # Feature availability bonus
    ta_features = sum(1 for k in match_data.keys() if 'ta_' in k)
    jeff_features = sum(
        1 for k in match_data.keys() if any(x in k for x in ['serve_pts', 'winners_total', 'pressure_performance']))
    api_features = sum(1 for k in match_data.keys() if any(x in k for x in ['ranking', 'odds', 'h2h']))

    if ta_features > 10:
        score += 0.25
    elif jeff_features > 8:
        score += 0.15
    elif api_features > 5:
        score += 0.1

    return min(score, 1.0)


def predict_match_unified(args, hist, jeff_data, defaults):
    """Enhanced prediction function that tries multiple composite_id variations"""

    match_date = pd.to_datetime(args.date).date()

    # Generate multiple tournament name variations
    tournament_base = args.tournament.lower().strip()
    tournament_variations = [
        tournament_base,  # Original
        tournament_base.replace(' ', '_'),  # Spaces to underscores
        tournament_base.replace('_', ' '),  # Underscores to spaces
        tournament_base.replace('-', ' '),  # Dashes to spaces
        tournament_base.replace(' ', ''),  # Remove spaces
        f"atp {tournament_base}",  # Add ATP prefix
        f"wta {tournament_base}",  # Add WTA prefix
        tournament_base.replace('atp ', ''),  # Remove ATP prefix
        tournament_base.replace('wta ', ''),  # Remove WTA prefix
    ]

    # Generate multiple player name variations
    def get_name_variations(player_name):
        base = normalize_name(player_name)
        variations = [base]

        # Try different formats
        parts = player_name.lower().split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            variations.extend([
                f"{last}_{first[0]}",  # lastname_f
                f"{first[0]}_{last}",  # f_lastname
                f"{first}_{last}",  # firstname_lastname
                f"{last}_{first}"  # lastname_firstname
            ])

        return list(set(variations))

    p1_variations = get_name_variations(args.player1)
    p2_variations = get_name_variations(args.player2)

    print(
        f"Trying {len(tournament_variations)} tournament × {len(p1_variations)} × {len(p2_variations)} = {len(tournament_variations) * len(p1_variations) * len(p2_variations)} combinations")

    # Try all combinations
    for tournament in tournament_variations:
        for p1 in p1_variations:
            for p2 in p2_variations:
                # Try both player orders
                for player1, player2 in [(p1, p2), (p2, p1)]:
                    comp_id = f"{match_date.strftime('%Y%m%d')}-{tournament}-{player1}-{player2}"

                    row = hist[hist["composite_id"] == comp_id]

                    if not row.empty:
                        print(f"✅ Found match: {comp_id}")

                        # Determine if players are swapped
                        match_row = row.iloc[0]
                        match_dict = match_row.to_dict()

                        # If we swapped players, swap the features back
                        if (player1, player2) == (p2, p1):
                            print("  → Players were swapped, correcting features...")
                            swapped_dict = {}
                            for key, value in match_dict.items():
                                if key.startswith('winner_'):
                                    swapped_dict[key.replace('winner_', 'loser_')] = value
                                elif key.startswith('loser_'):
                                    swapped_dict[key.replace('loser_', 'winner_')] = value
                                else:
                                    swapped_dict[key] = value
                            match_dict = swapped_dict

                        # Extract features and predict
                        p1_features = extract_unified_features(match_dict, 'winner')
                        p2_features = extract_unified_features(match_dict, 'loser')
                        match_context = extract_unified_match_context(match_dict)

                        # Show data source info
                        source_rank = match_dict.get('source_rank', 3)
                        data_sources = {1: 'Tennis Abstract', 2: 'API-Tennis', 3: 'Tennis Data Files'}
                        print(f"  → Data source: {data_sources.get(source_rank, 'Unknown')} (rank: {source_rank})")
                        print(f"  → Data quality: {match_context['data_quality_score']:.2f}")

                        # Show feature breakdown
                        print(f"\n=== UNIFIED FEATURE ANALYSIS ===")
                        print(f"Surface: {match_context.get('surface', 'Unknown')}")
                        print(
                            f"Rankings: P1={match_context.get('p1_ranking', 'N/A')}, P2={match_context.get('p2_ranking', 'N/A')}")
                        print(
                            f"H2H Record: {match_context.get('h2h_matches', 0)} matches, P1 win rate: {match_context.get('p1_h2h_win_pct', 0.5):.1%}")

                        if match_context.get('implied_prob_p1'):
                            print(
                                f"Market Odds: P1={match_context.get('implied_prob_p1'):.1%}, P2={match_context.get('implied_prob_p2'):.1%}")

                        print(f"\n=== PLAYER FEATURES ===")
                        for feature_name, p1_val in p1_features.items():
                            p2_val = p2_features.get(feature_name, 0)
                            print(f"{feature_name}: P1={p1_val:.3f}, P2={p2_val:.3f}")

                        # Run prediction
                        model = UnifiedBayesianTennisModel()
                        prob = model.simulate_match(p1_features, p2_features, match_context, best_of=args.best_of)

                        # Show results
                        print(f"\n=== PREDICTION RESULTS ===")
                        print(f"P({args.player1} wins) = {prob:.3f}")
                        print(f"P({args.player2} wins) = {1 - prob:.3f}")

                        return prob

    print("❌ No match found with any variation")
    return None

# Helper functions (should be imported from tennis_updated.py)
def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """Build composite match ID"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"


def normalize_tournament_name(name, gender=None):
    """Normalize tournament names"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(' ', '_')


def normalize_name(name):
    """Normalize player names"""
    if pd.isna(name):
        return ""
    return str(name).replace('.', '').lower().replace(' ', '_')

#%%
    # Run enhanced prediction
    model = BayesianTennisModel()
    prob = predict_match_unified(args, hist, jeff_data, defaults)

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


# Main Execution Block #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis match win probability predictor")
    parser.add_argument("--player1", required=True, help="Name of player 1")
    parser.add_argument("--player2", required=True, help="Name of player 2")
    parser.add_argument("--date", required=True, help="Match date in YYYY-MM-DD")
    parser.add_argument("--tournament", required=True, help="Tournament name")
    parser.add_argument("--gender", choices=["M", "W"], required=True, help="Gender: M or W")
    parser.add_argument("--best_of", type=int, default=3, help="Sets in match, default 3")
    args = parser.parse_args()

    print("🎾 TENNIS MATCH PREDICTION SYSTEM 🎾\n")

    # Load or generate data with Tennis Abstract integration
    hist, jeff_data, defaults = load_from_cache_with_scraping()
    if hist is None:
        print("No cache found. Generating full historical dataset...")
        hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=False)
        # Add Tennis Abstract scraping
        hist = run_automated_tennis_abstract_integration(hist)
        save_to_cache(hist, jeff_data, defaults)
        print("Historical data with Tennis Abstract integration cached for future use.")
    else:
        print("Loaded historical data from cache with Tennis Abstract integration.")

    # Integrate recent API data with full feature set
    print("Integrating recent API data with full feature extraction...")
    hist = integrate_api_tennis_data_incremental(hist)
    save_to_cache(hist, jeff_data, defaults)

    print(f"\n=== MATCH DETAILS ===")
    print(f"Date: {args.date}")
    print(f"Tournament: {args.tournament}")
    print(f"Player 1: {args.player1}")
    print(f"Player 2: {args.player2}")
    print(f"Gender: {args.gender}")
    print(f"Best of: {args.best_of}")

    # Run enhanced prediction with flexible lookup
    prob = predict_match_unified(args, hist, jeff_data, defaults)

    if prob is not None:
        print(f"\n=== FINAL PREDICTION ===")
        print(f"🏆 P({args.player1} wins) = {prob:.3f}")
        print(f"🏆 P({args.player2} wins) = {1 - prob:.3f}")

        # Add confidence assessment
        confidence = "High" if abs(prob - 0.5) > 0.2 else "Medium" if abs(prob - 0.5) > 0.1 else "Low"
        print(f"🎯 Prediction confidence: {confidence}")

    else:
        print("\nPREDICTION FAILED")
        print("No match data found. Possible reasons:")
        print("- Match not in dataset (check date, tournament, player names)")
        print("- Tournament name mismatch (try different format)")
        print("- Players not in our database")

        print(f"\nSuggestions:")
        print(f"- Try 'Wimbledon' instead of '{args.tournament}'")
        print(f"- Check player name spelling")
        print(f"- Verify match date")

    print("\nPREDICTION COMPLETE ")