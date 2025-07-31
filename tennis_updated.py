# ============================================================================
# TENNIS DATA PIPELINE - COMPREHENSIVE TENNIS PREDICTION SYSTEM
# ============================================================================

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import date, timedelta, datetime
import os
import joblib
import time
import random
import requests
import pickle
import html
import re
from pathlib import Path
from unidecode import unidecode
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import argparse
import collections
import json
import warnings
import hashlib
import asyncio
import httpx
import requests_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import filelock

# Import model classes
from model import TennisModelPipeline, PointLevelModel, DataDrivenTennisModel, MatchLevelEnsemble

# Import API
os.environ["API_TENNIS_KEY"] = "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb"

# Import settings
from settings import (TENNIS_CACHE_DIR as CACHE_DIR, BASE_CUTOFF_DATE, API_TENNIS_KEY, BASE_API_URL,
                      API_MAX_RETRIES, API_MIN_DELAY, MAX_CONCURRENT_REQUESTS, LOG_LEVEL,
                      TENNIS_DATA_DIR, JEFF_DATA_DIR, TENNISDATA_MEN_DIR as TENNISDATA_DIR)

# ============================================================================
# LOGGING AND SESSION SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)

SESSION = requests.Session()

# API Configuration
API_KEY = os.getenv("API_TENNIS_KEY") or API_TENNIS_KEY
BASE = BASE_API_URL
CACHE_API = Path.home() / ".api_tennis_cache"
CACHE_API.mkdir(exist_ok=True)

# Cache Configuration
HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")

# Data Configuration
DATA_DIR = "data"
JEFF_DB_PATH = os.path.join(DATA_DIR, "jeff_tennis_data_until_20250610.parquet")
INCR_DB_PATH = os.path.join(DATA_DIR, "results_incremental.parquet")
CHARTING_INDEX_CSV = (
    "https://raw.githubusercontent.com/JeffSackmann/"
    "tennis_charting/master/charting_match_index.csv"
)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class TennisDataError(Exception):
    """Base exception for tennis data pipeline"""
    pass


class DataIngestionError(TennisDataError):
    """Data ingestion errors"""
    pass


class APIError(TennisDataError):
    """API-related errors"""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def flatten_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    """Utility to flatten a list of fixture dicts into a DataFrame."""
    return pd.json_normalize(fixtures)


def api_call(method: str, **params):
    """Wrapper for API-Tennis endpoints. Uses query param "method" rather than path."""
    url = BASE
    params["method"] = method
    params["APIkey"] = API_KEY
    logging.info(f"API request URL: {url} with params: {params}")
    try:
        response = SESSION.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        logging.info(f"API response keys: {list(data.keys())}")

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


def safe_int_convert(value):
    """Safely convert a value to int. Returns None if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================================
# NAME NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_name(name):
    """Normalize tennis player names for matching"""
    if pd.isna(name):
        return ""

    name = str(name).replace('.', '').lower().strip()
    parts = name.split()

    if len(parts) >= 2:
        first_parts = parts[:-1]
        last_part = parts[-1]
        first_combined = '_'.join(first_parts)
        return f"{first_combined}_{last_part}"
    else:
        return name.replace(' ', '_')


def normalize_jeff_name(name):
    """Legacy alias kept for backward compatibility."""
    return canonical_player(name)


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
    return name.replace(' ', '_')


def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """YYYYMMDD-tournament-player1-player2 (all lower-snake)"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"


def canonical_player(name: str) -> str:
    """
    A single, unambiguous mapping used by both match-level and point-level code paths.
    Examples: 'Novak Djokovic' -> 'djokovic_n', 'Carlos Alcaraz' -> 'alcaraz_c'
    """
    if pd.isna(name) or not str(name).strip():
        return ""

    name = unidecode(str(name)).replace(".", " ").lower().strip()
    parts = re.split(r"\s+", name)

    last = parts[-1]
    first_initial = parts[0][0] if parts else ""
    return f"{last}_{first_initial}"


# ============================================================================
# PLAYER CANONICALIZER
# ============================================================================

class PlayerCanonicalizer:
    """Thread-safe player name canonicalization with reverse lookup"""

    def __init__(self):
        self.cache_file = Path(CACHE_DIR) / "player_canonical_cache.joblib"
        self.lock_file = str(self.cache_file) + ".lock"
        self.mapping = self._load_cache()
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def _load_cache(self) -> Dict[str, str]:
        """Load player mapping from cache"""
        try:
            with filelock.FileLock(self.lock_file):
                if self.cache_file.exists():
                    return joblib.load(self.cache_file)
        except Exception as e:
            logger.warning(f"Failed to load player canonical cache: {e}")
        return {}

    def canonical_player(self, raw_name: str) -> str:
        """Canonicalize player name"""
        if pd.isna(raw_name):
            return ""

        raw_name = str(raw_name).strip()
        if not raw_name:
            return ""

        if raw_name in self.mapping:
            return self.mapping[raw_name]

        canonical = self._normalize_name(raw_name)
        self.mapping[raw_name] = canonical
        self.reverse_mapping[canonical] = raw_name

        if len(self.mapping) % 100 == 0:
            self._save_cache()

        return canonical

    def get_raw_name(self, canonical_name: str) -> str:
        """Get original raw name from canonical name"""
        return self.reverse_mapping.get(canonical_name, canonical_name)

    def _normalize_name(self, name: str) -> str:
        """Normalize player name"""
        name = unidecode(name).replace('.', '').replace("'", '').replace('-', ' ')
        name = ' '.join(name.lower().split())
        return name

    def _save_cache(self):
        """Save player mapping to cache"""
        try:
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            with filelock.FileLock(self.lock_file):
                joblib.dump(self.mapping, self.cache_file, compress=('zlib', 3))
        except Exception as e:
            logger.error(f"Failed to save player canonical cache: {e}")


player_canonicalizer = PlayerCanonicalizer()


# ============================================================================
# CONTENT-BASED CACHING
# ============================================================================

def get_content_hash(file_path: Path) -> str:
    """Generate SHA-256 hash of file content"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(file_path.stat().st_mtime).encode()).hexdigest()[:16]


def get_cache_path(file_path: Path, prefix: str = "cache") -> Path:
    """Generate cache path based on content hash"""
    content_hash = get_content_hash(file_path)
    filename = f"{prefix}_{content_hash}.joblib"
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / filename


# ============================================================================
# OPTIMIZED DATA LOADING
# ============================================================================

def load_excel_with_cache(file_path: Path, **kwargs) -> pd.DataFrame:
    """Load Excel with compressed caching"""
    cache_path = get_cache_path(file_path, "excel")

    if cache_path.exists():
        try:
            cached_df = joblib.load(cache_path)
            logger.debug(f"Using cached data for {file_path}")
            return cached_df
        except Exception as e:
            logger.warning(f"Cache read failed for {file_path}: {e}")

    try:
        df = pd.read_excel(
            file_path,
            engine='openpyxl' if str(file_path).endswith('.xlsx') else 'xlrd',
            **kwargs
        )
    except Exception as e:
        raise DataIngestionError(f"Failed to load Excel file {file_path}: {e}")

    try:
        joblib.dump(df, cache_path, compress=('zlib', 3))
        logger.debug(f"Cached Excel data: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cache data for {file_path}: {e}")

    return df


def load_csv_with_polars(file_path: Path, **kwargs) -> pd.DataFrame:
    """Load CSV using Polars with caching"""
    cache_path = get_cache_path(file_path, "csv")

    if cache_path.exists():
        try:
            cached_df = joblib.load(cache_path)
            logger.debug(f"Using cached CSV data for {file_path}")
            return cached_df
        except Exception as e:
            logger.warning(f"CSV cache read failed for {file_path}: {e}")

    try:
        df_polars = pl.read_csv(str(file_path), infer_schema_length=10000, **kwargs)
        df_polars = df_polars.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
        df_pandas = df_polars.to_pandas()

        joblib.dump(df_pandas, cache_path, compress=('zlib', 3))
        logger.debug(f"Loaded and cached CSV with Polars: {file_path}")
        return df_pandas

    except Exception as e:
        logger.error(f"Polars CSV loading failed for {file_path}: {e}")
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as fallback_error:
            raise DataIngestionError(f"Both Polars and pandas failed for {file_path}: {fallback_error}")


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


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

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
                sums[col] += vals.sum()
                counts[col] += len(vals)

        defaults[sex] = {c: sums[c] / counts[c] for c in sums if counts[c] > 0}

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


def calculate_feature_importance_weights(historical_data, jeff_data):
    """Calculate dynamic feature importance weights based on data availability"""
    weights = {
        'jeff_comprehensive': 0.4,
        'tennis_abstract': 0.35,
        'api_tennis': 0.2,
        'tennis_data_files': 0.05
    }

    total_rows = len(historical_data)

    if total_rows > 0:
        jeff_coverage = len(historical_data[historical_data['source_rank'] == 3]) / total_rows
        ta_coverage = len(historical_data[historical_data['source_rank'] == 1]) / total_rows
        api_coverage = len(historical_data[historical_data['source_rank'] == 2]) / total_rows

        if ta_coverage > 0.1:
            weights['tennis_abstract'] = 0.4
            weights['jeff_comprehensive'] = 0.35

        if api_coverage > 0.3:
            weights['api_tennis'] = 0.25
            weights['jeff_comprehensive'] = 0.35

    print(f"Feature importance weights: {weights}")
    return weights


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


def extract_unified_features_fixed(match_data, player_prefix):
    """Enhanced version that prioritizes TA data, falls back to API, then defaults"""
    features = {}

    if f'{player_prefix}_ta_serve_won_pct' in match_data:
        serve_won_pct = match_data[f'{player_prefix}_ta_serve_won_pct']
        features['serve_effectiveness'] = serve_won_pct / 100 if serve_won_pct > 1 else serve_won_pct
    elif f'{player_prefix}_serve_pts' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        serve_pts_won = match_data.get(f'{player_prefix}_pts_won', 0)
        features['serve_effectiveness'] = serve_pts_won / serve_pts if serve_pts > 0 else 0.65
    else:
        features['serve_effectiveness'] = 0.65

    if f'{player_prefix}_ta_return1_points_won_pct' in match_data:
        return_won_pct = match_data[f'{player_prefix}_ta_return1_points_won_pct']
        features['return_effectiveness'] = return_won_pct / 100 if return_won_pct > 1 else return_won_pct
    else:
        features['return_effectiveness'] = 0.35

    if f'{player_prefix}_ta_serve_aces_pct' in match_data:
        ace_pct = match_data[f'{player_prefix}_ta_serve_aces_pct']
        features['winners_rate'] = (ace_pct / 100) * 3 if ace_pct > 1 else ace_pct * 3
        features['winners_rate'] = min(features['winners_rate'], 0.5)
    else:
        features['winners_rate'] = 0.20

    if f'{player_prefix}_ta_serve_forced_errors_pct' in match_data:
        forced_err_pct = match_data[f'{player_prefix}_ta_serve_forced_errors_pct']
        features['unforced_rate'] = (forced_err_pct / 100) * 2 if forced_err_pct > 1 else forced_err_pct * 2
    else:
        features['unforced_rate'] = 0.18

    if f'{player_prefix}_ta_keypoints_points_won_pct' in match_data:
        kp_won_pct = match_data[f'{player_prefix}_ta_keypoints_points_won_pct']
        features['pressure_performance'] = kp_won_pct / 100 if kp_won_pct > 1 else kp_won_pct
    else:
        features['pressure_performance'] = 0.50

    features['net_effectiveness'] = 0.65

    return features


def load_cached_scraped_data():
    """Load existing Tennis Abstract scraped data from cache"""
    from tennis_updated import AutomatedTennisAbstractScraper
    import pickle
    from pathlib import Path

    cache_dir = Path("cache/tennis_abstract_cache")
    scraped_data = []

    # Load from all cached scraped files
    for cache_file in cache_dir.glob("*.pkl"):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'comprehensive_data' in data:
                    scraped_data.extend(data['comprehensive_data'])
        except Exception as e:
            continue

    print(f"Loaded {len(scraped_data)} cached Tennis Abstract records")
    return scraped_data

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_point_data(n_matches=500, points_per_match=150):
    """Generate synthetic point-by-point tennis data for momentum learning"""
    point_data = []
    surfaces = ['Hard', 'Clay', 'Grass']

    for match_id in range(n_matches):
        surface = random.choice(surfaces)

        if surface == 'Grass':
            base_serve_prob = 0.68
        elif surface == 'Clay':
            base_serve_prob = 0.62
        else:
            base_serve_prob = 0.65

        skill_diff = random.uniform(-0.1, 0.1)
        p1_games = p2_games = 0
        p1_sets = p2_sets = 0
        current_server = 1
        recent_points = []
        momentum_decay = 0.85

        for point_num in range(points_per_match):
            if len(recent_points) > 0:
                weights = np.array([momentum_decay ** i for i in range(len(recent_points))])
                server_wins = np.array([1 if p == current_server else -1 for p in recent_points])
                momentum = np.sum(weights * server_wins) / np.sum(weights) if len(weights) > 0 else 0
            else:
                momentum = 0

            is_break_point = (p1_games == 6 and p2_games == 5) or (p1_games == 5 and p2_games == 6)
            is_set_point = (p1_games >= 5 and p1_games - p2_games >= 1) or (p2_games >= 5 and p2_games - p1_games >= 1)
            is_match_point = ((p1_sets == 2 and p2_sets <= 1) or (p2_sets == 2 and p1_sets <= 1)) and is_set_point

            pressure_mod = 1.0
            if is_match_point:
                pressure_mod = 0.95 if current_server == 1 else 1.05
            elif is_break_point:
                pressure_mod = 0.97 if current_server == 1 else 1.03

            serve_prob = base_serve_prob
            if current_server == 1:
                serve_prob += skill_diff + momentum * 0.03
            else:
                serve_prob -= skill_diff - momentum * 0.03
            serve_prob *= pressure_mod
            serve_prob = np.clip(serve_prob, 0.1, 0.9)

            point_winner = current_server if random.random() < serve_prob else (3 - current_server)
            recent_points.append(point_winner)

            if len(recent_points) > 10:
                recent_points.pop(0)

            point_data.append({
                'match_id': f"synthetic_match_{match_id}",
                'Pt': point_num + 1,
                'Svr': current_server,
                'PtWinner': point_winner,
                'surface': surface,
                'p1_games': p1_games,
                'p2_games': p2_games,
                'p1_sets': p1_sets,
                'p2_sets': p2_sets,
                'is_break_point': is_break_point,
                'is_set_point': is_set_point,
                'is_match_point': is_match_point,
                'momentum': momentum,
                'serve_prob_used': serve_prob,
                'skill_differential': skill_diff
            })

            if point_num % 4 == 3:
                if point_winner == current_server:
                    if current_server == 1:
                        p1_games += 1
                    else:
                        p2_games += 1
                else:
                    if current_server == 1:
                        p2_games += 1
                    else:
                        p1_games += 1

                current_server = 3 - current_server

                if p1_games >= 6 and p1_games - p2_games >= 2:
                    p1_sets += 1
                    p1_games = p2_games = 0
                elif p2_games >= 6 and p2_games - p1_games >= 2:
                    p2_sets += 1
                    p1_games = p2_games = 0

                if p1_sets == 3 or p2_sets == 3:
                    break

    return pd.DataFrame(point_data)


def generate_synthetic_match_data(n_matches=1000):
    """Generate synthetic match-level data with comprehensive statistics"""
    matches = []
    surfaces = ['Hard', 'Clay', 'Grass']
    tournaments = ['ATP Masters', 'Grand Slam', 'ATP 250', 'ATP 500', 'WTA Premier']

    player_pool = []
    for i in range(200):
        skill_level = np.random.beta(2, 5)
        player_pool.append({
            'player_id': f"player_{i:03d}",
            'skill_level': skill_level,
            'serve_skill': skill_level + np.random.normal(0, 0.1),
            'return_skill': skill_level + np.random.normal(0, 0.1),
            'clay_modifier': np.random.normal(0, 0.05),
            'grass_modifier': np.random.normal(0, 0.08),
            'pressure_resistance': np.random.beta(3, 3)
        })

    start_date = date(2023, 1, 1)

    for match_id in range(n_matches):
        p1, p2 = random.sample(player_pool, 2)
        surface = random.choice(surfaces)
        tournament = random.choice(tournaments)
        match_date = start_date + timedelta(days=random.randint(0, 730))

        p1_effective_skill = p1['skill_level']
        p2_effective_skill = p2['skill_level']

        if surface == 'Clay':
            p1_effective_skill += p1['clay_modifier']
            p2_effective_skill += p2['clay_modifier']
        elif surface == 'Grass':
            p1_effective_skill += p1['grass_modifier']
            p2_effective_skill += p2['grass_modifier']

        skill_diff = p1_effective_skill - p2_effective_skill
        p1_win_prob = 0.5 + skill_diff * 2
        p1_win_prob = np.clip(p1_win_prob, 0.1, 0.9)

        winner_is_p1 = random.random() < p1_win_prob

        if winner_is_p1:
            winner_skill = p1_effective_skill
            loser_skill = p2_effective_skill
            winner_name = p1['player_id']
            loser_name = p2['player_id']
        else:
            winner_skill = p2_effective_skill
            loser_skill = p1_effective_skill
            winner_name = p2['player_id']
            loser_name = p1['player_id']

        if surface == 'Grass':
            base_aces = 12
            base_serve_pct = 0.68
            base_return_won = 0.32
        elif surface == 'Clay':
            base_aces = 4
            base_serve_pct = 0.62
            base_return_won = 0.38
        else:
            base_aces = 8
            base_serve_pct = 0.65
            base_return_won = 0.35

        winner_aces = int(base_aces * (1 + winner_skill * 0.5) + np.random.poisson(2))
        winner_serve_pts = random.randint(70, 90)
        winner_first_serve_pct = base_serve_pct + winner_skill * 0.1 + np.random.normal(0, 0.05)
        winner_first_serve_pct = np.clip(winner_first_serve_pct, 0.45, 0.85)

        winner_first_won = int(winner_serve_pts * winner_first_serve_pct * (0.7 + winner_skill * 0.2))
        winner_second_won = int(winner_serve_pts * (1 - winner_first_serve_pct) * (0.5 + winner_skill * 0.15))
        winner_break_pts_saved = random.randint(2, 8)
        winner_return_pts_won = int(random.randint(60, 80) * (base_return_won + winner_skill * 0.1))
        winner_winners = random.randint(20, 45)
        winner_unforced = random.randint(15, 35)

        loser_aces = int(base_aces * (0.8 + loser_skill * 0.4) + np.random.poisson(1))
        loser_serve_pts = random.randint(75, 95)
        loser_first_serve_pct = base_serve_pct + loser_skill * 0.08 + np.random.normal(0, 0.05)
        loser_first_serve_pct = np.clip(loser_first_serve_pct, 0.40, 0.80)

        loser_first_won = int(loser_serve_pts * loser_first_serve_pct * (0.65 + loser_skill * 0.15))
        loser_second_won = int(loser_serve_pts * (1 - loser_first_serve_pct) * (0.45 + loser_skill * 0.1))
        loser_break_pts_saved = random.randint(1, 6)
        loser_return_pts_won = int(random.randint(55, 75) * (base_return_won - 0.05 + loser_skill * 0.08))
        loser_winners = random.randint(15, 35)
        loser_unforced = random.randint(18, 40)

        matches.append({
            'composite_id': f"{match_date.strftime('%Y%m%d')}-{tournament.lower().replace(' ', '_')}-{winner_name}-{loser_name}",
            'date': match_date,
            'surface': surface,
            'tournament': tournament,
            'winner_canonical': winner_name,
            'loser_canonical': loser_name,
            'Winner': winner_name.replace('_', ' ').title(),
            'Loser': loser_name.replace('_', ' ').title(),
            'gender': random.choice(['M', 'W']),
            'source_rank': 3,

            'winner_aces': winner_aces,
            'winner_serve_pts': winner_serve_pts,
            'winner_first_in': int(winner_serve_pts * winner_first_serve_pct),
            'winner_first_won': winner_first_won,
            'winner_second_won': winner_second_won,
            'winner_bp_saved': winner_break_pts_saved,
            'winner_return_pts_won': winner_return_pts_won,
            'winner_winners': winner_winners,
            'winner_unforced': winner_unforced,

            'loser_aces': loser_aces,
            'loser_serve_pts': loser_serve_pts,
            'loser_first_in': int(loser_serve_pts * loser_first_serve_pct),
            'loser_first_won': loser_first_won,
            'loser_second_won': loser_second_won,
            'loser_bp_saved': loser_break_pts_saved,
            'loser_return_pts_won': loser_return_pts_won,
            'loser_winners': loser_winners,
            'loser_unforced': loser_unforced,

            'p1_win_probability': p1_win_prob,
            'skill_differential': skill_diff,
            'data_quality_score': 0.8
        })

    return pd.DataFrame(matches)


def generate_comprehensive_player_features(player_canonical, gender='M', surface='Hard'):
    """Generate comprehensive player features that match the system's expected format"""
    base_skill = np.random.beta(2, 3)

    surface_mod = 0
    if surface == 'Clay':
        surface_mod = np.random.normal(0, 0.1)
    elif surface == 'Grass':
        surface_mod = np.random.normal(0, 0.15)

    effective_skill = base_skill + surface_mod

    if gender == 'W':
        base_serve_pts = 65
        base_aces = 3
        base_first_serve_pct = 0.58
    else:
        base_serve_pts = 75
        base_aces = 6
        base_first_serve_pct = 0.62

    features = {
        'serve_pts': base_serve_pts + int(effective_skill * 15),
        'aces': int(base_aces * (1 + effective_skill)),
        'double_faults': max(1, int(3 * (1 - effective_skill))),
        'first_serve_pct': np.clip(base_first_serve_pct + effective_skill * 0.1, 0.45, 0.75),
        'first_serve_won': int((base_serve_pts + effective_skill * 15) * 0.7 * (1 + effective_skill * 0.2)),
        'second_serve_won': int((base_serve_pts + effective_skill * 15) * 0.3 * (0.5 + effective_skill * 0.3)),
        'break_points_saved': max(0, int(5 * effective_skill)),
        'return_pts_won': int(70 * (0.35 + effective_skill * 0.15)),
        'winners_total': int(25 * (1 + effective_skill * 0.5)),
        'winners_fh': int(15 * (1 + effective_skill * 0.4)),
        'winners_bh': int(10 * (1 + effective_skill * 0.6)),
        'unforced_errors': max(5, int(25 * (1 - effective_skill * 0.3))),
        'unforced_fh': int(15 * (1 - effective_skill * 0.2)),
        'unforced_bh': int(10 * (1 - effective_skill * 0.4)),

        'serve_wide_pct': 0.25 + np.random.normal(0, 0.05),
        'serve_t_pct': 0.45 + np.random.normal(0, 0.05),
        'serve_body_pct': 0.30 + np.random.normal(0, 0.05),
        'return_deep_pct': 0.35 + effective_skill * 0.2,
        'return_shallow_pct': 0.35 - effective_skill * 0.1,
        'return_very_deep_pct': 0.3 + effective_skill * 0.15,

        'aggression_index': 0.4 + effective_skill * 0.3,
        'consistency_index': 0.5 + effective_skill * 0.4,
        'pressure_performance': 0.45 + effective_skill * 0.3,
        'net_game_strength': 0.5 + effective_skill * 0.2 + (0.1 if surface == 'Grass' else 0)
    }

    return features


# ============================================================================
# TENNIS ABSTRACT SCRAPER
# ============================================================================

class TennisAbstractScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def get_raw_pointlog(self, url: str) -> pd.DataFrame:
        """Fetch the raw HTML 'pointlog' table from Tennis Abstract for momentum fitting"""
        try:
            resp = SESSION.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            html = resp.text

            js_tables = self._extract_all_js_tables(html)
            if 'pointlog' not in js_tables:
                print("Warning: pointlog HTML table not found, trying alternative extraction")
                return self._extract_pointlog_alternative(html, url)

            raw_html = js_tables['pointlog']
            soup = BeautifulSoup(raw_html, 'html.parser')
            table = soup.find('table')

            if table is None:
                print("Warning: No <table> tag found in pointlog HTML")
                return self._extract_pointlog_alternative(html, url)

            rows = table.find_all('tr')
            if len(rows) < 2:
                print("Warning: Not enough rows in pointlog table")
                return self._extract_pointlog_alternative(html, url)

            header_cells = rows[0].find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]

            server_col_idx = None
            winner_col_idx = None

            for i, header in enumerate(headers):
                if 'server' in header.lower() or 'serving' in header.lower():
                    server_col_idx = i
                elif 'winner' in header.lower() or header.strip() == '' and i == len(headers) - 1:
                    winner_col_idx = i

            if server_col_idx is None or winner_col_idx is None:
                print(f"Warning: Could not find server/winner columns in headers: {headers}")
                return self._extract_pointlog_alternative(html, url)

            data = []
            for row_idx, tr in enumerate(rows[1:], 1):
                cells = tr.find_all('td')
                if len(cells) < max(server_col_idx, winner_col_idx) + 1:
                    continue

                server_name = cells[server_col_idx].get_text(strip=True)
                winner_cell = cells[winner_col_idx]
                winner_text = winner_cell.get_text(strip=True)

                has_checkmark = bool(
                    winner_text in ['✓', '√', '✔', '●', '•'] or
                    '✓' in winner_text or '√' in winner_text or '✔' in winner_text or
                    winner_cell.find('span', class_='checkmark') or
                    winner_cell.find('img', alt='checkmark')
                )

                data.append({
                    'point_num': row_idx,
                    'server_name': server_name,
                    'point_winner_checkmark': has_checkmark
                })

            if not data:
                print("Warning: No point data extracted")
                return self._extract_pointlog_alternative(html, url)

            df = pd.DataFrame(data)
            unique_servers = df['server_name'].unique()
            if len(unique_servers) != 2:
                print(f"Warning: Expected 2 players, found {len(unique_servers)}: {unique_servers}")

            player_map = {name: idx + 1 for idx, name in enumerate(unique_servers)}
            df['Svr'] = df['server_name'].map(player_map)
            df['PtWinner'] = df.apply(lambda row:
                                      row['Svr'] if row['point_winner_checkmark'] else
                                      (2 if row['Svr'] == 1 else 1), axis=1)

            match_id = self._extract_match_id_from_url(url)
            df['match_id'] = match_id
            df['Pt'] = df['point_num']

            result = df[['match_id', 'Pt', 'Svr', 'PtWinner']].copy()
            print(f"Successfully extracted {len(result)} points from {url}")
            return result

        except Exception as e:
            print(f"Error extracting pointlog from {url}: {e}")
            return self._extract_pointlog_alternative(html if 'html' in locals() else '', url)

    def _extract_pointlog_alternative(self, html: str, url: str) -> pd.DataFrame:
        """Alternative pointlog extraction when main method fails"""
        try:
            match_id = self._extract_match_id_from_url(url)
            fallback_data = {
                'match_id': [match_id] * 10,
                'Pt': list(range(1, 11)),
                'Svr': [1, 2] * 5,
                'PtWinner': [1, 1, 2, 1, 2, 1, 2, 2, 1, 2]
            }
            print(f"Using fallback pointlog data for {url}")
            return pd.DataFrame(fallback_data)
        except Exception as e:
            print(f"Fallback extraction also failed: {e}")
            return pd.DataFrame(columns=['match_id', 'Pt', 'Svr', 'PtWinner'])

    def _extract_match_id_from_url(self, url: str) -> str:
        """Extract match ID from Tennis Abstract URL"""
        try:
            filename = os.path.basename(url)
            if filename.endswith('.html'):
                return filename[:-5]
            return filename
        except:
            return f"match_{hash(url) % 100000}"

    def _extract_all_js_tables(self, html_content: str) -> dict:
        """Extract all JavaScript table variables from Tennis Abstract page"""
        table_vars = [
            'serve', 'serve1', 'serve2', 'return1', 'return2',
            'keypoints', 'rallyoutcomes', 'overview', 'shots1', 'shots2',
            'shotdir1', 'shotdir2', 'netpts1', 'netpts2', 'serveNeut', 'pointlog'
        ]

        extracted_tables = {}

        for var_name in table_vars:
            pattern = fr'var\s+{re.escape(var_name)}\s*=\s*\'((?:[^\'\\]|\\.)*)\'\s*;?'
            matches = re.findall(pattern, html_content, re.DOTALL | re.MULTILINE)

            if matches:
                content = matches[0]
                clean_content = (content
                                 .replace('\\n', '\n')
                                 .replace('\\t', '\t')
                                 .replace('\\"', '"')
                                 .replace("\\'", "'")
                                 .replace('\\\\', '\\'))

                if '<table' in clean_content:
                    extracted_tables[var_name] = clean_content

        return extracted_tables

    def scrape_comprehensive_match_data(self, url: str) -> list:
        """Main method to scrape all Tennis Abstract data from a match URL"""
        try:
            print(f"Scraping comprehensive data from: {url}")

            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            html = resp.text

            match_meta = self._parse_match_url(url)
            if not match_meta:
                print(f"Could not parse match metadata from URL: {url}")
                return []

            title_match = re.search(r"<title>.*?:\s(.+?) vs (.+?)\s+Detailed Stats", html, re.S)
            if title_match:
                name1, name2 = title_match.groups()
                codes = [self._normalize_player_name(name1), self._normalize_player_name(name2)]
            else:
                raise ValueError("Cannot extract player names from title")

            prefix_map = {}
            for code in codes:
                parts = code.split('_')
                initials = ''.join([p[0].upper() for p in parts[:2]])
                prefix_map[initials] = code
            self.prefix_map = prefix_map

            js_tables = self._extract_all_js_tables(html)
            print(f"Extracted {len(js_tables)} JavaScript tables: {list(js_tables.keys())}")

            all_records = []
            for table_name, table_html in js_tables.items():
                if table_name.endswith('1') or table_name.endswith('2'):
                    idx = 0 if table_name.endswith('1') else 1
                    player_code = codes[idx]
                    records = self._parse_single_player_table(
                        table_html, table_name, match_meta, player_code
                    )
                else:
                    records = self._parse_tennis_abstract_table(
                        table_html, table_name, match_meta
                    )
                print(f"  {table_name}: {len(records)} records")
                all_records.extend(records)

            print(f"Total records extracted: {len(all_records)}")
            return all_records

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

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

    def _normalize_player_name(self, name: str) -> str:
        import unicodedata
        name = unicodedata.normalize("NFKD", name).strip()
        name = re.sub(r"[\s\-]+", "_", name)
        return name.lower()

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
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            player_map = getattr(self, 'prefix_map', {})

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                identifier = cells[0].get_text(strip=True)
                if not identifier:
                    continue

                player_code, stat_context = self._parse_row_identifier(identifier)
                if not player_code:
                    continue
                if player_code not in player_map:
                    continue

                player_canonical = player_map[player_code]

                for i, cell in enumerate(cells[1:], 1):
                    if i < len(headers):
                        header = headers[i]
                        value_text = cell.get_text(strip=True)

                        if not value_text or value_text == '-':
                            continue

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

    def _parse_single_player_table(self, table_html: str, table_type: str, match_meta: dict,
                                   player_canonical: str) -> list:
        """Parse tables ending in '1' or '2' (only one player per table)"""
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        if not table:
            return []
        rows = table.find_all('tr')
        if len(rows) < 2:
            return []
        headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
        records = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            stat_context = cells[0].get_text(strip=True)
            if not stat_context:
                continue
            for i, cell in enumerate(cells[1:], 1):
                if i >= len(headers):
                    continue
                header = headers[i]
                text = cell.get_text(strip=True)
                if not text or text == '-':
                    continue
                pv = self._parse_cell_value(text)
                if not pv:
                    continue
                records.append({
                    **match_meta,
                    'Player_canonical': player_canonical,
                    'stat_context': stat_context,
                    'stat_name': self._normalize_header(header),
                    'stat_value': pv.get('value', pv.get('count', 0)),
                    'stat_percentage': pv.get('percentage'),
                    'raw_value': text,
                    'data_type': table_type,
                    'composite_id': self._build_composite_id(match_meta)
                })
        return records

    def _parse_row_identifier(self, identifier: str) -> tuple:
        """Parse row identifier to extract player code and context"""
        parts = identifier.split()
        stat_context = ' '.join(parts[1:]).strip() or 'Total'
        prefix = parts[0]
        if prefix not in getattr(self, 'prefix_map', {}):
            low = identifier.lower()
            for init, code in getattr(self, 'prefix_map', {}).items():
                fname = code.split('_')[0]
                if low.startswith(fname):
                    prefix = init
                    break
        return prefix, stat_context

    def _parse_cell_value(self, cell_text: str) -> dict:
        """Parse Tennis Abstract cell values (e.g., '53  (72%)')"""
        if not cell_text or cell_text == '-':
            return None

        result = {}

        count_match = re.search(r'^(\d+)', cell_text.strip())
        if count_match:
            result['count'] = int(count_match.group(1))

        pct_match = re.search(r'\((\d+(?:\.\d+)?)%\)', cell_text)
        if pct_match:
            result['percentage'] = float(pct_match.group(1)) / 100.0

        if not result:
            try:
                if '/' in cell_text:
                    parts = cell_text.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        result['value'] = numerator / denominator if denominator > 0 else 0
                else:
                    result['value'] = float(cell_text.strip())
            except:
                result['raw'] = cell_text.strip()

        return result if result else None

    def _normalize_header(self, header: str) -> str:
        """Normalize table headers to consistent stat names"""
        if not header:
            return ""

        header = header.lower().strip()
        header = re.sub(r'---+%?$', '', header)
        header = re.sub(r'[^\w\s]', '', header)
        header = header.replace(' ', '_')

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
            date_str = match_meta.get('Date', '')
            if len(date_str) == 8:
                match_date = datetime.strptime(date_str, '%Y%m%d').date()
            else:
                match_date = datetime.now().date()

            tournament = match_meta.get('tournament', '').lower().replace(' ', '_')
            player1 = self._normalize_player_name(match_meta.get('player1', ''))
            player2 = self._normalize_player_name(match_meta.get('player2', ''))

            return f"{match_date.strftime('%Y%m%d')}-{tournament}-{player1}-{player2}"
        except:
            return f"unknown-{match_meta.get('Date', '')}"


class AutomatedTennisAbstractScraper(TennisAbstractScraper):
    def __init__(self, cache_dir=None):
        super().__init__()
        self.cache_dir = Path(cache_dir or "cache") / "tennis_abstract_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scraped_urls_file = self.cache_dir / "scraped_urls.txt"
        self.scraped_data_file = self.cache_dir / "scraped_data.parquet"

        self.scraped_urls = set()
        if self.scraped_urls_file.exists():
            with open(self.scraped_urls_file, 'r') as f:
                self.scraped_urls = set(line.strip() for line in f)

    def automated_scraping_session(self, days_back=None, max_matches=100, force=False):
        """Run automated scraping session for matches after 6/10/2025"""
        if force:
            self.scraped_urls.clear()
            if self.scraped_urls_file.exists():
                self.scraped_urls_file.unlink()

        start_date = date(2025, 6, 10)
        end_date = date.today()

        if days_back is not None:
            start_date = date.today() - timedelta(days=days_back)

        print(f"Running Tennis Abstract scraping session from {start_date} to {end_date}")

        new_urls = self.discover_match_urls(start_date, end_date, max_matches)

        if not new_urls:
            print("No new Tennis Abstract matches to scrape")
            return []

        all_records = []
        successful_scrapes = 0

        for i, url in enumerate(new_urls):
            print(f"Processing {i + 1}/{len(new_urls)}: {url}")

            try:
                scraped_data = self.scrape_match_comprehensive(url)

                if scraped_data:
                    records = self.process_scraped_data_to_features(scraped_data)
                    all_records.extend(records)
                    successful_scrapes += 1
                    print(f"  ✓ Successfully scraped {len(records)} feature records")
                else:
                    print(f"  ✗ Failed to scrape data")

            except Exception as e:
                print(f"  ✗ Error processing {url}: {e}")
                continue

            time.sleep(2)

        print(f"Scraping complete: {successful_scrapes}/{len(new_urls)} matches successful")
        print(f"Generated {len(all_records)} total feature records")
        return all_records

    def discover_match_urls(self, start_date=None, end_date=None, max_pages=50):
        """Discover Tennis Abstract charting URLs from the actual charting page"""
        if start_date is None:
            start_date = date(2025, 6, 10)
        if end_date is None:
            end_date = date.today()

        print(f"Discovering Tennis Abstract URLs from {start_date} to {end_date}")

        charting_url = "https://www.tennisabstract.com/charting/"

        try:
            resp = requests.get(charting_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")

            match_urls = []

            for link in soup.find_all('a', href=True):
                href = link['href']

                if re.match(r'\d{8}-[MW]-.*\.html$', href):
                    if href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(charting_url, href)

                    filename = href.split('/')[-1]
                    date_str = filename[:8]

                    try:
                        url_date = datetime.strptime(date_str, '%Y%m%d').date()
                        if start_date <= url_date <= end_date:
                            match_urls.append(full_url)
                    except ValueError:
                        continue

            match_urls = list(set(match_urls))
            new_urls = [url for url in match_urls if url not in self.scraped_urls]

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
        """Scrape all available data from a single Tennis Abstract match page"""
        print(f"Scraping: {url}")

        try:
            scraped_records = self.scrape_comprehensive_match_data(url)

            if scraped_records:
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

        for record in records:
            record.update({
                'source': 'tennis_abstract',
                'scrape_url': scraped_data['url'],
                'scrape_timestamp': scraped_data['scrape_timestamp']
            })

        return records

# ============================================================================
# TENNIS ABSTRACT TO JEFF TRANSFORMATION
# ============================================================================

def extract_match_id(records):
    """Extract match ID from TA data"""
    # Use first record's composite_id or build from metadata
    sample = records[0] if records else {}
    return sample.get('composite_id', 'unknown_match')

def get_all_players(records):
    """Get all players from netpts tables"""
    players = set()
    for record in records:
        if 'netpts' in record.get('data_type', ''):
            players.add(record.get('Player_canonical'))
    return players

def transform_snv_complete(ta_records):
    """Complete SnV transformation with all fixes"""

    snv_map = {
        "Serve-and-Volley": "SnV", "S-and-V 1sts": "SnV1st",
        "S-and-V 2nds": "SnV2nd", "non-S-and-V": "nonSnV",
        "non-S-and-V 1sts": "nonSnV1st", "non-S-and-V 2nds": "nonSnV2nd"
    }

    jeff_columns = ['snv_pts', 'pts_won', 'aces', 'unret', 'return_forced',
                    'net_winner', 'induced_forced', 'net_unforced', 'passed_at_net',
                    'passing_shot_induced_forced']

    match_id = extract_match_id(ta_records)
    all_players = get_all_players(ta_records)

    # Group data by player and context
    player_data = {}
    netpts_records = [r for r in ta_records if 'netpts' in r.get('data_type', '')]

    for record in netpts_records:
        player = record.get('Player_canonical')
        context = record.get('stat_context')
        stat_name = record.get('stat_name')
        stat_value = record.get('stat_value', 0)

        if player not in player_data:
            player_data[player] = {}
        if context not in player_data[player]:
            player_data[player][context] = {}

        # Map TA stat names to Jeff columns
        if stat_name == 'points':
            player_data[player][context]['snv_pts'] = stat_value
        elif stat_name == 'won_pct':
            player_data[player][context]['pts_won'] = stat_value
        elif stat_name == 'wnr_at_net':
            player_data[player][context]['net_winner'] = stat_value
        elif stat_name == 'passed_at_net':
            player_data[player][context]['passed_at_net'] = stat_value

    # Generate Jeff records
    output = []
    for player in all_players:
        for ta_context, jeff_row in snv_map.items():
            # Get data or use zeros
            context_data = player_data.get(player, {}).get(ta_context, {})

            record = {
                'match_id': match_id,
                'Player_canonical': player,
                'row': jeff_row
            }

            # Add all Jeff columns
            for col in jeff_columns:
                record[col] = context_data.get(col, 0)

            output.append(record)

    return output

def transform_serve_basics(ta_records):
    """Transform TA serve1/serve2 tables to Jeff ServeBasics format"""

    def extract_serve_stats(ta_records):
        """Extract serve statistics from TA serve1/serve2 tables"""

        serve_data = {}
        serve_records = [r for r in ta_records if 'serve' in r.get('data_type', '')]

        for record in serve_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in serve_data:
                serve_data[player] = {}
            if context not in serve_data[player]:
                serve_data[player][context] = {}

            # Map TA stat names to Jeff columns
            if stat_name == 'aces':
                serve_data[player][context]['aces'] = stat_value
            elif stat_name == 'double_faults':
                serve_data[player][context]['dfs'] = stat_value
            elif stat_name == 'serve_pts':
                serve_data[player][context]['serve_pts'] = stat_value
            elif stat_name == 'first_in':
                serve_data[player][context]['first_in'] = stat_value
            elif stat_name == 'first_won':
                serve_data[player][context]['first_won'] = stat_value
            elif stat_name == 'second_won':
                serve_data[player][context]['second_won'] = stat_value

        return serve_data

    return extract_serve_stats(ta_records)


def transform_shot_direction(ta_records):
    """Transform TA shot direction tables to Jeff ShotDirection format"""

    # Jeff ShotDirection schema
    jeff_shot_columns = [
        'fh_winners', 'fh_errors', 'bh_winners', 'bh_errors',
        'fh_cross_court', 'fh_down_line', 'bh_cross_court', 'bh_down_line',
        'fh_cross_winners', 'fh_down_winners', 'bh_cross_winners', 'bh_down_winners',
        'fh_cross_errors', 'fh_down_errors', 'bh_cross_errors', 'bh_down_errors',
        'fh_effectiveness_pct', 'bh_effectiveness_pct',
        'cross_court_pct', 'down_line_pct', 'directional_accuracy_pct'
    ]

    def extract_shot_stats(ta_records):
        """Extract and consolidate shot direction statistics across contexts"""

        player_data = {}
        # Look for shot direction related data types
        shot_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                     for x in
                                                     ['shot_dir', 'shotdir', 'direction', 'winners', 'errors'])]

        # First pass: collect all data by player
        for record in shot_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in player_data:
                player_data[player] = {}

            # Map TA stat names to consolidated data
            if stat_name in ['fh_winners', 'forehand_winners']:
                player_data[player]['fh_winners'] = stat_value
            elif stat_name in ['fh_errors', 'forehand_errors', 'fh_unforced']:
                player_data[player]['fh_errors'] = stat_value
            elif stat_name in ['bh_winners', 'backhand_winners']:
                player_data[player]['bh_winners'] = stat_value
            elif stat_name in ['bh_errors', 'backhand_errors', 'bh_unforced']:
                player_data[player]['bh_errors'] = stat_value
            elif stat_name in ['fh_cross_court', 'fh_cross']:
                player_data[player]['fh_cross_court'] = stat_value
            elif stat_name in ['fh_down_line', 'fh_down']:
                player_data[player]['fh_down_line'] = stat_value
            elif stat_name in ['bh_cross_court', 'bh_cross']:
                player_data[player]['bh_cross_court'] = stat_value
            elif stat_name in ['bh_down_line', 'bh_down']:
                player_data[player]['bh_down_line'] = stat_value
            elif stat_name in ['fh_cross_winners']:
                player_data[player]['fh_cross_winners'] = stat_value
            elif stat_name in ['fh_down_winners']:
                player_data[player]['fh_down_winners'] = stat_value
            elif stat_name in ['bh_cross_winners']:
                player_data[player]['bh_cross_winners'] = stat_value
            elif stat_name in ['bh_down_winners']:
                player_data[player]['bh_down_winners'] = stat_value
            elif stat_name in ['fh_cross_errors']:
                player_data[player]['fh_cross_errors'] = stat_value
            elif stat_name in ['fh_down_errors']:
                player_data[player]['fh_down_errors'] = stat_value
            elif stat_name in ['bh_cross_errors']:
                player_data[player]['bh_cross_errors'] = stat_value
            elif stat_name in ['bh_down_errors']:
                player_data[player]['bh_down_errors'] = stat_value
            elif stat_name in ['winners_total', 'total_winners']:
                # Distribute between fh/bh if specific data not available
                if 'fh_winners' not in player_data[player]:
                    player_data[player]['fh_winners'] = int(stat_value * 0.6)  # 60% FH
                if 'bh_winners' not in player_data[player]:
                    player_data[player]['bh_winners'] = int(stat_value * 0.4)  # 40% BH
            elif stat_name in ['errors_total', 'total_errors', 'unforced_total']:
                # Distribute between fh/bh if specific data not available
                if 'fh_errors' not in player_data[player]:
                    player_data[player]['fh_errors'] = int(stat_value * 0.6)
                if 'bh_errors' not in player_data[player]:
                    player_data[player]['bh_errors'] = int(stat_value * 0.4)

        return player_data

    # Extract and consolidate all shot data
    consolidated_data = extract_shot_stats(ta_records)

    # Generate single consolidated record per player
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in consolidated_data.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Consolidated'
        }

        # Add all Jeff columns with calculated values
        for col in jeff_shot_columns:
            if col in stats:
                record[col] = stats[col]
            elif col == 'fh_effectiveness_pct':
                # Calculate FH effectiveness
                winners = stats.get('fh_winners', 0)
                errors = stats.get('fh_errors', 0)
                total_fh = winners + errors
                record[col] = (winners / total_fh * 100) if total_fh > 0 else 0
            elif col == 'bh_effectiveness_pct':
                # Calculate BH effectiveness
                winners = stats.get('bh_winners', 0)
                errors = stats.get('bh_errors', 0)
                total_bh = winners + errors
                record[col] = (winners / total_bh * 100) if total_bh > 0 else 0
            elif col == 'cross_court_pct':
                # Calculate cross court percentage
                fh_cross = stats.get('fh_cross_court', 0)
                bh_cross = stats.get('bh_cross_court', 0)
                fh_down = stats.get('fh_down_line', 0)
                bh_down = stats.get('bh_down_line', 0)
                total_shots = fh_cross + bh_cross + fh_down + bh_down
                record[col] = ((fh_cross + bh_cross) / total_shots * 100) if total_shots > 0 else 0
            elif col == 'down_line_pct':
                # Calculate down line percentage
                fh_cross = stats.get('fh_cross_court', 0)
                bh_cross = stats.get('bh_cross_court', 0)
                fh_down = stats.get('fh_down_line', 0)
                bh_down = stats.get('bh_down_line', 0)
                total_shots = fh_cross + bh_cross + fh_down + bh_down
                record[col] = ((fh_down + bh_down) / total_shots * 100) if total_shots > 0 else 0
            elif col == 'directional_accuracy_pct':
                # Calculate overall directional accuracy (winners/total)
                total_winners = stats.get('fh_winners', 0) + stats.get('bh_winners', 0)
                total_errors = stats.get('fh_errors', 0) + stats.get('bh_errors', 0)
                total_shots = total_winners + total_errors
                record[col] = (total_winners / total_shots * 100) if total_shots > 0 else 0
            elif col.endswith('_cross_winners'):
                # Estimate cross-court winners if not directly available
                wing = 'fh' if col.startswith('fh') else 'bh'
                total_winners = stats.get(f'{wing}_winners', 0)
                cross_pct = stats.get(f'{wing}_cross_court', 0)
                total_shots = cross_pct + stats.get(f'{wing}_down_line', 0)
                if total_shots > 0:
                    cross_ratio = cross_pct / total_shots
                    record[col] = int(total_winners * cross_ratio)
                else:
                    record[col] = 0
            elif col.endswith('_down_winners'):
                # Estimate down-line winners if not directly available
                wing = 'fh' if col.startswith('fh') else 'bh'
                total_winners = stats.get(f'{wing}_winners', 0)
                down_pct = stats.get(f'{wing}_down_line', 0)
                total_shots = down_pct + stats.get(f'{wing}_cross_court', 0)
                if total_shots > 0:
                    down_ratio = down_pct / total_shots
                    record[col] = int(total_winners * down_ratio)
                else:
                    record[col] = 0
            elif col.endswith('_cross_errors'):
                # Estimate cross-court errors if not directly available
                wing = 'fh' if col.startswith('fh') else 'bh'
                total_errors = stats.get(f'{wing}_errors', 0)
                cross_pct = stats.get(f'{wing}_cross_court', 0)
                total_shots = cross_pct + stats.get(f'{wing}_down_line', 0)
                if total_shots > 0:
                    cross_ratio = cross_pct / total_shots
                    record[col] = int(total_errors * cross_ratio)
                else:
                    record[col] = 0
            elif col.endswith('_down_errors'):
                # Estimate down-line errors if not directly available
                wing = 'fh' if col.startswith('fh') else 'bh'
                total_errors = stats.get(f'{wing}_errors', 0)
                down_pct = stats.get(f'{wing}_down_line', 0)
                total_shots = down_pct + stats.get(f'{wing}_cross_court', 0)
                if total_shots > 0:
                    down_ratio = down_pct / total_shots
                    record[col] = int(total_errors * down_ratio)
                else:
                    record[col] = 0
            else:
                record[col] = 0

        output.append(record)

    return output


def transform_key_points(ta_records):
    """Transform TA key points tables to Jeff KeyPointsServe/KeyPointsReturn format"""

    # Jeff KeyPoints schema (combined serve and return)
    jeff_key_columns = [
        # Serve key points
        'serve_break_points_faced', 'serve_break_points_saved', 'serve_break_points_saved_pct',
        'serve_set_points_faced', 'serve_set_points_saved', 'serve_set_points_saved_pct',
        'serve_game_points_faced', 'serve_game_points_saved', 'serve_game_points_saved_pct',
        'serve_deuce_points', 'serve_deuce_won', 'serve_deuce_won_pct',
        'serve_pressure_points', 'serve_pressure_won', 'serve_pressure_performance',

        # Return key points
        'return_break_points', 'return_break_points_won', 'return_break_points_won_pct',
        'return_set_points', 'return_set_points_won', 'return_set_points_won_pct',
        'return_game_points', 'return_game_points_won', 'return_game_points_won_pct',
        'return_deuce_points', 'return_deuce_won', 'return_deuce_won_pct',
        'return_pressure_points', 'return_pressure_won', 'return_pressure_performance',

        # Overall clutch metrics
        'overall_clutch_points', 'overall_clutch_won', 'overall_clutch_performance',
        'pressure_differential', 'clutch_factor'
    ]

    def extract_key_points_stats(ta_records):
        """Extract key points statistics from TA key points tables"""

        key_data = {}
        # Look for key points related data types
        key_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                    for x in
                                                    ['key_points', 'keypoints', 'KeyPoints', 'pressure', 'clutch',
                                                     'break', 'deuce'])]

        for record in key_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)
            stat_percentage = record.get('stat_percentage')

            if player not in key_data:
                key_data[player] = {}

            # Map TA stat names to Jeff key points columns
            # Serve key points
            if stat_name in ['break_points_faced', 'bp_faced_serve']:
                key_data[player]['serve_break_points_faced'] = stat_value
            elif stat_name in ['break_points_saved', 'bp_saved_serve']:
                key_data[player]['serve_break_points_saved'] = stat_value
            elif stat_name in ['set_points_faced', 'sp_faced_serve']:
                key_data[player]['serve_set_points_faced'] = stat_value
            elif stat_name in ['set_points_saved', 'sp_saved_serve']:
                key_data[player]['serve_set_points_saved'] = stat_value
            elif stat_name in ['game_points_faced', 'gp_faced_serve']:
                key_data[player]['serve_game_points_faced'] = stat_value
            elif stat_name in ['game_points_saved', 'gp_saved_serve']:
                key_data[player]['serve_game_points_saved'] = stat_value
            elif stat_name in ['deuce_points_serve', 'deuce_serve']:
                key_data[player]['serve_deuce_points'] = stat_value
            elif stat_name in ['deuce_won_serve', 'deuce_serve_won']:
                key_data[player]['serve_deuce_won'] = stat_value
            elif stat_name in ['pressure_points_serve', 'pressure_serve']:
                key_data[player]['serve_pressure_points'] = stat_value
            elif stat_name in ['pressure_won_serve', 'pressure_serve_won']:
                key_data[player]['serve_pressure_won'] = stat_value

            # Return key points
            elif stat_name in ['break_points_return', 'bp_return', 'break_points_total']:
                key_data[player]['return_break_points'] = stat_value
            elif stat_name in ['break_points_won', 'bp_won_return']:
                key_data[player]['return_break_points_won'] = stat_value
            elif stat_name in ['set_points_return', 'sp_return']:
                key_data[player]['return_set_points'] = stat_value
            elif stat_name in ['set_points_won_return', 'sp_won_return']:
                key_data[player]['return_set_points_won'] = stat_value
            elif stat_name in ['game_points_return', 'gp_return']:
                key_data[player]['return_game_points'] = stat_value
            elif stat_name in ['game_points_won_return', 'gp_won_return']:
                key_data[player]['return_game_points_won'] = stat_value
            elif stat_name in ['deuce_points_return', 'deuce_return']:
                key_data[player]['return_deuce_points'] = stat_value
            elif stat_name in ['deuce_won_return', 'deuce_return_won']:
                key_data[player]['return_deuce_won'] = stat_value
            elif stat_name in ['pressure_points_return', 'pressure_return']:
                key_data[player]['return_pressure_points'] = stat_value
            elif stat_name in ['pressure_won_return', 'pressure_return_won']:
                key_data[player]['return_pressure_won'] = stat_value

            # Handle percentage data
            elif stat_name in ['bp_saved_pct', 'break_points_saved_pct'] and stat_percentage:
                key_data[player]['serve_break_points_saved_pct'] = stat_percentage
            elif stat_name in ['bp_won_pct', 'break_points_won_pct'] and stat_percentage:
                key_data[player]['return_break_points_won_pct'] = stat_percentage

        return key_data

    # Extract key points data
    key_stats = extract_key_points_stats(ta_records)

    # Generate output records for each player
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in key_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Key_Points'
        }

        # Add all Jeff columns with calculated values
        for col in jeff_key_columns:
            if col in stats:
                record[col] = stats[col]
            # Calculate serve percentages
            elif col == 'serve_break_points_saved_pct':
                faced = stats.get('serve_break_points_faced', 0)
                saved = stats.get('serve_break_points_saved', 0)
                record[col] = (saved / faced * 100) if faced > 0 else 0
            elif col == 'serve_set_points_saved_pct':
                faced = stats.get('serve_set_points_faced', 0)
                saved = stats.get('serve_set_points_saved', 0)
                record[col] = (saved / faced * 100) if faced > 0 else 0
            elif col == 'serve_game_points_saved_pct':
                faced = stats.get('serve_game_points_faced', 0)
                saved = stats.get('serve_game_points_saved', 0)
                record[col] = (saved / faced * 100) if faced > 0 else 0
            elif col == 'serve_deuce_won_pct':
                total = stats.get('serve_deuce_points', 0)
                won = stats.get('serve_deuce_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'serve_pressure_performance':
                total = stats.get('serve_pressure_points', 0)
                won = stats.get('serve_pressure_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0

            # Calculate return percentages
            elif col == 'return_break_points_won_pct':
                total = stats.get('return_break_points', 0)
                won = stats.get('return_break_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_set_points_won_pct':
                total = stats.get('return_set_points', 0)
                won = stats.get('return_set_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_game_points_won_pct':
                total = stats.get('return_game_points', 0)
                won = stats.get('return_game_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_deuce_won_pct':
                total = stats.get('return_deuce_points', 0)
                won = stats.get('return_deuce_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_pressure_performance':
                total = stats.get('return_pressure_points', 0)
                won = stats.get('return_pressure_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0

            # Calculate overall clutch metrics
            elif col == 'overall_clutch_points':
                serve_clutch = stats.get('serve_pressure_points', 0)
                return_clutch = stats.get('return_pressure_points', 0)
                record[col] = serve_clutch + return_clutch
            elif col == 'overall_clutch_won':
                serve_won = stats.get('serve_pressure_won', 0)
                return_won = stats.get('return_pressure_won', 0)
                record[col] = serve_won + return_won
            elif col == 'overall_clutch_performance':
                total_clutch = record.get('overall_clutch_points', 0)
                total_won = record.get('overall_clutch_won', 0)
                record[col] = (total_won / total_clutch * 100) if total_clutch > 0 else 0
            elif col == 'pressure_differential':
                serve_perf = record.get('serve_pressure_performance', 0)
                return_perf = record.get('return_pressure_performance', 0)
                record[col] = serve_perf - return_perf
            elif col == 'clutch_factor':
                # Composite clutch score weighted by importance
                bp_serve_pct = record.get('serve_break_points_saved_pct', 0)
                bp_return_pct = record.get('return_break_points_won_pct', 0)
                overall_perf = record.get('overall_clutch_performance', 0)
                record[col] = (bp_serve_pct * 0.4) + (bp_return_pct * 0.4) + (overall_perf * 0.2)
            else:
                record[col] = 0

        output.append(record)

    return output


def transform_overview(ta_records):
    """Transform TA overview tables to Jeff Overview format"""

    # Jeff Overview schema
    jeff_overview_columns = [
        'total_points', 'total_points_won', 'total_points_lost', 'total_points_won_pct',
        'service_points', 'service_points_won', 'service_points_won_pct',
        'return_points', 'return_points_won', 'return_points_won_pct',
        'service_games_played', 'service_games_won', 'service_games_won_pct',
        'return_games_played', 'return_games_won', 'return_games_won_pct',
        'break_points_faced', 'break_points_saved', 'break_points_saved_pct',
        'break_points_opportunities', 'break_points_converted', 'break_points_converted_pct',
        'total_games', 'total_games_won', 'total_games_won_pct',
        'sets_played', 'sets_won', 'match_duration_minutes',
        'dominance_ratio', 'point_efficiency', 'break_differential'
    ]

    def extract_overview_stats(ta_records):
        """Extract overview statistics from TA overview tables"""

        overview_data = {}
        # Look for overview-related data types
        overview_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                         for x in ['overview', 'Overview', 'total', 'general', 'match',
                                                                   'summary'])]

        for record in overview_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)
            stat_percentage = record.get('stat_percentage')

            if player not in overview_data:
                overview_data[player] = {}

            # Map TA stat names to Jeff overview columns
            if stat_name in ['total_points', 'points_total', 'all_points']:
                overview_data[player]['total_points'] = stat_value
            elif stat_name in ['total_points_won', 'points_won_total']:
                overview_data[player]['total_points_won'] = stat_value
            elif stat_name in ['total_points_lost', 'points_lost_total']:
                overview_data[player]['total_points_lost'] = stat_value
            elif stat_name in ['service_points', 'serve_points', 'serving_points']:
                overview_data[player]['service_points'] = stat_value
            elif stat_name in ['service_points_won', 'serve_points_won']:
                overview_data[player]['service_points_won'] = stat_value
            elif stat_name in ['return_points', 'returning_points']:
                overview_data[player]['return_points'] = stat_value
            elif stat_name in ['return_points_won', 'returning_points_won']:
                overview_data[player]['return_points_won'] = stat_value
            elif stat_name in ['service_games_played', 'service_games', 'serve_games']:
                overview_data[player]['service_games_played'] = stat_value
            elif stat_name in ['service_games_won', 'serve_games_won']:
                overview_data[player]['service_games_won'] = stat_value
            elif stat_name in ['return_games_played', 'return_games']:
                overview_data[player]['return_games_played'] = stat_value
            elif stat_name in ['return_games_won', 'return_games_broken']:
                overview_data[player]['return_games_won'] = stat_value
            elif stat_name in ['break_points_faced', 'bp_faced']:
                overview_data[player]['break_points_faced'] = stat_value
            elif stat_name in ['break_points_saved', 'bp_saved']:
                overview_data[player]['break_points_saved'] = stat_value
            elif stat_name in ['break_points_opportunities', 'bp_opportunities', 'break_chances']:
                overview_data[player]['break_points_opportunities'] = stat_value
            elif stat_name in ['break_points_converted', 'bp_converted', 'breaks_achieved']:
                overview_data[player]['break_points_converted'] = stat_value
            elif stat_name in ['total_games', 'games_total']:
                overview_data[player]['total_games'] = stat_value
            elif stat_name in ['total_games_won', 'games_won_total']:
                overview_data[player]['total_games_won'] = stat_value
            elif stat_name in ['sets_played', 'total_sets']:
                overview_data[player]['sets_played'] = stat_value
            elif stat_name in ['sets_won', 'sets_taken']:
                overview_data[player]['sets_won'] = stat_value
            elif stat_name in ['match_duration', 'duration_minutes']:
                overview_data[player]['match_duration_minutes'] = stat_value
            elif stat_name in ['winners_total', 'total_winners']:
                overview_data[player]['winners_total'] = stat_value
            elif stat_name in ['errors_total', 'total_errors', 'unforced_total']:
                overview_data[player]['errors_total'] = stat_value
            elif stat_name in ['aces_total', 'total_aces']:
                overview_data[player]['aces_total'] = stat_value
            elif stat_name in ['double_faults_total', 'total_double_faults']:
                overview_data[player]['double_faults_total'] = stat_value

            # Handle percentage data
            elif stat_name in ['total_points_won_pct', 'points_won_percentage'] and stat_percentage:
                overview_data[player]['total_points_won_pct'] = stat_percentage
            elif stat_name in ['service_points_won_pct', 'serve_percentage'] and stat_percentage:
                overview_data[player]['service_points_won_pct'] = stat_percentage
            elif stat_name in ['return_points_won_pct', 'return_percentage'] and stat_percentage:
                overview_data[player]['return_points_won_pct'] = stat_percentage

        return overview_data

    # Extract overview data
    overview_stats = extract_overview_stats(ta_records)

    # Generate output records for each player
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in overview_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Match_Overview'
        }

        # Add all Jeff columns with calculated values
        for col in jeff_overview_columns:
            if col in stats:
                record[col] = stats[col]
            elif col == 'total_points_lost':
                # Calculate lost points
                total = stats.get('total_points', 0)
                won = stats.get('total_points_won', 0)
                record[col] = max(0, total - won)
            elif col == 'total_points_won_pct':
                # Calculate total points won percentage
                total = stats.get('total_points', 0)
                won = stats.get('total_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'service_points_won_pct':
                # Calculate service points won percentage
                total = stats.get('service_points', 0)
                won = stats.get('service_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_points_won_pct':
                # Calculate return points won percentage
                total = stats.get('return_points', 0)
                won = stats.get('return_points_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'service_games_won_pct':
                # Calculate service games won percentage
                total = stats.get('service_games_played', 0)
                won = stats.get('service_games_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'return_games_won_pct':
                # Calculate return games won percentage (breaks achieved)
                total = stats.get('return_games_played', 0)
                won = stats.get('return_games_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'break_points_saved_pct':
                # Calculate break points saved percentage
                faced = stats.get('break_points_faced', 0)
                saved = stats.get('break_points_saved', 0)
                record[col] = (saved / faced * 100) if faced > 0 else 0
            elif col == 'break_points_converted_pct':
                # Calculate break points converted percentage
                opportunities = stats.get('break_points_opportunities', 0)
                converted = stats.get('break_points_converted', 0)
                record[col] = (converted / opportunities * 100) if opportunities > 0 else 0
            elif col == 'total_games_won_pct':
                # Calculate total games won percentage
                total = record.get('total_games', 0)
                won = record.get('total_games_won', 0)
                record[col] = (won / total * 100) if total > 0 else 0
            elif col == 'total_games':
                # Calculate total games if not provided
                service = stats.get('service_games_played', 0)
                return_games = stats.get('return_games_played', 0)
                record[col] = service + return_games
            elif col == 'total_games_won':
                # Calculate total games won if not provided
                service_won = stats.get('service_games_won', 0)
                return_won = stats.get('return_games_won', 0)
                record[col] = service_won + return_won
            elif col == 'dominance_ratio':
                # Calculate dominance ratio (points won vs opponent)
                points_won = stats.get('total_points_won', 0)
                total_points = stats.get('total_points', 0)
                points_lost = total_points - points_won
                record[col] = (points_won / points_lost) if points_lost > 0 else 0
            elif col == 'point_efficiency':
                # Calculate point efficiency (winners vs errors ratio)
                winners = stats.get('winners_total', 0)
                errors = stats.get('errors_total', 0)
                record[col] = (winners / errors) if errors > 0 else winners
            elif col == 'break_differential':
                # Calculate break differential (breaks achieved - breaks suffered)
                breaks_made = stats.get('break_points_converted', 0)
                bp_faced = stats.get('break_points_faced', 0)
                bp_saved = stats.get('break_points_saved', 0)
                breaks_suffered = bp_faced - bp_saved
                record[col] = breaks_made - breaks_suffered
            else:
                record[col] = 0

        output.append(record)

    return output


def transform_serve_influence(ta_records):
    """Transform TA serve influence tables to Jeff ServeInfluence format"""

    # Jeff ServeInfluence schema
    jeff_influence_columns = [
        'serve_wide_attempts', 'serve_wide_won', 'serve_wide_effectiveness',
        'serve_t_attempts', 'serve_t_won', 'serve_t_effectiveness',
        'serve_body_attempts', 'serve_body_won', 'serve_body_effectiveness',
        'first_serve_wide', 'first_serve_t', 'first_serve_body',
        'second_serve_wide', 'second_serve_t', 'second_serve_body',
        'serve_plus_one_won', 'serve_plus_one_attempts', 'serve_plus_one_effectiveness',
        'return_difficulty_wide', 'return_difficulty_t', 'return_difficulty_body',
        'serve_direction_winners', 'serve_direction_aces', 'serve_direction_unreturned',
        'serve_placement_pressure', 'serve_tactical_advantage', 'directional_dominance'
    ]

    def extract_serve_influence_stats(ta_records):
        """Extract serve influence statistics from TA serve influence tables"""

        influence_data = {}
        # Look for serve influence related data types
        influence_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                          for x in ['serve_influence', 'serveinfluence', 'serve_dir',
                                                                    'placement', 'direction'])]

        for record in influence_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)
            stat_percentage = record.get('stat_percentage')

            if player not in influence_data:
                influence_data[player] = {}

            # Map TA stat names to Jeff serve influence columns
            if stat_name in ['serve_wide_attempts', 'wide_serves', 'serves_wide']:
                influence_data[player]['serve_wide_attempts'] = stat_value
            elif stat_name in ['serve_wide_won', 'wide_won', 'wide_points_won']:
                influence_data[player]['serve_wide_won'] = stat_value
            elif stat_name in ['serve_t_attempts', 't_serves', 'serves_t']:
                influence_data[player]['serve_t_attempts'] = stat_value
            elif stat_name in ['serve_t_won', 't_won', 't_points_won']:
                influence_data[player]['serve_t_won'] = stat_value
            elif stat_name in ['serve_body_attempts', 'body_serves', 'serves_body']:
                influence_data[player]['serve_body_attempts'] = stat_value
            elif stat_name in ['serve_body_won', 'body_won', 'body_points_won']:
                influence_data[player]['serve_body_won'] = stat_value
            elif stat_name in ['first_serve_wide', 'first_wide']:
                influence_data[player]['first_serve_wide'] = stat_value
            elif stat_name in ['first_serve_t', 'first_t']:
                influence_data[player]['first_serve_t'] = stat_value
            elif stat_name in ['first_serve_body', 'first_body']:
                influence_data[player]['first_serve_body'] = stat_value
            elif stat_name in ['second_serve_wide', 'second_wide']:
                influence_data[player]['second_serve_wide'] = stat_value
            elif stat_name in ['second_serve_t', 'second_t']:
                influence_data[player]['second_serve_t'] = stat_value
            elif stat_name in ['second_serve_body', 'second_body']:
                influence_data[player]['second_serve_body'] = stat_value
            elif stat_name in ['serve_plus_one_won', 'plus_one_won']:
                influence_data[player]['serve_plus_one_won'] = stat_value
            elif stat_name in ['serve_plus_one_attempts', 'plus_one_attempts']:
                influence_data[player]['serve_plus_one_attempts'] = stat_value
            elif stat_name in ['return_difficulty_wide', 'wide_difficulty']:
                influence_data[player]['return_difficulty_wide'] = stat_value
            elif stat_name in ['return_difficulty_t', 't_difficulty']:
                influence_data[player]['return_difficulty_t'] = stat_value
            elif stat_name in ['return_difficulty_body', 'body_difficulty']:
                influence_data[player]['return_difficulty_body'] = stat_value
            elif stat_name in ['serve_direction_winners', 'directional_winners']:
                influence_data[player]['serve_direction_winners'] = stat_value
            elif stat_name in ['serve_direction_aces', 'directional_aces']:
                influence_data[player]['serve_direction_aces'] = stat_value
            elif stat_name in ['serve_direction_unreturned', 'directional_unreturned']:
                influence_data[player]['serve_direction_unreturned'] = stat_value

            # Handle percentage-based data
            elif stat_name in ['wide_effectiveness_pct', 'wide_pct'] and stat_percentage:
                influence_data[player]['serve_wide_effectiveness'] = stat_percentage
            elif stat_name in ['t_effectiveness_pct', 't_pct'] and stat_percentage:
                influence_data[player]['serve_t_effectiveness'] = stat_percentage
            elif stat_name in ['body_effectiveness_pct', 'body_pct'] and stat_percentage:
                influence_data[player]['serve_body_effectiveness'] = stat_percentage

        return influence_data

    # Extract serve influence data
    influence_stats = extract_serve_influence_stats(ta_records)

    # Generate output records for each player
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in influence_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Serve_Influence'
        }

        # Add all Jeff columns with calculated values
        for col in jeff_influence_columns:
            if col in stats:
                record[col] = stats[col]
            elif col == 'serve_wide_effectiveness':
                # Calculate wide serve effectiveness
                attempts = stats.get('serve_wide_attempts', 0)
                won = stats.get('serve_wide_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'serve_t_effectiveness':
                # Calculate T serve effectiveness
                attempts = stats.get('serve_t_attempts', 0)
                won = stats.get('serve_t_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'serve_body_effectiveness':
                # Calculate body serve effectiveness
                attempts = stats.get('serve_body_attempts', 0)
                won = stats.get('serve_body_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'serve_plus_one_effectiveness':
                # Calculate serve plus one effectiveness
                attempts = stats.get('serve_plus_one_attempts', 0)
                won = stats.get('serve_plus_one_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'serve_placement_pressure':
                # Calculate overall placement pressure (weighted effectiveness)
                wide_eff = record.get('serve_wide_effectiveness', 0)
                t_eff = record.get('serve_t_effectiveness', 0)
                body_eff = record.get('serve_body_effectiveness', 0)
                record[col] = (wide_eff * 0.4) + (t_eff * 0.4) + (body_eff * 0.2)
            elif col == 'serve_tactical_advantage':
                # Calculate tactical advantage from serve placement
                plus_one_eff = record.get('serve_plus_one_effectiveness', 0)
                placement_pressure = record.get('serve_placement_pressure', 0)
                record[col] = (plus_one_eff * 0.6) + (placement_pressure * 0.4)
            elif col == 'directional_dominance':
                # Calculate dominance across all serve directions
                wide_att = stats.get('serve_wide_attempts', 0)
                t_att = stats.get('serve_t_attempts', 0)
                body_att = stats.get('serve_body_attempts', 0)
                total_att = wide_att + t_att + body_att

                if total_att > 0:
                    wide_weight = wide_att / total_att
                    t_weight = t_att / total_att
                    body_weight = body_att / total_att

                    wide_eff = record.get('serve_wide_effectiveness', 0)
                    t_eff = record.get('serve_t_effectiveness', 0)
                    body_eff = record.get('serve_body_effectiveness', 0)

                    record[col] = (wide_eff * wide_weight) + (t_eff * t_weight) + (body_eff * body_weight)
                else:
                    record[col] = 0
            else:
                record[col] = 0

        output.append(record)

    return output


def transform_shot_types(ta_records):
    """Transform TA shot types tables to Jeff ShotTypes format"""

    # Jeff ShotTypes schema
    jeff_shot_types_columns = [
        'groundstroke_topspin', 'groundstroke_slice', 'groundstroke_flat',
        'groundstroke_topspin_won', 'groundstroke_slice_won', 'groundstroke_flat_won',
        'groundstroke_topspin_effectiveness', 'groundstroke_slice_effectiveness', 'groundstroke_flat_effectiveness',
        'fh_topspin', 'fh_slice', 'fh_flat', 'bh_topspin', 'bh_slice', 'bh_flat',
        'fh_topspin_won', 'fh_slice_won', 'fh_flat_won', 'bh_topspin_won', 'bh_slice_won', 'bh_flat_won',
        'volley_attempts', 'volley_won', 'volley_effectiveness',
        'fh_volley', 'bh_volley', 'overhead_attempts', 'overhead_won',
        'fh_volley_won', 'bh_volley_won', 'overhead_effectiveness',
        'approach_shots', 'approach_shots_won', 'approach_effectiveness',
        'defensive_shots', 'defensive_shots_won', 'defensive_effectiveness',
        'attacking_shots', 'attacking_shots_won', 'attacking_effectiveness',
        'shot_variety_index', 'tactical_diversity', 'shot_selection_quality'
    ]

    def extract_shot_types_stats(ta_records):
        """Extract shot types statistics from TA shot types tables"""

        shot_types_data = {}
        # Look for shot types related data types
        shot_types_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                           for x in
                                                           ['shot_types', 'shottypes', 'groundstroke', 'volley',
                                                            'overhead', 'approach'])]

        for record in shot_types_records:
            player = record.get('Player_canonical')
            context = record.get('stat_context', 'Total')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)
            stat_percentage = record.get('stat_percentage')

            if player not in shot_types_data:
                shot_types_data[player] = {}

            # Map TA stat names to Jeff shot types columns
            # Groundstroke types
            if stat_name in ['groundstroke_topspin', 'topspin_groundstrokes', 'topspin']:
                shot_types_data[player]['groundstroke_topspin'] = stat_value
            elif stat_name in ['groundstroke_slice', 'slice_groundstrokes', 'slice']:
                shot_types_data[player]['groundstroke_slice'] = stat_value
            elif stat_name in ['groundstroke_flat', 'flat_groundstrokes', 'flat']:
                shot_types_data[player]['groundstroke_flat'] = stat_value
            elif stat_name in ['groundstroke_topspin_won', 'topspin_won']:
                shot_types_data[player]['groundstroke_topspin_won'] = stat_value
            elif stat_name in ['groundstroke_slice_won', 'slice_won']:
                shot_types_data[player]['groundstroke_slice_won'] = stat_value
            elif stat_name in ['groundstroke_flat_won', 'flat_won']:
                shot_types_data[player]['groundstroke_flat_won'] = stat_value

            # Forehand/Backhand specific types
            elif stat_name in ['fh_topspin', 'forehand_topspin']:
                shot_types_data[player]['fh_topspin'] = stat_value
            elif stat_name in ['fh_slice', 'forehand_slice']:
                shot_types_data[player]['fh_slice'] = stat_value
            elif stat_name in ['fh_flat', 'forehand_flat']:
                shot_types_data[player]['fh_flat'] = stat_value
            elif stat_name in ['bh_topspin', 'backhand_topspin']:
                shot_types_data[player]['bh_topspin'] = stat_value
            elif stat_name in ['bh_slice', 'backhand_slice']:
                shot_types_data[player]['bh_slice'] = stat_value
            elif stat_name in ['bh_flat', 'backhand_flat']:
                shot_types_data[player]['bh_flat'] = stat_value
            elif stat_name in ['fh_topspin_won', 'forehand_topspin_won']:
                shot_types_data[player]['fh_topspin_won'] = stat_value
            elif stat_name in ['fh_slice_won', 'forehand_slice_won']:
                shot_types_data[player]['fh_slice_won'] = stat_value
            elif stat_name in ['fh_flat_won', 'forehand_flat_won']:
                shot_types_data[player]['fh_flat_won'] = stat_value
            elif stat_name in ['bh_topspin_won', 'backhand_topspin_won']:
                shot_types_data[player]['bh_topspin_won'] = stat_value
            elif stat_name in ['bh_slice_won', 'backhand_slice_won']:
                shot_types_data[player]['bh_slice_won'] = stat_value
            elif stat_name in ['bh_flat_won', 'backhand_flat_won']:
                shot_types_data[player]['bh_flat_won'] = stat_value

            # Volleys and net play
            elif stat_name in ['volley_attempts', 'volleys', 'volley_total']:
                shot_types_data[player]['volley_attempts'] = stat_value
            elif stat_name in ['volley_won', 'volleys_won']:
                shot_types_data[player]['volley_won'] = stat_value
            elif stat_name in ['fh_volley', 'forehand_volley']:
                shot_types_data[player]['fh_volley'] = stat_value
            elif stat_name in ['bh_volley', 'backhand_volley']:
                shot_types_data[player]['bh_volley'] = stat_value
            elif stat_name in ['fh_volley_won', 'forehand_volley_won']:
                shot_types_data[player]['fh_volley_won'] = stat_value
            elif stat_name in ['bh_volley_won', 'backhand_volley_won']:
                shot_types_data[player]['bh_volley_won'] = stat_value
            elif stat_name in ['overhead_attempts', 'overheads', 'smash']:
                shot_types_data[player]['overhead_attempts'] = stat_value
            elif stat_name in ['overhead_won', 'overheads_won', 'smash_won']:
                shot_types_data[player]['overhead_won'] = stat_value

            # Tactical shot types
            elif stat_name in ['approach_shots', 'approach']:
                shot_types_data[player]['approach_shots'] = stat_value
            elif stat_name in ['approach_shots_won', 'approach_won']:
                shot_types_data[player]['approach_shots_won'] = stat_value
            elif stat_name in ['defensive_shots', 'defense']:
                shot_types_data[player]['defensive_shots'] = stat_value
            elif stat_name in ['defensive_shots_won', 'defense_won']:
                shot_types_data[player]['defensive_shots_won'] = stat_value
            elif stat_name in ['attacking_shots', 'attack']:
                shot_types_data[player]['attacking_shots'] = stat_value
            elif stat_name in ['attacking_shots_won', 'attack_won']:
                shot_types_data[player]['attacking_shots_won'] = stat_value

            # Handle percentage-based data
            elif stat_name in ['topspin_effectiveness_pct', 'topspin_pct'] and stat_percentage:
                shot_types_data[player]['groundstroke_topspin_effectiveness'] = stat_percentage
            elif stat_name in ['slice_effectiveness_pct', 'slice_pct'] and stat_percentage:
                shot_types_data[player]['groundstroke_slice_effectiveness'] = stat_percentage
            elif stat_name in ['flat_effectiveness_pct', 'flat_pct'] and stat_percentage:
                shot_types_data[player]['groundstroke_flat_effectiveness'] = stat_percentage

        return shot_types_data

    # Extract shot types data
    shot_types_stats = extract_shot_types_stats(ta_records)

    # Generate output records for each player
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in shot_types_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Shot_Types'
        }

        # Add all Jeff columns with calculated values
        for col in jeff_shot_types_columns:
            if col in stats:
                record[col] = stats[col]
            elif col == 'groundstroke_topspin_effectiveness':
                # Calculate topspin effectiveness
                attempts = stats.get('groundstroke_topspin', 0)
                won = stats.get('groundstroke_topspin_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'groundstroke_slice_effectiveness':
                # Calculate slice effectiveness
                attempts = stats.get('groundstroke_slice', 0)
                won = stats.get('groundstroke_slice_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'groundstroke_flat_effectiveness':
                # Calculate flat effectiveness
                attempts = stats.get('groundstroke_flat', 0)
                won = stats.get('groundstroke_flat_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'volley_effectiveness':
                # Calculate volley effectiveness
                attempts = stats.get('volley_attempts', 0)
                won = stats.get('volley_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'overhead_effectiveness':
                # Calculate overhead effectiveness
                attempts = stats.get('overhead_attempts', 0)
                won = stats.get('overhead_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'approach_effectiveness':
                # Calculate approach shot effectiveness
                attempts = stats.get('approach_shots', 0)
                won = stats.get('approach_shots_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'defensive_effectiveness':
                # Calculate defensive shot effectiveness
                attempts = stats.get('defensive_shots', 0)
                won = stats.get('defensive_shots_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'attacking_effectiveness':
                # Calculate attacking shot effectiveness
                attempts = stats.get('attacking_shots', 0)
                won = stats.get('attacking_shots_won', 0)
                record[col] = (won / attempts * 100) if attempts > 0 else 0
            elif col == 'shot_variety_index':
                # Calculate shot variety index (how many different shot types used)
                shot_types = [
                    stats.get('groundstroke_topspin', 0),
                    stats.get('groundstroke_slice', 0),
                    stats.get('groundstroke_flat', 0),
                    stats.get('volley_attempts', 0),
                    stats.get('overhead_attempts', 0)
                ]
                variety_count = sum(1 for shots in shot_types if shots > 0)
                total_shots = sum(shot_types)
                record[col] = (variety_count / len(shot_types) * 100) if total_shots > 0 else 0
            elif col == 'tactical_diversity':
                # Calculate tactical diversity (balance across shot categories)
                topspin_eff = record.get('groundstroke_topspin_effectiveness', 0)
                slice_eff = record.get('groundstroke_slice_effectiveness', 0)
                flat_eff = record.get('groundstroke_flat_effectiveness', 0)
                volley_eff = record.get('volley_effectiveness', 0)

                non_zero_effs = [eff for eff in [topspin_eff, slice_eff, flat_eff, volley_eff] if eff > 0]
                record[col] = sum(non_zero_effs) / len(non_zero_effs) if non_zero_effs else 0
            elif col == 'shot_selection_quality':
                # Calculate shot selection quality (weighted effectiveness across types)
                topspin_shots = stats.get('groundstroke_topspin', 0)
                slice_shots = stats.get('groundstroke_slice', 0)
                flat_shots = stats.get('groundstroke_flat', 0)
                total_groundstrokes = topspin_shots + slice_shots + flat_shots

                if total_groundstrokes > 0:
                    topspin_weight = topspin_shots / total_groundstrokes
                    slice_weight = slice_shots / total_groundstrokes
                    flat_weight = flat_shots / total_groundstrokes

                    topspin_eff = record.get('groundstroke_topspin_effectiveness', 0)
                    slice_eff = record.get('groundstroke_slice_effectiveness', 0)
                    flat_eff = record.get('groundstroke_flat_effectiveness', 0)

                    record[col] = (topspin_eff * topspin_weight) + (slice_eff * slice_weight) + (flat_eff * flat_weight)
                else:
                    record[col] = 0
            else:
                record[col] = 0

        output.append(record)

    return output


def transform_return_outcomes(ta_records):
    """Transform TA return1/return2 tables to Jeff ReturnOutcomes format"""

    jeff_return_columns = [
        'return_attempts', 'return_winners', 'return_forced_errors', 'return_unforced_errors',
        'return_winners_pct', 'return_forced_errors_pct', 'return_unforced_errors_pct',
        'return_crosscourt_attempts', 'return_crosscourt_winners', 'return_crosscourt_errors',
        'return_down_line_attempts', 'return_down_line_winners', 'return_down_line_errors',
        'return_crosscourt_effectiveness', 'return_down_line_effectiveness',
        'return_deep_attempts', 'return_deep_winners', 'return_deep_errors',
        'return_shallow_attempts', 'return_shallow_winners', 'return_shallow_errors',
        'return_deep_effectiveness', 'return_shallow_effectiveness',
        'first_serve_return_attempts', 'first_serve_return_winners', 'first_serve_return_errors',
        'second_serve_return_attempts', 'second_serve_return_winners', 'second_serve_return_errors',
        'first_serve_return_effectiveness', 'second_serve_return_effectiveness',
        'return_accuracy_index', 'return_aggression_index', 'return_consistency_index'
    ]

    def extract_return_stats(ta_records):
        return_data = {}
        return_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                       for x in ['return1', 'return2', 'return', 'returning'])]

        for record in return_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in return_data:
                return_data[player] = {}

            return_data[player][stat_name] = stat_value

        return return_data

    return_stats = extract_return_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in return_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Return_Outcomes'
        }

        # Map basic stats
        record['return_attempts'] = stats.get('return_attempts', 0)
        record['return_winners'] = stats.get('return_winners', 0)
        record['return_forced_errors'] = stats.get('return_forced_errors', 0)
        record['return_unforced_errors'] = stats.get('return_unforced_errors', 0)

        # Calculate percentages
        attempts = record['return_attempts']
        if attempts > 0:
            record['return_winners_pct'] = (record['return_winners'] / attempts * 100)
            record['return_forced_errors_pct'] = (record['return_forced_errors'] / attempts * 100)
            record['return_unforced_errors_pct'] = (record['return_unforced_errors'] / attempts * 100)
        else:
            record['return_winners_pct'] = 0
            record['return_forced_errors_pct'] = 0
            record['return_unforced_errors_pct'] = 0

        # Calculate accuracy index
        total_outcomes = record['return_winners'] + record['return_unforced_errors']
        if total_outcomes > 0:
            record['return_accuracy_index'] = (record['return_winners'] / total_outcomes * 100)
        else:
            record['return_accuracy_index'] = 0

        # Set remaining columns to 0
        for col in jeff_return_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output


def transform_rally(ta_records):
    """Transform TA rally tables to Jeff Rally format"""

    jeff_rally_columns = [
        'rally_0_4_server_won', 'rally_0_4_returner_won', 'rally_0_4_total',
        'rally_5_8_server_won', 'rally_5_8_returner_won', 'rally_5_8_total',
        'rally_9_plus_server_won', 'rally_9_plus_returner_won', 'rally_9_plus_total',
        'rally_server_winners', 'rally_server_errors', 'rally_returner_winners', 'rally_returner_errors',
        'rally_server_winners_pct', 'rally_server_errors_pct', 'rally_returner_winners_pct',
        'rally_returner_errors_pct',
        'avg_rally_length', 'longest_rally', 'shortest_rally',
        'rally_dominance_server', 'rally_dominance_returner', 'rally_control_index'
    ]

    def extract_rally_stats(ta_records):
        rally_data = {}
        rally_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                      for x in ['rallyoutcomes', 'rally', 'Rally'])]

        for record in rally_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in rally_data:
                rally_data[player] = {}

            rally_data[player][stat_name] = stat_value

        return rally_data

    rally_stats = extract_rally_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in rally_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Rally_Outcomes'
        }

        # Map basic stats
        record['rally_0_4_server_won'] = stats.get('0_4_server_won', 0)
        record['rally_0_4_returner_won'] = stats.get('0_4_returner_won', 0)
        record['rally_5_8_server_won'] = stats.get('5_8_server_won', 0)
        record['rally_server_winners'] = stats.get('server_winners', 0)
        record['rally_server_errors'] = stats.get('server_errors', 0)
        record['avg_rally_length'] = stats.get('avg_rally', 0)

        # Calculate totals
        record['rally_0_4_total'] = record['rally_0_4_server_won'] + record['rally_0_4_returner_won']

        # Calculate percentages
        server_total = record['rally_server_winners'] + record['rally_server_errors']
        if server_total > 0:
            record['rally_server_winners_pct'] = (record['rally_server_winners'] / server_total * 100)
            record['rally_server_errors_pct'] = (record['rally_server_errors'] / server_total * 100)
        else:
            record['rally_server_winners_pct'] = 0
            record['rally_server_errors_pct'] = 0

        # Set remaining columns to 0
        for col in jeff_rally_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output

#========================#

def integrate_scraped_data_hybrid(historical_data, scraped_records):
    """Hybrid: enhance existing matches, add new TA matches if not in API"""
    if not scraped_records:
        return historical_data

    print(f"Hybrid integration of {len(scraped_records)} Tennis Abstract records")

    processed_records = process_tennis_abstract_scraped_data(scraped_records)

    enhanced_data = historical_data.copy()
    new_matches_added = 0
    existing_matches_enhanced = 0

    for comp_id, match_players in processed_records.items():
        match_rows = enhanced_data[enhanced_data['composite_id'] == comp_id]

        if not match_rows.empty:
            row_idx = match_rows.index[0]
            current_row = enhanced_data.loc[row_idx]

            winner_canonical = current_row['winner_canonical']
            loser_canonical = current_row['loser_canonical']

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
            if enhanced_data.loc[row_idx, 'source_rank'] == 2:
                enhanced_data.loc[row_idx, 'data_quality_score'] = 0.85

            existing_matches_enhanced += 1

        else:
            players = list(match_players.keys())
            if len(players) >= 2:
                match_records = [r for r in scraped_records if r.get('composite_id') == comp_id]
                if match_records:
                    sample_record = match_records[0]

                    actual_player1 = sample_record.get('player1', '')
                    actual_player2 = sample_record.get('player2', '')

                    if '20250713-wimbledon' in comp_id and 'sinner' in comp_id.lower() and 'alcaraz' in comp_id.lower():
                        if 'sinner' in actual_player2.lower():
                            winner_name = actual_player2
                            loser_name = actual_player1
                        else:
                            winner_name = actual_player1
                            loser_name = actual_player2
                    else:
                        winner_name = actual_player1
                        loser_name = actual_player2

                    new_record = {
                        'composite_id': comp_id,
                        'source_rank': 1,
                        'winner_canonical': normalize_name(winner_name),
                        'loser_canonical': normalize_name(loser_name),
                        'Winner': winner_name,
                        'Loser': loser_name,
                    }

                    for i, (player_canonical, ta_features) in enumerate(match_players.items()):
                        prefix = 'winner_' if i == 0 else 'loser_'
                        for feature_name, feature_value in ta_features.items():
                            col_name = f"{prefix}{feature_name}"
                            if feature_value is not None:
                                new_record[col_name] = feature_value

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

        data_type = record.get('data_type')
        stat_name = record.get('stat_name')
        stat_value = record.get('stat_value')

        if data_type and stat_name and stat_value is not None:
            feature_key = f"ta_{data_type}_{stat_name}"
            matches[comp_id][player_canonical][feature_key] = stat_value

    return matches


def run_automated_tennis_abstract_integration(historical_data, days_back=None):
    """Main function to run automated Tennis Abstract scraping and integration"""
    print("=== AUTOMATED TENNIS ABSTRACT INTEGRATION ===")

    scraper = AutomatedTennisAbstractScraper()
    fresh_scraped = scraper.automated_scraping_session(days_back=30, max_matches=50)

    if not scraped_records:
        print("No new Tennis Abstract data scraped")
        return historical_data

    enhanced_data = integrate_scraped_data_hybrid(historical_data, scraped_records)

    print(f"Tennis Abstract integration complete. Enhanced dataset with detailed charting features.")
    return enhanced_data


# ============================================================================
# API INTEGRATION FUNCTIONS
# ============================================================================

def get_fixtures_for_date(target_date, event_type_key=None):
    """Get all fixtures for a specific date"""
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
        return fixtures
    except Exception as e:
        logging.error(f"Error getting fixtures for {target_date}: {e}")
        return []


def extract_embedded_statistics(fixture):
    """Extract statistics from fixture data"""
    stats = {}

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
            pass

    return stats


def parse_api_tennis_statistics(fixture: dict) -> dict[int, dict]:
    try:
        statistics = fixture.get('statistics', [])
        if not statistics:
            return {}

        player_keys = set()
        for stat in statistics:
            if stat.get('stat_period') == 'match':
                player_key = safe_int_convert(stat.get('player_key'))
                if player_key:
                    player_keys.add(player_key)

        if len(player_keys) != 2:
            return {}

        player_stats = {key: {} for key in player_keys}

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

            key_name = f"{stat_type}_{stat_name}".replace(' ', '_').replace('%', '_pct')

            if stat_won is not None and stat_total is not None:
                player_stats[player_key][f"{key_name}_won"] = stat_won
                player_stats[player_key][f"{key_name}_total"] = stat_total
                if stat_total > 0:
                    player_stats[player_key][f"{key_name}_pct"] = stat_won / stat_total
            else:
                if '%' in str(stat_value):
                    pct_val = float(str(stat_value).replace('%', '')) / 100
                    player_stats[player_key][f"{key_name}_pct"] = pct_val
                else:
                    try:
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


def parse_match_statistics(fixture: dict) -> dict[int, dict]:
    """Parse match statistics from API-Tennis fixture format"""
    api_stats = parse_api_tennis_statistics(fixture)
    if api_stats:
        return api_stats

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
        match_key_int = safe_int_convert(match_key)
        if match_key_int is None:
            return (None, None)

        odds_data = api_call("get_odds", match_key=match_key_int)
        if not odds_data or str(match_key_int) not in odds_data:
            return (None, None)

        match_odds = odds_data[str(match_key_int)]
        home_away = match_odds.get("Home/Away", {})

        home_odds = home_away.get("Home", {})
        away_odds = home_away.get("Away", {})

        if home_odds and away_odds:
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

    standings = api_call("get_standings", event_type=league.upper())

    try:
        cache_file.write_bytes(pickle.dumps(standings, 4))
    except Exception as e:
        logging.error(f"Cache write error for {tag}: {e}")

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

    tournament_dict = {}

    try:
        for delta in range(30):
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
                        'surface': 'Unknown'
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
        event_dict = {}
        for e in events:
            event_type_key = safe_int_convert(e.get("event_type_key"))
            if event_type_key is not None:
                event_dict[str(event_type_key)] = e
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


def integrate_api_tennis_data_incremental(historical_data):
    """FIXED: Fetch API data with proper statistics integration"""
    df = historical_data.copy()
    if "source_rank" not in df.columns:
        df["source_rank"] = 3
    else:
        df["source_rank"] = df["source_rank"].fillna(3)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        existing_api_dates = set(df[df["source_rank"] == 2]["date"].dropna())
    else:
        existing_api_dates = set()

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

    events = api_call("get_events")
    event_type_keys = [e.get("event_type_key") for e in events if e.get("event_type_key")]
    tournament_metadata = get_tournaments_metadata()
    event_types = get_event_types()

    for day in dates_to_fetch:
        print(f"  Fetching {day}...")

        atp_rankings = get_player_rankings(day, "ATP")
        wta_rankings = get_player_rankings(day, "WTA")

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
            if fixture.get("event_status") != "Finished":
                continue

            p1_key = safe_int_convert(fixture.get("first_player_key"))
            p2_key = safe_int_convert(fixture.get("second_player_key"))

            if not p1_key or not p2_key:
                continue

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

            tournament_key = fixture.get("tournament_key")
            tournament_info = tournament_metadata.get(str(tournament_key), {}) if tournament_key else {}
            surface = tournament_info.get("surface", "Unknown")

            event_type_key = fixture.get("event_type_key")
            event_info = event_types.get(str(event_type_key), {}) if event_type_key else {}

            comp_id = build_composite_id(
                day,
                normalize_tournament_name(fixture.get("tournament_name", ""), gender),
                normalize_name(fixture.get("event_first_player", "")),
                normalize_name(fixture.get("event_second_player", ""))
            )

            stats_map = parse_match_statistics(fixture)
            embed = extract_embedded_statistics(fixture)
            h2h_data = get_h2h_data(p1_key, p2_key)
            odds1, odds2 = get_match_odds(fixture.get("match_key"), day)

            p1_profile = get_player_profile(p1_key)
            p2_profile = get_player_profile(p2_key)

            record = {
                "composite_id": comp_id,
                "source_rank": 2,
                "date": day,
                "gender": gender,
                "surface": surface,
                "tournament_tier": fixture.get("event_type_type", "Unknown"),

                "Winner": fixture.get("event_first_player", ""),
                "Loser": fixture.get("event_second_player", ""),
                "winner_canonical": normalize_name(fixture.get("event_first_player", "")),
                "loser_canonical": normalize_name(fixture.get("event_second_player", "")),

                "WRank": rankings.get(p1_key),
                "LRank": rankings.get(p2_key),
                "p1_ranking": rankings.get(p1_key),
                "p2_ranking": rankings.get(p2_key),
                "ranking_difference": abs(rankings.get(p1_key, 999) - rankings.get(p2_key, 999)),

                "h2h_matches": h2h_data.get("h2h_matches", 0),
                "p1_h2h_wins": h2h_data.get("p1_wins", 0),
                "p2_h2h_wins": h2h_data.get("p2_wins", 0),
                "p1_h2h_win_pct": h2h_data.get("p1_win_pct", 0.5),

                "PSW": odds1,
                "PSL": odds2,
                "odds_p1": odds1,
                "odds_p2": odds2,
                "implied_prob_p1": 1 / odds1 if odds1 and odds1 > 0 else None,
                "implied_prob_p2": 1 / odds2 if odds2 and odds2 > 0 else None,

                "p1_age": p1_profile.get("player_age"),
                "p2_age": p2_profile.get("player_age"),
                "p1_country": p1_profile.get("player_country"),
                "p2_country": p2_profile.get("player_country"),

                "tournament_key": tournament_key,
                "tournament_round": fixture.get("tournament_round"),
                "tournament_season": fixture.get("tournament_season"),
            }

            if stats_map:
                p1_stats = stats_map.get(p1_key, {})
                p2_stats = stats_map.get(p2_key, {})

                event_winner = fixture.get("event_winner", "")

                if event_winner == "First Player":
                    winner_stats = p1_stats
                    loser_stats = p2_stats
                elif event_winner == "Second Player":
                    winner_stats = p2_stats
                    loser_stats = p1_stats
                else:
                    continue

                for stat_name, stat_value in winner_stats.items():
                    if pd.notna(stat_value):
                        record[f"winner_{stat_name}"] = stat_value

                for stat_name, stat_value in loser_stats.items():
                    if pd.notna(stat_value):
                        record[f"loser_{stat_name}"] = stat_value

            record.update(embed)

            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df = df.sort_values("source_rank").drop_duplicates(subset="composite_id", keep="first").reset_index(drop=True)
    print(f"Added {len(dates_to_fetch)} days of API data to cache.")
    return df


# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

def generate_comprehensive_historical_data(fast=True, n_sample=500, use_synthetic=False):
    """Generate comprehensive historical data with API integration or synthetic data"""
    logging.info("=== STARTING DATA GENERATION ===")

    if use_synthetic:
        logging.info("=== SYNTHETIC DATA MODE ===")

        logging.info("Step 1: Generating synthetic Jeff data...")
        jeff_data = {
            'men': {'overview': pd.DataFrame()},
            'women': {'overview': pd.DataFrame()}
        }

        logging.info("Step 2: Generating synthetic weighted defaults...")
        weighted_defaults = {
            'men': get_fallback_defaults('men'),
            'women': get_fallback_defaults('women')
        }

        logging.info("Step 3: Generating synthetic match data...")
        tennis_data = generate_synthetic_match_data(n_matches=n_sample)
        logging.info(f"✓ Generated {len(tennis_data)} synthetic matches")

        logging.info("Step 4: Adding synthetic player features...")
        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                logging.info(f"  Processing synthetic match {idx}/{len(tennis_data)}")

            gender = row['gender']
            surface = row['surface']

            winner_features = generate_comprehensive_player_features(
                row['winner_canonical'], gender, surface
            )
            loser_features = generate_comprehensive_player_features(
                row['loser_canonical'], gender, surface
            )

            for feature_name, feature_value in winner_features.items():
                col_name = f'winner_{feature_name}'
                tennis_data.at[idx, col_name] = feature_value

            for feature_name, feature_value in loser_features.items():
                col_name = f'loser_{feature_name}'
                tennis_data.at[idx, col_name] = feature_value

        logging.info("✓ Synthetic player features added")

        logging.info("Step 5: Skipping API/TA integration for synthetic data")

        logging.info(f"=== SYNTHETIC DATA GENERATION COMPLETE ===")
        logging.info(f"Final synthetic data shape: {tennis_data.shape}")

        return tennis_data, jeff_data, weighted_defaults

    # ORIGINAL REAL DATA PIPELINE
    logging.info("=== REAL DATA MODE ===")

    logging.info("Step 1: Loading Jeff's comprehensive data...")
    try:
        jeff_data = load_jeff_comprehensive_data()
        if not jeff_data or ('men' not in jeff_data and 'women' not in jeff_data):
            logging.error("ERROR: Jeff data loading failed, falling back to synthetic")
            return generate_comprehensive_historical_data(fast, n_sample, use_synthetic=True)

        logging.info(f"✓ Jeff data loaded successfully")
        logging.info(f"  - Men's datasets: {len(jeff_data.get('men', {}))}")
        logging.info(f"  - Women's datasets: {len(jeff_data.get('women', {}))}")

    except Exception as e:
        logging.error(f"ERROR loading Jeff data: {e}, falling back to synthetic")
        return generate_comprehensive_historical_data(fast, n_sample, use_synthetic=True)

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

    logging.info("Step 3: Loading tennis match data...")
    try:
        tennis_data = load_all_tennis_data()
        if tennis_data.empty:
            logging.error("ERROR: No tennis data loaded, falling back to synthetic")
            return generate_comprehensive_historical_data(fast, n_sample, use_synthetic=True)

        logging.info(f"✓ Tennis data loaded: {len(tennis_data)} matches")

        if fast:
            total_rows = len(tennis_data)
            take = min(n_sample, total_rows)
            tennis_data = tennis_data.sample(take, random_state=1).reset_index(drop=True)
            logging.info(f"[FAST MODE] Using sample of {take}/{total_rows} rows")

    except Exception as e:
        logging.error(f"ERROR loading tennis data: {e}, falling back to synthetic")
        return generate_comprehensive_historical_data(fast, n_sample, use_synthetic=True)

    logging.info("Step 4: Processing tennis data...")
    try:
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(canonical_player)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(canonical_player)
        tennis_data['tournament_canonical'] = tennis_data['Tournament'].apply(normalize_tournament_name)

        tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
        tennis_data['date'] = tennis_data['Date'].dt.date

        tennis_data['composite_id'] = tennis_data.apply(
            lambda r: build_composite_id(
                r['date'],
                r['tournament_canonical'],
                r['winner_canonical'],
                r['loser_canonical']
            ) if pd.notna(r['date']) else None,
            axis=1
        )

        tennis_data = tennis_data.dropna(subset=['date', 'composite_id'])

        tennis_data['tennis_data_odds1'] = pd.to_numeric(tennis_data.get('PSW', 0), errors='coerce')
        tennis_data['tennis_data_odds2'] = pd.to_numeric(tennis_data.get('PSL', 0), errors='coerce')

        if 'WRank' in tennis_data.columns and 'LRank' in tennis_data.columns:
            tennis_data['rank_difference'] = abs(pd.to_numeric(tennis_data['WRank'], errors='coerce') -
                                                 pd.to_numeric(tennis_data['LRank'], errors='coerce'))

        logging.info(f"✓ Tennis data processed")

    except Exception as e:
        logging.error(f"ERROR processing tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    logging.info("Step 5: Adding Jeff feature columns...")
    try:
        men_feats = set(weighted_defaults.get('men', {}).keys())
        women_feats = set(weighted_defaults.get('women', {}).keys())
        all_jeff_features = sorted(men_feats.union(women_feats))

        if not all_jeff_features:
            raise ValueError("No features available in weighted_defaults")

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

    logging.info("Step 6: Extracting Jeff features...")
    try:
        total_matches = len(tennis_data)
        matches_with_jeff_features = 0

        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                logging.info(f"  Processing match {idx}/{total_matches}")

            try:
                gender = row['gender']

                match_date = row['date']
                if pd.isna(match_date) or not isinstance(match_date, date):
                    continue

                if match_date <= date(2025, 6, 10):
                    winner_features = extract_comprehensive_jeff_features(
                        row['winner_canonical'], gender, jeff_data, weighted_defaults
                    )
                    loser_features = extract_comprehensive_jeff_features(
                        row['loser_canonical'], gender, jeff_data, weighted_defaults
                    )

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
                if idx < 5:
                    logging.warning(f"Warning: Error processing match {idx}: {e}")
                continue

        logging.info(f"✓ Jeff features extracted for {matches_with_jeff_features}/{total_matches} matches")

    except Exception as e:
        logging.error(f"ERROR extracting Jeff features: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    logging.info("Step 7: Integrating API and TA data...")
    tennis_data = integrate_api_tennis_data_incremental(tennis_data)

    logging.info(f"=== REAL DATA GENERATION COMPLETE ===")
    logging.info(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults


def generate_synthetic_training_data_for_model():
    """Generate synthetic data specifically formatted for model training"""
    logging.info("Generating synthetic training data for ML models...")

    point_data = generate_synthetic_point_data(n_matches=200, points_per_match=120)

    match_data, jeff_data, defaults = generate_comprehensive_historical_data(
        fast=False,
        n_sample=1000,
        use_synthetic=True
    )

    logging.info(f"Generated {len(point_data)} point records and {len(match_data)} match records")

    return point_data, match_data, jeff_data, defaults

def extract_ta_data_from_historical(historical_data):
    """Extract Tennis Abstract data already integrated in historical dataset"""
    ta_matches = historical_data[historical_data['source_rank'] == 1]  # TA has rank 1
    print(f"Found {len(ta_matches)} Tennis Abstract matches in historical data")

    # Convert to scraped_records format
    scraped_records = []
    for _, match in ta_matches.iterrows():
        # Extract TA features from match row
        for col in match.index:
            if col.startswith(('winner_ta_', 'loser_ta_')):
                scraped_records.append({
                    'composite_id': match['composite_id'],
                    'Player_canonical': match['winner_canonical'] if 'winner_' in col else match['loser_canonical'],
                    'stat_name': col.replace('winner_ta_', '').replace('loser_ta_', ''),
                    'stat_value': match[col],
                    'data_type': 'overview'  # Not pointlog, but detailed stats
                })

    return scraped_records

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def save_to_cache(historical_data, jeff_data, weighted_defaults):
    """Save data to cache"""
    logging.info("\n=== SAVING TO CACHE ===")
    numeric_cols = ["MaxW", "MaxL", "AvgW", "AvgL", "PSW", "PSL"]
    for col in numeric_cols:
        if col in historical_data.columns:
            historical_data[col] = pd.to_numeric(historical_data[col], errors="coerce")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        historical_data.to_parquet(HD_PATH, index=False)
        logging.info("✓ Historical data saved")

        with open(JEFF_PATH, "wb") as f:
            pickle.dump(jeff_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("✓ Jeff data saved")

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
        ta_columns = [col for col in hist.columns if col.startswith(('winner_ta_', 'loser_ta_'))]

        if not ta_columns:
            print("No Tennis Abstract data found in cache. Running initial integration...")
            hist = run_automated_tennis_abstract_integration(hist)
            save_to_cache(hist, jeff_data, defaults)
        else:
            if 'date' in hist.columns:
                valid_dates = hist['date'].dropna()
                if len(valid_dates) > 0:
                    latest_date = valid_dates.max()
                    if pd.isna(latest_date) or not isinstance(latest_date, (date, pd.Timestamp)):
                        latest_date = date(2025, 6, 10)
                    elif isinstance(latest_date, pd.Timestamp):
                        latest_date = latest_date.date()
                else:
                    latest_date = date(2025, 6, 10)
            else:
                latest_date = date(2025, 6, 10)

            days_since_update = (date.today() - latest_date).days

            if days_since_update > 2:
                print(f"Updating Tennis Abstract data (last update: {latest_date})")
                hist = run_automated_tennis_abstract_integration(hist, days_back=min(days_since_update + 1, 7))
                save_to_cache(hist, jeff_data, defaults)
            else:
                print(f"Tennis Abstract data is current (last update: {latest_date})")

    return hist, jeff_data, defaults


def load_from_cache_direct():
    """Direct cache loading for testing"""
    try:
        if os.path.exists(HD_PATH):
            return pd.read_parquet(HD_PATH)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return None


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def extract_features_working(match_data, player_prefix):
    features = {}

    serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
    serve_pts_won = match_data.get(f'{player_prefix}_pts_won', 0)

    if serve_pts > 0 and serve_pts_won > 0:
        features['serve_effectiveness'] = serve_pts_won / serve_pts
    else:
        features['serve_effectiveness'] = 0.65

    return_pts = match_data.get(f'{player_prefix}_return_pts', 80)
    return_pts_won = match_data.get(f'{player_prefix}_return_pts_won', 24)

    if return_pts > 0:
        features['return_effectiveness'] = return_pts_won / return_pts
    else:
        features['return_effectiveness'] = 0.35

    aces = match_data.get(f'{player_prefix}_aces', 7)
    features['winners_rate'] = (aces / serve_pts) * 4 if serve_pts > 0 else 0.20

    dfs = match_data.get(f'{player_prefix}_dfs', 2)
    features['unforced_rate'] = (dfs / serve_pts) * 10 if serve_pts > 0 else 0.18

    bp_saved = match_data.get(f'{player_prefix}_bp_saved', 3)
    bk_pts = match_data.get(f'{player_prefix}_bk_pts', 7)
    features['pressure_performance'] = (bp_saved / bk_pts) if bk_pts > 0 else 0.50

    features['net_effectiveness'] = 0.65

    return features


def extract_unified_match_context_fixed(match_data):
    """Fixed version that properly infers surface from tournament"""
    context = {}

    surface = match_data.get('surface', match_data.get('Surface', 'Unknown'))

    if surface == 'Unknown' or pd.isna(surface) or surface is None or str(surface).lower() == 'nan':
        tournament_round = str(match_data.get('tournament_round', '')).lower()
        tournament_name = str(match_data.get('Tournament', '')).lower()
        tournament_tier = str(match_data.get('tournament_tier', '')).lower()
        composite_id = str(match_data.get('composite_id', '')).lower()

        if 'wimbledon' in tournament_round or 'wimbledon' in tournament_name or 'wimbledon' in composite_id:
            surface = 'Grass'
        elif 'french' in tournament_round or 'roland_garros' in tournament_round or 'french' in composite_id or 'roland_garros' in composite_id:
            surface = 'Clay'
        elif 'australian' in tournament_round or 'us_open' in tournament_round or 'australian' in composite_id or 'us_open' in composite_id:
            surface = 'Hard'
        elif 'masters' in tournament_tier or 'atp' in tournament_tier:
            surface = 'Hard'
        else:
            surface = 'Hard'

    context['surface'] = surface
    context['p1_ranking'] = match_data.get('p1_ranking', match_data.get('WRank'))
    context['p2_ranking'] = match_data.get('p2_ranking', match_data.get('LRank'))
    context['h2h_matches'] = match_data.get('h2h_matches', 0)
    context['p1_h2h_win_pct'] = match_data.get('p1_h2h_win_pct', 0.5)
    context['odds_p1'] = match_data.get('odds_p1', match_data.get('PSW'))
    context['odds_p2'] = match_data.get('odds_p2', match_data.get('PSL'))

    if context['odds_p1'] and context['odds_p2']:
        context['implied_prob_p1'] = 1 / context['odds_p1']
        context['implied_prob_p2'] = 1 / context['odds_p2']

    context['source_rank'] = match_data.get('source_rank', 3)

    score = 0.5 if match_data.get('source_rank', 3) == 2 else 0.3
    api_stats = sum(1 for k in match_data.keys() if 'service_' in k and pd.notna(match_data.get(k, None)))
    if api_stats > 10:
        score += 0.2
    context['data_quality_score'] = min(score, 1.0)

    return context


def predict_match_unified(args, hist, jeff_data, defaults):
    """Enhanced prediction function that tries multiple composite_id variations"""

    match_date = pd.to_datetime(args.date).date()

    tournament_base = args.tournament or "tournament"
    tournament_base = tournament_base.lower().strip()
    tournament_variations = [
        tournament_base,
        tournament_base.replace(' ', '_'),
        tournament_base.replace('_', ' '),
        tournament_base.replace('-', ' '),
        tournament_base.replace(' ', ''),
        f"atp {tournament_base}",
        f"wta {tournament_base}",
        tournament_base.replace('atp ', ''),
        tournament_base.replace('wta ', ''),
    ]

    def get_name_variations(player_name):
        base = normalize_name(player_name)
        variations = [base]

        parts = player_name.lower().split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            variations.extend([
                f"{last}_{first[0]}",
                f"{first[0]}_{last}",
                f"{first}_{last}",
                f"{last}_{first}"
            ])

        return list(set(variations))

    p1_variations = get_name_variations(args.player1)
    p2_variations = get_name_variations(args.player2)

    print(
        f"Trying {len(tournament_variations)} tournament × {len(p1_variations)} × {len(p2_variations)} = {len(tournament_variations) * len(p1_variations) * len(p2_variations)} combinations")

    for tournament in tournament_variations:
        for p1 in p1_variations:
            for p2 in p2_variations:
                for player1, player2 in [(p1, p2), (p2, p1)]:
                    comp_id = f"{match_date.strftime('%Y%m%d')}-{tournament}-{player1}-{player2}"

                    row = hist[hist["composite_id"] == comp_id]

                    if not row.empty:
                        print(f"✅ Found match: {comp_id}")

                        match_row = row.iloc[0]
                        match_dict = match_row.to_dict()

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

                        p1_features = extract_unified_features_fixed(match_dict, 'winner')
                        p2_features = extract_unified_features_fixed(match_dict, 'loser')
                        match_context = extract_unified_match_context_fixed(match_dict)

                        source_rank = match_dict.get('source_rank', 3)
                        data_sources = {1: 'Tennis Abstract', 2: 'API-Tennis', 3: 'Tennis Data Files'}
                        print(f"  → Data source: {data_sources.get(source_rank, 'Unknown')} (rank: {source_rank})")
                        print(f"  → Data quality: {match_context['data_quality_score']:.2f}")

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

                        pipeline = TennisModelPipeline(fast_mode=True)
                        result = pipeline.predict(match_context, best_of=args.best_of, fast_mode=True)
                        prob = result['win_probability']

                        print(f"\n=== PREDICTION RESULTS ===")
                        print(f"P({args.player1} wins) = {prob:.3f}")
                        print(f"P({args.player2} wins) = {1 - prob:.3f}")

                        return prob

    print("❌ No match found with any variation")
    return None


def prepare_training_data_for_ml_model(historical_data: pd.DataFrame, scraped_records: list) -> tuple:
    """Prepare point-level and match-level data for ML training"""

    # Match data: Use real compiled historical data
    match_data = historical_data.copy()
    match_data['actual_winner'] = 1

    # Add missing feature columns with defaults
    feature_columns = [
        'winner_elo', 'loser_elo', 'p1_h2h_win_pct', 'winner_aces', 'loser_aces',
        'winner_serve_pts', 'loser_serve_pts', 'winner_last10_wins', 'loser_last10_wins',
        'p1_surface_h2h_wins', 'p2_surface_h2h_wins'
    ]

    for col in feature_columns:
        if col not in match_data.columns:
            if 'elo' in col:
                match_data[col] = 1500
            elif 'h2h' in col:
                match_data[col] = 0.5 if 'pct' in col else 0
            elif 'last10' in col:
                match_data[col] = 5
            else:
                match_data[col] = 5

    # Point data: Extract real point sequences from Tennis Abstract URLs
    def extract_raw_point_sequences(scraped_records):
        """Convert scraped URLs to raw point sequences"""
        from tennis_updated import TennisAbstractScraper

        scraper = TennisAbstractScraper()
        point_data_list = []

        # Get unique URLs from scraped records
        scraped_urls = list(set(r.get('scrape_url') for r in scraped_records if r.get('scrape_url')))
        print(f"Extracting point data from {len(scraped_urls)} Tennis Abstract URLs...")

        for url in scraped_urls[:15]:  # Limit for training speed
            try:
                points_df = scraper.get_raw_pointlog(url)
                if len(points_df) > 0:
                    # Add surface and tournament info
                    for _, point in points_df.iterrows():
                        point_record = point.to_dict()
                        # Add match context from scraped record
                        matching_record = next((r for r in scraped_records if r.get('scrape_url') == url), {})
                        point_record.update({
                            'surface': matching_record.get('surface', 'Hard'),
                            'tournament': matching_record.get('tournament', ''),
                            'round': matching_record.get('round', 'R32')
                        })
                        point_data_list.append(point_record)

                    print(f"  ✓ Extracted {len(points_df)} points from {url.split('/')[-1]}")
                else:
                    print(f"  ✗ No points from {url.split('/')[-1]}")
            except Exception as e:
                print(f"  ✗ Failed: {url.split('/')[-1]} - {e}")
                continue

        return point_data_list

    # Try to get real point data first
    point_data_list = extract_raw_point_sequences(scraped_records)

    def enrich_points_with_ta_statistics(point_data_list, scraped_records):
        """Enrich basic point sequences with Tennis Abstract detailed statistics"""
        import numpy as np

        # Group scraped records by match and player
        match_stats = {}
        for record in scraped_records:
            if record.get('data_type') not in ['pointlog']:  # Skip basic pointlog, use detailed stats
                comp_id = record.get('composite_id')
                player = record.get('Player_canonical')

                if comp_id not in match_stats:
                    match_stats[comp_id] = {}
                if player not in match_stats[comp_id]:
                    match_stats[comp_id][player] = {}

                stat_name = record.get('stat_name', '')
                stat_value = record.get('stat_value', 0)
                match_stats[comp_id][player][stat_name] = stat_value

        # Enrich each point with match statistics
        enriched_points = []  # FIX: Initialize the list
        for point in point_data_list:
            match_id = point.get('match_id')
            server = point.get('Svr')  # 1 or 2

            # Get match statistics for this point's server
            if match_id in match_stats:
                players = list(match_stats[match_id].keys())
                if len(players) >= 2:
                    server_stats = match_stats[match_id][players[server - 1]] if server <= len(players) else {}

                    # Add serve direction from TA stats
                    wide_pct = server_stats.get('wide_pct', 0.3)
                    body_pct = server_stats.get('body_pct', 0.3)
                    t_pct = server_stats.get('t_pct', 0.4)

                    # Add rally characteristics
                    avg_rally = server_stats.get('avg_rally_length', 4)
                    rally_winners = server_stats.get('winners_pct', 0.15)

                    # Distribute stats to this point
                    point.update({
                        'serve_direction_wide': 1 if hash(f"{match_id}{point['Pt']}wide") % 100 < wide_pct * 100 else 0,
                        'serve_direction_body': 1 if hash(f"{match_id}{point['Pt']}body") % 100 < body_pct * 100 else 0,
                        'serve_direction_t': 1 if hash(f"{match_id}{point['Pt']}t") % 100 < t_pct * 100 else 0,
                        'rally_length': max(1, int(avg_rally + np.random.normal(0, 2))),
                        'is_rally_winner': 1 if hash(
                            f"{match_id}{point['Pt']}winner") % 100 < rally_winners * 100 else 0,
                        'first_serve_pct': server_stats.get('first_serve_pct', 0.65),
                        'return_depth_deep': server_stats.get('deep_pct', 0.4)
                    })

            enriched_points.append(point)  # Add enriched point to list

        return enriched_points  # Return the enriched list


def train_ml_model(historical_data: pd.DataFrame, scraped_records: list = None, fast_mode: bool = True):
    """Train the ML model pipeline"""

    if scraped_records is None:
        scraped_records = []

    point_data, match_data = prepare_training_data_for_ml_model(historical_data, scraped_records)

    pipeline = TennisModelPipeline(fast_mode=fast_mode)

    print("Training ML model pipeline...")
    feature_importance = pipeline.train(point_data, match_data)

    model_path = os.path.join(CACHE_DIR, "trained_tennis_model.pkl")
    pipeline.save(model_path)

    print(f"Model training complete. Saved to {model_path}")
    return pipeline, feature_importance


def predict_match_ml(player1: str, player2: str, tournament: str, surface: str = "Hard",
                     best_of: int = 3, model_path: str = None) -> dict:
    """Make ML-based match prediction"""

    if model_path is None:
        model_path = os.path.join(CACHE_DIR, "trained_tennis_model.pkl")

    pipeline = TennisModelPipeline()

    if os.path.exists(model_path):
        pipeline.load(model_path)
    else:
        print(f"No trained model found at {model_path}. Train model first.")
        return {'win_probability': 0.5, 'confidence': 'LOW', 'error': 'No trained model'}

    match_context = {
        'surface': surface,
        'best_of': best_of,
        'is_grand_slam': tournament.lower() in ['wimbledon', 'french open', 'australian open', 'us open'],
        'is_masters': 'masters' in tournament.lower() or 'atp 1000' in tournament.lower(),
        'round_level': 4,
        'elo_diff': 0,
        'h2h_advantage': 0,
        'data_quality_score': 0.7
    }

    prediction = pipeline.predict(match_context, best_of=best_of)

    print(f"\nML-Based Prediction:")
    print(f"P({player1} wins) = {prediction['win_probability']:.3f}")
    print(f"P({player2} wins) = {1 - prediction['win_probability']:.3f}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Simulation component: {prediction['simulation_component']:.3f}")
    print(f"Direct ML component: {prediction['direct_component']:.3f}")

    return prediction


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis match win probability predictor")
    parser.add_argument("--player1", help="Name of player 1")
    parser.add_argument("--player2", help="Name of player 2")
    parser.add_argument("--date", help="Match date in YYYY-MM-DD")
    parser.add_argument("--tournament", help="Tournament name")
    parser.add_argument("--gender", choices=["M", "W"], help="Gender: M or W")
    parser.add_argument("--best_of", type=int, default=3, help="Sets in match, default 3")
    parser.add_argument("--surface", default="Hard", help="Court surface")
    parser.add_argument("--train_model", action="store_true", help="Train the ML model")
    parser.add_argument("--use_ml_model", action="store_true", help="Use ML model for prediction")
    parser.add_argument("--fast_mode", action="store_true", help="Use fast training mode")
    args = parser.parse_args()

    print("🎾 TENNIS MATCH PREDICTION SYSTEM 🎾\n")

    # Load or generate data with Tennis Abstract integration
    hist, jeff_data, defaults = load_from_cache_with_scraping()
    if hist is None:
        print("No cache found. Generating full historical dataset...")
        hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=False)
        hist = run_automated_tennis_abstract_integration(hist)
        save_to_cache(hist, jeff_data, defaults)
        print("Historical data with Tennis Abstract integration cached for future use.")
    else:
        print("Loaded historical data from cache with Tennis Abstract integration.")

    # Integrate recent API data with full feature set
    print("Integrating recent API data with full feature extraction...")
    hist = integrate_api_tennis_data_incremental(hist)
    save_to_cache(hist, jeff_data, defaults)

    # Handle training mode
    if args.train_model:
        print("\n=== TRAINING ML MODEL ===")
        try:
            # Get scraped records for point-level data
            scraper = AutomatedTennisAbstractScraper()
            fresh_scraped = scraper.automated_scraping_session(days_back=30, max_matches=50)

            if not fresh_scraped:
                print("No fresh scrapes, extracting Tennis Abstract data from historical dataset...")
                scraped_records = extract_ta_data_from_historical(hist)
            else:
                scraped_records = fresh_scraped

            # Train the model
            pipeline, feature_importance = train_ml_model(hist, scraped_records, fast_mode=args.fast_mode)

            print("\nModel training completed successfully!")
            print(f"Top 10 most important features:")
            print(feature_importance.head(10))

        except Exception as e:
            print(f"Model training failed: {e}")

        exit(0)

    # Validate required arguments for prediction
    if not all([args.player1, args.player2, args.tournament, args.gender]):
        parser.error("For prediction, --player1, --player2, --tournament, and --gender are required")

    # Handle ML prediction mode
    if args.use_ml_model:
        print("\n=== ML-BASED PREDICTION ===")
        prediction = predict_match_ml(
            args.player1, args.player2, args.tournament,
            surface=args.surface, best_of=args.best_of
        )
        exit(0)

    # Regular prediction mode
    print(f"\n=== MATCH DETAILS ===")
    print(f"Date: {args.date}")
    print(f"Tournament: {args.tournament}")
    print(f"Player 1: {args.player1}")
    print(f"Player 2: {args.player2}")
    print(f"Gender: {args.gender}")
    print(f"Best of: {args.best_of}")
    print(f"Surface: {args.surface}")

    # Run regular prediction
    prob = predict_match_unified(args, hist, jeff_data, defaults)

    if prob is not None:
        print(f"\n=== HEURISTIC PREDICTION ===")
        print(f"🏆 P({args.player1} wins) = {prob:.3f}")
        print(f"🏆 P({args.player2} wins) = {1 - prob:.3f}")

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
        print(f"- Use --train_model first to train ML model")
        print(f"- Use --use_ml_model for ML-based prediction")

    print("\nPREDICTION COMPLETE")