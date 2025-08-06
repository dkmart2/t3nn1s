# ============================================================================
# TENNIS DATA PIPELINE - COMPREHENSIVE TENNIS PREDICTION SYSTEM
# ============================================================================

# ============================================================================
# IMPORTS / CONFIGURATION
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

                    # ADDED: Store original match_id for Tennis Abstract scraping
                    if key == 'matches' and 'match_id' in df.columns:
                        df['jeff_original_id'] = df['match_id'].copy()
                        logging.info(f"Stored original match_ids for {gender} matches")

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

def extract_player_name_from_canonical(player_canonical):
    """Extract searchable name from canonical format (surname_initial)"""
    if not player_canonical or pd.isna(player_canonical):
        return ""

    parts = str(player_canonical).split('_')
    if len(parts) >= 1:
        # Capitalize first part (surname)
        return parts[0].capitalize()
    return str(player_canonical).capitalize()


def safe_column_access(df, columns, default_value=0):
    """Safely access DataFrame columns, filling missing ones with default"""
    result = {}
    for col in columns:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = pd.Series([default_value] * len(df), index=df.index)
    return pd.DataFrame(result, index=df.index)


def find_player_data_robust(df, player_canonical):
    """ENHANCED: Multi-strategy player matching"""
    if df.empty:
        return pd.DataFrame()

    # Extract search components
    search_name = extract_player_name_from_canonical(player_canonical)

    # Strategy 1: Direct canonical match
    if 'Player_canonical' in df.columns:
        direct_match = df[df['Player_canonical'] == player_canonical]
        if not direct_match.empty:
            return direct_match

    # Strategy 2: Player column match
    if 'player' in df.columns:
        player_match = df[df['player'].str.contains(search_name, case=False, na=False)]
        if not player_match.empty:
            return player_match

    # Strategy 3: Server/Returner columns (for Rally data)
    if 'server' in df.columns:
        server_match = df[df['server'].str.contains(search_name, case=False, na=False)]
        if not server_match.empty:
            return server_match

    if 'returner' in df.columns:
        returner_match = df[df['returner'].str.contains(search_name, case=False, na=False)]
        if not returner_match.empty:
            return returner_match

    # Strategy 4: Fuzzy name matching
    possible_cols = ['Player_canonical', 'player', 'server', 'returner']
    for col in possible_cols:
        if col in df.columns:
            fuzzy_match = df[df[col].str.contains(search_name[:4], case=False, na=False)]
            if not fuzzy_match.empty:
                return fuzzy_match

    return pd.DataFrame()


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


# ============================================================================
# JEFF FILES EXTRACTION
# ============================================================================

def extract_serve_basics_features(player_canonical, gender, jeff_data):
    """Extract 16 ServeBasics features with robust error handling"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'serve_basics' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['serve_basics']
    player_data = find_player_data(df, player_canonical)

    if player_data.empty:
        return {}

    # Find Total row
    total_data = player_data[player_data['row'] == 'Total']
    if total_data.empty:
        return {}

    # Define expected columns with fallbacks
    expected_cols = ['pts', 'pts_won', 'aces', 'unret', 'forced_err', 'pts_won_lte_3_shots', 'wide', 'body', 't']
    safe_data = safe_column_access(total_data, expected_cols)
    totals = safe_data.sum()

    features = {}
    serve_pts = totals['pts']

    if serve_pts > 0:
        features['sb_serve_pts'] = float(serve_pts)
        features['sb_aces'] = float(totals['aces'])
        features['sb_ace_rate'] = features['sb_aces'] / serve_pts
        features['sb_wide_pct'] = float(totals['wide']) / serve_pts
        features['sb_body_pct'] = float(totals['body']) / serve_pts
        features['sb_t_pct'] = float(totals['t']) / serve_pts
        features['sb_pts_won'] = float(totals['pts_won'])
        features['sb_serve_win_pct'] = features['sb_pts_won'] / serve_pts
        features['sb_unret'] = float(totals['unret'])
        features['sb_unret_pct'] = features['sb_unret'] / serve_pts
        features['sb_forced_err'] = float(totals['forced_err'])
        features['sb_forced_err_pct'] = features['sb_forced_err'] / serve_pts
        features['sb_quick_points'] = float(totals['pts_won_lte_3_shots'])
        features['sb_quick_points_pct'] = features['sb_quick_points'] / serve_pts
        features['sb_service_dominance'] = (features['sb_aces'] + features['sb_unret'] + features['sb_forced_err']) / serve_pts
        features['sb_placement_variety'] = 1 - max(features['sb_wide_pct'], features['sb_body_pct'], features['sb_t_pct'])

    return features

def extract_key_points_serve_features(player_canonical, gender, jeff_data):
    """Extract 35+ features from KeyPointsServe with robust player matching"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'key_points_serve' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['key_points_serve']
    player_data = find_player_data(df, player_canonical)

    if player_data.empty:
        return {}

    features = {}

    # Break Point Performance (BP row)
    bp_data = player_data[player_data['row'] == 'BP']
    if not bp_data.empty:
        expected_cols = ['pts', 'pts_won', 'first_in', 'aces', 'svc_winners', 'rally_winners', 'rally_forced', 'unforced', 'dfs']
        safe_data = safe_column_access(bp_data, expected_cols)
        bp_totals = safe_data.sum()

        bp_pts = bp_totals['pts']
        if bp_pts > 0:
            features['kps_bp_pts'] = float(bp_pts)
            features['kps_bp_pts_won'] = float(bp_totals['pts_won'])
            features['kps_bp_first_in'] = float(bp_totals['first_in'])
            features['kps_bp_aces'] = float(bp_totals['aces'])
            features['kps_bp_svc_winners'] = float(bp_totals['svc_winners'])
            features['kps_bp_rally_winners'] = float(bp_totals['rally_winners'])
            features['kps_bp_rally_forced'] = float(bp_totals['rally_forced'])
            features['kps_bp_unforced'] = float(bp_totals['unforced'])
            features['kps_bp_dfs'] = float(bp_totals['dfs'])

            features['kps_bp_save_pct'] = features['kps_bp_pts_won'] / bp_pts
            features['kps_bp_first_serve_pct'] = features['kps_bp_first_in'] / bp_pts
            features['kps_bp_ace_rate'] = features['kps_bp_aces'] / bp_pts
            features['kps_bp_service_winner_rate'] = features['kps_bp_svc_winners'] / bp_pts
            features['kps_bp_error_rate'] = features['kps_bp_unforced'] / bp_pts
            features['kps_bp_df_rate'] = features['kps_bp_dfs'] / bp_pts

            total_winners = features['kps_bp_aces'] + features['kps_bp_svc_winners'] + features['kps_bp_rally_winners']
            features['kps_bp_total_winners'] = total_winners
            features['kps_bp_winner_rate'] = total_winners / bp_pts
            features['kps_bp_dominance'] = (total_winners + features['kps_bp_rally_forced']) / bp_pts

    # Game Point Performance (GP row)
    gp_data = player_data[player_data['row'] == 'GP']
    if not gp_data.empty:
        expected_cols = ['pts', 'pts_won', 'first_in', 'aces', 'svc_winners', 'rally_winners', 'rally_forced', 'unforced', 'dfs']
        safe_data = safe_column_access(gp_data, expected_cols)
        gp_totals = safe_data.sum()

        gp_pts = gp_totals['pts']
        if gp_pts > 0:
            features['kps_gp_pts'] = float(gp_pts)
            features['kps_gp_pts_won'] = float(gp_totals['pts_won'])
            features['kps_gp_first_in'] = float(gp_totals['first_in'])
            features['kps_gp_aces'] = float(gp_totals['aces'])
            features['kps_gp_svc_winners'] = float(gp_totals['svc_winners'])
            features['kps_gp_rally_winners'] = float(gp_totals['rally_winners'])
            features['kps_gp_rally_forced'] = float(gp_totals['rally_forced'])
            features['kps_gp_unforced'] = float(gp_totals['unforced'])
            features['kps_gp_dfs'] = float(gp_totals['dfs'])

            features['kps_gp_conversion_pct'] = features['kps_gp_pts_won'] / gp_pts
            features['kps_gp_first_serve_pct'] = features['kps_gp_first_in'] / gp_pts
            features['kps_gp_ace_rate'] = features['kps_gp_aces'] / gp_pts
            features['kps_gp_service_winner_rate'] = features['kps_gp_svc_winners'] / gp_pts
            features['kps_gp_error_rate'] = features['kps_gp_unforced'] / gp_pts
            features['kps_gp_df_rate'] = features['kps_gp_dfs'] / gp_pts

    # Overall Key Point Performance (STotal row)
    total_data = player_data[player_data['row'] == 'STotal']
    if not total_data.empty:
        expected_cols = ['pts', 'pts_won', 'first_in', 'aces', 'svc_winners', 'rally_winners', 'rally_forced', 'unforced', 'dfs']
        safe_data = safe_column_access(total_data, expected_cols)
        totals = safe_data.sum()

        total_pts = totals['pts']
        if total_pts > 0:
            features['kps_total_pts'] = float(total_pts)
            features['kps_total_pts_won'] = float(totals['pts_won'])
            features['kps_total_aces'] = float(totals['aces'])
            features['kps_total_win_pct'] = features['kps_total_pts_won'] / total_pts
            features['kps_total_first_in'] = float(totals['first_in'])
            features['kps_total_svc_winners'] = float(totals['svc_winners'])
            features['kps_total_rally_winners'] = float(totals['rally_winners'])
            features['kps_total_unforced'] = float(totals['unforced'])
            features['kps_total_dfs'] = float(totals['dfs'])

            features['kps_total_first_serve_pct'] = features['kps_total_first_in'] / total_pts
            features['kps_total_ace_rate'] = features['kps_total_aces'] / total_pts
            total_service_winners = features['kps_total_aces'] + features['kps_total_svc_winners']
            features['kps_total_service_dominance'] = total_service_winners / total_pts
            features['kps_total_rally_effectiveness'] = features['kps_total_rally_winners'] / total_pts
            features['kps_total_error_rate'] = features['kps_total_unforced'] / total_pts
            features['kps_total_winner_error_ratio'] = (total_service_winners + features['kps_total_rally_winners']) / \
                                                       features['kps_total_unforced'] if features['kps_total_unforced'] > 0 else 10.0

    # Comparative pressure metrics
    bp_save_pct = features.get('kps_bp_save_pct', 0)
    gp_conversion_pct = features.get('kps_gp_conversion_pct', 0)
    total_win_pct = features.get('kps_total_win_pct', 0)

    if bp_save_pct > 0 and gp_conversion_pct > 0:
        features['kps_pressure_differential'] = gp_conversion_pct - bp_save_pct
        features['kps_clutch_consistency'] = min(bp_save_pct, gp_conversion_pct) / max(bp_save_pct, gp_conversion_pct) if max(bp_save_pct, gp_conversion_pct) > 0 else 0

    pressure_contexts = sum([1 for x in [bp_save_pct, gp_conversion_pct] if x > 0])
    features['kps_pressure_experience'] = pressure_contexts

    if pressure_contexts > 0:
        clutch_scores = [x for x in [bp_save_pct, gp_conversion_pct] if x > 0]
        features['kps_average_clutch_performance'] = sum(clutch_scores) / len(clutch_scores)
        features['kps_clutch_floor'] = min(clutch_scores)
        features['kps_clutch_ceiling'] = max(clutch_scores)
        features['kps_clutch_range'] = features['kps_clutch_ceiling'] - features['kps_clutch_floor']

    return features

def extract_key_points_return_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 35+ features from KeyPointsReturn pressure returning (was 6 basic)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'key_points_return' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['key_points_return']

    # Handle name mapping - use contains for partial matching
    if 'Player_canonical' in df.columns:
        player_data = df[df['Player_canonical'] == player_canonical]
    else:
        # Extract surname from canonical format for matching
        if player_canonical == 'sinner_j':
            search_name = 'Sinner'
        elif player_canonical == 'alcaraz_c':
            search_name = 'Alcaraz'
        elif player_canonical == 'djokovic_n':
            search_name = 'Djokovic'
        else:
            # Generic extraction: take part before underscore, capitalize
            search_name = player_canonical.split('_')[0].capitalize()

        player_data = df[df['player'].str.contains(search_name, case=False, na=False)]

    if player_data.empty:
        return {}

    features = {}

    # === BREAK POINT OPPORTUNITIES (BPO row) - Creating break points ===
    bpo_data = player_data[player_data['row'] == 'BPO']
    if not bpo_data.empty:
        bpo_totals = bpo_data[['pts', 'pts_won', 'rally_winners', 'rally_forced', 'unforced']].sum()

        bpo_pts = bpo_totals['pts']
        if bpo_pts > 0:
            # Raw break point opportunity data (5 features)
            features['kpr_bpo_pts'] = float(bpo_pts)
            features['kpr_bpo_pts_won'] = float(bpo_totals['pts_won'])
            features['kpr_bpo_rally_winners'] = float(bpo_totals['rally_winners'])
            features['kpr_bpo_rally_forced'] = float(bpo_totals['rally_forced'])
            features['kpr_bpo_unforced'] = float(bpo_totals['unforced'])

            # Break point conversion percentages (4 features)
            features['kpr_bpo_conversion_pct'] = features['kpr_bpo_pts_won'] / bpo_pts
            features['kpr_bpo_winner_rate'] = features['kpr_bpo_rally_winners'] / bpo_pts
            features['kpr_bpo_forced_error_rate'] = features['kpr_bpo_rally_forced'] / bpo_pts
            features['kpr_bpo_error_rate'] = features['kpr_bpo_unforced'] / bpo_pts

            # Break point tactical metrics (3 features)
            total_positive = features['kpr_bpo_rally_winners'] + features['kpr_bpo_rally_forced']
            features['kpr_bpo_total_positive'] = total_positive
            features['kpr_bpo_positive_rate'] = total_positive / bpo_pts
            features['kpr_bpo_aggression_ratio'] = features['kpr_bpo_rally_winners'] / features['kpr_bpo_unforced'] if \
            features['kpr_bpo_unforced'] > 0 else 10.0

    # === GAME POINT FACED (GPF row) - Defending against game points ===
    gpf_data = player_data[player_data['row'] == 'GPF']
    if not gpf_data.empty:
        gpf_totals = gpf_data[['pts', 'pts_won', 'rally_winners', 'rally_forced', 'unforced']].sum()

        gpf_pts = gpf_totals['pts']
        if gpf_pts > 0:
            # Raw game point defense data (5 features)
            features['kpr_gpf_pts'] = float(gpf_pts)
            features['kpr_gpf_pts_won'] = float(gpf_totals['pts_won'])
            features['kpr_gpf_rally_winners'] = float(gpf_totals['rally_winners'])
            features['kpr_gpf_rally_forced'] = float(gpf_totals['rally_forced'])
            features['kpr_gpf_unforced'] = float(gpf_totals['unforced'])

            # Game point defense percentages (4 features)
            features['kpr_gpf_defense_pct'] = features['kpr_gpf_pts_won'] / gpf_pts
            features['kpr_gpf_winner_rate'] = features['kpr_gpf_rally_winners'] / gpf_pts
            features['kpr_gpf_forced_error_rate'] = features['kpr_gpf_rally_forced'] / gpf_pts
            features['kpr_gpf_error_rate'] = features['kpr_gpf_unforced'] / gpf_pts

            # Game point defense tactical metrics (3 features)
            total_defensive_positive = features['kpr_gpf_rally_winners'] + features['kpr_gpf_rally_forced']
            features['kpr_gpf_total_positive'] = total_defensive_positive
            features['kpr_gpf_positive_rate'] = total_defensive_positive / gpf_pts
            features['kpr_gpf_defense_quality'] = (features['kpr_gpf_pts_won'] + total_defensive_positive) / gpf_pts

    # === DEUCE RETURN (DeuceR row) - Return games in deuce situations ===
    deuce_data = player_data[player_data['row'] == 'DeuceR']
    if not deuce_data.empty:
        deuce_totals = deuce_data[['pts', 'pts_won', 'rally_winners', 'rally_forced', 'unforced']].sum()

        deuce_pts = deuce_totals['pts']
        if deuce_pts > 0:
            # Raw deuce return data (5 features)
            features['kpr_deuce_pts'] = float(deuce_pts)
            features['kpr_deuce_pts_won'] = float(deuce_totals['pts_won'])
            features['kpr_deuce_rally_winners'] = float(deuce_totals['rally_winners'])
            features['kpr_deuce_rally_forced'] = float(deuce_totals['rally_forced'])
            features['kpr_deuce_unforced'] = float(deuce_totals['unforced'])

            # Deuce return percentages (4 features)
            features['kpr_deuce_win_pct'] = features['kpr_deuce_pts_won'] / deuce_pts
            features['kpr_deuce_winner_rate'] = features['kpr_deuce_rally_winners'] / deuce_pts
            features['kpr_deuce_forced_error_rate'] = features['kpr_deuce_rally_forced'] / deuce_pts
            features['kpr_deuce_error_rate'] = features['kpr_deuce_unforced'] / deuce_pts

            # Deuce return tactical metrics (3 features)
            deuce_positive = features['kpr_deuce_rally_winners'] + features['kpr_deuce_rally_forced']
            features['kpr_deuce_total_positive'] = deuce_positive
            features['kpr_deuce_positive_rate'] = deuce_positive / deuce_pts
            features['kpr_deuce_effectiveness'] = (features['kpr_deuce_pts_won'] + deuce_positive) / deuce_pts

    # === OVERALL RETURN PERFORMANCE (RTotal row) ===
    total_data = player_data[player_data['row'] == 'RTotal']
    if not total_data.empty:
        totals = total_data[['pts', 'pts_won', 'rally_winners', 'rally_forced', 'unforced']].sum()

        total_pts = totals['pts']
        if total_pts > 0:
            # Overall return key point data (original + new = 5 features)
            features['kpr_total_pts'] = float(total_pts)
            features['kpr_total_pts_won'] = float(totals['pts_won'])
            features['kpr_total_rally_winners'] = float(totals['rally_winners'])
            features['kpr_total_rally_forced'] = float(totals['rally_forced'])
            features['kpr_total_unforced'] = float(totals['unforced'])

            # Overall return tactical metrics (6 features)
            features['kpr_total_win_pct'] = features['kpr_total_pts_won'] / total_pts
            features['kpr_total_winner_rate'] = features['kpr_total_rally_winners'] / total_pts
            features['kpr_total_forced_error_rate'] = features['kpr_total_rally_forced'] / total_pts
            features['kpr_total_error_rate'] = features['kpr_total_unforced'] / total_pts
            total_return_positive = features['kpr_total_rally_winners'] + features['kpr_total_rally_forced']
            features['kpr_total_positive_rate'] = total_return_positive / total_pts
            features['kpr_total_winner_error_ratio'] = features['kpr_total_rally_winners'] / features[
                'kpr_total_unforced'] if features['kpr_total_unforced'] > 0 else 10.0

    # === COMPARATIVE RETURN PRESSURE METRICS (8 features) ===
    bpo_conversion = features.get('kpr_bpo_conversion_pct', 0)
    gpf_defense = features.get('kpr_gpf_defense_pct', 0)
    deuce_win = features.get('kpr_deuce_win_pct', 0)
    total_win = features.get('kpr_total_win_pct', 0)

    if bpo_conversion > 0 and gpf_defense > 0:
        # Attack vs defense differential - how much better at creating breaks vs defending game points
        features['kpr_attack_defense_differential'] = bpo_conversion - gpf_defense
        features['kpr_pressure_versatility'] = min(bpo_conversion, gpf_defense) / max(bpo_conversion,
                                                                                      gpf_defense) if max(
            bpo_conversion, gpf_defense) > 0 else 0

    if deuce_win > 0 and total_win > 0:
        features['kpr_deuce_vs_average'] = deuce_win - total_win

    # Return pressure situation variety (how many different pressure contexts player faced)
    pressure_contexts = sum([1 for x in [bpo_conversion, gpf_defense, deuce_win] if x > 0])
    features['kpr_pressure_experience'] = pressure_contexts

    # Overall return pressure performance index
    if pressure_contexts > 0:
        pressure_scores = [x for x in [bpo_conversion, gpf_defense, deuce_win] if x > 0]
        features['kpr_average_pressure_performance'] = sum(pressure_scores) / len(pressure_scores)
        features['kpr_pressure_floor'] = min(pressure_scores)
        features['kpr_pressure_ceiling'] = max(pressure_scores)
        features['kpr_pressure_range'] = features['kpr_pressure_ceiling'] - features['kpr_pressure_floor']

    # === RETURN AGGRESSION ANALYSIS (6 features) ===
    # Analyze aggressive return patterns across pressure situations
    bpo_winner_rate = features.get('kpr_bpo_winner_rate', 0)
    gpf_winner_rate = features.get('kpr_gpf_winner_rate', 0)
    deuce_winner_rate = features.get('kpr_deuce_winner_rate', 0)
    total_winner_rate = features.get('kpr_total_winner_rate', 0)

    winner_rates = [x for x in [bpo_winner_rate, gpf_winner_rate, deuce_winner_rate] if x > 0]
    if winner_rates:
        features['kpr_average_aggression'] = sum(winner_rates) / len(winner_rates)
        features['kpr_clutch_aggression'] = max(winner_rates)  # Peak aggression in pressure

        # Situational aggression adaptation
        if bpo_winner_rate > 0 and gpf_winner_rate > 0:
            features[
                'kpr_aggression_adaptation'] = bpo_winner_rate - gpf_winner_rate  # Should be positive (more aggressive on break points)

    # Error discipline under pressure
    bpo_error_rate = features.get('kpr_bpo_error_rate', 0)
    gpf_error_rate = features.get('kpr_gpf_error_rate', 0)
    deuce_error_rate = features.get('kpr_deuce_error_rate', 0)

    error_rates = [x for x in [bpo_error_rate, gpf_error_rate, deuce_error_rate] if x > 0]
    if error_rates:
        features['kpr_average_error_discipline'] = 1 - (
                    sum(error_rates) / len(error_rates))  # Inverted so higher is better
        features['kpr_pressure_discipline'] = 1 - max(error_rates)  # Best discipline in worst situation

        # Pressure error control differential
        if bpo_error_rate > 0 and gpf_error_rate > 0:
            features[
                'kpr_situational_discipline'] = gpf_error_rate - bpo_error_rate  # Should be negative (fewer errors when attacking)

    return features


def extract_net_points_features(player_canonical, gender, jeff_data):
    """
    EXPANDED: Extract 35+ features from NetPoints comprehensive net game analysis (was 7 basic)
    Analyzes net point effectiveness, approach shots, passing shot defense, and net positioning
    """
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'net_points' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['net_points']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # OVERALL NET POINT PERFORMANCE (NetPoints row)
    netpoints_data = player_data[player_data['row'] == 'NetPoints']
    if not netpoints_data.empty:
        np_row = netpoints_data.iloc[0]

        # Raw net point data (6 features)
        features['np_total_attempts'] = float(np_row['net_pts'])
        features['np_points_won'] = float(np_row['pts_won'])
        features['np_net_winners'] = float(np_row['net_winner'])
        features['np_induced_forced'] = float(np_row['induced_forced'])
        features['np_unforced_errors'] = float(np_row['net_unforced'])
        features['np_passed_at_net'] = float(np_row['passed_at_net'])

        # Net point success rates (5 features)
        if features['np_total_attempts'] > 0:
            features['np_success_rate'] = (features['np_points_won'] / features['np_total_attempts']) * 100
            features['np_winner_rate'] = (features['np_net_winners'] / features['np_total_attempts']) * 100
            features['np_error_rate'] = (features['np_unforced_errors'] / features['np_total_attempts']) * 100
            features['np_passed_rate'] = (features['np_passed_at_net'] / features['np_total_attempts']) * 100
            features['np_pressure_rate'] = (features['np_induced_forced'] / features['np_total_attempts']) * 100
        else:
            features['np_success_rate'] = 0
            features['np_winner_rate'] = 0
            features['np_error_rate'] = 0
            features['np_passed_rate'] = 0
            features['np_pressure_rate'] = 0

    # APPROACH SHOT ANALYSIS (Approach row)
    approach_data = player_data[player_data['row'] == 'Approach']
    if not approach_data.empty:
        app_row = approach_data.iloc[0]

        # Raw approach shot data (6 features)
        features['np_approach_attempts'] = float(app_row['net_pts'])
        features['np_approach_won'] = float(app_row['pts_won'])
        features['np_approach_winners'] = float(app_row['net_winner'])
        features['np_approach_forced'] = float(app_row['induced_forced'])
        features['np_approach_errors'] = float(app_row['net_unforced'])
        features['np_approach_passed'] = float(app_row['passed_at_net'])

        # Approach effectiveness rates (4 features)
        if features['np_approach_attempts'] > 0:
            features['np_approach_success_rate'] = (features['np_approach_won'] / features[
                'np_approach_attempts']) * 100
            features['np_approach_winner_rate'] = (features['np_approach_winners'] / features[
                'np_approach_attempts']) * 100
            features['np_approach_error_rate'] = (features['np_approach_errors'] / features[
                'np_approach_attempts']) * 100
            features['np_approach_passed_rate'] = (features['np_approach_passed'] / features[
                'np_approach_attempts']) * 100
        else:
            features['np_approach_success_rate'] = 0
            features['np_approach_winner_rate'] = 0
            features['np_approach_error_rate'] = 0
            features['np_approach_passed_rate'] = 0

    # RALLY NET POINTS ANALYSIS (NetPointsRallies vs NetPoints comparison)
    rally_netpoints_data = player_data[player_data['row'] == 'NetPointsRallies']
    if not rally_netpoints_data.empty:
        rally_np_row = rally_netpoints_data.iloc[0]

        # Rally net point raw data (3 features)
        features['np_rally_attempts'] = float(rally_np_row['net_pts'])
        features['np_rally_won'] = float(rally_np_row['pts_won'])
        features['np_rally_winners'] = float(rally_np_row['net_winner'])

        # Rally vs total comparison (2 features)
        if features.get('np_total_attempts', 0) > 0:
            features['np_rally_percentage'] = (features['np_rally_attempts'] / features['np_total_attempts']) * 100
        else:
            features['np_rally_percentage'] = 0

        if features['np_rally_attempts'] > 0:
            features['np_rally_success_rate'] = (features['np_rally_won'] / features['np_rally_attempts']) * 100
        else:
            features['np_rally_success_rate'] = 0

    # APPROACH RALLY ANALYSIS (ApproachRallies)
    rally_approach_data = player_data[player_data['row'] == 'ApproachRallies']
    if not rally_approach_data.empty:
        rally_app_row = rally_approach_data.iloc[0]

        # Rally approach raw data (2 features)
        features['np_approach_rally_attempts'] = float(rally_app_row['net_pts'])
        features['np_approach_rally_won'] = float(rally_app_row['pts_won'])

        # Rally approach effectiveness (1 feature)
        if features['np_approach_rally_attempts'] > 0:
            features['np_approach_rally_success_rate'] = (features['np_approach_rally_won'] / features[
                'np_approach_rally_attempts']) * 100
        else:
            features['np_approach_rally_success_rate'] = 0

    # TACTICAL NET GAME METRICS (6 features)
    # Net game dominance (winners vs errors)
    net_winners = features.get('np_net_winners', 0)
    net_errors = features.get('np_unforced_errors', 0)
    if net_errors > 0:
        features['np_winner_error_ratio'] = net_winners / net_errors
    else:
        features['np_winner_error_ratio'] = net_winners

    # Net aggression index (winners + pressure created)
    total_attempts = features.get('np_total_attempts', 0)
    if total_attempts > 0:
        aggression_points = net_winners + features.get('np_induced_forced', 0)
        features['np_aggression_index'] = (aggression_points / total_attempts) * 100
    else:
        features['np_aggression_index'] = 0

    # Passing shot defense effectiveness
    passed = features.get('np_passed_at_net', 0)
    if total_attempts > 0:
        features['np_passing_defense_rate'] = ((total_attempts - passed) / total_attempts) * 100
    else:
        features['np_passing_defense_rate'] = 0

    # Net positioning quality (approach conversion rate)
    approach_attempts = features.get('np_approach_attempts', 0)
    approach_won = features.get('np_approach_won', 0)
    if approach_attempts > 0:
        features['np_positioning_quality'] = (approach_won / approach_attempts) * 100
    else:
        features['np_positioning_quality'] = 0

    # Net game efficiency (points won per total shots)
    total_shots = 0
    points_won = features.get('np_points_won', 0)

    # Calculate total shots if available in any row
    for _, row in player_data.iterrows():
        if pd.notna(row['total_shots']) and row['total_shots'] > total_shots:
            total_shots = float(row['total_shots'])

    if total_shots > 0 and points_won > 0:
        features['np_shot_efficiency'] = points_won / (total_shots / points_won) if points_won > 0 else 0
    else:
        features['np_shot_efficiency'] = 0

    # Overall net game effectiveness
    success_rate = features.get('np_success_rate', 0)
    winner_rate = features.get('np_winner_rate', 0)
    error_rate = features.get('np_error_rate', 0)
    features['np_overall_effectiveness'] = (success_rate + winner_rate - error_rate) / 2

    return features


def extract_rally_features(player_canonical, gender, jeff_data):
    """Extract rally features with robust player matching"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'rally' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['rally']
    search_name = extract_player_name_from_canonical(player_canonical)

    # Find player as server or returner
    player_as_server = df[df['server'].str.contains(search_name, case=False, na=False)]
    player_as_returner = df[df['returner'].str.contains(search_name, case=False, na=False)]

    features = {}

    # Overall Rally Performance (Total row)
    total_as_server = player_as_server[player_as_server['row'] == 'Total']
    total_as_returner = player_as_returner[player_as_returner['row'] == 'Total']

    if not total_as_server.empty:
        server_total = total_as_server.iloc[0]
        expected_cols = ['pts', 'pl1_won', 'pl1_winners', 'pl1_forced', 'pl1_unforced']

        # Safe column access
        pts = server_total.get('pts', 0) if 'pts' in server_total else 0
        pl1_won = server_total.get('pl1_won', 0) if 'pl1_won' in server_total else 0

        features['rally_serve_total_pts'] = float(pts)
        features['rally_serve_pts_won'] = float(pl1_won)
        features['rally_serve_winners'] = float(server_total.get('pl1_winners', 0))
        features['rally_serve_forced'] = float(server_total.get('pl1_forced', 0))
        features['rally_serve_unforced'] = float(server_total.get('pl1_unforced', 0))

        if pts > 0:
            features['rally_serve_win_pct'] = (features['rally_serve_pts_won'] / pts) * 100

    if not total_as_returner.empty:
        returner_total = total_as_returner.iloc[0]

        pts = returner_total.get('pts', 0) if 'pts' in returner_total else 0
        pl2_won = returner_total.get('pl2_won', 0) if 'pl2_won' in returner_total else 0

        features['rally_return_total_pts'] = float(pts)
        features['rally_return_pts_won'] = float(pl2_won)
        features['rally_return_winners'] = float(returner_total.get('pl2_winners', 0))
        features['rally_return_forced'] = float(returner_total.get('pl2_forced', 0))
        features['rally_return_unforced'] = float(returner_total.get('pl2_unforced', 0))

        if pts > 0:
            features['rally_return_win_pct'] = (features['rally_return_pts_won'] / pts) * 100

    # Rally length categories with safe access
    rally_categories = ['1-3', '4-6', '7-9', '10']
    rally_names = ['short', 'medium', 'long', 'very_long']

    for i, (category, name) in enumerate(zip(rally_categories, rally_names)):
        serve_data = player_as_server[player_as_server['row'] == category]
        if not serve_data.empty:
            serve_row = serve_data.iloc[0]
            pts = serve_row.get('pts', 0) if 'pts' in serve_row else 0
            pl1_won = serve_row.get('pl1_won', 0) if 'pl1_won' in serve_row else 0

            features[f'rally_serve_{name}_pts'] = float(pts)
            features[f'rally_serve_{name}_won'] = float(pl1_won)
            features[f'rally_serve_{name}_winners'] = float(serve_row.get('pl1_winners', 0))

            if pts > 0:
                features[f'rally_serve_{name}_win_pct'] = (pl1_won / pts) * 100

    return features

def extract_serve_direction_features(player_canonical, gender, jeff_data):
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'serve_direction' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['serve_direction']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    total_data = player_data[player_data['row'] == 'Total']
    if total_data.empty:
        return {}

    totals = total_data[['deuce_wide', 'deuce_middle', 'deuce_t', 'ad_wide', 'ad_middle', 'ad_t', 'err_net', 'err_wide',
                         'err_deep']].sum()

    features = {}
    total_serves = totals['deuce_wide'] + totals['deuce_middle'] + totals['deuce_t'] + totals['ad_wide'] + totals[
        'ad_middle'] + totals['ad_t']
    total_errors = totals['err_net'] + totals['err_wide'] + totals['err_deep']

    if total_serves > 0:
        features['sd_deuce_wide_pct'] = totals['deuce_wide'] / total_serves
        features['sd_deuce_middle_pct'] = totals['deuce_middle'] / total_serves
        features['sd_deuce_t_pct'] = totals['deuce_t'] / total_serves
        features['sd_ad_wide_pct'] = totals['ad_wide'] / total_serves
        features['sd_ad_middle_pct'] = totals['ad_middle'] / total_serves
        features['sd_ad_t_pct'] = totals['ad_t'] / total_serves
        features['sd_error_rate'] = total_errors / total_serves if total_serves > 0 else 0
        features['sd_deuce_side_pct'] = (totals['deuce_wide'] + totals['deuce_middle'] + totals[
            'deuce_t']) / total_serves
        features['sd_ad_side_pct'] = (totals['ad_wide'] + totals['ad_middle'] + totals['ad_t']) / total_serves

    return features


def extract_return_outcomes_features(player_canonical, gender, jeff_data):
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'return_outcomes' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['return_outcomes']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # BASIC RETURN STATS (from Total row)
    total_data = player_data[player_data['row'] == 'Total']
    if not total_data.empty:
        totals = total_data[['pts', 'pts_won', 'returnable', 'returnable_won',
                             'in_play', 'in_play_won', 'winners', 'total_shots']].sum()

        return_pts = totals['pts']
        if return_pts > 0:
            # Raw data features (8)
            features['ro_return_pts'] = float(return_pts)
            features['ro_pts_won'] = float(totals['pts_won'])
            features['ro_returnable'] = float(totals['returnable'])
            features['ro_returnable_won'] = float(totals['returnable_won'])
            features['ro_in_play'] = float(totals['in_play'])
            features['ro_in_play_won'] = float(totals['in_play_won'])
            features['ro_winners'] = float(totals['winners'])
            features['ro_total_shots'] = float(totals['total_shots'])

            # Calculated percentages (7)
            features['ro_return_win_pct'] = features['ro_pts_won'] / return_pts
            features['ro_returnable_pct'] = features['ro_returnable'] / return_pts
            features['ro_returnable_win_pct'] = features['ro_returnable_won'] / features['ro_returnable'] if features[
                                                                                                                 'ro_returnable'] > 0 else 0
            features['ro_in_play_pct'] = features['ro_in_play'] / return_pts
            features['ro_in_play_win_pct'] = features['ro_in_play_won'] / features['ro_in_play'] if features[
                                                                                                        'ro_in_play'] > 0 else 0
            features['ro_winner_pct'] = features['ro_winners'] / return_pts
            features['ro_shots_per_point'] = features['ro_total_shots'] / return_pts

    # SERVE-SPECIFIC RETURNS (v1st vs v2nd)
    first_serve_data = player_data[player_data['row'] == 'v1st']
    second_serve_data = player_data[player_data['row'] == 'v2nd']

    if not first_serve_data.empty:
        first_totals = first_serve_data[['pts', 'pts_won']].sum()
        if first_totals['pts'] > 0:
            features['ro_first_serve_win_pct'] = first_totals['pts_won'] / first_totals['pts']

    if not second_serve_data.empty:
        second_totals = second_serve_data[['pts', 'pts_won']].sum()
        if second_totals['pts'] > 0:
            features['ro_second_serve_win_pct'] = second_totals['pts_won'] / second_totals['pts']

    # Compare first vs second serve return effectiveness
    if 'ro_first_serve_win_pct' in features and 'ro_second_serve_win_pct' in features:
        features['ro_second_serve_advantage'] = features['ro_second_serve_win_pct'] - features['ro_first_serve_win_pct']
        features['ro_first_vs_second_diff'] = abs(features['ro_second_serve_advantage'])

    # SHOT TYPE BREAKDOWN (fh vs bh, gs vs sl)
    fh_data = player_data[player_data['row'] == 'fh']
    bh_data = player_data[player_data['row'] == 'bh']
    gs_data = player_data[player_data['row'] == 'gs']
    sl_data = player_data[player_data['row'] == 'sl']

    if not fh_data.empty:
        fh_totals = fh_data[['pts', 'pts_won']].sum()
        if fh_totals['pts'] > 0:
            features['ro_fh_win_pct'] = fh_totals['pts_won'] / fh_totals['pts']

    if not bh_data.empty:
        bh_totals = bh_data[['pts', 'pts_won']].sum()
        if bh_totals['pts'] > 0:
            features['ro_bh_win_pct'] = bh_totals['pts_won'] / bh_totals['pts']

    if 'ro_fh_win_pct' in features and 'ro_bh_win_pct' in features:
        features['ro_fh_vs_bh_diff'] = abs(features['ro_fh_win_pct'] - features['ro_bh_win_pct'])

    if not gs_data.empty and not sl_data.empty:
        gs_totals = gs_data[['pts', 'pts_won']].sum()
        sl_totals = sl_data[['pts', 'pts_won']].sum()
        total_shot_types = gs_totals['pts'] + sl_totals['pts']
        if total_shot_types > 0:
            features['ro_groundstroke_pct'] = gs_totals['pts'] / total_shot_types

    # COURT POSITION (D vs A)
    deuce_data = player_data[player_data['row'] == 'D']
    ad_data = player_data[player_data['row'] == 'A']

    if not deuce_data.empty:
        deuce_totals = deuce_data[['pts', 'pts_won']].sum()
        if deuce_totals['pts'] > 0:
            features['ro_deuce_win_pct'] = deuce_totals['pts_won'] / deuce_totals['pts']

    if not ad_data.empty:
        ad_totals = ad_data[['pts', 'pts_won']].sum()
        if ad_totals['pts'] > 0:
            features['ro_ad_win_pct'] = ad_totals['pts_won'] / ad_totals['pts']

    if 'ro_deuce_win_pct' in features and 'ro_ad_win_pct' in features:
        features['ro_court_balance'] = abs(features['ro_deuce_win_pct'] - features['ro_ad_win_pct'])

    # DIRECTION-SPECIFIC (4=wide, 5=body, 6=T)
    wide_data = player_data[player_data['row'] == '4']
    body_data = player_data[player_data['row'] == '5']
    t_data = player_data[player_data['row'] == '6']

    direction_wins = []

    if not wide_data.empty:
        wide_totals = wide_data[['pts', 'pts_won']].sum()
        if wide_totals['pts'] > 0:
            features['ro_wide_win_pct'] = wide_totals['pts_won'] / wide_totals['pts']
            direction_wins.append(features['ro_wide_win_pct'])

    if not body_data.empty:
        body_totals = body_data[['pts', 'pts_won']].sum()
        if body_totals['pts'] > 0:
            features['ro_body_win_pct'] = body_totals['pts_won'] / body_totals['pts']
            direction_wins.append(features['ro_body_win_pct'])

    if not t_data.empty:
        t_totals = t_data[['pts', 'pts_won']].sum()
        if t_totals['pts'] > 0:
            features['ro_t_win_pct'] = t_totals['pts_won'] / t_totals['pts']
            direction_wins.append(features['ro_t_win_pct'])

    # Direction variety (how consistent across directions)
    if len(direction_wins) >= 2:
        features['ro_direction_variety'] = 1 - (max(direction_wins) - min(direction_wins))

    # ADVANCED METRICS
    if 'ro_returnable_pct' in features and 'ro_in_play_pct' in features:
        features['ro_return_depth_ability'] = features['ro_in_play_pct'] / features['ro_returnable_pct'] if features[
                                                                                                                'ro_returnable_pct'] > 0 else 0

    if 'ro_winner_pct' in features and 'ro_return_win_pct' in features:
        features['ro_return_aggression'] = features['ro_winner_pct'] / features['ro_return_win_pct'] if features[
                                                                                                            'ro_return_win_pct'] > 0 else 0

    return features


def extract_return_depth_features(player_canonical, gender, jeff_data):
    gender_key = 'men' if gender == 'M' else 'women'
    if gender_key not in jeff_data or 'return_depth' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['return_depth']
    player_data = find_player_data(df, player_canonical)
    if player_data.empty:
        return {}

    features = {}

    # Basic depth stats (from Total row)
    total_data = player_data[player_data['row'] == 'Total']
    if not total_data.empty:
        expected_cols = ['returnable', 'shallow', 'deep', 'very_deep', 'unforced', 'err_net', 'err_deep', 'err_wide', 'err_wide_deep']
        safe_data = safe_column_access(total_data, expected_cols)
        totals = safe_data.sum()

        returnable = totals['returnable']
        if returnable > 0:
            features['rd_returnable'] = float(returnable)
            features['rd_shallow'] = float(totals['shallow'])
            features['rd_deep'] = float(totals['deep'])
            features['rd_very_deep'] = float(totals['very_deep'])
            features['rd_unforced'] = float(totals['unforced'])
            features['rd_err_net'] = float(totals['err_net'])
            features['rd_err_deep'] = float(totals['err_deep'])
            features['rd_err_wide'] = float(totals['err_wide'])
            features['rd_err_wide_deep'] = float(totals['err_wide_deep'])

            features['rd_shallow_pct'] = features['rd_shallow'] / returnable
            features['rd_deep_pct'] = features['rd_deep'] / returnable
            features['rd_very_deep_pct'] = features['rd_very_deep'] / returnable

            total_errors = features['rd_unforced'] + features['rd_err_net'] + features['rd_err_deep'] + features['rd_err_wide'] + features['rd_err_wide_deep']
            if total_errors > 0:
                features['rd_error_rate'] = total_errors / returnable
                features['rd_net_error_pct'] = features['rd_err_net'] / total_errors
                features['rd_depth_error_pct'] = features['rd_err_deep'] / total_errors
                features['rd_wide_error_pct'] = (features['rd_err_wide'] + features['rd_err_wide_deep']) / total_errors

            features['rd_depth_aggression'] = features['rd_very_deep_pct'] / features['rd_deep_pct'] if features['rd_deep_pct'] > 0 else 0
            features['rd_depth_variety'] = 1 - max(features['rd_shallow_pct'], features['rd_deep_pct'], features['rd_very_deep_pct'])
            features['rd_depth_control'] = (features['rd_deep_pct'] + features['rd_very_deep_pct']) - features['rd_shallow_pct']

    return features


def extract_serve_influence_features(player_canonical, gender, jeff_data):
    """Extract ServeInfluence features with fixed row access"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'serve_influence' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['serve_influence']
    player_data = find_player_data(df, player_canonical)

    if player_data.empty:
        return {}

    features = {}

    # First Serve Influence (row = 1)
    first_serve_data = player_data[player_data['row'] == 1]
    if not first_serve_data.empty:
        row = first_serve_data.iloc[0]

        pts = row.get('pts', 0) if 'pts' in row else 0
        features['si_first_serve_pts'] = float(pts)

        # Rally length win percentages
        rally_cols = ['won_1+', 'won_2+', 'won_3+', 'won_4+', 'won_5+', 'won_6+', 'won_7+', 'won_8+', 'won_9+',
                      'won_10+']
        rally_values = []

        for i, col in enumerate(rally_cols, 1):
            val = row.get(col) if col in row else None
            if pd.notna(val) and val != '-':
                try:
                    # Remove % sign and convert to decimal
                    pct_val = float(str(val).replace('%', '')) / 100 if '%' in str(val) else float(val)
                    features[f'si_first_{i}plus_win_pct'] = pct_val
                    rally_values.append(pct_val)
                except (ValueError, TypeError):
                    features[f'si_first_{i}plus_win_pct'] = 0
            else:
                features[f'si_first_{i}plus_win_pct'] = 0

        # Calculate first serve influence metrics
        if len(rally_values) >= 3:
            features['si_first_immediate_advantage'] = rally_values[0] - rally_values[2] if rally_values[2] > 0 else \
            rally_values[0] - 0.5
            features['si_first_serve_decay'] = (rally_values[0] - rally_values[-1]) / len(rally_values) if len(
                rally_values) > 1 else 0
            features['si_first_consistency'] = 1 - (max(rally_values) - min(rally_values)) if rally_values else 0

    # Second Serve Influence (row = 2)
    second_serve_data = player_data[player_data['row'] == 2]
    if not second_serve_data.empty:
        row = second_serve_data.iloc[0]

        pts = row.get('pts', 0) if 'pts' in row else 0
        features['si_second_serve_pts'] = float(pts)

        rally_values_2nd = []
        for i, col in enumerate(rally_cols, 1):
            val = row.get(col) if col in row else None
            if pd.notna(val) and val != '-':
                try:
                    pct_val = float(str(val).replace('%', '')) / 100 if '%' in str(val) else float(val)
                    features[f'si_second_{i}plus_win_pct'] = pct_val
                    rally_values_2nd.append(pct_val)
                except (ValueError, TypeError):
                    features[f'si_second_{i}plus_win_pct'] = 0
            else:
                features[f'si_second_{i}plus_win_pct'] = 0

        if len(rally_values_2nd) >= 3:
            features['si_second_immediate_advantage'] = rally_values_2nd[0] - rally_values_2nd[2] if rally_values_2nd[
                                                                                                         2] > 0 else \
            rally_values_2nd[0] - 0.5
            features['si_second_serve_decay'] = (rally_values_2nd[0] - rally_values_2nd[-1]) / len(
                rally_values_2nd) if len(rally_values_2nd) > 1 else 0
            features['si_second_consistency'] = 1 - (
                        max(rally_values_2nd) - min(rally_values_2nd)) if rally_values_2nd else 0

    # Comparative metrics
    first_1plus = features.get('si_first_1plus_win_pct', 0)
    second_1plus = features.get('si_second_1plus_win_pct', 0)
    if first_1plus > 0 and second_1plus > 0:
        features['si_serve_type_advantage'] = first_1plus - second_1plus

    return features


def extract_shot_direction_features(player_canonical, gender, jeff_data):
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'shot_direction' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['shot_direction']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # BASIC SHOT DIRECTION TOTALS (from Total row)
    total_data = player_data[player_data['row'] == 'Total']
    if not total_data.empty:
        totals = total_data[['crosscourt', 'down_middle', 'down_the_line', 'inside_out', 'inside_in']].sum()

        total_shots = totals.sum()
        if total_shots > 0:
            # Raw counts (5)
            features['shotd_crosscourt'] = float(totals['crosscourt'])
            features['shotd_down_middle'] = float(totals['down_middle'])
            features['shotd_down_line'] = float(totals['down_the_line'])
            features['shotd_inside_out'] = float(totals['inside_out'])
            features['shotd_inside_in'] = float(totals['inside_in'])
            features['shotd_total_shots'] = float(total_shots)

            # Direction percentages (5)
            features['shotd_crosscourt_pct'] = features['shotd_crosscourt'] / total_shots
            features['shotd_down_middle_pct'] = features['shotd_down_middle'] / total_shots
            features['shotd_down_line_pct'] = features['shotd_down_line'] / total_shots
            features['shotd_inside_out_pct'] = features['shotd_inside_out'] / total_shots
            features['shotd_inside_in_pct'] = features['shotd_inside_in'] / total_shots

            # Tactical groupings (3)
            features['shotd_crosscourt_dominant'] = features['shotd_crosscourt_pct']
            features['shotd_attacking_shots_pct'] = (features['shotd_down_line'] + features[
                'shotd_inside_out']) / total_shots
            features['shotd_direction_variety'] = 1 - max(features['shotd_crosscourt_pct'],
                                                          features['shotd_down_middle_pct'],
                                                          features['shotd_down_line_pct'],
                                                          features['shotd_inside_out_pct'])

    # FOREHAND DIRECTION PATTERNS (row = F)
    fh_data = player_data[player_data['row'] == 'F']
    if not fh_data.empty:
        fh_totals = fh_data[['crosscourt', 'down_middle', 'down_the_line', 'inside_out', 'inside_in']].sum()
        fh_total = fh_totals.sum()

        if fh_total > 0:
            features['shotd_fh_crosscourt_pct'] = fh_totals['crosscourt'] / fh_total
            features['shotd_fh_down_line_pct'] = fh_totals['down_the_line'] / fh_total
            features['shotd_fh_inside_out_pct'] = fh_totals['inside_out'] / fh_total
            features['shotd_fh_variety'] = 1 - max(fh_totals['crosscourt'], fh_totals['down_middle'],
                                                   fh_totals['down_the_line'], fh_totals['inside_out']) / fh_total

    # BACKHAND DIRECTION PATTERNS (row = B)
    bh_data = player_data[player_data['row'] == 'B']
    if not bh_data.empty:
        bh_totals = bh_data[['crosscourt', 'down_middle', 'down_the_line', 'inside_out', 'inside_in']].sum()
        bh_total = bh_totals.sum()

        if bh_total > 0:
            features['shotd_bh_crosscourt_pct'] = bh_totals['crosscourt'] / bh_total
            features['shotd_bh_down_line_pct'] = bh_totals['down_the_line'] / bh_total
            features['shotd_bh_inside_out_pct'] = bh_totals['inside_out'] / bh_total
            features['shotd_bh_variety'] = 1 - max(bh_totals['crosscourt'], bh_totals['down_middle'],
                                                   bh_totals['down_the_line'], bh_totals['inside_out']) / bh_total

    # SLICE DIRECTION PATTERNS (row = S)
    slice_data = player_data[player_data['row'] == 'S']
    if not slice_data.empty:
        slice_totals = slice_data[['crosscourt', 'down_middle', 'down_the_line', 'inside_out', 'inside_in']].sum()
        slice_total = slice_totals.sum()

        if slice_total > 0:
            features['shotd_slice_crosscourt_pct'] = slice_totals['crosscourt'] / slice_total
            features['shotd_slice_total'] = float(slice_total)

    # COMPARATIVE SHOT TYPE METRICS
    if 'shotd_fh_crosscourt_pct' in features and 'shotd_bh_crosscourt_pct' in features:
        features['shotd_fh_vs_bh_crosscourt_diff'] = abs(
            features['shotd_fh_crosscourt_pct'] - features['shotd_bh_crosscourt_pct'])
        features['shotd_fh_vs_bh_downline_diff'] = abs(
            features.get('shotd_fh_down_line_pct', 0) - features.get('shotd_bh_down_line_pct', 0))

        # Overall shot direction consistency
        fh_variety = features.get('shotd_fh_variety', 0)
        bh_variety = features.get('shotd_bh_variety', 0)
        features['shotd_overall_consistency'] = (fh_variety + bh_variety) / 2 if bh_variety > 0 else fh_variety

    return features


def extract_shot_dir_outcomes_features(player_canonical, gender, jeff_data):
    """FIXED: Extract 40+ features from ShotDirOutcomes with proper data aggregation"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'shot_dir_outcomes' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['shot_dir_outcomes']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # Aggregate across all matches for this player
    aggregated = player_data.groupby('row')[
        ['shots', 'pt_ending', 'winners', 'induced_forced', 'unforced', 'shots_in_pts_won', 'shots_in_pts_lost']].sum()

    # Shot type aggregations (F, B, S)
    shot_types = {'F': 'forehand', 'B': 'backhand', 'S': 'slice'}
    for shot_prefix, shot_name in shot_types.items():
        # Get all rows starting with this shot type
        shot_rows = [row for row in aggregated.index if row.startswith(f'{shot_prefix}-')]
        if shot_rows:
            shot_totals = aggregated.loc[shot_rows].sum()

            total_shots = shot_totals['shots']
            if total_shots > 0:
                features[f'sdo_{shot_name}_shots'] = int(total_shots)
                features[f'sdo_{shot_name}_winners'] = int(shot_totals['winners'])
                features[f'sdo_{shot_name}_errors'] = int(shot_totals['unforced'])
                features[f'sdo_{shot_name}_pt_ending'] = int(shot_totals['pt_ending'])

                # Effectiveness metrics
                features[f'sdo_{shot_name}_winner_rate'] = (shot_totals['winners'] / total_shots) * 100
                features[f'sdo_{shot_name}_error_rate'] = (shot_totals['unforced'] / total_shots) * 100
                features[f'sdo_{shot_name}_pt_ending_rate'] = (shot_totals['pt_ending'] / total_shots) * 100
                features[f'sdo_{shot_name}_effectiveness'] = ((shot_totals['winners'] - shot_totals[
                    'unforced']) / total_shots) * 100

                # Win efficiency
                total_outcome_shots = shot_totals['shots_in_pts_won'] + shot_totals['shots_in_pts_lost']
                if total_outcome_shots > 0:
                    features[f'sdo_{shot_name}_win_efficiency'] = (shot_totals[
                                                                       'shots_in_pts_won'] / total_outcome_shots) * 100

    # Direction aggregations (XC, DTL, DTM, IO, II)
    directions = {'XC': 'crosscourt', 'DTL': 'down_line', 'DTM': 'down_middle', 'IO': 'inside_out', 'II': 'inside_in'}
    for dir_suffix, dir_name in directions.items():
        # Get all rows ending with this direction
        dir_rows = [row for row in aggregated.index if row.endswith(f'-{dir_suffix}')]
        if dir_rows:
            dir_totals = aggregated.loc[dir_rows].sum()

            total_shots = dir_totals['shots']
            if total_shots > 0:
                features[f'sdo_{dir_name}_shots'] = int(total_shots)
                features[f'sdo_{dir_name}_winners'] = int(dir_totals['winners'])
                features[f'sdo_{dir_name}_errors'] = int(dir_totals['unforced'])

                # Effectiveness metrics
                features[f'sdo_{dir_name}_winner_rate'] = (dir_totals['winners'] / total_shots) * 100
                features[f'sdo_{dir_name}_error_rate'] = (dir_totals['unforced'] / total_shots) * 100
                features[f'sdo_{dir_name}_effectiveness'] = ((dir_totals['winners'] - dir_totals[
                    'unforced']) / total_shots) * 100

    # Specific combinations
    key_combinations = ['F-XC', 'F-DTL', 'F-IO', 'B-XC', 'B-DTL']
    for combo in key_combinations:
        if combo in aggregated.index:
            combo_data = aggregated.loc[combo]
            shots = combo_data['shots']

            if shots > 0:
                combo_key = combo.lower().replace('-', '_')
                features[f'sdo_{combo_key}_shots'] = int(shots)
                features[f'sdo_{combo_key}_winners'] = int(combo_data['winners'])
                features[f'sdo_{combo_key}_errors'] = int(combo_data['unforced'])
                features[f'sdo_{combo_key}_winner_rate'] = (combo_data['winners'] / shots) * 100
                features[f'sdo_{combo_key}_error_rate'] = (combo_data['unforced'] / shots) * 100
                features[f'sdo_{combo_key}_effectiveness'] = ((combo_data['winners'] - combo_data[
                    'unforced']) / shots) * 100

    # Overall metrics
    total_shots = aggregated['shots'].sum()
    total_winners = aggregated['winners'].sum()
    total_errors = aggregated['unforced'].sum()

    if total_shots > 0:
        features['sdo_total_shots'] = int(total_shots)
        features['sdo_total_winners'] = int(total_winners)
        features['sdo_total_errors'] = int(total_errors)
        features['sdo_overall_winner_rate'] = (total_winners / total_shots) * 100
        features['sdo_overall_error_rate'] = (total_errors / total_shots) * 100
        features['sdo_overall_effectiveness'] = ((total_winners - total_errors) / total_shots) * 100
        features['sdo_winner_error_ratio'] = total_winners / total_errors if total_errors > 0 else total_winners

    # Shot distribution
    fh_shots = features.get('sdo_forehand_shots', 0)
    bh_shots = features.get('sdo_backhand_shots', 0)
    total_groundstrokes = fh_shots + bh_shots

    if total_groundstrokes > 0:
        features['sdo_fh_shot_preference'] = (fh_shots / total_groundstrokes) * 100
        features['sdo_fh_vs_bh_effectiveness'] = features.get('sdo_forehand_effectiveness', 0) - features.get(
            'sdo_backhand_effectiveness', 0)

    # Direction preferences
    xc_shots = features.get('sdo_crosscourt_shots', 0)
    dtl_shots = features.get('sdo_down_line_shots', 0)
    dtm_shots = features.get('sdo_down_middle_shots', 0)
    io_shots = features.get('sdo_inside_out_shots', 0)

    total_directional = xc_shots + dtl_shots + dtm_shots + io_shots
    if total_directional > 0:
        features['sdo_crosscourt_preference'] = (xc_shots / total_directional) * 100
        features['sdo_attacking_shot_rate'] = ((dtl_shots + io_shots) / total_directional) * 100
        features['sdo_conservative_shot_rate'] = ((xc_shots + dtm_shots) / total_directional) * 100

    return features



def extract_shot_types_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 40+ features from ShotTypes tactical positioning data (was 1 placeholder)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'shot_types' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['shot_types']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # Total overview
    total_data = player_data[player_data['row'] == 'Total']
    if not total_data.empty:
        totals = total_data.iloc[0]
        features['st_total_shots'] = float(totals['shots'])
        features['st_total_pt_ending'] = float(totals['pt_ending'])
        features['st_total_winners'] = float(totals['winners'])
        features['st_total_induced_forced'] = float(totals['induced_forced'])
        features['st_total_unforced'] = float(totals['unforced'])
        features['st_total_serve_return'] = float(totals['serve_return'])
        features['st_total_shots_pts_won'] = float(totals['shots_in_pts_won'])
        features['st_total_shots_pts_lost'] = float(totals['shots_in_pts_lost'])

        # Calculate rates
        if features['st_total_shots'] > 0:
            features['st_pt_ending_rate'] = (features['st_total_pt_ending'] / features['st_total_shots']) * 100
            features['st_winner_rate'] = (features['st_total_winners'] / features['st_total_shots']) * 100
            features['st_error_rate'] = (features['st_total_unforced'] / features['st_total_shots']) * 100
            features['st_forced_error_rate'] = (features['st_total_induced_forced'] / features['st_total_shots']) * 100

        # Win efficiency
        total_outcome_shots = features['st_total_shots_pts_won'] + features['st_total_shots_pts_lost']
        if total_outcome_shots > 0:
            features['st_win_efficiency'] = (features['st_total_shots_pts_won'] / total_outcome_shots) * 100

        # Winner to error ratio
        total_errors = features['st_total_unforced'] + features['st_total_induced_forced']
        if total_errors > 0:
            features['st_winner_error_ratio'] = features['st_total_winners'] / total_errors
        else:
            features['st_winner_error_ratio'] = features['st_total_winners']

    # Court positioning analysis
    fside_data = player_data[player_data['row'] == 'Fside']
    bside_data = player_data[player_data['row'] == 'Bside']

    if not fside_data.empty:
        fside = fside_data.iloc[0]
        features['st_fside_shots'] = float(fside['shots'])
        features['st_fside_winners'] = float(fside['winners'])
        features['st_fside_errors'] = float(fside['unforced'])

        if features['st_fside_shots'] > 0:
            features['st_fside_winner_rate'] = (features['st_fside_winners'] / features['st_fside_shots']) * 100
            features['st_fside_error_rate'] = (features['st_fside_errors'] / features['st_fside_shots']) * 100

    if not bside_data.empty:
        bside = bside_data.iloc[0]
        features['st_bside_shots'] = float(bside['shots'])
        features['st_bside_winners'] = float(bside['winners'])
        features['st_bside_errors'] = float(bside['unforced'])

        if features['st_bside_shots'] > 0:
            features['st_bside_winner_rate'] = (features['st_bside_winners'] / features['st_bside_shots']) * 100
            features['st_bside_error_rate'] = (features['st_bside_errors'] / features['st_bside_shots']) * 100

    # Court side preference
    fside_shots = features.get('st_fside_shots', 0)
    bside_shots = features.get('st_bside_shots', 0)
    if fside_shots + bside_shots > 0:
        features['st_fside_preference'] = (fside_shots / (fside_shots + bside_shots)) * 100

        # Side effectiveness comparison
        fside_wr = features.get('st_fside_winner_rate', 0)
        bside_wr = features.get('st_bside_winner_rate', 0)
        features['st_side_winner_differential'] = fside_wr - bside_wr

    # Groundstroke analysis
    fgs_data = player_data[player_data['row'] == 'Fgs']
    bgs_data = player_data[player_data['row'] == 'Bgs']

    if not fgs_data.empty:
        fgs = fgs_data.iloc[0]
        features['st_fgs_shots'] = float(fgs['shots'])
        features['st_fgs_winners'] = float(fgs['winners'])
        if features['st_fgs_shots'] > 0:
            features['st_fgs_winner_rate'] = (features['st_fgs_winners'] / features['st_fgs_shots']) * 100

    if not bgs_data.empty:
        bgs = bgs_data.iloc[0]
        features['st_bgs_shots'] = float(bgs['shots'])
        features['st_bgs_winners'] = float(bgs['winners'])
        if features['st_bgs_shots'] > 0:
            features['st_bgs_winner_rate'] = (features['st_bgs_winners'] / features['st_bgs_shots']) * 100

    # Wing preference
    fgs_shots = features.get('st_fgs_shots', 0)
    bgs_shots = features.get('st_bgs_shots', 0)
    if fgs_shots + bgs_shots > 0:
        features['st_fh_groundstroke_pct'] = (fgs_shots / (fgs_shots + bgs_shots)) * 100

    # Tactical positioning
    base_data = player_data[player_data['row'] == 'Base']
    net_data = player_data[player_data['row'] == 'Net']

    if not base_data.empty:
        base = base_data.iloc[0]
        features['st_baseline_shots'] = float(base['shots'])
        features['st_baseline_winners'] = float(base['winners'])
        if features['st_baseline_shots'] > 0:
            features['st_baseline_winner_rate'] = (features['st_baseline_winners'] / features[
                'st_baseline_shots']) * 100

    if not net_data.empty:
        net = net_data.iloc[0]
        features['st_net_shots'] = float(net['shots'])
        features['st_net_winners'] = float(net['winners'])
        if features['st_net_shots'] > 0:
            features['st_net_winner_rate'] = (features['st_net_winners'] / features['st_net_shots']) * 100

    # Court position preference
    baseline_shots = features.get('st_baseline_shots', 0)
    net_shots = features.get('st_net_shots', 0)
    if baseline_shots + net_shots > 0:
        features['st_net_approach_rate'] = (net_shots / (baseline_shots + net_shots)) * 100

    # Specific shot types
    shot_type_rows = ['Sl', 'Dr', 'Sw', 'Gs']
    for shot_type in shot_type_rows:
        shot_data = player_data[player_data['row'] == shot_type]
        if not shot_data.empty:
            shot = shot_data.iloc[0]
            features[f'st_{shot_type.lower()}_shots'] = float(shot['shots'])
            features[f'st_{shot_type.lower()}_winners'] = float(shot['winners'])

            if float(shot['shots']) > 0:
                features[f'st_{shot_type.lower()}_winner_rate'] = (float(shot['winners']) / float(shot['shots'])) * 100

    # Wing-specific analysis (F/B rows)
    f_data = player_data[player_data['row'] == 'F']
    b_data = player_data[player_data['row'] == 'B']

    if not f_data.empty:
        f_row = f_data.iloc[0]
        features['st_f_shots'] = float(f_row['shots'])
        features['st_f_winners'] = float(f_row['winners'])
        features['st_f_errors'] = float(f_row['unforced'])

        if features['st_f_shots'] > 0:
            features['st_f_effectiveness'] = ((features['st_f_winners'] - features['st_f_errors']) / features[
                'st_f_shots']) * 100

    if not b_data.empty:
        b_row = b_data.iloc[0]
        features['st_b_shots'] = float(b_row['shots'])
        features['st_b_winners'] = float(b_row['winners'])
        features['st_b_errors'] = float(b_row['unforced'])

        if features['st_b_shots'] > 0:
            features['st_b_effectiveness'] = ((features['st_b_winners'] - features['st_b_errors']) / features[
                'st_b_shots']) * 100

    # Wing effectiveness comparison
    f_eff = features.get('st_f_effectiveness', 0)
    b_eff = features.get('st_b_effectiveness', 0)
    features['st_wing_effectiveness_diff'] = f_eff - b_eff

    # Coded shot analysis (R, S, U, Y, J)
    coded_shots = ['R', 'S', 'U', 'Y', 'J']
    coded_total = 0

    for shot_code in coded_shots:
        shot_data = player_data[player_data['row'] == shot_code]
        if not shot_data.empty:
            shot = shot_data.iloc[0]
            shot_count = float(shot['shots'])
            features[f'st_{shot_code.lower()}_shots'] = shot_count
            coded_total += shot_count

    features['st_coded_shots_total'] = coded_total

    # Overall tactical style metrics
    total_shots = features.get('st_total_shots', 0)
    if total_shots > 0:
        features['st_aggressive_positioning'] = features.get('st_net_approach_rate', 0)

        slice_shots = features.get('st_sl_shots', 0)
        features['st_slice_usage_rate'] = (slice_shots / total_shots) * 100

        drop_shots = features.get('st_dr_shots', 0)
        features['st_drop_shot_rate'] = (drop_shots / total_shots) * 100

    return features


def extract_snv_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 35+ features from SnV (Serve & Volley) data (was 1 placeholder)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'snv' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['snv']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # SnV (Serve & Volley) analysis
    snv_data = player_data[player_data['row'] == 'SnV']
    if not snv_data.empty:
        snv = snv_data.iloc[0]
        features['snv_total_pts'] = float(snv['snv_pts'])
        features['snv_pts_won'] = float(snv['pts_won'])
        features['snv_aces'] = float(snv['aces'])
        features['snv_unret'] = float(snv['unret'])
        features['snv_return_forced'] = float(snv['return_forced'])
        features['snv_net_winner'] = float(snv['net_winner'])
        features['snv_induced_forced'] = float(snv['induced_forced'])
        features['snv_net_unforced'] = float(snv['net_unforced'])
        features['snv_passed_at_net'] = float(snv['passed_at_net'])
        features['snv_passing_shot_induced_forced'] = float(snv['passing_shot_induced_forced'])
        features['snv_total_shots'] = float(snv['total_shots'])

        # Calculate SnV success rates
        if features['snv_total_pts'] > 0:
            features['snv_success_rate'] = (features['snv_pts_won'] / features['snv_total_pts']) * 100
            features['snv_ace_rate'] = (features['snv_aces'] / features['snv_total_pts']) * 100
            features['snv_unreturned_rate'] = (features['snv_unret'] / features['snv_total_pts']) * 100
            features['snv_net_winner_rate'] = (features['snv_net_winner'] / features['snv_total_pts']) * 100
            features['snv_passed_rate'] = (features['snv_passed_at_net'] / features['snv_total_pts']) * 100
            features['snv_error_rate'] = (features['snv_net_unforced'] / features['snv_total_pts']) * 100

        # Net effectiveness
        net_attempts = features['snv_net_winner'] + features['snv_net_unforced'] + features['snv_passed_at_net']
        if net_attempts > 0:
            features['snv_net_effectiveness'] = (features['snv_net_winner'] / net_attempts) * 100

        # SnV efficiency
        if features['snv_total_shots'] > 0:
            features['snv_shots_per_point'] = features['snv_total_shots'] / features['snv_total_pts']

    # SnV1st (First serve S&V) analysis
    snv1st_data = player_data[player_data['row'] == 'SnV1st']
    if not snv1st_data.empty:
        snv1st = snv1st_data.iloc[0]
        features['snv1st_pts'] = float(snv1st['snv_pts'])
        features['snv1st_pts_won'] = float(snv1st['pts_won'])
        features['snv1st_aces'] = float(snv1st['aces'])
        features['snv1st_net_winner'] = float(snv1st['net_winner'])
        features['snv1st_passed'] = float(snv1st['passed_at_net'])
        features['snv1st_net_unforced'] = float(snv1st['net_unforced'])

        if features['snv1st_pts'] > 0:
            features['snv1st_success_rate'] = (features['snv1st_pts_won'] / features['snv1st_pts']) * 100
            features['snv1st_ace_rate'] = (features['snv1st_aces'] / features['snv1st_pts']) * 100
            features['snv1st_net_winner_rate'] = (features['snv1st_net_winner'] / features['snv1st_pts']) * 100

    # nonSnV (Baseline play) analysis
    nonsnv_data = player_data[player_data['row'] == 'nonSnV']
    if not nonsnv_data.empty:
        nonsnv = nonsnv_data.iloc[0]
        features['baseline_total_pts'] = float(nonsnv['snv_pts'])
        features['baseline_pts_won'] = float(nonsnv['pts_won'])
        features['baseline_aces'] = float(nonsnv['aces'])
        features['baseline_unret'] = float(nonsnv['unret'])
        features['baseline_return_forced'] = float(nonsnv['return_forced'])
        features['baseline_net_winner'] = float(nonsnv['net_winner'])
        features['baseline_net_unforced'] = float(nonsnv['net_unforced'])
        features['baseline_passed_at_net'] = float(nonsnv['passed_at_net'])
        features['baseline_total_shots'] = float(nonsnv['total_shots'])

        # Calculate baseline success rates
        if features['baseline_total_pts'] > 0:
            features['baseline_success_rate'] = (features['baseline_pts_won'] / features['baseline_total_pts']) * 100
            features['baseline_ace_rate'] = (features['baseline_aces'] / features['baseline_total_pts']) * 100
            features['baseline_net_winner_rate'] = (features['baseline_net_winner'] / features[
                'baseline_total_pts']) * 100

        # Baseline efficiency
        if features['baseline_total_shots'] > 0:
            features['baseline_shots_per_point'] = features['baseline_total_shots'] / features['baseline_total_pts']

    # nonSnV1st (Baseline first serve) analysis
    nonsnv1st_data = player_data[player_data['row'] == 'nonSnV1st']
    if not nonsnv1st_data.empty:
        nonsnv1st = nonsnv1st_data.iloc[0]
        features['baseline_1st_pts'] = float(nonsnv1st['snv_pts'])
        features['baseline_1st_won'] = float(nonsnv1st['pts_won'])
        features['baseline_1st_aces'] = float(nonsnv1st['aces'])

        if features['baseline_1st_pts'] > 0:
            features['baseline_1st_success_rate'] = (features['baseline_1st_won'] / features['baseline_1st_pts']) * 100
            features['baseline_1st_ace_rate'] = (features['baseline_1st_aces'] / features['baseline_1st_pts']) * 100

    # nonSnV2nd (Baseline second serve) analysis
    nonsnv2nd_data = player_data[player_data['row'] == 'nonSnV2nd']
    if not nonsnv2nd_data.empty:
        nonsnv2nd = nonsnv2nd_data.iloc[0]
        features['baseline_2nd_pts'] = float(nonsnv2nd['snv_pts'])
        features['baseline_2nd_won'] = float(nonsnv2nd['pts_won'])

        if features['baseline_2nd_pts'] > 0:
            features['baseline_2nd_success_rate'] = (features['baseline_2nd_won'] / features['baseline_2nd_pts']) * 100

    # Tactical comparisons
    snv_pts = features.get('snv_total_pts', 0)
    baseline_pts = features.get('baseline_total_pts', 0)
    total_serve_pts = snv_pts + baseline_pts

    if total_serve_pts > 0:
        # SnV frequency
        features['snv_frequency'] = (snv_pts / total_serve_pts) * 100

        # Style classification
        snv_freq = features['snv_frequency']
        if snv_freq > 20:
            features['snv_style'] = 3  # Frequent S&V
        elif snv_freq > 5:
            features['snv_style'] = 2  # Occasional S&V
        else:
            features['snv_style'] = 1  # Rare S&V

    # Success rate comparisons
    snv_sr = features.get('snv_success_rate', 0)
    baseline_sr = features.get('baseline_success_rate', 0)
    if snv_sr > 0 and baseline_sr > 0:
        features['snv_vs_baseline_advantage'] = snv_sr - baseline_sr
        features['snv_effectiveness_ratio'] = snv_sr / baseline_sr

    # First serve comparisons
    snv1st_sr = features.get('snv1st_success_rate', 0)
    baseline1st_sr = features.get('baseline_1st_success_rate', 0)
    if snv1st_sr > 0 and baseline1st_sr > 0:
        features['snv_1st_vs_baseline_1st'] = snv1st_sr - baseline1st_sr

    # Net approach analysis
    snv_net_approaches = features.get('snv_net_winner', 0) + features.get('snv_passed_at_net', 0) + features.get(
        'snv_net_unforced', 0)
    baseline_net_approaches = features.get('baseline_net_winner', 0) + features.get('baseline_passed_at_net',
                                                                                    0) + features.get(
        'baseline_net_unforced', 0)

    if snv_pts > 0:
        features['snv_net_approach_rate'] = (snv_net_approaches / snv_pts) * 100
    if baseline_pts > 0:
        features['baseline_net_approach_rate'] = (baseline_net_approaches / baseline_pts) * 100

    # Risk assessment
    if snv_pts > 0:
        # High-reward outcomes
        snv_quick_wins = features.get('snv_aces', 0) + features.get('snv_unret', 0) + features.get('snv_net_winner', 0)
        features['snv_quick_win_rate'] = (snv_quick_wins / snv_pts) * 100

        # High-risk failures
        snv_failures = features.get('snv_passed_at_net', 0) + features.get('snv_net_unforced', 0)
        features['snv_failure_rate'] = (snv_failures / snv_pts) * 100

        # Risk-reward ratio
        if snv_failures > 0:
            features['snv_risk_reward'] = snv_quick_wins / snv_failures
        else:
            features['snv_risk_reward'] = snv_quick_wins

    # Shot economy comparison
    snv_shots_pp = features.get('snv_shots_per_point', 0)
    baseline_shots_pp = features.get('baseline_shots_per_point', 0)
    if snv_shots_pp > 0 and baseline_shots_pp > 0:
        features['snv_shot_economy'] = baseline_shots_pp - snv_shots_pp  # Positive = SnV more efficient

    return features


def extract_sv_break_split_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 40+ features from SvBreakSplit serve performance by game score (was 1 placeholder)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'sv_break_split' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['sv_break_split']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # Score situation analysis
    score_situations = ['d', 'a', '4', '5', '6']

    for situation in score_situations:
        situation_data = player_data[player_data['row'] == situation]
        if not situation_data.empty:
            sit = situation_data.iloc[0]

            # First serve metrics
            features[f'svbs_{situation}_1st_pts'] = float(sit['first_pts'])
            features[f'svbs_{situation}_1st_won'] = float(sit['first_pts_won'])
            features[f'svbs_{situation}_1st_aces'] = float(sit['first_aces'])
            features[f'svbs_{situation}_1st_unret'] = float(sit['first_unret'])
            features[f'svbs_{situation}_1st_forced'] = float(sit['first_forced'])
            features[f'svbs_{situation}_1st_quick'] = float(sit['first_won_lte_3_shots'])

            # Second serve metrics
            features[f'svbs_{situation}_2nd_pts'] = float(sit['second_pts'])
            features[f'svbs_{situation}_2nd_won'] = float(sit['second_pts_won'])
            features[f'svbs_{situation}_2nd_aces'] = float(sit['second_aces'])
            features[f'svbs_{situation}_2nd_unret'] = float(sit['second_unret'])
            features[f'svbs_{situation}_2nd_forced'] = float(sit['second_forced'])
            features[f'svbs_{situation}_2nd_quick'] = float(sit['second_won_lte_3_shots'])

            # Calculate success rates
            if features[f'svbs_{situation}_1st_pts'] > 0:
                features[f'svbs_{situation}_1st_success_rate'] = (features[f'svbs_{situation}_1st_won'] / features[
                    f'svbs_{situation}_1st_pts']) * 100
                features[f'svbs_{situation}_1st_ace_rate'] = (features[f'svbs_{situation}_1st_aces'] / features[
                    f'svbs_{situation}_1st_pts']) * 100
                features[f'svbs_{situation}_1st_quick_rate'] = (features[f'svbs_{situation}_1st_quick'] / features[
                    f'svbs_{situation}_1st_pts']) * 100

            if features[f'svbs_{situation}_2nd_pts'] > 0:
                features[f'svbs_{situation}_2nd_success_rate'] = (features[f'svbs_{situation}_2nd_won'] / features[
                    f'svbs_{situation}_2nd_pts']) * 100
                features[f'svbs_{situation}_2nd_quick_rate'] = (features[f'svbs_{situation}_2nd_quick'] / features[
                    f'svbs_{situation}_2nd_pts']) * 100

            # Combined serve effectiveness
            total_pts = features[f'svbs_{situation}_1st_pts'] + features[f'svbs_{situation}_2nd_pts']
            total_won = features[f'svbs_{situation}_1st_won'] + features[f'svbs_{situation}_2nd_won']
            if total_pts > 0:
                features[f'svbs_{situation}_overall_success_rate'] = (total_won / total_pts) * 100

    # Detailed score+situation combinations
    detailed_situations = ['4d', '4a', '5d', '5a', '6d', '6a']

    for situation in detailed_situations:
        situation_data = player_data[player_data['row'] == situation]
        if not situation_data.empty:
            sit = situation_data.iloc[0]

            # Key metrics for detailed situations
            first_pts = float(sit['first_pts'])
            first_won = float(sit['first_pts_won'])
            second_pts = float(sit['second_pts'])
            second_won = float(sit['second_pts_won'])

            features[f'svbs_{situation}_total_pts'] = first_pts + second_pts
            features[f'svbs_{situation}_total_won'] = first_won + second_won

            if features[f'svbs_{situation}_total_pts'] > 0:
                features[f'svbs_{situation}_success_rate'] = (features[f'svbs_{situation}_total_won'] / features[
                    f'svbs_{situation}_total_pts']) * 100

    # Pressure situation analysis
    # Deuce performance
    deuce_sr = features.get('svbs_d_overall_success_rate', 0)
    features['svbs_deuce_performance'] = deuce_sr

    # Advantage performance
    adv_sr = features.get('svbs_a_overall_success_rate', 0)
    features['svbs_advantage_performance'] = adv_sr

    # Pressure differential (advantage vs deuce)
    if deuce_sr > 0 and adv_sr > 0:
        features['svbs_pressure_differential'] = adv_sr - deuce_sr

    # Score situation effectiveness
    score_4_sr = features.get('svbs_4_overall_success_rate', 0)
    score_5_sr = features.get('svbs_5_overall_success_rate', 0)
    score_6_sr = features.get('svbs_6_overall_success_rate', 0)

    # Game point situations (40-0, 40-15, 40-30)
    game_point_situations = [score_4_sr, score_5_sr, score_6_sr]
    valid_gp_situations = [x for x in game_point_situations if x > 0]

    if valid_gp_situations:
        features['svbs_game_point_avg_performance'] = sum(valid_gp_situations) / len(valid_gp_situations)
        features['svbs_game_point_consistency'] = min(valid_gp_situations) / max(valid_gp_situations) if max(
            valid_gp_situations) > 0 else 0

    # First vs second serve comparison across situations
    first_serve_rates = []
    second_serve_rates = []

    for situation in ['d', 'a', '4', '5', '6']:
        first_rate = features.get(f'svbs_{situation}_1st_success_rate', 0)
        second_rate = features.get(f'svbs_{situation}_2nd_success_rate', 0)

        if first_rate > 0:
            first_serve_rates.append(first_rate)
        if second_rate > 0:
            second_serve_rates.append(second_rate)

    if first_serve_rates:
        features['svbs_1st_serve_avg_performance'] = sum(first_serve_rates) / len(first_serve_rates)
        features['svbs_1st_serve_consistency'] = min(first_serve_rates) / max(first_serve_rates) if max(
            first_serve_rates) > 0 else 0

    if second_serve_rates:
        features['svbs_2nd_serve_avg_performance'] = sum(second_serve_rates) / len(second_serve_rates)
        features['svbs_2nd_serve_consistency'] = min(second_serve_rates) / max(second_serve_rates) if max(
            second_serve_rates) > 0 else 0

    # Serve type effectiveness gap
    if first_serve_rates and second_serve_rates:
        avg_first = sum(first_serve_rates) / len(first_serve_rates)
        avg_second = sum(second_serve_rates) / len(second_serve_rates)
        features['svbs_serve_type_gap'] = avg_first - avg_second

    # Quick point analysis (≤3 shots)
    quick_situations = []
    for situation in ['d', 'a', '4', '5', '6']:
        first_quick_rate = features.get(f'svbs_{situation}_1st_quick_rate', 0)
        second_quick_rate = features.get(f'svbs_{situation}_2nd_quick_rate', 0)

        if first_quick_rate > 0 or second_quick_rate > 0:
            combined_rate = (first_quick_rate + second_quick_rate) / (
                2 if first_quick_rate > 0 and second_quick_rate > 0 else 1)
            quick_situations.append(combined_rate)

    if quick_situations:
        features['svbs_quick_point_ability'] = sum(quick_situations) / len(quick_situations)

    # Clutch performance assessment
    clutch_situations = ['d', 'a', '6']  # Deuce, advantage, 40-30
    clutch_rates = []

    for situation in clutch_situations:
        rate = features.get(f'svbs_{situation}_overall_success_rate', 0)
        if rate > 0:
            clutch_rates.append(rate)

    if clutch_rates:
        features['svbs_clutch_performance'] = sum(clutch_rates) / len(clutch_rates)

        # Clutch vs non-clutch comparison
        non_clutch_rates = []
        for situation in ['4', '5']:  # 40-0, 40-15 (easier situations)
            rate = features.get(f'svbs_{situation}_overall_success_rate', 0)
            if rate > 0:
                non_clutch_rates.append(rate)

        if non_clutch_rates:
            avg_non_clutch = sum(non_clutch_rates) / len(non_clutch_rates)
            avg_clutch = sum(clutch_rates) / len(clutch_rates)
            features['svbs_clutch_vs_easy'] = avg_clutch - avg_non_clutch

    return features


def extract_sv_break_total_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 35+ features from SvBreakTotal aggregate serve performance by game score (was 1 placeholder)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'sv_break_total' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['sv_break_total']
    player_data = df[df['Player_canonical'] == player_canonical]

    if player_data.empty:
        return {}

    features = {}

    # Score situation analysis
    score_situations = ['d', 'a', '4', '5', '6']

    for situation in score_situations:
        situation_data = player_data[player_data['row'] == situation]
        if not situation_data.empty:
            sit = situation_data.iloc[0]

            # Raw serve metrics
            features[f'svbt_{situation}_pts'] = float(sit['pts'])
            features[f'svbt_{situation}_pts_won'] = float(sit['pts_won'])
            features[f'svbt_{situation}_aces'] = float(sit['aces'])
            features[f'svbt_{situation}_unret'] = float(sit['unret'])
            features[f'svbt_{situation}_forced_err'] = float(sit['forced_err'])
            features[f'svbt_{situation}_quick_pts'] = float(sit['pts_won_lte_3_shots'])
            features[f'svbt_{situation}_first_in'] = float(sit['first_in'])
            features[f'svbt_{situation}_dfs'] = float(sit['dfs'])

            # Calculate success rates
            if features[f'svbt_{situation}_pts'] > 0:
                features[f'svbt_{situation}_success_rate'] = (features[f'svbt_{situation}_pts_won'] / features[
                    f'svbt_{situation}_pts']) * 100
                features[f'svbt_{situation}_ace_rate'] = (features[f'svbt_{situation}_aces'] / features[
                    f'svbt_{situation}_pts']) * 100
                features[f'svbt_{situation}_unreturned_rate'] = (features[f'svbt_{situation}_unret'] / features[
                    f'svbt_{situation}_pts']) * 100
                features[f'svbt_{situation}_forced_error_rate'] = (features[f'svbt_{situation}_forced_err'] / features[
                    f'svbt_{situation}_pts']) * 100
                features[f'svbt_{situation}_quick_point_rate'] = (features[f'svbt_{situation}_quick_pts'] / features[
                    f'svbt_{situation}_pts']) * 100
                features[f'svbt_{situation}_df_rate'] = (features[f'svbt_{situation}_dfs'] / features[
                    f'svbt_{situation}_pts']) * 100

            # First serve percentage
            if features[f'svbt_{situation}_pts'] > 0:
                features[f'svbt_{situation}_first_serve_pct'] = (features[f'svbt_{situation}_first_in'] / features[
                    f'svbt_{situation}_pts']) * 100

            # Serve dominance (aces + unreturned + forced errors)
            serve_dominance = features[f'svbt_{situation}_aces'] + features[f'svbt_{situation}_unret'] + features[
                f'svbt_{situation}_forced_err']
            if features[f'svbt_{situation}_pts'] > 0:
                features[f'svbt_{situation}_dominance_rate'] = (serve_dominance / features[
                    f'svbt_{situation}_pts']) * 100

    # Detailed score+situation combinations
    detailed_situations = ['4d', '4a', '5d', '5a', '6d', '6a']

    for situation in detailed_situations:
        situation_data = player_data[player_data['row'] == situation]
        if not situation_data.empty:
            sit = situation_data.iloc[0]

            pts = float(sit['pts'])
            pts_won = float(sit['pts_won'])

            features[f'svbt_{situation}_pts'] = pts
            features[f'svbt_{situation}_pts_won'] = pts_won

            if pts > 0:
                features[f'svbt_{situation}_success_rate'] = (pts_won / pts) * 100

    # Pressure situation analysis
    deuce_sr = features.get('svbt_d_success_rate', 0)
    adv_sr = features.get('svbt_a_success_rate', 0)

    features['svbt_deuce_performance'] = deuce_sr
    features['svbt_advantage_performance'] = adv_sr

    if deuce_sr > 0 and adv_sr > 0:
        features['svbt_pressure_differential'] = adv_sr - deuce_sr

    # Game score effectiveness
    score_4_sr = features.get('svbt_4_success_rate', 0)
    score_5_sr = features.get('svbt_5_success_rate', 0)
    score_6_sr = features.get('svbt_6_success_rate', 0)

    game_scores = [score_4_sr, score_5_sr, score_6_sr]
    valid_scores = [x for x in game_scores if x > 0]

    if valid_scores:
        features['svbt_game_score_avg_performance'] = sum(valid_scores) / len(valid_scores)
        features['svbt_game_score_consistency'] = min(valid_scores) / max(valid_scores) if max(valid_scores) > 0 else 0

    # Ace performance across situations
    ace_rates = []
    for situation in ['d', 'a', '4', '5', '6']:
        rate = features.get(f'svbt_{situation}_ace_rate', 0)
        if rate > 0:
            ace_rates.append(rate)

    if ace_rates:
        features['svbt_overall_ace_consistency'] = min(ace_rates) / max(ace_rates) if max(ace_rates) > 0 else 0
        features['svbt_avg_ace_rate'] = sum(ace_rates) / len(ace_rates)

    # First serve consistency
    first_serve_pcts = []
    for situation in ['d', 'a', '4', '5', '6']:
        pct = features.get(f'svbt_{situation}_first_serve_pct', 0)
        if pct > 0:
            first_serve_pcts.append(pct)

    if first_serve_pcts:
        features['svbt_first_serve_consistency'] = min(first_serve_pcts) / max(first_serve_pcts) if max(
            first_serve_pcts) > 0 else 0
        features['svbt_avg_first_serve_pct'] = sum(first_serve_pcts) / len(first_serve_pcts)

    # Double fault performance
    df_rates = []
    for situation in ['d', 'a', '4', '5', '6']:
        rate = features.get(f'svbt_{situation}_df_rate', 0)
        df_rates.append(rate)  # Include 0s for DF analysis

    if df_rates:
        features['svbt_avg_df_rate'] = sum(df_rates) / len(df_rates)
        features['svbt_df_pressure_impact'] = max(df_rates) - min(df_rates)  # Higher = more pressure-sensitive

    # Quick point ability
    quick_rates = []
    for situation in ['d', 'a', '4', '5', '6']:
        rate = features.get(f'svbt_{situation}_quick_point_rate', 0)
        if rate > 0:
            quick_rates.append(rate)

    if quick_rates:
        features['svbt_avg_quick_point_rate'] = sum(quick_rates) / len(quick_rates)
        features['svbt_quick_point_consistency'] = min(quick_rates) / max(quick_rates) if max(quick_rates) > 0 else 0

    # Serve dominance analysis
    dominance_rates = []
    for situation in ['d', 'a', '4', '5', '6']:
        rate = features.get(f'svbt_{situation}_dominance_rate', 0)
        if rate > 0:
            dominance_rates.append(rate)

    if dominance_rates:
        features['svbt_avg_dominance_rate'] = sum(dominance_rates) / len(dominance_rates)
        features['svbt_dominance_consistency'] = min(dominance_rates) / max(dominance_rates) if max(
            dominance_rates) > 0 else 0

    # Clutch vs easy situation comparison
    clutch_situations = ['d', 'a', '6']  # Deuce, advantage, 40-30
    easy_situations = ['4', '5']  # 40-0, 40-15

    clutch_rates = [features.get(f'svbt_{s}_success_rate', 0) for s in clutch_situations if
                    features.get(f'svbt_{s}_success_rate', 0) > 0]
    easy_rates = [features.get(f'svbt_{s}_success_rate', 0) for s in easy_situations if
                  features.get(f'svbt_{s}_success_rate', 0) > 0]

    if clutch_rates and easy_rates:
        avg_clutch = sum(clutch_rates) / len(clutch_rates)
        avg_easy = sum(easy_rates) / len(easy_rates)
        features['svbt_clutch_vs_easy_differential'] = avg_clutch - avg_easy

        features['svbt_clutch_performance'] = avg_clutch
        features['svbt_easy_situation_performance'] = avg_easy

    # Detailed situation effectiveness (4d, 4a, etc.)
    detailed_scores = {}
    for situation in detailed_situations:
        rate = features.get(f'svbt_{situation}_success_rate', 0)
        if rate > 0:
            detailed_scores[situation] = rate

    if detailed_scores:
        features['svbt_detailed_situation_avg'] = sum(detailed_scores.values()) / len(detailed_scores)

        # Deuce vs advantage in specific scores
        deuce_detailed = [detailed_scores.get(f'{score}d', 0) for score in ['4', '5', '6']]
        adv_detailed = [detailed_scores.get(f'{score}a', 0) for score in ['4', '5', '6']]

        valid_deuce = [x for x in deuce_detailed if x > 0]
        valid_adv = [x for x in adv_detailed if x > 0]

        if valid_deuce and valid_adv:
            avg_deuce_detailed = sum(valid_deuce) / len(valid_deuce)
            avg_adv_detailed = sum(valid_adv) / len(valid_adv)
            features['svbt_detailed_deuce_vs_adv'] = avg_adv_detailed - avg_deuce_detailed

    # Overall effectiveness metrics
    all_success_rates = []
    for situation in score_situations:
        rate = features.get(f'svbt_{situation}_success_rate', 0)
        if rate > 0:
            all_success_rates.append(rate)

    if all_success_rates:
        features['svbt_overall_avg_performance'] = sum(all_success_rates) / len(all_success_rates)
        features['svbt_overall_consistency'] = min(all_success_rates) / max(all_success_rates) if max(
            all_success_rates) > 0 else 0
        features['svbt_performance_range'] = max(all_success_rates) - min(all_success_rates)

    return features


def extract_matches_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 45+ features from Matches metadata (was 1 placeholder)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'matches' not in jeff_data[gender_key]:
        return {}

    matches_df = jeff_data[gender_key]['matches']

    # Find matches where player participated
    player_matches = matches_df[
        (matches_df['Player 1'].str.contains(player_canonical.replace('_', ' '), case=False, na=False)) |
        (matches_df['Player 2'].str.contains(player_canonical.replace('_', ' '), case=False, na=False))
        ]

    if player_matches.empty:
        return {}

    features = {}

    # === BASIC MATCH VOLUME ===
    features['matches_total_matches'] = len(player_matches)
    features['matches_unique_tournaments'] = player_matches['Tournament'].nunique()
    features['matches_unique_opponents'] = len(set(
        list(player_matches['Player 1'].unique()) + list(player_matches['Player 2'].unique())
    )) - 1  # Subtract self

    # === SURFACE ANALYSIS ===
    surface_counts = player_matches['Surface'].value_counts()
    total_surface_matches = len(player_matches[player_matches['Surface'].notna()])

    for surface in ['Clay', 'Hard', 'Grass']:
        count = surface_counts.get(surface, 0)
        features[f'matches_{surface.lower()}_matches'] = count
        features[f'matches_{surface.lower()}_pct'] = count / total_surface_matches if total_surface_matches > 0 else 0

    features['matches_surface_variety'] = player_matches['Surface'].nunique()
    features['matches_most_common_surface'] = surface_counts.index[0] if len(surface_counts) > 0 else 'Unknown'

    # === TOURNAMENT ANALYSIS ===
    # Tournament tiers based on names
    grand_slams = ['Roland Garros', 'Wimbledon', 'US Open', 'Australian Open']
    masters_keywords = ['Masters', 'WTA 1000', 'ATP Masters']

    tournament_analysis = player_matches['Tournament'].str.lower()

    features['matches_grand_slam_count'] = sum(
        tournament_analysis.str.contains(gs.lower(), na=False).sum()
        for gs in grand_slams
    )

    features['matches_masters_count'] = sum(
        tournament_analysis.str.contains(keyword.lower(), na=False).sum()
        for keyword in masters_keywords
    )

    # ITF/Challenger level
    features['matches_itf_count'] = tournament_analysis.str.contains('itf', na=False).sum()
    features['matches_challenger_count'] = tournament_analysis.str.contains('challenger', na=False).sum()

    # === ROUND ANALYSIS ===
    round_counts = player_matches['Round'].value_counts()

    # Deep run analysis
    deep_rounds = ['F', 'SF', 'QF']
    features['matches_finals'] = round_counts.get('F', 0)
    features['matches_semifinals'] = round_counts.get('SF', 0)
    features['matches_quarterfinals'] = round_counts.get('QF', 0)
    features['matches_deep_runs'] = sum(round_counts.get(r, 0) for r in deep_rounds)

    # Early rounds
    early_rounds = ['R128', 'R64', 'R32', 'R16']
    features['matches_early_rounds'] = sum(round_counts.get(r, 0) for r in early_rounds)

    if len(player_matches) > 0:
        features['matches_deep_run_pct'] = features['matches_deep_runs'] / len(player_matches)
        features['matches_final_reach_pct'] = features['matches_finals'] / len(player_matches)
    else:
        features['matches_deep_run_pct'] = 0
        features['matches_final_reach_pct'] = 0

    # === HANDEDNESS ANALYSIS ===
    # Determine player's handedness
    player_1_mask = player_matches['Player 1'].str.contains(player_canonical.replace('_', ' '), case=False, na=False)
    player_2_mask = player_matches['Player 2'].str.contains(player_canonical.replace('_', ' '), case=False, na=False)

    player_handedness = None
    if player_1_mask.any():
        player_handedness = player_matches[player_1_mask]['Pl 1 hand'].iloc[0]
    elif player_2_mask.any():
        player_handedness = player_matches[player_2_mask]['Pl 2 hand'].iloc[0]

    features['matches_player_handedness'] = player_handedness

    # Opponent handedness patterns
    opponent_handedness = []
    for _, match in player_matches.iterrows():
        if pd.notna(match['Player 1']) and player_canonical.replace('_', ' ').lower() in match['Player 1'].lower():
            if pd.notna(match['Pl 2 hand']):
                opponent_handedness.append(match['Pl 2 hand'])
        elif pd.notna(match['Player 2']) and player_canonical.replace('_', ' ').lower() in match['Player 2'].lower():
            if pd.notna(match['Pl 1 hand']):
                opponent_handedness.append(match['Pl 1 hand'])

    if opponent_handedness:
        righty_opponents = opponent_handedness.count('R')
        lefty_opponents = opponent_handedness.count('L')

        features['matches_vs_righties'] = righty_opponents
        features['matches_vs_lefties'] = lefty_opponents
        features['matches_vs_righties_pct'] = righty_opponents / len(opponent_handedness)
        features['matches_vs_lefties_pct'] = lefty_opponents / len(opponent_handedness)
        features['matches_handedness_variety'] = len(set(opponent_handedness))
    else:
        features['matches_vs_righties'] = 0
        features['matches_vs_lefties'] = 0
        features['matches_vs_righties_pct'] = 0
        features['matches_vs_lefties_pct'] = 0
        features['matches_handedness_variety'] = 0

    # === MATCH FORMAT ANALYSIS ===
    best_of_counts = player_matches['Best of'].value_counts()
    features['matches_best_of_3'] = best_of_counts.get(3, 0)
    features['matches_best_of_5'] = best_of_counts.get(5, 0)

    if len(player_matches) > 0:
        features['matches_best_of_3_pct'] = features['matches_best_of_3'] / len(player_matches)
        features['matches_best_of_5_pct'] = features['matches_best_of_5'] / len(player_matches)
    else:
        features['matches_best_of_3_pct'] = 0
        features['matches_best_of_5_pct'] = 0

    # Tiebreak analysis
    final_tb_counts = player_matches['Final TB?'].value_counts()
    features['matches_final_tb_advantage'] = final_tb_counts.get('A', 0)  # Advantage sets
    features['matches_final_tb_tiebreak'] = final_tb_counts.get('1', 0)  # Tiebreak sets

    # === TEMPORAL ANALYSIS ===
    if 'Date' in player_matches.columns:
        player_matches['Date'] = pd.to_datetime(player_matches['Date'], format='%Y%m%d', errors='coerce')
        valid_dates = player_matches[player_matches['Date'].notna()]

        if len(valid_dates) > 0:
            # Date range
            features['matches_date_span_days'] = (valid_dates['Date'].max() - valid_dates['Date'].min()).days
            features['matches_most_recent_year'] = valid_dates['Date'].max().year
            features['matches_earliest_year'] = valid_dates['Date'].min().year

            # Seasonal patterns
            valid_dates['month'] = valid_dates['Date'].dt.month
            monthly_counts = valid_dates['month'].value_counts()

            # Season definitions (Northern Hemisphere)
            spring_months = [3, 4, 5]
            summer_months = [6, 7, 8]
            fall_months = [9, 10, 11]
            winter_months = [12, 1, 2]

            features['matches_spring_count'] = sum(monthly_counts.get(m, 0) for m in spring_months)
            features['matches_summer_count'] = sum(monthly_counts.get(m, 0) for m in summer_months)
            features['matches_fall_count'] = sum(monthly_counts.get(m, 0) for m in fall_months)
            features['matches_winter_count'] = sum(monthly_counts.get(m, 0) for m in winter_months)

            total_dated = len(valid_dates)
            features['matches_spring_pct'] = features['matches_spring_count'] / total_dated
            features['matches_summer_pct'] = features['matches_summer_count'] / total_dated
            features['matches_fall_pct'] = features['matches_fall_count'] / total_dated
            features['matches_winter_pct'] = features['matches_winter_count'] / total_dated

    # === CHARTING QUALITY ===
    charting_counts = player_matches['Charted by'].value_counts()
    features['matches_unique_charters'] = len(charting_counts)
    features['matches_most_frequent_charter'] = charting_counts.index[0] if len(charting_counts) > 0 else 'Unknown'
    features['matches_charter_consistency'] = charting_counts.max() / len(player_matches) if len(
        player_matches) > 0 else 0

    # === DERIVED METRICS ===
    # Tournament level consistency
    if features['matches_total_matches'] > 0:
        features['matches_avg_tournaments_per_match'] = features['matches_unique_tournaments'] / features[
            'matches_total_matches']
        features['matches_tournament_loyalty'] = 1 - (
                    features['matches_unique_tournaments'] / features['matches_total_matches'])
    else:
        features['matches_avg_tournaments_per_match'] = 0
        features['matches_tournament_loyalty'] = 0

    # Experience diversity index
    surface_diversity = features['matches_surface_variety'] / 3  # Max 3 surfaces
    round_diversity = len(round_counts) / 8  # Approximate max rounds
    tournament_diversity = min(features['matches_unique_tournaments'] / 10, 1)  # Cap at 1

    features['matches_experience_diversity'] = (surface_diversity + round_diversity + tournament_diversity) / 3

    return features


def extract_overview_features(player_canonical, gender, jeff_data):
    """EXPANDED: Extract 25+ features from Overview match statistics (was missing entirely)"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'overview' not in jeff_data[gender_key]:
        return {}

    df = jeff_data[gender_key]['overview']

    # Map canonical name back to full name for matching
    # djokovic_n -> look for "Djokovic" in player names
    if player_canonical == 'djokovic_n':
        player_data = df[df['player'].str.contains('Djokovic', case=False, na=False)]
    elif player_canonical == 'sinner_j':
        player_data = df[df['player'].str.contains('Sinner', case=False, na=False)]
    elif player_canonical == 'alcaraz_c':
        player_data = df[df['player'].str.contains('Alcaraz', case=False, na=False)]
    elif player_canonical == 'sabalenka_a':
        player_data = df[df['player'].str.contains('Sabalenka', case=False, na=False)]
    else:
        # Generic mapping: extract surname from canonical name
        surname = player_canonical.split('_')[0].capitalize()
        player_data = df[df['player'].str.contains(surname, case=False, na=False)]

    if player_data.empty:
        return {}

    # Get only Total rows (aggregate across all matches)
    total_data = player_data[player_data['set'] == 'Total']
    if total_data.empty:
        return {}

    # Sum across all Total rows for this player
    aggregated = total_data[['serve_pts', 'aces', 'dfs', 'first_in', 'first_won', 'second_won',
                             'bk_pts', 'bp_saved', 'return_pts', 'return_pts_won', 'winners',
                             'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']].sum()

    features = {}

    # Serve statistics
    serve_pts = aggregated['serve_pts']
    if serve_pts > 0:
        features['ov_serve_pts'] = int(serve_pts)
        features['ov_aces'] = int(aggregated['aces'])
        features['ov_double_faults'] = int(aggregated['dfs'])
        features['ov_first_in'] = int(aggregated['first_in'])
        features['ov_first_won'] = int(aggregated['first_won'])
        features['ov_second_won'] = int(aggregated['second_won'])

        # Serve percentages
        features['ov_ace_rate'] = (aggregated['aces'] / serve_pts) * 100
        features['ov_df_rate'] = (aggregated['dfs'] / serve_pts) * 100
        features['ov_first_serve_pct'] = (aggregated['first_in'] / serve_pts) * 100
        features['ov_first_serve_win_pct'] = (aggregated['first_won'] / aggregated['first_in']) * 100 if aggregated[
                                                                                                             'first_in'] > 0 else 0

        second_serves = serve_pts - aggregated['first_in']
        features['ov_second_serve_win_pct'] = (aggregated[
                                                   'second_won'] / second_serves) * 100 if second_serves > 0 else 0
        features['ov_serve_dominance'] = ((aggregated['aces'] + aggregated['first_won'] + aggregated[
            'second_won']) / serve_pts) * 100

    # Return statistics
    return_pts = aggregated['return_pts']
    if return_pts > 0:
        features['ov_return_pts'] = int(return_pts)
        features['ov_return_pts_won'] = int(aggregated['return_pts_won'])
        features['ov_return_win_pct'] = (aggregated['return_pts_won'] / return_pts) * 100

    # Winners and errors
    total_winners = aggregated['winners']
    total_errors = aggregated['unforced']

    features['ov_winners_total'] = int(total_winners)
    features['ov_winners_fh'] = int(aggregated['winners_fh'])
    features['ov_winners_bh'] = int(aggregated['winners_bh'])
    features['ov_unforced_total'] = int(total_errors)
    features['ov_unforced_fh'] = int(aggregated['unforced_fh'])
    features['ov_unforced_bh'] = int(aggregated['unforced_bh'])

    # Shot distribution
    if total_winners > 0:
        features['ov_fh_winner_pct'] = (aggregated['winners_fh'] / total_winners) * 100
        features['ov_bh_winner_pct'] = (aggregated['winners_bh'] / total_winners) * 100

    if total_errors > 0:
        features['ov_fh_error_pct'] = (aggregated['unforced_fh'] / total_errors) * 100
        features['ov_bh_error_pct'] = (aggregated['unforced_bh'] / total_errors) * 100

    # Effectiveness ratios
    if total_errors > 0:
        features['ov_winner_error_ratio'] = total_winners / total_errors

    if aggregated['unforced_fh'] > 0:
        features['ov_fh_effectiveness'] = aggregated['winners_fh'] / aggregated['unforced_fh']

    if aggregated['unforced_bh'] > 0:
        features['ov_bh_effectiveness'] = aggregated['winners_bh'] / aggregated['unforced_bh']

    # Break point performance (using correct column name bk_pts)
    bk_pts = aggregated['bk_pts']  # break points faced
    if bk_pts > 0:
        features['ov_bp_saved'] = int(aggregated['bp_saved'])
        features['ov_bp_faced'] = int(bk_pts)
        features['ov_bp_save_pct'] = (aggregated['bp_saved'] / bk_pts) * 100

    # Overall tactical metrics
    total_points = serve_pts + return_pts
    if total_points > 0:
        total_points_won = aggregated['first_won'] + aggregated['second_won'] + aggregated['return_pts_won']
        features['ov_total_points'] = int(total_points)
        features['ov_total_points_won'] = int(total_points_won)
        features['ov_point_win_pct'] = (total_points_won / total_points) * 100

    return features


# Test the fixed Overview function
jeff_data = load_jeff_comprehensive_data()

for player_canonical, gender, name in [('djokovic_n', 'M', 'Djokovic'), ('sinner_j', 'M', 'Sinner'),
                                       ('alcaraz_c', 'M', 'Alcaraz')]:
    print(f"--- {name} Overview ---")
    features = extract_overview_features(player_canonical, gender, jeff_data)
    print(f"Features extracted: {len(features)}")

    if features:
        print(f"Total points: {features.get('ov_total_points', 0)}")
        print(f"Ace rate: {features.get('ov_ace_rate', 0):.1f}%")
        print(f"First serve %: {features.get('ov_first_serve_pct', 0):.1f}%")
        print(f"Return win %: {features.get('ov_return_win_pct', 0):.1f}%")
        print(f"Winner/Error ratio: {features.get('ov_winner_error_ratio', 0):.2f}")
        print(f"Break point save %: {features.get('ov_bp_save_pct', 0):.1f}%")
        print(f"Point win %: {features.get('ov_point_win_pct', 0):.1f}%")

        print(f"All features: {sorted(features.keys())}")
    else:
        print("No Overview data found")
    print()


def extract_comprehensive_jeff_features(player_canonical, gender, jeff_data, weighted_defaults=None):
    """Enhanced feature extraction with ALL Jeff data files and robust error handling"""
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data:
        return get_fallback_defaults(gender_key)

    if weighted_defaults and gender_key in weighted_defaults:
        features = weighted_defaults[gender_key].copy()
    else:
        features = get_fallback_defaults(gender_key)

    # Extract from Overview if available
    if 'overview' in jeff_data[gender_key]:
        try:
            overview_df = jeff_data[gender_key]['overview']
            player_overview = find_player_data(overview_df, player_canonical)

            if not player_overview.empty:
                total_overview = player_overview[player_overview['set'] == 'Total']
                if not total_overview.empty:
                    latest = total_overview.iloc[-1]
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
        except Exception as e:
            pass  # Continue with other extractions

    # Extract from all specialized files
    extraction_functions = [
        extract_serve_basics_features,
        extract_key_points_serve_features,
        extract_key_points_return_features,
        extract_net_points_features,
        extract_rally_features,
        extract_serve_direction_features,
        extract_return_outcomes_features,
        extract_return_depth_features,
        extract_serve_influence_features,
        extract_shot_direction_features,
        extract_shot_dir_outcomes_features,
        extract_shot_types_features,
        extract_snv_features,
        extract_sv_break_split_features,
        extract_sv_break_total_features,
        extract_matches_features,
        extract_overview_features,
        extract_jeff_notation_features,
    ]

    for func in extraction_functions:
        try:
            new_features = func(player_canonical, gender, jeff_data)
            features.update(new_features)
        except Exception as e:
            # Log error but continue with other extractions
            continue

    # Calculate derived features
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


#________________________________#

import sys
import os
import traceback
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_individual_functions():
    """Test each extraction function individually"""

    print("=== TESTING INDIVIDUAL EXTRACTION FUNCTIONS ===\n")

    try:
        from tennis_updated import load_jeff_comprehensive_data, canonical_player

        # Import new extraction functions
        from jeff_extraction_functions import (
            extract_serve_basics_features,
            extract_key_points_serve_features,
            extract_key_points_return_features,
            extract_net_points_features,
            extract_rally_features,
            extract_serve_direction_features,
            extract_return_outcomes_features,
            extract_return_depth_features,
            extract_serve_influence_features,
            extract_shot_direction_features,
            extract_shot_dir_outcomes_features,
            extract_shot_types_features,
            extract_snv_features,
            extract_sv_break_split_features,
            extract_sv_break_total_features,
            extract_matches_features
        )

        print("✅ All imports successful")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    print("Loading Jeff data...")
    jeff_data = load_jeff_comprehensive_data()
    print(f"✅ Jeff data loaded successfully")

    # Test functions
    functions_to_test = [
        ('ServeBasics', extract_serve_basics_features),
        ('KeyPointsServe', extract_key_points_serve_features),
        ('KeyPointsReturn', extract_key_points_return_features),
        ('NetPoints', extract_net_points_features),
        ('Rally', extract_rally_features),
        ('ServeDirection', extract_serve_direction_features),
        ('ReturnOutcomes', extract_return_outcomes_features),
        ('ReturnDepth', extract_return_depth_features),
        ('ServeInfluence', extract_serve_influence_features),
        ('ShotDirection', extract_shot_direction_features),
        ('ShotDirOutcomes', extract_shot_dir_outcomes_features),
        ('ShotTypes', extract_shot_types_features),
        ('ServeAndVolley', extract_snv_features),
        ('SvBreakSplit', extract_sv_break_split_features),
        ('SvBreakTotal', extract_sv_break_total_features),
        ('Matches', extract_matches_features)
    ]

    # Test players
    test_players = [
        ('djokovic_n', 'M'),
        ('federer_r', 'M'),
        ('serena_w', 'W')
    ]

    results = defaultdict(dict)

    for name, func in functions_to_test:
        print(f"\n--- Testing {name} ---")

        for player, gender in test_players:
            try:
                features = func(player, gender, jeff_data)
                results[name][player] = len(features)

                if features:
                    sample_keys = list(features.keys())[:3]
                    sample_values = [features[k] for k in sample_keys]
                    print(f"  {player}: {len(features)} features - {dict(zip(sample_keys, sample_values))}")
                else:
                    print(f"  {player}: No features extracted (expected for some players)")

            except Exception as e:
                print(f"  ❌ {player}: Error - {e}")
                results[name][player] = -1

    # Summary
    print(f"\n=== INDIVIDUAL FUNCTION TEST SUMMARY ===")

    total_functions = len(functions_to_test)
    successful_functions = 0

    for name, func in functions_to_test:
        player_results = results[name]
        success_count = sum(1 for v in player_results.values() if v >= 0)
        total_features = sum(v for v in player_results.values() if v > 0)

        if success_count == len(test_players):
            print(f"✅ {name}: All players successful, {total_features} total features")
            successful_functions += 1
        else:
            print(f"❌ {name}: {success_count}/{len(test_players)} players successful")

    print(f"\n✅ {successful_functions}/{total_functions} functions working correctly")
    return successful_functions == total_functions


def test_integration():
    """Test integration with main pipeline"""

    print("\n=== TESTING PIPELINE INTEGRATION ===\n")

    try:
        from tennis_updated import (
            load_jeff_comprehensive_data,
            extract_comprehensive_jeff_features,
            canonical_player
        )

        print("✅ Main pipeline imports successful")

    except ImportError as e:
        print(f"❌ Pipeline import error: {e}")
        return False

    jeff_data = load_jeff_comprehensive_data()

    # Test integrated extraction
    test_players = [
        ('djokovic_n', 'M'),
        ('federer_r', 'M'),
        ('nadal_r', 'M'),
        ('serena_w', 'W')
    ]

    print("Testing integrated extraction...")

    for player, gender in test_players:
        try:
            # Test original function (should include new extractions)
            features = extract_comprehensive_jeff_features(player, gender, jeff_data)

            # Count features by category
            categories = {
                'sb_': 'ServeBasics',
                'kps_': 'KeyPointsServe',
                'kpr_': 'KeyPointsReturn',
                'np_': 'NetPoints',
                'rally_': 'Rally',
                'sd_': 'ServeDirection',
                'ro_': 'ReturnOutcomes',
                'rd_': 'ReturnDepth',
                'si_': 'ServeInfluence',
                'shotd_': 'ShotDirection',
                'sdo_': 'ShotDirOutcomes',
                'st_': 'ShotTypes',
                'snv_': 'ServeAndVolley',
                'svbs_': 'SvBreakSplit',
                'svbt_': 'SvBreakTotal',
                'matches_': 'Matches',
                'jeff_': 'JeffNotation'
            }

            print(f"\n{player} ({gender}): {len(features)} total features")

            category_counts = {}
            for prefix, name in categories.items():
                count = len([k for k in features.keys() if k.startswith(prefix)])
                if count > 0:
                    category_counts[name] = count
                    print(f"  {name}: {count}")

            # Verify we have features from multiple categories
            if len(category_counts) >= 5:
                print(f"  ✅ Multiple categories extracted successfully")
            else:
                print(f"  ⚠️  Only {len(category_counts)} categories with features")

        except Exception as e:
            print(f"❌ {player}: Integration error - {e}")
            traceback.print_exc()
            return False

    print(f"\n✅ Pipeline integration test successful")
    return True


def test_data_quality():
    """Test data quality and validation"""

    print("\n=== TESTING DATA QUALITY ===\n")

    try:
        from tennis_updated import load_jeff_comprehensive_data
        from jeff_extraction_functions import extract_all_jeff_features

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    jeff_data = load_jeff_comprehensive_data()

    # Test data quality
    test_player = 'djokovic_n'
    features = extract_all_jeff_features(test_player, 'M', jeff_data)

    print(f"Testing data quality for {test_player}...")

    # Check for reasonable values
    quality_checks = [
        ('sb_first_serve_pct', 0.0, 1.0, 'First serve percentage'),
        ('sb_ace_rate', 0.0, 0.5, 'Ace rate'),
        ('kps_bp_save_pct', 0.0, 1.0, 'Break point save percentage'),
        ('np_net_win_pct', 0.0, 1.0, 'Net win percentage'),
        ('rally_short_pct', 0.0, 1.0, 'Short rally percentage'),
        ('sd_placement_variety', 0.0, 1.0, 'Serve placement variety'),
        ('ro_return_effectiveness', 0.0, 1.0, 'Return effectiveness'),
        ('st_forehand_pct', 0.0, 1.0, 'Forehand percentage')
    ]

    passed_checks = 0
    total_checks = 0

    for feature_key, min_val, max_val, description in quality_checks:
        if feature_key in features:
            value = features[feature_key]
            total_checks += 1

            if min_val <= value <= max_val:
                print(f"  ✅ {description}: {value:.3f} (valid range)")
                passed_checks += 1
            else:
                print(f"  ❌ {description}: {value:.3f} (outside range {min_val}-{max_val})")
        else:
            print(f"  ⚠️  {description}: Not found in features")

    # Check for NaN values
    nan_count = sum(1 for v in features.values() if str(v).lower() == 'nan')
    if nan_count == 0:
        print(f"  ✅ No NaN values found")
    else:
        print(f"  ⚠️  {nan_count} NaN values found")

    print(f"\n✅ Data quality: {passed_checks}/{total_checks} checks passed")
    return passed_checks >= total_checks * 0.8  # 80% success rate


def test_performance():
    """Test extraction performance"""

    print("\n=== TESTING PERFORMANCE ===\n")

    import time

    try:
        from tennis_updated import load_jeff_comprehensive_data
        from jeff_extraction_functions import extract_all_jeff_features

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    print("Loading data...")
    start_time = time.time()
    jeff_data = load_jeff_comprehensive_data()
    load_time = time.time() - start_time
    print(f"✅ Data loading: {load_time:.2f} seconds")

    # Test extraction speed
    test_player = 'djokovic_n'
    iterations = 10

    print(f"Testing extraction speed ({iterations} iterations)...")
    start_time = time.time()

    for i in range(iterations):
        features = extract_all_jeff_features(test_player, 'M', jeff_data)

    total_time = time.time() - start_time
    avg_time = total_time / iterations

    print(f"✅ Average extraction time: {avg_time:.3f} seconds per player")
    print(f"✅ Estimated time for 1000 players: {avg_time * 1000 / 60:.1f} minutes")

    # Memory usage estimation
    import sys
    feature_size = sys.getsizeof(features)
    print(f"✅ Feature dictionary size: {feature_size} bytes")
    print(f"✅ Estimated memory for 1000 players: {feature_size * 1000 / 1024 / 1024:.1f} MB")

    return avg_time < 1.0  # Should be under 1 second per player


def main():
    """Run all tests"""

    print("JEFF SACKMANN EXTRACTION FUNCTION TEST SUITE")
    print("=" * 50)

    test_results = []

    # Run tests
    tests = [
        ("Individual Functions", test_individual_functions),
        ("Pipeline Integration", test_integration),
        ("Data Quality", test_data_quality),
        ("Performance", test_performance)
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")

        try:
            result = test_func()
            test_results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            traceback.print_exc()
            test_results.append((test_name, False))

    # Final summary
    print(f"\n{'=' * 20} FINAL RESULTS {'=' * 20}")

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Ready for production deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Review errors before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

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
# JEFF NOTATION PARSER - Extract features from point-by-point shot sequences
# ============================================================================

class JeffNotationParser:
    """Parse Jeff Sackmann's shot notation to extract tactical features"""
    
    def __init__(self):
        # Shot direction mappings
        self.directions = {
            '1': 'wide_deuce', '2': 'wide_ad', '3': 'down_line', 
            '4': 'wide', '5': 'body', '6': 'center', 
            '7': 'inside_out', '8': 'crosscourt', '9': 'down_middle'
        }
        
        # Shot type mappings  
        self.shot_types = {
            'f': 'forehand', 'b': 'backhand', 'v': 'volley',
            'o': 'overhead', 'l': 'lob', 'd': 'drop', 
            'h': 'half_volley', 'z': 'slice'
        }
        
        # Shot outcomes
        self.outcomes = {
            '*': 'winner', '#': 'error', '@': 'forced_error',
            '!': 'ace', '=': 'let', '+': 'net'
        }

    def parse_point(self, notation):
        """Parse a single point notation into structured data"""
        if pd.isna(notation) or not notation:
            return {}
            
        notation = str(notation).strip()
        shots = []
        serve_info = {}
        
        # First character is usually serve direction
        if notation and notation[0].isdigit():
            serve_info['serve_direction'] = self.directions.get(notation[0], 'unknown')
            notation = notation[1:]
        
        # Parse remaining shots
        i = 0
        shot_count = 0
        while i < len(notation):
            char = notation[i]
            
            # Shot type
            if char.lower() in self.shot_types:
                shot = {
                    'shot_num': shot_count,
                    'shot_type': self.shot_types[char.lower()],
                    'direction': None,
                    'outcome': None
                }
                
                # Look ahead for direction
                if i + 1 < len(notation) and notation[i + 1].isdigit():
                    shot['direction'] = self.directions.get(notation[i + 1], 'unknown')
                    i += 1
                
                # Look ahead for outcome
                if i + 1 < len(notation) and notation[i + 1] in self.outcomes:
                    shot['outcome'] = self.outcomes[notation[i + 1]]
                    i += 1
                    
                shots.append(shot)
                shot_count += 1
                
            # Direct outcome (ace, error, etc.)
            elif char in self.outcomes:
                if shots:
                    shots[-1]['outcome'] = self.outcomes[char]
                else:
                    serve_info['serve_outcome'] = self.outcomes[char]
            
            i += 1
        
        return {
            'serve_info': serve_info,
            'shots': shots,
            'rally_length': len(shots),
            'point_notation': str(notation)
        }

    def extract_player_patterns(self, player_points):
        """Extract tactical patterns from all points for a player"""
        if not player_points or len(player_points) == 0:
            return self._get_default_patterns()
            
        patterns = {
            'total_points': len(player_points),
            'serve_patterns': {},
            'return_patterns': {},
            'rally_patterns': {},
            'pressure_patterns': {},
            'shot_distribution': {}
        }
        
        serve_points = []
        return_points = []
        
        for point_data in player_points:
            parsed = self.parse_point(point_data.get('notation', ''))
            
            if not parsed:
                continue
                
            # Categorize by serve/return
            if point_data.get('serving', True):  # Default assume serving
                serve_points.append(parsed)
            else:
                return_points.append(parsed)
        
        # Extract serve patterns
        patterns['serve_patterns'] = self._extract_serve_patterns(serve_points)
        
        # Extract return patterns  
        patterns['return_patterns'] = self._extract_return_patterns(return_points)
        
        # Extract rally patterns from all points
        all_parsed = [self.parse_point(p.get('notation', '')) for p in player_points]
        patterns['rally_patterns'] = self._extract_rally_patterns(all_parsed)
        
        # Extract pressure performance
        patterns['pressure_patterns'] = self._extract_pressure_patterns(player_points)
        
        return patterns

    def _extract_serve_patterns(self, serve_points):
        """Extract serving patterns"""
        if not serve_points:
            return self._get_default_serve_patterns()
            
        patterns = {
            'total_serves': len(serve_points),
            'direction_distribution': {},
            'ace_rate': 0,
            'service_winner_rate': 0,
            'avg_rally_length': 0
        }
        
        directions = []
        aces = 0
        service_winners = 0
        rally_lengths = []
        
        for point in serve_points:
            serve_info = point.get('serve_info', {})
            direction = serve_info.get('serve_direction')
            if direction:
                directions.append(direction)
                
            outcome = serve_info.get('serve_outcome')
            if outcome == 'ace':
                aces += 1
            elif outcome == 'winner':
                service_winners += 1
                
            rally_lengths.append(point.get('rally_length', 0))
        
        # Calculate distributions
        if directions:
            from collections import Counter
            dir_counts = Counter(directions)
            total = len(directions)
            patterns['direction_distribution'] = {
                k: v / total for k, v in dir_counts.items()
            }
        
        patterns['ace_rate'] = aces / len(serve_points) if serve_points else 0
        patterns['service_winner_rate'] = service_winners / len(serve_points) if serve_points else 0
        patterns['avg_rally_length'] = np.mean(rally_lengths) if rally_lengths else 0
        
        return patterns

    def _extract_return_patterns(self, return_points):
        """Extract return patterns"""
        if not return_points:
            return self._get_default_return_patterns()
            
        patterns = {
            'total_returns': len(return_points),
            'return_winner_rate': 0,
            'return_error_rate': 0,
            'avg_rally_length': 0,
            'shot_type_distribution': {}
        }
        
        winners = 0
        errors = 0
        rally_lengths = []
        first_shots = []
        
        for point in return_points:
            shots = point.get('shots', [])
            
            if shots:
                first_shot = shots[0]
                first_shots.append(first_shot.get('shot_type', 'unknown'))
                
                # Check for return winners/errors
                for shot in shots:
                    outcome = shot.get('outcome')
                    if outcome == 'winner':
                        winners += 1
                        break
                    elif outcome in ['error', 'forced_error']:
                        errors += 1
                        break
                        
            rally_lengths.append(point.get('rally_length', 0))
        
        patterns['return_winner_rate'] = winners / len(return_points) if return_points else 0
        patterns['return_error_rate'] = errors / len(return_points) if return_points else 0
        patterns['avg_rally_length'] = np.mean(rally_lengths) if rally_lengths else 0
        
        # Shot type distribution for returns
        if first_shots:
            from collections import Counter
            shot_counts = Counter(first_shots)
            total = len(first_shots)
            patterns['shot_type_distribution'] = {
                k: v / total for k, v in shot_counts.items()
            }
        
        return patterns

    def _extract_rally_patterns(self, all_points):
        """Extract rally characteristics"""
        if not all_points:
            return self._get_default_rally_patterns()
            
        patterns = {
            'avg_rally_length': 0,
            'short_rally_rate': 0,  # <= 3 shots
            'medium_rally_rate': 0,  # 4-9 shots  
            'long_rally_rate': 0,   # >= 10 shots
            'net_approach_rate': 0,
            'winner_to_error_ratio': 1.0
        }
        
        rally_lengths = []
        net_approaches = 0
        winners = 0
        errors = 0
        
        for point in all_points:
            if not point:
                continue
                
            length = point.get('rally_length', 0)
            rally_lengths.append(length)
            
            shots = point.get('shots', [])
            
            # Check for net approaches
            for shot in shots:
                if shot.get('shot_type') == 'volley':
                    net_approaches += 1
                    break
            
            # Count winners and errors
            for shot in shots:
                outcome = shot.get('outcome')
                if outcome == 'winner':
                    winners += 1
                elif outcome in ['error', 'forced_error']:
                    errors += 1
        
        if rally_lengths:
            patterns['avg_rally_length'] = np.mean(rally_lengths)
            total_rallies = len(rally_lengths)
            patterns['short_rally_rate'] = sum(1 for x in rally_lengths if x <= 3) / total_rallies
            patterns['medium_rally_rate'] = sum(1 for x in rally_lengths if 4 <= x <= 9) / total_rallies  
            patterns['long_rally_rate'] = sum(1 for x in rally_lengths if x >= 10) / total_rallies
        
        patterns['net_approach_rate'] = net_approaches / len(all_points) if all_points else 0
        patterns['winner_to_error_ratio'] = winners / errors if errors > 0 else 2.0
        
        return patterns

    def _extract_pressure_patterns(self, all_points):
        """Extract performance under pressure"""
        if not all_points:
            return self._get_default_pressure_patterns()
            
        patterns = {
            'break_point_conversion': 0.5,
            'deuce_performance': 0.5,
            'set_point_performance': 0.5,
            'clutch_shot_accuracy': 0.5
        }
        
        # This would require score context which isn't in basic notation
        # For now, return defaults - could be enhanced with game score data
        
        return patterns

    def _get_default_patterns(self):
        """Default patterns when no data available"""
        return {
            'serve_patterns': self._get_default_serve_patterns(),
            'return_patterns': self._get_default_return_patterns(), 
            'rally_patterns': self._get_default_rally_patterns(),
            'pressure_patterns': self._get_default_pressure_patterns()
        }

    def _get_default_serve_patterns(self):
        return {
            'direction_distribution': {'wide': 0.3, 'body': 0.4, 'center': 0.3},
            'ace_rate': 0.08,
            'service_winner_rate': 0.15,
            'avg_rally_length': 2.5
        }

    def _get_default_return_patterns(self):
        return {
            'return_winner_rate': 0.05,
            'return_error_rate': 0.15,
            'avg_rally_length': 4.2,
            'shot_type_distribution': {'forehand': 0.6, 'backhand': 0.4}
        }

    def _get_default_rally_patterns(self):
        return {
            'avg_rally_length': 3.8,
            'short_rally_rate': 0.6,
            'medium_rally_rate': 0.3,
            'long_rally_rate': 0.1,
            'net_approach_rate': 0.08,
            'winner_to_error_ratio': 1.2
        }

    def _get_default_pressure_patterns(self):
        return {
            'break_point_conversion': 0.42,
            'deuce_performance': 0.52,
            'set_point_performance': 0.65,
            'clutch_shot_accuracy': 0.48
        }

def extract_jeff_notation_features(player_canonical, gender, jeff_data):
    gender_key = 'men' if gender == 'M' else 'women'

    if gender_key not in jeff_data or 'points_2020s' not in jeff_data[gender_key]:
        return {}

    points_df = jeff_data[gender_key]['points_2020s']
    player_points = []

    for _, row in points_df.iterrows():
        match_id = row['match_id']
        parts = match_id.split('-')

        if len(parts) >= 6:
            player1_canonical = canonical_player(parts[-2].replace('_', ' '))
            player2_canonical = canonical_player(parts[-1].replace('_', ' '))

            server = row['Svr']
            current_player = player1_canonical if server == 1 else player2_canonical

            if current_player == player_canonical:
                notation = row.get('1st', '') or row.get('2nd', '')
                if notation and pd.notna(notation):
                    player_points.append({
                        'notation': notation,
                        'serving': True,
                        'match_id': match_id
                    })

    if not player_points:
        return {}

    # Parse with Jeff notation parser
    parser = JeffNotationParser()
    patterns = parser.extract_player_patterns(player_points)

    # Convert to flat feature dictionary (COMPLETE VERSION - 17 features)
    features = {}

    # Serve features
    serve_patterns = patterns.get('serve_patterns', {})
    features['jeff_ace_rate'] = serve_patterns.get('ace_rate', 0.08)
    features['jeff_service_winner_rate'] = serve_patterns.get('service_winner_rate', 0.15)
    features['jeff_serve_rally_length'] = serve_patterns.get('avg_rally_length', 2.5)

    # Direction preferences
    direction_dist = serve_patterns.get('direction_distribution', {})
    features['jeff_serve_wide_pct'] = direction_dist.get('wide', 0.3)
    features['jeff_serve_body_pct'] = direction_dist.get('body', 0.4)
    features['jeff_serve_center_pct'] = direction_dist.get('center', 0.3)

    # Return features
    return_patterns = patterns.get('return_patterns', {})
    features['jeff_return_winner_rate'] = return_patterns.get('return_winner_rate', 0.05)
    features['jeff_return_error_rate'] = return_patterns.get('return_error_rate', 0.15)
    features['jeff_return_rally_length'] = return_patterns.get('avg_rally_length', 4.2)

    # Rally features
    rally_patterns = patterns.get('rally_patterns', {})
    features['jeff_avg_rally_length'] = rally_patterns.get('avg_rally_length', 3.8)
    features['jeff_short_rally_rate'] = rally_patterns.get('short_rally_rate', 0.6)
    features['jeff_long_rally_rate'] = rally_patterns.get('long_rally_rate', 0.1)
    features['jeff_net_approach_rate'] = rally_patterns.get('net_approach_rate', 0.08)
    features['jeff_winner_error_ratio'] = rally_patterns.get('winner_to_error_ratio', 1.2)

    # Derived tactical features
    features['jeff_aggression_index'] = (
                                                features['jeff_service_winner_rate'] +
                                                features['jeff_return_winner_rate'] +
                                                features['jeff_net_approach_rate']
                                        ) / 3

    features['jeff_consistency_index'] = 1 - features['jeff_return_error_rate']

    features['jeff_serve_placement_variety'] = 1 - max(
        features['jeff_serve_wide_pct'],
        features['jeff_serve_body_pct'],
        features['jeff_serve_center_pct']
    )

    return features

# ============================================================================
# DATA INTEGRATION
# ============================================================================

def integrate_jeff_notation_into_pipeline(historical_data, jeff_data):
    """Add Jeff notation features to the historical dataset"""
    print("Integrating Jeff notation features...")
    
    # Add Jeff notation feature columns
    jeff_features = [
        'jeff_ace_rate', 'jeff_service_winner_rate', 'jeff_serve_rally_length',
        'jeff_serve_wide_pct', 'jeff_serve_body_pct', 'jeff_serve_center_pct',
        'jeff_return_winner_rate', 'jeff_return_error_rate', 'jeff_return_rally_length',
        'jeff_avg_rally_length', 'jeff_short_rally_rate', 'jeff_long_rally_rate',
        'jeff_net_approach_rate', 'jeff_winner_error_ratio', 'jeff_aggression_index',
        'jeff_consistency_index', 'jeff_serve_placement_variety'
    ]
    
    # Add columns for winner and loser
    for feature in jeff_features:
        for prefix in ['winner', 'loser']:
            col_name = f"{prefix}_{feature}"
            if col_name not in historical_data.columns:
                historical_data[col_name] = np.nan
    
    matches_updated = 0
    total_matches = len(historical_data)
    
    for idx, row in historical_data.iterrows():
        if idx % 1000 == 0:
            print(f"Processing Jeff notation for match {idx}/{total_matches}")
            
        # Skip if already has Jeff notation features
        if pd.notna(row.get('winner_jeff_ace_rate')):
            continue
            
        try:
            winner_canonical = row.get('winner_canonical', '')
            loser_canonical = row.get('loser_canonical', '')
            gender = row.get('gender', 'M')
            
            if not winner_canonical or not loser_canonical:
                continue
                
            # Extract features for winner
            winner_jeff_features = extract_jeff_notation_features(
                winner_canonical, gender, jeff_data
            )
            
            # Extract features for loser  
            loser_jeff_features = extract_jeff_notation_features(
                loser_canonical, gender, jeff_data
            )
            
            # Update dataframe
            for feature_name, feature_value in winner_jeff_features.items():
                col_name = f'winner_{feature_name}'
                if col_name in historical_data.columns:
                    historical_data.at[idx, col_name] = feature_value
                    
            for feature_name, feature_value in loser_jeff_features.items():
                col_name = f'loser_{feature_name}'
                if col_name in historical_data.columns:
                    historical_data.at[idx, col_name] = feature_value
            
            if winner_jeff_features or loser_jeff_features:
                matches_updated += 1
                
        except Exception as e:
            if idx < 10:  # Only log first few errors
                print(f"Error processing Jeff notation for match {idx}: {e}")
            continue
    
    print(f"Updated {matches_updated} matches with Jeff notation features")
    return historical_data


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


def transform_serve_direction(ta_records):
    """Transform TA serve direction tables to Jeff ServeDirection format"""

    jeff_serve_direction_columns = [
        'serve_wide_attempts', 'serve_wide_won', 'serve_wide_pct', 'serve_wide_effectiveness',
        'serve_t_attempts', 'serve_t_won', 'serve_t_pct', 'serve_t_effectiveness',
        'serve_body_attempts', 'serve_body_won', 'serve_body_pct', 'serve_body_effectiveness',
        'first_serve_wide', 'first_serve_t', 'first_serve_body',
        'second_serve_wide', 'second_serve_t', 'second_serve_body',
        'serve_wide_aces', 'serve_t_aces', 'serve_body_aces',
        'serve_wide_winners', 'serve_t_winners', 'serve_body_winners',
        'serve_wide_errors', 'serve_t_errors', 'serve_body_errors',
        'serve_direction_variety', 'serve_direction_dominance', 'optimal_direction_index'
    ]

    def extract_serve_direction_stats(ta_records):
        direction_data = {}
        direction_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                          for x in
                                                          ['serve1', 'serve2', 'serve_direction', 'serveNeut'])]

        for record in direction_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in direction_data:
                direction_data[player] = {}

            direction_data[player][stat_name] = stat_value

        return direction_data

    direction_stats = extract_serve_direction_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in direction_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Serve_Direction'
        }

        # Map basic stats
        record['serve_wide_attempts'] = stats.get('wide_attempts', 0)
        record['serve_wide_won'] = stats.get('wide_won', 0)
        record['serve_t_attempts'] = stats.get('t_attempts', 0)
        record['serve_t_won'] = stats.get('t_won', 0)
        record['serve_body_attempts'] = stats.get('body_attempts', 0)
        record['serve_body_won'] = stats.get('body_won', 0)

        # Calculate percentages
        total_serves = record['serve_wide_attempts'] + record['serve_t_attempts'] + record['serve_body_attempts']
        if total_serves > 0:
            record['serve_wide_pct'] = (record['serve_wide_attempts'] / total_serves * 100)
            record['serve_t_pct'] = (record['serve_t_attempts'] / total_serves * 100)
            record['serve_body_pct'] = (record['serve_body_attempts'] / total_serves * 100)
        else:
            record['serve_wide_pct'] = 0
            record['serve_t_pct'] = 0
            record['serve_body_pct'] = 0

        # Calculate effectiveness
        if record['serve_wide_attempts'] > 0:
            record['serve_wide_effectiveness'] = (record['serve_wide_won'] / record['serve_wide_attempts'] * 100)
        else:
            record['serve_wide_effectiveness'] = 0

        if record['serve_t_attempts'] > 0:
            record['serve_t_effectiveness'] = (record['serve_t_won'] / record['serve_t_attempts'] * 100)
        else:
            record['serve_t_effectiveness'] = 0

        # Set remaining columns to 0
        for col in jeff_serve_direction_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output


def transform_shot_dir_outcomes(ta_records):
    """Transform TA shot direction outcome tables to Jeff ShotDirOutcomes format"""

    jeff_shot_dir_outcomes_columns = [
        'crosscourt_attempts', 'crosscourt_winners', 'crosscourt_errors', 'crosscourt_effectiveness',
        'down_line_attempts', 'down_line_winners', 'down_line_errors', 'down_line_effectiveness',
        'inside_out_attempts', 'inside_out_winners', 'inside_out_errors', 'inside_out_effectiveness',
        'inside_in_attempts', 'inside_in_winners', 'inside_in_errors', 'inside_in_effectiveness',
        'fh_crosscourt_attempts', 'fh_crosscourt_winners', 'fh_crosscourt_errors', 'fh_crosscourt_effectiveness',
        'fh_down_line_attempts', 'fh_down_line_winners', 'fh_down_line_errors', 'fh_down_line_effectiveness',
        'bh_crosscourt_attempts', 'bh_crosscourt_winners', 'bh_crosscourt_errors', 'bh_crosscourt_effectiveness',
        'bh_down_line_attempts', 'bh_down_line_winners', 'bh_down_line_errors', 'bh_down_line_effectiveness',
        'crosscourt_pct', 'down_line_pct', 'inside_out_pct', 'inside_in_pct',
        'directional_accuracy', 'directional_aggression', 'directional_consistency'
    ]

    def extract_shot_dir_outcomes_stats(ta_records):
        outcomes_data = {}
        outcomes_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                         for x in ['shotdir1', 'shotdir2', 'shot_dir_outcomes',
                                                                   'direction_outcomes'])]

        for record in outcomes_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in outcomes_data:
                outcomes_data[player] = {}

            outcomes_data[player][stat_name] = stat_value

        return outcomes_data

    outcomes_stats = extract_shot_dir_outcomes_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in outcomes_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Shot_Dir_Outcomes'
        }

        # Map basic stats
        record['crosscourt_attempts'] = stats.get('crosscourt_attempts', 0)
        record['crosscourt_winners'] = stats.get('crosscourt_winners', 0)
        record['crosscourt_errors'] = stats.get('crosscourt_errors', 0)
        record['down_line_attempts'] = stats.get('down_line_attempts', 0)
        record['down_line_winners'] = stats.get('down_line_winners', 0)
        record['down_line_errors'] = stats.get('down_line_errors', 0)
        record['fh_crosscourt_winners'] = stats.get('fh_crosscourt_winners', 0)
        record['fh_down_line_winners'] = stats.get('fh_down_line_winners', 0)

        # Calculate total attempts
        total_attempts = record['crosscourt_attempts'] + record['down_line_attempts']

        # Calculate percentages
        if total_attempts > 0:
            record['crosscourt_pct'] = (record['crosscourt_attempts'] / total_attempts * 100)
            record['down_line_pct'] = (record['down_line_attempts'] / total_attempts * 100)
        else:
            record['crosscourt_pct'] = 0
            record['down_line_pct'] = 0

        # Calculate effectiveness
        if record['crosscourt_attempts'] > 0:
            cc_total_outcomes = record['crosscourt_winners'] + record['crosscourt_errors']
            record['crosscourt_effectiveness'] = (
                        record['crosscourt_winners'] / cc_total_outcomes * 100) if cc_total_outcomes > 0 else 0
        else:
            record['crosscourt_effectiveness'] = 0

        if record['down_line_attempts'] > 0:
            dl_total_outcomes = record['down_line_winners'] + record['down_line_errors']
            record['down_line_effectiveness'] = (
                        record['down_line_winners'] / dl_total_outcomes * 100) if dl_total_outcomes > 0 else 0
        else:
            record['down_line_effectiveness'] = 0

        # Set remaining columns to 0
        for col in jeff_shot_dir_outcomes_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output


def transform_net_points(ta_records):
    """Transform TA net points tables to Jeff NetPoints format"""

    jeff_net_points_columns = [
        'net_points_attempted', 'net_points_won', 'net_points_lost', 'net_points_won_pct',
        'approach_shots', 'approach_shots_won', 'approach_shots_lost', 'approach_effectiveness',
        'volleys_attempted', 'volleys_won', 'volleys_lost', 'volley_effectiveness',
        'overheads_attempted', 'overheads_won', 'overheads_lost', 'overhead_effectiveness',
        'net_winners', 'net_errors', 'net_forced_errors', 'net_unforced_errors',
        'passed_at_net', 'passing_shots_against', 'passing_shot_defense_pct',
        'fh_volleys', 'bh_volleys', 'fh_volley_winners', 'bh_volley_winners',
        'net_game_dominance', 'net_tactical_advantage', 'net_positioning_effectiveness'
    ]

    def extract_net_points_stats(ta_records):
        net_data = {}
        net_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                    for x in ['netpts1', 'netpts2', 'net_points', 'netpoints'])]

        for record in net_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in net_data:
                net_data[player] = {}

            net_data[player][stat_name] = stat_value

        return net_data

    net_stats = extract_net_points_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in net_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Net_Points'
        }

        # Map basic stats
        record['net_points_attempted'] = stats.get('net_points_attempted', 0)
        record['net_points_won'] = stats.get('net_points_won', 0)
        record['net_points_lost'] = stats.get('net_points_lost', 0)
        record['approach_shots'] = stats.get('approach_shots', 0)
        record['approach_shots_won'] = stats.get('approach_shots_won', 0)
        record['volleys_attempted'] = stats.get('volleys_attempted', 0)
        record['volleys_won'] = stats.get('volleys_won', 0)
        record['net_winners'] = stats.get('net_winners', 0)
        record['net_errors'] = stats.get('net_errors', 0)
        record['passed_at_net'] = stats.get('passed_at_net', 0)

        # Calculate percentages
        if record['net_points_attempted'] > 0:
            record['net_points_won_pct'] = (record['net_points_won'] / record['net_points_attempted'] * 100)
        else:
            record['net_points_won_pct'] = 0

        if record['approach_shots'] > 0:
            record['approach_effectiveness'] = (record['approach_shots_won'] / record['approach_shots'] * 100)
        else:
            record['approach_effectiveness'] = 0

        if record['volleys_attempted'] > 0:
            record['volley_effectiveness'] = (record['volleys_won'] / record['volleys_attempted'] * 100)
        else:
            record['volley_effectiveness'] = 0

        # Set remaining columns to 0
        for col in jeff_net_points_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output


def transform_sv_break_split(ta_records):
    """Transform TA serve/break data to Jeff SvBreakSplit format"""

    jeff_sv_break_split_columns = [
        'service_games_played', 'service_games_won', 'service_games_lost', 'service_hold_pct',
        'break_points_faced', 'break_points_saved', 'break_points_saved_pct',
        'break_points_lost', 'break_points_lost_pct',
        'deuce_games', 'deuce_games_won', 'deuce_games_lost', 'deuce_games_won_pct',
        'love_holds', 'fifteen_holds', 'thirty_holds', 'forty_holds', 'deuce_holds',
        'easy_holds_pct', 'pressure_holds_pct', 'clutch_holds_pct',
        'first_serve_hold_pct', 'second_serve_hold_pct',
        'service_dominance_index', 'hold_consistency', 'pressure_resistance'
    ]

    def extract_sv_break_split_stats(ta_records):
        split_data = {}
        split_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                      for x in ['serve1', 'serve2', 'keypoints', 'overview'])]

        for record in split_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in split_data:
                split_data[player] = {}

            split_data[player][stat_name] = stat_value

        return split_data

    split_stats = extract_sv_break_split_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in split_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Sv_Break_Split'
        }

        # Map basic stats
        record['service_games_played'] = stats.get('service_games_played', 0)
        record['service_games_won'] = stats.get('service_games_won', 0)
        record['service_games_lost'] = stats.get('service_games_lost', 0)
        record['break_points_faced'] = stats.get('break_points_faced', 0)
        record['break_points_saved'] = stats.get('break_points_saved', 0)
        record['deuce_games'] = stats.get('deuce_games', 0)
        record['deuce_games_won'] = stats.get('deuce_games_won', 0)
        record['love_holds'] = stats.get('love_holds', 0)
        record['easy_holds'] = stats.get('easy_holds', 0)

        # Calculate derived stats
        if record['service_games_played'] > 0:
            record['service_hold_pct'] = (record['service_games_won'] / record['service_games_played'] * 100)
        else:
            record['service_hold_pct'] = 0

        if record['break_points_faced'] > 0:
            record['break_points_saved_pct'] = (record['break_points_saved'] / record['break_points_faced'] * 100)
            record['break_points_lost'] = record['break_points_faced'] - record['break_points_saved']
            record['break_points_lost_pct'] = (record['break_points_lost'] / record['break_points_faced'] * 100)
        else:
            record['break_points_saved_pct'] = 0
            record['break_points_lost'] = 0
            record['break_points_lost_pct'] = 0

        if record['deuce_games'] > 0:
            record['deuce_games_won_pct'] = (record['deuce_games_won'] / record['deuce_games'] * 100)
        else:
            record['deuce_games_won_pct'] = 0

        # Set remaining columns to 0
        for col in jeff_sv_break_split_columns:
            if col not in record:
                record[col] = 0

        output.append(record)

    return output


def transform_sv_break_total(ta_records):
    """Transform TA serve/break data to Jeff SvBreakTotal format"""

    jeff_sv_break_total_columns = [
        'total_service_points', 'total_service_points_won', 'total_service_points_lost', 'total_service_points_won_pct',
        'total_return_points', 'total_return_points_won', 'total_return_points_lost', 'total_return_points_won_pct',
        'total_break_points_faced', 'total_break_points_saved', 'total_break_points_saved_pct',
        'total_break_points_created', 'total_break_points_converted', 'total_break_points_converted_pct',
        'total_service_games', 'total_service_games_won', 'total_service_games_lost', 'total_service_hold_pct',
        'total_return_games', 'total_return_games_won', 'total_return_games_lost', 'total_break_pct',
        'net_break_differential', 'break_point_efficiency', 'service_dominance_total',
        'return_aggression_total', 'pressure_point_performance', 'clutch_factor_total'
    ]

    def extract_sv_break_total_stats(ta_records):
        total_data = {}
        total_records = [r for r in ta_records if any(x in r.get('data_type', '')
                                                      for x in ['overview', 'serve1', 'serve2', 'return1', 'return2',
                                                                'keypoints'])]

        for record in total_records:
            player = record.get('Player_canonical')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value', 0)

            if player not in total_data:
                total_data[player] = {}

            total_data[player][stat_name] = stat_value

        return total_data

    total_stats = extract_sv_break_total_stats(ta_records)
    output = []
    match_id = extract_match_id(ta_records)

    for player, stats in total_stats.items():
        record = {
            'match_id': match_id,
            'Player_canonical': player,
            'context': 'Sv_Break_Total'
        }

        # Map basic stats
        record['total_service_points'] = stats.get('total_service_points', 0)
        record['total_service_points_won'] = stats.get('total_service_points_won', 0)
        record['total_return_points'] = stats.get('total_return_points', 0)
        record['total_return_points_won'] = stats.get('total_return_points_won', 0)
        record['total_break_points_faced'] = stats.get('total_break_points_faced', 0)
        record['total_break_points_saved'] = stats.get('total_break_points_saved', 0)
        record['total_break_points_created'] = stats.get('total_break_points_created', 0)
        record['total_break_points_converted'] = stats.get('total_break_points_converted', 0)
        record['total_service_games'] = stats.get('total_service_games', 0)
        record['total_service_games_won'] = stats.get('total_service_games_won', 0)

        # Calculate derived stats
        if record['total_service_points'] > 0:
            record['total_service_points_won_pct'] = (
                        record['total_service_points_won'] / record['total_service_points'] * 100)
        else:
            record['total_service_points_won_pct'] = 0

        if record['total_return_points'] > 0:
            record['total_return_points_won_pct'] = (
                        record['total_return_points_won'] / record['total_return_points'] * 100)
        else:
            record['total_return_points_won_pct'] = 0

        if record['total_break_points_faced'] > 0:
            record['total_break_points_saved_pct'] = (
                        record['total_break_points_saved'] / record['total_break_points_faced'] * 100)
        else:
            record['total_break_points_saved_pct'] = 0

        if record['total_break_points_created'] > 0:
            record['total_break_points_converted_pct'] = (
                        record['total_break_points_converted'] / record['total_break_points_created'] * 100)
        else:
            record['total_break_points_converted_pct'] = 0

        if record['total_service_games'] > 0:
            record['total_service_hold_pct'] = (record['total_service_games_won'] / record['total_service_games'] * 100)
        else:
            record['total_service_hold_pct'] = 0

        # Calculate break differential
        breaks_achieved = record['total_break_points_converted']
        breaks_suffered = record['total_break_points_faced'] - record['total_break_points_saved']
        record['net_break_differential'] = breaks_achieved - breaks_suffered

        # Set remaining columns to 0
        for col in jeff_sv_break_total_columns:
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

    if not fresh_scraped:  # FIXED: was scraped_records
        print("No new Tennis Abstract data scraped")
        return historical_data

    enhanced_data = integrate_scraped_data_hybrid(historical_data, fresh_scraped)  # FIXED: was scraped_records

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


def infer_surface_from_tournament(tournament_name, tournament_round=""):
    """Map tournament names to surfaces"""
    name = str(tournament_name).lower()
    round_info = str(tournament_round).lower()

    # Grand Slams
    if 'wimbledon' in name or 'wimbledon' in round_info:
        return 'Grass'
    elif any(x in name for x in ['french', 'roland', 'garros']):
        return 'Clay'
    elif 'us open' in name or 'australian' in name:
        return 'Hard'

    # ATP/WTA Tours - Clay
    elif any(x in name for x in
             ['monte carlo', 'madrid', 'rome', 'barcelona', 'hamburg', 'gstaad', 'umag', 'bastad', 'kitzbuhel',
              'bucharest', 'marrakech', 'estoril', 'munich', 'geneva']):
        return 'Clay'

    # ATP/WTA Tours - Hard
    elif any(x in name for x in
             ['masters', 'miami', 'indian wells', 'cincinnati', 'toronto', 'montreal', 'paris masters', 'shanghai',
              'beijing', 'tokyo', 'dubai', 'doha', 'acapulco', 'delray beach', 'memphis', 'las vegas', 'winston salem',
              'washington', 'atlanta', 'newport', 'los cabos']):
        return 'Hard'

    # ATP/WTA Tours - Grass
    elif any(x in name for x in ['halle', 'queens', 'eastbourne', 's-hertogenbosch', 'stuttgart', 'mallorca']):
        return 'Grass'

    # Default to Hard (most common)
    return 'Hard'


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
                    tournament_name = fixture.get('tournament_name', '')
                    tournament_round = fixture.get('tournament_round', '')

                    tournament_dict[str(tournament_key)] = {
                        'tournament_name': tournament_name,
                        'tournament_key': tournament_key,
                        'tournament_round': tournament_round,
                        'tournament_season': fixture.get('tournament_season'),
                        'event_type_type': fixture.get('event_type_type'),
                        'surface': infer_surface_from_tournament(tournament_name, tournament_round)
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
    """FIXED: API integration that properly stores set scores"""
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
    print(f"Added {len(dates_to_fetch)} days of API data with set scores to cache.")
    return df


# ============================================================================
# DATA PROCESSING
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

    logging.info("Step 7: Adding Jeff notation features...")
    tennis_data = integrate_jeff_notation_into_pipeline(tennis_data, jeff_data)

    logging.info("Step 8: Integrating API and TA data...")
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


if __name__ == "__main__":
    """Pure data pipeline execution"""

    print("🎾 TENNIS DATA PIPELINE 🎾")

    # Generate complete dataset
    hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=False)

    # Integrate all data sources
    hist = integrate_api_tennis_data_incremental(hist)

    # Save final dataset
    save_to_cache(hist, jeff_data, defaults)

    print(f"✓ Dataset complete: {hist.shape}")
    print(f"✓ Jeff features: {len([c for c in hist.columns if 'jeff_' in c])}")
    print(f"✓ Saved to cache for model.py")
