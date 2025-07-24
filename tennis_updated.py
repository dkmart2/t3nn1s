# ============================================================================
# TENNIS DATA PIPELINE - FOCUSED IMPLEMENTATION
# ============================================================================

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import date, timedelta, datetime
import os
import joblib
import time
from pathlib import Path
import filelock
from unidecode import unidecode
import hashlib
import asyncio
import httpx
import requests_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Import only what we need from settings
from settings import CACHE_DIR, BASE_CUTOFF_DATE, API_TENNIS_KEY, BASE_API_URL, API_MAX_RETRIES, API_MIN_DELAY, \
    MAX_CONCURRENT_REQUESTS, LOG_LEVEL

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)


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
# PLAYER CANONICALIZER
# ============================================================================

class PlayerCanonicalizer:
    """Thread-safe player name canonicalization"""

    def __init__(self):
        self.cache_file = CACHE_DIR / "player_canonical_cache.joblib"
        self.lock_file = str(self.cache_file) + ".lock"
        self.mapping = self._load_cache()

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

        if len(self.mapping) % 100 == 0:
            self._save_cache()

        return canonical

    def _normalize_name(self, name: str) -> str:
        """Normalize player name"""
        name = unidecode(name).replace('.', '').replace("'", '').replace('-', ' ')
        name = ' '.join(name.lower().split())
        return name

    def _save_cache(self):
        """Save player mapping to cache"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / filename


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

    # Cache with compression
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
        # Load with Polars and convert string columns to categorical
        df_polars = pl.read_csv(str(file_path), infer_schema_length=10000, **kwargs)
        df_polars = df_polars.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
        df_pandas = df_polars.to_pandas()

        # Cache the result
        joblib.dump(df_pandas, cache_path, compress=('zlib', 3))
        logger.debug(f"Loaded and cached CSV with Polars: {file_path}")
        return df_pandas

    except Exception as e:
        logger.error(f"Polars CSV loading failed for {file_path}: {e}")
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as fallback_error:
            raise DataIngestionError(f"Both Polars and pandas failed for {file_path}: {fallback_error}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_tennis_data() -> pd.DataFrame:
    """Load tennis match data with date constraints"""
    logger.info("Loading tennis match data")

    dataframes = []
    errors = []

    # Men's and women's data files (2020+)
    all_files = []
    for year in range(2020, 2026):
        for gender in ['m', 'w']:
            file_path = Path(f"tennis_data/{year}_{gender}.xlsx")
            if file_path.exists():
                all_files.append((file_path, 'M' if gender == 'm' else 'W'))

    # Historical CSV files
    historical_files = [
        (Path("tennis_data/match_history_men.csv"), 'M'),
        (Path("tennis_data/match_history_women.csv"), 'W')
    ]
    all_files.extend([(f, g) for f, g in historical_files if f.exists()])

    # Load files in parallel
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, os.cpu_count() or 2)) as executor:
        future_to_file = {}

        for file_path, gender in all_files:
            if str(file_path).endswith('.xlsx'):
                future = executor.submit(load_excel_with_cache, file_path)
            else:
                future = executor.submit(load_csv_with_polars, file_path)
            future_to_file[future] = (file_path, gender)

        for future in future_to_file:
            file_path, gender = future_to_file[future]
            try:
                df = future.result(timeout=60)

                # Add metadata
                df['gender'] = gender
                df['data_source'] = 'excel'  # Will be overridden by higher priority sources

                # Add canonical names
                if 'Winner' in df.columns:
                    df['winner_canonical'] = df['Winner'].apply(player_canonicalizer.canonical_player)
                if 'Loser' in df.columns:
                    df['loser_canonical'] = df['Loser'].apply(player_canonicalizer.canonical_player)

                # Process dates and apply cutoff
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df['date'] = df['Date'].dt.date
                    df = df.dropna(subset=['date'])
                    df = df[df['date'] <= BASE_CUTOFF_DATE]

                # Create composite ID
                if all(col in df.columns for col in ['date', 'winner_canonical', 'loser_canonical']):
                    df['composite_id'] = df.apply(
                        lambda
                            row: f"{row['date'].strftime('%Y%m%d')}-{row['winner_canonical']}-{row['loser_canonical']}",
                        axis=1
                    )

                dataframes.append(df)
                logger.info(f"Loaded {len(df):,} matches from {file_path}")

            except Exception as e:
                error_msg = f"Failed to load {file_path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

    if not dataframes:
        raise DataIngestionError(f"No data loaded successfully. Errors: {errors}")

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    logger.info(f"Combined tennis data: {len(combined_df):,} total matches")

    # Runtime guardrail
    if len(combined_df) < 10000:
        raise DataIngestionError(f"Insufficient tennis data: {len(combined_df)} matches (minimum 10,000)")

    return combined_df


def load_jeff_comprehensive_data():
    """Load Jeff charting data - abort on failure"""
    logger.info("Loading Jeff comprehensive data")

    cache_path = CACHE_DIR / "jeff_data.joblib"

    if cache_path.exists():
        try:
            jeff_data = joblib.load(cache_path)
            logger.info("Loaded Jeff data from cache")
            return jeff_data
        except Exception as e:
            logger.warning(f"Failed to load Jeff data from cache: {e}")

    jeff_data_dir = Path("tennis_data/jeff")
    if not jeff_data_dir.exists():
        raise DataIngestionError(f"Jeff data directory not found: {jeff_data_dir}")

    jeff_data = {'men': {}, 'women': {}}

    # Load charting files
    file_patterns = {
        'men': ['chartingm*.csv'],
        'women': ['chartingw*.csv']
    }

    for gender, patterns in file_patterns.items():
        for pattern in patterns:
            for file_path in jeff_data_dir.glob(pattern):
                try:
                    filename = file_path.stem
                    df = load_csv_with_polars(file_path)

                    if not df.empty:
                        jeff_data[gender][filename] = df
                        logger.info(f"Loaded Jeff {gender} data: {filename} ({len(df):,} rows)")

                except Exception as e:
                    logger.error(f"Failed to load Jeff file {file_path}: {e}")

    # Runtime guardrail
    overview_rows = 0
    for gender in jeff_data:
        if 'overview' in jeff_data[gender]:
            overview_rows += len(jeff_data[gender]['overview'])

    if overview_rows < 10000:
        raise DataIngestionError(f"Insufficient Jeff overview data: {overview_rows} rows (minimum 10,000)")

    # Cache the loaded data
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(jeff_data, cache_path, compress=('zlib', 3))
        logger.info("Cached Jeff comprehensive data")
    except Exception as e:
        logger.warning(f"Failed to cache Jeff data: {e}")

    return jeff_data


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_comprehensive_jeff_features(df: pd.DataFrame, jeff_data: Dict) -> pd.DataFrame:
    """Extract Jeff features using aggregated data (no Total filter)"""
    logger.info("Extracting comprehensive Jeff features")

    if df.empty:
        return df

    # Pre-compute feature matrices for both genders
    feature_matrices = {}

    for gender in ['M', 'W']:
        gender_key = 'men' if gender == 'M' else 'women'

        if gender_key not in jeff_data or 'overview' not in jeff_data[gender_key]:
            continue

        overview_df = jeff_data[gender_key]['overview']
        if overview_df.empty:
            continue

        # Use all overview data (no Total filter as requested)
        features_matrix = compute_features_vectorized(overview_df)
        features_matrix['gender'] = gender

        feature_matrices[gender] = features_matrix

    # Combine and merge features
    if feature_matrices:
        all_features = pd.concat(feature_matrices.values(), ignore_index=True)

        # Merge winner features
        winner_features = all_features.add_prefix('winner_')
        winner_features.rename(columns={'winner_Player_canonical': 'winner_canonical',
                                        'winner_gender': 'gender'}, inplace=True)

        df = df.merge(winner_features, on=['winner_canonical', 'gender'], how='left')

        # Merge loser features
        loser_features = all_features.add_prefix('loser_')
        loser_features.rename(columns={'loser_Player_canonical': 'loser_canonical',
                                       'loser_gender': 'gender'}, inplace=True)

        df = df.merge(loser_features, on=['loser_canonical', 'gender'], how='left')

    logger.info(f"Jeff feature extraction complete: {len(df):,} matches")
    return df


def compute_features_vectorized(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features vectorized"""
    features_df = stats_df.copy()

    # Vectorized ratio calculations
    serve_pts = features_df.get('serve_pts', pd.Series([80] * len(features_df))).replace(0, 1)

    if 'first_in' in features_df.columns:
        features_df['first_serve_pct'] = features_df['first_in'] / serve_pts
    if 'aces' in features_df.columns:
        features_df['ace_rate'] = features_df['aces'] / serve_pts
    if 'dfs' in features_df.columns:
        features_df['df_rate'] = features_df['dfs'] / serve_pts

    # Return calculations
    if 'return_pts' in features_df.columns and 'return_pts_won' in features_df.columns:
        return_pts = features_df['return_pts'].replace(0, 1)
        features_df['return_pts_won_pct'] = features_df['return_pts_won'] / return_pts

    # Winner/error ratios
    if 'winners' in features_df.columns and 'unforced' in features_df.columns:
        features_df['winner_error_ratio'] = features_df['winners'] / features_df['unforced'].replace(0, 1)

    return features_df


# ============================================================================
# API INTEGRATION
# ============================================================================

def get_cached_session() -> requests_cache.CachedSession:
    """Get cached session"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=str(CACHE_DIR / 'api_cache'),
        expire_after=3600
    )


async def fetch_api_data(date_range: List[date]) -> pd.DataFrame:
    """Fetch API data for date range"""
    logger.info(f"Fetching API data for {len(date_range)} dates")

    if not API_TENNIS_KEY:
        raise APIError("API_TENNIS_KEY not configured")

    all_fixtures = []

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:

        for target_date in date_range:
            try:
                # Get events
                events_url = f"{BASE_API_URL}?method=get_events&key={API_TENNIS_KEY}"
                events_response = await client.get(events_url)
                await asyncio.sleep(API_MIN_DELAY)
                events_response.raise_for_status()
                events_data = events_response.json()

                # Find tennis events
                tennis_events = [
                                    event for event in events_data
                                    if any(term in event.get('event_type_type', '').lower()
                                           for term in ['atp', 'wta', 'tennis'])
                                ][:3]  # Limit to 3 events

                # Get fixtures for each event
                for event in tennis_events:
                    fixtures_url = (
                        f"{BASE_API_URL}?method=get_fixtures"
                        f"&key={API_TENNIS_KEY}"
                        f"&date_start={target_date.isoformat()}"
                        f"&date_stop={target_date.isoformat()}"
                        f"&event_type_key={event.get('event_type_key')}"
                    )

                    fixtures_response = await client.get(fixtures_url)
                    await asyncio.sleep(API_MIN_DELAY)
                    fixtures_response.raise_for_status()
                    fixtures_data = fixtures_response.json()

                    if isinstance(fixtures_data, list):
                        all_fixtures.extend(fixtures_data)

            except Exception as e:
                logger.warning(f"API fetch failed for {target_date}: {e}")
                continue

    # Convert fixtures to DataFrame
    if not all_fixtures:
        return pd.DataFrame()

    api_matches = []
    for fixture in all_fixtures:
        try:
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            match_date = pd.to_datetime(fixture.get('date_start')).date()
            winner_name = participants[0].get('name', '')
            loser_name = participants[1].get('name', '')

            match_data = {
                'date': match_date,
                'Winner': winner_name,
                'Loser': loser_name,
                'winner_canonical': player_canonicalizer.canonical_player(winner_name),
                'loser_canonical': player_canonicalizer.canonical_player(loser_name),
                'Tournament': fixture.get('event_name', ''),
                'gender': 'M',  # Default - would need better inference
                'data_source': 'api'
            }

            match_data[
                'composite_id'] = f"{match_date.strftime('%Y%m%d')}-{match_data['winner_canonical']}-{match_data['loser_canonical']}"
            api_matches.append(match_data)

        except Exception as e:
            logger.warning(f"Failed to convert fixture: {e}")

    if api_matches:
        api_df = pd.DataFrame(api_matches)
        logger.info(f"Converted {len(api_df)} API fixtures to matches")
        return api_df

    return pd.DataFrame()


# ============================================================================
# DATA INTEGRATION WITH HIERARCHY ENFORCEMENT
# ============================================================================

def integrate_all_data_sources(tennis_data: pd.DataFrame, jeff_data: Dict) -> pd.DataFrame:
    """Integrate all data sources with hierarchy: Jeff/TA > API > Excel"""
    logger.info("Integrating all data sources")

    # Step 1: Extract Jeff features (highest priority data enhancement)
    tennis_data = extract_comprehensive_jeff_features(tennis_data, jeff_data)

    # Step 2: Fetch API data for dates after cutoff
    recent_dates = []
    end_date = date.today()
    current_date = BASE_CUTOFF_DATE + timedelta(days=1)

    while current_date <= end_date and len(recent_dates) < 30:  # Limit to 30 days
        recent_dates.append(current_date)
        current_date += timedelta(days=1)

    if recent_dates:
        try:
            api_data = asyncio.run(fetch_api_data(recent_dates))
            if not api_data.empty:
                # Add API data to tennis_data
                tennis_data = pd.concat([tennis_data, api_data], ignore_index=True, sort=False)
                logger.info(f"Added {len(api_data)} API matches")
        except Exception as e:
            logger.warning(f"API integration failed: {e}")

    # Step 3: TA integration (when available) - overwrite API/Excel for same composite_id
    # For now, just mark existing data source priorities
    # Jeff features already applied, API data already added

    # Step 4: Enforce data hierarchy - keep highest priority version of each match
    if 'composite_id' in tennis_data.columns:
        # Define source priority (lower number = higher priority)
        source_priority = {'jeff': 1, 'ta': 1, 'api': 2, 'excel': 3}
        tennis_data['source_priority'] = tennis_data['data_source'].map(source_priority).fillna(3)

        # Sort by priority and keep first occurrence of each composite_id
        tennis_data = tennis_data.sort_values(['composite_id', 'source_priority'])
        tennis_data = tennis_data.drop_duplicates(subset=['composite_id'], keep='first')
        tennis_data = tennis_data.drop(columns=['source_priority'])

        logger.info(f"After deduplication: {len(tennis_data):,} unique matches")

    return tennis_data


# ============================================================================
# POINT DATA PREPARATION
# ============================================================================

def prepare_training_data_for_ml_model(tennis_data: pd.DataFrame, jeff_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training data with surface one-hot encoding and rally length"""
    logger.info("Preparing training data for ML model")

    # Load point-by-point data
    point_files = []
    jeff_data_dir = Path("tennis_data/jeff")

    for pattern in ['chartingmpoints*.csv', 'chartingwpoints*.csv']:
        point_files.extend(jeff_data_dir.glob(pattern))

    if not point_files:
        raise DataIngestionError("No point data files found")

    point_dataframes = []
    for file_path in point_files:
        try:
            df = load_csv_with_polars(file_path)
            if not df.empty:
                point_dataframes.append(df)
                logger.info(f"Loaded point data: {file_path.name} ({len(df):,} points)")
        except Exception as e:
            logger.warning(f"Failed to load point file {file_path}: {e}")

    if not point_dataframes:
        raise DataIngestionError("No point data loaded")

    point_df = pd.concat(point_dataframes, ignore_index=True)

    # Runtime guardrail
    if len(point_df) < 100000:
        raise DataIngestionError(f"Insufficient point data: {len(point_df)} points (minimum 100,000)")

    # Add rally length and other features
    if 'rallyCount' in point_df.columns:
        point_df['rally_length'] = point_df['rallyCount']
    else:
        point_df['rally_length'] = 1  # Default if not available

    logger.info(f"Point data prepared: {len(point_df):,} points")

    # Add surface one-hot encoding to tennis_data
    if 'Surface' in tennis_data.columns:
        surface_dummies = pd.get_dummies(tennis_data['Surface'], prefix='surface')
        tennis_data = pd.concat([tennis_data, surface_dummies], axis=1)

    return tennis_data, point_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main pipeline execution"""
    logger.info("Starting tennis data pipeline")

    try:
        # Load base tennis data
        tennis_data = load_all_tennis_data()

        # Load Jeff data
        jeff_data = load_jeff_comprehensive_data()

        # Integrate all data sources with hierarchy
        tennis_data = integrate_all_data_sources(tennis_data, jeff_data)

        # Prepare training data
        tennis_data, point_df = prepare_training_data_for_ml_model(tennis_data, jeff_data)

        # Save processed data
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tennis_output = CACHE_DIR / "processed_tennis_data.joblib"
        point_output = CACHE_DIR / "processed_point_data.joblib"

        joblib.dump(tennis_data, tennis_output, compress=('zlib', 3))
        joblib.dump(point_df, point_output, compress=('zlib', 3))

        logger.info(f"Pipeline complete: {len(tennis_data):,} matches, {len(point_df):,} points")
        return tennis_data, point_df

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main_pipeline()