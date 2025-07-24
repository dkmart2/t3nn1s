# ============================================================================
# TENNIS DATA PIPELINE - DATA LAYER CORRECTIONS
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
import argparse
import collections
import json
import hashlib
import asyncio
import aiohttp
import httpx
import requests_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
import pytest
from dataclasses import dataclass, asdict
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import settings
from contextlib import contextmanager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# SETTINGS SAFEGUARDS WITH DEFAULTS
# ============================================================================

def get_setting(key: str, default: Any) -> Any:
    """Safely get setting with fallback default"""
    return getattr(settings, key, default)


# Core directories with safe defaults
TENNIS_DATA_DIR = Path(os.getenv("TENNIS_DATA_DIR", Path.home() / "tennis_data"))
TENNIS_CACHE_DIR = Path(get_setting('TENNIS_CACHE_DIR', TENNIS_DATA_DIR / "cache"))
EXCEL_CACHE_DIR = Path(get_setting('EXCEL_CACHE_DIR', TENNIS_CACHE_DIR / "excel"))
API_CACHE_DIR = Path(get_setting('API_CACHE_DIR', TENNIS_CACHE_DIR / "api"))
JEFF_DATA_DIR = Path(get_setting('JEFF_DATA_DIR', TENNIS_DATA_DIR / "jeff"))

# API settings with safe defaults
API_TENNIS_KEY = get_setting('API_TENNIS_KEY', "")
API_MAX_RETRIES = get_setting('API_MAX_RETRIES', 3)
API_CACHE_TTL = get_setting('API_CACHE_TTL', 3600)
API_MIN_DELAY = get_setting('API_MIN_DELAY', 0.5)
BASE_API_URL = get_setting('BASE_API_URL', "https://api.api-tennis.com/tennis/")

# Processing settings
MAX_CONCURRENT_REQUESTS = get_setting('MAX_CONCURRENT_REQUESTS', 5)
BASE_CUTOFF_DATE = get_setting('BASE_CUTOFF_DATE', date(2025, 6, 10))

# Logging settings
LOG_LEVEL = get_setting('LOG_LEVEL', 'INFO')
LOG_TO_CONSOLE = get_setting('LOG_TO_CONSOLE', True)
LOG_TO_FILE = get_setting('LOG_TO_FILE', True)


# Schema versions from settings with hash-based computation
def compute_schema_version(column_list: List[str]) -> str:
    """Compute schema version with content hash"""
    major, minor = "2", "1"
    content_hash = hashlib.sha256(str(sorted(column_list)).encode()).hexdigest()[:8]
    return f"{major}.{minor}.{content_hash}"


expected_columns = ['Winner', 'Loser', 'date', 'Tournament', 'Surface', 'gender']
SCHEMA_VERSION = get_setting('HISTORICAL_DATA_SCHEMA_VERSION', compute_schema_version(expected_columns))
WEIGHTED_DEFAULTS_SCHEMA_VERSION = get_setting('WEIGHTED_DEFAULTS_SCHEMA_VERSION', "1.2.a1b2c3d4")

# Ensure directories exist
for directory in [TENNIS_CACHE_DIR, EXCEL_CACHE_DIR, API_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PROMETHEUS METRICS WITH DUPLICATION GUARDS
# ============================================================================

def create_metric_if_not_exists(metric_type, name, description, labels=None):
    """Create metric only if not already registered"""
    if name not in prometheus_client.REGISTRY._names_to_collectors:
        if metric_type == 'histogram':
            return Histogram(name, description)
        elif metric_type == 'counter':
            return Counter(name, description, labels or [])
        elif metric_type == 'gauge':
            return Gauge(name, description)
    else:
        return prometheus_client.REGISTRY._names_to_collectors[name]


FEATURE_EXTRACTION_TIME = create_metric_if_not_exists('histogram', 'tennis_feature_extraction_seconds',
                                                      'Time spent extracting features')
API_REQUESTS_TOTAL = create_metric_if_not_exists('counter', 'tennis_api_requests_total', 'Total API requests',
                                                 ['method', 'status'])
CACHE_HITS_TOTAL = create_metric_if_not_exists('counter', 'tennis_cache_hits_total', 'Cache hits', ['cache_type'])
DATA_QUALITY_SCORE = create_metric_if_not_exists('gauge', 'tennis_data_quality_score', 'Data quality score')
MATCHES_PROCESSED = create_metric_if_not_exists('counter', 'tennis_matches_processed_total', 'Total matches processed')
SCRAPING_ERRORS = create_metric_if_not_exists('counter', 'tennis_scraping_errors_total', 'Scraping errors', ['source'])


# ============================================================================
# LAZY JSON LOGGING WITH PERFORMANCE OPTIMIZATION
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get properly configured logger with lazy JSON formatting"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))

        import uuid
        correlation_id = str(uuid.uuid4())[:8]

        class LazyStructuredFormatter(logging.Formatter):
            def format(self, record):
                # Only format as JSON if logging level permits
                if logger.isEnabledFor(logging.INFO):
                    log_entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": record.levelname,
                        "module": record.name,
                        "function": record.funcName,
                        "line": record.lineno,
                        "message": record.getMessage(),
                        "correlation_id": correlation_id
                    }
                    return json.dumps(log_entry)
                else:
                    return record.getMessage()

        formatter = LazyStructuredFormatter()

        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if LOG_TO_FILE:
            file_handler = logging.FileHandler(
                TENNIS_CACHE_DIR / "tennis_pipeline.log", mode='a'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class TennisDataError(Exception):
    """Base exception for tennis data pipeline"""
    pass


class CacheError(TennisDataError):
    """Cache-related errors"""
    pass


class DataIngestionError(TennisDataError):
    """Data ingestion errors"""
    pass


class APIError(TennisDataError):
    """API-related errors"""
    pass


class FeatureExtractionError(TennisDataError):
    """Feature extraction errors"""
    pass


# ============================================================================
# CACHED SESSION WITH RETRY STRATEGY
# ============================================================================

def get_cached_session() -> requests_cache.CachedSession:
    """Get cached session with exponential backoff"""
    session = requests_cache.CachedSession(
        cache_name=str(API_CACHE_DIR / 'api_cache'),
        expire_after=API_CACHE_TTL
    )

    retry_strategy = Retry(
        total=API_MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ============================================================================
# THREAD-SAFE PLAYER CANONICALIZER WITH FILE LOCKING
# ============================================================================

class PlayerCanonicalizer:
    """Thread-safe player name canonicalization with file locking"""

    def __init__(self):
        self.cache_file = TENNIS_CACHE_DIR / "player_canonical_cache.joblib"
        self.lock_file = str(self.cache_file) + ".lock"
        self.mapping = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Load player mapping from compressed cache with file locking"""
        try:
            with filelock.FileLock(self.lock_file):
                if self.cache_file.exists():
                    return joblib.load(self.cache_file)
        except Exception as e:
            logger.warning(f"Failed to load player canonical cache: {e}")
        return {}

    def canonical_player(self, raw_name: str) -> str:
        """Single source of truth for player name canonicalization"""
        if pd.isna(raw_name):
            return ""

        raw_name = str(raw_name).strip()
        if not raw_name:
            return ""

        if raw_name in self.mapping:
            CACHE_HITS_TOTAL.labels(cache_type='player_canonical').inc()
            return self.mapping[raw_name]

        # Compute canonical name
        canonical = self._normalize_name(raw_name)
        self.mapping[raw_name] = canonical

        # Persist to disk every 100 new mappings
        if len(self.mapping) % 100 == 0:
            self._save_cache()

        return canonical

    def _normalize_name(self, name: str) -> str:
        """Normalize player name with comprehensive rules"""
        name = unidecode(name).replace('.', '').replace("'", '').replace('-', ' ')
        name = ' '.join(name.lower().split())
        return name

    def _save_cache(self):
        """Save player mapping to compressed cache with file locking"""
        try:
            with filelock.FileLock(self.lock_file):
                joblib.dump(self.mapping, self.cache_file, compress=('zlib', 3))
        except Exception as e:
            logger.error(f"Failed to save player canonical cache: {e}")


player_canonicalizer = PlayerCanonicalizer()


# ============================================================================
# DETERMINISTIC CONTENT-BASED CACHE INVALIDATION
# ============================================================================

def get_content_hash(file_path: Path) -> str:
    """Generate SHA-256 hash of file content for deterministic invalidation"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        # Fallback to mtime if content can't be read
        return hashlib.sha256(str(file_path.stat().st_mtime).encode()).hexdigest()[:16]


def get_cache_path(file_path: Path, prefix: str = "cache") -> Path:
    """Generate cache path with schema version and content hash"""
    content_hash = get_content_hash(file_path)
    filename = f"{prefix}_{SCHEMA_VERSION}_{content_hash}.joblib"
    return EXCEL_CACHE_DIR / filename


# ============================================================================
# OPTIMIZED EXCEL CACHING WITH SCHEMA VERSIONING
# ============================================================================

def load_excel_with_compressed_cache(file_path: Path, **read_excel_kwargs) -> pd.DataFrame:
    """Load Excel with compressed joblib caching and schema versioning"""

    cache_path = get_cache_path(file_path, "excel")

    # Check if cached version exists and matches schema
    if cache_path.exists():
        try:
            cached_df = joblib.load(cache_path)
            CACHE_HITS_TOTAL.labels(cache_type='excel_compressed').inc()
            logger.info(f"Using cached data for {file_path}")
            return cached_df
        except Exception as e:
            logger.warning(f"Cache read failed for {file_path}: {e}")

    # Load Excel with optimized parameters
    with FEATURE_EXTRACTION_TIME.time():
        try:
            df = pd.read_excel(
                file_path,
                engine='openpyxl' if str(file_path).endswith('.xlsx') else 'xlrd',
                dtype=get_optimized_dtypes(),
                **read_excel_kwargs
            )
        except Exception as e:
            raise DataIngestionError(f"Failed to load Excel file {file_path}: {e}")

    # Convert string columns to categorical for memory efficiency
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')

    # Cache with compression and schema version
    try:
        joblib.dump(df, cache_path, compress=('zlib', 3))
        logger.info(f"Cached Excel as compressed joblib: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cache data for {file_path}: {e}")

    return df


# ============================================================================
# DYNAMIC THREAD POOL SIZING
# ============================================================================

def get_optimal_workers() -> int:
    """Calculate optimal worker count based on CPU and settings"""
    cpu_count = os.cpu_count() or 2
    return min(MAX_CONCURRENT_REQUESTS, cpu_count * 2)


# ============================================================================
# VECTORIZED DATA LOADING WITH PROPER ERROR HANDLING
# ============================================================================

def load_all_tennis_data() -> pd.DataFrame:
    """Vectorized loading of all tennis match data with error aggregation"""

    logger.info("Starting vectorized tennis data loading")

    dataframes = []
    errors = []

    # Men's data
    men_files = []
    for year in range(2020, 2026):
        file_path = TENNIS_DATA_DIR / f"{year}_m.xlsx"
        if file_path.exists():
            men_files.append(file_path)

    # Women's data
    women_files = []
    for year in range(2020, 2026):
        file_path = TENNIS_DATA_DIR / f"{year}_w.xlsx"
        if file_path.exists():
            women_files.append(file_path)

    # Historical data
    historical_files = [
        TENNIS_DATA_DIR / "match_history_men.csv",
        TENNIS_DATA_DIR / "match_history_women.csv"
    ]

    all_files = men_files + women_files + [f for f in historical_files if f.exists()]

    # Parallel loading with optimal worker count
    with ThreadPoolExecutor(max_workers=get_optimal_workers()) as executor:
        future_to_file = {}

        for file_path in all_files:
            if str(file_path).endswith('.xlsx'):
                future = executor.submit(load_excel_with_compressed_cache, file_path)
            else:
                future = executor.submit(load_large_csv_with_polars, file_path)
            future_to_file[future] = file_path

        for future in future_to_file:
            file_path = future_to_file[future]
            try:
                df = future.result(timeout=60)

                # Add metadata
                if 'men' in str(file_path) or '_m.xlsx' in str(file_path):
                    df['gender'] = 'M'
                elif 'women' in str(file_path) or '_w.xlsx' in str(file_path):
                    df['gender'] = 'W'
                else:
                    df['gender'] = 'M'  # Default

                # Add canonical names using unified canonicalizer
                if 'Winner' in df.columns:
                    df['winner_canonical'] = df['Winner'].apply(player_canonicalizer.canonical_player)
                if 'Loser' in df.columns:
                    df['loser_canonical'] = df['Loser'].apply(player_canonicalizer.canonical_player)

                dataframes.append(df)
                logger.info(f"Loaded {len(df):,} matches from {file_path}")

            except Exception as e:
                error_msg = f"Failed to load {file_path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

    if not dataframes:
        raise DataIngestionError(f"No data loaded successfully. Errors: {errors}")

    if errors:
        logger.warning(f"Data loading completed with {len(errors)} errors: {errors}")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)

    # Process dates uniformly
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
        combined_df['date'] = combined_df['Date'].dt.date
        combined_df = combined_df.dropna(subset=['date'])

    logger.info(f"Combined tennis data: {len(combined_df):,} total matches")
    return combined_df


# ============================================================================
# OPTIMIZED POLARS CSV LOADING WITH SPECIFIC ERROR HANDLING
# ============================================================================

def load_large_csv_with_polars(file_path: Path, **kwargs) -> pd.DataFrame:
    """Load large CSV using Polars with specific error handling"""

    cache_path = get_cache_path(file_path, "csv_polars")

    if cache_path.exists():
        try:
            cached_df = joblib.load(cache_path)
            CACHE_HITS_TOTAL.labels(cache_type='csv_polars').inc()
            return cached_df
        except Exception as e:
            logger.warning(f"Polars cache read failed for {file_path}: {e}")

    try:
        with FEATURE_EXTRACTION_TIME.time():
            # Load with Polars and apply categorical casting at Polars level
            df_polars = pl.read_csv(
                str(file_path),
                infer_schema_length=10000,
                **kwargs
            )

            # Convert string columns to categorical at Polars level for efficiency
            df_polars = df_polars.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))

            # Convert to Pandas - categoricals are preserved
            df_pandas = df_polars.to_pandas()

            # Apply optimized numeric dtypes
            dtype_map = get_optimized_dtypes()
            for col in df_pandas.columns:
                if col in dtype_map and col in df_pandas.columns:
                    try:
                        df_pandas[col] = df_pandas[col].astype(dtype_map[col])
                    except Exception:
                        continue

        # Cache the result with compression
        try:
            joblib.dump(df_pandas, cache_path, compress=('zlib', 3))
        except Exception as e:
            logger.warning(f"Failed to cache Polars CSV for {file_path}: {e}")

        memory_mb = df_pandas.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info(f"Loaded CSV with Polars: {file_path}, {len(df_pandas):,} rows, {memory_mb:.1f}MB")

        return df_pandas

    except (pl.exceptions.ColumnNotFoundError, FileNotFoundError) as e:
        raise DataIngestionError(f"Polars CSV loading failed for {file_path}: {e}")
    except Exception as e:
        logger.error(f"Polars CSV loading failed, falling back to pandas: {e}")
        try:
            df = pd.read_csv(file_path, **kwargs)
            # Apply categorical conversion manually for pandas fallback
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            return df
        except Exception as fallback_error:
            raise DataIngestionError(f"Both Polars and pandas failed for {file_path}: {fallback_error}")


# ============================================================================
# OPTIMIZED DTYPE MAPPING
# ============================================================================

def get_optimized_dtypes() -> Dict[str, str]:
    """Return optimized dtypes with expanded coverage"""
    return {
        'ATP': 'int16', 'WTA': 'int16',
        'WRank': 'float32', 'LRank': 'float32',
        'W1': 'int8', 'L1': 'int8', 'W2': 'int8', 'L2': 'int8',
        'W3': 'int8', 'L3': 'int8', 'W4': 'int8', 'L4': 'int8',
        'W5': 'int8', 'L5': 'int8', 'Wsets': 'int8', 'Lsets': 'int8',
        'Best of': 'int8',
        'B365W': 'float32', 'B365L': 'float32',
        'PSW': 'float32', 'PSL': 'float32',
        'MaxW': 'float32', 'MaxL': 'float32',
        'AvgW': 'float32', 'AvgL': 'float32',
        'serve_pts': 'int16', 'aces': 'int16', 'dfs': 'int16',
        'first_in': 'int16', 'first_won': 'int16', 'second_in': 'int16',
        'second_won': 'int16', 'return_pts': 'int16', 'return_pts_won': 'int16',
        'winners': 'int16', 'winners_fh': 'int16', 'winners_bh': 'int16',
        'unforced': 'int16', 'unforced_fh': 'int16', 'unforced_bh': 'int16'
    }


# ============================================================================
# VECTORIZED JEFF FEATURE EXTRACTION WITH METADATA VERSIONING
# ============================================================================

@dataclass
class WeightedDefaultsMetadata:
    schema_version: str
    creation_date: str
    jeff_data_hash: str
    column_count: int
    men_features: int
    women_features: int


def load_jeff_comprehensive_data():
    """Load Jeff charting data with proper error handling and metadata versioning"""

    cache_path = TENNIS_CACHE_DIR / f"jeff_data_v{SCHEMA_VERSION}.joblib"

    if cache_path.exists():
        try:
            jeff_data = joblib.load(cache_path)
            CACHE_HITS_TOTAL.labels(cache_type='jeff_data').inc()
            logger.info("Loaded Jeff data from cache")
            return jeff_data
        except Exception as e:
            logger.warning(f"Failed to load Jeff data from cache: {e}")

    if not JEFF_DATA_DIR.exists():
        raise DataIngestionError(f"Jeff data directory not found: {JEFF_DATA_DIR}")

    jeff_data = {'men': {}, 'women': {}}

    # Define file patterns for each gender
    file_patterns = {
        'men': ['chartingm*.csv'],
        'women': ['chartingw*.csv']
    }

    for gender, patterns in file_patterns.items():
        for pattern in patterns:
            for file_path in JEFF_DATA_DIR.glob(pattern):
                try:
                    filename = file_path.stem
                    df = load_large_csv_with_polars(file_path)

                    if not df.empty:
                        jeff_data[gender][filename] = df
                        logger.info(f"Loaded Jeff {gender} data: {filename} ({len(df):,} rows)")

                except Exception as e:
                    logger.error(f"Failed to load Jeff file {file_path}: {e}")

    # Cache the loaded data with compression
    try:
        joblib.dump(jeff_data, cache_path, compress=('zlib', 3))
        logger.info("Cached Jeff comprehensive data")
    except Exception as e:
        logger.warning(f"Failed to cache Jeff data: {e}")

    return jeff_data


def calculate_comprehensive_weighted_defaults_versioned(jeff_data: Dict) -> Tuple[Dict, WeightedDefaultsMetadata]:
    """Calculate weighted defaults with schema versioning and metadata validation"""

    # Calculate data hash for cache invalidation
    jeff_hash = hashlib.sha256(str(sorted(jeff_data.keys())).encode()).hexdigest()[:16]

    # Check if cached version exists and is valid
    cache_path = TENNIS_CACHE_DIR / f"weighted_defaults_v{WEIGHTED_DEFAULTS_SCHEMA_VERSION}.joblib"

    if cache_path.exists():
        try:
            cached_data = joblib.load(cache_path)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                defaults, metadata = cached_data
                # Validate metadata
                if (metadata.schema_version == WEIGHTED_DEFAULTS_SCHEMA_VERSION and
                        metadata.jeff_data_hash == jeff_hash):
                    CACHE_HITS_TOTAL.labels(cache_type='weighted_defaults').inc()
                    logger.info("Using cached weighted defaults")
                    return defaults, metadata
        except Exception as e:
            logger.warning(f"Failed to load cached weighted defaults: {e}")

    defaults = {"men": {}, "women": {}}
    skip = {"matches", "points_2020s", "points_2010s", "points_to2009"}

    total_columns = 0

    for sex in ("men", "women"):
        all_numeric_data = []

        for name, df in jeff_data.get(sex, {}).items():
            if name in skip or df is None or df.empty:
                continue

            # Select numeric columns only
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                all_numeric_data.append(numeric_df)
                total_columns += len(numeric_df.columns)

        if all_numeric_data:
            # Combine all numeric data for this gender
            combined_numeric = pd.concat(all_numeric_data, ignore_index=True, sort=False)

            # Vectorized mean calculation
            defaults[sex] = combined_numeric.mean().to_dict()

    # Create metadata
    metadata = WeightedDefaultsMetadata(
        schema_version=WEIGHTED_DEFAULTS_SCHEMA_VERSION,
        creation_date=datetime.now().isoformat(),
        jeff_data_hash=jeff_hash,
        column_count=total_columns,
        men_features=len(defaults["men"]),
        women_features=len(defaults["women"])
    )

    # Cache with metadata
    try:
        joblib.dump((defaults, metadata), cache_path, compress=('zlib', 3))
        logger.info("Cached weighted defaults with metadata")
    except Exception as e:
        logger.warning(f"Failed to cache weighted defaults: {e}")

    return defaults, metadata


def inject_jeff_features_vectorized(df: pd.DataFrame, jeff_data: Dict, weighted_defaults: Dict) -> pd.DataFrame:
    """Fully vectorized Jeff feature injection using merge operations"""

    logger.info("Starting fully vectorized Jeff feature injection")

    if df.empty:
        return df

    with FEATURE_EXTRACTION_TIME.time():
        # Pre-compute feature matrices for both genders
        feature_matrices = {}

        for gender in ['M', 'W']:
            gender_key = 'men' if gender == 'M' else 'women'

            if gender_key not in jeff_data or 'overview' not in jeff_data[gender_key]:
                continue

            overview_df = jeff_data[gender_key]['overview']
            if overview_df.empty:
                continue

            total_stats = overview_df[overview_df['set'] == 'Total'].copy()
            if total_stats.empty:
                continue

            # Vectorized feature computation
            features_matrix = compute_features_vectorized(total_stats)
            features_matrix['gender'] = gender

            feature_matrices[gender] = features_matrix

        # Combine all feature matrices
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

        # Fill missing values with weighted defaults
        expected_features = get_expected_jeff_features()
        for feature in expected_features:
            if feature in df.columns:
                gender_col = 'gender'
                if gender_col in df.columns:
                    # Vectorized fillna by gender
                    for gender in ['M', 'W']:
                        gender_key = 'men' if gender == 'M' else 'women'
                        mask = df['gender'] == gender
                        default_val = weighted_defaults.get(gender_key, {}).get(
                            feature.replace('winner_', '').replace('loser_', ''), 0
                        )
                        df.loc[mask, feature] = df.loc[mask, feature].fillna(default_val)

    MATCHES_PROCESSED.inc(len(df))
    logger.info(f"Vectorized Jeff feature injection complete: {len(df):,} matches")

    return df


def compute_features_vectorized(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorized feature computation"""

    features_df = stats_df.copy()

    # Vectorized ratio calculations
    serve_pts = features_df['serve_pts'].replace(0, 1)  # Avoid division by zero

    features_df['first_serve_pct'] = features_df['first_in'] / serve_pts
    features_df['serve_pts_won_pct'] = (features_df['first_won'] + features_df['second_won']) / serve_pts
    features_df['ace_rate'] = features_df['aces'] / serve_pts
    features_df['df_rate'] = features_df['dfs'] / serve_pts

    # Return points calculations
    return_pts = features_df['return_pts'].replace(0, 1)
    features_df['return_pts_won_pct'] = features_df['return_pts_won'] / return_pts

    # Winner/error ratios
    features_df['winner_error_ratio'] = features_df['winners'] / features_df['unforced'].replace(0, 1)

    return features_df


# ============================================================================
# ASYNC API WITH UNIFIED ERROR HANDLING AND DELAYS
# ============================================================================

async def fetch_fixtures_async_batch(date_range: List[date], batch_size: int = 5) -> List[Dict]:
    """Async batch API fetching with delays and unified error handling"""

    logger.info(f"Starting async batch fixture fetch: {len(date_range)} dates")

    all_fixtures = []

    # Use httpx.AsyncClient with specific timeout and retry settings
    timeout = httpx.Timeout(30.0, connect=10.0)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Process in batches
        for i in range(0, len(date_range), batch_size):
            batch_dates = date_range[i:i + batch_size]

            tasks = []
            for target_date in batch_dates:
                task = fetch_fixtures_for_date_async(client, target_date)
                tasks.append(task)

            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, APIError):
                        API_REQUESTS_TOTAL.labels(method='get_fixtures', status='error').inc()
                        logger.error(f"API error: {result}")
                    elif isinstance(result, Exception):
                        SCRAPING_ERRORS.labels(source='api').inc()
                        logger.error(f"Unexpected error: {result}")
                    elif isinstance(result, list):
                        all_fixtures.extend(result)

                # Exponential backoff between batches
                await asyncio.sleep(min(2 ** (i // batch_size), 10))

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                SCRAPING_ERRORS.labels(source='api_batch').inc()

    logger.info(f"Async batch fetch complete: {len(all_fixtures)} fixtures")
    return all_fixtures


async def fetch_fixtures_for_date_async(client: httpx.AsyncClient, target_date: date) -> List[Dict]:
    """Async fetch fixtures with mandatory delays and unified error handling"""

    try:
        # Get tennis events with mandatory delay
        events_url = f"{BASE_API_URL}?method=get_events&key={API_TENNIS_KEY}"

        try:
            events_response = await client.get(events_url)
            await asyncio.sleep(API_MIN_DELAY)  # Mandatory delay between calls
            events_response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise APIError(f"Events request failed: {type(e).__name__}: {e}")

        events_data = events_response.json()

        # Find tennis events
        tennis_events = [
            event for event in events_data
            if any(term in event.get('event_type_type', '').lower()
                   for term in ['atp', 'wta', 'tennis', 'singles'])
        ]

        all_fixtures = []

        # Fetch fixtures for each tennis event (limit to avoid rate limits)
        for event in tennis_events[:3]:
            fixtures_url = (
                f"{BASE_API_URL}?method=get_fixtures"
                f"&key={API_TENNIS_KEY}"
                f"&date_start={target_date.isoformat()}"
                f"&date_stop={target_date.isoformat()}"
                f"&event_type_key={event.get('event_type_key')}"
                f"&timezone=UTC"
            )

            try:
                fixtures_response = await client.get(fixtures_url)
                await asyncio.sleep(API_MIN_DELAY)  # Mandatory delay between calls
                fixtures_response.raise_for_status()
                fixtures_data = fixtures_response.json()

                if isinstance(fixtures_data, list):
                    all_fixtures.extend(fixtures_data)

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                raise APIError(f"Fixtures request failed for {event.get('event_type_key')}: {type(e).__name__}: {e}")

        API_REQUESTS_TOTAL.labels(method='get_fixtures', status='success').inc()
        return all_fixtures

    except APIError:
        raise
    except Exception as e:
        API_REQUESTS_TOTAL.labels(method='get_fixtures', status='error').inc()
        raise APIError(f"Unexpected error fetching fixtures for {target_date}: {e}")


# ============================================================================
# ENHANCED DEDUPLICATION WITH CATEGORICAL CONVERSION TRACKING
# ============================================================================

def deduplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced deduplication with categorical conversion tracking and data quality metrics"""

    if df.empty or 'composite_id' not in df.columns:
        return df

    original_count = len(df)

    with FEATURE_EXTRACTION_TIME.time():
        # Check if categorical conversion already done
        if not df.attrs.get('cats_to_str_done', False):
            # Convert categorical columns to string for groupby operations
            categorical_cols = df.select_dtypes(include=['category']).columns
            for col in categorical_cols:
                df[col] = df[col].astype(str)
            df.attrs['cats_to_str_done'] = True

        # Sort by source_rank (ascending) so higher priority sources come first
        df_sorted = df.sort_values(['composite_id', 'source_rank'])

        # Keep first occurrence (highest priority) of each composite_id
        df_deduped = df_sorted.drop_duplicates(subset=['composite_id'], keep='first')

        # Convert back to categorical for memory efficiency if conversion was done
        if df.attrs.get('cats_to_str_done', False):
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df_deduped.columns and df_deduped[col].nunique() / len(df_deduped) < 0.5:
                    df_deduped[col] = df_deduped[col].astype('category')

    duplicates_removed = original_count - len(df_deduped)

    if duplicates_removed > 0:
        logger.info(f"Deduplication complete: {duplicates_removed} duplicates removed, {len(df_deduped)} remaining")

        # Update data quality score - moved inside deduplication as requested
        quality_score = len(df_deduped) / original_count if original_count > 0 else 1.0
        DATA_QUALITY_SCORE.set(quality_score)

    return df_deduped


# ============================================================================
# UTILITY FUNCTIONS WITH PROPER TYPE ANNOTATIONS
# ============================================================================

def get_expected_jeff_features() -> List[str]:
    """Return list of expected Jeff feature columns"""
    base_features = [
        'serve_pts', 'aces', 'dfs', 'first_serve_pct', 'serve_pts_won_pct',
        'ace_rate', 'df_rate', 'return_pts_won_pct', 'winner_error_ratio',
        'winners', 'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh'
    ]

    expected = []
    for prefix in ['winner_', 'loser_']:
        for feature in base_features:
            expected.append(f"{prefix}{feature}")

    return expected


def get_fallback_defaults(gender_key: str) -> Dict[str, float]:
    """Get fallback default values with type hints"""
    base_defaults = {
        'serve_pts': 80.0, 'aces': 6.0, 'dfs': 3.0, 'first_serve_pct': 0.62,
        'serve_pts_won_pct': 0.65, 'ace_rate': 0.075, 'df_rate': 0.04,
        'return_pts_won_pct': 0.38, 'winner_error_ratio': 1.14,
        'winners': 25.0, 'winners_fh': 15.0, 'winners_bh': 10.0,
        'unforced': 22.0, 'unforced_fh': 12.0, 'unforced_bh': 10.0
    }

    if gender_key == 'women':
        base_defaults.update({
            'serve_pts': 75.0, 'aces': 4.0, 'first_serve_pct': 0.60,
            'serve_pts_won_pct': 0.62, 'ace_rate': 0.053
        })

    return base_defaults


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def main_pipeline() -> pd.DataFrame:
    """Main pipeline execution with comprehensive error handling"""

    logger.info("Starting tennis data pipeline")

    try:
        # Step 1: Load all tennis data
        logger.info("Step 1: Loading tennis match data")
        tennis_data = load_all_tennis_data()

        if tennis_data.empty:
            raise DataIngestionError("No tennis data loaded")

        # Step 2: Load Jeff data
        logger.info("Step 2: Loading Jeff comprehensive data")
        jeff_data = load_jeff_comprehensive_data()

        # Step 3: Calculate weighted defaults
        logger.info("Step 3: Calculating weighted defaults")
        weighted_defaults, metadata = calculate_comprehensive_weighted_defaults_versioned(jeff_data)

        # Step 4: Inject Jeff features
        logger.info("Step 4: Injecting Jeff features")
        tennis_data = inject_jeff_features_vectorized(tennis_data, jeff_data, weighted_defaults)

        # Step 5: Deduplicate matches (data quality metrics updated inside)
        logger.info("Step 5: Deduplicating matches")
        tennis_data = deduplicate_matches(tennis_data)

        # Step 6: Save processed data
        logger.info("Step 6: Saving processed data")
        output_path = TENNIS_CACHE_DIR / f"processed_tennis_data_v{SCHEMA_VERSION}.joblib"
        joblib.dump(tennis_data, output_path, compress=('zlib', 3))

        logger.info(f"Pipeline complete: {len(tennis_data):,} matches processed")
        return tennis_data

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main_pipeline()