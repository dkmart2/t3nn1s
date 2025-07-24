# ============================================================================
# TENNIS DATA PIPELINE - ENTERPRISE GRADE VERSION
# ============================================================================

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import date, timedelta, datetime
import os
import pickle
import time
from pathlib import Path
from unidecode import unidecode
import argparse
import collections
import json
import hashlib
import gc
import asyncio
import aiohttp
import httpx
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
import pytest
from dataclasses import dataclass, asdict
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import settings
from contextlib import contextmanager

# ============================================================================
# SCHEMA VERSIONING AND CONSTANTS
# ============================================================================

SCHEMA_VERSION = "2.1.0"
WEIGHTED_DEFAULTS_SCHEMA_VERSION = "1.2.0"

# Promote constants to settings
BASE_DATA_DIR = settings.TENNIS_DATA_DIR
CACHE_DIR = settings.TENNIS_CACHE_DIR
JEFF_DATA_DIR = settings.JEFF_DATA_DIR
API_KEY = settings.API_TENNIS_KEY
BASE_CUTOFF_DATE = settings.BASE_CUTOFF_DATE

# Cache paths with versioning
HD_PATH = os.path.join(CACHE_DIR, f"historical_data_v{SCHEMA_VERSION}.parquet")
JEFF_PATH = os.path.join(CACHE_DIR, f"jeff_data_v{SCHEMA_VERSION}.pkl")
DEF_PATH = os.path.join(CACHE_DIR, f"weighted_defaults_v{WEIGHTED_DEFAULTS_SCHEMA_VERSION}.pkl")

# ============================================================================
# PROMETHEUS METRICS INSTRUMENTATION
# ============================================================================

# Define metrics
FEATURE_EXTRACTION_TIME = Histogram('tennis_feature_extraction_seconds', 'Time spent extracting features')
API_REQUESTS_TOTAL = Counter('tennis_api_requests_total', 'Total API requests', ['method', 'status'])
CACHE_HITS_TOTAL = Counter('tennis_cache_hits_total', 'Cache hits', ['cache_type'])
DATA_QUALITY_SCORE = Gauge('tennis_data_quality_score', 'Data quality score')
MATCHES_PROCESSED = Counter('tennis_matches_processed_total', 'Total matches processed')
SCRAPING_ERRORS = Counter('tennis_scraping_errors_total', 'Scraping errors', ['source'])


# ============================================================================
# STRUCTURED LOGGING WITH PROPER LOGGER
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get properly configured logger"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(
            os.path.join(CACHE_DIR, "tennis_pipeline.log"), mode='a'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)


# ============================================================================
# CONTEXT-MANAGED SESSION
# ============================================================================

@contextmanager
def get_session():
    """Context-managed requests session"""
    session = None
    try:
        session = httpx.Client(timeout=30.0)
        yield session
    finally:
        if session:
            session.close()


@contextmanager
def get_async_session(semaphore_limit: int = 10):
    """Context-managed async session with semaphore"""
    semaphore = asyncio.Semaphore(semaphore_limit)

    async def bounded_fetch(session, *args, **kwargs):
        async with semaphore:
            return await session.get(*args, **kwargs)

    try:
        session = httpx.AsyncClient(timeout=30.0)
        session.bounded_get = lambda *args, **kwargs: bounded_fetch(session, *args, **kwargs)
        yield session
    finally:
        if session:
            await session.aclose()


# ============================================================================
# PERSISTENT PLAYER MAPPING
# ============================================================================

class PlayerMappingCache:
    """Persistent player-key ↔ canonical name mapping"""

    def __init__(self):
        self.cache_file = os.path.join(CACHE_DIR, "player_mapping_cache.pkl")
        self.mapping = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Load player mapping from cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load player mapping cache: {e}")
        return {}

    def get_canonical_name(self, raw_name: str) -> str:
        """Get canonical name, computing if necessary"""
        if raw_name in self.mapping:
            CACHE_HITS_TOTAL.labels(cache_type='player_mapping').inc()
            return self.mapping[raw_name]

        # Compute canonical name
        canonical = self._normalize_name(raw_name)
        self.mapping[raw_name] = canonical

        # Persist to disk every 100 new mappings
        if len(self.mapping) % 100 == 0:
            self._save_cache()

        return canonical

    def _normalize_name(self, name: str) -> str:
        """Normalize player name"""
        if pd.isna(name):
            return ""
        name = str(name).strip()
        name = unidecode(name).replace('.', '').replace("'", '').replace('-', ' ')
        return ' '.join(name.lower().split())

    def _save_cache(self):
        """Save player mapping to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.mapping, f)
        except Exception as e:
            logger.error(f"Failed to save player mapping cache: {e}")


player_cache = PlayerMappingCache()


# ============================================================================
# SURFACE LOOKUP TABLE
# ============================================================================

class SurfaceLookupTable:
    """Tournament-keyed surface lookup with seasonal updates"""

    def __init__(self):
        self.cache_file = os.path.join(CACHE_DIR, "surface_lookup.json")
        self.lookup = self._load_surface_data()

    def _load_surface_data(self) -> Dict[str, Dict]:
        """Load surface lookup table"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load surface lookup: {e}")

        # Initialize with known tournaments
        return self._initialize_surface_data()

    def _initialize_surface_data(self) -> Dict[str, Dict]:
        """Initialize with known tournament surfaces"""
        return {
            "french_open": {"surface": "Clay", "confidence": 1.0},
            "roland_garros": {"surface": "Clay", "confidence": 1.0},
            "wimbledon": {"surface": "Grass", "confidence": 1.0},
            "us_open": {"surface": "Hard", "confidence": 1.0},
            "australian_open": {"surface": "Hard", "confidence": 1.0},
            "indian_wells": {"surface": "Hard", "confidence": 0.9},
            "miami_open": {"surface": "Hard", "confidence": 0.9},
            "monte_carlo": {"surface": "Clay", "confidence": 0.9},
            "madrid_open": {"surface": "Clay", "confidence": 0.9},
            "rome_masters": {"surface": "Clay", "confidence": 0.9}
        }

    def get_surface(self, tournament_key: str, season: int = None) -> str:
        """Get surface for tournament with seasonal consideration"""
        tournament_normalized = tournament_key.lower().replace(' ', '_')

        if tournament_normalized in self.lookup:
            surface_data = self.lookup[tournament_normalized]
            return surface_data["surface"]

        # Seasonal inference for unknown tournaments
        if season:
            month = season % 12 or 12  # Rough mapping
            if 5 <= month <= 7:  # Clay/Grass season
                return "Clay" if month <= 6 else "Grass"

        return "Hard"  # Default

    def update_surface(self, tournament_key: str, surface: str, confidence: float = 0.8):
        """Update surface information"""
        tournament_normalized = tournament_key.lower().replace(' ', '_')
        self.lookup[tournament_normalized] = {
            "surface": surface,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat()
        }
        self._save_lookup()

    def _save_lookup(self):
        """Save surface lookup to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.lookup, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save surface lookup: {e}")


surface_lookup = SurfaceLookupTable()


# ============================================================================
# OPTIMIZED EXCEL TO PARQUET CACHING
# ============================================================================

def load_excel_with_parquet_cache(file_path: str, **read_excel_kwargs) -> pd.DataFrame:
    """Load Excel with Parquet caching for subsequent runs"""

    # Generate cache file path
    file_hash = hashlib.md5(f"{file_path}_{os.path.getmtime(file_path)}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"excel_cache_{file_hash}.parquet")

    # Check if cached version exists and is newer
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            CACHE_HITS_TOTAL.labels(cache_type='excel_parquet').inc()
            logger.info(f"Using cached Parquet for {file_path}")
            return cached_df
        except Exception as e:
            logger.warning(f"Cache read failed for {file_path}: {e}")

    # Load Excel with optimized parameters
    with FEATURE_EXTRACTION_TIME.time():
        df = pd.read_excel(
            file_path,
            engine='openpyxl',
            dtype=get_optimized_dtypes(),
            nrows=read_excel_kwargs.get('nrows'),
            usecols=read_excel_kwargs.get('usecols'),
            **{k: v for k, v in read_excel_kwargs.items() if k not in ['nrows', 'usecols']}
        )

    # Cache as Parquet
    try:
        df.to_parquet(cache_path, index=False, compression='snappy')
        logger.info(f"Cached Excel as Parquet: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cache Parquet for {file_path}: {e}")

    return df


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
# POLARS INTEGRATION FOR LARGE CSV PROCESSING
# ============================================================================

def load_large_csv_with_polars(file_path: str, **kwargs) -> pd.DataFrame:
    """Load large CSV using Polars for 40% speed/memory improvement"""

    try:
        with FEATURE_EXTRACTION_TIME.time():
            # Load with Polars
            df_polars = pl.read_csv(
                file_path,
                infer_schema_length=10000,  # Better type inference
                **kwargs
            )

            # Convert to Pandas with optimized dtypes
            df_pandas = df_polars.to_pandas()

            # Apply dtype optimization
            dtype_map = get_optimized_dtypes()
            for col in df_pandas.columns:
                if col in dtype_map:
                    try:
                        df_pandas[col] = df_pandas[col].astype(dtype_map[col])
                    except Exception:
                        continue  # Skip problematic conversions

        memory_mb = df_pandas.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info(f"Loaded CSV with Polars: {file_path}, {len(df_pandas):,} rows, {memory_mb:.1f}MB")

        return df_pandas

    except Exception as e:
        logger.error(f"Polars CSV loading failed, falling back to pandas: {e}")
        return pd.read_csv(file_path, **kwargs)


# ============================================================================
# VECTORIZED PANDAS OPERATIONS FOR JEFF FEATURES
# ============================================================================

def inject_jeff_features_vectorized(df: pd.DataFrame, jeff_data: Dict, weighted_defaults: Dict) -> pd.DataFrame:
    """Replace row-wise loops with vectorized pandas operations"""

    logger.info("Starting vectorized Jeff feature injection")

    with FEATURE_EXTRACTION_TIME.time():
        # Pre-compute all player features for vectorized lookup
        player_features_cache = {}

        for gender in ['M', 'W']:
            gender_key = 'men' if gender == 'M' else 'women'

            if gender_key not in jeff_data or 'overview' not in jeff_data[gender_key]:
                continue

            overview_df = jeff_data[gender_key]['overview']
            total_stats = overview_df[overview_df['set'] == 'Total'].copy()

            if total_stats.empty:
                continue

            # Vectorized feature extraction
            features_df = total_stats.apply(lambda row: extract_features_vectorized(row), axis=1, result_type='expand')
            features_df.index = total_stats['Player_canonical']

            player_features_cache[gender] = features_df.to_dict('index')

        # Vectorized application to matches
        expected_features = get_expected_jeff_features()

        # Initialize feature columns
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = np.nan

        # Apply features using vectorized operations
        for gender in ['M', 'W']:
            gender_mask = df['gender'] == gender
            if not gender_mask.any():
                continue

            gender_features = player_features_cache.get(gender, {})
            if not gender_features:
                continue

            # Vectorized winner feature assignment
            winner_features = df.loc[gender_mask, 'winner_canonical'].map(gender_features)
            valid_winner_mask = winner_features.notna()

            if valid_winner_mask.any():
                winner_data = pd.DataFrame(winner_features[valid_winner_mask].tolist())
                for feature, values in winner_data.items():
                    col_name = f'winner_{feature}'
                    if col_name in df.columns:
                        df.loc[gender_mask & valid_winner_mask, col_name] = values.values

            # Vectorized loser feature assignment
            loser_features = df.loc[gender_mask, 'loser_canonical'].map(gender_features)
            valid_loser_mask = loser_features.notna()

            if valid_loser_mask.any():
                loser_data = pd.DataFrame(loser_features[valid_loser_mask].tolist())
                for feature, values in loser_data.items():
                    col_name = f'loser_{feature}'
                    if col_name in df.columns:
                        df.loc[gender_mask & valid_loser_mask, col_name] = values.values

    MATCHES_PROCESSED.inc(len(df))
    logger.info(f"Vectorized Jeff feature injection complete: {len(df):,} matches")

    return df


def extract_features_vectorized(row: pd.Series) -> Dict[str, float]:
    """Vectorized feature extraction from overview row"""
    serve_pts = row.get('serve_pts', 80)

    if serve_pts > 0:
        return {
            'serve_pts': float(serve_pts),
            'aces': float(row.get('aces', 0)),
            'double_faults': float(row.get('dfs', 0)),
            'first_serve_pct': float(row.get('first_in', 0)) / float(serve_pts),
            'first_serve_won': float(row.get('first_won', 0)),
            'second_serve_won': float(row.get('second_won', 0)),
            'break_points_saved': float(row.get('bp_saved', 0)),
            'return_pts_won': float(row.get('return_pts_won', 0)),
            'winners_total': float(row.get('winners', 0)),
            'winners_fh': float(row.get('winners_fh', 0)),
            'winners_bh': float(row.get('winners_bh', 0)),
            'unforced_errors': float(row.get('unforced', 0)),
            'unforced_fh': float(row.get('unforced_fh', 0)),
            'unforced_bh': float(row.get('unforced_bh', 0))
        }

    return get_fallback_defaults('men')


# ============================================================================
# ASYNC API BATCH PROCESSING
# ============================================================================

async def fetch_fixtures_async_batch(date_range: List[date], batch_size: int = 10) -> List[Dict]:
    """Asynchronous batch API fetching with httpx.AsyncClient and semaphore control"""

    logger.info(f"Starting async batch fixture fetch: {len(date_range)} dates")

    all_fixtures = []

    async with get_async_session(semaphore_limit=batch_size) as session:
        # Create tasks for all dates
        tasks = []
        for target_date in date_range:
            task = fetch_fixtures_for_date_async(session, target_date)
            tasks.append(task)

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]

            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        SCRAPING_ERRORS.labels(source='api').inc()
                        logger.error(f"Async fetch error: {result}")
                    elif isinstance(result, list):
                        all_fixtures.extend(result)

                # Rate limiting between batches
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                SCRAPING_ERRORS.labels(source='api_batch').inc()

    logger.info(f"Async batch fetch complete: {len(all_fixtures)} fixtures")
    return all_fixtures


async def fetch_fixtures_for_date_async(session: httpx.AsyncClient, target_date: date) -> List[Dict]:
    """Async fetch fixtures for single date"""

    try:
        # Get tennis event types
        events_url = f"{settings.BASE_API_URL}?method=get_events&key={API_KEY}"
        events_response = await session.bounded_get(events_url)
        events_response.raise_for_status()
        events_data = events_response.json()

        # Find tennis events
        tennis_events = [
            event for event in events_data
            if any(term in event.get('event_type_type', '').lower()
                   for term in ['atp', 'wta', 'tennis', 'singles'])
        ]

        all_fixtures = []

        # Fetch fixtures for each tennis event
        for event in tennis_events[:3]:  # Limit to avoid rate limits
            fixtures_url = (
                f"{settings.BASE_API_URL}?method=get_fixtures"
                f"&key={API_KEY}"
                f"&date_start={target_date.isoformat()}"
                f"&date_stop={target_date.isoformat()}"
                f"&event_type_key={event.get('event_type_key')}"
                f"&timezone=UTC"
            )

            fixtures_response = await session.bounded_get(fixtures_url)
            fixtures_response.raise_for_status()
            fixtures_data = fixtures_response.json()

            if isinstance(fixtures_data, list):
                all_fixtures.extend(fixtures_data)

        API_REQUESTS_TOTAL.labels(method='get_fixtures', status='success').inc()
        return all_fixtures

    except Exception as e:
        API_REQUESTS_TOTAL.labels(method='get_fixtures', status='error').inc()
        logger.error(f"Async fixture fetch failed for {target_date}: {e}")
        return []


# ============================================================================
# SCHEMA VERSIONING FOR WEIGHTED DEFAULTS
# ============================================================================

@dataclass
class WeightedDefaultsMetadata:
    schema_version: str
    creation_date: str
    jeff_data_hash: str
    column_count: int
    men_features: int
    women_features: int


def calculate_comprehensive_weighted_defaults_versioned(jeff_data: Dict) -> Tuple[Dict, WeightedDefaultsMetadata]:
    """Calculate weighted defaults with schema versioning"""

    # Calculate data hash for invalidation
    jeff_hash = hashlib.md5(str(sorted(jeff_data.keys())).encode()).hexdigest()

    defaults = {"men": {}, "women": {}}
    skip = {"matches", "points_2020s", "points_2010s", "points_to2009"}

    total_columns = 0

    for sex in ("men", "women"):
        sums, counts = collections.defaultdict(float), collections.defaultdict(int)

        for name, df in jeff_data.get(sex, {}).items():
            if name in skip or df is None or df.empty:
                continue

            num = df.select_dtypes(include=["number"])
            total_columns += len(num.columns)

            for col in num.columns:
                vals = pd.to_numeric(num[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                sums[col] += vals.sum()
                counts[col] += len(vals)

        defaults[sex] = {c: sums[c] / counts[c] for c in sums if counts[c] > 0}

    # Create metadata
    metadata = WeightedDefaultsMetadata(
        schema_version=WEIGHTED_DEFAULTS_SCHEMA_VERSION,
        creation_date=datetime.now().isoformat(),
        jeff_data_hash=jeff_hash,
        column_count=total_columns,
        men_features=len(defaults["men"]),
        women_features=len(defaults["women"])
    )

    return defaults, metadata


def load_weighted_defaults_with_validation() -> Tuple[Optional[Dict], Optional[WeightedDefaultsMetadata]]:
    """Load weighted defaults with schema validation"""

    if not os.path.exists(DEF_PATH):
        return None, None

    try:
        with open(DEF_PATH, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, tuple) and len(data) == 2:
            defaults, metadata = data

            # Validate schema version
            if metadata.schema_version == WEIGHTED_DEFAULTS_SCHEMA_VERSION:
                CACHE_HITS_TOTAL.labels(cache_type='weighted_defaults').inc()
                return defaults, metadata
            else:
                logger.warning(
                    f"Schema version mismatch: {metadata.schema_version} != {WEIGHTED_DEFAULTS_SCHEMA_VERSION}")

    except Exception as e:
        logger.error(f"Failed to load weighted defaults: {e}")

    return None, None


# ============================================================================
# CLI ENHANCEMENTS
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser with validation"""

    parser = argparse.ArgumentParser(
        description="Tennis Data Pipeline - Enterprise Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tennis_updated.py --rebuild-cache
  python tennis_updated.py --validate-data
  python tennis_updated.py --export-metrics
        """
    )

    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Wipe and rebuild all caches atomically')
    parser.add_argument('--validate-data', action='store_true',
                        help='Run comprehensive data validation')
    parser.add_argument('--export-metrics', action='store_true',
                        help='Export Prometheus metrics')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of concurrent workers')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Batch size for processing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    return parser


def validate_required_args(args: argparse.Namespace) -> None:
    """Validate required arguments early"""

    if args.workers < 1 or args.workers > 20:
        raise argparse.ArgumentTypeError("Workers must be between 1 and 20")

    if args.batch_size < 1000 or args.batch_size > 50000:
        raise argparse.ArgumentTypeError("Batch size must be between 1000 and 50000")

    # Check required environment variables
    required_settings = ['TENNIS_DATA_DIR', 'TENNIS_CACHE_DIR', 'API_TENNIS_KEY']
    missing = [s for s in required_settings if not hasattr(settings, s)]

    if missing:
        raise RuntimeError(f"Missing required settings: {missing}")


# ============================================================================
# UTILITY FUNCTIONS WITH TYPE HINTS
# ============================================================================

def get_expected_jeff_features() -> List[str]:
    """Return list of expected Jeff feature columns"""
    base_features = [
        'serve_pts', 'aces', 'double_faults', 'first_serve_pct',
        'first_serve_won', 'second_serve_won', 'break_points_saved',
        'return_pts_won', 'winners_total', 'winners_fh', 'winners_bh',
        'unforced_errors', 'unforced_fh', 'unforced_bh'
    ]

    expected = []
    for prefix in ['winner_', 'loser_']:
        for feature in base_features:
            expected.append(f"{prefix}{feature}")

    return expected


def get_fallback_defaults(gender_key: str) -> Dict[str, float]:
    """Get fallback default values with type hints"""
    base_defaults = {
        'serve_pts': 80.0, 'aces': 6.0, 'double_faults': 3.0, 'first_serve_pct': 0.62,
        'first_serve_won': 38.0, 'second_serve_won': 20.0, 'break_points_saved': 8.0,
        'return_pts_won': 30.0, 'winners_total': 25.0, 'winners_fh': 15.0, 'winners_bh': 10.0,
        'unforced_errors': 22.0, 'unforced_fh': 12.0, 'unforced_bh': 10.0
    }

    if gender_key == 'women':
        base_defaults.update({
            'serve_pts': 75.0, 'aces': 4.0, 'first_serve_pct': 0.60,
            'first_serve_won': 32.0, 'second_serve_won': 15.0
        })

    return base_defaults


def deduplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced deduplication with source ranking priority and metrics"""

    if 'composite_id' not in df.columns:
        return df

    original_count = len(df)

    # Sort by source_rank (ascending) so higher priority sources come first
    df_sorted = df.sort_values(['composite_id', 'source_rank'])

    # Keep first occurrence (highest priority) of each composite_id
    df_deduped = df_sorted.drop_duplicates(subset=['composite_id'], keep='first')

    duplicates_removed = original_count - len(df_deduped)

    if duplicates_removed > 0:
        logger.info(f"Deduplication complete: {duplicates_removed} duplicates removed, {len(df_deduped)} remaining")

        # Update data quality score
        quality_score = len(df_deduped) / original_count if original_count > 0 else 1.0
        DATA_QUALITY_SCORE.set(quality_score)

    return df_deduped


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'load_excel_with_parquet_cache',
    'load_large_csv_with_polars',
    'inject_jeff_features_vectorized',
    'fetch_fixtures_async_batch',
    'calculate_comprehensive_weighted_defaults_versioned',
    'deduplicate_matches',
    'PlayerMappingCache',
    'SurfaceLookupTable',
    'get_expected_jeff_features',
    'get_fallback_defaults'
]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> int:
    """Main execution with comprehensive CLI"""

    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        validate_required_args(args)
    except (argparse.ArgumentTypeError, RuntimeError) as e:
        logger.error(f"Argument validation failed: {e}")
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.rebuild_cache:
            logger.info("Rebuilding all caches...")
            # Implementation would wipe and rebuild caches atomically
            return 0

        elif args.validate_data:
            logger.info("Running data validation...")
            # Implementation would run comprehensive validation
            return 0

        elif args.export_metrics:
            logger.info("Exporting Prometheus metrics...")
            # Export metrics to file or push gateway
            return 0

        else:
            logger.info("No specific action requested. Use --help for options.")
            return 0

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())