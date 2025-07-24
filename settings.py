# ============================================================================
# TENNIS PIPELINE SETTINGS - CORRECTED CONFIGURATION
# ============================================================================

import os
import hashlib
from datetime import date
from pathlib import Path
from typing import List

# ============================================================================
# ENVIRONMENT-BASED DIRECTORY PATHS
# ============================================================================

# Base directories with proper environment variable handling
TENNIS_DATA_DIR = Path(os.getenv("TENNIS_DATA_DIR", Path.home() / "tennis_data"))
TENNIS_CACHE_DIR = Path(os.getenv("TENNIS_CACHE_DIR", TENNIS_DATA_DIR / "cache"))
JEFF_DATA_DIR = Path(os.getenv("JEFF_DATA_DIR", TENNIS_DATA_DIR / "jeff"))

# Derived cache subdirectories - all derive from TENNIS_CACHE_DIR
EXCEL_CACHE_DIR = TENNIS_CACHE_DIR / "excel"
API_CACHE_DIR = TENNIS_CACHE_DIR / "api"
METRICS_DIR = TENNIS_CACHE_DIR / "metrics"

# Derived data directories
TENNISDATA_MEN_DIR = TENNIS_DATA_DIR / "tennisdata_men"
TENNISDATA_WOMEN_DIR = TENNIS_DATA_DIR / "tennisdata_women"

# Ensure directories exist
for directory in [TENNIS_CACHE_DIR, EXCEL_CACHE_DIR, API_CACHE_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CONFIGURATION WITH SAFE DEFAULTS
# ============================================================================

# API-Tennis configuration
API_TENNIS_KEY = os.getenv("API_TENNIS_KEY", "")
BASE_API_URL = os.getenv("BASE_API_URL", "https://api.api-tennis.com/tennis/")

# Rate limiting with safe defaults
API_REQUESTS_PER_SECOND = float(os.getenv("API_REQUESTS_PER_SECOND", "2.0"))
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_MIN_DELAY = float(os.getenv("API_MIN_DELAY", "0.5"))

# Concurrent processing with safe defaults
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "10000"))

# ============================================================================
# DATA PROCESSING SETTINGS WITH DEFAULTS
# ============================================================================

# Cutoff date for Jeff Sackmann data
BASE_CUTOFF_DATE = date(2025, 6, 10)

# Processing parameters with safe defaults
MEMORY_OPTIMIZATION_ENABLED = os.getenv("MEMORY_OPTIMIZATION_ENABLED", "true").lower() == "true"
USE_POLARS_FOR_LARGE_CSV = os.getenv("USE_POLARS_FOR_LARGE_CSV", "true").lower() == "true"
VECTORIZED_OPERATIONS = os.getenv("VECTORIZED_OPERATIONS", "true").lower() == "true"

# Batch processing sizes with safe defaults
FEATURE_EXTRACTION_BATCH_SIZE = int(os.getenv("FEATURE_EXTRACTION_BATCH_SIZE", "10000"))
JEFF_PROCESSING_BATCH_SIZE = int(os.getenv("JEFF_PROCESSING_BATCH_SIZE", "5000"))

# Data quality thresholds with safe defaults
MIN_DATA_QUALITY_SCORE = float(os.getenv("MIN_DATA_QUALITY_SCORE", "0.8"))
FEATURE_COVERAGE_THRESHOLD = float(os.getenv("FEATURE_COVERAGE_THRESHOLD", "0.1"))

# ============================================================================
# CACHING CONFIGURATION WITH SAFE DEFAULTS
# ============================================================================

# Cache TTL (Time To Live) in seconds with safe defaults
API_CACHE_TTL = int(os.getenv("API_CACHE_TTL", "3600"))  # 1 hour
EXCEL_CACHE_TTL = int(os.getenv("EXCEL_CACHE_TTL", "86400"))  # 24 hours
PLAYER_MAPPING_CACHE_TTL = int(os.getenv("PLAYER_MAPPING_CACHE_TTL", "604800"))  # 1 week

# Cache size limits (in MB) with safe defaults
MAX_CACHE_SIZE_MB = int(os.getenv("MAX_CACHE_SIZE_MB", "1000"))  # 1 GB
CACHE_CLEANUP_THRESHOLD = float(os.getenv("CACHE_CLEANUP_THRESHOLD", "0.8"))  # 80% full

# Cache validation with safe defaults
VALIDATE_CACHE_ON_STARTUP = os.getenv("VALIDATE_CACHE_ON_STARTUP", "true").lower() == "true"
AUTO_REBUILD_INVALID_CACHE = os.getenv("AUTO_REBUILD_INVALID_CACHE", "true").lower() == "true"

# ============================================================================
# HASH-BASED SCHEMA VERSIONING
# ============================================================================

def compute_schema_version(column_list: List[str], major: str = "2", minor: str = "1") -> str:
    """Compute schema version with content hash for deterministic versioning"""
    content_hash = hashlib.sha256(str(sorted(column_list)).encode()).hexdigest()[:8]
    return f"{major}.{minor}.{content_hash}"

# Expected columns for different data schemas
HISTORICAL_DATA_EXPECTED_COLUMNS = [
    'Winner', 'Loser', 'date', 'Tournament', 'Surface', 'gender',
    'winner_canonical', 'loser_canonical', 'composite_id', 'source_rank'
]

JEFF_DATA_EXPECTED_COLUMNS = [
    'Player_canonical', 'set', 'serve_pts', 'aces', 'dfs', 'first_in',
    'first_won', 'second_won', 'return_pts', 'return_pts_won', 'winners',
    'unforced', 'winners_fh', 'winners_bh', 'unforced_fh', 'unforced_bh'
]

WEIGHTED_DEFAULTS_EXPECTED_COLUMNS = [
    'serve_pts', 'aces', 'dfs', 'first_serve_pct', 'serve_pts_won_pct',
    'ace_rate', 'df_rate', 'return_pts_won_pct', 'winner_error_ratio',
    'winners', 'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh'
]

PLAYER_MAPPING_EXPECTED_COLUMNS = ['raw_name', 'canonical_name']

SURFACE_LOOKUP_EXPECTED_COLUMNS = ['tournament_key', 'surface', 'confidence']

# Compute schema versions based on expected columns
HISTORICAL_DATA_SCHEMA_VERSION = compute_schema_version(HISTORICAL_DATA_EXPECTED_COLUMNS)
JEFF_DATA_SCHEMA_VERSION = compute_schema_version(JEFF_DATA_EXPECTED_COLUMNS)
WEIGHTED_DEFAULTS_SCHEMA_VERSION = compute_schema_version(WEIGHTED_DEFAULTS_EXPECTED_COLUMNS)
PLAYER_MAPPING_SCHEMA_VERSION = compute_schema_version(PLAYER_MAPPING_EXPECTED_COLUMNS)
SURFACE_LOOKUP_SCHEMA_VERSION = compute_schema_version(SURFACE_LOOKUP_EXPECTED_COLUMNS)

# Version compatibility with safe defaults
MIN_SUPPORTED_SCHEMA_VERSION = "2.0.00000000"
AUTO_MIGRATE_SCHEMAS = os.getenv("AUTO_MIGRATE_SCHEMAS", "true").lower() == "true"

# ============================================================================
# LOGGING CONFIGURATION WITH SAFE DEFAULTS
# ============================================================================

# Log levels with safe defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s")

# Log file settings with safe defaults
LOG_FILE_MAX_SIZE_MB = int(os.getenv("LOG_FILE_MAX_SIZE_MB", "100"))
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# Structured logging with safe defaults
ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
LOG_JSON_FORMAT = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"

# ============================================================================
# MONITORING AND METRICS WITH SAFE DEFAULTS
# ============================================================================

# Prometheus metrics with safe defaults
ENABLE_PROMETHEUS_METRICS = os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "60"))  # seconds

# Performance monitoring with safe defaults
ENABLE_PERFORMANCE_PROFILING = os.getenv("ENABLE_PERFORMANCE_PROFILING", "false").lower() == "true"
PROFILE_MEMORY_USAGE = os.getenv("PROFILE_MEMORY_USAGE", "true").lower() == "true"
PROFILE_EXECUTION_TIME = os.getenv("PROFILE_EXECUTION_TIME", "true").lower() == "true"

# Health checks with safe defaults
ENABLE_HEALTH_CHECKS = os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # 5 minutes

# ============================================================================
# SCRAPING CONFIGURATION WITH SAFE DEFAULTS
# ============================================================================

# Tennis Abstract scraping with safe defaults
TENNIS_ABSTRACT_BASE_URL = "https://www.tennisabstract.com"
SCRAPING_DELAY_SECONDS = float(os.getenv("SCRAPING_DELAY_SECONDS", "1.0"))
MAX_SCRAPING_RETRIES = int(os.getenv("MAX_SCRAPING_RETRIES", "3"))

# User agent for scraping
USER_AGENT = "TennisDataPipeline/2.1.0 (https://github.com/tennis-analytics/pipeline)"

# Scraping limits with safe defaults
MAX_MATCHES_PER_SCRAPE = int(os.getenv("MAX_MATCHES_PER_SCRAPE", "100"))
MAX_DAYS_BACK_SCRAPE = int(os.getenv("MAX_DAYS_BACK_SCRAPE", "7"))

# ============================================================================
# DATA VALIDATION SETTINGS WITH SAFE DEFAULTS
# ============================================================================

# Validation thresholds with safe defaults
MIN_MATCHES_PER_GENDER = int(os.getenv("MIN_MATCHES_PER_GENDER", "1000"))
MIN_FEATURE_COVERAGE = float(os.getenv("MIN_FEATURE_COVERAGE", "0.7"))
MAX_DUPLICATE_RATE = float(os.getenv("MAX_DUPLICATE_RATE", "0.05"))  # 5%

# Data quality checks with safe defaults
ENABLE_DATA_QUALITY_CHECKS = os.getenv("ENABLE_DATA_QUALITY_CHECKS", "true").lower() == "true"
STRICT_VALIDATION_MODE = os.getenv("STRICT_VALIDATION_MODE", "false").lower() == "true"
HALT_ON_VALIDATION_FAILURE = os.getenv("HALT_ON_VALIDATION_FAILURE", "false").lower() == "true"

# Anomaly detection with safe defaults
ENABLE_ANOMALY_DETECTION = os.getenv("ENABLE_ANOMALY_DETECTION", "true").lower() == "true"
ANOMALY_THRESHOLD_SIGMA = float(os.getenv("ANOMALY_THRESHOLD_SIGMA", "3.0"))

# ============================================================================
# FEATURE ENGINEERING SETTINGS WITH SAFE DEFAULTS
# ============================================================================

# Jeff feature extraction with safe defaults
ENABLE_JEFF_FEATURES = os.getenv("ENABLE_JEFF_FEATURES", "true").lower() == "true"
JEFF_FEATURE_TIMEOUT_SECONDS = int(os.getenv("JEFF_FEATURE_TIMEOUT_SECONDS", "300"))  # 5 minutes

# Tennis Abstract features with safe defaults
ENABLE_TA_FEATURES = os.getenv("ENABLE_TA_FEATURES", "true").lower() == "true"
TA_FEATURE_PRIORITY = int(os.getenv("TA_FEATURE_PRIORITY", "1"))  # Highest priority

# Surface and tournament inference with safe defaults
ENABLE_SURFACE_INFERENCE = os.getenv("ENABLE_SURFACE_INFERENCE", "true").lower() == "true"
ENABLE_GENDER_INFERENCE = os.getenv("ENABLE_GENDER_INFERENCE", "true").lower() == "true"
SURFACE_CONFIDENCE_THRESHOLD = float(os.getenv("SURFACE_CONFIDENCE_THRESHOLD", "0.7"))

# ============================================================================
# DEVELOPMENT AND TESTING WITH SAFE DEFAULTS
# ============================================================================

# Development mode with safe defaults
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
ENABLE_DEBUG_LOGGING = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"

# Testing with safe defaults
RUN_UNIT_TESTS_ON_STARTUP = os.getenv("RUN_UNIT_TESTS_ON_STARTUP", "false").lower() == "true"
ENABLE_INTEGRATION_TESTS = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"

# Synthetic data (for testing) with safe defaults
ALLOW_SYNTHETIC_DATA = os.getenv("ALLOW_SYNTHETIC_DATA", "false").lower() == "true"
SYNTHETIC_DATA_SIZE = int(os.getenv("SYNTHETIC_DATA_SIZE", "1000"))

# ============================================================================
# SECURITY SETTINGS WITH SAFE DEFAULTS
# ============================================================================

# API security with safe defaults
VALIDATE_SSL_CERTIFICATES = os.getenv("VALIDATE_SSL_CERTIFICATES", "true").lower() == "true"
API_KEY_ROTATION_ENABLED = os.getenv("API_KEY_ROTATION_ENABLED", "false").lower() == "true"

# File permissions with safe defaults
SECURE_FILE_PERMISSIONS = os.getenv("SECURE_FILE_PERMISSIONS", "true").lower() == "true"
CACHE_FILE_PERMISSIONS = 0o600  # Read/write for owner only

# ============================================================================
# RUNTIME SETTINGS VALIDATION
# ============================================================================

def validate_settings():
    """Validate settings on import with comprehensive checks"""
    errors = []

    # Check required directories
    if not TENNIS_DATA_DIR.exists() and not os.getenv("SKIP_DATA_DIR_CHECK"):
        errors.append(f"TENNIS_DATA_DIR does not exist: {TENNIS_DATA_DIR}")

    # Check API key if not in development mode
    if not DEVELOPMENT_MODE and (not API_TENNIS_KEY or API_TENNIS_KEY == "your_api_key_here"):
        errors.append("API_TENNIS_KEY not properly configured")

    # Validate numeric ranges
    if API_REQUESTS_PER_SECOND <= 0 or API_REQUESTS_PER_SECOND > 10:
        errors.append("API_REQUESTS_PER_SECOND must be between 0 and 10")

    if BATCH_SIZE_DEFAULT < 1000 or BATCH_SIZE_DEFAULT > 100000:
        errors.append("BATCH_SIZE_DEFAULT must be between 1000 and 100000")

    if MAX_CONCURRENT_REQUESTS < 1 or MAX_CONCURRENT_REQUESTS > 20:
        errors.append("MAX_CONCURRENT_REQUESTS must be between 1 and 20")

    # Validate paths are Path objects
    path_settings = [
        ('TENNIS_DATA_DIR', TENNIS_DATA_DIR),
        ('TENNIS_CACHE_DIR', TENNIS_CACHE_DIR),
        ('EXCEL_CACHE_DIR', EXCEL_CACHE_DIR),
        ('API_CACHE_DIR', API_CACHE_DIR)
    ]

    for name, path_obj in path_settings:
        if not isinstance(path_obj, Path):
            errors.append(f"{name} must be a Path object, got {type(path_obj)}")

    # Validate schema versions format
    schema_versions = [
        ('HISTORICAL_DATA_SCHEMA_VERSION', HISTORICAL_DATA_SCHEMA_VERSION),
        ('JEFF_DATA_SCHEMA_VERSION', JEFF_DATA_SCHEMA_VERSION),
        ('WEIGHTED_DEFAULTS_SCHEMA_VERSION', WEIGHTED_DEFAULTS_SCHEMA_VERSION)
    ]

    for name, version in schema_versions:
        if not isinstance(version, str) or len(version.split('.')) != 3:
            errors.append(f"{name} format invalid: {version}")

    if errors:
        raise ValueError(f"Settings validation failed: {'; '.join(errors)}")

# Validate settings on import unless explicitly skipped
if os.getenv("SKIP_SETTINGS_VALIDATION", "false").lower() != "true":
    validate_settings()

# ============================================================================
# FEATURE FLAGS WITH SAFE DEFAULTS
# ============================================================================

class FeatureFlags:
    """Feature flags for experimental functionality with safe defaults"""

    # Experimental features with safe defaults
    ENABLE_POLARS_INTEGRATION = os.getenv("FF_ENABLE_POLARS", "true").lower() == "true"
    ENABLE_ASYNC_PROCESSING = os.getenv("FF_ENABLE_ASYNC", "true").lower() == "true"
    ENABLE_VECTORIZED_FEATURES = os.getenv("FF_ENABLE_VECTORIZED", "true").lower() == "true"

    # Advanced caching with safe defaults
    ENABLE_DISTRIBUTED_CACHE = os.getenv("FF_DISTRIBUTED_CACHE", "false").lower() == "true"
    ENABLE_CACHE_COMPRESSION = os.getenv("FF_CACHE_COMPRESSION", "true").lower() == "true"

    # ML integration with safe defaults
    ENABLE_FEATURE_STORE = os.getenv("FF_FEATURE_STORE", "false").lower() == "true"
    ENABLE_MODEL_VERSIONING = os.getenv("FF_MODEL_VERSIONING", "true").lower() == "true"

    # Monitoring with safe defaults
    ENABLE_DISTRIBUTED_TRACING = os.getenv("FF_DISTRIBUTED_TRACING", "false").lower() == "true"
    ENABLE_CUSTOM_METRICS = os.getenv("FF_CUSTOM_METRICS", "true").lower() == "true"

# ============================================================================
# EXPORT ALL SETTINGS
# ============================================================================

__all__ = [
    # Directories
    'TENNIS_DATA_DIR', 'TENNIS_CACHE_DIR', 'JEFF_DATA_DIR',
    'EXCEL_CACHE_DIR', 'API_CACHE_DIR',

    # API settings
    'API_TENNIS_KEY', 'BASE_API_URL', 'API_REQUESTS_PER_SECOND',
    'API_MAX_RETRIES', 'API_CACHE_TTL', 'API_MIN_DELAY',

    # Processing settings
    'BASE_CUTOFF_DATE', 'BATCH_SIZE_DEFAULT', 'MAX_CONCURRENT_REQUESTS',

    # Schema versions
    'HISTORICAL_DATA_SCHEMA_VERSION', 'JEFF_DATA_SCHEMA_VERSION',
    'WEIGHTED_DEFAULTS_SCHEMA_VERSION', 'PLAYER_MAPPING_SCHEMA_VERSION',
    'SURFACE_LOOKUP_SCHEMA_VERSION',

    # Logging
    'LOG_LEVEL', 'LOG_TO_CONSOLE', 'LOG_TO_FILE',

    # Feature flags
    'FeatureFlags',

    # Schema computation
    'compute_schema_version',

    # Validation
    'validate_settings'
]