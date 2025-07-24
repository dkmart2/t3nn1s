# ============================================================================
# TENNIS PIPELINE SETTINGS - ENTERPRISE CONFIGURATION
# ============================================================================

import os
from datetime import date
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Environment-overrideable base directories
TENNIS_DATA_DIR = os.getenv("TENNIS_DATA_DIR", os.path.expanduser("~/Desktop/data"))
TENNIS_CACHE_DIR = os.getenv("TENNIS_CACHE_DIR", os.path.expanduser("~/Desktop/data/cache"))
JEFF_DATA_DIR = os.getenv("JEFF_DATA_DIR", os.path.join(TENNIS_DATA_DIR, "Jeff 6.14.25"))

# Derived paths
TENNISDATA_MEN_DIR = os.path.join(TENNIS_DATA_DIR, "tennisdata_men")
TENNISDATA_WOMEN_DIR = os.path.join(TENNIS_DATA_DIR, "tennisdata_women")

# Cache subdirectories
EXCEL_CACHE_DIR = os.path.join(TENNIS_CACHE_DIR, "excel_cache")
API_CACHE_DIR = os.path.join(TENNIS_CACHE_DIR, "api_cache")
METRICS_DIR = os.path.join(TENNIS_CACHE_DIR, "metrics")

# Ensure directories exist
for directory in [TENNIS_CACHE_DIR, EXCEL_CACHE_DIR, API_CACHE_DIR, METRICS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# API-Tennis configuration
API_TENNIS_KEY = os.getenv("API_TENNIS_KEY", "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb")
BASE_API_URL = "https://api.api-tennis.com/tennis/"

# Rate limiting
API_REQUESTS_PER_SECOND = float(os.getenv("API_REQUESTS_PER_SECOND", "2.0"))
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))

# Concurrent processing
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "10000"))

# ============================================================================
# DATA PROCESSING SETTINGS
# ============================================================================

# Cutoff date for Jeff Sackmann data
BASE_CUTOFF_DATE = date(2025, 6, 10)

# Processing parameters
MEMORY_OPTIMIZATION_ENABLED = os.getenv("MEMORY_OPTIMIZATION_ENABLED", "true").lower() == "true"
USE_POLARS_FOR_LARGE_CSV = os.getenv("USE_POLARS_FOR_LARGE_CSV", "true").lower() == "true"
VECTORIZED_OPERATIONS = os.getenv("VECTORIZED_OPERATIONS", "true").lower() == "true"

# Batch processing sizes
FEATURE_EXTRACTION_BATCH_SIZE = int(os.getenv("FEATURE_EXTRACTION_BATCH_SIZE", "10000"))
JEFF_PROCESSING_BATCH_SIZE = int(os.getenv("JEFF_PROCESSING_BATCH_SIZE", "5000"))

# Data quality thresholds
MIN_DATA_QUALITY_SCORE = float(os.getenv("MIN_DATA_QUALITY_SCORE", "0.8"))
FEATURE_COVERAGE_THRESHOLD = float(os.getenv("FEATURE_COVERAGE_THRESHOLD", "0.1"))

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

# Cache TTL (Time To Live) in seconds
API_CACHE_TTL = int(os.getenv("API_CACHE_TTL", "3600"))  # 1 hour
EXCEL_CACHE_TTL = int(os.getenv("EXCEL_CACHE_TTL", "86400"))  # 24 hours
PLAYER_MAPPING_CACHE_TTL = int(os.getenv("PLAYER_MAPPING_CACHE_TTL", "604800"))  # 1 week

# Cache size limits (in MB)
MAX_CACHE_SIZE_MB = int(os.getenv("MAX_CACHE_SIZE_MB", "1000"))  # 1 GB
CACHE_CLEANUP_THRESHOLD = float(os.getenv("CACHE_CLEANUP_THRESHOLD", "0.8"))  # 80% full

# Cache validation
VALIDATE_CACHE_ON_STARTUP = os.getenv("VALIDATE_CACHE_ON_STARTUP", "true").lower() == "true"
AUTO_REBUILD_INVALID_CACHE = os.getenv("AUTO_REBUILD_INVALID_CACHE", "true").lower() == "true"

# ============================================================================
# SCHEMA VERSIONING
# ============================================================================

# Current schema versions
HISTORICAL_DATA_SCHEMA_VERSION = "2.1.0"
JEFF_DATA_SCHEMA_VERSION = "1.3.0"
WEIGHTED_DEFAULTS_SCHEMA_VERSION = "1.2.0"
PLAYER_MAPPING_SCHEMA_VERSION = "1.1.0"
SURFACE_LOOKUP_SCHEMA_VERSION = "1.0.0"

# Version compatibility
MIN_SUPPORTED_SCHEMA_VERSION = "2.0.0"
AUTO_MIGRATE_SCHEMAS = os.getenv("AUTO_MIGRATE_SCHEMAS", "true").lower() == "true"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log levels
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s")

# Log file settings
LOG_FILE_MAX_SIZE_MB = int(os.getenv("LOG_FILE_MAX_SIZE_MB", "100"))
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# Structured logging
ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
LOG_JSON_FORMAT = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"

# ============================================================================
# MONITORING AND METRICS
# ============================================================================

# Prometheus metrics
ENABLE_PROMETHEUS_METRICS = os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "60"))  # seconds

# Performance monitoring
ENABLE_PERFORMANCE_PROFILING = os.getenv("ENABLE_PERFORMANCE_PROFILING", "false").lower() == "true"
PROFILE_MEMORY_USAGE = os.getenv("PROFILE_MEMORY_USAGE", "true").lower() == "true"
PROFILE_EXECUTION_TIME = os.getenv("PROFILE_EXECUTION_TIME", "true").lower() == "true"

# Health checks
ENABLE_HEALTH_CHECKS = os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # 5 minutes

# ============================================================================
# SCRAPING CONFIGURATION
# ============================================================================

# Tennis Abstract scraping
TENNIS_ABSTRACT_BASE_URL = "https://www.tennisabstract.com"
SCRAPING_DELAY_SECONDS = float(os.getenv("SCRAPING_DELAY_SECONDS", "1.0"))
MAX_SCRAPING_RETRIES = int(os.getenv("MAX_SCRAPING_RETRIES", "3"))

# User agent for scraping
USER_AGENT = "TennisDataPipeline/2.1.0 (https://github.com/tennis-analytics/pipeline)"

# Scraping limits
MAX_MATCHES_PER_SCRAPE = int(os.getenv("MAX_MATCHES_PER_SCRAPE", "100"))
MAX_DAYS_BACK_SCRAPE = int(os.getenv("MAX_DAYS_BACK_SCRAPE", "7"))

# ============================================================================
# DATA VALIDATION SETTINGS
# ============================================================================

# Validation thresholds
MIN_MATCHES_PER_GENDER = int(os.getenv("MIN_MATCHES_PER_GENDER", "1000"))
MIN_FEATURE_COVERAGE = float(os.getenv("MIN_FEATURE_COVERAGE", "0.7"))
MAX_DUPLICATE_RATE = float(os.getenv("MAX_DUPLICATE_RATE", "0.05"))  # 5%

# Data quality checks
ENABLE_DATA_QUALITY_CHECKS = os.getenv("ENABLE_DATA_QUALITY_CHECKS", "true").lower() == "true"
STRICT_VALIDATION_MODE = os.getenv("STRICT_VALIDATION_MODE", "false").lower() == "true"
HALT_ON_VALIDATION_FAILURE = os.getenv("HALT_ON_VALIDATION_FAILURE", "false").lower() == "true"

# Anomaly detection
ENABLE_ANOMALY_DETECTION = os.getenv("ENABLE_ANOMALY_DETECTION", "true").lower() == "true"
ANOMALY_THRESHOLD_SIGMA = float(os.getenv("ANOMALY_THRESHOLD_SIGMA", "3.0"))

# ============================================================================
# FEATURE ENGINEERING SETTINGS
# ============================================================================

# Jeff feature extraction
ENABLE_JEFF_FEATURES = os.getenv("ENABLE_JEFF_FEATURES", "true").lower() == "true"
JEFF_FEATURE_TIMEOUT_SECONDS = int(os.getenv("JEFF_FEATURE_TIMEOUT_SECONDS", "300"))  # 5 minutes

# Tennis Abstract features
ENABLE_TA_FEATURES = os.getenv("ENABLE_TA_FEATURES", "true").lower() == "true"
TA_FEATURE_PRIORITY = int(os.getenv("TA_FEATURE_PRIORITY", "1"))  # Highest priority

# Surface and tournament inference
ENABLE_SURFACE_INFERENCE = os.getenv("ENABLE_SURFACE_INFERENCE", "true").lower() == "true"
ENABLE_GENDER_INFERENCE = os.getenv("ENABLE_GENDER_INFERENCE", "true").lower() == "true"
SURFACE_CONFIDENCE_THRESHOLD = float(os.getenv("SURFACE_CONFIDENCE_THRESHOLD", "0.7"))

# ============================================================================
# DEVELOPMENT AND TESTING
# ============================================================================

# Development mode
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
ENABLE_DEBUG_LOGGING = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"

# Testing
RUN_UNIT_TESTS_ON_STARTUP = os.getenv("RUN_UNIT_TESTS_ON_STARTUP", "false").lower() == "true"
ENABLE_INTEGRATION_TESTS = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"

# Synthetic data (for testing)
ALLOW_SYNTHETIC_DATA = os.getenv("ALLOW_SYNTHETIC_DATA", "false").lower() == "true"
SYNTHETIC_DATA_SIZE = int(os.getenv("SYNTHETIC_DATA_SIZE", "1000"))

# ============================================================================
# SECURITY SETTINGS
# ============================================================================

# API security
VALIDATE_SSL_CERTIFICATES = os.getenv("VALIDATE_SSL_CERTIFICATES", "true").lower() == "true"
API_KEY_ROTATION_ENABLED = os.getenv("API_KEY_ROTATION_ENABLED", "false").lower() == "true"

# File permissions
SECURE_FILE_PERMISSIONS = os.getenv("SECURE_FILE_PERMISSIONS", "true").lower() == "true"
CACHE_FILE_PERMISSIONS = 0o600  # Read/write for owner only


# ============================================================================
# RUNTIME SETTINGS VALIDATION
# ============================================================================

def validate_settings():
    """Validate settings on import"""
    errors = []

    # Check required directories
    if not os.path.exists(TENNIS_DATA_DIR):
        errors.append(f"TENNIS_DATA_DIR does not exist: {TENNIS_DATA_DIR}")

    # Check API key
    if not API_TENNIS_KEY or API_TENNIS_KEY == "your_api_key_here":
        errors.append("API_TENNIS_KEY not properly configured")

    # Validate numeric ranges
    if API_REQUESTS_PER_SECOND <= 0 or API_REQUESTS_PER_SECOND > 10:
        errors.append("API_REQUESTS_PER_SECOND must be between 0 and 10")

    if BATCH_SIZE_DEFAULT < 1000 or BATCH_SIZE_DEFAULT > 100000:
        errors.append("BATCH_SIZE_DEFAULT must be between 1000 and 100000")

    if errors:
        raise ValueError(f"Settings validation failed: {'; '.join(errors)}")


# Validate settings on import
if os.getenv("SKIP_SETTINGS_VALIDATION", "false").lower() != "true":
    validate_settings()


# ============================================================================
# FEATURE FLAGS
# ============================================================================

class FeatureFlags:
    """Feature flags for experimental functionality"""

    # Experimental features
    ENABLE_POLARS_INTEGRATION = os.getenv("FF_ENABLE_POLARS", "true").lower() == "true"
    ENABLE_ASYNC_PROCESSING = os.getenv("FF_ENABLE_ASYNC", "true").lower() == "true"
    ENABLE_VECTORIZED_FEATURES = os.getenv("FF_ENABLE_VECTORIZED", "true").lower() == "true"

    # Advanced caching
    ENABLE_DISTRIBUTED_CACHE = os.getenv("FF_DISTRIBUTED_CACHE", "false").lower() == "true"
    ENABLE_CACHE_COMPRESSION = os.getenv("FF_CACHE_COMPRESSION", "true").lower() == "true"

    # ML integration
    ENABLE_FEATURE_STORE = os.getenv("FF_FEATURE_STORE", "false").lower() == "true"
    ENABLE_MODEL_VERSIONING = os.getenv("FF_MODEL_VERSIONING", "true").lower() == "true"

    # Monitoring
    ENABLE_DISTRIBUTED_TRACING = os.getenv("FF_DISTRIBUTED_TRACING", "false").lower() == "true"
    ENABLE_CUSTOM_METRICS = os.getenv("FF_CUSTOM_METRICS", "true").lower() == "true"


# ============================================================================
# EXPORT ALL SETTINGS
# ============================================================================

__all__ = [
    # Directories
    'TENNIS_DATA_DIR', 'TENNIS_CACHE_DIR', 'JEFF_DATA_DIR',

    # API settings
    'API_TENNIS_KEY', 'BASE_API_URL', 'API_REQUESTS_PER_SECOND',

    # Processing settings
    'BASE_CUTOFF_DATE', 'BATCH_SIZE_DEFAULT', 'MAX_CONCURRENT_REQUESTS',

    # Schema versions
    'HISTORICAL_DATA_SCHEMA_VERSION', 'WEIGHTED_DEFAULTS_SCHEMA_VERSION',

    # Feature flags
    'FeatureFlags',

    # Validation
    'validate_settings'
]