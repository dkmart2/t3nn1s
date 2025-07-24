#!/usr/bin/env python3
# ============================================================================
# TENNIS PIPELINE CLI UTILITIES - ENTERPRISE OPERATIONS
# ============================================================================

import argparse
import os
import shutil
import sys
import time
import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import logging
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import prometheus_client
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

import settings
import tennis_updated_enterprise as pipeline


# ============================================================================
# CLI UTILITIES CLASS
# ============================================================================

class TennisPipelineCLI:
    """Comprehensive CLI utilities for tennis data pipeline"""

    def __init__(self):
        self.logger = pipeline.get_logger(__name__)
        self.start_time = time.time()

    def rebuild_cache_atomic(self, backup: bool = True) -> bool:
        """Atomically rebuild all caches with optional backup"""
        self.logger.info("Starting atomic cache rebuild")

        try:
            # Step 1: Create backup if requested
            if backup:
                backup_dir = self._create_cache_backup()
                self.logger.info(f"Cache backup created: {backup_dir}")

            # Step 2: Validate prerequisites
            self._validate_rebuild_prerequisites()

            # Step 3: Wipe existing caches
            self._wipe_caches()

            # Step 4: Rebuild in dependency order
            success = self._rebuild_caches_ordered()

            if success:
                self.logger.info("Cache rebuild completed successfully")
                if backup:
                    self._cleanup_old_backups()
                return True
            else:
                self.logger.error("Cache rebuild failed")
                if backup:
                    self._restore_from_backup(backup_dir)
                return False

        except Exception as e:
            self.logger.error(f"Cache rebuild failed with exception: {e}")
            if backup and 'backup_dir' in locals():
                self._restore_from_backup(backup_dir)
            return False

    def _create_cache_backup(self) -> str:
        """Create timestamped backup of current cache"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(settings.TENNIS_CACHE_DIR, f"backup_{timestamp}")

        os.makedirs(backup_dir, exist_ok=True)

        # Copy all cache files
        cache_files = [
            'historical_data*.parquet',
            'jeff_data*.pkl',
            'weighted_defaults*.pkl',
            'player_mapping_cache.pkl',
            'surface_lookup.json',
            'pointlog_cache.parquet',
            'pointlog_index.pkl'
        ]

        for pattern in cache_files:
            for file_path in Path(settings.TENNIS_CACHE_DIR).glob(pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, backup_dir)
                    self.logger.debug(f"Backed up: {file_path.name}")

        return backup_dir

    def _validate_rebuild_prerequisites(self):
        """Validate prerequisites for cache rebuild"""
        # Check disk space
        disk_usage = shutil.disk_usage(settings.TENNIS_CACHE_DIR)
        free_gb = disk_usage.free / (1024 ** 3)

        if free_gb < 5.0:  # Need at least 5GB free
            raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB free, need 5GB")

        # Check data directories exist
        if not os.path.exists(settings.TENNIS_DATA_DIR):
            raise RuntimeError(f"Data directory not found: {settings.TENNIS_DATA_DIR}")

        if not os.path.exists(settings.JEFF_DATA_DIR):
            raise RuntimeError(f"Jeff data directory not found: {settings.JEFF_DATA_DIR}")

        # Check API connectivity
        if not self._test_api_connectivity():
            self.logger.warning("API connectivity test failed - some features may be limited")

        self.logger.info("Prerequisites validation passed")

    def _test_api_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            # Simple API test
            with pipeline.get_session() as session:
                response = session.get(f"{settings.BASE_API_URL}?method=get_events&key={settings.API_TENNIS_KEY}")
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"API connectivity test failed: {e}")
            return False

    def _wipe_caches(self):
        """Safely wipe existing caches"""
        cache_patterns = [
            'historical_data*.parquet',
            'jeff_data*.pkl',
            'weighted_defaults*.pkl',
            'excel_cache_*.parquet',
            'player_mapping_cache.pkl',
            'surface_lookup.json'
        ]

        wiped_count = 0
        for pattern in cache_patterns:
            for file_path in Path(settings.TENNIS_CACHE_DIR).glob(pattern):
                try:
                    file_path.unlink()
                    wiped_count += 1
                    self.logger.debug(f"Wiped cache file: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to wipe {file_path}: {e}")

        self.logger.info(f"Wiped {wiped_count} cache files")

    def _rebuild_caches_ordered(self) -> bool:
        """Rebuild caches in dependency order"""
        try:
            # Step 1: Load and process Jeff data
            self.logger.info("Step 1: Loading Jeff comprehensive data...")
            jeff_data = pipeline.load_jeff_comprehensive_data()

            if not jeff_data:
                raise RuntimeError("Failed to load Jeff data")

            # Step 2: Calculate weighted defaults
            self.logger.info("Step 2: Calculating weighted defaults...")
            defaults, metadata = pipeline.calculate_comprehensive_weighted_defaults_versioned(jeff_data)

            # Step 3: Load tennis match data
            self.logger.info("Step 3: Loading tennis match data...")
            tennis_data = pipeline.load_all_tennis_data()

            if tennis_data.empty:
                raise RuntimeError("Failed to load tennis match data")

            # Step 4: Process match data
            self.logger.info("Step 4: Processing tennis match data...")
            tennis_data = self._process_tennis_match_data(tennis_data)

            # Step 5: Extract Jeff features (vectorized)
            self.logger.info("Step 5: Extracting Jeff features...")
            tennis_data = pipeline.inject_jeff_features_vectorized(tennis_data, jeff_data, defaults)

            # Step 6: Integrate API data
            self.logger.info("Step 6: Integrating API data...")
            date_range = self._get_api_date_range()
            api_fixtures = asyncio.run(pipeline.fetch_fixtures_async_batch(date_range))

            if api_fixtures:
                tennis_data = self._integrate_api_fixtures(tennis_data, api_fixtures)

            # Step 7: Run Tennis Abstract integration
            self.logger.info("Step 7: Running Tennis Abstract integration...")
            scraper = pipeline.ProductionTennisAbstractScraper()
            scraped_records = scraper.automated_scraping_session(days_back=7)

            if scraped_records:
                tennis_data = pipeline.integrate_scraped_data_hybrid(tennis_data, scraped_records)

            # Step 8: Final deduplication
            self.logger.info("Step 8: Final deduplication...")
            tennis_data = pipeline.deduplicate_matches(tennis_data)

            # Step 9: Save all caches
            self.logger.info("Step 9: Saving caches...")
            self._save_all_caches(tennis_data, jeff_data, defaults, metadata)

            self.logger.info("Cache rebuild completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Cache rebuild failed: {e}")
            return False

    def _process_tennis_match_data(self, tennis_data: pd.DataFrame) -> pd.DataFrame:
        """Process tennis match data with optimizations"""
        # Add canonical names using persistent cache
        player_cache = pipeline.PlayerMappingCache()

        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(player_cache.get_canonical_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(player_cache.get_canonical_name)

        # Process dates
        if 'Date' in tennis_data.columns:
            tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
            tennis_data['date'] = tennis_data['Date'].dt.date

        # Drop rows with invalid dates
        tennis_data = tennis_data.dropna(subset=['date'])

        # Add surface inference
        surface_lookup = pipeline.SurfaceLookupTable()
        if 'Tournament' in tennis_data.columns:
            tennis_data['Surface'] = tennis_data.apply(
                lambda row: surface_lookup.get_surface(
                    row.get('Tournament', ''),
                    season=row['date'].month if pd.notna(row['date']) else None
                ), axis=1
            )

        self.logger.info(f"Processed {len(tennis_data):,} tennis matches")
        return tennis_data

    def _get_api_date_range(self) -> List[date]:
        """Get date range for API data fetching"""
        start_date = settings.BASE_CUTOFF_DATE
        end_date = date.today()

        date_range = []
        current = start_date
        while current <= end_date:
            date_range.append(current)
            current += timedelta(days=1)

        # Limit to reasonable range for rebuild
        if len(date_range) > 30:
            date_range = date_range[-30:]  # Last 30 days

        return date_range

    def _integrate_api_fixtures(self, tennis_data: pd.DataFrame, fixtures: List[Dict]) -> pd.DataFrame:
        """Integrate API fixtures into tennis data"""
        api_matches = []

        for fixture in fixtures:
            try:
                match_data = self._convert_fixture_to_match(fixture)
                if match_data:
                    api_matches.append(match_data)
            except Exception as e:
                self.logger.warning(f"Failed to convert fixture: {e}")

        if api_matches:
            api_df = pd.DataFrame(api_matches)
            tennis_data = pd.concat([tennis_data, api_df], ignore_index=True)
            self.logger.info(f"Integrated {len(api_matches)} API matches")

        return tennis_data

    def _convert_fixture_to_match(self, fixture: Dict) -> Optional[Dict]:
        """Convert API fixture to match format"""
        try:
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                return None

            match_date = pd.to_datetime(fixture.get('date_start')).date()

            return {
                'composite_id': f"{match_date.strftime('%Y%m%d')}-api-{fixture.get('fixture_key', 'unknown')}",
                'date': match_date,
                'Winner': participants[0].get('name'),
                'Loser': participants[1].get('name'),
                'winner_canonical': pipeline.player_cache.get_canonical_name(participants[0].get('name', '')),
                'loser_canonical': pipeline.player_cache.get_canonical_name(participants[1].get('name', '')),
                'Tournament': fixture.get('event_name'),
                'Surface': pipeline.surface_lookup.get_surface(fixture.get('event_name', '')),
                'gender': 'M',  # Would need better inference
                'source_rank': 2,  # API priority
                'api_fixture_id': fixture.get('fixture_key')
            }
        except Exception as e:
            self.logger.warning(f"Failed to convert fixture: {e}")
            return None

    def _save_all_caches(self, tennis_data: pd.DataFrame, jeff_data: Dict,
                         defaults: Dict, metadata) -> None:
        """Save all cache files"""

        # Save historical data
        hd_path = os.path.join(settings.TENNIS_CACHE_DIR,
                               f"historical_data_v{settings.HISTORICAL_DATA_SCHEMA_VERSION}.parquet")
        tennis_data.to_parquet(hd_path, index=False, compression='snappy')

        # Save Jeff data
        jeff_path = os.path.join(settings.TENNIS_CACHE_DIR,
                                 f"jeff_data_v{settings.JEFF_DATA_SCHEMA_VERSION}.pkl")
        with open(jeff_path, 'wb') as f:
            pickle.dump(jeff_data, f, protocol=4)

        # Save weighted defaults with metadata
        def_path = os.path.join(settings.TENNIS_CACHE_DIR,
                                f"weighted_defaults_v{settings.WEIGHTED_DEFAULTS_SCHEMA_VERSION}.pkl")
        with open(def_path, 'wb') as f:
            pickle.dump((defaults, metadata), f, protocol=4)

        self.logger.info("All caches saved successfully")

    def _restore_from_backup(self, backup_dir: str):
        """Restore caches from backup directory"""
        self.logger.info(f"Restoring caches from backup: {backup_dir}")

        for backup_file in Path(backup_dir).glob('*'):
            if backup_file.is_file():
                dest_path = os.path.join(settings.TENNIS_CACHE_DIR, backup_file.name)
                shutil.copy2(backup_file, dest_path)
                self.logger.debug(f"Restored: {backup_file.name}")

        self.logger.info("Cache restore completed")

    def _cleanup_old_backups(self, keep_count: int = 5):
        """Cleanup old backup directories"""
        backup_dirs = []
        for path in Path(settings.TENNIS_CACHE_DIR).glob('backup_*'):
            if path.is_dir():
                backup_dirs.append(path)

        # Sort by modification time (newest first)
        backup_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old backups
        for old_backup in backup_dirs[keep_count:]:
            try:
                shutil.rmtree(old_backup)
                self.logger.info(f"Cleaned up old backup: {old_backup.name}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup backup {old_backup}: {e}")

    def validate_data_comprehensive(self) -> Dict:
        """Run comprehensive data validation"""
        self.logger.info("Starting comprehensive data validation")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_status': 'PASS',
            'warnings': [],
            'errors': []
        }

        try:
            # Load current data
            historical_data = pd.read_parquet(
                os.path.join(settings.TENNIS_CACHE_DIR,
                             f"historical_data_v{settings.HISTORICAL_DATA_SCHEMA_VERSION}.parquet")
            )

            # Check 1: Data completeness
            completeness = self._validate_data_completeness(historical_data)
            validation_results['checks']['completeness'] = completeness

            # Check 2: Data quality
            quality = self._validate_data_quality(historical_data)
            validation_results['checks']['quality'] = quality

            # Check 3: Schema compliance
            schema = self._validate_schema_compliance(historical_data)
            validation_results['checks']['schema'] = schema

            # Check 4: Feature coverage
            features = self._validate_feature_coverage(historical_data)
            validation_results['checks']['features'] = features

            # Check 5: Temporal consistency
            temporal = self._validate_temporal_consistency(historical_data)
            validation_results['checks']['temporal'] = temporal

            # Determine overall status
            failed_checks = [k for k, v in validation_results['checks'].items()
                             if v.get('status') == 'FAIL']

            if failed_checks:
                validation_results['overall_status'] = 'FAIL'
                validation_results['errors'].extend(failed_checks)

            self.logger.info(f"Data validation completed: {validation_results['overall_status']}")

        except Exception as e:
            validation_results['overall_status'] = 'ERROR'
            validation_results['errors'].append(f"Validation failed: {e}")
            self.logger.error(f"Data validation error: {e}")

        return validation_results

    def _validate_data_completeness(self, data: pd.DataFrame) -> Dict:
        """Validate data completeness"""
        total_matches = len(data)

        # Check gender distribution
        gender_counts = data['gender'].value_counts()
        men_matches = gender_counts.get('M', 0)
        women_matches = gender_counts.get('W', 0)

        # Check required columns
        required_cols = ['Winner', 'Loser', 'date', 'Tournament']
        missing_cols = [col for col in required_cols if col not in data.columns]

        return {
            'status': 'PASS' if not missing_cols and total_matches > 1000 else 'FAIL',
            'total_matches': total_matches,
            'men_matches': men_matches,
            'women_matches': women_matches,
            'missing_columns': missing_cols,
            'min_threshold': 1000
        }

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Validate data quality metrics"""
        # Duplicate rate
        duplicate_rate = 1 - (data['composite_id'].nunique() / len(data))

        # Null rates for key columns
        null_rates = {}
        for col in ['Winner', 'Loser', 'date']:
            if col in data.columns:
                null_rates[col] = data[col].isna().mean()

        # Invalid date rate
        if 'date' in data.columns:
            invalid_dates = data['date'].isna().sum()
            invalid_date_rate = invalid_dates / len(data)
        else:
            invalid_date_rate = 1.0

        quality_issues = []
        if duplicate_rate > 0.05:  # 5% threshold
            quality_issues.append(f"High duplicate rate: {duplicate_rate:.1%}")

        if invalid_date_rate > 0.01:  # 1% threshold
            quality_issues.append(f"High invalid date rate: {invalid_date_rate:.1%}")

        return {
            'status': 'PASS' if not quality_issues else 'FAIL',
            'duplicate_rate': duplicate_rate,
            'null_rates': null_rates,
            'invalid_date_rate': invalid_date_rate,
            'issues': quality_issues
        }

    def _validate_schema_compliance(self, data: pd.DataFrame) -> Dict:
        """Validate schema compliance"""
        expected_dtypes = {
            'Winner': 'object',
            'Loser': 'object',
            'gender': 'object',
            'source_rank': 'int64'
        }

        dtype_issues = []
        for col, expected_dtype in expected_dtypes.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                if expected_dtype not in actual_dtype:
                    dtype_issues.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")

        return {
            'status': 'PASS' if not dtype_issues else 'FAIL',
            'dtype_issues': dtype_issues,
            'columns_count': len(data.columns)
        }

    def _validate_feature_coverage(self, data: pd.DataFrame) -> Dict:
        """Validate Jeff feature coverage"""
        expected_features = pipeline.get_expected_jeff_features()

        missing_features = [f for f in expected_features if f not in data.columns]
        present_features = [f for f in expected_features if f in data.columns]

        # Calculate coverage rates
        coverage_rates = {}
        for feature in present_features:
            coverage_rates[feature] = data[feature].notna().mean()

        low_coverage_features = [f for f, rate in coverage_rates.items()
                                 if rate < settings.FEATURE_COVERAGE_THRESHOLD]

        return {
            'status': 'PASS' if len(missing_features) < 5 and len(low_coverage_features) < 10 else 'WARN',
            'missing_features': len(missing_features),
            'present_features': len(present_features),
            'low_coverage_features': len(low_coverage_features),
            'avg_coverage': sum(coverage_rates.values()) / len(coverage_rates) if coverage_rates else 0
        }

    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Dict:
        """Validate temporal consistency"""
        if 'date' not in data.columns:
            return {'status': 'FAIL', 'error': 'No date column'}

        valid_dates = data['date'].dropna()

        if valid_dates.empty:
            return {'status': 'FAIL', 'error': 'No valid dates'}

        min_date = valid_dates.min()
        max_date = valid_dates.max()
        date_range_days = (max_date - min_date).days if isinstance(max_date, date) else 0

        # Check for future dates
        today = date.today()
        future_dates = sum(1 for d in valid_dates if isinstance(d, date) and d > today)

        return {
            'status': 'PASS' if future_dates == 0 else 'WARN',
            'min_date': str(min_date),
            'max_date': str(max_date),
            'date_range_days': date_range_days,
            'future_dates': future_dates
        }

    def export_prometheus_metrics(self, output_file: Optional[str] = None) -> str:
        """Export Prometheus metrics"""
        try:
            metrics_data = generate_latest()

            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(metrics_data)
                self.logger.info(f"Metrics exported to: {output_file}")
            else:
                # Print to stdout
                print(metrics_data.decode('utf-8'))

            return metrics_data.decode('utf-8')

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return ""

    def system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'system': {},
            'cache': {},
            'data': {}
        }

        # System info
        status['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage(settings.TENNIS_CACHE_DIR).percent
        }

        # Cache info
        cache_files = {
            'historical_data': f"historical_data_v{settings.HISTORICAL_DATA_SCHEMA_VERSION}.parquet",
            'jeff_data': f"jeff_data_v{settings.JEFF_DATA_SCHEMA_VERSION}.pkl",
            'weighted_defaults': f"weighted_defaults_v{settings.WEIGHTED_DEFAULTS_SCHEMA_VERSION}.pkl"
        }

        for cache_name, filename in cache_files.items():
            cache_path = os.path.join(settings.TENNIS_CACHE_DIR, filename)
            if os.path.exists(cache_path):
                stat = os.stat(cache_path)
                status['cache'][cache_name] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024 ** 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                status['cache'][cache_name] = {'exists': False}

        return status


# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================

def main():
    """Main CLI entry point with comprehensive argument parsing"""

    parser = argparse.ArgumentParser(
        description="Tennis Data Pipeline - Enterprise CLI Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rebuild all caches with backup
  python cli_utilities.py --rebuild-cache --backup

  # Run comprehensive validation
  python cli_utilities.py --validate-data --output validation_report.json

  # Export Prometheus metrics
  python cli_utilities.py --export-metrics --output metrics.txt

  # Show system status
  python cli_utilities.py --status
        """
    )

    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--rebuild-cache', action='store_true',
                              help='Rebuild all caches atomically')
    action_group.add_argument('--validate-data', action='store_true',
                              help='Run comprehensive data validation')
    action_group.add_argument('--export-metrics', action='store_true',
                              help='Export Prometheus metrics')
    action_group.add_argument('--status', action='store_true',
                              help='Show system status')

    # Options
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before rebuild (use with --rebuild-cache)')
    parser.add_argument('--output', type=str,
                        help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of concurrent workers')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize CLI
    cli = TennisPipelineCLI()

    try:
        if args.rebuild_cache:
            success = cli.rebuild_cache_atomic(backup=args.backup)
            return 0 if success else 1

        elif args.validate_data:
            results = cli.validate_data_comprehensive()

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Validation results saved to: {args.output}")
            else:
                print(json.dumps(results, indent=2))

            return 0 if results['overall_status'] == 'PASS' else 1

        elif args.export_metrics:
            cli.export_prometheus_metrics(args.output)
            return 0

        elif args.status:
            status = cli.system_status()

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(status, f, indent=2)
                print(f"Status saved to: {args.output}")
            else:
                print(json.dumps(status, indent=2))

            return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        cli.logger.error(f"CLI operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())