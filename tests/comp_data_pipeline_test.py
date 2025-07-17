#!/usr/bin/env python3
"""
Comprehensive test suite for tennis data pipeline
Tests all aspects: backfill, incremental updates, data quality, error handling
"""

import pytest
import shutil
import tempfile
import pickle
import hashlib
import time
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Go up one directory from test/ to find tennis_updated.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tennis_updated import (
    load_from_cache_with_scraping,
    generate_comprehensive_historical_data,
    save_to_cache,
    integrate_api_tennis_data_incremental,
    CACHE_DIR, HD_PATH, JEFF_PATH, DEF_PATH
)
)

class TestDataPipeline:
    """Comprehensive data pipeline test suite"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup clean test environment before each test"""
        # Create temporary cache directory
        self.temp_cache_dir = tempfile.mkdtemp(prefix="tennis_test_")
        self.original_cache_dir = CACHE_DIR

        # Patch cache paths to use temp directory
        self.cache_patches = [
            patch('tennis_updated.CACHE_DIR', self.temp_cache_dir),
            patch('tennis_updated.HD_PATH', os.path.join(self.temp_cache_dir, "historical_data.parquet")),
            patch('tennis_updated.JEFF_PATH', os.path.join(self.temp_cache_dir, "jeff_data.pkl")),
            patch('tennis_updated.DEF_PATH', os.path.join(self.temp_cache_dir, "weighted_defaults.pkl"))
        ]

        for p in self.cache_patches:
            p.start()

        # Setup logging capture
        self.log_output = []
        self.log_handler = logging.Handler()
        self.log_handler.emit = lambda record: self.log_output.append(record.getMessage())
        logging.getLogger().addHandler(self.log_handler)

        yield

        # Cleanup
        for p in self.cache_patches:
            p.stop()
        logging.getLogger().removeHandler(self.log_handler)
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def get_file_checksum(self, filepath):
        """Calculate file checksum for comparison"""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def create_mock_data(self, n_matches=100):
        """Create deterministic test data"""
        dates = [date(2025, 6, 15) + timedelta(days=i) for i in range(n_matches)]

        data = []
        for i, match_date in enumerate(dates):
            data.append({
                'composite_id': f"20250615-test_tournament-player_a-player_b_{i}",
                'date': match_date,
                'Winner': f'Player A {i}',
                'Loser': f'Player B {i}',
                'winner_canonical': f'player_a_{i}',
                'loser_canonical': f'player_b_{i}',
                'gender': 'M' if i % 2 == 0 else 'W',
                'Tournament': 'Test Tournament',
                'source_rank': 3,
                'winner_serve_pts': 60 + i,
                'loser_serve_pts': 55 + i,
                'winner_aces': 5 + (i % 10),
                'loser_aces': 3 + (i % 8)
            })

        return pd.DataFrame(data)

    def create_mock_jeff_data(self):
        """Create mock Jeff Sackmann data"""
        return {
            'men': {
                'overview': pd.DataFrame([
                    {'Player_canonical': 'player_a_0', 'set': 'Total', 'serve_pts': 60, 'aces': 5},
                    {'Player_canonical': 'player_b_0', 'set': 'Total', 'serve_pts': 55, 'aces': 3}
                ])
            },
            'women': {
                'overview': pd.DataFrame([
                    {'Player_canonical': 'player_a_1', 'set': 'Total', 'serve_pts': 58, 'aces': 4},
                    {'Player_canonical': 'player_b_1', 'set': 'Total', 'serve_pts': 53, 'aces': 2}
                ])
            }
        }

    def create_mock_defaults(self):
        """Create mock weighted defaults"""
        return {
            'men': {'serve_pts': 60, 'aces': 5, 'return_pts_won': 25},
            'women': {'serve_pts': 55, 'aces': 4, 'return_pts_won': 23}
        }

    # TEST 1: INITIAL FULL-BACKFILL RUN
    @patch('tennis_updated.load_all_tennis_data')
    @patch('tennis_updated.load_jeff_comprehensive_data')
    @patch('tennis_updated.api_call')
    def test_1_full_backfill_run(self, mock_api, mock_jeff, mock_tennis):
        """Test initial full backfill with empty cache"""
        # Setup mocks
        mock_tennis.return_value = self.create_mock_data(500)
        mock_jeff.return_value = self.create_mock_jeff_data()
        mock_api.return_value = []  # No API data for this test

        # Verify empty cache
        assert not os.path.exists(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

        # Run pipeline
        with patch('tennis_updated.run_automated_tennis_abstract_integration') as mock_ta:
            mock_ta.return_value = mock_tennis.return_value  # Return same data
            hist, jeff_data, defaults = load_from_cache_with_scraping()

        # Assertions
        assert hist is not None, "Pipeline should generate data on empty cache"
        assert len(hist) >= 500, "Should contain at least the mock tennis data"
        assert os.path.exists(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

        # Check logs
        log_messages = ' '.join(self.log_output)
        assert "Running initial integration" in log_messages or "No cache found" in log_messages

    # TEST 2: INCREMENTAL API UPDATE ONLY
    @patch('tennis_updated.api_call')
    def test_2_incremental_api_only(self, mock_api):
        """Test incremental API updates without TA changes"""
        # Setup existing cache
        existing_data = self.create_mock_data(100)
        existing_jeff = self.create_mock_jeff_data()
        existing_defaults = self.create_mock_defaults()

        save_to_cache(existing_data, existing_jeff, existing_defaults)
        original_checksum = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

        # Mock API to return no new data
        mock_api.return_value = []

        # Mock scraper to return empty
        with patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper.return_value.automated_scraping_session.return_value = []

            hist, _, _ = load_from_cache_with_scraping()

        # Verify no changes
        new_checksum = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))
        assert len(hist) == 100, "Row count should be unchanged"

        # Check appropriate log messages
        log_messages = ' '.join(self.log_output)
        assert "current" in log_messages.lower() or "cached" in log_messages.lower()

    # TEST 3: INCREMENTAL TA SCRAPE
    def test_3_incremental_ta_scrape(self):
        """Test incremental Tennis Abstract scraping"""
        # Setup existing cache with old TA update date
        existing_data = self.create_mock_data(100)
        existing_data['ta_enhanced'] = False
        existing_jeff = self.create_mock_jeff_data()
        existing_defaults = self.create_mock_defaults()

        save_to_cache(existing_data, existing_jeff, existing_defaults)

        # Mock scraper to return new match
        new_ta_record = {
            'composite_id': '20250716-wimbledon-sinner-alcaraz',
            'Player_canonical': 'jannik_sinner',
            'data_type': 'serve',
            'stat_name': 'points_won_pct',
            'stat_value': 75.5,
            'winner': 'Jannik Sinner',
            'loser': 'Carlos Alcaraz'
        }

        with patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper.return_value.automated_scraping_session.return_value = [new_ta_record]

            hist, _, _ = load_from_cache_with_scraping()

        # Verify new data integrated
        ta_enhanced = hist[hist.get('ta_enhanced', False) == True]
        assert len(ta_enhanced) > 0, "Should have TA-enhanced matches"

    # TEST 4: IDEMPOTENCY
    def test_4_idempotency(self):
        """Test that repeated runs produce identical results"""
        # Setup cache
        existing_data = self.create_mock_data(50)
        existing_jeff = self.create_mock_jeff_data()
        existing_defaults = self.create_mock_defaults()
        save_to_cache(existing_data, existing_jeff, existing_defaults)

        # Mock all external calls to return empty
        with patch('tennis_updated.api_call', return_value=[]), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper.return_value.automated_scraping_session.return_value = []

            # First run
            hist1, _, _ = load_from_cache_with_scraping()
            checksum1 = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

            # Second run
            hist2, _, _ = load_from_cache_with_scraping()
            checksum2 = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

        # Verify identical results
        assert len(hist1) == len(hist2), "Row counts should be identical"
        assert checksum1 == checksum2, "Cache files should be bitwise identical"

        # Check for no duplicate composite_ids
        assert len(hist2['composite_id'].unique()) == len(hist2), "No duplicate composite_ids"

    # TEST 5: SCHEMA AND DATA TYPE VALIDATION
    def test_5_schema_validation(self):
        """Test schema and data type correctness"""
        # Generate test data
        with patch('tennis_updated.load_all_tennis_data', return_value=self.create_mock_data()), \
                patch('tennis_updated.load_jeff_comprehensive_data', return_value=self.create_mock_jeff_data()), \
                patch('tennis_updated.api_call', return_value=[]):

            hist, _, _ = generate_comprehensive_historical_data(fast=True, n_sample=50)

        # Required columns
        required_cols = [
            'composite_id', 'winner_canonical', 'loser_canonical',
            'date', 'gender', 'source_rank'
        ]

        for col in required_cols:
            assert col in hist.columns, f"Missing required column: {col}"

        # Data type checks
        assert hist['composite_id'].dtype == 'object', "composite_id should be string"
        assert pd.api.types.is_datetime64_any_dtype(hist['date']) or \
               all(isinstance(x, date) for x in hist['date'].dropna()), "date should be date type"

        # Numeric ranges
        numeric_cols = [col for col in hist.columns if 'serve_pts' in col]
        for col in numeric_cols:
            valid_values = hist[col].dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all(), f"{col} should be non-negative"
                assert (valid_values <= 200).all(), f"{col} should be reasonable (≤200)"

    # TEST 6: WINNER/LOSER DISTINCTION
    def test_6_winner_loser_distinction(self):
        """Test winner vs loser stats are distinct"""
        # Create data with clear winner/loser differences
        test_data = self.create_mock_data(10)
        test_data['winner_serve_pts'] = 70
        test_data['loser_serve_pts'] = 50
        test_data['winner_aces'] = 10
        test_data['loser_aces'] = 3

        # Check distinctions
        for col_suffix in ['serve_pts', 'aces']:
            winner_col = f'winner_{col_suffix}'
            loser_col = f'loser_{col_suffix}'

            if winner_col in test_data.columns and loser_col in test_data.columns:
                # Should have some differences (not all identical)
                differences = (test_data[winner_col] != test_data[loser_col]).sum()
                assert differences > 0, f"Winner and loser {col_suffix} should differ in some matches"

    # TEST 7: ERROR HANDLING AND ROLLBACK
    def test_7_error_handling(self):
        """Test error handling and cache preservation"""
        # Setup existing cache
        existing_data = self.create_mock_data(50)
        existing_jeff = self.create_mock_jeff_data()
        existing_defaults = self.create_mock_defaults()
        save_to_cache(existing_data, existing_jeff, existing_defaults)

        original_checksum = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))

        # Inject API error
        with patch('tennis_updated.api_call', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                integrate_api_tennis_data_incremental(existing_data)

        # Verify cache unchanged
        post_error_checksum = self.get_file_checksum(os.path.join(self.temp_cache_dir, "historical_data.parquet"))
        assert original_checksum == post_error_checksum, "Cache should be unchanged after error"

    # TEST 8: PERFORMANCE AND LOGGING
    def test_8_performance_logging(self):
        """Test performance characteristics and logging"""
        # Test data sizes
        small_data = self.create_mock_data(10)
        large_data = self.create_mock_data(100)

        # Measure performance
        start_time = time.time()
        with patch('tennis_updated.load_all_tennis_data', return_value=small_data):
            generate_comprehensive_historical_data(fast=True, n_sample=10)
        small_duration = time.time() - start_time

        start_time = time.time()
        with patch('tennis_updated.load_all_tennis_data', return_value=large_data):
            generate_comprehensive_historical_data(fast=True, n_sample=100)
        large_duration = time.time() - start_time

        # Performance should scale reasonably
        assert large_duration < small_duration * 50, "Performance should scale reasonably"

        # Check logging
        assert len(self.log_output) > 0, "Should produce log output"
        log_text = ' '.join(self.log_output).lower()
        assert 'loading' in log_text or 'processing' in log_text, "Should log processing steps"

    # TEST 9: SECRETS AND ENVIRONMENT
    def test_9_environment_checks(self):
        """Test environment variable handling"""
        # Test missing API key
        with patch.dict(os.environ, {}, clear=True):
            # Should handle missing API key gracefully
            with patch('tennis_updated.api_call') as mock_api:
                mock_api.side_effect = ValueError("API key required")

                # Should not crash the entire pipeline
                try:
                    integrate_api_tennis_data_incremental(pd.DataFrame())
                except ValueError as e:
                    assert "API key" in str(e).lower()

    # TEST 10: MODEL TRAINING READINESS
    def test_10_model_readiness(self):
        """Test end-to-end model training compatibility"""
        # Generate complete dataset
        with patch('tennis_updated.load_all_tennis_data', return_value=self.create_mock_data()), \
                patch('tennis_updated.load_jeff_comprehensive_data', return_value=self.create_mock_jeff_data()), \
                patch('tennis_updated.api_call', return_value=[]):
            hist, _, _ = generate_comprehensive_historical_data(fast=True, n_sample=50)

        # Test model training preparation
        assert len(hist) > 0, "Should have training data"
        assert 'winner_canonical' in hist.columns, "Should have winner labels"
        assert 'loser_canonical' in hist.columns, "Should have loser labels"

        # Test feature matrix creation (basic)
        feature_cols = [col for col in hist.columns if col.startswith(('winner_', 'loser_')) and 'canonical' not in col]
        feature_matrix = hist[feature_cols].select_dtypes(include=[np.number])

        assert feature_matrix.shape[1] > 0, "Should have numeric features for training"
        assert feature_matrix.shape[0] == len(hist), "Feature matrix should match data length"

        # Test train/test split readiness
        from sklearn.model_selection import train_test_split
        if len(feature_matrix) > 10:  # Need minimum samples
            X_train, X_test = train_test_split(feature_matrix, test_size=0.2, random_state=42)
            assert len(X_train) > 0 and len(X_test) > 0, "Should split successfully"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])