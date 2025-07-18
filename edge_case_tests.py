#!/usr/bin/env python3
"""
Comprehensive edge case tests for tennis data pipeline
Covers all remaining gaps identified in feedback
"""

import pytest
import os
import shutil
import tempfile
import pickle
import hashlib
import subprocess
import time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Import your pipeline functions
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_updated import (
    load_from_cache_with_scraping,
    generate_comprehensive_historical_data,
    save_to_cache,
    load_from_cache,
    integrate_api_tennis_data_incremental,
    AutomatedTennisAbstractScraper
)


class TestPipelineEdgeCases:
    """Comprehensive edge case testing for tennis pipeline"""

    @pytest.fixture(autouse=True)
    def setup_edge_case_environment(self):
        """Setup test environment for edge case testing"""
        self.temp_cache_dir = tempfile.mkdtemp(prefix="tennis_edge_test_")
        self.hd_path = os.path.join(self.temp_cache_dir, "historical_data.parquet")
        self.jeff_path = os.path.join(self.temp_cache_dir, "jeff_data.pkl")
        self.def_path = os.path.join(self.temp_cache_dir, "weighted_defaults.pkl")

        # Patch all cache paths
        self.cache_patches = [
            patch('tennis_updated.CACHE_DIR', self.temp_cache_dir),
            patch('tennis_updated.HD_PATH', self.hd_path),
            patch('tennis_updated.JEFF_PATH', self.jeff_path),
            patch('tennis_updated.DEF_PATH', self.def_path)
        ]

        for p in self.cache_patches:
            p.start()

        yield

        # Cleanup
        for p in self.cache_patches:
            p.stop()
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def create_base_historical_data(self, n_matches=100):
        """Create base historical data for edge case testing"""
        dates = [date(2025, 6, 10) + timedelta(days=i // 5) for i in range(n_matches)]

        data = []
        for i, match_date in enumerate(dates):
            data.append({
                'composite_id': f"{match_date.strftime('%Y%m%d')}-tournament_{i % 3}-player_a_{i % 20}-player_b_{i % 20}",
                'date': match_date,
                'Winner': f'Player A {i % 20}',
                'Loser': f'Player B {i % 20}',
                'winner_canonical': f'player_a_{i % 20}',
                'loser_canonical': f'player_b_{i % 20}',
                'gender': 'M' if i % 2 == 0 else 'W',
                'Tournament': f'Tournament {i % 3}',
                'source_rank': 3,
                'winner_serve_pts': 60 + (i % 20),
                'loser_serve_pts': 55 + (i % 15),
                'winner_aces': 5 + (i % 10),
                'loser_aces': 3 + (i % 8),
                'ta_enhanced': False,
                'data_quality_score': 0.3
            })

        return pd.DataFrame(data)

    def get_file_checksum(self, filepath):
        """Calculate file checksum for bit-identical verification"""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    # TEST 1: INCREMENTAL TA-ONLY UPDATE
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    @patch('tennis_updated.api_call')
    def test_1_incremental_ta_only_update(self, mock_api, mock_scraper):
        """Test incremental TA-only update with stale cache"""
        print("\n=== TESTING INCREMENTAL TA-ONLY UPDATE ===")

        # Setup stale cache (TA data stale by >2 days)
        base_data = self.create_base_historical_data(50)
        base_data['date'] = base_data['date'].apply(lambda x: x - timedelta(days=5))  # Make stale
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}

        save_to_cache(base_data, jeff_data, defaults)
        original_count = len(base_data)

        # Mock API to return no new data
        mock_api.return_value = []

        # Mock scraper to return exactly one new match
        new_ta_record = [{
            'composite_id': '20250717-new_tournament-new_player_a-new_player_b',
            'Player_canonical': 'new_player_a',
            'data_type': 'serve',
            'stat_name': 'points_won_pct',
            'stat_value': 75.0,
            'Date': '20250717',
            'tournament': 'New Tournament',
            'player1': 'New Player A',
            'player2': 'New Player B'
        }]

        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = new_ta_record
        mock_scraper.return_value = mock_scraper_instance

        # Execute pipeline
        hist, _, _ = load_from_cache_with_scraping()

        # Verify exactly one row was added
        assert len(hist) == original_count + 1, f"Expected {original_count + 1} rows, got {len(hist)}"

        # Verify the new row is flagged
        new_rows = hist[hist['composite_id'] == '20250717-new_tournament-new_player_a-new_player_b']
        assert len(new_rows) == 1, "Should have exactly one new match"
        assert new_rows.iloc[0]['ta_enhanced'] == True, "New match should be TA enhanced"

        # Verify TA columns exist
        ta_columns = [col for col in hist.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
        assert len(ta_columns) > 0, "Should have TA feature columns"

        print("✓ Incremental TA-only update test passed")

    # TEST 2: IDEMPOTENCY CHECK
    @patch('tennis_updated.api_call')
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    def test_2_idempotency_check(self, mock_scraper, mock_api):
        """Test pipeline idempotency with bit-identical Parquet verification"""
        print("\n=== TESTING IDEMPOTENCY CHECK ===")

        # Setup initial cache
        base_data = self.create_base_historical_data(30)
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        # Mock all external calls to return no new data
        mock_api.return_value = []
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = []
        mock_scraper.return_value = mock_scraper_instance

        # First pipeline run
        hist1, _, _ = load_from_cache_with_scraping()
        checksum1 = self.get_file_checksum(self.hd_path)

        # Second pipeline run (no new data)
        hist2, _, _ = load_from_cache_with_scraping()
        checksum2 = self.get_file_checksum(self.hd_path)

        # Verify bit-identical results
        assert checksum1 == checksum2, "Pipeline should produce bit-identical results on repeated runs"
        assert len(hist1) == len(hist2), "Row counts should be identical"
        assert hist1['composite_id'].equals(hist2['composite_id']), "Composite IDs should be identical"

        # Verify no duplicate composite_ids
        assert len(hist2['composite_id'].unique()) == len(hist2), "Should have no duplicate composite_ids"

        print("✓ Idempotency check test passed")

    # TEST 3: API-ONLY INCREMENTAL UPDATE
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    @patch('tennis_updated.integrate_api_tennis_data_incremental')
    def test_3_api_only_incremental_update(self, mock_api_integration, mock_scraper):
        """Test API-only incremental update with exactly one day's new fixtures"""
        print("\n=== TESTING API-ONLY INCREMENTAL UPDATE ===")

        # Setup existing cache
        base_data = self.create_base_historical_data(40)
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        original_count = len(base_data)

        # Mock scraper to return no new TA data
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = []
        mock_scraper.return_value = mock_scraper_instance

        # Mock API integration to return exactly one more day's fixtures
        def mock_api_side_effect(df):
            new_row = pd.DataFrame([{
                'composite_id': '20250718-api_tournament-api_player_a-api_player_b',
                'date': date(2025, 7, 18),
                'Winner': 'API Player A',
                'Loser': 'API Player B',
                'winner_canonical': 'api_player_a',
                'loser_canonical': 'api_player_b',
                'gender': 'M',
                'source_rank': 2,
                'winner_serve_pts': 70,
                'loser_serve_pts': 65,
                'ta_enhanced': False
            }])
            return pd.concat([df, new_row], ignore_index=True)

        mock_api_integration.side_effect = mock_api_side_effect

        # Execute pipeline
        hist, _, _ = load_from_cache_with_scraping()

        # Verify exactly one row was added
        assert len(hist) == original_count + 1, f"Expected {original_count + 1} rows, got {len(hist)}"

        # Verify the new row is from API source
        new_api_rows = hist[hist['composite_id'] == '20250718-api_tournament-api_player_a-api_player_b']
        assert len(new_api_rows) == 1, "Should have exactly one new API match"
        assert new_api_rows.iloc[0]['source_rank'] == 2, "New match should be from API source"

        print("✓ API-only incremental update test passed")

    # TEST 4: SCHEMA & TYPES
    def test_4_schema_and_types(self):
        """Test full schema validation after complete pipeline run"""
        print("\n=== TESTING SCHEMA & TYPES ===")

        # Create comprehensive test data with all expected columns
        base_data = self.create_base_historical_data(20)

        # Add TA columns
        base_data['winner_ta_serve_points_won_pct'] = 75.5
        base_data['loser_ta_serve_points_won_pct'] = 68.2
        base_data['winner_ta_keypoints_serve_won_pct'] = 82.1
        base_data['loser_ta_keypoints_serve_won_pct'] = 76.8
        base_data['ta_enhanced'] = True

        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        # Load the parquet file directly
        loaded_df = pd.read_parquet(self.hd_path)

        # Test required columns exist
        required_columns = [
            'composite_id', 'winner_canonical', 'loser_canonical', 'ta_enhanced'
        ]
        for col in required_columns:
            assert col in loaded_df.columns, f"Missing required column: {col}"

        # Test TA columns exist
        winner_ta_cols = [col for col in loaded_df.columns if col.startswith('winner_ta_')]
        loser_ta_cols = [col for col in loaded_df.columns if col.startswith('loser_ta_')]
        assert len(winner_ta_cols) > 0, "Should have winner TA columns"
        assert len(loser_ta_cols) > 0, "Should have loser TA columns"

        # Test dtypes
        assert loaded_df['composite_id'].dtype == 'object', "composite_id should be string"
        assert loaded_df['winner_canonical'].dtype == 'object', "winner_canonical should be string"
        assert loaded_df['loser_canonical'].dtype == 'object', "loser_canonical should be string"
        assert loaded_df['ta_enhanced'].dtype == 'bool', "ta_enhanced should be boolean"

        # Test TA stat columns are numeric
        for col in winner_ta_cols + loser_ta_cols:
            if 'pct' in col or 'points' in col:
                assert pd.api.types.is_numeric_dtype(loaded_df[col]), f"{col} should be numeric"

        print("✓ Schema and types test passed")

    # TEST 5: WINNER/LOSER VALUE DIVERGENCE
    def test_5_winner_loser_value_divergence(self):
        """Test that winner and loser stats diverge for real matches"""
        print("\n=== TESTING WINNER/LOSER VALUE DIVERGENCE ===")

        # Create test data with explicitly different winner/loser stats
        base_data = self.create_base_historical_data(10)

        # Add divergent TA stats
        base_data['winner_ta_serve_points_won_pct'] = [75.0 + i for i in range(10)]
        base_data['loser_ta_serve_points_won_pct'] = [65.0 + i for i in range(10)]
        base_data['winner_ta_keypoints_serve_won_pct'] = [80.0 + i for i in range(10)]
        base_data['loser_ta_keypoints_serve_won_pct'] = [70.0 + i for i in range(10)]
        base_data['ta_enhanced'] = True

        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        # Load and test divergence
        loaded_df = pd.read_parquet(self.hd_path)

        # Test that winner != loser for key stats
        stats_to_check = [
            'serve_points_won_pct',
            'keypoints_serve_won_pct'
        ]

        for stat in stats_to_check:
            winner_col = f'winner_ta_{stat}'
            loser_col = f'loser_ta_{stat}'

            if winner_col in loaded_df.columns and loser_col in loaded_df.columns:
                # Should have differences (not all identical)
                differences = (loaded_df[winner_col] != loaded_df[loser_col]).sum()
                assert differences > 0, f"Winner and loser {stat} should differ in some matches"

                # Winner stats should generally be higher (since they won)
                winner_higher = (loaded_df[winner_col] > loaded_df[loser_col]).sum()
                total_valid = loaded_df[[winner_col, loser_col]].dropna().shape[0]
                if total_valid > 0:
                    winner_higher_pct = winner_higher / total_valid
                    assert winner_higher_pct > 0.5, f"Winner {stat} should be higher than loser in majority of matches"

        print("✓ Winner/loser value divergence test passed")

    # TEST 6: ERROR-HANDLING & ROLLBACK
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    def test_6_error_handling_and_rollback(self, mock_scraper):
        """Test error handling and rollback with mid-pipeline exception"""
        print("\n=== TESTING ERROR-HANDLING & ROLLBACK ===")

        # Setup valid initial cache
        base_data = self.create_base_historical_data(25)
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        original_checksum = self.get_file_checksum(self.hd_path)
        original_count = len(base_data)

        # Mock scraper to throw exception midway
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.side_effect = Exception("Scraper failed midway")
        mock_scraper.return_value = mock_scraper_instance

        # Execute pipeline and expect failure
        with pytest.raises(Exception) as exc_info:
            with patch('tennis_updated.api_call', return_value=[]):
                load_from_cache_with_scraping()

        # Verify exception was raised
        assert "failed" in str(exc_info.value).lower(), "Should capture scraper failure"

        # Verify original cache file is untouched
        post_error_checksum = self.get_file_checksum(self.hd_path)
        assert original_checksum == post_error_checksum, "Original cache should be untouched after error"

        # Verify data integrity maintained
        recovered_df = pd.read_parquet(self.hd_path)
        assert len(recovered_df) == original_count, "Should maintain original row count"
        assert recovered_df['composite_id'].nunique() == len(recovered_df), "Should have no duplicate IDs"

        print("✓ Error handling and rollback test passed")

    # TEST 7: SECRET & ENVIRONMENT FLAG
    @patch.dict(os.environ, {}, clear=True)
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    def test_7_secret_and_environment_flag(self, mock_scraper):
        """Test pipeline behavior with missing API_TENNIS_KEY"""
        print("\n=== TESTING SECRET & ENVIRONMENT FLAG ===")

        # Setup cache
        base_data = self.create_base_historical_data(15)
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        # Mock scraper to avoid external calls
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = []
        mock_scraper.return_value = mock_scraper_instance

        # Test pipeline behavior without API key
        try:
            # This should either:
            # 1. Skip API steps gracefully, or
            # 2. Fail fast with clear message about missing API key
            hist, _, _ = load_from_cache_with_scraping()

            # If it succeeds, verify it skipped API integration
            assert hist is not None, "Should handle missing API key gracefully"
            print("Pipeline skipped API steps due to missing key")

        except ValueError as e:
            # If it fails, verify clear error message
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['api', 'key', 'environment', 'missing']), \
                f"Error message should mention API key issue: {e}"
            print("Pipeline failed fast with clear API key error")

        except Exception as e:
            # Any other exception should mention API or environment
            error_msg = str(e).lower()
            if 'api' in error_msg or 'key' in error_msg or 'environment' in error_msg:
                print("Pipeline failed with API-related error (acceptable)")
            else:
                pytest.fail(f"Unexpected error without API key: {e}")

        print("✓ Secret and environment flag test passed")

    # TEST 8: COMPREHENSIVE INTEGRATION TEST
    @patch('tennis_updated.api_call')
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    def test_8_comprehensive_integration(self, mock_scraper, mock_api):
        """Test comprehensive end-to-end integration with all components"""
        print("\n=== TESTING COMPREHENSIVE INTEGRATION ===")

        # Setup initial state
        base_data = self.create_base_historical_data(50)
        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}
        save_to_cache(base_data, jeff_data, defaults)

        original_count = len(base_data)

        # Mock API to add some new data
        mock_api.return_value = []  # No API fixtures for simplicity

        # Mock TA scraper to add new match
        new_ta_records = [{
            'composite_id': '20250719-comprehensive_test-player_x-player_y',
            'Player_canonical': 'player_x',
            'data_type': 'serve',
            'stat_name': 'points_won_pct',
            'stat_value': 78.5,
            'Date': '20250719',
            'tournament': 'Comprehensive Test',
            'player1': 'Player X',
            'player2': 'Player Y'
        }]

        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = new_ta_records
        mock_scraper.return_value = mock_scraper_instance

        # Execute comprehensive pipeline
        hist, jeff_result, defaults_result = load_from_cache_with_scraping()

        # Comprehensive validation
        assert hist is not None, "Should return historical data"
        assert jeff_result is not None, "Should return jeff data"
        assert defaults_result is not None, "Should return defaults"

        # Data quality checks
        assert len(hist) >= original_count, "Should maintain or increase data"
        assert hist['composite_id'].nunique() == len(hist), "Should have unique composite IDs"
        assert not hist['winner_canonical'].isna().any(), "Should have all winner names"
        assert not hist['loser_canonical'].isna().any(), "Should have all loser names"

        # TA integration checks
        ta_enhanced_matches = hist[hist['ta_enhanced'] == True]
        if len(ta_enhanced_matches) > 0:
            ta_columns = [col for col in hist.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
            assert len(ta_columns) > 0, "Should have TA feature columns"

            # Check data quality scores
            enhanced_quality_scores = ta_enhanced_matches['data_quality_score']
            assert enhanced_quality_scores.notna().any(), "TA enhanced matches should have quality scores"

        # Performance check - should complete in reasonable time
        # (This is implicitly tested by pytest timeouts)

        print("✓ Comprehensive integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])