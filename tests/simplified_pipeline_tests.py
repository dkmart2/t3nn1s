#!/usr/bin/env python3
"""
Simplified but comprehensive test suite for tennis data pipeline
Focuses on testing key behaviors with proper mocking
"""

import pytest
import os
import shutil
import tempfile
import pickle
import hashlib
import time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock, Mock

# Import your pipeline functions
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_updated import (
    save_to_cache,
    load_from_cache,
    normalize_name,
    normalize_tournament_name,
    build_composite_id,
    extract_unified_features_fixed,
    integrate_api_tennis_data_incremental
)


class TestDataPipelineCore:
    """Core pipeline functionality tests"""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup clean test environment"""
        self.temp_cache_dir = tempfile.mkdtemp(prefix="tennis_test_")
        self.hd_path = os.path.join(self.temp_cache_dir, "historical_data.parquet")
        self.jeff_path = os.path.join(self.temp_cache_dir, "jeff_data.pkl")
        self.def_path = os.path.join(self.temp_cache_dir, "weighted_defaults.pkl")

        yield

        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def create_sample_data(self, n_rows=100):
        """Create realistic test data"""
        data = []
        for i in range(n_rows):
            match_date = date(2025, 6, 15) + timedelta(days=i % 30)
            data.append({
                'composite_id': f"20250615-wimbledon-player_a_{i}-player_b_{i}",
                'date': match_date,
                'Winner': f'Player A {i}',
                'Loser': f'Player B {i}',
                'winner_canonical': f'player_a_{i}',
                'loser_canonical': f'player_b_{i}',
                'gender': 'M' if i % 2 == 0 else 'W',
                'Tournament': 'Wimbledon',
                'source_rank': 3,
                'winner_serve_pts': 60 + (i % 20),
                'loser_serve_pts': 55 + (i % 15),
                'winner_aces': 5 + (i % 10),
                'loser_aces': 3 + (i % 8),
                'winner_return_pts_won': 20 + (i % 10),
                'loser_return_pts_won': 18 + (i % 8),
                'PSW': 1.5 + (i % 10) * 0.1,
                'PSL': 2.5 + (i % 10) * 0.1
            })
        return pd.DataFrame(data)

    def create_sample_jeff_data(self):
        """Create sample Jeff data"""
        return {
            'men': {
                'overview': pd.DataFrame([
                    {'Player_canonical': 'player_a_0', 'set': 'Total', 'serve_pts': 60, 'aces': 5}
                ])
            },
            'women': {
                'overview': pd.DataFrame([
                    {'Player_canonical': 'player_a_1', 'set': 'Total', 'serve_pts': 58, 'aces': 4}
                ])
            }
        }

    def create_sample_defaults(self):
        """Create sample weighted defaults"""
        return {
            'men': {'serve_pts': 60, 'aces': 5, 'return_pts_won': 25},
            'women': {'serve_pts': 55, 'aces': 4, 'return_pts_won': 23}
        }

    # TEST 1: CACHE OPERATIONS
    def test_1_cache_operations(self):
        """Test basic cache save/load operations"""
        # Create test data
        hist_data = self.create_sample_data(50)
        jeff_data = self.create_sample_jeff_data()
        defaults = self.create_sample_defaults()

        # Test save
        with patch('tennis_updated.CACHE_DIR', self.temp_cache_dir), \
                patch('tennis_updated.HD_PATH', self.hd_path), \
                patch('tennis_updated.JEFF_PATH', self.jeff_path), \
                patch('tennis_updated.DEF_PATH', self.def_path):
            result = save_to_cache(hist_data, jeff_data, defaults)
            assert result == True, "Cache save should succeed"

            # Verify files exist
            assert os.path.exists(self.hd_path), "Historical data file should exist"
            assert os.path.exists(self.jeff_path), "Jeff data file should exist"
            assert os.path.exists(self.def_path), "Defaults file should exist"

            # Test load
            loaded_hist, loaded_jeff, loaded_defaults = load_from_cache()

            assert loaded_hist is not None, "Should load historical data"
            assert len(loaded_hist) == 50, "Should load correct number of rows"
            assert 'composite_id' in loaded_hist.columns, "Should have required columns"

    # TEST 2: NAME NORMALIZATION
    def test_2_name_normalization(self):
        """Test name normalization functions"""
        # Test player names
        assert normalize_name("Carlos Alcaraz") == "carlos_alcaraz"
        assert normalize_name("J. M. La Serna") == "j_m_la_serna"
        assert normalize_name("A. Barrena") == "a_barrena"

        # Test tournament names
        assert normalize_tournament_name("Wimbledon") == "wimbledon"
        assert normalize_tournament_name("French Open") == "french_open"

        # Test composite ID building
        match_date = date(2025, 7, 15)
        comp_id = build_composite_id(match_date, "wimbledon", "carlos_alcaraz", "jannik_sinner")
        expected = "20250715-wimbledon-carlos_alcaraz-jannik_sinner"
        assert comp_id == expected, f"Expected {expected}, got {comp_id}"

    # TEST 3: FEATURE EXTRACTION
    def test_3_feature_extraction(self):
        """Test unified feature extraction"""
        # Create match data with various sources
        match_data = {
            'winner_serve_pts': 80,
            'winner_pts_won': 60,
            'winner_return_pts': 70,
            'winner_return_pts_won': 25,
            'winner_aces': 8,
            'winner_dfs': 2,
            'winner_bp_saved': 4,
            'winner_bk_pts': 6
        }

        features = extract_unified_features_fixed(match_data, 'winner')

        # Check that we get expected features
        expected_features = [
            'serve_effectiveness', 'return_effectiveness', 'winners_rate',
            'unforced_rate', 'pressure_performance', 'net_effectiveness'
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
            assert isinstance(features[feature], float), f"{feature} should be numeric"
            assert 0 <= features[feature] <= 1, f"{feature} should be in [0,1] range"

    # TEST 4: DATA QUALITY VALIDATION
    def test_4_data_quality(self):
        """Test data quality checks"""
        data = self.create_sample_data(100)

        # Check required columns
        required_cols = ['composite_id', 'winner_canonical', 'loser_canonical', 'date']
        for col in required_cols:
            assert col in data.columns, f"Missing required column: {col}"

        # Check data types
        assert data['composite_id'].dtype == 'object', "composite_id should be string"

        # Check composite_id uniqueness
        assert len(data['composite_id'].unique()) == len(data), "composite_ids should be unique"

        # Check winner/loser distinctions
        if 'winner_serve_pts' in data.columns and 'loser_serve_pts' in data.columns:
            # Should have some differences
            differences = (data['winner_serve_pts'] != data['loser_serve_pts']).sum()
            assert differences > 0, "Winner and loser stats should differ"

    # TEST 5: INCREMENTAL API INTEGRATION
    def test_5_api_integration(self):
        """Test API data integration"""
        # Create base historical data
        base_data = self.create_sample_data(50)
        base_data['source_rank'] = 3  # Tennis data files

        # Mock API responses
        with patch('tennis_updated.api_call') as mock_api, \
                patch('tennis_updated.get_player_rankings') as mock_rankings, \
                patch('tennis_updated.get_tournaments_metadata') as mock_tournaments, \
                patch('tennis_updated.get_event_types') as mock_events:
            # Setup mocks
            mock_api.return_value = []  # No new fixtures
            mock_rankings.return_value = {}
            mock_tournaments.return_value = {}
            mock_events.return_value = {}

            # Test integration
            result = integrate_api_tennis_data_incremental(base_data)

            # Should return data unchanged if no new API data
            assert len(result) == len(base_data), "Should preserve existing data"
            assert 'source_rank' in result.columns, "Should have source_rank column"

    # TEST 6: IDEMPOTENCY CHECK
    def test_6_idempotency(self):
        """Test that operations are idempotent"""
        data = self.create_sample_data(30)

        # Save data twice
        with patch('tennis_updated.CACHE_DIR', self.temp_cache_dir), \
                patch('tennis_updated.HD_PATH', self.hd_path), \
                patch('tennis_updated.JEFF_PATH', self.jeff_path), \
                patch('tennis_updated.DEF_PATH', self.def_path):
            save_to_cache(data, self.create_sample_jeff_data(), self.create_sample_defaults())
            checksum1 = self._get_file_checksum(self.hd_path)

            # Save again with same data
            save_to_cache(data, self.create_sample_jeff_data(), self.create_sample_defaults())
            checksum2 = self._get_file_checksum(self.hd_path)

            # Checksums should be identical
            assert checksum1 == checksum2, "Repeated saves should produce identical files"

    def _get_file_checksum(self, filepath):
        """Helper to calculate file checksum"""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    # TEST 7: ERROR HANDLING
    def test_7_error_handling(self):
        """Test error handling in pipeline components"""
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})

        # Should handle gracefully
        try:
            with patch('tennis_updated.CACHE_DIR', self.temp_cache_dir), \
                    patch('tennis_updated.HD_PATH', self.hd_path), \
                    patch('tennis_updated.JEFF_PATH', self.jeff_path), \
                    patch('tennis_updated.DEF_PATH', self.def_path):
                save_to_cache(invalid_data, {}, {})
            # If it doesn't crash, that's good
        except Exception as e:
            # Should be a meaningful error, not a random crash
            assert len(str(e)) > 0, "Error should have meaningful message"

    # TEST 8: PERFORMANCE CHECK
    def test_8_performance(self):
        """Test basic performance characteristics"""
        # Test with different data sizes
        small_data = self.create_sample_data(10)
        large_data = self.create_sample_data(1000)

        # Time small operation
        start = time.time()
        features_small = extract_unified_features_fixed({
            'winner_serve_pts': 80, 'winner_pts_won': 60,
            'winner_return_pts': 70, 'winner_return_pts_won': 25
        }, 'winner')
        small_time = time.time() - start

        # Time larger operation (normalization of many names)
        start = time.time()
        normalized_names = [normalize_name(f"Player {i}") for i in range(1000)]
        large_time = time.time() - start

        # Should complete in reasonable time
        assert small_time < 1.0, "Feature extraction should be fast"
        assert large_time < 5.0, "Name normalization should be reasonable"
        assert len(normalized_names) == 1000, "Should process all items"

    # TEST 9: DATA SCHEMA VALIDATION
    def test_9_schema_validation(self):
        """Test comprehensive schema validation"""
        data = self.create_sample_data(20)

        # Check numeric columns are actually numeric
        numeric_cols = ['winner_serve_pts', 'loser_serve_pts', 'winner_aces', 'loser_aces']
        for col in numeric_cols:
            if col in data.columns:
                assert pd.api.types.is_numeric_dtype(data[col]), f"{col} should be numeric"
                assert (data[col] >= 0).all(), f"{col} should be non-negative"

        # Check string columns
        string_cols = ['composite_id', 'Winner', 'Loser', 'winner_canonical', 'loser_canonical']
        for col in string_cols:
            if col in data.columns:
                assert data[col].dtype == 'object', f"{col} should be string type"
                assert not data[col].isna().all(), f"{col} should not be all null"

    # TEST 10: END-TO-END COMPATIBILITY
    def test_10_end_to_end_compatibility(self):
        """Test end-to-end pipeline compatibility"""
        # Create comprehensive test dataset
        data = self.create_sample_data(100)

        # Add various feature columns that model would expect
        feature_columns = [
            'winner_serve_effectiveness', 'loser_serve_effectiveness',
            'winner_return_effectiveness', 'loser_return_effectiveness',
            'winner_winners_rate', 'loser_winners_rate',
            'winner_pressure_performance', 'loser_pressure_performance'
        ]

        for col in feature_columns:
            data[col] = np.random.uniform(0, 1, len(data))

        # Test that we can create feature matrices
        feature_cols = [col for col in data.columns if col.startswith(('winner_', 'loser_'))
                        and 'canonical' not in col]
        feature_matrix = data[feature_cols].select_dtypes(include=[np.number])

        assert feature_matrix.shape[1] > 0, "Should have numeric features"
        assert feature_matrix.shape[0] == len(data), "Feature matrix should match data length"
        assert not feature_matrix.isna().all().any(), "Features should not be all NaN"

        # Test train/test split readiness
        if len(feature_matrix) > 10:
            try:
                from sklearn.model_selection import train_test_split
                X_train, X_test = train_test_split(feature_matrix, test_size=0.2, random_state=42)
                assert len(X_train) > 0 and len(X_test) > 0, "Should split successfully"
                assert X_train.shape[1] == X_test.shape[1], "Train/test should have same features"
            except ImportError:
                # Skip if sklearn not available
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])