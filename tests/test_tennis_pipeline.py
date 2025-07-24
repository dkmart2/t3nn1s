# ============================================================================
# COMPREHENSIVE UNIT TESTS - TENNIS DATA PIPELINE
# ============================================================================

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, List

# Import modules under test
import tennis_updated_enterprise as pipeline
import settings


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    temp_dir = tempfile.mkdtemp(prefix="tennis_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_match_data():
    """Create sample tennis match data"""
    return pd.DataFrame({
        'composite_id': ['20250101-tournament-playerA-playerB', '20250102-tournament-playerC-playerD'],
        'date': [date(2025, 1, 1), date(2025, 1, 2)],
        'Winner': ['Player A', 'Player C'],
        'Loser': ['Player B', 'Player D'],
        'winner_canonical': ['player_a', 'player_c'],
        'loser_canonical': ['player_b', 'player_d'],
        'gender': ['M', 'W'],
        'Tournament': ['Test Tournament', 'Test Tournament'],
        'Surface': ['Hard', 'Clay'],
        'source_rank': [3, 2]
    })


@pytest.fixture
def sample_jeff_data():
    """Create sample Jeff charting data"""
    return {
        'men': {
            'overview': pd.DataFrame({
                'Player_canonical': ['player_a', 'player_b'],
                'set': ['Total', 'Total'],
                'serve_pts': [80, 75],
                'aces': [6, 4],
                'dfs': [3, 5],
                'first_in': [50, 45],
                'first_won': [38, 35],
                'second_won': [20, 18],
                'bp_saved': [8, 6],
                'return_pts_won': [30, 25],
                'winners': [25, 20],
                'winners_fh': [15, 12],
                'winners_bh': [10, 8],
                'unforced': [22, 25],
                'unforced_fh': [12, 15],
                'unforced_bh': [10, 10]
            })
        },
        'women': {
            'overview': pd.DataFrame({
                'Player_canonical': ['player_c', 'player_d'],
                'set': ['Total', 'Total'],
                'serve_pts': [75, 70],
                'aces': [4, 3],
                'dfs': [2, 4],
                'first_in': [45, 42],
                'first_won': [32, 30],
                'second_won': [15, 14],
                'bp_saved': [6, 5],
                'return_pts_won': [28, 26],
                'winners': [20, 18],
                'winners_fh': [12, 11],
                'winners_bh': [8, 7],
                'unforced': [18, 20],
                'unforced_fh': [10, 11],
                'unforced_bh': [8, 9]
            })
        }
    }


@pytest.fixture
def mock_api_response():
    """Mock API response data"""
    return [
        {
            'fixture_key': 'test_fixture_1',
            'event_name': 'Test Tournament',
            'date_start': '2025-01-01T10:00:00Z',
            'participants': [
                {'name': 'Player A', 'participant_key': 'player_a_key'},
                {'name': 'Player B', 'participant_key': 'player_b_key'}
            ],
            'scores': [
                {'participant_1_score': 6, 'participant_2_score': 4},
                {'participant_1_score': 6, 'participant_2_score': 2}
            ]
        }
    ]


# ============================================================================
# UNIT TESTS - CORE FUNCTIONALITY
# ============================================================================

class TestPlayerMappingCache:
    """Test player name mapping and caching"""

    def test_canonical_name_generation(self, temp_cache_dir):
        """Test canonical name generation"""
        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            cache = pipeline.PlayerMappingCache()

            # Test basic normalization
            assert cache.get_canonical_name("Novak Djokovic") == "novak djokovic"
            assert cache.get_canonical_name("Rafael Nadal-Parera") == "rafael nadal parera"
            assert cache.get_canonical_name("Jo-Wilfried Tsonga") == "jo wilfried tsonga"

            # Test edge cases
            assert cache.get_canonical_name("") == ""
            assert cache.get_canonical_name(None) == ""
            assert cache.get_canonical_name("  Test Player  ") == "test player"

    def test_cache_persistence(self, temp_cache_dir):
        """Test cache persistence across instances"""
        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            # First instance
            cache1 = pipeline.PlayerMappingCache()
            cache1.get_canonical_name("Test Player One")
            cache1._save_cache()

            # Second instance should load cached data
            cache2 = pipeline.PlayerMappingCache()
            assert "Test Player One" in cache2.mapping
            assert cache2.get_canonical_name("Test Player One") == "test player one"


class TestSurfaceLookupTable:
    """Test surface lookup and inference"""

    def test_known_tournament_surface(self, temp_cache_dir):
        """Test surface lookup for known tournaments"""
        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            lookup = pipeline.SurfaceLookupTable()

            assert lookup.get_surface("French Open") == "Clay"
            assert lookup.get_surface("Wimbledon") == "Grass"
            assert lookup.get_surface("US Open") == "Hard"
            assert lookup.get_surface("Australian Open") == "Hard"

    def test_seasonal_inference(self, temp_cache_dir):
        """Test seasonal surface inference"""
        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            lookup = pipeline.SurfaceLookupTable()

            # Clay season (May-June)
            assert lookup.get_surface("Unknown Tournament", season=5) == "Clay"
            assert lookup.get_surface("Unknown Tournament", season=6) == "Clay"

            # Grass season (July)
            assert lookup.get_surface("Unknown Tournament", season=7) == "Grass"

            # Hard court default
            assert lookup.get_surface("Unknown Tournament", season=1) == "Hard"
            assert lookup.get_surface("Unknown Tournament", season=8) == "Hard"

    def test_surface_update(self, temp_cache_dir):
        """Test surface update functionality"""
        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            lookup = pipeline.SurfaceLookupTable()

            # Update surface for new tournament
            lookup.update_surface("New Tournament", "Clay", confidence=0.9)
            assert lookup.get_surface("New Tournament") == "Clay"


class TestDataProcessing:
    """Test data processing functions"""

    def test_dtype_optimization(self):
        """Test dtype optimization mapping"""
        dtypes = pipeline.get_optimized_dtypes()

        # Check key dtypes are properly mapped
        assert dtypes['WRank'] == 'float32'
        assert dtypes['W1'] == 'int8'
        assert dtypes['serve_pts'] == 'int16'
        assert dtypes['aces'] == 'int16'

    def test_fallback_defaults(self):
        """Test fallback default values"""
        men_defaults = pipeline.get_fallback_defaults('men')
        women_defaults = pipeline.get_fallback_defaults('women')

        # Check men's defaults
        assert men_defaults['serve_pts'] == 80.0
        assert men_defaults['aces'] == 6.0
        assert men_defaults['first_serve_pct'] == 0.62

        # Check women's defaults are different
        assert women_defaults['serve_pts'] == 75.0
        assert women_defaults['aces'] == 4.0
        assert women_defaults['first_serve_pct'] == 0.60

    def test_expected_jeff_features(self):
        """Test expected Jeff feature list generation"""
        features = pipeline.get_expected_jeff_features()

        # Check feature count and structure
        assert len(features) == 28  # 14 base features * 2 (winner/loser)

        # Check winner/loser prefixes
        winner_features = [f for f in features if f.startswith('winner_')]
        loser_features = [f for f in features if f.startswith('loser_')]

        assert len(winner_features) == 14
        assert len(loser_features) == 14

        # Check specific features exist
        assert 'winner_serve_pts' in features
        assert 'loser_aces' in features
        assert 'winner_first_serve_pct' in features


class TestVectorizedOperations:
    """Test vectorized processing functions"""

    def test_extract_features_vectorized(self):
        """Test vectorized feature extraction from overview row"""
        test_row = pd.Series({
            'serve_pts': 80,
            'aces': 6,
            'dfs': 3,
            'first_in': 50,
            'first_won': 38,
            'second_won': 20,
            'bp_saved': 8,
            'return_pts_won': 30,
            'winners': 25,
            'winners_fh': 15,
            'winners_bh': 10,
            'unforced': 22,
            'unforced_fh': 12,
            'unforced_bh': 10
        })

        features = pipeline.extract_features_vectorized(test_row)

        # Check calculated values
        assert features['serve_pts'] == 80.0
        assert features['aces'] == 6.0
        assert features['first_serve_pct'] == 50.0 / 80.0  # first_in / serve_pts
        assert features['winners_total'] == 25.0

    def test_inject_jeff_features_vectorized(self, sample_match_data, sample_jeff_data):
        """Test vectorized Jeff feature injection"""
        # Mock weighted defaults
        weighted_defaults = {
            'men': pipeline.get_fallback_defaults('men'),
            'women': pipeline.get_fallback_defaults('women')
        }

        result_df = pipeline.inject_jeff_features_vectorized(
            sample_match_data, sample_jeff_data, weighted_defaults
        )

        # Check that features were added
        expected_features = pipeline.get_expected_jeff_features()
        for feature in expected_features:
            assert feature in result_df.columns

        # Check that values were populated (should not all be NaN)
        assert not result_df['winner_serve_pts'].isna().all()
        assert not result_df['loser_serve_pts'].isna().all()


class TestDeduplication:
    """Test match deduplication logic"""

    def test_source_rank_priority(self):
        """Test deduplication respects source ranking"""
        test_data = pd.DataFrame({
            'composite_id': ['match1', 'match1', 'match2', 'match2'],
            'source_rank': [1, 3, 2, 1],  # Lower rank = higher priority
            'data_source': ['TA', 'Historical', 'API', 'TA']
        })

        deduplicated = pipeline.deduplicate_matches(test_data)

        # Should keep 2 unique matches
        assert len(deduplicated) == 2

        # Should keep higher priority sources
        match1_row = deduplicated[deduplicated['composite_id'] == 'match1'].iloc[0]
        match2_row = deduplicated[deduplicated['composite_id'] == 'match2'].iloc[0]

        assert match1_row['source_rank'] == 1  # TA has priority
        assert match2_row['source_rank'] == 1  # TA has priority

    def test_no_duplicates_case(self):
        """Test deduplication with no duplicates"""
        test_data = pd.DataFrame({
            'composite_id': ['match1', 'match2', 'match3'],
            'source_rank': [1, 2, 3]
        })

        deduplicated = pipeline.deduplicate_matches(test_data)

        # Should return unchanged
        assert len(deduplicated) == 3
        pd.testing.assert_frame_equal(deduplicated.reset_index(drop=True),
                                      test_data.reset_index(drop=True))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAsyncAPI:
    """Test asynchronous API processing"""

    @pytest.mark.asyncio
    async def test_fetch_fixtures_for_date_async(self, mock_api_response):
        """Test async fixture fetching"""
        target_date = date(2025, 1, 1)

        # Mock the session and responses
        mock_session = MagicMock()

        # Mock events response
        events_response = MagicMock()
        events_response.json.return_value = [
            {'event_type_key': 1, 'event_type_type': 'ATP Singles'}
        ]
        events_response.raise_for_status = MagicMock()

        # Mock fixtures response
        fixtures_response = MagicMock()
        fixtures_response.json.return_value = mock_api_response
        fixtures_response.raise_for_status = MagicMock()

        mock_session.bounded_get.side_effect = [events_response, fixtures_response]

        # Test the function
        with patch('tennis_updated_enterprise.settings.BASE_API_URL', 'https://test.api.com'):
            with patch('tennis_updated_enterprise.settings.API_KEY', 'test_key'):
                fixtures = await pipeline.fetch_fixtures_for_date_async(mock_session, target_date)

        # Verify results
        assert len(fixtures) == 1
        assert fixtures[0]['fixture_key'] == 'test_fixture_1'

    @pytest.mark.asyncio
    async def test_fetch_fixtures_async_batch(self, mock_api_response):
        """Test async batch fixture fetching"""
        date_range = [date(2025, 1, 1), date(2025, 1, 2)]

        with patch('tennis_updated_enterprise.fetch_fixtures_for_date_async') as mock_fetch:
            mock_fetch.return_value = mock_api_response

            fixtures = await pipeline.fetch_fixtures_async_batch(date_range, batch_size=2)

            # Should have called fetch for each date
            assert mock_fetch.call_count == 2
            assert len(fixtures) == 2  # 2 dates * 1 fixture each


class TestCaching:
    """Test caching mechanisms"""

    def test_excel_parquet_caching(self, temp_cache_dir):
        """Test Excel to Parquet caching"""
        # Create a temporary Excel file (mock)
        test_file = os.path.join(temp_cache_dir, "test.xlsx")

        # Mock DataFrame to return
        mock_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        with patch('tennis_updated_enterprise.CACHE_DIR', temp_cache_dir):
            with patch('pandas.read_excel', return_value=mock_df) as mock_read_excel:
                with patch('os.path.exists', return_value=True):
                    with patch('os.path.getmtime', return_value=1234567):
                        # First call should read Excel and cache
                        result1 = pipeline.load_excel_with_parquet_cache(test_file)
                        assert mock_read_excel.call_count == 1
                        pd.testing.assert_frame_equal(result1, mock_df)

                        # Second call should use cache (mock Parquet read)
                        with patch('pandas.read_parquet', return_value=mock_df) as mock_read_parquet:
                            result2 = pipeline.load_excel_with_parquet_cache(test_file)
                            assert mock_read_parquet.call_count == 1
                            pd.testing.assert_frame_equal(result2, mock_df)


# ============================================================================
# SCHEMA VERSIONING TESTS
# ============================================================================

class TestSchemaVersioning:
    """Test schema versioning and validation"""

    def test_weighted_defaults_metadata_creation(self, sample_jeff_data):
        """Test weighted defaults metadata creation"""
        defaults, metadata = pipeline.calculate_comprehensive_weighted_defaults_versioned(sample_jeff_data)

        # Check metadata structure
        assert isinstance(metadata, pipeline.WeightedDefaultsMetadata)
        assert metadata.schema_version == pipeline.WEIGHTED_DEFAULTS_SCHEMA_VERSION
        assert metadata.men_features > 0
        assert metadata.women_features > 0
        assert len(metadata.jeff_data_hash) == 32  # MD5 hash length

    def test_schema_version_validation(self, temp_cache_dir):
        """Test schema version validation"""
        # Create metadata with wrong version
        wrong_metadata = pipeline.WeightedDefaultsMetadata(
            schema_version="0.9.0",  # Wrong version
            creation_date="2025-01-01",
            jeff_data_hash="test_hash",
            column_count=100,
            men_features=50,
            women_features=50
        )

        test_defaults = {"men": {}, "women": {}}

        # Save with wrong version
        def_path = os.path.join(temp_cache_dir, "test_defaults.pkl")
        with open(def_path, 'wb') as f:
            pickle.dump((test_defaults, wrong_metadata), f)

        # Test loading with validation
        with patch('tennis_updated_enterprise.DEF_PATH', def_path):
            defaults, metadata = pipeline.load_weighted_defaults_with_validation()

            # Should return None due to version mismatch
            assert defaults is None
            assert metadata is None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()

        # Should handle empty DataFrame gracefully
        result = pipeline.deduplicate_matches(empty_df)
        assert len(result) == 0

        # Feature injection should handle empty DataFrame
        result = pipeline.inject_jeff_features_vectorized(empty_df, {}, {})
        assert len(result) == 0

    def test_missing_columns_handling(self):
        """Test handling of missing columns"""
        df_no_composite_id = pd.DataFrame({
            'other_column': [1, 2, 3]
        })

        # Should return unchanged if no composite_id column
        result = pipeline.deduplicate_matches(df_no_composite_id)
        pd.testing.assert_frame_equal(result, df_no_composite_id)

    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        # Test with None values in name normalization
        cache = pipeline.PlayerMappingCache()
        result = cache._normalize_name(None)
        assert result == ""

        # Test with numeric values
        result = cache._normalize_name(12345)
        assert result == "12345"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    def test_large_dataset_processing(self):
        """Test processing of large datasets"""
        # Create large dataset
        n_matches = 10000
        large_df = pd.DataFrame({
            'composite_id': [f'match_{i}' for i in range(n_matches)],
            'winner_canonical': [f'player_{i % 100}' for i in range(n_matches)],
            'loser_canonical': [f'player_{(i + 1) % 100}' for i in range(n_matches)],
            'gender': ['M' if i % 2 == 0 else 'W' for i in range(n_matches)]
        })

        # Test deduplication performance
        start_time = time.time()
        result = pipeline.deduplicate_matches(large_df)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for 10k records)
        assert end_time - start_time < 1.0
        assert len(result) == n_matches  # No duplicates in this case

    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        # This test would monitor memory usage during processing
        # Implementation would use memory profiling tools
        pass


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests if executed directly
    os.system("python -m pytest test_tennis_pipeline.py -v")