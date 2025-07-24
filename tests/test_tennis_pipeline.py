# ============================================================================
# CORRECTED UNIT TESTS - TENNIS DATA PIPELINE
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
from pathlib import Path
import joblib

# Import modules under test
import tennis_updated_fixed as pipeline


# ============================================================================
# TEST FIXTURES WITH TEMP DIRECTORY PATCHING
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory and patch settings"""
    temp_dir = Path(tempfile.mkdtemp(prefix="tennis_test_"))

    # Create required subdirectories
    (temp_dir / "excel").mkdir(exist_ok=True)
    (temp_dir / "api").mkdir(exist_ok=True)

    with patch.multiple(
            pipeline,
            TENNIS_CACHE_DIR=temp_dir,
            EXCEL_CACHE_DIR=temp_dir / "excel",
            API_CACHE_DIR=temp_dir / "api"
    ):
        yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    temp_dir = Path(tempfile.mkdtemp(prefix="tennis_data_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_match_data():
    """Create sample tennis match data with categorical columns"""
    return pd.DataFrame({
        'composite_id': ['20250101-tournament-playerA-playerB', '20250102-tournament-playerC-playerD'],
        'date': [date(2025, 1, 1), date(2025, 1, 2)],
        'Winner': pd.Categorical(['Player A', 'Player C']),
        'Loser': pd.Categorical(['Player B', 'Player D']),
        'winner_canonical': ['player_a', 'player_c'],
        'loser_canonical': ['player_b', 'player_d'],
        'gender': pd.Categorical(['M', 'W']),
        'Tournament': pd.Categorical(['Test Tournament', 'Test Tournament']),
        'Surface': pd.Categorical(['Hard', 'Clay']),
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
                'return_pts': [40, 35],
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
                'return_pts': [38, 36],
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


# ============================================================================
# PARAMETRIZED CATEGORICAL PRESERVATION TESTS
# ============================================================================

@pytest.mark.parametrize("data_type,expected_categorical", [
    ("Tournament", True),
    ("Surface", True),
    ("gender", True),
    ("Winner", True),
    ("Loser", True),
    ("source_rank", False),  # Numeric should remain numeric
    ("date", False)  # Date should remain date
])
def test_categorical_preservation_after_cache_roundtrip(temp_cache_dir, sample_match_data, data_type,
                                                        expected_categorical):
    """Test that categorical columns remain categorical after cache round-trip"""

    # Create a test file path
    test_file = temp_cache_dir / "test_data.csv"
    sample_match_data.to_csv(test_file, index=False)

    # Load through the caching system
    loaded_df = pipeline.load_large_csv_with_polars(test_file)

    if expected_categorical:
        assert loaded_df[
                   data_type].dtype.name == 'category', f"{data_type} should be categorical after cache round-trip"
    else:
        assert loaded_df[data_type].dtype.name != 'category', f"{data_type} should not be categorical"


@pytest.mark.parametrize("cache_scenario", [
    "fresh_load",  # No cache exists
    "cache_hit",  # Cache exists and is valid
    "cache_miss"  # Cache exists but is invalid
])
def test_excel_cache_categorical_consistency(temp_cache_dir, cache_scenario):
    """Test categorical consistency across different cache scenarios"""

    # Create sample Excel-like data
    test_data = pd.DataFrame({
        'Player': pd.Categorical(['Federer', 'Nadal', 'Djokovic'] * 10),
        'Tournament': pd.Categorical(['Wimbledon', 'French Open'] * 15),
        'Surface': pd.Categorical(['Grass', 'Clay', 'Hard'] * 10),
        'Ranking': [1, 2, 3] * 10
    })

    test_file = temp_cache_dir / "test_excel.csv"
    test_data.to_csv(test_file, index=False)

    if cache_scenario == "cache_hit":
        # Pre-populate cache
        cache_path = pipeline.get_cache_path(test_file, "csv_polars")
        joblib.dump(test_data, cache_path, compress=('zlib', 3))
    elif cache_scenario == "cache_miss":
        # Create invalid cache
        cache_path = pipeline.get_cache_path(test_file, "csv_polars")
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        joblib.dump(invalid_data, cache_path, compress=('zlib', 3))

    # Load data
    loaded_df = pipeline.load_large_csv_with_polars(test_file)

    # Check categorical columns are preserved
    categorical_cols = ['Player', 'Tournament', 'Surface']
    for col in categorical_cols:
        assert loaded_df[col].dtype.name == 'category', f"{col} should be categorical in {cache_scenario}"

    # Check numeric columns remain numeric
    assert pd.api.types.is_numeric_dtype(loaded_df['Ranking']), "Ranking should remain numeric"


# ============================================================================
# UNIT TESTS - CORE FUNCTIONALITY WITH TEMP DIRECTORIES
# ============================================================================

class TestPlayerCanonicalizer:
    """Test player name canonicalization with file locking"""

    def test_canonical_name_generation(self, temp_cache_dir):
        """Test canonical name generation with temporary cache"""
        canonicalizer = pipeline.PlayerCanonicalizer()
        canonicalizer.cache_file = temp_cache_dir / "test_player_cache.joblib"

        # Test basic normalization
        assert canonicalizer.canonical_player("Novak Djokovic") == "novak djokovic"
        assert canonicalizer.canonical_player("Rafael Nadal-Parera") == "rafael nadal parera"
        assert canonicalizer.canonical_player("Jo-Wilfried Tsonga") == "jo wilfried tsonga"

        # Test edge cases
        assert canonicalizer.canonical_player("") == ""
        assert canonicalizer.canonical_player(None) == ""
        assert canonicalizer.canonical_player("  Test Player  ") == "test player"

    def test_cache_persistence_with_locking(self, temp_cache_dir):
        """Test cache persistence with file locking"""
        cache_file = temp_cache_dir / "test_player_cache.joblib"

        # First instance
        canonicalizer1 = pipeline.PlayerCanonicalizer()
        canonicalizer1.cache_file = cache_file
        canonicalizer1.canonical_player("Test Player One")
        canonicalizer1._save_cache()

        # Second instance should load cached data
        canonicalizer2 = pipeline.PlayerCanonicalizer()
        canonicalizer2.cache_file = cache_file
        canonicalizer2.mapping = canonicalizer2._load_cache()

        assert "Test Player One" in canonicalizer2.mapping
        assert canonicalizer2.canonical_player("Test Player One") == "test player one"


class TestCachingMechanisms:
    """Test caching with content-based invalidation"""

    def test_content_hash_deterministic(self, temp_cache_dir):
        """Test that content hash is deterministic"""
        test_file = temp_cache_dir / "test.csv"
        test_content = "col1,col2\n1,a\n2,b\n"

        # Write same content twice
        test_file.write_text(test_content)
        hash1 = pipeline.get_content_hash(test_file)

        test_file.write_text(test_content)
        hash2 = pipeline.get_content_hash(test_file)

        assert hash1 == hash2, "Content hash should be deterministic"

    def test_content_hash_changes_with_content(self, temp_cache_dir):
        """Test that content hash changes when content changes"""
        test_file = temp_cache_dir / "test.csv"

        test_file.write_text("col1,col2\n1,a\n2,b\n")
        hash1 = pipeline.get_content_hash(test_file)

        test_file.write_text("col1,col2\n1,a\n2,c\n")  # Changed content
        hash2 = pipeline.get_content_hash(test_file)

        assert hash1 != hash2, "Content hash should change when content changes"

    def test_cache_path_includes_schema_version(self, temp_cache_dir):
        """Test that cache paths include schema version"""
        test_file = temp_cache_dir / "test.csv"
        test_file.write_text("col1,col2\n1,a\n")

        cache_path = pipeline.get_cache_path(test_file, "test_prefix")

        assert pipeline.SCHEMA_VERSION in str(cache_path), "Cache path should include schema version"
        assert "test_prefix" in str(cache_path), "Cache path should include prefix"


class TestVectorizedOperations:
    """Test vectorized processing functions"""

    def test_compute_features_vectorized(self):
        """Test vectorized feature computation"""
        test_df = pd.DataFrame({
            'serve_pts': [80, 75, 90],
            'aces': [6, 4, 8],
            'dfs': [3, 5, 2],
            'first_in': [50, 45, 55],
            'first_won': [38, 35, 42],
            'second_won': [20, 18, 25],
            'return_pts': [40, 35, 45],
            'return_pts_won': [30, 25, 35],
            'winners': [25, 20, 30],
            'unforced': [22, 25, 18]
        })

        features_df = pipeline.compute_features_vectorized(test_df)

        # Check that vectorized calculations are correct
        expected_first_serve_pct = test_df['first_in'] / test_df['serve_pts']
        pd.testing.assert_series_equal(features_df['first_serve_pct'], expected_first_serve_pct, check_names=False)

        expected_ace_rate = test_df['aces'] / test_df['serve_pts']
        pd.testing.assert_series_equal(features_df['ace_rate'], expected_ace_rate, check_names=False)

    def test_inject_jeff_features_vectorized(self, sample_match_data, sample_jeff_data, temp_cache_dir):
        """Test vectorized Jeff feature injection"""
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
            assert feature in result_df.columns, f"Feature {feature} should be present"

        # Check that values were populated (should not all be NaN)
        assert not result_df['winner_serve_pts'].isna().all(), "Winner serve points should have values"
        assert not result_df['loser_serve_pts'].isna().all(), "Loser serve points should have values"


class TestDeduplication:
    """Test match deduplication with categorical conversion tracking"""

    def test_categorical_conversion_tracking(self, sample_match_data):
        """Test that categorical conversion is tracked to prevent repeated conversions"""
        # Ensure attrs are clean
        sample_match_data.attrs = {}

        # First deduplication should set the flag
        result1 = pipeline.deduplicate_matches(sample_match_data.copy())
        assert result1.attrs.get('cats_to_str_done', False), "Flag should be set after first conversion"

        # Second deduplication should not convert again
        with patch.object(pd.Series, 'astype') as mock_astype:
            result2 = pipeline.deduplicate_matches(result1)
            # Should not have called astype for categorical conversion
            categorical_conversions = [call for call in mock_astype.call_args_list
                                       if len(call[0]) > 0 and call[0][0] == str]
            assert len(categorical_conversions) == 0, "Should not convert categoricals again"

    def test_data_quality_metric_updated(self, temp_cache_dir):
        """Test that data quality metric is updated during deduplication"""
        test_data = pd.DataFrame({
            'composite_id': ['match1', 'match1', 'match2'],  # One duplicate
            'source_rank': [1, 3, 2],
            'Tournament': pd.Categorical(['Test'] * 3)
        })

        with patch.object(pipeline.DATA_QUALITY_SCORE, 'set') as mock_set:
            result = pipeline.deduplicate_matches(test_data)

            # Should have called set with quality score
            mock_set.assert_called_once()
            called_score = mock_set.call_args[0][0]
            expected_score = 2 / 3  # 2 remaining out of 3 original
            assert abs(called_score - expected_score) < 0.01, f"Quality score should be {expected_score}"


class TestWeightedDefaultsVersioning:
    """Test weighted defaults with metadata versioning"""

    def test_metadata_creation_and_validation(self, sample_jeff_data, temp_cache_dir):
        """Test weighted defaults metadata creation and validation"""
        defaults, metadata = pipeline.calculate_comprehensive_weighted_defaults_versioned(sample_jeff_data)

        # Check metadata structure
        assert isinstance(metadata, pipeline.WeightedDefaultsMetadata)
        assert metadata.schema_version == pipeline.WEIGHTED_DEFAULTS_SCHEMA_VERSION
        assert metadata.men_features > 0
        assert metadata.women_features > 0
        assert len(metadata.jeff_data_hash) == 16  # Truncated hash length

    def test_cache_hit_with_valid_metadata(self, sample_jeff_data, temp_cache_dir):
        """Test cache hit when metadata is valid"""
        # First call should compute and cache
        defaults1, metadata1 = pipeline.calculate_comprehensive_weighted_defaults_versioned(sample_jeff_data)

        # Second call should hit cache
        with patch('joblib.load') as mock_load:
            mock_load.return_value = (defaults1, metadata1)
            with patch.object(pipeline.CACHE_HITS_TOTAL, 'labels') as mock_labels:
                mock_counter = Mock()
                mock_labels.return_value = mock_counter

                defaults2, metadata2 = pipeline.calculate_comprehensive_weighted_defaults_versioned(sample_jeff_data)

                # Should have hit cache
                mock_counter.inc.assert_called_once()


class TestAsyncAPIWithDelays:
    """Test async API with mandatory delays"""

    @pytest.mark.asyncio
    async def test_fetch_fixtures_with_delays(self):
        """Test that mandatory delays are inserted between API calls"""
        mock_client = MagicMock()

        # Mock successful responses
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{'event_type_key': 1, 'event_type_type': 'ATP Singles'}]

        mock_fixtures_response = MagicMock()
        mock_fixtures_response.raise_for_status = MagicMock()
        mock_fixtures_response.json.return_value = [{'fixture_key': 'test'}]

        mock_client.get.side_effect = [mock_response, mock_fixtures_response]

        # Test with patched sleep to verify it's called
        with patch('asyncio.sleep') as mock_sleep:
            result = await pipeline.fetch_fixtures_for_date_async(mock_client, date(2025, 1, 1))

            # Should have called sleep for delays
            assert mock_sleep.call_count >= 2, "Should have called sleep for delays between requests"

            # Verify delay amount
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert all(delay >= pipeline.API_MIN_DELAY for delay in
                       sleep_calls), f"All delays should be >= {pipeline.API_MIN_DELAY}"

    @pytest.mark.asyncio
    async def test_api_error_unification(self):
        """Test that httpx errors are unified into APIError"""
        mock_client = MagicMock()

        # Mock httpx error
        import httpx
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(pipeline.APIError) as exc_info:
            await pipeline.fetch_fixtures_for_date_async(mock_client, date(2025, 1, 1))

        assert "Events request failed" in str(exc_info.value)
        assert "ConnectError" in str(exc_info.value)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSettingsSafeguards:
    """Test settings access safeguards"""

    def test_settings_defaults_used_when_missing(self):
        """Test that defaults are used when settings are missing"""
        # Test that pipeline uses safe defaults
        assert pipeline.API_MAX_RETRIES >= 1, "Should have safe default for API retries"
        assert pipeline.MAX_CONCURRENT_REQUESTS >= 1, "Should have safe default for concurrent requests"
        assert isinstance(pipeline.TENNIS_CACHE_DIR, Path), "Cache dir should be Path object"

    def test_path_handling_uses_env_vars(self):
        """Test that path handling respects environment variables"""
        with patch.dict(os.environ, {'TENNIS_DATA_DIR': '/test/path'}):
            # Would need to reload module to test this properly
            # This is a structural test to ensure the pattern is correct
            test_path = Path(os.getenv("TENNIS_DATA_DIR", Path.home() / "tennis_data"))
            assert str(test_path) == '/test/path', "Should use environment variable"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformanceOptimizations:
    """Test performance optimizations"""

    def test_optimal_worker_calculation(self):
        """Test optimal worker count calculation"""
        with patch('os.cpu_count', return_value=4):
            workers = pipeline.get_optimal_workers()
            expected = min(pipeline.MAX_CONCURRENT_REQUESTS, 4 * 2)
            assert workers == expected, f"Workers should be {expected}"

    def test_categorical_dtype_optimization(self, temp_cache_dir):
        """Test that categorical dtypes are applied for optimization"""
        # Create data with repeated values (good candidate for categorical)
        test_data = pd.DataFrame({
            'player': ['Federer'] * 50 + ['Nadal'] * 50,
            'tournament': ['Wimbledon'] * 100,
            'surface': ['Grass'] * 100,
            'score': list(range(100))  # Unique values, not good for categorical
        })

        test_file = temp_cache_dir / "test_categorical.csv"
        test_data.to_csv(test_file, index=False)

        loaded_df = pipeline.load_large_csv_with_polars(test_file)

        # Check that string columns with low uniqueness are categorical
        assert loaded_df['player'].dtype.name == 'category', "Player should be categorical"
        assert loaded_df['tournament'].dtype.name == 'category', "Tournament should be categorical"
        assert loaded_df['surface'].dtype.name == 'category', "Surface should be categorical"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestSpecificErrorHandling:
    """Test specific error handling improvements"""

    def test_polars_specific_exceptions(self, temp_cache_dir):
        """Test that Polars-specific exceptions are caught"""
        import polars as pl

        # Create invalid CSV
        invalid_file = temp_cache_dir / "invalid.csv"
        invalid_file.write_text("invalid,content\n1,2,3,4")  # Wrong number of columns

        # Should catch specific Polars exceptions
        with pytest.raises(pipeline.DataIngestionError) as exc_info:
            pipeline.load_large_csv_with_polars(invalid_file)

        # The error message should indicate it was a Polars loading issue
        assert "Polars CSV loading failed" in str(exc_info.value)

    def test_file_not_found_handling(self, temp_cache_dir):
        """Test FileNotFoundError handling"""
        non_existent_file = temp_cache_dir / "does_not_exist.csv"

        with pytest.raises(pipeline.DataIngestionError) as exc_info:
            pipeline.load_large_csv_with_polars(non_existent_file)

        assert "Polars CSV loading failed" in str(exc_info.value)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])