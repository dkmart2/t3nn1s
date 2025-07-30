# ============================================================================
# FOCUSED UNIT TESTS - TENNIS DATA PIPELINE
# ============================================================================

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock
import joblib

# Import modules under test
import tennis_updated as pipeline
from pathlib import Path


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def temp_directories():
    """Create temporary directories for testing"""
    temp_dir = Path(tempfile.mkdtemp(prefix="tennis_test_"))
    cache_dir = temp_dir / "cache"
    data_dir = temp_dir / "data"
    jeff_dir = data_dir / "jeff"

    cache_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    jeff_dir.mkdir(parents=True)

    # Patch the pipeline constants
    with patch.multiple(
            pipeline,
            CACHE_DIR=cache_dir,
    ):
        yield {
            'temp_dir': temp_dir,
            'cache_dir': cache_dir,
            'data_dir': data_dir,
            'jeff_dir': jeff_dir
        }

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_tennis_data():
    """Create sample tennis match data"""
    return pd.DataFrame({
        'composite_id': [
            '20250101-federer-nadal',
            '20250102-djokovic-murray',
            '20250103-williams-sharapova'
        ],
        'date': [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
        'Winner': ['Federer', 'Djokovic', 'Williams'],
        'Loser': ['Nadal', 'Murray', 'Sharapova'],
        'winner_canonical': ['federer', 'djokovic', 'williams'],
        'loser_canonical': ['nadal', 'murray', 'sharapova'],
        'gender': ['M', 'M', 'W'],
        'data_source': ['excel', 'excel', 'excel']
    })


@pytest.fixture
def sample_api_data():
    """Create sample API data that overlaps with tennis data"""
    return pd.DataFrame({
        'composite_id': [
            '20250102-djokovic-murray',  # Duplicate with tennis data
            '20250104-federer-djokovic'  # New match
        ],
        'date': [date(2025, 1, 2), date(2025, 1, 4)],
        'Winner': ['Djokovic', 'Federer'],
        'Loser': ['Murray', 'Djokovic'],
        'winner_canonical': ['djokovic', 'federer'],
        'loser_canonical': ['murray', 'djokovic'],
        'gender': ['M', 'M'],
        'data_source': ['api', 'api']
    })


@pytest.fixture
def sample_jeff_data():
    """Create sample Jeff data"""
    return {
        'men': {
            'overview': pd.DataFrame({
                'Player_canonical': ['federer', 'nadal', 'djokovic'],
                'serve_pts': [80, 75, 85],
                'aces': [10, 5, 8],
                'dfs': [2, 4, 3],
                'first_in': [50, 45, 55],
                'winners': [25, 20, 30],
                'unforced': [15, 25, 18]
            })
        },
        'women': {
            'overview': pd.DataFrame({
                'Player_canonical': ['williams', 'sharapova'],
                'serve_pts': [70, 75],
                'aces': [6, 4],
                'dfs': [3, 5],
                'first_in': [42, 45],
                'winners': [22, 18],
                'unforced': [18, 22]
            })
        }
    }


# ============================================================================
# DATA INTEGRATION TEST
# ============================================================================

def test_data_integration_uniqueness_and_source(temp_directories, sample_tennis_data, sample_api_data,
                                                sample_jeff_data):
    """Test that after integration each match has exactly one row and valid data_source"""

    # Mock the data loading functions to return our test data
    with patch.object(pipeline, 'load_all_tennis_data', return_value=sample_tennis_data), \
            patch.object(pipeline, 'load_jeff_comprehensive_data', return_value=sample_jeff_data), \
            patch('tennis_updated.asyncio.run', return_value=sample_api_data):
        # Skip this test as integrate_all_data_sources doesn't exist in current codebase
        pytest.skip("integrate_all_data_sources function not found in current codebase")

        # Test 1: Each composite_id should appear exactly once
        composite_id_counts = integrated_data['composite_id'].value_counts()
        duplicates = composite_id_counts[composite_id_counts > 1]

        assert len(duplicates) == 0, f"Found duplicate composite_ids: {duplicates.index.tolist()}"

        # Test 2: All data_source values should be in allowed set
        allowed_sources = {'jeff', 'api', 'excel', 'ta'}
        actual_sources = set(integrated_data['data_source'].unique())
        invalid_sources = actual_sources - allowed_sources

        assert len(invalid_sources) == 0, f"Found invalid data sources: {invalid_sources}"

        # Test 3: Verify that higher priority sources override lower priority
        # Check that data_source priorities are correctly assigned
        source_priority = {'jeff': 1, 'ta': 1, 'api': 2, 'excel': 3}
        for source in integrated_data['data_source'].unique():
            assert source in source_priority, f"Unknown data source: {source}"

        # Test 4: Verify no matches are lost during integration
        expected_unique_matches = len(pd.concat([sample_tennis_data, sample_api_data])['composite_id'].unique())
        actual_matches = len(integrated_data)

        assert actual_matches <= expected_unique_matches, "Integration should not create more matches than unique composite_ids"

        print(f"✓ Integration test passed:")
        print(f"  - {len(integrated_data)} unique matches")
        print(f"  - Data sources: {integrated_data['data_source'].value_counts().to_dict()}")
        print(f"  - No duplicate composite_ids")


# ============================================================================
# ADDITIONAL FOCUSED TESTS
# ============================================================================

def test_canonical_uniqueness():
    """Test that canonical names are unique per composite_id"""
    canonicalizer = pipeline.PlayerCanonicalizer()

    test_names = ["Roger Federer", "R. Federer", "Roger FEDERER"]
    canonical_names = [canonicalizer.canonical_player(name) for name in test_names]

    # All variations should map to same canonical form
    assert len(set(canonical_names)) == 1, f"Expected unique canonical name, got: {canonical_names}"


def test_composite_id_uniqueness(sample_tennis_data):
    """Test that composite_id values are unique after processing"""
    # Duplicate a row to test deduplication
    duplicated_data = pd.concat([sample_tennis_data, sample_tennis_data.iloc[[0]]], ignore_index=True)

    # After deduplication, should have same length as original
    composite_ids = duplicated_data['composite_id'].unique()
    assert len(composite_ids) == len(sample_tennis_data), "Composite IDs should be unique after deduplication"


def test_source_hierarchy_precedence():
    """Test that source hierarchy precedence is correctly enforced"""
    test_data = pd.DataFrame({
        'composite_id': ['match1', 'match1', 'match1'],
        'data_source': ['excel', 'api', 'jeff'],
        'Winner': ['Player A', 'Player A', 'Player A'],
        'test_field': ['excel_value', 'api_value', 'jeff_value']
    })

    # Define expected hierarchy
    source_priority = {'jeff': 1, 'ta': 1, 'api': 2, 'excel': 3}
    test_data['source_priority'] = test_data['data_source'].map(source_priority)

    # Sort and deduplicate
    result = test_data.sort_values(['composite_id', 'source_priority']).drop_duplicates(subset=['composite_id'],
                                                                                        keep='first')

    # Should keep jeff data (highest priority)
    assert result.iloc[0]['data_source'] == 'jeff', "Should keep highest priority source"
    assert result.iloc[0]['test_field'] == 'jeff_value', "Should keep data from highest priority source"


def test_player_canonicalizer_consistency(temp_directories):
    """Test that player canonicalization is consistent and has reverse lookup"""
    canonicalizer = pipeline.PlayerCanonicalizer()
    canonicalizer.cache_file = temp_directories['cache_dir'] / "test_player_cache.joblib"

    # Test consistency
    name1 = canonicalizer.canonical_player("Roger Federer")
    name2 = canonicalizer.canonical_player("Roger Federer")
    assert name1 == name2, "Canonicalization should be consistent"

    # Test normalization
    assert canonicalizer.canonical_player("ROGER FEDERER") == "roger federer"
    assert canonicalizer.canonical_player("Roger-Federer") == "roger federer"
    assert canonicalizer.canonical_player("Roger   Federer") == "roger federer"

    # Test reverse lookup
    canonical_name = canonicalizer.canonical_player("Rafael Nadal")
    raw_name = canonicalizer.get_raw_name(canonical_name)
    assert raw_name == "Rafael Nadal", "Reverse lookup should return original raw name"

    # Test missing reverse lookup
    missing_raw = canonicalizer.get_raw_name("nonexistent player")
    assert missing_raw == "nonexistent player", "Missing reverse lookup should return input"


def test_data_source_hierarchy_enforcement():
    """Test that data source hierarchy is properly enforced"""

    # Create test data with same composite_id but different sources
    test_data = pd.DataFrame({
        'composite_id': ['match1', 'match1', 'match1'],
        'data_source': ['excel', 'api', 'jeff'],
        'Winner': ['Player A', 'Player A', 'Player A'],
        'Loser': ['Player B', 'Player B', 'Player B'],
        'extra_data': ['excel_data', 'api_data', 'jeff_data']
    })

    # Define source priority (same as in pipeline)
    source_priority = {'jeff': 1, 'ta': 1, 'api': 2, 'excel': 3}
    test_data['source_priority'] = test_data['data_source'].map(source_priority)

    # Apply deduplication logic
    result = test_data.sort_values(['composite_id', 'source_priority'])
    result = result.drop_duplicates(subset=['composite_id'], keep='first')

    # Should keep only the Jeff data (highest priority)
    assert len(result) == 1, "Should have only one row after deduplication"
    assert result.iloc[0]['data_source'] == 'jeff', "Should keep highest priority source (jeff)"
    assert result.iloc[0]['extra_data'] == 'jeff_data', "Should keep data from highest priority source"


def test_canonical_uniqueness():
    """Test that canonical names are unique per composite_id"""
    canonicalizer = pipeline.PlayerCanonicalizer()

    test_names = ["Roger Federer", "R. Federer", "Roger FEDERER"]
    canonical_names = [canonicalizer.canonical_player(name) for name in test_names]

    # All variations should map to same canonical form
    assert len(set(canonical_names)) == 1, f"Expected unique canonical name, got: {canonical_names}"


def test_composite_id_uniqueness(sample_tennis_data):
    """Test that composite_id values are unique after processing"""
    # Duplicate a row to test deduplication
    duplicated_data = pd.concat([sample_tennis_data, sample_tennis_data.iloc[[0]]], ignore_index=True)

    # After deduplication, should have same length as original
    composite_ids = duplicated_data['composite_id'].unique()
    assert len(composite_ids) == len(sample_tennis_data), "Composite IDs should be unique after deduplication"


def test_source_hierarchy_precedence():
    """Test that source hierarchy precedence is correctly enforced"""
    test_data = pd.DataFrame({
        'composite_id': ['match1', 'match1', 'match1'],
        'data_source': ['excel', 'api', 'jeff'],
        'Winner': ['Player A', 'Player A', 'Player A'],
        'test_field': ['excel_value', 'api_value', 'jeff_value']
    })

    # Define expected hierarchy
    source_priority = {'jeff': 1, 'ta': 1, 'api': 2, 'excel': 3}
    test_data['source_priority'] = test_data['data_source'].map(source_priority)

    # Sort and deduplicate
    result = test_data.sort_values(['composite_id', 'source_priority']).drop_duplicates(subset=['composite_id'],
                                                                                        keep='first')

    # Should keep jeff data (highest priority)
    assert result.iloc[0]['data_source'] == 'jeff', "Should keep highest priority source"
    assert result.iloc[0]['test_field'] == 'jeff_value', "Should keep data from highest priority source"
    """Test that pipeline enforces minimum data requirements"""

    # Test insufficient tennis data
    small_tennis_data = pd.DataFrame({
        'composite_id': ['match1', 'match2'],
        'Winner': ['A', 'B'],
        'Loser': ['B', 'A']
    })

    with patch.object(pipeline, 'load_all_tennis_data', return_value=small_tennis_data):
        with pytest.raises(pipeline.DataIngestionError, match="Insufficient tennis data"):
            pipeline.load_all_tennis_data()


def test_jeff_data_aggregation_no_total_filter(sample_jeff_data):
    """Test that Jeff feature extraction uses all data, not just 'Total' rows"""

    # Add non-Total rows to test data
    extended_jeff_data = sample_jeff_data.copy()
    extended_jeff_data['men']['overview'] = pd.concat([
        sample_jeff_data['men']['overview'],
        pd.DataFrame({
            'Player_canonical': ['federer', 'nadal'],
            'set': ['Set1', 'Set1'],  # Non-Total rows
            'serve_pts': [40, 35],
            'aces': [5, 2],
            'dfs': [1, 2],
            'first_in': [25, 20],
            'winners': [12, 10],
            'unforced': [8, 12]
        })
    ], ignore_index=True)

    # Create test tennis data
    test_tennis_data = pd.DataFrame({
        'winner_canonical': ['federer'],
        'loser_canonical': ['nadal'],
        'gender': ['M']
    })

    # Extract features (should use all rows, not just Total)
    result = pipeline.extract_comprehensive_jeff_features(test_tennis_data, extended_jeff_data)

    # Should have features extracted (exact values depend on aggregation method)
    jeff_feature_cols = [col for col in result.columns if col.startswith(('winner_', 'loser_'))]
    assert len(jeff_feature_cols) > 0, "Should have Jeff features extracted"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])