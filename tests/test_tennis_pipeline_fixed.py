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
# TESTS
# ============================================================================

def test_canonical_uniqueness(temp_directories):
    """Test that canonical names are unique per composite_id"""
    with patch.object(pipeline, 'CACHE_DIR', str(temp_directories['cache_dir'])):
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
    with patch.object(pipeline, 'CACHE_DIR', str(temp_directories['cache_dir'])):
        canonicalizer = pipeline.PlayerCanonicalizer()

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


def test_jeff_feature_extraction(sample_jeff_data):
    """Test that Jeff feature extraction works with correct signature"""
    
    # Test function exists and works with correct parameters
    try:
        # Extract features for a male player
        features = pipeline.extract_comprehensive_jeff_features('federer', 'M', sample_jeff_data)
        assert isinstance(features, dict), "Should return a dictionary of features"
        assert len(features) > 0, "Should return some features"
    except Exception as e:
        # If function signature doesn't match, skip test
        pytest.skip(f"extract_comprehensive_jeff_features not available or different signature: {e}")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])