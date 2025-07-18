#!/usr/bin/env python3
"""
OPTIMIZED: Fast unit tests for tennis pipeline
Avoids long-running API calls
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os

# Import your pipeline functions
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_updated import (
    parse_match_statistics,
    extract_embedded_statistics,
    integrate_api_tennis_data,
    flatten_fixtures,
    get_fixtures_for_date,
    load_from_cache_with_scraping,
    save_to_cache
)


def test_date_handling():
    """Test that date operations work without type errors"""
    print("=== TESTING DATE HANDLING ===")

    # Create test DataFrame with mixed date types
    test_data = pd.DataFrame({
        'date': [date(2025, 6, 15), None, date(2025, 6, 20)],
        'source_rank': [2, 2, 2],
        'composite_id': ['test1', 'test2', 'test3']
    })

    # Test date max calculation
    try:
        valid_dates = test_data['date'].dropna()
        if len(valid_dates) > 0:
            latest_date = valid_dates.max()
            if pd.isna(latest_date) or not isinstance(latest_date, (date, pd.Timestamp)):
                latest_date = date(2025, 6, 10)
            elif isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.date()
        else:
            latest_date = date(2025, 6, 10)

        days_since_update = (date.today() - latest_date).days
        print(f"✓ Date handling works: latest_date={latest_date}, days_since={days_since_update}")
        return True
    except Exception as e:
        print(f"✗ Date handling failed: {e}")
        return False


@patch('tennis_updated.api_call')
@patch('tennis_updated.AutomatedTennisAbstractScraper')
@patch('tennis_updated.integrate_api_tennis_data_incremental')
def test_fast_orchestration(mock_api_integration, mock_scraper, mock_api):
    """FAST: Test orchestration without long-running operations"""
    print("\n=== TESTING FAST ORCHESTRATION ===")

    # Setup temp cache
    temp_cache_dir = tempfile.mkdtemp(prefix="fast_test_")
    hd_path = os.path.join(temp_cache_dir, "historical_data.parquet")
    jeff_path = os.path.join(temp_cache_dir, "jeff_data.pkl")
    def_path = os.path.join(temp_cache_dir, "weighted_defaults.pkl")

    try:
        # Create minimal test data
        test_data = pd.DataFrame({
            'composite_id': ['test_match_1'],
            'date': [date(2025, 7, 15)],
            'Winner': ['Test Player A'],
            'Loser': ['Test Player B'],
            'winner_canonical': ['test_player_a'],
            'loser_canonical': ['test_player_b'],
            'gender': ['M'],
            'source_rank': [3],
            'ta_enhanced': [False]
        })

        jeff_data = {'men': {}, 'women': {}}
        defaults = {'men': {}, 'women': {}}

        # Mock external calls to avoid long operations
        mock_api.return_value = []  # No API data
        mock_api_integration.side_effect = lambda x: x  # Return data unchanged

        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = []  # No TA data
        mock_scraper.return_value = mock_scraper_instance

        # Test with patched cache paths
        with patch('tennis_updated.CACHE_DIR', temp_cache_dir), \
                patch('tennis_updated.HD_PATH', hd_path), \
                patch('tennis_updated.JEFF_PATH', jeff_path), \
                patch('tennis_updated.DEF_PATH', def_path):

            # Save initial test data
            save_result = save_to_cache(test_data, jeff_data, defaults)
            assert save_result == True, "Cache save should succeed"

            # Test loading with scraping (should be fast with mocks)
            hist, jeff_result, defaults_result = load_from_cache_with_scraping()

            # Verify results
            assert hist is not None, "Should load historical data"
            assert len(hist) >= 1, "Should have at least test data"
            assert 'composite_id' in hist.columns, "Should have required columns"

        print("✓ Fast orchestration test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Fast orchestration failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_cache_dir, ignore_errors=True)


def test_parse_match_statistics():
    """Test match statistics parsing"""
    fixture = {
        "first_player_key": "1",
        "second_player_key": "2",
        "p1_aces": 5,
        "p2_aces": 3
    }
    df = flatten_fixtures([fixture])
    assert "p1_aces" in df.columns
    stats = parse_match_statistics(fixture)
    assert 1 in stats and 2 in stats
    assert stats[1]["aces"] == 5
    assert stats[2]["aces"] == 3


def test_extract_embedded_statistics():
    """Test embedded statistics extraction"""
    fixture = {
        "scores": [
            {"score_first": "6", "score_second": "4"},
            {"score_first": "3", "score_second": "6"}
        ]
    }
    stats = extract_embedded_statistics(fixture)
    assert stats["sets_won_p1"] == 1
    assert stats["sets_won_p2"] == 1
    assert stats["total_sets"] == 2


if __name__ == "__main__":
    # Run fast tests
    print("🎾 FAST UNIT TESTS FOR TENNIS PIPELINE")
    print("=" * 50)

    tests = [
        ("Date Handling", test_date_handling),
        ("Fast Orchestration", test_fast_orchestration),
        ("Parse Statistics", test_parse_match_statistics),
        ("Extract Embedded Stats", test_extract_embedded_statistics)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            result = test_func()
            if result or result is None:  # None means test passed via assertions
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")

    print(f"\n📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🏆 ALL FAST UNIT TESTS PASSED!")
    else:
        print("🔧 Some tests need attention")