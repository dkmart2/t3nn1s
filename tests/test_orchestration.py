#!/usr/bin/env python3
"""
Test script for ETL pipeline orchestration
"""

import sys
import os
import traceback
from datetime import date
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_updated import (
    load_from_cache_with_scraping,
    generate_comprehensive_historical_data,
    save_to_cache,
    integrate_api_tennis_data_incremental
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


def test_full_orchestration():
    """Test the complete ETL pipeline orchestration"""
    print("\n=== TESTING FULL ORCHESTRATION ===")

    try:
        print("Step 1: Testing cache loading with scraping...")
        hist, jeff_data, defaults = load_from_cache_with_scraping()

        if hist is None:
            print("  No cache found, generating data...")
            hist, jeff_data, defaults = generate_comprehensive_historical_data(fast=True, n_sample=100)
            if hist.empty:
                print("  ✗ Data generation failed")
                return False
            print(f"  ✓ Generated data: {len(hist)} matches")

            print("  Saving to cache...")
            if save_to_cache(hist, jeff_data, defaults):
                print("  ✓ Cache saved successfully")
            else:
                print("  ✗ Cache save failed")
                return False
        else:
            print(f"  ✓ Loaded from cache: {len(hist)} matches")

        print("Step 2: Testing incremental API integration...")
        original_count = len(hist)
        hist_updated = integrate_api_tennis_data_incremental(hist)
        print(f"  ✓ API integration complete: {len(hist_updated)} matches (was {original_count})")

        print("Step 3: Checking data quality...")

        # Check date column integrity
        if 'date' in hist_updated.columns:
            null_dates = hist_updated['date'].isna().sum()
            valid_dates = len(hist_updated) - null_dates
            print(f"  Date integrity: {valid_dates}/{len(hist_updated)} valid dates")

        # Check Tennis Abstract features
        ta_columns = [col for col in hist_updated.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
        ta_enhanced = len(hist_updated[hist_updated[ta_columns].notna().any(axis=1)]) if ta_columns else 0
        print(f"  TA enhancement: {ta_enhanced} matches with TA features ({len(ta_columns)} TA columns)")

        # Check source distribution
        if 'source_rank' in hist_updated.columns:
            source_dist = hist_updated['source_rank'].value_counts().sort_index()
            print(f"  Source distribution: {dict(source_dist)}")

        print("Step 4: Final cache update...")
        if save_to_cache(hist_updated, jeff_data, defaults):
            print("  ✓ Final cache saved")
        else:
            print("  ✗ Final cache save failed")
            return False

        print("\n✓ ORCHESTRATION TEST PASSED")
        return True

    except Exception as e:
        print(f"\n✗ ORCHESTRATION TEST FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("TENNIS ETL PIPELINE ORCHESTRATION TEST\n")

    # Test individual components first
    date_test_passed = test_date_handling()

    if date_test_passed:
        orchestration_test_passed = test_full_orchestration()

        if orchestration_test_passed:
            print("\n🎾 ALL TESTS PASSED - PIPELINE READY")
            sys.exit(0)
        else:
            print("\n❌ ORCHESTRATION FAILED")
            sys.exit(1)
    else:
        print("\n❌ DATE HANDLING FAILED")
        sys.exit(1)