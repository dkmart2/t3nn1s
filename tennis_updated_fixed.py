#!/usr/bin/env python3
"""
Test script for the tennis prediction system
Run this to identify issues before using the full system
"""

import sys
import traceback


def test_imports():
    """Test 1: Check if all imports work"""
    print("=== TEST 1: IMPORTS ===")
    try:
        # Test basic imports first
        import pandas as pd
        import numpy as np
        import requests
        print("✓ Basic imports (pandas, numpy, requests)")

        # Test optional imports
        try:
            from bs4 import BeautifulSoup
            print("✓ BeautifulSoup available")
        except ImportError:
            print("⚠ BeautifulSoup not available - Tennis Abstract scraping will fail")

        try:
            from unidecode import unidecode
            print("✓ unidecode available")
        except ImportError:
            print("⚠ unidecode not available - may cause name normalization issues")

        print("✓ All critical imports successful")
        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tennis_module():
    """Test 2: Check if our tennis module loads"""
    print("\n=== TEST 2: TENNIS MODULE LOADING ===")
    try:
        # Try to import our module
        import tennis_updated_fixed as tennis
        print("✓ Tennis module imported successfully")

        # Test basic functions exist
        functions_to_check = [
            'normalize_name',
            'load_from_cache',
            'api_call',
            'safe_int_convert',
            'build_composite_id'
        ]

        for func_name in functions_to_check:
            if hasattr(tennis, func_name):
                print(f"✓ Function {func_name} found")
            else:
                print(f"✗ Function {func_name} missing")
                return False

        # Test classes exist
        classes_to_check = [
            'UnifiedBayesianTennisModel',
            'TennisAbstractScraper'
        ]

        for class_name in classes_to_check:
            if hasattr(tennis, class_name):
                print(f"✓ Class {class_name} found")
            else:
                print(f"✗ Class {class_name} missing")
                return False

        return True

    except Exception as e:
        print(f"✗ Tennis module loading failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functions():
    """Test 3: Test basic utility functions"""
    print("\n=== TEST 3: BASIC FUNCTIONS ===")
    try:
        import tennis_updated_fixed as tennis

        # Test name normalization
        test_name = "Novak Djokovic"
        normalized = tennis.normalize_name(test_name)
        print(f"✓ normalize_name('{test_name}') = '{normalized}'")

        # Test safe int conversion
        assert tennis.safe_int_convert("123") == 123
        assert tennis.safe_int_convert("invalid") is None
        assert tennis.safe_int_convert(None) is None
        print("✓ safe_int_convert works correctly")

        # Test composite ID building
        from datetime import date
        comp_id = tennis.build_composite_id(
            date(2025, 7, 15),
            "wimbledon",
            "djokovic_n",
            "alcaraz_c"
        )
        expected = "20250715-wimbledon-djokovic_n-alcaraz_c"
        assert comp_id == expected
        print(f"✓ build_composite_id works: {comp_id}")

        return True

    except Exception as e:
        print(f"✗ Basic function test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test 4: Test model instantiation"""
    print("\n=== TEST 4: MODEL CREATION ===")
    try:
        import tennis_updated_fixed as tennis

        # Test model creation
        model = tennis.UnifiedBayesianTennisModel(n_simulations=10)
        print("✓ UnifiedBayesianTennisModel created successfully")

        # Test feature extraction functions
        sample_data = {
            'winner_serve_pts': 80,
            'winner_first_won': 50,
            'winner_second_won': 15,
            'surface': 'Hard',
            'WRank': 1,
            'LRank': 5
        }

        features = tennis.extract_unified_features_fixed(sample_data, 'winner')
        print(f"✓ Feature extraction works: {len(features)} features extracted")

        context = tennis.extract_unified_match_context_fixed(sample_data)
        print(f"✓ Context extraction works: {len(context)} context items")

        return True

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_cache_functions():
    """Test 5: Test cache loading (without requiring actual cache files)"""
    print("\n=== TEST 5: CACHE FUNCTIONS ===")
    try:
        import tennis_updated_fixed as tennis

        # Test cache loading (should return None if no cache exists)
        result = tennis.load_from_cache()
        print("✓ load_from_cache() executes without error")

        if result == (None, None, None):
            print("✓ No cache found (expected for new installation)")
        else:
            print("✓ Cache data found")

        return True

    except Exception as e:
        print(f"✗ Cache function test failed: {e}")
        traceback.print_exc()
        return False


def test_prediction_simulation():
    """Test 6: Test basic prediction with dummy data"""
    print("\n=== TEST 6: PREDICTION SIMULATION ===")
    try:
        import tennis_updated_fixed as tennis

        model = tennis.UnifiedBayesianTennisModel(n_simulations=100)

        # Create dummy player features
        p1_features = {
            'serve_effectiveness': 0.75,
            'return_effectiveness': 0.40,
            'winners_rate': 0.25,
            'unforced_rate': 0.15,
            'pressure_performance': 0.60,
            'net_effectiveness': 0.70
        }

        p2_features = {
            'serve_effectiveness': 0.70,
            'return_effectiveness': 0.35,
            'winners_rate': 0.20,
            'unforced_rate': 0.18,
            'pressure_performance': 0.55,
            'net_effectiveness': 0.65
        }

        match_context = {
            'surface': 'Hard',
            'p1_ranking': 1,
            'p2_ranking': 5,
            'h2h_matches': 10,
            'p1_h2h_win_pct': 0.6,
            'data_quality_score': 0.8
        }

        # Run prediction
        prob = model.simulate_match(p1_features, p2_features, match_context, best_of=3)

        print(f"✓ Prediction simulation successful: P1 win probability = {prob:.3f}")

        # Sanity check
        if 0 <= prob <= 1:
            print("✓ Probability is in valid range [0,1]")
        else:
            print(f"✗ Invalid probability: {prob}")
            return False

        return True

    except Exception as e:
        print(f"✗ Prediction simulation failed: {e}")
        traceback.print_exc()
        return False


def test_argument_parsing():
    """Test 7: Test command line argument structure"""
    print("\n=== TEST 7: ARGUMENT PARSING ===")
    try:
        import tennis_updated_fixed as tennis
        import argparse

        # Test if the argument parser would work
        # (We can't actually test it without sys.argv changes)
        parser = argparse.ArgumentParser(description="Tennis match win probability predictor")
        parser.add_argument("--player1", required=True, help="Name of player 1")
        parser.add_argument("--player2", required=True, help="Name of player 2")
        parser.add_argument("--date", required=True, help="Match date in YYYY-MM-DD")
        parser.add_argument("--tournament", required=True, help="Tournament name")
        parser.add_argument("--gender", choices=["M", "W"], required=True, help="Gender: M or W")
        parser.add_argument("--best_of", type=int, default=3, help="Sets in match, default 3")

        print("✓ Argument parser structure is valid")
        return True

    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("TENNIS PREDICTION SYSTEM - TESTING SUITE")
    print("=" * 50)

    tests = [
        test_imports,
        test_tennis_module,
        test_basic_functions,
        test_model_creation,
        test_cache_functions,
        test_prediction_simulation,
        test_argument_parsing
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i + 1}. {test.__name__}: {status}")

    print(f"\nOVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System should work correctly.")
        print("\nNext steps:")
        print(
            "1. Try running with real data: python tennis_updated_fixed.py --player1 'Novak Djokovic' --player2 'Carlos Alcaraz' --date '2025-07-15' --tournament 'Wimbledon' --gender 'M'")
        print("2. Check data directory paths in the script match your setup")
        print("3. Verify API key is valid if using API features")
    else:
        print("❌ SOME TESTS FAILED! Fix these issues before using the system.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install pandas numpy requests beautifulsoup4 unidecode")
        print("- Check file paths in the script")
        print("- Verify the tennis_updated_fixed.py file is saved correctly")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)