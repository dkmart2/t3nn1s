#!/usr/bin/env python3
"""
Comprehensive test for Jeff extraction integration
Run this after quick test passes
"""

import time
import pandas as pd
import numpy as np
from collections import defaultdict


def test_feature_quality():
    """Test that extracted features have reasonable values"""

    print("=== TESTING FEATURE QUALITY ===\n")

    from tennis_updated import load_jeff_comprehensive_data, extract_comprehensive_jeff_features

    jeff_data = load_jeff_comprehensive_data()

    # Test multiple players
    test_players = [
        ('djokovic_n', 'M'),
        ('federer_r', 'M'),
        ('nadal_r', 'M'),
        ('serena_w', 'W'),
        ('azarenka_v', 'W')
    ]

    quality_issues = []

    for player, gender in test_players:
        print(f"Testing {player}...")

        try:
            features = extract_comprehensive_jeff_features(player, gender, jeff_data)

            # Check for reasonable percentage values
            percentage_features = [k for k in features.keys() if 'pct' in k or 'rate' in k]

            for feature in percentage_features:
                value = features.get(feature, 0)
                if value < 0 or value > 1:
                    quality_issues.append(f"{player}: {feature} = {value} (outside 0-1 range)")

            # Check for NaN values
            nan_features = [k for k, v in features.items() if pd.isna(v)]
            if nan_features:
                quality_issues.append(f"{player}: {len(nan_features)} NaN features")

            # Check for reasonable serve stats
            sb_serve_pts = features.get('sb_serve_pts', 0)
            if sb_serve_pts > 0:
                sb_aces = features.get('sb_aces', 0)
                ace_rate = sb_aces / sb_serve_pts
                if ace_rate > 0.5:  # More than 50% aces is unrealistic
                    quality_issues.append(f"{player}: Unrealistic ace rate {ace_rate:.3f}")

            # Check break point stats make sense
            kps_bp_faced = features.get('kps_bp_faced', 0)
            kps_bp_saved = features.get('kps_bp_saved', 0)
            if kps_bp_saved > kps_bp_faced:
                quality_issues.append(f"{player}: More BP saved than faced")

        except Exception as e:
            quality_issues.append(f"{player}: Extraction error - {e}")

    if quality_issues:
        print("⚠️  Quality issues found:")
        for issue in quality_issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(quality_issues) > 10:
            print(f"   ... and {len(quality_issues) - 10} more")
    else:
        print("✅ All feature quality checks passed!")

    return len(quality_issues) == 0


def test_performance():
    """Test extraction performance"""

    print("\n=== TESTING PERFORMANCE ===\n")

    from tennis_updated import load_jeff_comprehensive_data, extract_comprehensive_jeff_features

    # Time data loading
    print("Testing data loading speed...")
    start_time = time.time()
    jeff_data = load_jeff_comprehensive_data()
    load_time = time.time() - start_time
    print(f"✅ Data loading: {load_time:.2f} seconds")

    # Time extraction
    print("Testing extraction speed...")
    test_player = 'djokovic_n'
    iterations = 5

    times = []
    for i in range(iterations):
        start_time = time.time()
        features = extract_comprehensive_jeff_features(test_player, 'M', jeff_data)
        times.append(time.time() - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"✅ Average extraction time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"✅ Features extracted: {len(features)}")

    # Estimate for full dataset
    estimated_1000 = avg_time * 1000 / 60
    print(f"📊 Estimated time for 1000 players: {estimated_1000:.1f} minutes")

    # Memory estimation
    import sys
    feature_size = sys.getsizeof(features)
    estimated_memory = feature_size * 1000 / 1024 / 1024
    print(f"📊 Estimated memory for 1000 players: {estimated_memory:.1f} MB")

    return avg_time < 2.0  # Should be under 2 seconds


def test_pipeline_integration():
    """Test integration with main pipeline"""

    print("\n=== TESTING PIPELINE INTEGRATION ===\n")

    try:
        from tennis_updated import generate_comprehensive_historical_data

        print("Testing small dataset generation...")
        start_time = time.time()

        # Generate small test dataset
        historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(
            fast=True,
            n_sample=50,  # Small sample
            use_synthetic=False  # Use real data
        )

        generation_time = time.time() - start_time

        print(f"✅ Dataset generation: {generation_time:.2f} seconds")
        print(f"✅ Dataset shape: {historical_data.shape}")

        # Check for new feature columns
        new_feature_prefixes = ['sb_', 'kps_', 'kpr_', 'np_', 'rally_', 'sd_', 'ro_', 'rd_',
                                'si_', 'shotd_', 'sdo_', 'st_', 'snv_', 'svbs_', 'svbt_', 'matches_']

        found_columns = defaultdict(int)

        for col in historical_data.columns:
            for prefix in new_feature_prefixes:
                if prefix in col:
                    found_columns[prefix] += 1
                    break

        print(f"\n📊 New feature columns found:")
        total_new_cols = 0
        for prefix, count in found_columns.items():
            if count > 0:
                print(f"   {prefix[:-1]}: {count} columns")
                total_new_cols += count

        print(f"   Total new columns: {total_new_cols}")

        # Check feature population
        if total_new_cols > 0:
            sample_cols = []
            for prefix in new_feature_prefixes:
                prefix_cols = [col for col in historical_data.columns if prefix in col]
                if prefix_cols:
                    sample_cols.extend(prefix_cols[:2])  # Sample 2 from each category

            if sample_cols:
                non_null_rates = []
                for col in sample_cols[:10]:  # Check first 10 sample columns
                    non_null_count = historical_data[col].notna().sum()
                    rate = non_null_count / len(historical_data)
                    non_null_rates.append(rate)

                avg_population = np.mean(non_null_rates)
                print(f"📊 Average feature population rate: {avg_population:.1%}")

                if avg_population > 0.1:  # At least 10% populated
                    print("✅ Good feature population in pipeline")
                else:
                    print("⚠️  Low feature population - check integration")

        return total_new_cols > 50  # Should have many new columns

    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_coverage():
    """Test how many players have data in different files"""

    print("\n=== TESTING DATA COVERAGE ===\n")

    from tennis_updated import load_jeff_comprehensive_data, extract_comprehensive_jeff_features

    jeff_data = load_jeff_comprehensive_data()

    # Test common players
    common_players = [
        'djokovic_n', 'federer_r', 'nadal_r', 'murray_a', 'wawrinka_s',
        'serena_w', 'sharapova_m', 'halep_s', 'azarenka_v', 'kvitova_p'
    ]

    coverage_stats = defaultdict(int)

    for player in common_players:
        gender = 'M' if player.endswith('_n') or player.endswith('_r') or player.endswith('_a') or player.endswith(
            '_s') else 'W'

        try:
            features = extract_comprehensive_jeff_features(player, gender, jeff_data)

            # Count which categories have data
            categories = ['sb_', 'kps_', 'kpr_', 'np_', 'rally_', 'sd_', 'ro_', 'rd_',
                          'si_', 'shotd_', 'sdo_', 'st_', 'snv_', 'svbs_', 'svbt_', 'matches_']

            for category in categories:
                if any(k.startswith(category) for k in features.keys()):
                    coverage_stats[category] += 1

        except Exception as e:
            print(f"   ❌ {player}: {e}")

    print("📊 Data coverage across top players:")
    total_players = len(common_players)

    for category, count in sorted(coverage_stats.items()):
        rate = count / total_players
        print(f"   {category[:-1]}: {count}/{total_players} players ({rate:.0%})")

    # Overall coverage
    avg_coverage = np.mean(list(coverage_stats.values())) / total_players
    print(f"\n📊 Average coverage: {avg_coverage:.1%}")

    return avg_coverage > 0.5  # At least 50% average coverage


def run_comprehensive_tests():
    """Run all comprehensive tests"""

    print("COMPREHENSIVE JEFF EXTRACTION TESTS")
    print("=" * 50)

    tests = [
        ("Feature Quality", test_feature_quality),
        ("Performance", test_performance),
        ("Pipeline Integration", test_pipeline_integration),
        ("Data Coverage", test_data_coverage)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"⚠️  {test_name}: ISSUES FOUND")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Final summary
    print(f"\n{'=' * 20} COMPREHENSIVE TEST RESULTS {'=' * 20}")

    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "⚠️  FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")

    if passed_tests >= 3:  # At least 3/4 should pass
        print("🎉 COMPREHENSIVE TESTS MOSTLY SUCCESSFUL!")
        print("✅ Ready for production use")
    else:
        print("⚠️  Multiple test failures - review before production")

    return passed_tests >= 3


if __name__ == "__main__":
    run_comprehensive_tests()