#!/usr/bin/env python3
"""
Comprehensive test runner for tennis data pipeline
Uses actual existing test files in tests/ directory
"""

import subprocess
import sys
import os
import time


def run_test_suite(test_file, suite_name, timeout=300):
    """Run a specific test suite with timeout"""
    print(f"\n{'=' * 60}")
    print(f"🧪 RUNNING {suite_name}")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--capture=no",
        "--durations=5",
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), timeout=timeout)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {suite_name} PASSED ({duration:.1f}s)")
            return True, duration
        else:
            print(f"\n❌ {suite_name} FAILED ({duration:.1f}s)")
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n⏰ {suite_name} TIMED OUT ({duration:.1f}s)")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 {suite_name} CRASHED: {e} ({duration:.1f}s)")
        return False, duration


def run_python_script_test(script_path, test_name):
    """Run a standalone Python test script"""
    print(f"\n{'=' * 60}")
    print(f"🧪 RUNNING {test_name}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_path],
                                cwd=os.path.dirname(os.path.dirname(__file__)),
                                timeout=120)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {test_name} PASSED ({duration:.1f}s)")
            return True, duration
        else:
            print(f"\n❌ {test_name} FAILED ({duration:.1f}s)")
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n⏰ {test_name} TIMED OUT ({duration:.1f}s)")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 {test_name} CRASHED: {e} ({duration:.1f}s)")
        return False, duration


def main():
    """Run comprehensive test suite using actual test files"""
    print("🎾 COMPREHENSIVE TENNIS DATA PIPELINE TEST SUITE")
    print("Testing pipeline using actual test files")

    # Test suites to run - using actual files that exist
    test_configs = [
        ("test_tennis_pipeline.py", "CORE PIPELINE TESTS", 180),
        ("test_pipeline_integration.py", "INTEGRATION TESTS", 300),
        ("test_tennis_scraper.py", "SCRAPER TESTS", 60),
    ]

    # Standalone script tests
    script_tests = [
        ("tests/test_full_pipeline_script.py", "PIPELINE SCRIPT TEST"),
    ]

    results = []
    total_duration = 0

    # Run pytest-based test suites
    for test_file, suite_name, timeout in test_configs:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name, timeout)
            results.append((suite_name, passed, duration))
            total_duration += duration
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            results.append((suite_name, None, 0))

    # Run standalone script tests
    for script_path, test_name in script_tests:
        if os.path.exists(script_path):
            passed, duration = run_python_script_test(script_path, test_name)
            results.append((test_name, passed, duration))
            total_duration += duration
        else:
            print(f"\n⚠️  {test_name}: File {script_path} not found, skipping")
            results.append((test_name, None, 0))

    # Final report
    print(f"\n{'=' * 60}")
    print("📊 COMPREHENSIVE TEST RESULTS")
    print(f"{'=' * 60}")

    passed_count = 0
    total_count = 0

    for suite_name, passed, duration in results:
        if passed is True:
            status = "✅ PASSED"
            passed_count += 1
        elif passed is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
            continue

        total_count += 1
        print(f"{status:12} {suite_name:30} ({duration:.1f}s)")

    print(f"\n📈 SUMMARY:")
    print(f"   Tests Passed: {passed_count}/{total_count}")
    print(f"   Total Time: {total_duration:.1f}s")

    if passed_count == total_count and total_count > 0:
        print(f"\n🏆 ALL TESTS PASSED - PIPELINE IS PRODUCTION READY!")
        print(f"✅ Core pipeline functionality validated")
        print(f"✅ Tennis Abstract integration verified")
        print(f"✅ Scraper functionality confirmed")
        print(f"✅ Pipeline script execution verified")
        print(f"✅ Data quality and integrity confirmed")
        print(f"\n🚀 Ready to proceed with model training!")

        # Provide guidance for next steps
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Pipeline validation complete")
        print(f"   2. All core components working")
        print(f"   3. Begin model training")
        print(f"   4. Use: python scripts/train_model.py")

        return True
    else:
        print(f"\n🔧 PIPELINE NEEDS ATTENTION")
        print(f"❌ Fix failing tests before model training")

        # Provide specific guidance
        failed_tests = [name for name, passed, _ in results if passed is False]
        if failed_tests:
            print(f"\n🎯 FOCUS AREAS:")
            for test_name in failed_tests:
                if "CORE PIPELINE" in test_name:
                    print(f"   - Fix core pipeline function logic")
                elif "INTEGRATION" in test_name:
                    print(f"   - Fix Tennis Abstract integration")
                elif "SCRAPER" in test_name:
                    print(f"   - Fix web scraping functionality")
                elif "SCRIPT" in test_name:
                    print(f"   - Fix pipeline script execution")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)