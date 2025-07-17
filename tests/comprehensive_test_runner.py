#!/usr/bin/env python3
"""
Flexible test runner for tennis data pipeline
Supports fast development tests and comprehensive integration tests
"""

import subprocess
import sys
import os
import time
import argparse


def run_test_suite(test_file, suite_name, markers=None, timeout=300):
    """Run a specific test suite with optional markers"""
    print(f"\n{'=' * 70}")
    print(f"🧪 RUNNING {suite_name}")
    if markers:
        print(f"   Markers: {', '.join(markers)}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--capture=no",
        "--durations=10",
    ]

    # Add marker filters
    if markers:
        for marker in markers:
            if marker.startswith('not '):
                cmd.extend(["-m", marker])
            else:
                cmd.extend(["-m", marker])

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


def run_fast_tests():
    """Run fast development tests (mocked)"""
    print("🏃‍♂️ FAST DEVELOPMENT TESTS")
    print("Using mocks for rapid feedback")

    test_configs = [
        ("test_tennis_pipeline.py", "CORE PIPELINE TESTS (FAST)", ["not integration", "not slow"], 60),
        ("test_pipeline_integration.py", "INTEGRATION TESTS (FAST)", ["not slow"], 120),
        ("test_tennis_scraper.py", "SCRAPER TESTS", None, 60),
    ]

    results = []
    total_duration = 0

    for test_file, suite_name, markers, timeout in test_configs:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name, markers, timeout)
            results.append((suite_name, passed, duration))
            total_duration += duration
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            results.append((suite_name, None, 0))

    return results, total_duration


def run_integration_tests():
    """Run comprehensive integration tests (real APIs)"""
    print("🔍 COMPREHENSIVE INTEGRATION TESTS")
    print("Using real APIs and external services")

    test_configs = [
        ("test_tennis_pipeline.py", "REAL API INTEGRATION TESTS", ["integration"], 600),
        ("edge_case_tests.py", "EDGE CASE TESTS", None, 600),
        ("test_pipeline_integration.py", "FULL INTEGRATION TESTS", None, 300),
        ("test_tennis_scraper.py", "SCRAPER TESTS", None, 60),
    ]

    results = []
    total_duration = 0

    for test_file, suite_name, markers, timeout in test_configs:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name, markers, timeout)
            results.append((suite_name, passed, duration))
            total_duration += duration
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            results.append((suite_name, None, 0))

    return results, total_duration


def print_results(results, total_duration, test_type=""):
    """Print formatted test results"""
    print(f"\n{'=' * 70}")
    print(f"📊 {test_type} TEST RESULTS")
    print(f"{'=' * 70}")

    passed_count = 0
    total_count = 0

    for result in results:
        if len(result) == 4:  # All tests format
            suite_name, passed, duration, phase = result
            phase_tag = f"[{phase}]"
        else:  # Fast/Integration only format
            suite_name, passed, duration = result
            phase_tag = ""

        if passed is True:
            status = "✅ PASSED"
            passed_count += 1
        elif passed is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
            continue

        total_count += 1
        print(f"{status:12} {phase_tag:8} {suite_name:35} ({duration:.1f}s)")

    print(f"\n📈 SUMMARY:")
    print(f"   Tests Passed: {passed_count}/{total_count}")
    print(f"   Total Time: {total_duration:.1f}s")
    print(f"   Success Rate: {(passed_count / total_count) * 100:.1f}%" if total_count > 0 else "   Success Rate: N/A")

    return passed_count == total_count and total_count > 0


def main():
    """Main test runner with command line options"""
    parser = argparse.ArgumentParser(description="Tennis Pipeline Test Runner")
    parser.add_argument("--mode", choices=["fast", "integration", "all"], default="fast",
                        help="Test mode: fast (mocked), integration (real APIs), or all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("🎾 TENNIS DATA PIPELINE TEST RUNNER")
    print(f"Mode: {args.mode.upper()}")

    if args.mode == "fast":
        print("\n🏃‍♂️ Running FAST tests for development feedback")
        print("• Uses mocks for external APIs")
        print("• Optimized for speed")
        print("• Safe for frequent execution")
        results, duration = run_fast_tests()
        success = print_results(results, duration, "FAST DEVELOPMENT")

        if success:
            print(f"\n✅ FAST TESTS PASSED - READY FOR DEVELOPMENT")
            print(f"💡 For production validation, run: python {sys.argv[0]} --mode integration")

    elif args.mode == "integration":
        print("\n🔍 Running INTEGRATION tests for production validation")
        print("• Uses real APIs and external services")
        print("• Comprehensive edge case coverage")
        print("• May be slow due to network calls")
        results, duration = run_integration_tests()
        success = print_results(results, duration, "INTEGRATION")

        if success:
            print(f"\n✅ INTEGRATION TESTS PASSED - READY FOR PRODUCTION")

    if not success:
        print(f"\n🔧 TESTS NEED ATTENTION")
        print(f"❌ Fix failing tests before proceeding")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)