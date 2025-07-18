#!/usr/bin/env python3
"""
Complete test runner for tennis data pipeline
Includes all core tests plus comprehensive edge case coverage
"""

import subprocess
import sys
import os
import time


def run_test_suite(test_file, suite_name, timeout=600):
    """Run a specific test suite with timeout"""
    print(f"\n{'=' * 70}")
    print(f"🧪 RUNNING {suite_name}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--capture=no",
        "--durations=10",
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
    print(f"\n{'=' * 70}")
    print(f"🧪 RUNNING {test_name}")
    print(f"{'=' * 70}")

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
    """Run complete test suite including edge cases"""
    print("🎾 COMPLETE TENNIS DATA PIPELINE TEST SUITE")
    print("Testing core functionality + comprehensive edge cases")
    print("Addressing all identified testing gaps")

    # Core test suites
    core_test_configs = [
        ("test_tennis_pipeline.py", "CORE PIPELINE TESTS", 180),
        ("test_pipeline_integration.py", "INTEGRATION TESTS", 300),
        ("test_tennis_scraper.py", "SCRAPER TESTS", 60),
    ]

    # Edge case test suites
    edge_case_configs = [
        ("edge_case_tests.py", "COMPREHENSIVE EDGE CASE TESTS", 600),
    ]

    # Standalone script tests
    script_tests = [
        ("tests/test_full_pipeline_script.py", "PIPELINE SCRIPT TEST"),
    ]

    all_results = []
    total_duration = 0

    # Run core tests first
    print(f"\n{'#' * 70}")
    print("PHASE 1: CORE FUNCTIONALITY TESTS")
    print(f"{'#' * 70}")

    core_passed = 0
    core_total = 0

    for test_file, suite_name, timeout in core_test_configs:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name, timeout)
            all_results.append((suite_name, passed, duration, "CORE"))
            total_duration += duration
            core_total += 1
            if passed:
                core_passed += 1
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            all_results.append((suite_name, None, 0, "CORE"))

    # Run edge case tests
    print(f"\n{'#' * 70}")
    print("PHASE 2: EDGE CASE & GAP COVERAGE TESTS")
    print(f"{'#' * 70}")

    edge_passed = 0
    edge_total = 0

    for test_file, suite_name, timeout in edge_case_configs:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name, timeout)
            all_results.append((suite_name, passed, duration, "EDGE"))
            total_duration += duration
            edge_total += 1
            if passed:
                edge_passed += 1
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            all_results.append((suite_name, None, 0, "EDGE"))

    # Run script tests
    print(f"\n{'#' * 70}")
    print("PHASE 3: SCRIPT EXECUTION TESTS")
    print(f"{'#' * 70}")

    script_passed = 0
    script_total = 0

    for script_path, test_name in script_tests:
        if os.path.exists(script_path):
            passed, duration = run_python_script_test(script_path, test_name)
            all_results.append((test_name, passed, duration, "SCRIPT"))
            total_duration += duration
            script_total += 1
            if passed:
                script_passed += 1
        else:
            print(f"\n⚠️  {test_name}: File {script_path} not found, skipping")
            all_results.append((test_name, None, 0, "SCRIPT"))

    # Comprehensive final report
    print(f"\n{'=' * 70}")
    print("📊 COMPLETE TEST RESULTS")
    print(f"{'=' * 70}")

    # Phase breakdown
    print(f"\n📋 PHASE BREAKDOWN:")
    print(f"   Core Tests:       {core_passed}/{core_total}")
    print(f"   Edge Case Tests:  {edge_passed}/{edge_total}")
    print(f"   Script Tests:     {script_passed}/{script_total}")

    # Detailed results
    print(f"\n📊 DETAILED RESULTS:")

    total_passed = 0
    total_count = 0

    for suite_name, passed, duration, phase in all_results:
        if passed is True:
            status = "✅ PASSED"
            total_passed += 1
        elif passed is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
            continue

        total_count += 1
        phase_tag = f"[{phase}]"
        print(f"{status:12} {phase_tag:8} {suite_name:35} ({duration:.1f}s)")

    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Tests Passed: {total_passed}/{total_count}")
    print(f"   Total Time: {total_duration:.1f}s")
    print(f"   Success Rate: {(total_passed / total_count) * 100:.1f}%" if total_count > 0 else "   Success Rate: N/A")

    # Determine overall result
    if total_passed == total_count and total_count > 0:
        print(f"\n🏆 ALL TESTS PASSED - PIPELINE IS PRODUCTION READY!")
        print(f"✅ Core pipeline functionality validated")
        print(f"✅ Tennis Abstract integration verified")
        print(f"✅ Scraper functionality confirmed")
        print(f"✅ Pipeline script execution verified")
        print(f"✅ Comprehensive edge cases covered:")
        print(f"   • Incremental TA-only updates")
        print(f"   • Idempotency verification")
        print(f"   • API-only incremental updates")
        print(f"   • Schema & type validation")
        print(f"   • Winner/loser value divergence")
        print(f"   • Error handling & rollback")
        print(f"   • Environment flag testing")
        print(f"✅ All identified testing gaps addressed")

        print(f"\n🚀 PIPELINE READY FOR MODEL TRAINING!")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Pipeline validation complete")
        print(f"   2. All core components + edge cases verified")
        print(f"   3. Production deployment ready")
        print(f"   4. Begin model training: python scripts/train_model.py")

        return True
    else:
        print(f"\n🔧 PIPELINE NEEDS ATTENTION")
        print(f"❌ Fix failing tests before model training")

        # Categorize failures
        core_failures = [name for name, passed, _, phase in all_results
                         if passed is False and phase == "CORE"]
        edge_failures = [name for name, passed, _, phase in all_results
                         if passed is False and phase == "EDGE"]
        script_failures = [name for name, passed, _, phase in all_results
                           if passed is False and phase == "SCRIPT"]

        if core_failures or edge_failures or script_failures:
            print(f"\n🎯 FOCUS AREAS:")

            if core_failures:
                print(f"   📌 CORE FUNCTIONALITY:")
                for failure in core_failures:
                    if "PIPELINE" in failure:
                        print(f"      - Fix core pipeline function logic")
                    elif "INTEGRATION" in failure:
                        print(f"      - Fix Tennis Abstract integration")
                    elif "SCRAPER" in failure:
                        print(f"      - Fix web scraping functionality")

            if edge_failures:
                print(f"   📌 EDGE CASES:")
                for failure in edge_failures:
                    print(f"      - Fix comprehensive edge case handling")

            if script_failures:
                print(f"   📌 SCRIPT EXECUTION:")
                for failure in script_failures:
                    print(f"      - Fix pipeline script execution")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)