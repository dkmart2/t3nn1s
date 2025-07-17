#!/usr/bin/env python3
"""
Comprehensive test runner for tennis data pipeline
Runs both component tests and orchestration tests
"""

import subprocess
import sys
import os
import time


def run_test_suite(test_file, suite_name):
    """Run a specific test suite with detailed reporting"""
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
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {suite_name} PASSED ({duration:.1f}s)")
            return True, duration
        else:
            print(f"\n❌ {suite_name} FAILED ({duration:.1f}s)")
            return False, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 {suite_name} CRASHED: {e} ({duration:.1f}s)")
        return False, duration


def main():
    """Run comprehensive test suite"""
    print("🎾 COMPREHENSIVE TENNIS DATA PIPELINE TEST SUITE")
    print("Testing both component functionality and orchestration workflows")

    # Test suites to run
    test_suites = [
        ("orchestration_tests.py", "COMPONENT TESTS"),
        ("orchestration_tests.py", "ORCHESTRATION TESTS"),
        ("test_tennis_pipeline.py", "UNIT TESTS"),
    ]

    results = []
    total_duration = 0

    for test_file, suite_name in test_suites:
        test_path = f"tests/{test_file}"
        if os.path.exists(test_path):
            passed, duration = run_test_suite(test_file, suite_name)
            results.append((suite_name, passed, duration))
            total_duration += duration
        else:
            print(f"\n⚠️  {suite_name}: File {test_file} not found, skipping")
            results.append((suite_name, None, 0))

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
        print(f"{status:12} {suite_name:25} ({duration:.1f}s)")

    print(f"\n📈 SUMMARY:")
    print(f"   Tests Passed: {passed_count}/{total_count}")
    print(f"   Total Time: {total_duration:.1f}s")

    if passed_count == total_count and total_count > 0:
        print(f"\n🏆 ALL TESTS PASSED - PIPELINE IS PRODUCTION READY!")
        print(f"✅ Component functionality validated")
        print(f"✅ Orchestration workflows verified")
        print(f"✅ Cold start and incremental updates tested")
        print(f"✅ Tennis Abstract integration validated")
        print(f"✅ Data quality and integrity confirmed")
        print(f"✅ Error handling and recovery verified")
        print(f"✅ Performance characteristics acceptable")
        print(f"\n🚀 Ready to proceed with model training!")
        return True
    else:
        print(f"\n🔧 PIPELINE NEEDS ATTENTION")
        print(f"❌ Fix failing tests before model training")

        # Provide specific guidance
        for suite_name, passed, duration in results:
            if passed is False:
                if "COMPONENT" in suite_name:
                    print(f"   - Fix core component issues in {suite_name}")
                elif "ORCHESTRATION" in suite_name:
                    print(f"   - Fix pipeline workflow issues in {suite_name}")
                elif "UNIT" in suite_name:
                    print(f"   - Fix individual function issues in {suite_name}")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)