#!/usr/bin/env python3
"""
Runner for simplified comprehensive data pipeline tests
"""

import subprocess
import sys
import os


def run_tests():
    """Run comprehensive test suite with detailed reporting"""

    print("🎾 TENNIS DATA PIPELINE VALIDATION SUITE")
    print("=" * 55)
    print("Testing core pipeline functionality and data quality")
    print()

    # Test command with detailed output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/simplified_pipeline_tests.py",  # Updated filename
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--capture=no",  # Show print statements
        "--durations=0",  # Show all test durations
        "-x",  # Stop on first failure
    ]

    try:
        print("Running pipeline component tests...")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

        if result.returncode == 0:
            print("\n" + "=" * 55)
            print("✅ ALL CORE TESTS PASSED")
            print("✅ Pipeline components are working correctly")
            print("✅ Data quality validation passed")
            print("✅ Cache operations working")
            print("✅ Feature extraction validated")
            print("✅ Ready for model training")
            print("=" * 55)
            return True
        else:
            print("\n" + "=" * 55)
            print("❌ SOME TESTS FAILED")
            print("❌ Fix failing components before proceeding")
            print("=" * 55)
            return False

    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)