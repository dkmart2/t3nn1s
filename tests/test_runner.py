#!/usr/bin/env python3
"""
Runner for comprehensive data pipeline tests
"""

import subprocess
import sys
import os


def run_tests():
    """Run comprehensive test suite with detailed reporting"""

    print("🎾 COMPREHENSIVE TENNIS DATA PIPELINE TEST SUITE")
    print("=" * 60)

    # Test command with detailed output
    cmd = [
        sys.executable, "-m", "pytest",
        "comp_data_pipeline_test.py",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--capture=no",  # Show print statements
        "--durations=10",  # Show 10 slowest tests
        "-x",  # Stop on first failure
    ]

    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

        if result.returncode == 0:
            print("\n✅ ALL TESTS PASSED - PIPELINE IS READY FOR PRODUCTION")
            print("\nYou can now proceed to model training with confidence.")
            return True
        else:
            print("\n❌ TESTS FAILED - PIPELINE NEEDS FIXES")
            print("\nFix the failing tests before proceeding to model training.")
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