#!/usr/bin/env python3
"""
Quick validation script to test fixes
"""
import subprocess
import sys
from pathlib import Path


def validate_test_syntax():
    """Check if test files have valid syntax"""
    test_files = [
        "test_tennis_pipeline.py",
        "test_pipeline_integration.py",
        "test_tennis_scraper.py"
    ]

    for test_file in test_files:
        if not Path(test_file).exists():
            return False, f"Missing test file: {test_file}"

        try:
            with open(test_file, "r") as f:
                compile(f.read(), test_file, "exec")
        except SyntaxError as e:
            return False, f"Syntax error in {test_file}: {e}"

    return True, "All test files have valid syntax"


def run_quick_test():
    """Run a single quick test to validate setup"""
    try:
        result = subprocess.run(
            "python -m pytest test_tennis_pipeline.py::test_parse_match_statistics -v",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return True, "Quick test passed"
        else:
            return False, f"Quick test failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Quick test timed out"
    except Exception as e:
        return False, f"Quick test error: {e}"


def main():
    """Main validation"""
    print("🔍 VALIDATING TEST SETUP")

    checks = [
        ("test file syntax", validate_test_syntax),
        ("quick test execution", run_quick_test)
    ]

    all_passed = True

    for check_name, check_func in checks:
        print(f"\nChecking {check_name}...", end=" ")
        try:
            success, message = check_func()
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                all_passed = False
        except Exception as e:
            print(f"❌ Error: {e}")
            all_passed = False

    print(f"\n{'=' * 50}")
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("Ready to run full test suite")
        return 0
    else:
        print("❌ VALIDATION FAILURES DETECTED")
        print("Fix issues before running full tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())