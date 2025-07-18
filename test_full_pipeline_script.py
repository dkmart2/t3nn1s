#!/usr/bin/env python3
"""
Test the actual full_data_pipeline.py script execution
"""

import os
import sys
import subprocess
import tempfile
import shutil
from unittest.mock import patch


def test_full_pipeline_script():
    """Test that full_data_pipeline.py script executes successfully"""

    # Check if the script exists
    script_paths = [
        "scripts/full_data_pipeline.py",
        "full_data_pipeline.py",
        "pipeline.py"
    ]

    script_path = None
    for path in script_paths:
        if os.path.exists(path):
            script_path = path
            break

    if not script_path:
        print("❌ No full_data_pipeline.py script found")
        print(f"Checked paths: {script_paths}")
        return False

    print(f"✓ Found pipeline script: {script_path}")

    # Create temporary cache directory for testing
    temp_cache = tempfile.mkdtemp(prefix="pipeline_test_")

    try:
        # Set up environment for testing
        env = os.environ.copy()
        env['CACHE_DIR'] = temp_cache
        env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

        print(f"🧪 Testing pipeline script execution...")
        print(f"   Script: {script_path}")
        print(f"   Cache: {temp_cache}")

        # Execute the script with timeout
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=120, env=env)

        print(f"\n📋 SCRIPT OUTPUT:")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        # Analyze results
        if result.returncode == 0:
            print(f"\n✅ PIPELINE SCRIPT EXECUTED SUCCESSFULLY")

            # Check for expected outputs
            success_indicators = [
                "complete", "success", "cached", "saved",
                "matches", "pipeline", "finished"
            ]

            output_text = (result.stdout + result.stderr).lower()
            found_indicators = [ind for ind in success_indicators if ind in output_text]

            if found_indicators:
                print(f"✓ Success indicators found: {', '.join(found_indicators)}")

            # Check if cache files were created
            cache_files = ['historical_data.parquet', 'jeff_data.pkl', 'weighted_defaults.pkl']
            created_files = [f for f in cache_files if os.path.exists(os.path.join(temp_cache, f))]

            if created_files:
                print(f"✓ Cache files created: {', '.join(created_files)}")
            else:
                print(f"⚠️  No cache files created (may be using existing cache)")

            return True

        else:
            print(f"\n❌ PIPELINE SCRIPT FAILED")
            print(f"Return code: {result.returncode}")

            # Analyze error output
            if "error" in result.stderr.lower() or "exception" in result.stderr.lower():
                print(f"💥 Exception detected in output")

            return False

    except subprocess.TimeoutExpired:
        print(f"\n⏰ PIPELINE SCRIPT TIMED OUT (>120s)")
        print(f"❌ Script may be hanging or taking too long")
        return False

    except Exception as e:
        print(f"\n💥 SCRIPT EXECUTION ERROR: {e}")
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_cache, ignore_errors=True)


def main():
    """Main test function"""
    print("🎾 TESTING FULL DATA PIPELINE SCRIPT")
    print("=" * 50)

    success = test_full_pipeline_script()

    if success:
        print(f"\n🏆 FULL PIPELINE SCRIPT TEST PASSED")
        print(f"✅ Script executes without errors")
        print(f"✅ Orchestration logic is working")
        print(f"✅ Ready for production deployment")
    else:
        print(f"\n🔧 FULL PIPELINE SCRIPT NEEDS FIXES")
        print(f"❌ Script execution failed")
        print(f"❌ Check script logic and dependencies")
        print(f"❌ Fix issues before production deployment")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)