#!/usr/bin/env python3
"""
Quick validation script to ensure pipeline is ready for full run
Tests critical components without running the full 120-hour process
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check environment setup"""
    logger.info("\n🔍 CHECKING ENVIRONMENT")
    logger.info("=" * 50)
    
    checks = []
    
    # Check data directories
    data_dirs = {
        'TENNIS_DATA_DIR': os.environ.get('TENNIS_DATA_DIR', '~/Desktop/data'),
        'Jeff Data': os.path.expanduser('~/Desktop/data/Jeff 6.14.25'),
        'Cache Dir': os.path.expanduser('~/Desktop/data/cache')
    }
    
    for name, path in data_dirs.items():
        path = os.path.expanduser(path)
        exists = os.path.exists(path)
        checks.append((name, exists))
        status = "✓" if exists else "✗"
        logger.info(f"{status} {name}: {path}")
    
    # Check Jeff's point sequence files
    jeff_base = os.path.expanduser('~/Desktop/data/Jeff 6.14.25')
    point_files = [
        'men/charting-m-points-2020s.csv',
        'men/charting-m-points-2010s.csv',
        'women/charting-w-points-2020s.csv'
    ]
    
    logger.info("\n📊 Jeff's Point Sequence Files:")
    for file in point_files:
        full_path = os.path.join(jeff_base, file)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / 1024 / 1024
            logger.info(f"  ✓ {file} ({size_mb:.1f} MB)")
            
            # Check for 1st/2nd columns
            try:
                df_sample = pd.read_csv(full_path, nrows=5)
                if '1st' in df_sample.columns and '2nd' in df_sample.columns:
                    logger.info(f"    ✓ Contains 1st/2nd serve columns")
                else:
                    logger.info(f"    ⚠️ Missing 1st/2nd serve columns")
            except Exception as e:
                logger.info(f"    ✗ Error reading: {e}")
        else:
            logger.info(f"  ✗ {file} NOT FOUND")
            checks.append((file, False))
    
    # Check API key
    api_key = os.environ.get('API_TENNIS_KEY')
    if api_key:
        logger.info(f"\n✓ API_TENNIS_KEY set ({len(api_key)} chars)")
    else:
        logger.info("\n⚠️ API_TENNIS_KEY not set in environment")
    
    return all(check[1] for check in checks)

def test_imports():
    """Test all required imports"""
    logger.info("\n📦 TESTING IMPORTS")
    logger.info("=" * 50)
    
    required_modules = [
        'tennis_updated',
        'model',
        'settings',
        'polars',
        'httpx',
        'requests_cache',
        'unidecode',
        'bs4'
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {module}")
        except ImportError as e:
            logger.info(f"  ✗ {module}: {e}")
            failed.append(module)
    
    if failed:
        logger.info(f"\n⚠️ Missing modules: {', '.join(failed)}")
        logger.info("Install with: pip install polars httpx requests-cache unidecode beautifulsoup4")
        return False
    
    return True

def validate_jeff_parsing():
    """Validate Jeff's point notation parsing"""
    logger.info("\n🎾 VALIDATING JEFF POINT PARSING")
    logger.info("=" * 50)
    
    # Check if parser exists
    try:
        from tennis_updated import parse_point
        logger.info("✗ parse_point function not found in tennis_updated.py")
        logger.info("  Jeff's point sequences will not be utilized!")
        return False
    except ImportError:
        logger.info("⚠️ No parse_point function found")
        logger.info("  Per CLAUDE.md, this is highest priority for 20% improvement")
        
        # Check if points are at least being loaded
        try:
            from tennis_updated import load_jeff_comprehensive_data
            jeff_data = load_jeff_comprehensive_data()
            
            if 'men' in jeff_data and 'points_2020s' in jeff_data['men']:
                points_df = jeff_data['men']['points_2020s']
                logger.info(f"\n✓ Points data loaded: {points_df.shape[0]} rows")
                
                if '1st' in points_df.columns:
                    sample = points_df['1st'].dropna().head(3)
                    logger.info(f"  Sample sequences: {sample.tolist()}")
                    logger.info("\n⚠️ Data available but NOT being parsed!")
                    logger.info("  Implement parse_shot_sequence() per CLAUDE.md")
            
        except Exception as e:
            logger.info(f"Error checking points: {e}")
    
    return True  # Don't block pipeline, just warn

def estimate_runtime():
    """Estimate full pipeline runtime"""
    logger.info("\n⏱️ RUNTIME ESTIMATION")
    logger.info("=" * 50)
    
    # Based on file sizes and complexity
    jeff_base = os.path.expanduser('~/Desktop/data/Jeff 6.14.25')
    total_size_mb = 0
    
    for root, dirs, files in os.walk(jeff_base):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                total_size_mb += os.path.getsize(path) / 1024 / 1024
    
    logger.info(f"Total Jeff data size: {total_size_mb:.0f} MB")
    
    # Rough estimates based on data size
    processing_rate_mb_per_hour = 10  # Conservative estimate
    estimated_hours = total_size_mb / processing_rate_mb_per_hour
    
    logger.info(f"Estimated processing time: {estimated_hours:.0f}-{estimated_hours*2:.0f} hours")
    logger.info(f"  (~{estimated_hours/24:.1f}-{estimated_hours*2/24:.1f} days)")
    
    logger.info("\n💡 RECOMMENDATIONS:")
    logger.info("  1. Use nohup or screen for long run")
    logger.info("  2. Monitor with: python tennis_pipeline_runner.py --monitor")
    logger.info("  3. Resume if interrupted: python tennis_pipeline_runner.py --resume")
    
    return True

def check_cache():
    """Check cache configuration"""
    logger.info("\n💾 CACHE CONFIGURATION")
    logger.info("=" * 50)
    
    from tennis_updated import CACHE_DIR, HD_PATH, JEFF_PATH, DEF_PATH
    
    cache_paths = {
        'Cache Directory': CACHE_DIR,
        'Historical Data': HD_PATH,
        'Jeff Data Cache': JEFF_PATH,
        'Defaults Cache': DEF_PATH
    }
    
    for name, path in cache_paths.items():
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        parent_exists = os.path.exists(parent)
        
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024 if os.path.isfile(path) else 0
            logger.info(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")
        elif parent_exists:
            logger.info(f"  ⚠️ {name}: Will be created at {path}")
        else:
            logger.info(f"  ✗ {name}: Parent directory doesn't exist: {parent}")
            
    return True

def main():
    logger.info("\n" + "=" * 60)
    logger.info("TENNIS PIPELINE VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_checks = []
    
    # Run all checks
    all_checks.append(("Environment", check_environment()))
    all_checks.append(("Imports", test_imports()))
    all_checks.append(("Cache", check_cache()))
    all_checks.append(("Jeff Parsing", validate_jeff_parsing()))
    all_checks.append(("Runtime", estimate_runtime()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in all_checks if result)
    total = len(all_checks)
    
    for check_name, result in all_checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name}: {status}")
    
    logger.info(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n✅ READY FOR FULL PIPELINE RUN")
        logger.info("\nTo start full run:")
        logger.info("  python tennis_pipeline_runner.py")
        logger.info("\nOr for background run:")
        logger.info("  nohup python tennis_pipeline_runner.py > pipeline.log 2>&1 &")
    else:
        logger.info("\n⚠️ ISSUES FOUND - Review before full run")
        logger.info("\nCritical issue: Jeff's point sequences not being parsed")
        logger.info("This is leaving 20%+ improvement on the table per CLAUDE.md")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())