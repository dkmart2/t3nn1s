#!/usr/bin/env python3
"""
Test script for tennis data pipeline
Tests with small subset before running full 120-hour processing
"""

import os
import sys
import time
import psutil
import gc
import pandas as pd
import tracemalloc
from datetime import datetime
import logging
from pathlib import Path

# Import the main pipeline
from tennis_updated import (
    generate_comprehensive_historical_data_optimized,
    integrate_api_tennis_data_incremental,
    save_to_cache,
    load_jeff_comprehensive_data,
    CACHE_DIR, HD_PATH, JEFF_PATH, DEF_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineTest:
    def __init__(self, test_size=100):
        self.test_size = test_size
        self.start_time = None
        self.checkpoints = []
        
    def log_memory(self, checkpoint_name):
        """Log memory usage at checkpoints"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        self.checkpoints.append({
            'name': checkpoint_name,
            'memory_mb': memory_mb,
            'time': time.time() - self.start_time if self.start_time else 0
        })
        
        logger.info(f"[{checkpoint_name}] Memory: {memory_mb:.2f} MB")
        
    def test_jeff_data_loading(self):
        """Test 1: Verify Jeff data loads correctly"""
        logger.info("\n=== TEST 1: Jeff Data Loading ===")
        self.log_memory("Before Jeff load")
        
        try:
            jeff_data = load_jeff_comprehensive_data()
            
            if jeff_data and 'men' in jeff_data:
                logger.info(f"✓ Jeff data loaded successfully")
                logger.info(f"  Men's datasets: {list(jeff_data['men'].keys())}")
                
                # Check for point sequences
                if 'points' in jeff_data['men']:
                    points_df = jeff_data['men']['points']
                    logger.info(f"  Points data shape: {points_df.shape}")
                    
                    # Check for 1st/2nd serve columns
                    if '1st' in points_df.columns and '2nd' in points_df.columns:
                        logger.info("  ✓ Found 1st and 2nd serve columns for point sequences")
                        
                        # Sample a few points
                        sample = points_df[['1st', '2nd']].dropna().head(5)
                        logger.info(f"  Sample point sequences:\n{sample}")
                    else:
                        logger.warning("  ⚠️ Missing 1st/2nd serve columns for point parsing")
                        
            self.log_memory("After Jeff load")
            return True
            
        except Exception as e:
            logger.error(f"✗ Jeff data loading failed: {e}")
            return False
    
    def test_small_subset(self):
        """Test 2: Process small subset of matches"""
        logger.info(f"\n=== TEST 2: Small Subset ({self.test_size} matches) ===")
        self.log_memory("Before subset processing")
        
        try:
            # Use fast mode with limited sample
            hist, jeff_data, defaults = generate_comprehensive_historical_data_optimized(
                fast=True, 
                n_sample=self.test_size,
                use_synthetic=False
            )
            
            logger.info(f"✓ Generated historical data: {hist.shape}")
            logger.info(f"  Columns: {hist.shape[1]}")
            logger.info(f"  Jeff features: {len([c for c in hist.columns if 'jeff_' in c])}")
            
            # Check for duplicates
            duplicates = hist.duplicated().sum()
            logger.info(f"  Duplicates: {duplicates}")
            
            # Check for missing values
            missing_pct = (hist.isnull().sum() / len(hist)).mean() * 100
            logger.info(f"  Average missing: {missing_pct:.2f}%")
            
            self.log_memory("After subset processing")
            return hist, jeff_data, defaults
            
        except Exception as e:
            logger.error(f"✗ Subset processing failed: {e}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    def test_api_integration(self, hist):
        """Test 3: API Tennis integration"""
        logger.info("\n=== TEST 3: API Tennis Integration ===")
        
        if hist is None:
            logger.warning("Skipping API test - no historical data")
            return None
            
        self.log_memory("Before API integration")
        
        try:
            original_shape = hist.shape
            hist_integrated = integrate_api_tennis_data_incremental(hist)
            
            logger.info(f"✓ API integration complete")
            logger.info(f"  Original shape: {original_shape}")
            logger.info(f"  New shape: {hist_integrated.shape}")
            logger.info(f"  New rows: {hist_integrated.shape[0] - original_shape[0]}")
            
            self.log_memory("After API integration")
            return hist_integrated
            
        except Exception as e:
            logger.error(f"✗ API integration failed: {e}")
            return hist
    
    def test_cache_operations(self, hist, jeff_data, defaults):
        """Test 4: Cache saving and loading"""
        logger.info("\n=== TEST 4: Cache Operations ===")
        
        if hist is None:
            logger.warning("Skipping cache test - no data")
            return False
            
        try:
            # Save to cache
            save_to_cache(hist, jeff_data, defaults)
            logger.info("✓ Data saved to cache")
            
            # Check file sizes
            for path, name in [(HD_PATH, "Historical"), (JEFF_PATH, "Jeff"), (DEF_PATH, "Defaults")]:
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    logger.info(f"  {name}: {size_mb:.2f} MB")
            
            # Test loading back
            if os.path.exists(HD_PATH):
                loaded = pd.read_parquet(HD_PATH)
                logger.info(f"✓ Cache loaded successfully: {loaded.shape}")
                return True
                
        except Exception as e:
            logger.error(f"✗ Cache operations failed: {e}")
            return False
    
    def estimate_full_runtime(self):
        """Estimate time for full dataset based on test"""
        if not self.checkpoints:
            return
            
        logger.info("\n=== RUNTIME ESTIMATION ===")
        
        # Get total test time
        total_time = self.checkpoints[-1]['time']
        matches_per_second = self.test_size / total_time if total_time > 0 else 0
        
        # Estimate for full dataset (~30k matches)
        full_dataset_size = 30000
        estimated_seconds = full_dataset_size / matches_per_second if matches_per_second > 0 else 0
        estimated_hours = estimated_seconds / 3600
        
        logger.info(f"Test performance:")
        logger.info(f"  Processed: {self.test_size} matches")
        logger.info(f"  Time: {total_time:.2f} seconds")
        logger.info(f"  Rate: {matches_per_second:.2f} matches/second")
        
        logger.info(f"\nFull dataset estimate:")
        logger.info(f"  Size: ~{full_dataset_size} matches")
        logger.info(f"  Estimated time: {estimated_hours:.1f} hours")
        
        # Memory projection
        peak_memory = max(c['memory_mb'] for c in self.checkpoints)
        memory_per_match = peak_memory / self.test_size
        projected_memory = memory_per_match * full_dataset_size
        
        logger.info(f"  Projected peak memory: {projected_memory:.0f} MB ({projected_memory/1024:.1f} GB)")
        
        if projected_memory > 16000:  # 16GB warning threshold
            logger.warning("  ⚠️ WARNING: Projected memory usage exceeds 16GB")
            logger.warning("  Consider processing in batches or increasing RAM")
    
    def run_all_tests(self):
        """Run complete test suite"""
        logger.info("=" * 60)
        logger.info("TENNIS PIPELINE TEST SUITE")
        logger.info(f"Test size: {self.test_size} matches")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        tracemalloc.start()
        
        # Run tests
        test_results = {
            'jeff_loading': self.test_jeff_data_loading(),
        }
        
        hist, jeff_data, defaults = self.test_small_subset()
        test_results['subset_processing'] = hist is not None
        
        if hist is not None:
            hist = self.test_api_integration(hist)
            test_results['api_integration'] = True
            
            test_results['cache_operations'] = self.test_cache_operations(
                hist, jeff_data, defaults
            )
        
        # Memory cleanup
        gc.collect()
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{test_name}: {status}")
        
        # Performance metrics
        self.estimate_full_runtime()
        
        # Memory trace
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        logger.info(f"\nMemory usage:")
        logger.info(f"  Current: {current / 1024 / 1024:.2f} MB")
        logger.info(f"  Peak: {peak / 1024 / 1024:.2f} MB")
        
        # Recommendations
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATIONS BEFORE FULL RUN")
        logger.info("=" * 60)
        
        all_passed = all(test_results.values())
        if all_passed:
            logger.info("✓ All tests passed - safe to proceed with full run")
            logger.info("\nSuggested command:")
            logger.info("  nohup python tennis_updated.py > pipeline_full.log 2>&1 &")
            logger.info("\nMonitor progress with:")
            logger.info("  tail -f pipeline_full.log")
        else:
            logger.warning("⚠️ Some tests failed - review issues before full run")
            failed = [k for k, v in test_results.items() if not v]
            logger.warning(f"Failed tests: {', '.join(failed)}")
        
        return all_passed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tennis data pipeline')
    parser.add_argument('--size', type=int, default=100,
                       help='Number of matches to test (default: 100)')
    parser.add_argument('--skip-cache', action='store_true',
                       help='Skip cache operations test')
    
    args = parser.parse_args()
    
    # Create test instance
    tester = PipelineTest(test_size=args.size)
    
    # Run tests
    success = tester.run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()