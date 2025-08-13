#!/usr/bin/env python3
"""
Production runner for tennis data pipeline with checkpoint/resume capability
Designed for long-running jobs (120+ hours) with progress monitoring
"""

import os
import sys
import time
import pickle
import json
import psutil
import gc
import pandas as pd
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import logging
import signal
import atexit

# Import pipeline components
from tennis_updated import (
    generate_comprehensive_historical_data_optimized,
    integrate_api_tennis_data_incremental,
    save_to_cache,
    load_jeff_comprehensive_data,
    CACHE_DIR, HD_PATH, JEFF_PATH, DEF_PATH
)

# Checkpoint configuration
CHECKPOINT_DIR = Path(CACHE_DIR) / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = CHECKPOINT_DIR / "pipeline_state.pkl"
PROGRESS_FILE = CHECKPOINT_DIR / "pipeline_progress.json"
JEFF_CACHE_FILE = CHECKPOINT_DIR / "jeff_data_cache.pkl"

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, batch_size=1000, checkpoint_interval=300):
        """
        Args:
            batch_size: Number of matches to process per batch
            checkpoint_interval: Seconds between automatic checkpoints
        """
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.start_time = None
        self.state = {
            'stage': 'init',  # init, jeff_loading, processing, api_integration, complete
            'processed_matches': 0,
            'total_matches': 0,
            'batch_num': 0,
            'errors': [],
            'memory_peaks': []
        }
        self.hist_data = None
        self.jeff_data = None
        self.defaults = None
        self.last_checkpoint = time.time()
        
        # Register cleanup handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        
    def signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        logger.info(f"\n⚠️ Received signal {signum}, saving checkpoint...")
        self.save_checkpoint()
        logger.info("✓ Checkpoint saved. Exiting...")
        sys.exit(0)
        
    def cleanup(self):
        """Final cleanup on exit"""
        if self.hist_data is not None:
            self.save_checkpoint()
            
    def log_memory(self):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.state['memory_peaks'].append(memory_mb)
        return memory_mb
        
    def save_checkpoint(self):
        """Save current state for resume capability"""
        try:
            checkpoint = {
                'state': self.state,
                'timestamp': datetime.now().isoformat(),
                'hist_shape': self.hist_data.shape if self.hist_data is not None else None,
            }
            
            # Save state
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            # Save data if available
            if self.hist_data is not None:
                temp_path = CHECKPOINT_DIR / "hist_data_checkpoint.parquet"
                self.hist_data.to_parquet(temp_path)
                
            # Save progress JSON for monitoring
            self.save_progress()
            
            logger.info(f"✓ Checkpoint saved: {self.state['processed_matches']} matches processed")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def save_progress(self):
        """Save human-readable progress file"""
        try:
            if self.start_time:
                elapsed = time.time() - self.start_time
                rate = self.state['processed_matches'] / elapsed if elapsed > 0 else 0
                eta_seconds = (self.state['total_matches'] - self.state['processed_matches']) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=eta_seconds)
            else:
                elapsed = 0
                rate = 0
                eta = None
                
            progress = {
                'stage': self.state['stage'],
                'processed': self.state['processed_matches'],
                'total': self.state['total_matches'],
                'percent': (self.state['processed_matches'] / self.state['total_matches'] * 100) 
                          if self.state['total_matches'] > 0 else 0,
                'elapsed_hours': elapsed / 3600,
                'rate_per_hour': rate * 3600,
                'eta': eta.isoformat() if eta else None,
                'memory_mb': self.log_memory(),
                'errors': len(self.state['errors']),
                'last_update': datetime.now().isoformat()
            }
            
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            
    def load_checkpoint(self):
        """Load previous checkpoint if exists"""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.state = checkpoint['state']
                logger.info(f"✓ Loaded checkpoint from {checkpoint['timestamp']}")
                logger.info(f"  Stage: {self.state['stage']}")
                logger.info(f"  Processed: {self.state['processed_matches']} matches")
                
                # Load data if exists
                temp_path = CHECKPOINT_DIR / "hist_data_checkpoint.parquet"
                if temp_path.exists():
                    self.hist_data = pd.read_parquet(temp_path)
                    logger.info(f"  Loaded data: {self.hist_data.shape}")
                    
                return True
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False
        return False
        
    def load_jeff_cached(self):
        """Load Jeff data with caching to avoid repeated loading"""
        if JEFF_CACHE_FILE.exists():
            try:
                logger.info("Loading cached Jeff data...")
                with open(JEFF_CACHE_FILE, 'rb') as f:
                    self.jeff_data = pickle.load(f)
                logger.info("✓ Loaded Jeff data from cache")
                return True
            except Exception as e:
                logger.error(f"Failed to load cached Jeff data: {e}")
                
        # Load fresh
        logger.info("Loading Jeff data (this takes ~30 seconds)...")
        self.jeff_data = load_jeff_comprehensive_data()
        
        # Cache for next time
        try:
            with open(JEFF_CACHE_FILE, 'wb') as f:
                pickle.dump(self.jeff_data, f)
            logger.info("✓ Jeff data cached for future runs")
        except Exception as e:
            logger.warning(f"Failed to cache Jeff data: {e}")
            
        return self.jeff_data is not None
        
    def run_with_monitoring(self):
        """Run pipeline with progress monitoring"""
        self.start_time = time.time()
        
        # Check for existing checkpoint
        resumed = self.load_checkpoint()
        
        if not resumed or self.state['stage'] == 'init':
            logger.info("\n" + "="*60)
            logger.info("STARTING TENNIS DATA PIPELINE")
            logger.info("="*60)
            self.state['stage'] = 'jeff_loading'
        else:
            logger.info("\n" + "="*60)
            logger.info("RESUMING TENNIS DATA PIPELINE")
            logger.info("="*60)
            
        try:
            # Stage 1: Load Jeff data (cached)
            if self.state['stage'] == 'jeff_loading':
                logger.info("\n📚 Stage 1: Loading Jeff data...")
                if not self.load_jeff_cached():
                    raise Exception("Failed to load Jeff data")
                self.state['stage'] = 'processing'
                self.save_checkpoint()
                
            # Stage 2: Generate comprehensive data
            if self.state['stage'] == 'processing':
                logger.info("\n⚙️ Stage 2: Processing historical data...")
                logger.info("NOTE: This is the long-running step (120+ hours)")
                
                # Pass cached Jeff data to avoid reloading
                self.hist_data, _, self.defaults = generate_comprehensive_historical_data_optimized(
                    fast=False
                )
                
                if self.hist_data is not None:
                    self.state['total_matches'] = len(self.hist_data)
                    self.state['processed_matches'] = len(self.hist_data)
                    logger.info(f"✓ Processed {self.state['total_matches']} matches")
                    
                self.state['stage'] = 'api_integration'
                self.save_checkpoint()
                
            # Stage 3: API integration
            if self.state['stage'] == 'api_integration':
                logger.info("\n🌐 Stage 3: Integrating API Tennis data...")
                original_size = len(self.hist_data)
                
                self.hist_data = integrate_api_tennis_data_incremental(self.hist_data)
                
                new_matches = len(self.hist_data) - original_size
                logger.info(f"✓ Added {new_matches} matches from API")
                
                self.state['stage'] = 'saving'
                self.save_checkpoint()
                
            # Stage 4: Save final results
            if self.state['stage'] == 'saving':
                logger.info("\n💾 Stage 4: Saving final dataset...")
                
                save_to_cache(self.hist_data, self.jeff_data, self.defaults)
                
                logger.info(f"✓ Final dataset saved: {self.hist_data.shape}")
                logger.info(f"  Historical data: {HD_PATH}")
                logger.info(f"  Jeff data: {JEFF_PATH}")
                logger.info(f"  Defaults: {DEF_PATH}")
                
                self.state['stage'] = 'complete'
                self.save_checkpoint()
                
            # Complete
            elapsed = time.time() - self.start_time
            hours = elapsed / 3600
            
            logger.info("\n" + "="*60)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("="*60)
            logger.info(f"Total time: {hours:.1f} hours")
            logger.info(f"Final dataset: {self.hist_data.shape}")
            logger.info(f"Jeff features: {len([c for c in self.hist_data.columns if 'jeff_' in c])}")
            logger.info(f"Peak memory: {max(self.state['memory_peaks']):.0f} MB")
            
            if self.state['errors']:
                logger.warning(f"Completed with {len(self.state['errors'])} errors")
                for error in self.state['errors'][-5:]:  # Show last 5 errors
                    logger.warning(f"  - {error}")
                    
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            self.state['errors'].append(str(e))
            self.save_checkpoint()
            raise
            
    def monitor_progress(self):
        """Print current progress (call from separate terminal)"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                
            print("\n" + "="*60)
            print("PIPELINE PROGRESS")
            print("="*60)
            print(f"Stage: {progress['stage']}")
            print(f"Progress: {progress['processed']}/{progress['total']} ({progress['percent']:.1f}%)")
            print(f"Elapsed: {progress['elapsed_hours']:.1f} hours")
            print(f"Rate: {progress['rate_per_hour']:.0f} matches/hour")
            print(f"Memory: {progress['memory_mb']:.0f} MB")
            
            if progress['eta']:
                eta = datetime.fromisoformat(progress['eta'])
                print(f"ETA: {eta.strftime('%Y-%m-%d %H:%M')}")
                
            print(f"Last update: {progress['last_update']}")
        else:
            print("No progress file found. Pipeline may not be running.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tennis data pipeline with monitoring')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--monitor', action='store_true',
                       help='Show current progress and exit')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    runner = PipelineRunner(batch_size=args.batch_size)
    
    if args.monitor:
        runner.monitor_progress()
    else:
        runner.run_with_monitoring()


if __name__ == "__main__":
    main()