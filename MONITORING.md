# Pipeline Monitoring Guide

## Current Status
✅ **Pipeline is running!**
- Started: 2025-08-08 22:54
- PID: 5070
- Current stage: Processing historical data (Step 5: Extracting Jeff features)
- CPU usage: ~91%
- Memory: ~1GB

## How to Monitor

### 1. Check if still running
```bash
ps aux | grep tennis_pipeline_runner
```

### 2. View latest log entries
```bash
tail -100 pipeline.log
```

### 3. Watch log in real-time
```bash
tail -f pipeline.log
```
Press Ctrl+C to stop watching

### 4. Check progress
```bash
python tennis_pipeline_runner.py --monitor
```

### 5. Check disk usage
```bash
du -sh ~/Desktop/data/cache/
```

## Expected Timeline
Based on current processing:
- Step 1: Loading Jeff data ✅ (complete)
- Step 2: Calculating defaults ✅ (complete)
- Step 3: Loading match data ✅ (complete)
- Step 4: Processing tennis data ✅ (complete)
- Step 5: Extracting Jeff features 🔄 (in progress)
- Step 6: API integration (pending)
- Step 7: Saving results (pending)

**Estimated completion**: 4-6 days from start

## What's Happening Now
The pipeline is extracting features from Jeff's comprehensive tennis statistics. This involves:
- Processing 25,451 matches
- Extracting ~87 features per player
- Building comprehensive player profiles
- Processing point-by-point sequences

## If Pipeline Stops
If the process stops for any reason:
```bash
# Resume from checkpoint
python tennis_pipeline_runner.py --resume
```

## Warning Signs
- Memory usage > 16GB
- Disk full errors
- Process not in ps aux output
- No new log entries for > 1 hour

## Files Being Created
- `pipeline.log` - Main log file
- `~/Desktop/data/cache/historical_data.parquet` - Final output (when complete)
- `~/Desktop/data/cache/jeff_data.pkl` - Jeff data cache
- `~/Desktop/data/cache/weighted_defaults.pkl` - Default values

## Next Steps
1. Let it run for several days
2. Check periodically with monitoring commands
3. When complete, you'll see "PIPELINE COMPLETE" in the log
4. Then you can run your models with the processed data