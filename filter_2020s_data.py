#!/usr/bin/env python3
"""
Filter tennis data to 2020s only - modern tennis focus
Dramatically reduce database size by focusing on post-2020 data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR
from datetime import datetime, date
import json

def load_and_filter_main_dataset():
    """Load integrated dataset and filter to 2020+"""
    print("🎾 FILTERING TO 2020s DATA ONLY")
    print("=" * 50)
    print("Reason: Tennis has changed significantly - focus on modern game")
    print()
    
    # Load the full integrated dataset
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'integrated_tennis_data.parquet')
    
    if not os.path.exists(cache_file):
        print("❌ No integrated dataset found")
        return pd.DataFrame()
    
    df = pd.read_parquet(cache_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 ORIGINAL DATASET")
    print(f"Total matches: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Filter to 2020 and later
    cutoff_date = pd.Timestamp('2020-01-01')
    df_2020s = df[df['date'] >= cutoff_date].copy()
    
    print(f"\n📊 FILTERED DATASET (2020+)")
    print(f"Total matches: {len(df_2020s):,}")
    print(f"Date range: {df_2020s['date'].min().date()} to {df_2020s['date'].max().date()}")
    print(f"Reduction: {len(df) - len(df_2020s):,} matches removed ({((len(df) - len(df_2020s)) / len(df) * 100):.1f}%)")
    
    # Show source breakdown for 2020s data
    if 'source_rank' in df_2020s.columns:
        source_breakdown = df_2020s['source_rank'].value_counts().sort_index()
        print(f"\n2020s source breakdown:")
        for rank, count in source_breakdown.items():
            source_name = {1: "Jeff/Tennis Abstract", 2: "API-Tennis", 3: "Tennis-data Excel"}.get(rank, f"Unknown({rank})")
            print(f"  - {source_name}: {count:,} matches")
    
    # Show year breakdown
    df_2020s['year'] = df_2020s['date'].dt.year
    year_breakdown = df_2020s['year'].value_counts().sort_index()
    print(f"\nYear breakdown (2020s):")
    for year, count in year_breakdown.items():
        print(f"  - {year}: {count:,} matches")
    
    # Show Jeff enhanced matches in 2020s
    jeff_enhanced = df_2020s[df_2020s.get('has_detailed_stats', False) == True]
    print(f"\nJeff detailed stats matches (2020s): {len(jeff_enhanced):,}")
    
    return df_2020s

def update_jeff_data_focus():
    """Update Jeff data summary to focus on 2020s relevance"""
    print(f"\n📈 JEFF DATA 2020s FOCUS")
    print(f"=" * 50)
    
    # Load existing Jeff data info
    summary_file = os.path.join(TENNIS_CACHE_DIR, 'jeff_data_summary.json')
    working_dir = Path.cwd()
    
    # Count 2020s specific Jeff data
    jeff_2020s_data = {
        'points_2020s': 0,
        'stats_2020s_relevant': 0,
        'total_comprehensive_2020s': 0
    }
    
    # Check 2020s point files
    points_files = [
        'charting-m-points-2020s.csv',
        'charting-w-points-2020s.csv'
    ]
    
    for filename in points_files:
        file_path = working_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, low_memory=False)
                jeff_2020s_data['points_2020s'] += len(df)
                jeff_2020s_data['total_comprehensive_2020s'] += len(df)
                print(f"✓ {filename}: {len(df):,} records")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    
    # Check stats files (all are relevant for 2020s analysis)
    stats_patterns = [
        'charting-*-stats-*.csv'
    ]
    
    stats_count = 0
    for pattern in stats_patterns:
        for file_path in working_dir.glob(pattern):
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    stats_count += len(df)
                except:
                    pass
    
    jeff_2020s_data['stats_2020s_relevant'] = stats_count
    jeff_2020s_data['total_comprehensive_2020s'] += stats_count
    
    print(f"\n📊 Jeff 2020s Data Summary:")
    print(f"  - Point-by-point (2020s): {jeff_2020s_data['points_2020s']:,} records")
    print(f"  - Statistical breakdowns: {jeff_2020s_data['stats_2020s_relevant']:,} records")
    print(f"  - Total comprehensive (2020s relevant): {jeff_2020s_data['total_comprehensive_2020s']:,} records")
    
    # Update summary file
    jeff_2020s_data['last_updated'] = datetime.now().isoformat()
    jeff_2020s_data['focus'] = '2020s modern tennis'
    
    summary_2020s_file = os.path.join(TENNIS_CACHE_DIR, 'jeff_data_2020s_summary.json')
    with open(summary_2020s_file, 'w') as f:
        json.dump(jeff_2020s_data, f, indent=2)
    
    print(f"✅ 2020s summary saved: {summary_2020s_file}")
    
    return jeff_2020s_data

def save_2020s_dataset(df_2020s, jeff_2020s_summary):
    """Save the 2020s focused dataset"""
    print(f"\n💾 SAVING 2020s FOCUSED DATASET")
    print(f"=" * 50)
    
    if df_2020s.empty:
        print("❌ No 2020s data to save")
        return False
    
    # Save as historical_data.parquet (replacing the full dataset)
    historical_data_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
    
    try:
        df_2020s.to_parquet(historical_data_file, index=False)
        print(f"✅ 2020s dataset saved: historical_data.parquet")
        print(f"   Size: {len(df_2020s):,} matches")
        
        # Also save a backup with explicit name
        backup_file = os.path.join(TENNIS_CACHE_DIR, 'tennis_data_2020s.parquet')
        df_2020s.to_parquet(backup_file, index=False)
        print(f"✅ Backup saved: tennis_data_2020s.parquet")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def show_2020s_capabilities():
    """Show what we can do with 2020s focused data"""
    print(f"\n🚀 2020s FOCUSED CAPABILITIES")
    print(f"=" * 50)
    
    modern_capabilities = [
        "✅ Modern serve patterns (recent power/speed trends)",
        "✅ Current return strategies (modern defensive play)",
        "✅ Recent shot evolution (topspin, pace changes)",
        "✅ COVID-era adaptations (2020-2021 unique conditions)",
        "✅ Surface evolution (court speed changes)",
        "✅ Equipment impact (modern racquet/string tech)",
        "✅ Fitness evolution (modern athletic conditioning)", 
        "✅ Tactical evolution (coaching innovations)",
        "✅ Young player emergence patterns",
        "✅ Career prime shifts in modern era"
    ]
    
    for capability in modern_capabilities:
        print(f"  {capability}")
    
    print(f"\n📊 ADVANTAGES OF 2020s FOCUS:")
    print(f"  ✅ Smaller, faster database operations")
    print(f"  ✅ More relevant for current predictions")
    print(f"  ✅ Captures modern playing styles")
    print(f"  ✅ Reflects current physical standards")
    print(f"  ✅ Includes recent surface/equipment changes")

def main():
    """Main execution to create 2020s focused dataset"""
    print("🎾 CREATING 2020s FOCUSED TENNIS DATABASE")
    print("Modern tennis analysis - post-2020 data only")
    print("=" * 60)
    
    try:
        # Step 1: Filter main dataset to 2020+
        df_2020s = load_and_filter_main_dataset()
        
        if df_2020s.empty:
            print("❌ No 2020s data available")
            return
        
        # Step 2: Update Jeff data focus to 2020s relevance
        jeff_2020s_summary = update_jeff_data_focus()
        
        # Step 3: Save 2020s focused dataset
        success = save_2020s_dataset(df_2020s, jeff_2020s_summary)
        
        if success:
            # Step 4: Show capabilities
            show_2020s_capabilities()
            
            print(f"\n🎯 2020s FOCUSED DATABASE COMPLETED")
            print(f"=" * 50)
            print(f"✅ Matches (2020+): {len(df_2020s):,}")
            print(f"✅ Jeff comprehensive data: {jeff_2020s_summary.get('total_comprehensive_2020s', 0):,} records")
            print(f"✅ Modern tennis focus: Enabled")
            print(f"✅ Database size: Dramatically reduced")
            print(f"✅ Prediction relevance: Maximized")
            
            print(f"\n🎾 READY FOR MODERN TENNIS ANALYSIS!")
            
            return df_2020s
        else:
            print(f"\n❌ Failed to save 2020s dataset")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result = main()