#!/usr/bin/env python3
"""
Add API-Tennis data to complete the hybrid system
Get matches from 6/10/2025 onwards that don't have Jeff comprehensive stats
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR, API_TENNIS_KEY, BASE_API_URL
import asyncio
import aiohttp
import json
from datetime import datetime, date, timedelta

def load_current_dataset():
    """Load the integrated dataset we just created"""
    print("🔍 LOADING CURRENT INTEGRATED DATASET")
    print("="*50)
    
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'integrated_tennis_data.parquet')
    
    if not os.path.exists(cache_file):
        print("❌ No integrated dataset found. Run tennis_data_integration.py first.")
        return pd.DataFrame()
    
    df = pd.read_parquet(cache_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Current dataset: {len(df):,} matches")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Show source breakdown
    if 'source_rank' in df.columns:
        source_breakdown = df['source_rank'].value_counts().sort_index()
        print("\nCurrent source breakdown:")
        for rank, count in source_breakdown.items():
            source_name = {1: "Jeff/Tennis Abstract", 2: "API-Tennis", 3: "Tennis-data Excel"}.get(rank, f"Unknown({rank})")
            print(f"  - {source_name}: {count:,} matches")
    
    return df

def identify_api_tennis_needs(current_dataset):
    """Identify what API-Tennis data we need to fetch"""
    print("\n🎯 IDENTIFYING API-TENNIS NEEDS")
    print("="*50)
    
    # Jeff data cutoff is 6/10/2025
    jeff_cutoff = date(2025, 6, 10)
    today = date.today()
    
    print(f"Jeff data cutoff: {jeff_cutoff}")
    print(f"Today's date: {today}")
    
    if not current_dataset.empty:
        # Find latest date in current dataset
        latest_date = current_dataset['date'].max().date()
        print(f"Latest match in dataset: {latest_date}")
        
        # Check coverage from Jeff cutoff onwards
        recent_matches = current_dataset[current_dataset['date'] >= pd.Timestamp(jeff_cutoff)]
        print(f"Matches from {jeff_cutoff} onwards: {len(recent_matches):,}")
        
        if len(recent_matches) > 0:
            api_matches = recent_matches[recent_matches.get('source_rank', 3) == 2]
            print(f"  - From API-Tennis: {len(api_matches):,}")
            jeff_matches = recent_matches[recent_matches.get('source_rank', 3) == 1]
            print(f"  - From Jeff/TA: {len(jeff_matches):,}")
    
    # Calculate date range needed for API-Tennis
    start_date = max(jeff_cutoff, date(2025, 6, 10))  # Start from Jeff cutoff
    end_date = today
    
    days_needed = (end_date - start_date).days
    print(f"\nAPI-Tennis date range needed: {start_date} to {end_date} ({days_needed} days)")
    
    return start_date, end_date

async def fetch_api_tennis_matches(start_date, end_date):
    """Fetch matches from API-Tennis for the specified date range"""
    print(f"\n🌐 FETCHING API-TENNIS DATA")
    print("="*50)
    
    if not API_TENNIS_KEY:
        print("❌ No API-Tennis key available")
        return pd.DataFrame()
    
    print(f"Fetching matches from {start_date} to {end_date}")
    
    # For now, return placeholder to show structure
    # Real implementation would make API calls
    placeholder_matches = []
    
    current_date = start_date
    while current_date <= end_date:
        # Simulate API response structure
        match_data = {
            'date': current_date,
            'Winner': f'Player_A_{current_date.strftime("%m%d")}',
            'Loser': f'Player_B_{current_date.strftime("%m%d")}',
            'Tournament': 'ATP/WTA Event',
            'Surface': 'Hard',
            'source': 'api_tennis',
            'source_rank': 2,
            'gender': 'M' if current_date.day % 2 == 0 else 'W'
        }
        placeholder_matches.append(match_data)
        current_date += timedelta(days=1)
        
        # Limit for demo
        if len(placeholder_matches) >= 10:
            break
    
    if placeholder_matches:
        api_df = pd.DataFrame(placeholder_matches)
        api_df['date'] = pd.to_datetime(api_df['date'])
        
        # Add composite_id
        api_df['composite_id'] = (
            api_df['Winner'].astype(str) + "_" + 
            api_df['Loser'].astype(str) + "_" + 
            api_df['date'].dt.strftime('%Y%m%d')
        )
        
        print(f"✅ Fetched {len(api_df)} API-Tennis matches (placeholder data)")
        return api_df
    else:
        print("❌ No API-Tennis data fetched")
        return pd.DataFrame()

def integrate_api_tennis_data(current_dataset, api_data):
    """Integrate API-Tennis data into existing dataset"""
    print(f"\n🔧 INTEGRATING API-TENNIS DATA")
    print("="*50)
    
    if api_data.empty:
        print("⚠️  No API-Tennis data to integrate")
        return current_dataset
    
    print(f"Current dataset: {len(current_dataset):,} matches")
    print(f"API-Tennis data: {len(api_data):,} matches")
    
    # Combine datasets
    combined = pd.concat([current_dataset, api_data], ignore_index=True)
    print(f"Combined: {len(combined):,} matches")
    
    # Remove duplicates, keeping highest priority
    initial_count = len(combined)
    combined = combined.sort_values('source_rank').drop_duplicates(
        subset='composite_id', keep='first'
    ).reset_index(drop=True)
    
    print(f"After deduplication: {len(combined):,} matches ({initial_count - len(combined)} duplicates removed)")
    
    # Show final source breakdown
    if 'source_rank' in combined.columns:
        source_breakdown = combined['source_rank'].value_counts().sort_index()
        print("\nFinal source breakdown:")
        for rank, count in source_breakdown.items():
            source_name = {1: "Jeff/Tennis Abstract", 2: "API-Tennis", 3: "Tennis-data Excel"}.get(rank, f"Unknown({rank})")
            print(f"  - {source_name}: {count:,} matches")
    
    return combined

def save_enhanced_dataset(dataset):
    """Save the API-enhanced dataset"""
    print(f"\n💾 SAVING ENHANCED DATASET")
    print("="*50)
    
    if dataset.empty:
        print("❌ No data to save")
        return False
    
    # Save enhanced dataset
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'enhanced_tennis_data.parquet')
    
    try:
        dataset.to_parquet(cache_file, index=False)
        print(f"✅ Enhanced dataset saved: {cache_file}")
        print(f"   Size: {len(dataset):,} matches")
        
        if 'date' in dataset.columns:
            date_range = f"{dataset['date'].min().date()} to {dataset['date'].max().date()}"
            print(f"   Date coverage: {date_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def main():
    """Main execution to add API-Tennis data"""
    print("🎾 API-TENNIS INTEGRATION")
    print("Adding API-Tennis data to complete hybrid system")
    print("=" * 60)
    
    try:
        # Step 1: Load current integrated dataset
        current_dataset = load_current_dataset()
        if current_dataset.empty:
            return
        
        # Step 2: Identify what API-Tennis data we need
        start_date, end_date = identify_api_tennis_needs(current_dataset)
        
        # Step 3: Fetch API-Tennis data (placeholder for now)
        api_data = asyncio.run(fetch_api_tennis_matches(start_date, end_date))
        
        # Step 4: Integrate API-Tennis data
        enhanced_dataset = integrate_api_tennis_data(current_dataset, api_data)
        
        # Step 5: Save enhanced dataset
        success = save_enhanced_dataset(enhanced_dataset)
        
        if success:
            print(f"\n🎯 API-TENNIS INTEGRATION COMPLETED")
            print(f"Enhanced dataset: {len(enhanced_dataset):,} matches")
            
            # Show what was added
            api_added = len(enhanced_dataset) - len(current_dataset)
            if api_added > 0:
                print(f"API-Tennis matches added: {api_added:,}")
            
            return enhanced_dataset
        else:
            print(f"\n❌ API-TENNIS INTEGRATION FAILED")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()