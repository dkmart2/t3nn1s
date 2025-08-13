#!/usr/bin/env python3
"""
Minimal test to check Jeff match data persistence
"""
import pandas as pd
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR

def load_jeff_match_records_simple():
    """Simple version of Jeff match loading"""
    jeff_matches = []
    working_dir = Path.cwd()
    
    print("=== LOADING JEFF MATCH RECORDS (SIMPLE) ===")
    
    # Load men's matches
    men_match_file = working_dir / "charting-m-matches.csv"
    if men_match_file.exists():
        print(f"Loading men's matches from: {men_match_file}")
        df_men = pd.read_csv(men_match_file)
        df_men['gender'] = 'M'
        jeff_matches.append(df_men)
        print(f"✓ Loaded {len(df_men)} men's matches")
    else:
        print(f"❌ Men's match file not found: {men_match_file}")
    
    # Load women's matches
    women_match_file = working_dir / "charting-w-matches.csv"  
    if women_match_file.exists():
        print(f"Loading women's matches from: {women_match_file}")
        df_women = pd.read_csv(women_match_file)
        df_women['gender'] = 'W'
        jeff_matches.append(df_women)
        print(f"✓ Loaded {len(df_women)} women's matches")
    else:
        print(f"❌ Women's match file not found: {women_match_file}")
    
    if not jeff_matches:
        print("❌ No Jeff match files found!")
        return pd.DataFrame()
    
    # Combine the dataframes
    combined_jeff = pd.concat(jeff_matches, ignore_index=True)
    print(f"✓ Combined Jeff matches: {len(combined_jeff)}")
    
    # Ensure date column is datetime
    if 'Date' in combined_jeff.columns:
        print(f"✓ Found Date column, converting to datetime...")
        # Handle invalid dates like "RR" by coercing to NaT
        combined_jeff['date'] = pd.to_datetime(combined_jeff['Date'], errors='coerce')
        
        # Check for any invalid dates
        invalid_dates = combined_jeff['date'].isna()
        if invalid_dates.sum() > 0:
            print(f"⚠️  Found {invalid_dates.sum()} invalid dates, removing them...")
            print("Sample invalid date entries:")
            print(combined_jeff[invalid_dates]['Date'].head().tolist())
            combined_jeff = combined_jeff[~invalid_dates]
        
        combined_jeff = combined_jeff.drop('Date', axis=1)
    elif 'date' not in combined_jeff.columns:
        print("❌ No Date/date column found!")
        print(f"Available columns: {list(combined_jeff.columns)}")
        return pd.DataFrame()
    
    # Set source_rank=1 for Jeff/Tennis Abstract data
    combined_jeff['source_rank'] = 1
    
    # Map Player 1/Player 2 to Winner/Loser (WARNING: assumes Player 1 wins - this is incorrect!)
    # This is just for testing persistence - real implementation needs proper winner determination
    if 'Player 1' in combined_jeff.columns and 'Player 2' in combined_jeff.columns:
        combined_jeff['Winner'] = combined_jeff['Player 1']
        combined_jeff['Loser'] = combined_jeff['Player 2']
        print("⚠️  WARNING: Incorrectly assuming Player 1 always wins (for test only)")
    
    # Generate composite_id to prevent dropna filtering
    if 'Winner' in combined_jeff.columns and 'Loser' in combined_jeff.columns:
        combined_jeff['composite_id'] = (
            combined_jeff['Winner'].astype(str) + "_" + 
            combined_jeff['Loser'].astype(str) + "_" + 
            combined_jeff['date'].dt.strftime('%Y%m%d')
        )
    
    print(f"✓ Jeff match records processed with source_rank=1")
    return combined_jeff

def test_direct_cache_save():
    """Test direct saving to cache without full pipeline"""
    
    # Load Jeff matches
    jeff_data = load_jeff_match_records_simple()
    if jeff_data.empty:
        return
    
    print(f"\n=== TESTING DIRECT CACHE SAVE ===")
    print(f"Jeff data shape: {jeff_data.shape}")
    print(f"Source ranks: {jeff_data['source_rank'].value_counts().to_dict()}")
    print(f"Date range: {jeff_data['date'].min()} to {jeff_data['date'].max()}")
    
    # Save directly to cache
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'test_jeff_only.parquet')
    jeff_data.to_parquet(cache_file, index=False)
    print(f"✓ Saved Jeff data to: {cache_file}")
    
    # Reload and verify
    reloaded = pd.read_parquet(cache_file)
    print(f"✓ Reloaded data shape: {reloaded.shape}")
    print(f"✓ Reloaded source ranks: {reloaded['source_rank'].value_counts().to_dict()}")
    
    # Check if any records were lost
    if len(reloaded) == len(jeff_data):
        print("✅ SUCCESS: All Jeff records persisted!")
    else:
        print(f"❌ FAILURE: Lost {len(jeff_data) - len(reloaded)} records during save/load")
    
    return reloaded

def check_current_cache():
    """Check what's currently in the main cache"""
    print(f"\n=== CHECKING CURRENT CACHE ===")
    
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
    if not os.path.exists(cache_file):
        print("❌ No historical_data.parquet found in cache")
        return
    
    df = pd.read_parquet(cache_file)
    print(f"✓ Main cache size: {len(df)} matches")
    
    if 'source_rank' in df.columns:
        source_breakdown = df['source_rank'].value_counts().sort_index()
        print("Source rank breakdown:")
        for rank, count in source_breakdown.items():
            source_name = {1: "Tennis Abstract/Jeff", 2: "API-Tennis", 3: "Excel"}.get(rank, f"Unknown({rank})")
            print(f"  - {source_name}: {count} matches")
    else:
        print("❌ No source_rank column in main cache")

if __name__ == "__main__":
    print("🧪 TESTING JEFF DATA PERSISTENCE")
    
    # Check current state
    check_current_cache()
    
    # Test direct persistence
    test_direct_cache_save()
    
    print("\n✅ Test completed!")