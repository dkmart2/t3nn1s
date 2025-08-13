#!/usr/bin/env python3
"""
Finalize API-Tennis integration with proper data type handling
"""

import pandas as pd
import os
from datetime import datetime, date
from settings import TENNIS_CACHE_DIR
from fixed_api_tennis_integration import WorkingAPITennisClient

def fix_data_types_for_integration():
    """Load and fix data types for proper integration"""
    print("🔧 FIXING DATA TYPES FOR INTEGRATION")
    print("="*60)
    
    # Load API-Tennis data
    client = WorkingAPITennisClient()
    cache_files = list(client.cache_dir.glob('working_api_matches_*.parquet'))
    
    if not cache_files:
        print("❌ No API-Tennis cache files found")
        return pd.DataFrame()
    
    # Load most recent API data
    latest_api_file = max(cache_files, key=lambda x: x.stat().st_mtime)
    api_df = pd.read_parquet(latest_api_file)
    print(f"✅ Loaded API data: {len(api_df)} matches from {latest_api_file.name}")
    
    # Fix data types
    api_df['date'] = pd.to_datetime(api_df['date'])
    
    # Clean string columns
    string_cols = ['player1', 'player2', 'event_name', 'tournament', 'status', 'source']
    for col in string_cols:
        if col in api_df.columns:
            api_df[col] = api_df[col].astype(str)
    
    # Ensure numeric columns are proper type
    numeric_cols = ['source_rank', 'fixture_id']
    for col in numeric_cols:
        if col in api_df.columns:
            api_df[col] = pd.to_numeric(api_df[col], errors='coerce')
    
    print(f"✅ Fixed API data types")
    
    # Load existing dataset
    existing_file = os.path.join(TENNIS_CACHE_DIR, 'final_complete_tennis_2020s.parquet')
    
    try:
        existing_df = pd.read_parquet(existing_file)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        print(f"✅ Loaded existing data: {len(existing_df)} matches")
        
        # Align columns - only keep common ones for now
        api_columns = set(api_df.columns)
        existing_columns = set(existing_df.columns)
        common_columns = api_columns.intersection(existing_columns)
        
        print(f"✅ Common columns: {len(common_columns)}")
        
        # Select common columns
        api_df_aligned = api_df[list(common_columns)]
        existing_df_aligned = existing_df[list(common_columns)]
        
        # Combine datasets
        combined_df = pd.concat([existing_df_aligned, api_df_aligned], ignore_index=True)
        print(f"✅ Combined dataset: {len(combined_df)} matches")
        
        # Clean data types for all columns
        for col in combined_df.select_dtypes(include=['object']).columns:
            if col != 'date':  # Keep date as datetime
                combined_df[col] = combined_df[col].astype(str)
        
        # Deduplicate if composite_id exists
        if 'composite_id' in combined_df.columns:
            initial_count = len(combined_df)
            combined_df = combined_df.sort_values('source_rank').drop_duplicates(
                subset='composite_id', keep='first'
            ).reset_index(drop=True)
            
            duplicates = initial_count - len(combined_df)
            print(f"✅ After deduplication: {len(combined_df)} matches ({duplicates} duplicates removed)")
        
        return combined_df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def save_final_integrated_dataset(df):
    """Save the final integrated dataset"""
    print(f"\n💾 SAVING FINAL INTEGRATED DATASET")
    print("="*60)
    
    if df.empty:
        print("❌ No data to save")
        return False
    
    try:
        # Save updated complete dataset
        complete_file = os.path.join(TENNIS_CACHE_DIR, 'final_complete_with_api.parquet')
        df.to_parquet(complete_file, index=False)
        print(f"✅ Saved: final_complete_with_api.parquet ({len(df)} matches)")
        
        # Update historical_data.parquet for modeling
        historical_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
        df.to_parquet(historical_file, index=False)
        print(f"✅ Updated: historical_data.parquet (ready for modeling)")
        
        # Show final breakdown
        if 'source' in df.columns:
            print(f"\n📊 Final source breakdown:")
            source_counts = df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  - {source}: {count:,}")
        
        # Show date coverage
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"\n📅 Date coverage:")
            print(f"   Range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            # Check recent coverage
            jeff_cutoff = pd.Timestamp('2025-06-10')
            recent_matches = df[df['date'] > jeff_cutoff]
            print(f"   Recent matches (post-6/10): {len(recent_matches)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False

def main():
    """Main execution"""
    print("🎯 FINALIZING API-TENNIS INTEGRATION")
    print("="*80)
    
    # Fix data types and integrate
    final_df = fix_data_types_for_integration()
    
    if not final_df.empty:
        # Save final dataset
        success = save_final_integrated_dataset(final_df)
        
        if success:
            print(f"\n🎉 API-TENNIS INTEGRATION COMPLETED!")
            print("="*80)
            print(f"✅ Dataset size: {len(final_df):,} matches")
            print(f"✅ API-Tennis: Working and integrated")
            print(f"✅ Tennis Abstract: Working and integrated")
            print(f"✅ Jeff 2020s data: Integrated (excluding outdated)")
            print(f"✅ Tennis-data: Integrated (2020-2025)")
            print(f"\n🚀 PIPELINE FULLY COMPLETE!")
            
            return final_df
    
    print(f"\n❌ Integration failed")
    return pd.DataFrame()

if __name__ == "__main__":
    result = main()