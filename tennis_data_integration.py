#!/usr/bin/env python3
"""
Streamlined Tennis Data Integration
Following user directive: "Load jeff's csvs and tennis-data, extract them, add them to database"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR
from datetime import datetime, date
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jeff_comprehensive_data():
    """Load ALL Jeff's data - the comprehensive stats that contain real tennis insights"""
    print("🎾 STEP 1: LOADING JEFF'S COMPREHENSIVE DATA")
    print("="*60)
    
    jeff_data = {'men': {}, 'women': {}}
    working_dir = Path.cwd()
    
    # All Jeff files - this is the rich data source
    jeff_patterns = {
        'men': 'charting-m-*.csv',
        'women': 'charting-w-*.csv'
    }
    
    total_records = 0
    total_files = 0
    
    for gender, pattern in jeff_patterns.items():
        print(f"\nLoading {gender}'s data...")
        jeff_data[gender] = {}
        
        for file_path in sorted(working_dir.glob(pattern)):
            filename = file_path.name
            try:
                df = pd.read_csv(file_path, low_memory=False)
                key = filename.replace('.csv', '').replace('charting-', '').replace(f'{gender[0].lower()}-', '')
                jeff_data[gender][key] = df
                total_records += len(df)
                total_files += 1
                print(f"  ✓ {filename}: {len(df):,} records")
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"\n✅ Jeff data loaded: {total_files} files, {total_records:,} total records")
    return jeff_data

def load_tennis_data_excel():
    """Load tennis-data Excel files - the match results base"""
    print("\n🎾 STEP 2: LOADING TENNIS-DATA EXCEL FILES")  
    print("="*60)
    
    base_path = Path.home() / "Desktop" / "data"
    all_matches = []
    
    # Directories to process
    data_dirs = [
        (base_path / "tennisdata_men", 'M'),
        (base_path / "tennisdata_women", 'W')
    ]
    
    total_matches = 0
    total_files = 0
    
    for data_dir, gender in data_dirs:
        if not data_dir.exists():
            print(f"⚠️  Directory not found: {data_dir}")
            continue
            
        print(f"\nLoading {gender} data from {data_dir.name}...")
        
        for year_file in sorted(data_dir.glob("*.xlsx")):
            try:
                df = pd.read_excel(year_file)
                
                # Standardize columns
                df['gender'] = gender
                df['source'] = 'tennis_data'
                df['source_rank'] = 3  # Lower priority than Jeff (source_rank=1)
                
                # Ensure date column
                if 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df['year'] = df['date'].dt.year
                
                # Create composite_id for deduplication
                if 'Winner' in df.columns and 'Loser' in df.columns and 'date' in df.columns:
                    df['composite_id'] = (
                        df['Winner'].astype(str) + "_" + 
                        df['Loser'].astype(str) + "_" + 
                        df['date'].dt.strftime('%Y%m%d')
                    )
                
                all_matches.append(df)
                total_matches += len(df)
                total_files += 1
                print(f"  ✓ {year_file.name}: {len(df):,} matches")
                
            except Exception as e:
                print(f"  ❌ Error loading {year_file.name}: {e}")
    
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        print(f"\n✅ Tennis-data loaded: {total_files} files, {total_matches:,} matches")
        return combined_matches
    else:
        print(f"\n❌ No tennis-data loaded")
        return pd.DataFrame()

def create_jeff_enhanced_matches(jeff_data, tennis_matches):
    """Create enhanced match records using Jeff's rich statistical data"""
    print("\n🎾 STEP 3: CREATING JEFF-ENHANCED MATCH RECORDS")
    print("="*60)
    
    jeff_matches = []
    
    for gender, data in jeff_data.items():
        print(f"\nProcessing {gender}'s Jeff data...")
        
        # Use Overview stats to get match-level data
        overview_key = 'stats-Overview'
        if overview_key in data:
            overview_df = data[overview_key].copy()
            print(f"  Found overview stats: {len(overview_df):,} player-match records")
            
            # Group by match_id to create match records
            match_records = []
            for match_id, group in overview_df.groupby('match_id'):
                if len(group) >= 2:  # Need at least 2 players
                    players = group[['player', 'set']].copy()
                    
                    # Get match date from match_id (format: YYYYMMDD-...)
                    try:
                        date_str = match_id.split('-')[0]
                        match_date = pd.to_datetime(date_str, format='%Y%m%d')
                    except:
                        continue  # Skip if can't parse date
                    
                    # Create match record (using first two players)
                    player_list = players['player'].unique()
                    if len(player_list) >= 2:
                        match_record = {
                            'match_id': match_id,
                            'date': match_date,
                            'Player_1': player_list[0],
                            'Player_2': player_list[1],
                            'gender': gender[0].upper(),
                            'source': 'jeff_comprehensive',
                            'source_rank': 1,  # Highest priority
                            'has_detailed_stats': True
                        }
                        match_records.append(match_record)
            
            if match_records:
                jeff_df = pd.DataFrame(match_records)
                jeff_matches.append(jeff_df)
                print(f"  ✓ Created {len(jeff_df):,} match records from Jeff stats")
    
    if jeff_matches:
        combined_jeff = pd.concat(jeff_matches, ignore_index=True)
        
        # Add composite_id (note: using Player_1/Player_2 since we don't know winner yet)
        combined_jeff['composite_id'] = (
            combined_jeff['Player_1'].astype(str) + "_" + 
            combined_jeff['Player_2'].astype(str) + "_" + 
            combined_jeff['date'].dt.strftime('%Y%m%d')
        )
        
        print(f"\n✅ Jeff matches created: {len(combined_jeff):,} total")
        print(f"   Date range: {combined_jeff['date'].min().date()} to {combined_jeff['date'].max().date()}")
        return combined_jeff
    else:
        print(f"\n❌ No Jeff matches created")
        return pd.DataFrame()

def integrate_datasets(tennis_matches, jeff_matches):
    """Integrate tennis-data and Jeff matches with proper prioritization"""
    print("\n🎾 STEP 4: INTEGRATING DATASETS")
    print("="*60)
    
    datasets = []
    
    # Add tennis-data as base
    if not tennis_matches.empty:
        print(f"✓ Tennis-data base: {len(tennis_matches):,} matches")
        datasets.append(tennis_matches)
    
    # Add Jeff matches (highest priority)
    if not jeff_matches.empty:
        print(f"✓ Jeff enhanced matches: {len(jeff_matches):,} matches")
        datasets.append(jeff_matches)
    
    if not datasets:
        print("❌ No datasets to integrate")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"✓ Combined dataset: {len(combined_data):,} matches")
    
    # Remove records with missing critical data
    initial_count = len(combined_data)
    combined_data = combined_data.dropna(subset=['date', 'composite_id'])
    print(f"✓ After removing invalid dates/IDs: {len(combined_data):,} matches ({initial_count - len(combined_data)} removed)")
    
    # Remove duplicates, keeping highest priority (lowest source_rank)
    if 'source_rank' in combined_data.columns and 'composite_id' in combined_data.columns:
        initial_count = len(combined_data)
        combined_data = combined_data.sort_values('source_rank').drop_duplicates(
            subset='composite_id', keep='first'
        ).reset_index(drop=True)
        print(f"✓ After deduplication: {len(combined_data):,} matches ({initial_count - len(combined_data)} duplicates removed)")
    
    # Show final source breakdown
    if 'source_rank' in combined_data.columns:
        source_breakdown = combined_data['source_rank'].value_counts().sort_index()
        print("\nFinal source breakdown:")
        for rank, count in source_breakdown.items():
            source_name = {1: "Jeff/Tennis Abstract", 2: "API-Tennis", 3: "Tennis-data Excel"}.get(rank, f"Unknown({rank})")
            print(f"  - {source_name}: {count:,} matches")
    
    return combined_data

def save_integrated_dataset(dataset, jeff_data):
    """Save the integrated dataset to cache"""
    print("\n🎾 STEP 5: SAVING INTEGRATED DATASET")
    print("="*60)
    
    if dataset.empty:
        print("❌ No data to save")
        return False
    
    # Save main dataset
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'integrated_tennis_data.parquet')
    
    try:
        # Handle data type issues for parquet
        dataset_clean = dataset.copy()
        
        # Convert problematic columns to string
        for col in dataset_clean.columns:
            if dataset_clean[col].dtype == 'object':
                try:
                    # Try to keep numeric columns numeric
                    pd.to_numeric(dataset_clean[col], errors='raise')
                except:
                    # Convert to string if not numeric
                    dataset_clean[col] = dataset_clean[col].astype(str)
        
        dataset_clean.to_parquet(cache_file, index=False)
        print(f"✅ Main dataset saved: {cache_file}")
        print(f"   Size: {len(dataset_clean):,} matches")
        
    except Exception as e:
        print(f"❌ Error saving to parquet: {e}")
        # Fallback to CSV
        csv_file = cache_file.replace('.parquet', '.csv')
        dataset.to_csv(csv_file, index=False)
        print(f"✅ Saved as CSV instead: {csv_file}")
    
    # Save Jeff data summary  
    jeff_summary = {
        'total_comprehensive_records': sum(len(data_dict.get(key, [])) 
                                         for data_dict in jeff_data.values() 
                                         for key in data_dict),
        'men_datasets': len(jeff_data.get('men', {})),
        'women_datasets': len(jeff_data.get('women', {})),
        'last_updated': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(TENNIS_CACHE_DIR, 'jeff_data_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(jeff_summary, f, indent=2)
    
    print(f"✅ Jeff summary saved: {summary_file}")
    print(f"   Comprehensive records: {jeff_summary['total_comprehensive_records']:,}")
    
    return True

def main():
    """Main execution following user's directive"""
    print("🎾 TENNIS DATA INTEGRATION PIPELINE")
    print("Following directive: Load jeff's csvs and tennis-data, extract them, add them to database")
    print("=" * 80)
    
    try:
        # Step 1: Load Jeff's comprehensive CSV data (the rich statistical data)
        jeff_data = load_jeff_comprehensive_data()
        
        # Step 2: Load tennis-data Excel files (match results base)
        tennis_matches = load_tennis_data_excel()
        
        # Step 3: Create enhanced match records from Jeff's statistical data
        jeff_matches = create_jeff_enhanced_matches(jeff_data, tennis_matches)
        
        # Step 4: Integrate datasets with proper prioritization
        integrated_dataset = integrate_datasets(tennis_matches, jeff_matches)
        
        # Step 5: Save to database/cache
        success = save_integrated_dataset(integrated_dataset, jeff_data)
        
        if success and not integrated_dataset.empty:
            print(f"\n🎯 INTEGRATION COMPLETED SUCCESSFULLY")
            print(f"Final dataset: {len(integrated_dataset):,} matches")
            
            # Show date coverage
            if 'date' in integrated_dataset.columns:
                date_range = f"{integrated_dataset['date'].min().date()} to {integrated_dataset['date'].max().date()}"
                print(f"Date coverage: {date_range}")
            
            # Show data richness
            jeff_enhanced = integrated_dataset[integrated_dataset.get('has_detailed_stats', False) == True]
            if not jeff_enhanced.empty:
                print(f"Matches with detailed Jeff stats: {len(jeff_enhanced):,}")
            
            return integrated_dataset
        else:
            print(f"\n❌ INTEGRATION FAILED")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"\n💥 ERROR DURING INTEGRATION: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()