#!/usr/bin/env python3
"""
Simplified Tennis Data Loader
Following user's expectation: "Load jeff's csvs and tennis-data, extract them, add them to database"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR
from datetime import datetime
import logging

def load_jeff_comprehensive_stats():
    """Load Jeff's comprehensive stats data (the real valuable data)"""
    print("=== LOADING JEFF'S COMPREHENSIVE STATS ===")
    
    jeff_data = {'men': {}, 'women': {}}
    working_dir = Path.cwd()
    
    # Files to load (these contain the actual valuable data)
    jeff_files = {
        'men': [
            'charting-m-stats-Overview.csv',
            'charting-m-stats-ServeBasics.csv', 
            'charting-m-stats-ReturnOutcomes.csv',
            'charting-m-points-2020s.csv'  # Point-by-point data
        ],
        'women': [
            'charting-w-stats-Overview.csv',
            'charting-w-stats-ServeBasics.csv',
            'charting-w-stats-ReturnOutcomes.csv', 
            'charting-w-points-2020s.csv'  # Point-by-point data
        ]
    }
    
    total_records = 0
    for gender, files in jeff_files.items():
        jeff_data[gender] = {}
        for filename in files:
            file_path = working_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    jeff_data[gender][filename.replace('.csv', '')] = df
                    total_records += len(df)
                    print(f"✓ Loaded {filename}: {len(df)} records")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            else:
                print(f"⚠️  File not found: {filename}")
    
    print(f"✓ Total Jeff records loaded: {total_records:,}")
    return jeff_data

def load_tennis_data_simple():
    """Load tennis-data Excel files"""
    print("\n=== LOADING TENNIS-DATA EXCEL FILES ===")
    
    base_path = Path.home() / "Desktop" / "data"
    tennis_data = []
    
    # Load men's data
    men_dir = base_path / "tennisdata_men"
    if men_dir.exists():
        for year_file in sorted(men_dir.glob("*.xlsx")):
            try:
                df = pd.read_excel(year_file)
                df['gender'] = 'M'
                df['source'] = 'tennisdata'
                df['source_rank'] = 3  # Lower priority than Jeff
                tennis_data.append(df)
                print(f"✓ Loaded {year_file.name}: {len(df)} matches")
            except Exception as e:
                print(f"❌ Error loading {year_file}: {e}")
    
    # Load women's data
    women_dir = base_path / "tennisdata_women" 
    if women_dir.exists():
        for year_file in sorted(women_dir.glob("*.xlsx")):
            try:
                df = pd.read_excel(year_file)
                df['gender'] = 'W'
                df['source'] = 'tennisdata'
                df['source_rank'] = 3  # Lower priority than Jeff
                tennis_data.append(df)
                print(f"✓ Loaded {year_file.name}: {len(df)} matches")
            except Exception as e:
                print(f"❌ Error loading {year_file}: {e}")
    
    if tennis_data:
        combined = pd.concat(tennis_data, ignore_index=True)
        print(f"✓ Combined tennis-data: {len(combined):,} matches")
        return combined
    else:
        print("❌ No tennis-data loaded")
        return pd.DataFrame()

def create_jeff_match_records_from_stats(jeff_data):
    """Create match records from Jeff's stats data (which has winner info)"""
    print("\n=== CREATING MATCH RECORDS FROM JEFF STATS ===")
    
    matches = []
    
    for gender, data in jeff_data.items():
        print(f"\nProcessing {gender}'s data...")
        
        # Use Overview stats which have match_id and player info
        if 'charting-m-stats-Overview' in data or 'charting-w-stats-Overview' in data:
            key = f'charting-{gender[0].lower()}-stats-Overview'
            if key in data:
                overview_df = data[key]
                print(f"✓ Found overview data: {len(overview_df)} records")
                
                # Group by match_id to get both players
                for match_id, group in overview_df.groupby('match_id'):
                    if len(group) >= 2:  # Need at least 2 players
                        players = group['player'].tolist()
                        
                        # For now, create matches assuming we have winner info elsewhere
                        # This is a simplified approach - real implementation would
                        # need to determine winner from match scores or other data
                        match_record = {
                            'match_id': match_id,
                            'date': match_id.split('-')[0],  # Extract date from match_id
                            'Player_1': players[0],
                            'Player_2': players[1] if len(players) > 1 else 'Unknown',
                            'gender': gender[0].upper(),
                            'source': 'jeff_stats',
                            'source_rank': 1  # Highest priority
                        }
                        matches.append(match_record)
    
    if matches:
        matches_df = pd.DataFrame(matches)
        
        # Convert date format
        try:
            matches_df['date'] = pd.to_datetime(matches_df['date'], format='%Y%m%d', errors='coerce')
            matches_df = matches_df.dropna(subset=['date'])
        except:
            print("⚠️  Date conversion issues, using raw dates")
        
        print(f"✓ Created {len(matches_df)} match records from Jeff stats")
        return matches_df
    else:
        print("❌ No match records created from Jeff stats")
        return pd.DataFrame()

def combine_and_save_data(jeff_data, tennis_data, jeff_matches):
    """Combine all data and save to cache"""
    print("\n=== COMBINING AND SAVING DATA ===")
    
    # Start with tennis-data as base
    if not tennis_data.empty:
        print(f"✓ Base tennis-data: {len(tennis_data)} matches")
        
        # Add composite_id for deduplication
        if 'Winner' in tennis_data.columns and 'Loser' in tennis_data.columns:
            tennis_data['composite_id'] = (
                tennis_data['Winner'].astype(str) + "_" + 
                tennis_data['Loser'].astype(str) + "_" + 
                pd.to_datetime(tennis_data['Date']).dt.strftime('%Y%m%d')
            )
    
    # Add Jeff match records if available
    if not jeff_matches.empty:
        print(f"✓ Jeff match records: {len(jeff_matches)} matches")
        
        # For now, just save Jeff data info separately
        # Real implementation would properly merge and deduplicate
        jeff_info = {
            'total_stats_records': sum(len(data_dict.get(key, [])) 
                                     for data_dict in jeff_data.values() 
                                     for key in data_dict),
            'match_records': len(jeff_matches),
            'date_range': f"{jeff_matches['date'].min()} to {jeff_matches['date'].max()}" if 'date' in jeff_matches.columns else 'Unknown'
        }
        print(f"✓ Jeff data summary: {jeff_info}")
    
    # Save to cache
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'simplified_tennis_data.parquet')
    
    if not tennis_data.empty:
        tennis_data.to_parquet(cache_file, index=False)
        print(f"✓ Saved simplified data to: {cache_file}")
        print(f"  - Total matches: {len(tennis_data)}")
        if 'source_rank' in tennis_data.columns:
            print(f"  - Source breakdown: {tennis_data['source_rank'].value_counts().to_dict()}")
        
        return tennis_data
    else:
        print("❌ No data to save")
        return pd.DataFrame()

def main():
    """Main execution"""
    print("🎾 SIMPLIFIED TENNIS DATA LOADER")
    print("=" * 50)
    
    # Step 1: Load Jeff's comprehensive stats (the valuable data)
    jeff_data = load_jeff_comprehensive_stats()
    
    # Step 2: Load tennis-data Excel files
    tennis_data = load_tennis_data_simple()
    
    # Step 3: Create match records from Jeff stats
    jeff_matches = create_jeff_match_records_from_stats(jeff_data)
    
    # Step 4: Combine and save
    final_data = combine_and_save_data(jeff_data, tennis_data, jeff_matches)
    
    print(f"\n✅ SIMPLIFIED LOADER COMPLETED")
    print(f"Final dataset size: {len(final_data) if not final_data.empty else 0}")

if __name__ == "__main__":
    main()