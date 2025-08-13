#!/usr/bin/env python3
"""
Create the final hybrid dataset combining:
1. Jeff Sackmann CSV data (pre-2025-06-10)
2. Tennis Abstract raw statistics (post-2025-06-10)  
3. API-Tennis context where available

This creates the complete Tennis prediction dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def create_final_hybrid_dataset():
    """Create the ultimate tennis dataset"""
    
    print("=== CREATING FINAL HYBRID TENNIS DATASET ===")
    
    # 1. Load Jeff Sackmann data (pre-6/10/2025)
    print("\n1. Loading Jeff Sackmann data...")
    jeff_files = [
        'charting-m-points-2010s.csv',
        'charting-m-points-2020s.csv', 
        'charting-m-points-to-2009.csv',
        'charting-m-stats-ReturnOutcomes.csv',
        'charting-m-stats-ShotTypes.csv',
        'charting-w-points-2020s.csv'
    ]
    
    jeff_dfs = []
    for file in jeff_files:
        file_path = Path(f'/Users/danielkim/Desktop/t3nn1s/{file}')
        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)
            print(f"  ✓ Loaded {file}: {len(df):,} records")
            jeff_dfs.append(df)
    
    if jeff_dfs:
        jeff_data = pd.concat(jeff_dfs, ignore_index=True)
        # Filter for pre-6/10/2025 only
        if 'date' in jeff_data.columns:
            # Convert date column to string for comparison
            jeff_data['date'] = jeff_data['date'].astype(str)
            jeff_data = jeff_data[jeff_data['date'] < '2025-06-10']
            print(f"  → Jeff data (pre-6/10): {len(jeff_data):,} records")
    else:
        jeff_data = pd.DataFrame()
        print("  ✗ No Jeff data found")
    
    # 2. Load Tennis Abstract raw data (post-6/10/2025)
    print("\n2. Loading Tennis Abstract raw data...")
    ta_raw_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_raw_stats/tennis_abstract_raw_20250812_010121.parquet')
    
    if ta_raw_file.exists():
        ta_data = pd.read_parquet(ta_raw_file)
        # Filter for post-6/10/2025 only
        ta_data['date'] = ta_data['date'].astype(str)
        ta_data = ta_data[ta_data['date'] >= '2025-06-10']
        print(f"  ✓ Tennis Abstract raw (post-6/10): {len(ta_data):,} records")
        print(f"    → Unique matches: {ta_data['match_id'].nunique()}")
        print(f"    → Date range: {ta_data['date'].min()} to {ta_data['date'].max()}")
    else:
        ta_data = pd.DataFrame()
        print("  ✗ No Tennis Abstract raw data found")
    
    # 3. Load API-Tennis data for context
    print("\n3. Loading API-Tennis context...")
    api_files = list(Path('/Users/danielkim/Desktop/t3nn1s').glob('*api_tennis*.parquet'))
    
    api_data = pd.DataFrame()
    for file in api_files:
        try:
            df = pd.read_parquet(file)
            if len(df) > 0:
                api_data = pd.concat([api_data, df], ignore_index=True)
                print(f"  ✓ Loaded {file.name}: {len(df):,} records")
        except:
            pass
    
    if len(api_data) > 0:
        print(f"  → Total API-Tennis data: {len(api_data):,} records")
    else:
        print("  → No API-Tennis data available")
    
    # 4. Standardize column structures
    print("\n4. Standardizing data structures...")
    
    # Define core Jeff columns
    jeff_core_cols = [
        'match_id', 'date', 'tournament', 'round', 'player', 'set',
        'serve_pts', 'aces', 'dfs', 'first_in', 'first_won', 
        'second_in', 'second_won', 'bk_pts', 'bp_saved',
        'return_pts', 'return_pts_won', 'winners', 'winners_fh', 
        'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh'
    ]
    
    # Standardize Jeff data
    if len(jeff_data) > 0:
        # Add missing columns with defaults
        for col in jeff_core_cols:
            if col not in jeff_data.columns:
                jeff_data[col] = 0
        
        # Add source marker
        jeff_data['source'] = 'Jeff_Sackmann'
        jeff_data['source_rank'] = 1
        
        # Select core columns
        jeff_standardized = jeff_data[jeff_core_cols + ['source', 'source_rank']].copy()
        print(f"  ✓ Standardized Jeff data: {len(jeff_standardized):,} records")
    else:
        jeff_standardized = pd.DataFrame()
    
    # Standardize Tennis Abstract data  
    if len(ta_data) > 0:
        # Add missing columns with defaults
        for col in jeff_core_cols:
            if col not in ta_data.columns:
                ta_data[col] = 0
        
        # Add source marker
        ta_data['source'] = 'Tennis_Abstract'
        ta_data['source_rank'] = 1  # Same quality as Jeff
        
        # Select core columns
        ta_standardized = ta_data[jeff_core_cols + ['source', 'source_rank']].copy()
        print(f"  ✓ Standardized Tennis Abstract data: {len(ta_standardized):,} records")
    else:
        ta_standardized = pd.DataFrame()
    
    # 5. Combine primary datasets
    print("\n5. Combining datasets...")
    
    primary_datasets = []
    if len(jeff_standardized) > 0:
        primary_datasets.append(jeff_standardized)
    if len(ta_standardized) > 0:
        primary_datasets.append(ta_standardized)
    
    if primary_datasets:
        combined_data = pd.concat(primary_datasets, ignore_index=True)
        print(f"  ✓ Combined dataset: {len(combined_data):,} records")
        
        # Remove duplicates (prefer Tennis Abstract for post-6/10 overlaps)
        print("  → Removing duplicates...")
        before_dedup = len(combined_data)
        
        # Create composite ID for deduplication
        combined_data['composite_id'] = combined_data['match_id'] + '_' + combined_data['player'] + '_' + combined_data['date'].astype(str)
        
        # Sort by source rank (1 = highest priority) and keep first
        combined_data = combined_data.sort_values(['composite_id', 'source_rank', 'source'])
        combined_data = combined_data.drop_duplicates(subset=['composite_id'], keep='first')
        combined_data = combined_data.drop('composite_id', axis=1)
        
        after_dedup = len(combined_data)
        print(f"  → Removed {before_dedup - after_dedup:,} duplicates")
        print(f"  → Final dataset: {after_dedup:,} records")
    else:
        combined_data = pd.DataFrame()
        print("  ✗ No data to combine")
    
    # 6. Add API-Tennis context
    if len(api_data) > 0 and len(combined_data) > 0:
        print("\n6. Adding API-Tennis context...")
        
        # Create match mapping for API context
        api_context = {}
        
        # Group API data by match for context
        if 'event_name' in api_data.columns and 'event_date' in api_data.columns:
            for _, row in api_data.iterrows():
                match_key = f"{row['event_date']}_{row['event_name']}"
                if match_key not in api_context:
                    api_context[match_key] = {
                        'api_tournament': row.get('event_name', ''),
                        'api_date': row.get('event_date', ''),
                        'api_surface': row.get('surface', ''),
                        'api_odds_home': row.get('odds_home', ''),
                        'api_odds_away': row.get('odds_away', ''),
                    }
        
        # Add API context where possible
        combined_data['api_tournament'] = ''
        combined_data['api_surface'] = ''
        combined_data['api_odds'] = ''
        
        matches_with_api = 0
        for idx, row in combined_data.iterrows():
            match_key = f"{row['date']}_{row['tournament']}"
            if match_key in api_context:
                context = api_context[match_key]
                combined_data.at[idx, 'api_tournament'] = context['api_tournament']
                combined_data.at[idx, 'api_surface'] = context['api_surface']
                combined_data.at[idx, 'api_odds'] = f"{context['api_odds_home']}/{context['api_odds_away']}"
                matches_with_api += 1
        
        print(f"  ✓ Added API context to {matches_with_api:,} records")
    
    # 7. Generate statistics and save
    print("\n7. Generating final statistics...")
    
    if len(combined_data) > 0:
        # Ensure date column is string for min/max operations
        combined_data['date'] = combined_data['date'].astype(str)
        
        # Date range
        date_min = combined_data['date'].min()
        date_max = combined_data['date'].max()
        
        # Source breakdown
        source_stats = combined_data['source'].value_counts()
        
        # Match statistics  
        total_matches = combined_data['match_id'].nunique()
        total_players = combined_data['player'].nunique()
        
        # Quality metrics
        non_zero_aces = (combined_data['aces'] > 0).sum()
        non_zero_winners = (combined_data['winners'] > 0).sum()
        avg_serve_pts = combined_data['serve_pts'].mean()
        avg_aces = combined_data['aces'].mean()
        
        print(f"\n=== FINAL DATASET STATISTICS ===")
        print(f"Total records: {len(combined_data):,}")
        print(f"Unique matches: {total_matches:,}")
        print(f"Unique players: {total_players:,}")
        print(f"Date range: {date_min} to {date_max}")
        print(f"\nSource breakdown:")
        for source, count in source_stats.items():
            print(f"  {source}: {count:,} records ({count/len(combined_data)*100:.1f}%)")
        print(f"\nData quality:")
        print(f"  Records with aces: {non_zero_aces:,} ({non_zero_aces/len(combined_data)*100:.1f}%)")
        print(f"  Records with winners: {non_zero_winners:,} ({non_zero_winners/len(combined_data)*100:.1f}%)")
        print(f"  Average serve points: {avg_serve_pts:.1f}")
        print(f"  Average aces: {avg_aces:.1f}")
        
        # 8. Save final dataset
        print(f"\n8. Saving final hybrid dataset...")
        
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/final_hybrid_dataset')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in multiple formats
        parquet_file = output_dir / f'hybrid_tennis_dataset_{timestamp}.parquet'
        csv_file = output_dir / f'hybrid_tennis_dataset_{timestamp}.csv'
        json_file = output_dir / f'hybrid_tennis_metadata_{timestamp}.json'
        
        # Ensure all string columns are properly typed
        string_cols = ['match_id', 'date', 'tournament', 'round', 'player', 'set', 'source']
        for col in string_cols:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].astype(str)
        
        combined_data.to_parquet(parquet_file, index=False, compression='gzip')
        combined_data.to_csv(csv_file, index=False)
        
        # Save metadata
        metadata = {
            'created': timestamp,
            'total_records': len(combined_data),
            'unique_matches': total_matches,
            'unique_players': total_players,
            'date_range': {'min': str(date_min), 'max': str(date_max)},
            'sources': source_stats.to_dict(),
            'quality_metrics': {
                'records_with_aces': int(non_zero_aces),
                'records_with_winners': int(non_zero_winners),
                'avg_serve_points': float(avg_serve_pts),
                'avg_aces': float(avg_aces)
            },
            'files': {
                'parquet': str(parquet_file),
                'csv': str(csv_file)
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Parquet: {parquet_file}")
        print(f"  ✓ CSV: {csv_file}")  
        print(f"  ✓ Metadata: {json_file}")
        
        # Sample data
        print(f"\n=== SAMPLE DATA ===")
        print(combined_data[['match_id', 'date', 'player', 'source', 'serve_pts', 'aces', 'winners']].head(10))
        
        return combined_data
    
    else:
        print("  ✗ No data to save")
        return None

if __name__ == "__main__":
    result = create_final_hybrid_dataset()
    
    if result is not None:
        print(f"\n🎾 SUCCESS: Created hybrid tennis dataset with {len(result):,} records!")
        print(f"   Combines Jeff Sackmann + Tennis Abstract + API-Tennis context")
        print(f"   Ready for machine learning and prediction models!")
    else:
        print(f"\n❌ FAILED: Could not create hybrid dataset")