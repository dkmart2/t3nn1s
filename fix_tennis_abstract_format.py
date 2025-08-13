#!/usr/bin/env python3
"""
Convert Tennis Abstract scraped data from stat_name/stat_value pairs 
to Jeff's CSV columnar format
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def reshape_tennis_abstract_to_jeff_format():
    """Reshape Tennis Abstract data to match Jeff's CSV structure"""
    
    # Load Tennis Abstract data
    ta_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete/complete_tennis_abstract_20250811_231157.parquet')
    ta_data = pd.read_parquet(ta_file)
    
    print(f"Loaded {len(ta_data):,} Tennis Abstract records")
    print(f"Columns: {list(ta_data.columns)}")
    
    # Group by match to pivot stats
    if 'url' in ta_data.columns and 'stat_name' in ta_data.columns:
        # Group by match identifier
        grouped = ta_data.groupby(['url', 'Player_canonical'])
        
        match_records = []
        
        for (url, player), group in grouped:
            # Create match record with Jeff's column structure
            match_record = {
                'match_id': url.split('/')[-1].replace('.html', ''),
                'player': player,
                'url': url
            }
            
            # Add metadata
            first_row = group.iloc[0]
            match_record['date'] = first_row.get('match_date', first_row.get('Date'))
            match_record['tournament'] = first_row.get('tournament')
            match_record['round'] = first_row.get('round')
            match_record['gender'] = first_row.get('gender')
            
            # Pivot stats into columns
            for _, row in group.iterrows():
                stat_name = row.get('stat_name', '')
                stat_value = row.get('stat_value', '')
                
                # Map to Jeff's column names
                column_mapping = {
                    'Aces': 'aces',
                    'Double Faults': 'dfs',
                    'First Serve %': 'first_serve_pct',
                    'First Serve In': 'first_in',
                    'First Serve Won': 'first_won',
                    'Second Serve Won': 'second_won',
                    'Service Points': 'serve_pts',
                    'Return Points': 'return_pts',
                    'Return Points Won': 'return_pts_won',
                    'Break Points': 'bk_pts',
                    'Break Points Saved': 'bp_saved',
                    'Winners': 'winners',
                    'Winners FH': 'winners_fh',
                    'Winners BH': 'winners_bh',
                    'Unforced Errors': 'unforced',
                    'Unforced FH': 'unforced_fh',
                    'Unforced BH': 'unforced_bh'
                }
                
                jeff_column = column_mapping.get(stat_name, stat_name.lower().replace(' ', '_'))
                
                # Convert value to appropriate type
                try:
                    if '%' in str(stat_value):
                        match_record[jeff_column] = float(stat_value.replace('%', ''))
                    elif stat_value and stat_value != '-':
                        match_record[jeff_column] = float(stat_value)
                except:
                    match_record[jeff_column] = stat_value
            
            match_records.append(match_record)
        
        # Convert to DataFrame
        jeff_format_df = pd.DataFrame(match_records)
        
        print(f"\nConverted to {len(jeff_format_df)} match-player records")
        print(f"Columns: {list(jeff_format_df.columns)[:20]}")
        
        # Sample comparison with Jeff's structure
        jeff_columns = ['match_id', 'player', 'set', 'serve_pts', 'aces', 'dfs', 
                       'first_in', 'first_won', 'second_in', 'second_won', 
                       'bk_pts', 'bp_saved', 'return_pts', 'return_pts_won', 
                       'winners', 'winners_fh', 'winners_bh', 'unforced', 
                       'unforced_fh', 'unforced_bh']
        
        # Check column alignment
        missing_cols = [col for col in jeff_columns if col not in jeff_format_df.columns]
        extra_cols = [col for col in jeff_format_df.columns if col not in jeff_columns + ['date', 'tournament', 'round', 'gender', 'url']]
        
        print(f"\nMissing Jeff columns: {missing_cols}")
        print(f"Extra columns: {extra_cols[:10]}")
        
        # Save converted data
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_jeff_format')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'tennis_abstract_jeff_format_{timestamp}.parquet'
        jeff_format_df.to_parquet(output_file, index=False)
        
        # Also save as CSV for inspection
        csv_file = output_dir / f'tennis_abstract_jeff_format_{timestamp}.csv'
        jeff_format_df.to_csv(csv_file, index=False)
        
        print(f"\nSaved converted data:")
        print(f"  Parquet: {output_file}")
        print(f"  CSV: {csv_file}")
        
        # Show sample records
        print(f"\nSample converted records:")
        sample_cols = ['match_id', 'player', 'aces', 'winners', 'serve_pts', 'first_won']
        available_cols = [col for col in sample_cols if col in jeff_format_df.columns]
        print(jeff_format_df[available_cols].head())
        
        return jeff_format_df
        
    else:
        print("ERROR: Expected columns not found in Tennis Abstract data")
        print(f"Available columns: {list(ta_data.columns)}")
        return None

if __name__ == "__main__":
    result = reshape_tennis_abstract_to_jeff_format()
    
    if result is not None:
        # Verify match count
        unique_matches = result['match_id'].nunique()
        print(f"\n=== CONVERSION COMPLETE ===")
        print(f"Total records: {len(result):,}")
        print(f"Unique matches: {unique_matches}")
        print(f"Average records per match: {len(result)/unique_matches:.1f}")
        
        # This should be ~392 records (196 matches × 2 players)