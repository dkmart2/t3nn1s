#!/usr/bin/env python3
"""
Properly convert Tennis Abstract data to Jeff's CSV format
Handling the actual stat names present in the data
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def convert_tennis_abstract_to_jeff():
    """Convert Tennis Abstract to Jeff format with proper stat mapping"""
    
    # Load data
    ta_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete/complete_tennis_abstract_20250811_231157.parquet')
    ta_data = pd.read_parquet(ta_file)
    
    print(f"Processing {len(ta_data):,} Tennis Abstract records")
    
    # First, let's understand the data structure better
    # Group by match and player to see all stats per player
    grouped = ta_data.groupby(['url', 'Player_canonical'])
    
    jeff_records = []
    
    for (url, player), group in grouped:
        # Create Jeff-style record
        match_id = url.split('/')[-1].replace('.html', '')
        
        # Get metadata from first row
        first_row = group.iloc[0]
        
        # Initialize record with Jeff column structure
        record = {
            'match_id': match_id,
            'player': player,
            'set': 'Total',  # Aggregate data
            'date': first_row.get('match_date'),
            'tournament': first_row.get('tournament'),
            'round': first_row.get('round'),
            'gender': first_row.get('gender')
        }
        
        # Create stat lookup from group
        stats_dict = {}
        for _, row in group.iterrows():
            stat_name = row.get('stat_name', '')
            stat_value = row.get('stat_value', '')
            stats_dict[stat_name] = stat_value
        
        # Map Tennis Abstract stats to Jeff columns
        # Some are direct mappings, others need calculation
        
        # Service stats
        if 'total_pts' in stats_dict:
            record['serve_pts'] = int(stats_dict.get('total_pts', 0))
        
        # Aces - need to calculate from percentage
        if 'aces_pct' in stats_dict and 'total_pts' in stats_dict:
            aces_pct = float(stats_dict.get('aces_pct', 0))
            total_pts = float(stats_dict.get('total_pts', 0))
            record['aces'] = int(total_pts * aces_pct / 100)
        elif 'a' in stats_dict:  # Alternative ace notation
            record['aces'] = int(stats_dict.get('a', 0))
        
        # Double faults
        if 'df' in stats_dict:
            record['dfs'] = int(stats_dict.get('df', 0))
        
        # First serve stats
        if '1stin' in stats_dict:
            record['first_in'] = int(stats_dict.get('1stin', 0))
        if '1st' in stats_dict:
            record['first_won'] = int(stats_dict.get('1st', 0))
        if '2nd' in stats_dict:
            record['second_won'] = int(stats_dict.get('2nd', 0))
        
        # Break points
        if 'bpsaved' in stats_dict:
            record['bp_saved'] = int(stats_dict.get('bpsaved', 0))
        
        # Return stats
        if 'returnable_serves' in stats_dict:
            record['return_pts'] = int(stats_dict.get('returnable_serves', 0))
        
        # Winners and errors
        if 'wnrs' in stats_dict:
            record['winners'] = int(stats_dict.get('wnrs', 0))
        elif 'svwnr' in stats_dict and 'rlywnr' in stats_dict:
            # Combine service winners and rally winners
            record['winners'] = int(stats_dict.get('svwnr', 0)) + int(stats_dict.get('rlywnr', 0))
        
        if 'ufe' in stats_dict:
            record['unforced'] = int(stats_dict.get('ufe', 0))
        
        # Rally/point outcome stats
        if 'winners_pct' in stats_dict and 'total_pts' in stats_dict:
            winners_pct = float(stats_dict.get('winners_pct', 0))
            total_pts = float(stats_dict.get('total_pts', 0))
            record['winners'] = int(total_pts * winners_pct / 100)
        
        # Points won
        if 'points_won_pct' in stats_dict and 'total_pts' in stats_dict:
            won_pct = float(stats_dict.get('points_won_pct', 0))
            total_pts = float(stats_dict.get('total_pts', 0))
            record['total_points_won'] = int(total_pts * won_pct / 100)
        elif 'won_pct' in stats_dict and 'points' in stats_dict:
            won_pct = float(stats_dict.get('won_pct', 0))
            points = float(stats_dict.get('points', 0))
            record['total_points_won'] = int(points * won_pct / 100)
        
        jeff_records.append(record)
    
    # Convert to DataFrame
    jeff_df = pd.DataFrame(jeff_records)
    
    print(f"\nConverted to {len(jeff_df)} match-player records")
    print(f"Unique matches: {jeff_df['match_id'].nunique()}")
    
    # Add missing Jeff columns with defaults
    jeff_required_cols = ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won', 
                          'second_in', 'second_won', 'bk_pts', 'bp_saved', 
                          'return_pts', 'return_pts_won', 'winners', 'winners_fh', 
                          'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']
    
    for col in jeff_required_cols:
        if col not in jeff_df.columns:
            jeff_df[col] = 0  # Default to 0 for missing stats
    
    # Calculate second serve stats if missing
    if 'serve_pts' in jeff_df.columns and 'first_in' in jeff_df.columns:
        jeff_df['second_in'] = jeff_df['serve_pts'] - jeff_df['first_in']
    
    # Save output
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_jeff_format')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save in Jeff's exact format
    jeff_cols_order = ['match_id', 'player', 'set', 'serve_pts', 'aces', 'dfs', 
                       'first_in', 'first_won', 'second_in', 'second_won', 
                       'bk_pts', 'bp_saved', 'return_pts', 'return_pts_won', 
                       'winners', 'winners_fh', 'winners_bh', 'unforced', 
                       'unforced_fh', 'unforced_bh']
    
    # Reorder columns to match Jeff's structure
    available_cols = [col for col in jeff_cols_order if col in jeff_df.columns]
    jeff_formatted = jeff_df[available_cols]
    
    # Save files
    parquet_file = output_dir / f'tennis_abstract_jeff_format_proper_{timestamp}.parquet'
    csv_file = output_dir / f'tennis_abstract_jeff_format_proper_{timestamp}.csv'
    
    jeff_formatted.to_parquet(parquet_file, index=False)
    jeff_formatted.to_csv(csv_file, index=False)
    
    print(f"\nSaved Jeff-formatted data:")
    print(f"  Parquet: {parquet_file}")
    print(f"  CSV: {csv_file}")
    
    # Show sample
    print(f"\nSample records (Jeff format):")
    print(jeff_formatted[['match_id', 'player', 'serve_pts', 'aces', 'winners', 'unforced']].head(10))
    
    # Verify statistics
    print(f"\n=== STATISTICS ===")
    print(f"Total records: {len(jeff_formatted)}")
    print(f"Unique matches: {jeff_formatted['match_id'].nunique()}")
    print(f"Non-zero aces: {(jeff_formatted['aces'] > 0).sum()} records")
    print(f"Non-zero winners: {(jeff_formatted['winners'] > 0).sum()} records")
    
    return jeff_formatted

if __name__ == "__main__":
    result = convert_tennis_abstract_to_jeff()
    
    # Append to existing Jeff data
    if result is not None:
        print("\nTo append to Jeff's existing data:")
        print("1. Load Jeff's charting-m-stats-Overview.csv")
        print("2. Filter for dates >= 2025-06-10")
        print("3. Append this converted Tennis Abstract data")
        print("4. Save as unified dataset")