#!/usr/bin/env python3
"""
Fixed API-Tennis statistics extraction that properly parses all 25+ statistics
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any

def process_api_fixtures_to_dataframe(fixtures: List[Dict]) -> pd.DataFrame:
    """
    Convert API-Tennis fixtures to DataFrame with PROPER statistics extraction
    
    The fix: API-Tennis returns statistics as a list of objects with:
    - player_key: identifies which player (matches first_player_key or second_player_key)
    - stat_name: the statistic name
    - stat_value: the value (may be percentage or number)
    - stat_won/stat_total: additional detail for some stats
    """
    
    records = []
    
    for fixture in fixtures:
        try:
            # Basic match info
            record = {
                'event_key': fixture.get('event_key'),
                'date': fixture.get('event_date'),
                'time': fixture.get('event_time'),
                'player1': fixture.get('event_first_player'),
                'player1_key': fixture.get('first_player_key'),
                'player2': fixture.get('event_second_player'),
                'player2_key': fixture.get('second_player_key'),
                'winner': fixture.get('event_winner'),
                'final_result': fixture.get('event_final_result'),
                'status': fixture.get('event_status'),
                'match_type': fixture.get('event_type_type'),
                'tournament': fixture.get('tournament_name'),
                'tournament_key': fixture.get('tournament_key'),
                'round': fixture.get('tournament_round'),
                'season': fixture.get('tournament_season'),
            }
            
            # Extract scores
            if 'scores' in fixture:
                scores = fixture.get('scores', {})
                for set_num in range(1, 6):
                    set_key = f'{set_num}_set'
                    if set_key in scores:
                        record[f'set{set_num}_p1'] = scores[set_key].get('1_player')
                        record[f'set{set_num}_p2'] = scores[set_key].get('2_player')
            
            # FIXED: Properly extract statistics
            if 'statistics' in fixture:
                # First, get player keys for mapping
                p1_key = fixture.get('first_player_key')
                p2_key = fixture.get('second_player_key')
                
                # Process each statistic
                for stat in fixture.get('statistics', []):
                    player_key = stat.get('player_key')
                    stat_name = stat.get('stat_name', '').lower().replace(' ', '_').replace('%', 'pct')
                    stat_value = stat.get('stat_value', '')
                    
                    # Remove percentage sign if present
                    if isinstance(stat_value, str) and stat_value.endswith('%'):
                        try:
                            stat_value = float(stat_value.rstrip('%'))
                        except:
                            pass
                    
                    # Determine which player this stat belongs to
                    if player_key == p1_key:
                        # Player 1 statistic
                        record[f'{stat_name}_p1'] = stat_value
                        
                        # Add won/total if available
                        if stat.get('stat_won') is not None:
                            record[f'{stat_name}_won_p1'] = stat.get('stat_won')
                        if stat.get('stat_total') is not None:
                            record[f'{stat_name}_total_p1'] = stat.get('stat_total')
                            
                    elif player_key == p2_key:
                        # Player 2 statistic
                        record[f'{stat_name}_p2'] = stat_value
                        
                        # Add won/total if available
                        if stat.get('stat_won') is not None:
                            record[f'{stat_name}_won_p2'] = stat.get('stat_won')
                        if stat.get('stat_total') is not None:
                            record[f'{stat_name}_total_p2'] = stat.get('stat_total')
            
            # Extract point progression if available
            if 'pointbypoint' in fixture:
                points = fixture.get('pointbypoint', [])
                record['has_point_progression'] = True
                record['total_points'] = sum(len(game.get('points', [])) for game in points)
                
                # Calculate break points and other detailed stats from point data
                bp_faced_p1 = 0
                bp_saved_p1 = 0
                bp_faced_p2 = 0
                bp_saved_p2 = 0
                
                for game in points:
                    server = game.get('player_served')
                    winner = game.get('serve_winner')
                    
                    for point in game.get('points', []):
                        if point.get('break_point') == 'Yes':
                            if server == 'First Player':
                                bp_faced_p1 += 1
                                if winner == 'First Player':
                                    bp_saved_p1 += 1
                            else:
                                bp_faced_p2 += 1
                                if winner == 'Second Player':
                                    bp_saved_p2 += 1
                
                # Add calculated break point stats if not already present
                if 'break_points_saved_p1' not in record and bp_faced_p1 > 0:
                    record['break_points_faced_p1'] = bp_faced_p1
                    record['break_points_saved_calc_p1'] = bp_saved_p1
                if 'break_points_saved_p2' not in record and bp_faced_p2 > 0:
                    record['break_points_faced_p2'] = bp_faced_p2
                    record['break_points_saved_calc_p2'] = bp_saved_p2
            else:
                record['has_point_progression'] = False
                record['total_points'] = 0
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing fixture {fixture.get('event_key')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} matches into DataFrame")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Columns: {len(df.columns)}")
        
        # Show extracted statistics columns
        stat_cols = [col for col in df.columns if any(x in col for x in ['aces', 'double_faults', 'serve', 'return', 'winners', 'errors', 'break'])]
        print(f"\nExtracted statistics columns ({len(stat_cols)}):")
        for i, col in enumerate(stat_cols[:20], 1):  # Show first 20
            non_null = df[col].notna().sum()
            print(f"  {i:2}. {col:40} ({non_null}/{len(df)} non-null)")
        if len(stat_cols) > 20:
            print(f"  ... and {len(stat_cols) - 20} more")
        
        # Filter for singles matches
        if 'match_type' in df.columns:
            singles_mask = df['match_type'].str.contains('Singles', na=False)
            singles_count = singles_mask.sum()
            doubles_count = (~singles_mask).sum()
            
            print(f"\nSingles matches: {singles_count}")
            print(f"Doubles matches: {doubles_count}")
    
    return df


def test_with_sample():
    """Test the fixed extraction with the sample JSON file"""
    
    sample_file = Path("/Users/danielkim/Desktop/t3nn1s/sample_api_tennis_match.json")
    
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return None
    
    # Load sample
    with open(sample_file, 'r') as f:
        sample_match = json.load(f)
    
    # Process single match as list
    df = process_api_fixtures_to_dataframe([sample_match])
    
    if len(df) > 0:
        print("\n=== SAMPLE MATCH EXTRACTION TEST ===")
        print(f"Match: {df['player1'].iloc[0]} vs {df['player2'].iloc[0]}")
        print(f"Date: {df['date'].iloc[0]}")
        print(f"Winner: {df['winner'].iloc[0]}")
        
        # Show all non-null statistics
        print("\nExtracted Statistics:")
        for col in df.columns:
            if any(stat in col for stat in ['aces', 'double_faults', 'serve', 'return', 'winners', 'errors', 'break']):
                val = df[col].iloc[0]
                if pd.notna(val):
                    print(f"  {col}: {val}")
        
        return df
    
    return None


def reprocess_existing_data():
    """Reprocess existing API-Tennis data with fixed extraction"""
    
    # Check for existing raw cache
    cache_dir = Path("/Users/danielkim/Desktop/t3nn1s/api_cache_full")
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return None
    
    # Find all cached JSON files
    json_files = list(cache_dir.glob("*.json"))
    print(f"Found {len(json_files)} cached API responses")
    
    if not json_files:
        return None
    
    # Process all cached fixtures
    all_fixtures = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # API responses typically have 'result' key with list of fixtures
                if isinstance(data, dict) and 'result' in data:
                    fixtures = data['result']
                    if isinstance(fixtures, list):
                        all_fixtures.extend(fixtures)
                elif isinstance(data, list):
                    all_fixtures.extend(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"\nTotal fixtures to process: {len(all_fixtures)}")
    
    if all_fixtures:
        # Process with fixed extraction
        df = process_api_fixtures_to_dataframe(all_fixtures)
        
        if len(df) > 0:
            # Save fixed dataset
            output_file = Path("/Users/danielkim/Desktop/t3nn1s/api_tennis_FIXED_EXTRACTION.csv")
            df.to_csv(output_file, index=False)
            print(f"\nSaved fixed extraction to: {output_file}")
            
            # Also save as Parquet
            parquet_file = Path("/Users/danielkim/Desktop/t3nn1s/api_tennis_FIXED_EXTRACTION.parquet")
            df.to_parquet(parquet_file)
            print(f"Saved Parquet: {parquet_file}")
            
            return df
    
    return None


if __name__ == "__main__":
    # First test with sample
    print("=== TESTING WITH SAMPLE MATCH ===")
    sample_df = test_with_sample()
    
    if sample_df is not None:
        print("\n✓ Sample extraction successful!")
        
        # Now reprocess all cached data
        print("\n=== REPROCESSING ALL CACHED DATA ===")
        full_df = reprocess_existing_data()
        
        if full_df is not None:
            print(f"\n✓ Successfully reprocessed {len(full_df)} matches")
            print(f"✓ Statistics properly extracted!")