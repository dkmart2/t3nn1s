#!/usr/bin/env python3
"""
Build Master Tennis Database - Denormalized Parquet Approach
Combines Jeff, Tennis Abstract, and API data with proper ID reconciliation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json
from match_id_reconciler import MatchIDReconciler
from intelligent_data_merger import IntelligentDataMerger

def load_jeff_comprehensive_data():
    """Load all Jeff CSV files"""
    jeff_files = {
        'points_2020s': 'charting-m-points-2020s.csv',
        'points_2010s': 'charting-m-points-2010s.csv', 
        'points_to_2009': 'charting-m-points-to-2009.csv',
        'return_outcomes': 'charting-m-stats-ReturnOutcomes.csv',
        'shot_types': 'charting-m-stats-ShotTypes.csv',
        'women_points': 'charting-w-points-2020s.csv'
    }
    
    jeff_data = {}
    base_dir = Path('/Users/danielkim/Desktop/t3nn1s')
    
    for file_key, filename in jeff_files.items():
        file_path = base_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)
            jeff_data[file_key] = df
            print(f"Loaded {filename}: {len(df):,} records")
    
    return jeff_data

def load_tennis_abstract_data():
    """Load Tennis Abstract scraped data"""
    ta_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_raw_stats/tennis_abstract_raw_20250812_010121.parquet')
    
    if ta_file.exists():
        df = pd.read_parquet(ta_file)
        print(f"Loaded Tennis Abstract data: {len(df):,} records")
        return df
    else:
        print("No Tennis Abstract data found")
        return pd.DataFrame()

def load_api_tennis_data():
    """Load API-Tennis data from compressed cache"""
    api_dir = Path('/Users/danielkim/Desktop/t3nn1s/data')
    api_files = list(api_dir.glob('**/compressed_data*.csv'))
    
    if api_files:
        # Use the most recent file
        latest_file = max(api_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_file, low_memory=False)
        print(f"Loaded API-Tennis data: {len(df):,} records from {latest_file.name}")
        return df
    else:
        print("No API-Tennis data found")
        return pd.DataFrame()

def get_ta_urls():
    """Get all Tennis Abstract URLs"""
    urls_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_196_urls.txt')
    
    if urls_file.exists():
        with open(urls_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    return []

def merge_post_june_data(api_data, ta_data, reconciler, merger):
    """Intelligently merge post-June 2025 API and TA data"""
    
    print("\n--- MERGING POST-JUNE DATA ---")
    BASE_CUTOFF_DATE = pd.to_datetime('2025-06-10').date()
    
    merged_records = {}
    
    # Filter API data for post-June matches
    api_data['date'] = pd.to_datetime(api_data['date'], errors='coerce')
    post_june_api = api_data[api_data['date'].dt.date > BASE_CUTOFF_DATE]
    
    if len(post_june_api) == 0:
        print("No post-June API data found")
        return merged_records
    
    print(f"Processing {len(post_june_api)} post-June API matches...")
    
    # Convert TA data to lookup dict by player and date
    ta_lookup = {}
    if len(ta_data) > 0:
        ta_data['date'] = pd.to_datetime(ta_data['date'], errors='coerce')
        for _, ta_row in ta_data.iterrows():
            if pd.notna(ta_row['date']):
                date_str = ta_row['date'].strftime('%Y-%m-%d')
                player = str(ta_row.get('player', '')).lower().replace(' ', '_')
                key = f"{date_str}_{player}"
                ta_lookup[key] = ta_row.to_dict()
    
    print(f"TA lookup has {len(ta_lookup)} player records")
    
    # Process each API match
    matches_processed = 0
    matches_enhanced = 0
    
    for _, api_row in post_june_api.iterrows():
        try:
            # Create match record for each player
            date_str = api_row['date'].strftime('%Y-%m-%d')
            
            # Extract players from API data (assuming columns exist)
            p1_name = str(api_row.get('home_team', api_row.get('player1', ''))).lower().replace(' ', '_')
            p2_name = str(api_row.get('away_team', api_row.get('player2', ''))).lower().replace(' ', '_')
            
            if not p1_name or not p2_name:
                continue
            
            # Build composite ID for this match
            tournament = str(api_row.get('tournament', api_row.get('event', 'unknown'))).lower()
            date_obj = api_row['date'].date()
            composite_id = reconciler.build_composite_id(date_obj, tournament, p1_name, p2_name)
            
            # Look for TA data for each player
            p1_ta_key = f"{date_str}_{p1_name}"
            p2_ta_key = f"{date_str}_{p2_name}"
            
            p1_ta = ta_lookup.get(p1_ta_key)
            p2_ta = ta_lookup.get(p2_ta_key)
            
            # Convert API row to match fixture format
            api_fixture = {
                'date': date_str,
                'tournament': tournament,
                'event_key': f"api_{composite_id}",
                'odds': {'home': api_row.get('home_odds'), 'away': api_row.get('away_odds')},
                'statistics': []
            }
            
            # Add API statistics to fixture format
            stat_fields = ['aces', 'double_faults', 'winners', 'unforced_errors', 'first_serve_percentage']
            for field in stat_fields:
                if f'{field}_home' in api_row:
                    api_fixture['statistics'].append({
                        'type': field,
                        'home': api_row.get(f'{field}_home'),
                        'away': api_row.get(f'{field}_away')
                    })
            
            # Use intelligent merger for each player
            p1_merged = merger.merge_api_and_ta_intelligently(api_fixture, p1_ta)
            p2_merged = merger.merge_api_and_ta_intelligently(api_fixture, p2_ta)
            
            # Combine into master record
            master_record = {
                'composite_id': composite_id,
                'date': api_row['date'],
                'tournament': tournament,
                'surface': api_row.get('surface'),
                'round': api_row.get('round'),
                'p1_name': p1_name,
                'p2_name': p2_name,
                
                # Data source tracking
                'data_source': 'api_ta_merged',
                'post_june_match': True,
                'p1_has_ta': bool(p1_ta),
                'p2_has_ta': bool(p2_ta),
            }
            
            # Add player stats with prefixes
            for prefix, player_data in [('p1', p1_merged), ('p2', p2_merged)]:
                for field, value in player_data.items():
                    if field not in ['date', 'tournament', 'event_key']:
                        master_record[f'{prefix}_{field}'] = value
            
            # Overall quality score
            p1_quality = p1_merged.get('data_completeness', {}).get('quality_score', 0.6)
            p2_quality = p2_merged.get('data_completeness', {}).get('quality_score', 0.6)
            master_record['overall_quality_score'] = (p1_quality + p2_quality) / 2
            
            merged_records[composite_id] = master_record
            matches_processed += 1
            
            if p1_ta or p2_ta:
                matches_enhanced += 1
        
        except Exception as e:
            print(f"Error processing API row: {e}")
            continue
    
    print(f"Post-June processing complete:")
    print(f"  Matches processed: {matches_processed}")
    print(f"  Matches with TA enhancement: {matches_enhanced}")
    print(f"  Enhancement rate: {matches_enhanced/matches_processed*100:.1f}%")
    
    return merged_records

def build_master_database():
    """Build denormalized master database with proper ID reconciliation"""
    
    print("=== BUILDING MASTER TENNIS DATABASE ===")
    
    # 1. Initialize reconciler and intelligent merger
    print("\n1. Building match ID crosswalk...")
    reconciler = MatchIDReconciler()
    reconciler.build_crosswalk_from_jeff_data('/Users/danielkim/Desktop/t3nn1s')
    
    # Add TA URLs to crosswalk
    ta_urls = get_ta_urls()
    reconciler.add_ta_urls(ta_urls)
    
    # Initialize intelligent data merger
    merger = IntelligentDataMerger()
    
    # 2. Load all data sources
    print("\n2. Loading data sources...")
    jeff_data = load_jeff_comprehensive_data()
    ta_data = load_tennis_abstract_data()
    api_data = load_api_tennis_data()
    
    # 3. Process Jeff data into master records
    print("\n3. Processing Jeff data...")
    master_records = {}
    
    # Start with points data (most comprehensive)
    points_datasets = ['points_2020s', 'points_2010s', 'points_to_2009', 'women_points']
    
    for dataset in points_datasets:
        if dataset not in jeff_data:
            continue
            
        df = jeff_data[dataset]
        if 'match_id' not in df.columns:
            continue
        
        print(f"  Processing {dataset}: {len(df):,} points")
        
        for match_id in df['match_id'].unique():
            if pd.isna(match_id):
                continue
            
            # Get composite ID
            composite_id = reconciler.get_composite_for_jeff(match_id)
            if not composite_id:
                continue
            
            # Initialize master record if not exists
            if composite_id not in master_records:
                parsed = reconciler.parse_jeff_id(match_id)
                if not parsed:
                    continue
                
                master_records[composite_id] = {
                    'composite_id': composite_id,
                    'jeff_match_id': match_id,
                    'date': pd.to_datetime(parsed['date']),
                    'gender': parsed['gender'],
                    'tournament': parsed['tournament'],
                    'round': parsed['round'],
                    'surface': None,
                    'score': None,
                    'p1_name': None,
                    'p2_name': None,
                    # Jeff point sequences
                    'jeff_points': [],
                    # Stats will be added later
                }
            
            # Add point sequences
            match_points = df[df['match_id'] == match_id]
            point_records = []
            
            for _, point in match_points.iterrows():
                point_record = {
                    'point_num': point.get('Pt', 0),
                    'server': point.get('Svr', 0),
                    'winner': point.get('PtWinner', 0),
                    'first_serve': point.get('1st', ''),
                    'second_serve': point.get('2nd', ''),
                    'score': point.get('Pts', ''),
                    'set1': point.get('Set1', 0),
                    'set2': point.get('Set2', 0),
                }
                point_records.append(point_record)
            
            master_records[composite_id]['jeff_points'] = point_records
    
    print(f"  Created {len(master_records)} master records from points data")
    
    # 4. Add Jeff statistics data
    print("\n4. Adding Jeff statistics...")
    stats_datasets = ['return_outcomes', 'shot_types']
    
    for dataset in stats_datasets:
        if dataset not in jeff_data:
            continue
            
        df = jeff_data[dataset]
        if 'match_id' not in df.columns or 'player' not in df.columns:
            continue
        
        print(f"  Processing {dataset}: {len(df):,} stat records")
        
        for match_id in df['match_id'].unique():
            if pd.isna(match_id):
                continue
                
            composite_id = reconciler.get_composite_for_jeff(match_id)
            if not composite_id or composite_id not in master_records:
                continue
            
            # Get stats for this match
            match_stats = df[df['match_id'] == match_id]
            
            # Group by player
            for player in match_stats['player'].unique():
                if pd.isna(player):
                    continue
                
                player_stats = match_stats[match_stats['player'] == player]
                
                # Determine if p1 or p2
                canonical_player = str(player).lower().replace(' ', '_')
                
                if master_records[composite_id]['p1_name'] is None:
                    master_records[composite_id]['p1_name'] = canonical_player
                    player_prefix = 'p1'
                elif master_records[composite_id]['p1_name'] == canonical_player:
                    player_prefix = 'p1'
                elif master_records[composite_id]['p2_name'] is None:
                    master_records[composite_id]['p2_name'] = canonical_player
                    player_prefix = 'p2'
                elif master_records[composite_id]['p2_name'] == canonical_player:
                    player_prefix = 'p2'
                else:
                    continue  # Skip unknown players
                
                # Add all stats with prefix
                for _, stat_row in player_stats.iterrows():
                    row_type = stat_row.get('row', 'Total')
                    
                    for col in stat_row.index:
                        if col not in ['match_id', 'player', 'row']:
                            col_name = f'{player_prefix}_jeff_{dataset}_{row_type}_{col}'
                            master_records[composite_id][col_name] = stat_row[col]
    
    # 5. Add Tennis Abstract data
    print("\n5. Adding Tennis Abstract data...")
    if len(ta_data) > 0:
        ta_added = 0
        
        for _, ta_row in ta_data.iterrows():
            # Match by date and players
            match_date = ta_row.get('date')
            player = ta_row.get('player', '').lower().replace(' ', '_')
            
            # Find matching composite ID
            matched_composite = None
            for composite_id, record in master_records.items():
                if (record['date'].strftime('%Y-%m-%d') == match_date and 
                    (record['p1_name'] == player or record['p2_name'] == player)):
                    matched_composite = composite_id
                    break
            
            if matched_composite:
                # Determine player prefix
                if master_records[matched_composite]['p1_name'] == player:
                    prefix = 'p1'
                elif master_records[matched_composite]['p2_name'] == player:
                    prefix = 'p2'
                else:
                    continue
                
                # Add TA stats
                for col in ta_row.index:
                    if col not in ['match_id', 'date', 'player', 'url']:
                        col_name = f'{prefix}_ta_{col}'
                        master_records[matched_composite][col_name] = ta_row[col]
                
                ta_added += 1
        
        print(f"  Added Tennis Abstract data to {ta_added} matches")
    
    # 6. Add post-June 2025 data with intelligent merging
    print("\n6. Processing post-June 2025 data...")
    if len(api_data) > 0:
        post_june_records = merge_post_june_data(api_data, ta_data, reconciler, merger)
        
        # Add post-June records to master records
        for composite_id, record in post_june_records.items():
            master_records[composite_id] = record
        
        print(f"  Added {len(post_june_records)} post-June matches")
    else:
        print("  No API data available for post-June processing")
    
    # 7. Convert to DataFrame and save
    print("\n7. Converting to DataFrame and saving...")
    master_df = pd.DataFrame.from_dict(master_records, orient='index')
    
    print(f"  Master database: {len(master_df):,} matches")
    print(f"  Date range: {master_df['date'].min()} to {master_df['date'].max()}")
    print(f"  Columns: {len(master_df.columns)}")
    
    # Add partitioning column
    master_df['year_month'] = master_df['date'].dt.to_period('M')
    
    # Create output directory
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/master_database')
    output_dir.mkdir(exist_ok=True)
    
    # Save as monthly Parquet files
    saved_files = []
    for period, group in master_df.groupby('year_month'):
        filepath = output_dir / f'{period}.parquet'
        group_clean = group.drop('year_month', axis=1)
        group_clean.to_parquet(filepath, engine='pyarrow', compression='gzip')
        saved_files.append(filepath)
        print(f"  Saved {len(group_clean)} matches to {filepath}")
    
    # Save complete database as single file
    complete_file = output_dir / 'complete_master_database.parquet'
    master_df_clean = master_df.drop('year_month', axis=1)
    master_df_clean.to_parquet(complete_file, engine='pyarrow', compression='gzip')
    print(f"  Saved complete database: {complete_file}")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_matches': len(master_df),
        'date_range': {
            'min': master_df['date'].min().isoformat(),
            'max': master_df['date'].max().isoformat()
        },
        'columns': len(master_df.columns),
        'data_sources': {
            'jeff_points': len([c for c in master_df.columns if 'jeff_points' in str(c)]),
            'jeff_stats': len([c for c in master_df.columns if 'jeff_' in c and 'points' not in c]),
            'ta_stats': len([c for c in master_df.columns if '_ta_' in c]),
        },
        'partitioned_files': [str(f) for f in saved_files],
        'complete_file': str(complete_file)
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata: {metadata_file}")
    
    # Display sample
    print(f"\n=== SAMPLE DATA ===")
    sample_cols = ['composite_id', 'date', 'tournament', 'p1_name', 'p2_name']
    available_cols = [c for c in sample_cols if c in master_df.columns]
    print(master_df[available_cols].head())
    
    print(f"\n=== STATISTICS ===")
    print(f"Total matches: {len(master_df):,}")
    print(f"Unique tournaments: {master_df['tournament'].nunique()}")
    print(f"Date coverage: {master_df['date'].min()} to {master_df['date'].max()}")
    print(f"Gender split: {master_df['gender'].value_counts().to_dict()}")
    
    # Point sequence stats
    point_counts = master_df['jeff_points'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    print(f"Point sequences: {(point_counts > 0).sum():,} matches with point data")
    print(f"Average points per match: {point_counts.mean():.1f}")
    
    return master_df, reconciler

if __name__ == "__main__":
    master_df, reconciler = build_master_database()
    print("\n🎾 MASTER DATABASE COMPLETE!")