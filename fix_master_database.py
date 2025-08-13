#!/usr/bin/env python3
"""
Fix and rebuild master database with proper data integration
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
JEFF_DATA_DIR = "/Users/danielkim/Desktop/data/Jeff 6.14.25"
TENNIS_DATA_DIR = "/Users/danielkim/Desktop/data"
PROJECT_DIR = "/Users/danielkim/Desktop/t3nn1s"

class ImprovedMatchIDReconciler:
    """Enhanced ID reconciliation preserving all metadata"""
    
    def __init__(self):
        self.jeff_to_composite = {}
        self.composite_to_jeff = {}
        self.ta_to_jeff = {}
        self.jeff_metadata = {}  # Store gender, round, etc.
        
    def parse_jeff_id(self, jeff_id):
        """Parse Jeff's match_id preserving ALL information"""
        # Format: 20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner
        parts = jeff_id.split('-', 4)
        
        if len(parts) < 5:
            return None
            
        return {
            'date': parts[0],
            'gender': parts[1],
            'tournament': parts[2],
            'round': parts[3],
            'players_str': parts[4],
            'full_id': jeff_id
        }
    
    def parse_ta_url(self, ta_url):
        """Extract Jeff ID from TA URL"""
        # Remove .html and extract the ID part
        if ta_url.endswith('.html'):
            # Format: https://.../20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html
            parts = ta_url.split('/')
            if parts:
                filename = parts[-1].replace('.html', '')
                return filename
        return None
    
    def jeff_to_standard_composite(self, jeff_id):
        """Convert Jeff ID to composite format, preserving metadata"""
        parsed = self.parse_jeff_id(jeff_id)
        if not parsed:
            return None
        
        # Store metadata for later use
        self.jeff_metadata[jeff_id] = {
            'gender': parsed['gender'],
            'round': parsed['round'],
            'tournament': parsed['tournament']
        }
        
        # Parse players
        players_str = parsed['players_str']
        # Simple split - handle complex names later if needed
        player_parts = players_str.split('-')
        
        if len(player_parts) >= 2:
            p1_jeff = player_parts[0]
            p2_jeff = '-'.join(player_parts[1:]) if len(player_parts) > 2 else player_parts[1]
            
            # Convert to canonical format
            p1_canonical = self.jeff_name_to_canonical(p1_jeff)
            p2_canonical = self.jeff_name_to_canonical(p2_jeff)
            
            # Create composite with metadata embedded
            date_str = parsed['date']
            tournament_norm = parsed['tournament'].lower()
            
            # Include gender and round in composite for uniqueness
            composite = f"{date_str}-{tournament_norm}-{p1_canonical}-{p2_canonical}"
            
            return composite
        
        return None
    
    def jeff_name_to_canonical(self, jeff_name):
        """Convert Jeff's First_Last to canonical last_f"""
        parts = jeff_name.split('_')
        if len(parts) >= 2:
            first = parts[0].lower()
            last = '_'.join(parts[1:]).lower()
            return f"{last}_{first[0]}"
        return jeff_name.lower()


def load_jeff_matches_with_metadata():
    """Load Jeff match files with Surface and Score"""
    logging.info("Loading Jeff matches with metadata...")
    
    all_matches = []
    
    for gender in ['men', 'women']:
        match_file = f"{JEFF_DATA_DIR}/{gender}/charting-m-matches.csv" if gender == 'men' else f"{JEFF_DATA_DIR}/{gender}/charting-w-matches.csv"
        
        if os.path.exists(match_file):
            df = pd.read_csv(match_file, low_memory=False)
            df['gender'] = 'M' if gender == 'men' else 'W'
            
            # Create Jeff match ID from components
            if 'match_id' not in df.columns:
                # Build match_id from Date, Player1, Player2, etc.
                if all(col in df.columns for col in ['Date', 'Player 1', 'Player 2']):
                    df['jeff_match_id'] = df.apply(
                        lambda x: f"{x['Date']}-{x['gender']}-{x.get('Tournament', 'Unknown')}-{x.get('Round', 'Unknown')}-{x['Player 1'].replace(' ', '_')}-{x['Player 2'].replace(' ', '_')}",
                        axis=1
                    )
            else:
                df['jeff_match_id'] = df['match_id']
            
            logging.info(f"Loaded {len(df)} {gender} matches with Surface/Score data")
            all_matches.append(df)
    
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    return pd.DataFrame()


def load_tennis_abstract_data():
    """Load and process Tennis Abstract data"""
    logging.info("Loading Tennis Abstract data...")
    
    ta_file = f"{PROJECT_DIR}/tennis_abstract_complete/complete_tennis_abstract_20250811_231157.parquet"
    
    if os.path.exists(ta_file):
        ta_df = pd.read_parquet(ta_file)
        logging.info(f"Loaded {len(ta_df)} Tennis Abstract records")
        
        # Pivot to match-level data
        # Group by match and aggregate stats
        match_groups = ta_df.groupby(['Date', 'gender', 'tournament', 'round', 'player1', 'player2', 'url'])
        
        ta_matches = []
        for (date, gender, tournament, round_str, p1, p2, url), group in match_groups:
            match_data = {
                'ta_date': date,
                'ta_gender': gender,
                'ta_tournament': tournament,
                'ta_round': round_str,
                'ta_player1': p1,
                'ta_player2': p2,
                'ta_url': url
            }
            
            # Aggregate stats for each player
            for player in [p1, p2]:
                player_stats = group[group['Player_canonical'] == player.lower().replace(' ', '_')]
                for _, stat_row in player_stats.iterrows():
                    stat_name = f"ta_{stat_row['stat_name']}_{stat_row['stat_context']}"
                    player_prefix = 'p1' if player == p1 else 'p2'
                    match_data[f"{player_prefix}_{stat_name}"] = stat_row['stat_value']
            
            ta_matches.append(match_data)
        
        ta_matches_df = pd.DataFrame(ta_matches)
        logging.info(f"Processed {len(ta_matches_df)} Tennis Abstract matches")
        return ta_matches_df
    
    return pd.DataFrame()


def parse_jeff_point_sequences(points_str):
    """Parse Jeff's point sequence notation into structured data"""
    # Handle numpy arrays and other types
    if isinstance(points_str, (list, np.ndarray)):
        points = points_str
    elif pd.isna(points_str):
        return None
    else:
        try:
            # If it's a string, try to parse it
            if isinstance(points_str, str) and points_str.startswith('['):
                points = ast.literal_eval(points_str)
            else:
                points = points_str
        except:
            return None
    
    # Convert numpy array to list if needed
    if isinstance(points, np.ndarray):
        try:
            points = points.tolist()
        except:
            return None
    
    if not isinstance(points, list):
        return None
    
    try:
        # Extract key statistics from point sequences
        total_points = len(points)
        aces = sum(1 for p in points if isinstance(p, dict) and 
                  (p.get('first_serve', '').endswith('*') or p.get('second_serve', '').endswith('*')))
        
        # Identify break points, momentum shifts, etc.
        break_points = 0
        for p in points:
            if isinstance(p, dict):
                score = p.get('score', '')
                if score in ['30-40', '40-AD', '0-40', '15-40']:
                    break_points += 1
        
        return {
            'total_points': total_points,
            'aces_from_sequence': aces,
            'break_points': break_points,
            'raw_sequence': points  # Keep raw for advanced analysis
        }
    except:
        return None


def build_fixed_master_database():
    """Build master database with all fixes applied"""
    logging.info("Building fixed master database...")
    
    # 1. Load Jeff comprehensive data from CSV files
    logging.info("Loading Jeff comprehensive data from CSV files...")
    jeff_data = {'men': {}, 'women': {}}
    
    # Load Jeff point files
    jeff_files = {
        'men': {
            'points_2020s': f"{PROJECT_DIR}/charting-m-points-2020s.csv",
            'points_2010s': f"{PROJECT_DIR}/charting-m-points-2010s.csv",
            'points_to_2009': f"{PROJECT_DIR}/charting-m-points-to-2009.csv",
            'return_outcomes': f"{PROJECT_DIR}/charting-m-stats-ReturnOutcomes.csv",
            'shot_types': f"{PROJECT_DIR}/charting-m-stats-ShotTypes.csv"
        },
        'women': {
            'points': f"{PROJECT_DIR}/charting-w-points-2020s.csv"
        }
    }
    
    for gender, files in jeff_files.items():
        for name, filepath in files.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, low_memory=False)
                # Add jeff_original_id if match_id exists
                if 'match_id' in df.columns:
                    df['jeff_original_id'] = df['match_id']
                jeff_data[gender][name] = df
                logging.info(f"Loaded {name}: {len(df)} records")
    
    # Also load from existing master database to get the data
    existing_master = f"{PROJECT_DIR}/master_database/complete_master_database.parquet"
    if os.path.exists(existing_master):
        existing_df = pd.read_parquet(existing_master)
        # Use this as our base for jeff_data structure
        jeff_data['matches'] = existing_df
        logging.info(f"Loaded existing master with {len(existing_df)} matches")
    
    # 2. Load Jeff matches with Surface/Score
    jeff_matches_df = load_jeff_matches_with_metadata()
    
    # 3. Load Tennis Abstract data
    ta_matches_df = load_tennis_abstract_data()
    
    # 4. Build improved reconciler
    reconciler = ImprovedMatchIDReconciler()
    
    # 5. Process and merge all data
    master_records = []
    
    # Process Jeff comprehensive data
    if 'men' in jeff_data:
        for dataset_name, df in jeff_data['men'].items():
            if 'jeff_original_id' in df.columns:
                for jeff_id in df['jeff_original_id'].unique():
                    if pd.notna(jeff_id):
                        # Get composite ID
                        composite = reconciler.jeff_to_standard_composite(jeff_id)
                        if composite:
                            reconciler.jeff_to_composite[jeff_id] = composite
                            reconciler.composite_to_jeff[composite] = jeff_id
    
    # Map TA URLs to Jeff IDs
    if not ta_matches_df.empty:
        for _, ta_match in ta_matches_df.iterrows():
            ta_url = ta_match.get('ta_url', '')
            if ta_url:
                jeff_id_from_url = reconciler.parse_ta_url(ta_url)
                if jeff_id_from_url and jeff_id_from_url in reconciler.jeff_to_composite:
                    reconciler.ta_to_jeff[ta_url] = jeff_id_from_url
    
    logging.info(f"Built reconciler with {len(reconciler.jeff_to_composite)} Jeff IDs and {len(reconciler.ta_to_jeff)} TA mappings")
    
    # 6. Create master records with all data
    processed_composites = set()
    
    # Process existing master database if we have it
    if 'matches' in jeff_data and not jeff_data['matches'].empty:
        matches_df = jeff_data['matches']
        
        for _, match in matches_df.iterrows():
            jeff_id = match.get('jeff_match_id')
            composite_id = match.get('composite_id')
            
            if not jeff_id or pd.isna(jeff_id):
                continue
            
            if composite_id in processed_composites:
                continue
            
            processed_composites.add(composite_id)
            
            # Start with existing data
            record = match.to_dict()
            
            # Add Surface and Score from jeff_matches_df
            if not jeff_matches_df.empty:
                # Fix the column access
                if 'jeff_match_id' in jeff_matches_df.columns:
                    match_meta = jeff_matches_df[jeff_matches_df['jeff_match_id'] == jeff_id]
                elif 'match_id' in jeff_matches_df.columns:
                    match_meta = jeff_matches_df[jeff_matches_df['match_id'] == jeff_id]
                else:
                    match_meta = pd.DataFrame()
                if not match_meta.empty:
                    record['surface'] = match_meta.iloc[0].get('Surface')
                    record['score'] = match_meta.iloc[0].get('Score') 
                    record['match_minutes'] = match_meta.iloc[0].get('Match minutes')
            
            # Parse point sequences if available
            if 'jeff_points' in record:
                point_data = parse_jeff_point_sequences(record['jeff_points'])
                if point_data:
                    record['total_points'] = point_data['total_points']
                    record['aces_from_sequence'] = point_data['aces_from_sequence']
                    record['break_points'] = point_data['break_points']
            
            # Add TA data
            if not ta_matches_df.empty:
                # Try to match by date and players
                for _, ta_match in ta_matches_df.iterrows():
                    ta_url = ta_match.get('ta_url', '')
                    jeff_id_from_url = reconciler.parse_ta_url(ta_url) if ta_url else None
                    
                    if jeff_id_from_url == jeff_id:
                        # Add all TA columns
                        for col in ta_match.index:
                            if col.startswith('p1_ta_') or col.startswith('p2_ta_'):
                                record[col] = ta_match[col]
                        break
            
            master_records.append(record)
    
    # Also process data by gender if we have structured Jeff data
    for gender in ['men', 'women']:
        if gender not in jeff_data:
            continue
        
        # Get base matches from points or other datasets
        for dataset_name in jeff_data[gender].keys():
            
            for _, match in matches_df.iterrows():
                jeff_id = match.get('jeff_original_id') or match.get('match_id')
                if not jeff_id or pd.isna(jeff_id):
                    continue
                
                composite_id = reconciler.jeff_to_standard_composite(jeff_id)
                if not composite_id or composite_id in processed_composites:
                    continue
                
                processed_composites.add(composite_id)
                
                # Parse Jeff ID for metadata
                parsed = reconciler.parse_jeff_id(jeff_id)
                if not parsed:
                    continue
                
                # Initialize record
                record = {
                    'composite_id': composite_id,
                    'jeff_match_id': jeff_id,
                    'date': pd.to_datetime(parsed['date'], format='%Y%m%d'),
                    'gender': parsed['gender'],
                    'tournament': parsed['tournament'],
                    'round': parsed['round']
                }
                
                # Add Surface and Score from jeff_matches_df
                if not jeff_matches_df.empty:
                    match_meta = jeff_matches_df[jeff_matches_df['jeff_match_id'] == jeff_id]
                    if not match_meta.empty:
                        record['surface'] = match_meta.iloc[0].get('Surface')
                        record['score'] = match_meta.iloc[0].get('Score')
                        record['match_minutes'] = match_meta.iloc[0].get('Match minutes')
                
                # Parse and add point sequences
                if 'jeff_points' in match.index:
                    point_data = parse_jeff_point_sequences(match['jeff_points'])
                    if point_data:
                        record['total_points'] = point_data['total_points']
                        record['aces_from_sequence'] = point_data['aces_from_sequence']
                        record['break_points'] = point_data['break_points']
                        record['jeff_points_parsed'] = json.dumps(point_data['raw_sequence'][:10])  # Sample
                
                # Add all Jeff stats
                for dataset_name in jeff_data[gender].keys():
                    if dataset_name == 'matches':
                        continue
                    
                    stats_df = jeff_data[gender][dataset_name]
                    if 'match_id' in stats_df.columns or 'jeff_original_id' in stats_df.columns:
                        match_stats = stats_df[
                            (stats_df.get('match_id', stats_df.get('jeff_original_id')) == jeff_id)
                        ]
                        
                        for _, stat_row in match_stats.iterrows():
                            player = stat_row.get('player')
                            if pd.isna(player):
                                continue
                            
                            # Determine player number
                            if 'p1_name' not in record:
                                record['p1_name'] = player.lower().replace(' ', '_')
                                prefix = 'p1'
                            elif record['p1_name'] == player.lower().replace(' ', '_'):
                                prefix = 'p1'
                            else:
                                if 'p2_name' not in record:
                                    record['p2_name'] = player.lower().replace(' ', '_')
                                prefix = 'p2'
                            
                            # Add stats
                            for col in stat_row.index:
                                if col not in ['match_id', 'jeff_original_id', 'player', 'set']:
                                    value = stat_row[col]
                                    if pd.notna(value):
                                        record[f'{prefix}_jeff_{dataset_name}_{col}'] = value
                
                # Add Tennis Abstract data if available
                if not ta_matches_df.empty:
                    # Find matching TA match
                    for _, ta_match in ta_matches_df.iterrows():
                        ta_url = ta_match.get('ta_url', '')
                        if ta_url in reconciler.ta_to_jeff:
                            if reconciler.ta_to_jeff[ta_url] == jeff_id:
                                # Add TA stats
                                for col in ta_match.index:
                                    if col.startswith('p1_ta_') or col.startswith('p2_ta_'):
                                        record[col] = ta_match[col]
                                break
                
                master_records.append(record)
    
    # Convert to DataFrame
    master_df = pd.DataFrame(master_records)
    logging.info(f"Created master database with {len(master_df)} matches and {len(master_df.columns)} columns")
    
    # 7. Save with improved structure
    output_dir = f"{PROJECT_DIR}/master_database_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete database
    complete_file = f"{output_dir}/complete_master_fixed.parquet"
    master_df.to_parquet(complete_file, engine='pyarrow', compression='snappy')
    
    # Save monthly partitions
    master_df['year_month'] = pd.to_datetime(master_df['date']).dt.to_period('M')
    for period, group in master_df.groupby('year_month'):
        filepath = f"{output_dir}/{period}.parquet"
        group.drop('year_month', axis=1).to_parquet(filepath, engine='pyarrow')
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_matches': len(master_df),
        'total_columns': len(master_df.columns),
        'date_range': {
            'min': master_df['date'].min().isoformat() if not master_df.empty else None,
            'max': master_df['date'].max().isoformat() if not master_df.empty else None
        },
        'data_completeness': {
            'surface': master_df['surface'].notna().sum() if 'surface' in master_df else 0,
            'score': master_df['score'].notna().sum() if 'score' in master_df else 0,
            'ta_data': len([c for c in master_df.columns if 'ta_' in c]),
            'jeff_points_parsed': master_df['total_points'].notna().sum() if 'total_points' in master_df else 0
        }
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save improved crosswalk
    crosswalk = {
        'jeff_to_composite': reconciler.jeff_to_composite,
        'composite_to_jeff': reconciler.composite_to_jeff,
        'ta_to_jeff': reconciler.ta_to_jeff,
        'jeff_metadata': reconciler.jeff_metadata
    }
    
    with open(f"{output_dir}/crosswalk_fixed.json", 'w') as f:
        json.dump(crosswalk, f, indent=2)
    
    logging.info(f"Master database saved to {output_dir}")
    
    # Print summary statistics
    print("\n=== MASTER DATABASE SUMMARY ===")
    print(f"Total matches: {len(master_df)}")
    print(f"Total columns: {len(master_df.columns)}")
    print(f"Date range: {master_df['date'].min()} to {master_df['date'].max()}")
    
    if 'surface' in master_df:
        print(f"Matches with surface: {master_df['surface'].notna().sum()} ({100*master_df['surface'].notna().sum()/len(master_df):.1f}%)")
    if 'score' in master_df:
        print(f"Matches with score: {master_df['score'].notna().sum()} ({100*master_df['score'].notna().sum()/len(master_df):.1f}%)")
    if 'total_points' in master_df:
        print(f"Matches with parsed points: {master_df['total_points'].notna().sum()} ({100*master_df['total_points'].notna().sum()/len(master_df):.1f}%)")
    
    ta_cols = [c for c in master_df.columns if 'ta_' in c]
    print(f"Tennis Abstract columns: {len(ta_cols)}")
    
    return master_df


if __name__ == "__main__":
    master_df = build_fixed_master_database()
    print("\n✅ Master database rebuild complete!")