#!/usr/bin/env python3
"""
Apply intelligent merger to API + Tennis Abstract data
"""

import pandas as pd
from pathlib import Path
from intelligent_data_merger import IntelligentDataMerger
import json

def load_api_data():
    """Load the FULL API-Tennis data"""
    # Try parquet first (faster)
    parquet_file = Path('/Users/danielkim/Desktop/t3nn1s/api_tennis_post_june_2025_FULL.parquet')
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        print(f"Loaded API data: {len(df)} matches")
        return df
    
    # Fallback to CSV
    csv_file = Path('/Users/danielkim/Desktop/t3nn1s/api_tennis_post_june_2025_FULL.csv')
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"Loaded API data: {len(df)} matches")
        return df
    
    return pd.DataFrame()

def load_ta_data():
    """Load Tennis Abstract data"""
    ta_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_raw_stats/tennis_abstract_raw_20250812_010121.parquet')
    if ta_file.exists():
        df = pd.read_parquet(ta_file)
        print(f"Loaded TA data: {len(df)} records")
        return df
    return pd.DataFrame()

def create_ta_lookup(ta_data):
    """Create lookup dict for TA data by date + player"""
    lookup = {}
    
    if len(ta_data) == 0:
        return lookup
    
    ta_data['date'] = pd.to_datetime(ta_data['date'], errors='coerce')
    
    for _, row in ta_data.iterrows():
        if pd.notna(row['date']):
            date_str = row['date'].strftime('%Y-%m-%d')
            player_cleaned = clean_ta_name(row.get('player', ''))
            key = f"{date_str}_{player_cleaned}"
            lookup[key] = row.to_dict()
    
    print(f"TA lookup created: {len(lookup)} player-date records")
    return lookup

def clean_player_name(name):
    """Clean player name for matching"""
    if pd.isna(name):
        return ""
    
    # Convert to string and clean
    name_str = str(name).strip()
    
    # Handle API format "M. Arnaldi" -> "arnaldi_m"
    if '. ' in name_str:
        parts = name_str.split('. ', 1)
        if len(parts) == 2:
            first_initial = parts[0].strip().lower()
            last_name = parts[1].strip().lower()
            # Clean last name (handle multi-word like "Van De Zandschulp")
            last_name = last_name.replace(' ', '_')
            return f"{last_name}_{first_initial}"
    
    # Handle regular format "John Smith" -> "smith_j"  
    parts = name_str.lower().split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        return f"{last}_{first[0]}"
    
    # Single name
    return name_str.lower().replace(' ', '_')

def clean_ta_name(name):
    """TA names are already in firstname_lastname format, convert to lastname_f"""
    if pd.isna(name):
        return ""
    
    name_str = str(name).strip().lower()
    
    if '_' in name_str:
        parts = name_str.split('_', 1)
        if len(parts) == 2:
            first = parts[0]
            last = parts[1]
            return f"{last}_{first[0]}"
    
    return name_str

def merge_api_ta_matches(api_data, ta_lookup):
    """Apply intelligent merger to singles matches only"""
    
    # Filter for singles matches
    singles_mask = api_data['match_type'].str.contains('Singles', na=False)
    singles_data = api_data[singles_mask].copy()
    
    print(f"Processing {len(singles_data)} singles matches (filtered from {len(api_data)} total)")
    
    merger = IntelligentDataMerger()
    merged_matches = []
    
    enhancement_count = 0
    
    for _, api_row in singles_data.iterrows():
        try:
            date_str = api_row['date']
            
            # Clean player names for matching
            p1_clean = clean_player_name(api_row.get('player1', ''))
            p2_clean = clean_player_name(api_row.get('player2', ''))
            
            # Look for TA data for each player
            p1_ta_key = f"{date_str}_{p1_clean}"
            p2_ta_key = f"{date_str}_{p2_clean}"
            
            p1_ta = ta_lookup.get(p1_ta_key)
            p2_ta = ta_lookup.get(p2_ta_key)
            
            # Convert API row to fixture format for merger
            api_fixture = {
                'date': date_str,
                'tournament': api_row.get('tournament'),
                'event_key': api_row.get('event_key'),
                'round': api_row.get('round'),
                'player1': api_row.get('player1'),
                'player2': api_row.get('player2'),
                'final_result': api_row.get('final_result'),
                'winner': api_row.get('winner'),
                'statistics': []  # Will be populated if detailed stats available
            }
            
            # Create merged records for each player
            p1_merged = merger.merge_api_and_ta_intelligently(api_fixture, p1_ta)
            p2_merged = merger.merge_api_and_ta_intelligently(api_fixture, p2_ta)
            
            # Combine into master match record
            match_record = {
                'composite_id': f"{date_str}-{api_row.get('tournament', 'unknown').lower()}-{p1_clean}-{p2_clean}",
                'api_event_key': api_row.get('event_key'),
                'date': date_str,
                'tournament': api_row.get('tournament'),
                'round': api_row.get('round'),
                'match_type': api_row.get('match_type'),
                'final_result': api_row.get('final_result'),
                'winner': api_row.get('winner'),
                'status': api_row.get('status'),
                
                # Player info
                'p1_name': p1_clean,
                'p2_name': p2_clean,
                'p1_display_name': api_row.get('player1'),
                'p2_display_name': api_row.get('player2'),
                
                # Data source info
                'data_source': 'api_ta_merged',
                'p1_has_ta': bool(p1_ta),
                'p2_has_ta': bool(p2_ta),
                'post_june_match': True,
            }
            
            # Add player stats with prefixes
            for prefix, player_data in [('p1', p1_merged), ('p2', p2_merged)]:
                for field, value in player_data.items():
                    if field not in ['date', 'tournament', 'event_key', 'round', 'player1', 'player2']:
                        match_record[f'{prefix}_{field}'] = value
            
            # Overall quality score
            p1_quality = p1_merged.get('data_completeness', {}).get('quality_score', 0.6)
            p2_quality = p2_merged.get('data_completeness', {}).get('quality_score', 0.6)
            match_record['overall_quality_score'] = (p1_quality + p2_quality) / 2
            
            merged_matches.append(match_record)
            
            if p1_ta or p2_ta:
                enhancement_count += 1
            
        except Exception as e:
            print(f"Error processing match {api_row.get('event_key')}: {e}")
            continue
    
    print(f"Processed {len(merged_matches)} matches")
    print(f"Enhanced with TA data: {enhancement_count} matches ({enhancement_count/len(merged_matches)*100:.1f}%)")
    
    return merged_matches

def save_merged_data(merged_matches):
    """Save merged data to files"""
    
    # Convert to DataFrame
    df = pd.DataFrame(merged_matches)
    
    # Save as CSV (large file - parquet recommended)
    csv_file = Path('/Users/danielkim/Desktop/t3nn1s/merged_api_ta_FULL.csv')
    df.to_csv(csv_file, index=False)
    
    # Save as Parquet (primary format)
    parquet_file = Path('/Users/danielkim/Desktop/t3nn1s/merged_api_ta_FULL.parquet')
    df.to_parquet(parquet_file)
    
    print(f"\nSaved merged data:")
    print(f"  CSV: {csv_file}")
    print(f"  Parquet: {parquet_file}")
    
    # Summary statistics
    print(f"\n=== MERGED DATA SUMMARY ===")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique tournaments: {df['tournament'].nunique()}")
    print(f"Matches with P1 TA enhancement: {df['p1_has_ta'].sum()}")
    print(f"Matches with P2 TA enhancement: {df['p2_has_ta'].sum()}")
    print(f"Average quality score: {df['overall_quality_score'].mean():.3f}")
    
    # Show quality distribution
    quality_bins = pd.cut(df['overall_quality_score'], bins=[0, 0.7, 0.9, 1.0], labels=['Low', 'Medium', 'High'])
    print(f"\nQuality distribution:")
    print(quality_bins.value_counts())
    
    # Sample enhanced matches
    enhanced = df[(df['p1_has_ta']) | (df['p2_has_ta'])]
    if len(enhanced) > 0:
        print(f"\n=== SAMPLE ENHANCED MATCH ===")
        sample = enhanced.iloc[0]
        print(f"Date: {sample['date']}")
        print(f"Tournament: {sample['tournament']}")
        print(f"Players: {sample['p1_display_name']} vs {sample['p2_display_name']}")
        print(f"P1 has TA: {sample['p1_has_ta']}")
        print(f"P2 has TA: {sample['p2_has_ta']}")
        print(f"Quality score: {sample['overall_quality_score']:.3f}")
        
        # Show some enhanced fields
        enhanced_fields = [col for col in sample.index if '_ta_' in col and pd.notna(sample[col])][:5]
        if enhanced_fields:
            print(f"Enhanced fields: {enhanced_fields}")
    
    return df

def main():
    """Main execution"""
    print("=== APPLYING INTELLIGENT API + TA MERGER ===")
    
    # Load data
    api_data = load_api_data()
    ta_data = load_ta_data()
    
    if len(api_data) == 0:
        print("No API data available")
        return
    
    # Create TA lookup
    ta_lookup = create_ta_lookup(ta_data)
    
    # Apply intelligent merger
    merged_matches = merge_api_ta_matches(api_data, ta_lookup)
    
    # Save results
    df = save_merged_data(merged_matches)
    
    print("\n✅ Intelligent merge complete!")
    return df

if __name__ == "__main__":
    df = main()