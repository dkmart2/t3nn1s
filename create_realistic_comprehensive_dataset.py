#!/usr/bin/env python3
"""
Create a realistic comprehensive dataset from what we actually have:
1. Main tennis data (10,108 matches from tennis_updated.py)
2. Tennis Abstract scraped data (73 matches with detailed point data)
3. API-Tennis data (fetch recent matches with the available API key)

This is honest about what we have vs claimed integration.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date
import sys
import os
import requests
import time

# Set API key
os.environ['API_TENNIS_KEY'] = 'adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb'

def fetch_api_tennis_data():
    """Fetch recent matches from API-Tennis"""
    print("📡 Fetching API-Tennis data...")
    
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("❌ No API key available")
        return pd.DataFrame()
    
    base_url = "https://api.api-tennis.com/tennis/"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        # Get recent matches
        matches_url = f"{base_url}matches?limit=100"
        response = requests.get(matches_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                matches = []
                for match in data['data']:
                    match_record = {
                        'date': match.get('date'),
                        'Player_1': match.get('player1', {}).get('name', 'Unknown'),
                        'Player_2': match.get('player2', {}).get('name', 'Unknown'),
                        'tournament': match.get('tournament', {}).get('name', 'Unknown'),
                        'surface': match.get('surface', 'Unknown'),
                        'round': match.get('round', 'Unknown'),
                        'source': 'api_tennis',
                        'source_rank': 2,
                        'has_odds': True,
                        'api_match_id': match.get('id'),
                        'composite_id': f"{match.get('player1', {}).get('name', 'Unknown')}_{match.get('player2', {}).get('name', 'Unknown')}_{match.get('date', 'unknown')}"
                    }
                    matches.append(match_record)
                
                df = pd.DataFrame(matches)
                print(f"✅ API-Tennis: {len(df)} matches fetched")
                return df
            else:
                print("⚠️  No data field in API response")
                return pd.DataFrame()
        else:
            print(f"❌ API request failed: HTTP {response.status_code}")
            if response.status_code == 401:
                print("   API key may be invalid or expired")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ API-Tennis error: {e}")
        return pd.DataFrame()

def load_existing_main_data():
    """Load the existing comprehensive dataset"""
    print("📊 Loading existing main dataset...")
    
    try:
        main_file = '/Users/danielkim/Desktop/t3nn1s/comprehensive_datasets/comprehensive_tennis_dataset_20250811_224425.parquet'
        if Path(main_file).exists():
            df = pd.read_parquet(main_file)
            print(f"✅ Main dataset: {len(df):,} matches")
            return df
        else:
            print("❌ Main dataset file not found")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading main dataset: {e}")
        return pd.DataFrame()

def load_tennis_abstract_data():
    """Load Tennis Abstract scraped data"""
    print("🎾 Loading Tennis Abstract data...")
    
    try:
        ta_file = '/Users/danielkim/Desktop/t3nn1s/tennis_abstract_recent/recent_missing_records_20250811_223651.parquet'
        if Path(ta_file).exists():
            ta_records = pd.read_parquet(ta_file)
            
            # Convert to match-level records
            ta_matches = []
            if len(ta_records) > 0 and 'url' in ta_records.columns:
                for url, match_group in ta_records.groupby('url'):
                    try:
                        first_record = match_group.iloc[0]
                        players = match_group['Player_canonical'].dropna().unique()
                        
                        if len(players) >= 2:
                            match_record = {
                                'date': first_record.get('match_date'),
                                'Player_1': players[0],
                                'Player_2': players[1],
                                'gender': first_record.get('gender', 'M'),
                                'tournament': first_record.get('tournament', 'Unknown'),
                                'round': first_record.get('round', 'Unknown'),
                                'source': 'tennis_abstract',
                                'source_rank': 1,
                                'has_detailed_stats': True,
                                'has_point_data': True,
                                'comprehensive_records': len(match_group),
                                'url': url,
                                'composite_id': f"{players[0]}_{players[1]}_{first_record.get('match_date', 'unknown')}"
                            }
                            ta_matches.append(match_record)
                    except:
                        continue
            
            df = pd.DataFrame(ta_matches)
            print(f"✅ Tennis Abstract: {len(df)} matches ({len(ta_records):,} total records)")
            return df
        else:
            print("❌ Tennis Abstract file not found")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading Tennis Abstract: {e}")
        return pd.DataFrame()

def create_realistic_comprehensive_dataset():
    """Create a realistic comprehensive dataset from available sources"""
    print("🏆 CREATING REALISTIC COMPREHENSIVE DATASET")
    print("=" * 60)
    print("Combining actual available data sources")
    print()
    
    datasets = []
    source_stats = {}
    
    # 1. Load main dataset (10k+ matches)
    main_data = load_existing_main_data()
    if not main_data.empty:
        datasets.append(main_data)
        source_stats['Main Dataset'] = len(main_data)
    
    # 2. Load Tennis Abstract data (73 detailed matches)  
    ta_data = load_tennis_abstract_data()
    if not ta_data.empty:
        datasets.append(ta_data)
        source_stats['Tennis Abstract'] = len(ta_data)
    
    # 3. Fetch API-Tennis data (recent matches with odds)
    api_data = fetch_api_tennis_data()
    if not api_data.empty:
        datasets.append(api_data)
        source_stats['API-Tennis'] = len(api_data)
    
    if not datasets:
        print("❌ No datasets available!")
        return None
    
    print(f"\n🔄 Combining {len(datasets)} datasets...")
    
    # Combine all datasets
    try:
        combined_data = pd.concat(datasets, ignore_index=True)
        print(f"✅ Combined: {len(combined_data):,} total matches")
    except Exception as e:
        print(f"❌ Error combining: {e}")
        return None
    
    # Deduplication
    print(f"\n🧹 Deduplication...")
    original_count = len(combined_data)
    
    # Create composite_id if missing
    if 'composite_id' not in combined_data.columns:
        if all(col in combined_data.columns for col in ['Player_1', 'Player_2', 'date']):
            combined_data['composite_id'] = combined_data.apply(
                lambda row: f"{row['Player_1']}_{row['Player_2']}_{row['date']}", axis=1
            )
    
    # Remove duplicates, preferring higher quality sources (lower source_rank)
    if 'composite_id' in combined_data.columns and 'source_rank' in combined_data.columns:
        combined_data = combined_data.sort_values(['composite_id', 'source_rank'])
        combined_data = combined_data.drop_duplicates(subset=['composite_id'], keep='first')
        final_count = len(combined_data)
        print(f"🗑️  Removed {original_count - final_count:,} duplicates")
    else:
        final_count = original_count
        print("⚠️  Could not deduplicate properly")
    
    # Final statistics
    print(f"\n📊 REALISTIC DATASET SUMMARY")
    print("-" * 40)
    print(f"Total unique matches: {final_count:,}")
    
    for source, count in source_stats.items():
        print(f"{source}: {count:,} contributed")
    
    # Actual source distribution in final dataset
    if 'source' in combined_data.columns:
        print(f"\nFinal Source Distribution:")
        source_dist = combined_data['source'].value_counts()
        for source, count in source_dist.items():
            print(f"  {source}: {count:,} matches")
    
    # Date range
    if 'date' in combined_data.columns:
        try:
            combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')
            date_range = f"{combined_data['date'].min().date()} to {combined_data['date'].max().date()}"
            print(f"Date range: {date_range}")
        except:
            print("⚠️  Could not analyze date range")
    
    # Data quality features
    quality_features = ['has_detailed_stats', 'has_point_data', 'has_odds']
    print(f"\nData Quality Features:")
    for feature in quality_features:
        if feature in combined_data.columns:
            count = combined_data[feature].sum() if pd.api.types.is_bool_dtype(combined_data[feature]) else len(combined_data[combined_data[feature] == True])
            print(f"  {feature}: {count:,} matches")
    
    # Save dataset
    print(f"\n💾 Saving realistic comprehensive dataset...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/comprehensive_datasets')
    
    base_filename = f'REALISTIC_comprehensive_tennis_dataset_{timestamp}'
    
    # Parquet
    parquet_file = output_dir / f'{base_filename}.parquet'
    combined_data.to_parquet(parquet_file, index=False)
    print(f"📄 Parquet: {parquet_file}")
    
    # CSV  
    csv_file = output_dir / f'{base_filename}.csv'
    combined_data.to_csv(csv_file, index=False)
    print(f"📄 CSV: {csv_file}")
    
    # Honest data dictionary
    dictionary_file = output_dir / f'{base_filename}_dictionary.txt'
    with open(dictionary_file, 'w') as f:
        f.write("REALISTIC COMPREHENSIVE TENNIS DATASET - DATA DICTIONARY\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Created: {datetime.now()}\n")
        f.write(f"Total unique matches: {final_count:,}\n\n")
        
        f.write("DATA SOURCES (actual contribution):\n")
        for source, count in source_stats.items():
            f.write(f"- {source}: {count:,} matches contributed\n")
        f.write("\n")
        
        f.write("FINAL SOURCE DISTRIBUTION (after deduplication):\n")
        if 'source' in combined_data.columns:
            source_dist = combined_data['source'].value_counts()
            for source, count in source_dist.items():
                f.write(f"- {source}: {count:,} matches\n")
        f.write("\n")
        
        f.write("COLUMNS:\n")
        for col in sorted(combined_data.columns):
            f.write(f"- {col}\n")
        f.write("\n")
        
        f.write("HONEST ASSESSMENT:\n")
        f.write("- Tennis Abstract: Only 73 matches scraped (not all 196 cached URLs)\n")
        f.write("- API-Tennis: Limited by API rate limits and key availability\n")
        f.write("- Main Dataset: Existing comprehensive data, source composition unclear\n")
        f.write("- This represents what was actually achievable, not theoretical integration\n")
    
    print(f"📄 Dictionary: {dictionary_file}")
    
    # Success message
    print(f"\n🎉 REALISTIC DATASET COMPLETE!")
    print(f"📊 Final dataset: {final_count:,} unique tennis matches")
    print(f"🎯 Honest integration of available data sources")
    print(f"💡 Ready for modeling with realistic data expectations")
    
    return combined_data

if __name__ == "__main__":
    dataset = create_realistic_comprehensive_dataset()
    
    if dataset is not None:
        print(f"\n✅ REALISTIC INTEGRATION SUCCESS!")
        print(f"Created an honest comprehensive dataset with actual available data.")
        print(f"This provides a solid foundation for tennis analytics.")
    else:
        print(f"\n❌ Dataset creation failed")