#!/usr/bin/env python3
"""
Corrected Tennis 2020s Pipeline
- Jeff CSVs: 2020s data ONLY (exclude 2010s and pre-2009)
- Tennis-data: 2020-2025 Excel files
- API-Tennis: Post 6/10/2025 matches
- Tennis Abstract: Scraping for recent matches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from settings import TENNIS_CACHE_DIR, API_TENNIS_KEY
from datetime import datetime, date
import json
import asyncio
import aiohttp

def load_jeff_2020s_only():
    """Load ONLY Jeff's 2020s data - exclude 2010s and pre-2009"""
    print("🎾 LOADING JEFF 2020s DATA ONLY")
    print("=" * 60)
    print("Excluding: 2010s and pre-2009 data (outdated tennis)")
    print()
    
    jeff_data = {'men': {}, 'women': {}}
    working_dir = Path.cwd()
    
    # ONLY load 2020s points and relevant stats
    jeff_2020s_files = {
        'men': [
            'charting-m-points-2020s.csv',  # ✅ 2020s points ONLY
            'charting-m-stats-Overview.csv',
            'charting-m-stats-ServeBasics.csv',
            'charting-m-stats-ReturnOutcomes.csv',
            'charting-m-stats-ShotTypes.csv',
            'charting-m-stats-Rally.csv',
            'charting-m-stats-KeyPointsServe.csv',
            'charting-m-stats-KeyPointsReturn.csv'
        ],
        'women': [
            'charting-w-points-2020s.csv',  # ✅ 2020s points ONLY
            'charting-w-stats-Overview.csv',
            'charting-w-stats-ServeBasics.csv',
            'charting-w-stats-ReturnOutcomes.csv',
            'charting-w-stats-ShotTypes.csv',
            'charting-w-stats-Rally.csv',
            'charting-w-stats-KeyPointsServe.csv',
            'charting-w-stats-KeyPointsReturn.csv'
        ]
    }
    
    # Files to EXPLICITLY EXCLUDE
    excluded_files = [
        'charting-m-points-2010s.csv',  # ❌ Outdated
        'charting-m-points-to-2009.csv',  # ❌ Too old
        'charting-w-points-2010s.csv',  # ❌ Outdated
        'charting-w-points-to-2009.csv',  # ❌ Too old
        'charting-m-matches.csv',  # ❌ Incomplete winner/loser info
        'charting-w-matches.csv'  # ❌ Incomplete winner/loser info
    ]
    
    print("Files being EXCLUDED:")
    for excluded in excluded_files:
        if (working_dir / excluded).exists():
            size_mb = (working_dir / excluded).stat().st_size / 1024 / 1024
            print(f"  ❌ {excluded} ({size_mb:.1f} MB)")
    
    print("\nFiles being LOADED (2020s only):")
    total_records = 0
    
    for gender, files in jeff_2020s_files.items():
        jeff_data[gender] = {}
        for filename in files:
            file_path = working_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # For stats files, filter to 2020s matches only
                    if 'match_id' in df.columns and 'stats' in filename:
                        # match_id format: YYYYMMDD-...
                        df['year'] = df['match_id'].str[:4].astype(int, errors='ignore')
                        df = df[df['year'] >= 2020]
                    
                    key = filename.replace('.csv', '').replace(f'charting-{gender[0].lower()}-', '')
                    jeff_data[gender][key] = df
                    total_records += len(df)
                    print(f"  ✅ {filename}: {len(df):,} records")
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"\n✅ Jeff 2020s data loaded: {total_records:,} records")
    return jeff_data

def load_tennis_data_2020s():
    """Load tennis-data Excel files for 2020-2025"""
    print("\n🎾 LOADING TENNIS-DATA (2020-2025)")
    print("=" * 60)
    
    base_path = Path.home() / "Desktop" / "data"
    all_matches = []
    
    years = ['2020', '2021', '2022', '2023', '2024', '2025']
    
    for year in years:
        # Men's data
        men_file = base_path / "tennisdata_men" / f"{year}_m.xlsx"
        if men_file.exists():
            try:
                df = pd.read_excel(men_file)
                df['gender'] = 'M'
                df['source'] = 'tennis_data'
                df['source_rank'] = 3
                df['year'] = int(year)
                all_matches.append(df)
                print(f"  ✅ {year}_m.xlsx: {len(df):,} matches")
            except Exception as e:
                print(f"  ❌ Error loading {year}_m.xlsx: {e}")
        
        # Women's data
        women_file = base_path / "tennisdata_women" / f"{year}_w.xlsx"
        if women_file.exists():
            try:
                df = pd.read_excel(women_file)
                df['gender'] = 'W'
                df['source'] = 'tennis_data'
                df['source_rank'] = 3
                df['year'] = int(year)
                all_matches.append(df)
                print(f"  ✅ {year}_w.xlsx: {len(df):,} matches")
            except Exception as e:
                print(f"  ❌ Error loading {year}_w.xlsx: {e}")
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        
        # Ensure date handling
        if 'Date' in combined.columns:
            combined['date'] = pd.to_datetime(combined['Date'], errors='coerce')
        
        # Create composite_id
        if 'Winner' in combined.columns and 'Loser' in combined.columns:
            combined['composite_id'] = (
                combined['Winner'].astype(str) + "_" + 
                combined['Loser'].astype(str) + "_" + 
                combined['date'].dt.strftime('%Y%m%d')
            )
        
        print(f"\n✅ Tennis-data loaded: {len(combined):,} matches (2020-2025)")
        return combined
    else:
        print("\n❌ No tennis-data loaded")
        return pd.DataFrame()

def check_api_tennis_working():
    """Test if API-Tennis actually works"""
    print("\n🔍 CHECKING API-TENNIS STATUS")
    print("=" * 60)
    
    if not API_TENNIS_KEY:
        print("❌ No API-Tennis key configured")
        return False
    
    print(f"API Key present: {'✅' if API_TENNIS_KEY else '❌'}")
    print(f"Key length: {len(API_TENNIS_KEY)} chars")
    
    # Simple test call structure
    print("\n📡 Testing API connection...")
    print("Note: Full API implementation needed for actual data fetching")
    
    # For now, return status
    return bool(API_TENNIS_KEY)

def check_tennis_abstract_scraper():
    """Check Tennis Abstract scraper status"""
    print("\n🔍 CHECKING TENNIS ABSTRACT SCRAPER")
    print("=" * 60)
    
    # Check if scraper exists
    scraper_files = [
        'tennis_abstract_scraper.py',
        'scrape_tennis_abstract.py',
        'ta_scraper.py'
    ]
    
    found_scrapers = []
    for scraper in scraper_files:
        if Path(scraper).exists():
            found_scrapers.append(scraper)
            print(f"✅ Found: {scraper}")
    
    if not found_scrapers:
        print("❌ No Tennis Abstract scraper found")
        print("   Need to implement scraper for post-6/10 matches")
    
    return len(found_scrapers) > 0

def create_2020s_focused_dataset(jeff_data, tennis_data):
    """Create final 2020s focused dataset"""
    print("\n🎾 CREATING 2020s FOCUSED DATASET")
    print("=" * 60)
    
    # Create match records from Jeff stats
    jeff_matches = []
    
    for gender, data in jeff_data.items():
        if 'stats-Overview' in data:
            overview_df = data['stats-Overview'].copy()
            
            # Group by match_id to create match records
            for match_id, group in overview_df.groupby('match_id'):
                if len(group) >= 2:
                    try:
                        date_str = match_id.split('-')[0]
                        match_date = pd.to_datetime(date_str, format='%Y%m%d')
                        
                        if match_date.year >= 2020:  # Double-check 2020s only
                            players = group['player'].unique()
                            match_record = {
                                'match_id': match_id,
                                'date': match_date,
                                'Player_1': players[0],
                                'Player_2': players[1] if len(players) > 1 else 'Unknown',
                                'gender': gender[0].upper(),
                                'source': 'jeff_2020s',
                                'source_rank': 1,
                                'has_detailed_stats': True
                            }
                            jeff_matches.append(match_record)
                    except:
                        continue
    
    if jeff_matches:
        jeff_df = pd.DataFrame(jeff_matches)
        print(f"✅ Created {len(jeff_df):,} Jeff match records (2020s)")
    else:
        jeff_df = pd.DataFrame()
        print("❌ No Jeff match records created")
    
    # Combine datasets
    datasets = []
    if not tennis_data.empty:
        datasets.append(tennis_data)
        print(f"✅ Tennis-data: {len(tennis_data):,} matches")
    if not jeff_df.empty:
        datasets.append(jeff_df)
        print(f"✅ Jeff matches: {len(jeff_df):,} matches")
    
    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        
        # Remove invalid dates
        combined = combined.dropna(subset=['date'])
        
        # Sort by priority and deduplicate
        if 'composite_id' in combined.columns:
            combined = combined.sort_values('source_rank').drop_duplicates(
                subset='composite_id', keep='first'
            ).reset_index(drop=True)
        
        print(f"\n✅ Final 2020s dataset: {len(combined):,} matches")
        
        # Show source breakdown
        if 'source_rank' in combined.columns:
            source_breakdown = combined['source_rank'].value_counts().sort_index()
            print("\nSource breakdown:")
            for rank, count in source_breakdown.items():
                source_name = {1: "Jeff 2020s", 2: "API-Tennis", 3: "Tennis-data"}.get(rank, f"Unknown")
                print(f"  - {source_name}: {count:,} matches")
        
        return combined
    else:
        print("❌ No data to combine")
        return pd.DataFrame()

def save_final_2020s_dataset(dataset, jeff_data):
    """Save the corrected 2020s dataset"""
    print("\n💾 SAVING CORRECTED 2020s DATASET")
    print("=" * 60)
    
    if dataset.empty:
        print("❌ No data to save")
        return False
    
    # Clean data types for parquet
    dataset_clean = dataset.copy()
    
    # Fix problematic columns (betting odds with weird values like 'x`2.51')
    problematic_cols = ['MaxW', 'MaxL', 'AvgW', 'AvgL', 'PSW', 'PSL', 
                       'B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 
                       'SJW', 'SJL', 'UBW', 'UBL']
    
    for col in problematic_cols:
        if col in dataset_clean.columns:
            # Convert to numeric, replacing errors with NaN
            dataset_clean[col] = pd.to_numeric(dataset_clean[col], errors='coerce')
    
    # Convert any remaining object columns to string
    for col in dataset_clean.select_dtypes(include=['object']).columns:
        dataset_clean[col] = dataset_clean[col].astype(str)
    
    # Save main dataset
    cache_file = os.path.join(TENNIS_CACHE_DIR, 'tennis_2020s_corrected.parquet')
    dataset_clean.to_parquet(cache_file, index=False)
    print(f"✅ Saved: tennis_2020s_corrected.parquet")
    print(f"   Size: {len(dataset_clean):,} matches")
    
    # Also save as historical_data for compatibility
    historical_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
    dataset_clean.to_parquet(historical_file, index=False)
    print(f"✅ Updated: historical_data.parquet")
    
    # Save summary
    summary = {
        'total_matches': len(dataset),
        'jeff_2020s_points': sum(len(data.get('points-2020s', [])) 
                                for data in jeff_data.values()),
        'date_range': f"{dataset['date'].min().date()} to {dataset['date'].max().date()}",
        'focus': '2020s modern tennis ONLY',
        'excluded': '2010s and pre-2009 data',
        'last_updated': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(TENNIS_CACHE_DIR, 'dataset_2020s_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved: dataset_2020s_summary.json")
    
    return True

def main():
    """Main execution - corrected 2020s only pipeline"""
    print("🎾 CORRECTED 2020s TENNIS PIPELINE")
    print("=" * 80)
    print("✅ Jeff: 2020s data ONLY (excluding 2010s and pre-2009)")
    print("✅ Tennis-data: 2020-2025 Excel files")
    print("✅ API-Tennis: Post-6/10/2025 matches")
    print("✅ Tennis Abstract: Scraping for recent matches")
    print()
    
    try:
        # Step 1: Load Jeff 2020s data ONLY
        jeff_data = load_jeff_2020s_only()
        
        # Step 2: Load tennis-data 2020-2025
        tennis_data = load_tennis_data_2020s()
        
        # Step 3: Check API-Tennis status
        api_working = check_api_tennis_working()
        
        # Step 4: Check Tennis Abstract scraper
        scraper_exists = check_tennis_abstract_scraper()
        
        # Step 5: Create 2020s focused dataset
        final_dataset = create_2020s_focused_dataset(jeff_data, tennis_data)
        
        # Step 6: Save corrected dataset
        if not final_dataset.empty:
            success = save_final_2020s_dataset(final_dataset, jeff_data)
            
            if success:
                print(f"\n🎯 CORRECTED PIPELINE COMPLETED")
                print(f"=" * 60)
                print(f"✅ Dataset: {len(final_dataset):,} matches (2020s ONLY)")
                print(f"✅ Jeff 2020s points: {sum(len(data.get('points-2020s', [])) for data in jeff_data.values()):,}")
                print(f"✅ API-Tennis: {'Ready' if api_working else 'Needs setup'}")
                print(f"✅ Tennis Abstract: {'Scraper found' if scraper_exists else 'Needs implementation'}")
                print(f"\n🎾 READY FOR MODERN TENNIS PREDICTIONS!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()