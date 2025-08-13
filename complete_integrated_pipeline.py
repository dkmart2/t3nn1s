#!/usr/bin/env python3
"""
Complete Integrated Tennis Pipeline
Combines all data sources:
1. Jeff 2020s data (excluding outdated 2010s and pre-2009)
2. Tennis-data Excel files (2020-2025)
3. API-Tennis for recent matches
4. Tennis Abstract scraping capability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import asyncio
from datetime import datetime, date, timedelta
import json
from settings import TENNIS_CACHE_DIR, API_TENNIS_KEY
import logging

# Import our modules
from tennis_2020s_only_pipeline import load_jeff_2020s_only, load_tennis_data_2020s
from api_tennis_integration import fetch_recent_api_tennis_data
from tennis_abstract_scraper import TennisAbstractScraper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_all_data_sources():
    """Integrate data from all sources with proper prioritization"""
    print("🎾 COMPLETE INTEGRATED TENNIS PIPELINE")
    print("="*70)
    print("Data Sources:")
    print("1. Jeff 2020s CSV data (highest priority)")
    print("2. Tennis-data Excel files (2020-2025)")
    print("3. API-Tennis (recent matches)")
    print("4. Tennis Abstract (supplementary)")
    print()
    
    all_data = []
    data_summary = {}
    
    # Step 1: Load Jeff 2020s data (source_rank=1)
    print("📚 STEP 1: Loading Jeff 2020s Data")
    print("-"*50)
    jeff_data = load_jeff_2020s_only()
    
    # Create match records from Jeff data
    jeff_matches = []
    jeff_point_count = 0
    
    for gender, datasets in jeff_data.items():
        # Count points
        if 'points-2020s' in datasets:
            jeff_point_count += len(datasets['points-2020s'])
        
        # Create match records from overview stats
        if 'stats-Overview' in datasets:
            overview_df = datasets['stats-Overview'].copy()
            
            for match_id, group in overview_df.groupby('match_id'):
                if len(group) >= 2:
                    try:
                        date_str = match_id.split('-')[0]
                        match_date = pd.to_datetime(date_str, format='%Y%m%d')
                        
                        players = group['player'].unique()
                        match_record = {
                            'match_id': match_id,
                            'date': match_date,
                            'Player_1': players[0],
                            'Player_2': players[1] if len(players) > 1 else 'Unknown',
                            'gender': gender[0].upper(),
                            'source': 'jeff_2020s',
                            'source_rank': 1,
                            'has_detailed_stats': True,
                            'has_point_data': True
                        }
                        jeff_matches.append(match_record)
                    except:
                        continue
    
    if jeff_matches:
        jeff_df = pd.DataFrame(jeff_matches)
        all_data.append(jeff_df)
        data_summary['Jeff 2020s'] = {
            'matches': len(jeff_df),
            'points': jeff_point_count,
            'source_rank': 1
        }
        print(f"✅ Jeff 2020s: {len(jeff_df):,} matches, {jeff_point_count:,} points")
    
    # Step 2: Load Tennis-data Excel files (source_rank=3)
    print(f"\n📊 STEP 2: Loading Tennis-Data Excel Files")
    print("-"*50)
    tennis_data = load_tennis_data_2020s()
    
    if not tennis_data.empty:
        tennis_data['source_rank'] = 3
        all_data.append(tennis_data)
        data_summary['Tennis-data'] = {
            'matches': len(tennis_data),
            'source_rank': 3
        }
        print(f"✅ Tennis-data: {len(tennis_data):,} matches")
    
    # Step 3: Fetch API-Tennis data (source_rank=2)
    print(f"\n🌐 STEP 3: Fetching API-Tennis Data")
    print("-"*50)
    
    if API_TENNIS_KEY:
        # Get recent matches from API-Tennis
        start_date = date(2025, 6, 10)  # Jeff cutoff
        end_date = date.today()
        
        try:
            api_data = asyncio.run(fetch_recent_api_tennis_data(start_date, end_date))
            
            if not api_data.empty:
                api_data['source_rank'] = 2
                all_data.append(api_data)
                data_summary['API-Tennis'] = {
                    'matches': len(api_data),
                    'source_rank': 2
                }
                print(f"✅ API-Tennis: {len(api_data):,} matches")
            else:
                print("⚠️  No API-Tennis data retrieved")
        except Exception as e:
            print(f"❌ API-Tennis error: {e}")
    else:
        print("❌ No API-Tennis key configured")
    
    # Step 4: Tennis Abstract scraping (source_rank=1)
    print(f"\n🔍 STEP 4: Tennis Abstract Scraping")
    print("-"*50)
    
    try:
        scraper = TennisAbstractScraper()
        ta_data = scraper.scrape_recent_matches(date(2025, 6, 10), date.today())
        
        if not ta_data.empty:
            ta_data['source_rank'] = 1
            all_data.append(ta_data)
            data_summary['Tennis Abstract'] = {
                'matches': len(ta_data),
                'source_rank': 1
            }
            print(f"✅ Tennis Abstract: {len(ta_data):,} matches")
        else:
            print("⚠️  No Tennis Abstract data scraped")
    except Exception as e:
        print(f"⚠️  Tennis Abstract scraping skipped: {e}")
    
    # Step 5: Combine and deduplicate
    print(f"\n🔧 STEP 5: Combining and Deduplicating Data")
    print("-"*50)
    
    if not all_data:
        print("❌ No data to combine")
        return pd.DataFrame(), data_summary
    
    # Combine all data sources
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined total: {len(combined_df):,} matches (before deduplication)")
    
    # Ensure date column
    if 'date' in combined_df.columns:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Create composite_id for deduplication
    if 'Winner' in combined_df.columns and 'Loser' in combined_df.columns:
        combined_df['composite_id'] = (
            combined_df['Winner'].astype(str) + "_" + 
            combined_df['Loser'].astype(str) + "_" + 
            combined_df['date'].dt.strftime('%Y%m%d')
        )
    elif 'Player_1' in combined_df.columns and 'Player_2' in combined_df.columns:
        # For matches without winner/loser designation
        combined_df['composite_id'] = combined_df.apply(
            lambda x: "_".join(sorted([str(x['Player_1']), str(x['Player_2'])])) + "_" + 
                     pd.to_datetime(x['date']).strftime('%Y%m%d'), axis=1
        )
    
    # Remove invalid dates
    combined_df = combined_df.dropna(subset=['date'])
    
    # Deduplicate, keeping highest priority (lowest source_rank)
    if 'composite_id' in combined_df.columns and 'source_rank' in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df.sort_values('source_rank').drop_duplicates(
            subset='composite_id', keep='first'
        ).reset_index(drop=True)
        print(f"After deduplication: {len(combined_df):,} matches ({initial_count - len(combined_df)} duplicates removed)")
    
    # Final source breakdown
    if 'source' in combined_df.columns:
        print(f"\n📊 Final Source Breakdown:")
        source_counts = combined_df['source'].value_counts()
        for source, count in source_counts.items():
            rank = combined_df[combined_df['source'] == source]['source_rank'].iloc[0] if 'source_rank' in combined_df.columns else 'N/A'
            print(f"  - {source}: {count:,} matches (rank={rank})")
    
    return combined_df, data_summary

def save_integrated_dataset(df, summary):
    """Save the fully integrated dataset"""
    print(f"\n💾 STEP 6: Saving Integrated Dataset")
    print("-"*50)
    
    if df.empty:
        print("❌ No data to save")
        return False
    
    # Clean data types for parquet
    df_clean = df.copy()
    
    # Fix problematic columns
    problematic_cols = ['MaxW', 'MaxL', 'AvgW', 'AvgL', 'PSW', 'PSL']
    for col in problematic_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert object columns to string
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str)
    
    # Save main dataset
    main_file = os.path.join(TENNIS_CACHE_DIR, 'complete_tennis_2020s.parquet')
    df_clean.to_parquet(main_file, index=False)
    print(f"✅ Saved: complete_tennis_2020s.parquet ({len(df_clean):,} matches)")
    
    # Also update historical_data.parquet
    historical_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
    df_clean.to_parquet(historical_file, index=False)
    print(f"✅ Updated: historical_data.parquet")
    
    # Save summary
    summary['total_matches'] = len(df_clean)
    summary['date_range'] = f"{df_clean['date'].min().date()} to {df_clean['date'].max().date()}"
    summary['last_updated'] = datetime.now().isoformat()
    
    summary_file = os.path.join(TENNIS_CACHE_DIR, 'complete_pipeline_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Summary saved: complete_pipeline_summary.json")
    
    return True

def show_data_coverage(df):
    """Show comprehensive data coverage analysis"""
    print(f"\n📈 DATA COVERAGE ANALYSIS")
    print("-"*50)
    
    if df.empty:
        print("No data to analyze")
        return
    
    # Date coverage
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()} ({date_range.days} days)")
        
        # Check for gaps
        jeff_cutoff = pd.Timestamp('2025-06-10')
        pre_cutoff = df[df['date'] <= jeff_cutoff]
        post_cutoff = df[df['date'] > jeff_cutoff]
        
        print(f"\nPre-6/10/2025 (Jeff coverage): {len(pre_cutoff):,} matches")
        print(f"Post-6/10/2025 (API/TA needed): {len(post_cutoff):,} matches")
    
    # Feature richness
    if 'has_detailed_stats' in df.columns:
        detailed = df['has_detailed_stats'].sum()
        print(f"\nMatches with detailed stats: {detailed:,} ({detailed/len(df)*100:.1f}%)")
    
    if 'has_point_data' in df.columns:
        points = df['has_point_data'].sum()
        print(f"Matches with point data: {points:,} ({points/len(df)*100:.1f}%)")
    
    # Gender coverage
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        print(f"\nGender breakdown:")
        for gender, count in gender_counts.items():
            print(f"  - {gender}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Year coverage
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        year_counts = df['year'].value_counts().sort_index()
        print(f"\nMatches by year:")
        for year, count in year_counts.items():
            print(f"  - {year}: {count:,}")

def main():
    """Run the complete integrated pipeline"""
    print("🚀 RUNNING COMPLETE INTEGRATED PIPELINE")
    print("="*70)
    print("Goal: Complete tennis data coverage from 2020 to today")
    print()
    
    try:
        # Integrate all data sources
        integrated_df, data_summary = integrate_all_data_sources()
        
        if integrated_df.empty:
            print("\n❌ No data integrated")
            return None
        
        # Save integrated dataset
        success = save_integrated_dataset(integrated_df, data_summary)
        
        if success:
            # Show coverage analysis
            show_data_coverage(integrated_df)
            
            print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"Total matches: {len(integrated_df):,}")
            print(f"Data sources integrated: {len(data_summary)}")
            print(f"Ready for: modeling, prediction, analysis")
            
            return integrated_df
        else:
            print(f"\n❌ Failed to save integrated dataset")
            return None
            
    except Exception as e:
        print(f"\n💥 Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()