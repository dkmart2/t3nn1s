#!/usr/bin/env python3
"""
Build Complete Master Database with Real Post-June Data
Combines Jeff + TA (pre-June) with API + TA (post-June)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json

def load_existing_master_data():
    """Load existing master database (pre-June 2025)"""
    master_file = Path('/Users/danielkim/Desktop/t3nn1s/master_database/complete_master_database.parquet')
    
    if master_file.exists():
        df = pd.read_parquet(master_file)
        print(f"Loaded existing master database: {len(df):,} matches")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    else:
        print("No existing master database found")
        return pd.DataFrame()

def load_post_june_merged_data():
    """Load post-June merged API + TA data"""
    merged_file = Path('/Users/danielkim/Desktop/t3nn1s/merged_api_ta_FULL.parquet')
    
    if merged_file.exists():
        df = pd.read_parquet(merged_file)
        print(f"Loaded post-June merged data: {len(df):,} matches")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    else:
        print("No post-June merged data found")
        return pd.DataFrame()

def harmonize_schemas(pre_june_df, post_june_df):
    """Harmonize column schemas between pre and post June data"""
    
    print("\n=== HARMONIZING SCHEMAS ===")
    
    # Get column sets
    pre_cols = set(pre_june_df.columns)
    post_cols = set(post_june_df.columns)
    
    print(f"Pre-June columns: {len(pre_cols)}")
    print(f"Post-June columns: {len(post_cols)}")
    
    # Find overlapping and unique columns
    common_cols = pre_cols & post_cols
    pre_only = pre_cols - post_cols
    post_only = post_cols - pre_cols
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Pre-June only: {len(pre_only)}")
    print(f"Post-June only: {len(post_only)}")
    
    # Create unified schema
    all_columns = pre_cols | post_cols
    
    # Add missing columns to each DataFrame
    for col in post_only:
        if col not in pre_june_df.columns:
            pre_june_df[col] = None
    
    for col in pre_only:
        if col not in post_june_df.columns:
            post_june_df[col] = None
    
    # Ensure consistent data types for common columns
    for col in common_cols:
        if col in ['date']:
            pre_june_df[col] = pd.to_datetime(pre_june_df[col], errors='coerce')
            post_june_df[col] = pd.to_datetime(post_june_df[col], errors='coerce')
    
    print(f"Unified schema: {len(all_columns)} columns")
    
    return pre_june_df, post_june_df, all_columns

def combine_databases(pre_june_df, post_june_df):
    """Combine pre and post June databases"""
    
    print("\n=== COMBINING DATABASES ===")
    
    # Harmonize schemas
    pre_harmonized, post_harmonized, all_columns = harmonize_schemas(pre_june_df, post_june_df)
    
    # Reorder columns to match
    column_order = sorted(all_columns)
    pre_harmonized = pre_harmonized[column_order]
    post_harmonized = post_harmonized[column_order]
    
    # Combine
    combined_df = pd.concat([pre_harmonized, post_harmonized], ignore_index=True)
    
    print(f"Combined database: {len(combined_df):,} matches")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Add data source indicators
    combined_df['data_era'] = combined_df['date'].apply(
        lambda x: 'pre_june_2025' if x <= pd.Timestamp('2025-06-10') else 'post_june_2025'
    )
    
    return combined_df

def analyze_combined_database(df):
    """Analyze the combined database"""
    
    print("\n=== COMBINED DATABASE ANALYSIS ===")
    
    # Basic stats
    print(f"Total matches: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total columns: {len(df.columns)}")
    
    # Era breakdown
    era_counts = df['data_era'].value_counts()
    print(f"\nData era breakdown:")
    for era, count in era_counts.items():
        print(f"  {era}: {count:,} matches")
    
    # Data source analysis
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        print(f"\nData sources:")
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} matches")
    
    # Quality analysis for post-June data
    post_june = df[df['data_era'] == 'post_june_2025']
    if len(post_june) > 0 and 'overall_quality_score' in post_june.columns:
        quality_stats = post_june['overall_quality_score'].describe()
        print(f"\nPost-June quality scores:")
        print(f"  Mean: {quality_stats['mean']:.3f}")
        print(f"  Min: {quality_stats['min']:.3f}")
        print(f"  Max: {quality_stats['max']:.3f}")
        
        # Enhanced matches
        if 'p1_has_ta' in post_june.columns:
            p1_enhanced = post_june['p1_has_ta'].sum()
            p2_enhanced = post_june['p2_has_ta'].sum()
            total_enhanced = ((post_june['p1_has_ta']) | (post_june['p2_has_ta'])).sum()
            print(f"\nPost-June TA enhancements:")
            print(f"  P1 enhanced: {p1_enhanced}")
            print(f"  P2 enhanced: {p2_enhanced}")
            print(f"  Matches with any enhancement: {total_enhanced}")
            print(f"  Enhancement rate: {total_enhanced/len(post_june)*100:.1f}%")
    
    # Column type analysis
    jeff_cols = [c for c in df.columns if 'jeff' in c.lower()]
    ta_cols = [c for c in df.columns if '_ta_' in c.lower()]
    api_cols = [c for c in df.columns if 'api' in c.lower() or 'event_key' in c]
    
    print(f"\nColumn categories:")
    print(f"  Jeff columns: {len(jeff_cols)}")
    print(f"  TA columns: {len(ta_cols)}")
    print(f"  API columns: {len(api_cols)}")
    
    return df

def save_complete_database(df):
    """Save the complete combined database"""
    
    print("\n=== SAVING COMPLETE DATABASE ===")
    
    # Create output directory
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/complete_master_database')
    output_dir.mkdir(exist_ok=True)
    
    # Save complete database
    complete_file = output_dir / 'complete_master_database_with_post_june.parquet'
    df.to_parquet(complete_file, engine='pyarrow', compression='gzip')
    print(f"Saved complete database: {complete_file}")
    
    # Save CSV sample (first 1000 rows)
    sample_file = output_dir / 'sample_1000_matches.csv'
    df.head(1000).to_csv(sample_file, index=False)
    print(f"Saved sample CSV: {sample_file}")
    
    # Save by era
    pre_june = df[df['data_era'] == 'pre_june_2025']
    post_june = df[df['data_era'] == 'post_june_2025']
    
    if len(pre_june) > 0:
        pre_file = output_dir / 'pre_june_2025.parquet'
        pre_june.to_parquet(pre_file)
        print(f"Saved pre-June data: {pre_file}")
    
    if len(post_june) > 0:
        post_file = output_dir / 'post_june_2025.parquet'
        post_june.to_parquet(post_file)
        print(f"Saved post-June data: {post_file}")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_matches': len(df),
        'date_range': {
            'min': df['date'].min().isoformat(),
            'max': df['date'].max().isoformat()
        },
        'columns': len(df.columns),
        'data_eras': {
            'pre_june_2025': len(pre_june),
            'post_june_2025': len(post_june)
        },
        'enhancements': {
            'post_june_ta_enhanced': int(((post_june.get('p1_has_ta', pd.Series([False]))) | 
                                         (post_june.get('p2_has_ta', pd.Series([False])))).sum()),
            'average_quality_score': float(post_june.get('overall_quality_score', pd.Series([0.6])).mean())
        }
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    
    return complete_file

def main():
    """Main execution"""
    
    print("=== BUILDING COMPLETE MASTER DATABASE ===")
    
    # Load existing data
    pre_june_df = load_existing_master_data()
    post_june_df = load_post_june_merged_data()
    
    if len(pre_june_df) == 0 and len(post_june_df) == 0:
        print("No data to combine!")
        return
    
    if len(post_june_df) == 0:
        print("No post-June data - using existing database only")
        combined_df = pre_june_df
    elif len(pre_june_df) == 0:
        print("No pre-June data - using post-June data only")
        combined_df = post_june_df
    else:
        # Combine both datasets
        combined_df = combine_databases(pre_june_df, post_june_df)
    
    # Analyze combined database
    analyzed_df = analyze_combined_database(combined_df)
    
    # Save complete database
    complete_file = save_complete_database(analyzed_df)
    
    print(f"\n🎾 COMPLETE MASTER DATABASE BUILT!")
    print(f"📁 Saved to: {complete_file}")
    print(f"📊 Total matches: {len(analyzed_df):,}")
    
    return analyzed_df

if __name__ == "__main__":
    df = main()