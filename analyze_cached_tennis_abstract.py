#!/usr/bin/env python3
"""
Analyze our cached Tennis Abstract matches to understand coverage
"""

import pandas as pd
from datetime import datetime
from collections import Counter

def analyze_cached_matches():
    """Analyze our 111 cached Tennis Abstract matches"""
    print("📊 ANALYZING CACHED TENNIS ABSTRACT MATCHES")
    print("="*70)
    
    urls_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
    
    try:
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except:
        print("❌ Cannot read cached URLs file")
        return
    
    print(f"Total cached URLs: {len(urls)}")
    
    # Parse match data
    matches_data = []
    
    for url in urls:
        filename = url.split('/')[-1].replace('.html', '')
        parts = filename.split('-')
        
        if len(parts) >= 5:
            try:
                date_str = parts[0]
                gender = parts[1]
                tournament = parts[2]
                round_name = parts[3]
                
                match_date = datetime.strptime(date_str, '%Y%m%d').date()
                
                matches_data.append({
                    'date': match_date,
                    'gender': gender,
                    'tournament': tournament,
                    'round': round_name,
                    'url': url,
                    'filename': filename
                })
            except Exception as e:
                print(f"Could not parse: {filename}")
                continue
    
    if not matches_data:
        print("❌ No valid match data parsed")
        return
    
    df = pd.DataFrame(matches_data)
    
    print(f"✅ Successfully parsed: {len(df)} matches")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Filter for post June 10, 2025
    cutoff_date = pd.to_datetime('2025-06-10').date()
    recent_df = df[df['date'] >= cutoff_date]
    
    print(f"\n📅 POST-JUNE 10 MATCHES: {len(recent_df)}")
    
    if recent_df.empty:
        print("❌ No matches found after June 10, 2025!")
        return
    
    # Tournament breakdown
    print(f"\n🏆 TOURNAMENT BREAKDOWN:")
    tournament_counts = recent_df['tournament'].value_counts()
    for tournament, count in tournament_counts.items():
        print(f"  - {tournament}: {count}")
    
    # Gender breakdown
    print(f"\n⚥ GENDER BREAKDOWN:")
    gender_counts = recent_df['gender'].value_counts()
    for gender, count in gender_counts.items():
        print(f"  - {gender}: {count}")
    
    # Monthly breakdown
    print(f"\n📊 MONTHLY BREAKDOWN:")
    recent_df['year_month'] = recent_df['date'].apply(lambda x: f"{x.year}-{x.month:02d}")
    monthly_counts = recent_df['year_month'].value_counts().sort_index()
    for month, count in monthly_counts.items():
        print(f"  - {month}: {count}")
    
    # Daily activity in recent weeks
    print(f"\n📈 RECENT DAILY ACTIVITY:")
    last_20_days = recent_df.nlargest(20, 'date')
    daily_counts = last_20_days['date'].value_counts().sort_index(ascending=False)
    for date, count in daily_counts.head(10).items():
        print(f"  - {date}: {count} matches")
    
    # Major tournaments
    major_tournaments = ['Wimbledon', 'Cincinnati', 'Montreal', 'Toronto', 'Olympics', 'Stuttgart', 'Queens_Club', 'Halle', 'Bad_Homburg']
    
    print(f"\n🎾 MAJOR TOURNAMENTS:")
    for tournament in major_tournaments:
        count = len(recent_df[recent_df['tournament'] == tournament])
        if count > 0:
            print(f"  - {tournament}: {count}")
    
    # Show latest matches
    print(f"\n🔥 LATEST 10 MATCHES:")
    latest_matches = recent_df.nlargest(10, 'date')
    for _, match in latest_matches.iterrows():
        players = '-'.join(match['filename'].split('-')[4:]).replace('.html', '')
        print(f"  - {match['date']}: {match['tournament']} {match['round']} ({players[:40]})")
    
    return len(recent_df)

if __name__ == "__main__":
    count = analyze_cached_matches()