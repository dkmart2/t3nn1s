#!/usr/bin/env python3
"""
Scrape all known Tennis Abstract URLs from previous work
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

def scrape_all_known_tennis_abstract():
    """Scrape all 111 known Tennis Abstract URLs"""
    print("🎾 SCRAPING ALL KNOWN TENNIS ABSTRACT MATCHES")
    print("="*70)
    
    # Load known URLs from cache
    urls_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
    
    with open(urls_file, 'r') as f:
        known_urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(known_urls)} known Tennis Abstract URLs")
    
    # Filter for recent matches (post 6/10/2025)
    recent_urls = []
    cutoff_date = datetime(2025, 6, 10)
    
    for url in known_urls:
        try:
            match_file = url.split('/')[-1]
            date_str = match_file[:8]
            match_date = datetime.strptime(date_str, '%Y%m%d')
            if match_date >= cutoff_date:
                recent_urls.append(url)
        except:
            continue
    
    print(f"Filtered to {len(recent_urls)} matches post-6/10/2025")
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    successful_urls = []
    
    cache_dir = Path('/Users/danielkim/tennis_data/cache') / 'tennis_abstract_comprehensive'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, url in enumerate(recent_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\n[{i}/{len(recent_urls)}] Scraping: {match_name}")
        
        try:
            # Get comprehensive match data
            records = scraper.scrape_comprehensive_match_data(url)
            
            if records and len(records) > 0:
                print(f"  ✅ Extracted {len(records)} records")
                successful_urls.append(url)
                
                # Parse match metadata from URL
                url_parts = url.split('/')[-1].replace('.html', '').split('-')
                if len(url_parts) >= 5:
                    date_str = url_parts[0]
                    gender = url_parts[1]
                    tournament = url_parts[2]
                    round_str = url_parts[3]
                    
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    
                    # Add metadata to all records
                    for record in records:
                        record['match_date'] = match_date
                        record['gender'] = gender
                        record['tournament'] = tournament
                        record['round'] = round_str
                        record['source'] = 'tennis_abstract'
                        record['source_rank'] = 1
                        record['has_detailed_stats'] = True
                        record['has_point_data'] = True
                        record['url'] = url
                    
                    all_records.extend(records)
                    
                    # Create match summary record
                    players = list(set(record.get('Player_canonical', record.get('Player', 'Unknown')) for record in records[:10]))[:2]
                    if len(players) >= 2:
                        match_record = {
                            'date': match_date,
                            'Player_1': players[0],
                            'Player_2': players[1],
                            'gender': gender,
                            'tournament': tournament,
                            'round': round_str,
                            'source': 'tennis_abstract',
                            'source_rank': 1,
                            'has_detailed_stats': True,
                            'has_point_data': True,
                            'comprehensive_records': len(records),
                            'url': url,
                            'composite_id': f"{players[0]}_{players[1]}_{date_str}"
                        }
                        all_matches.append(match_record)
            else:
                print(f"  ❌ No records extracted")
                
        except Exception as e:
            print(f"  💥 Error: {str(e)[:100]}")
        
        # Be respectful to Tennis Abstract
        if i % 10 == 0:
            print(f"  ⏸️  Pausing after {i} matches...")
            time.sleep(5)
        else:
            time.sleep(2)
    
    # Save results
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        records_df = pd.DataFrame(all_records)
        
        print(f"\n✅ COMPREHENSIVE TENNIS ABSTRACT SCRAPING COMPLETED!")
        print(f"   Successful URLs: {len(successful_urls)}")
        print(f"   Matches: {len(matches_df)}")
        print(f"   Detailed records: {len(records_df):,}")
        
        # Save both datasets
        matches_file = cache_dir / f'ta_matches_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        records_file = cache_dir / f'ta_records_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        
        matches_df.to_parquet(matches_file, index=False)
        records_df.to_parquet(records_file, index=False)
        
        print(f"💾 Cached TA matches: {matches_file}")
        print(f"💾 Cached TA records: {records_file}")
        
        # Show tournament breakdown
        if 'tournament' in matches_df.columns:
            print(f"\n📊 Tournament breakdown:")
            tournament_counts = matches_df['tournament'].value_counts()
            for tournament, count in tournament_counts.items():
                print(f"  - {tournament}: {count}")
        
        # Show date range
        if 'date' in matches_df.columns:
            print(f"\n📅 Date range:")
            print(f"   Earliest: {matches_df['date'].min()}")
            print(f"   Latest: {matches_df['date'].max()}")
        
        return matches_df, len(all_records)
    else:
        print(f"\n❌ No Tennis Abstract matches scraped successfully")
        return pd.DataFrame(), 0

if __name__ == "__main__":
    matches_df, total_records = scrape_all_known_tennis_abstract()
    
    if not matches_df.empty:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Scraped {len(matches_df)} Tennis Abstract matches")
        print(f"✅ Extracted {total_records:,} detailed records")
        print(f"🚀 Ready for comprehensive integration!")
    else:
        print(f"\n⚠️  Failed to scrape Tennis Abstract matches")