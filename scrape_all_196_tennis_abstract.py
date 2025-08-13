#!/usr/bin/env python3
"""
Scrape ALL 196 Tennis Abstract matches from the cached URLs
Currently we only have 73 - we need to get the remaining 123 that were cached but never scraped
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path

def scrape_all_196_tennis_abstract():
    """Scrape all 196 Tennis Abstract matches from cached URLs"""
    print("🎾 SCRAPING ALL 196 TENNIS ABSTRACT MATCHES")
    print("=" * 70)
    print("Goal: Get complete Tennis Abstract dataset from all cached URLs")
    print()
    
    # Load all cached URLs
    cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
    
    try:
        with open(cache_file, 'r') as f:
            all_urls = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(all_urls)} cached URLs")
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return None, 0
    
    if len(all_urls) != 196:
        print(f"⚠️  Expected 196 URLs, found {len(all_urls)}")
    
    # Check what we already have
    existing_file = '/Users/danielkim/Desktop/t3nn1s/tennis_abstract_recent/recent_missing_records_20250811_223651.parquet'
    existing_urls = set()
    
    try:
        if Path(existing_file).exists():
            existing_data = pd.read_parquet(existing_file)
            if 'url' in existing_data.columns:
                existing_urls = set(existing_data['url'].unique())
                print(f"✅ Found {len(existing_urls)} already scraped URLs")
            else:
                print("⚠️  Existing data has no URL column")
    except Exception as e:
        print(f"⚠️  Could not load existing data: {e}")
    
    # Find URLs we still need to scrape
    urls_to_scrape = [url for url in all_urls if url not in existing_urls]
    print(f"🎯 Need to scrape {len(urls_to_scrape)} additional URLs")
    print(f"📊 Status: {len(existing_urls)} done + {len(urls_to_scrape)} todo = {len(all_urls)} total")
    
    if not urls_to_scrape:
        print("✅ All URLs already scraped!")
        # Just load and return existing data
        try:
            return pd.read_parquet(existing_file), len(existing_data)
        except:
            return pd.DataFrame(), 0
    
    print(f"\n🚀 STARTING SCRAPE OF {len(urls_to_scrape)} REMAINING MATCHES")
    print("-" * 50)
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    successful_urls = []
    
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete')
    output_dir.mkdir(exist_ok=True)
    
    for i, url in enumerate(urls_to_scrape, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\\n[{i}/{len(urls_to_scrape)}] {match_name[:70]}")
        
        try:
            records = scraper.scrape_comprehensive_match_data(url)
            
            if records and len(records) > 0:
                print(f"  ✅ {len(records):,} records")
                successful_urls.append(url)
                
                # Parse metadata from URL
                url_parts = url.split('/')[-1].replace('.html', '').split('-')
                if len(url_parts) >= 5:
                    date_str = url_parts[0]
                    gender = url_parts[1]
                    tournament = url_parts[2]
                    round_str = url_parts[3]
                    
                    try:
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    except:
                        match_date = date_str
                    
                    # Add metadata to records
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
                    
                    # Create match summary
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
                print(f"  ❌ No records")
                
        except Exception as e:
            print(f"  💥 Error: {str(e)[:70]}")
        
        # Progress update every 25 matches
        if i % 25 == 0:
            print(f"\\n📊 Progress: {i}/{len(urls_to_scrape)} ({i/len(urls_to_scrape)*100:.1f}%)")
            print(f"   Successful: {len(successful_urls)}")
            print(f"   Records: {len(all_records):,}")
        
        time.sleep(1.5)  # Be respectful
    
    # Combine with existing data
    print(f"\\n🔄 Combining with existing data...")
    final_records = all_records.copy()
    final_matches = all_matches.copy()
    
    if existing_urls:
        try:
            existing_data = pd.read_parquet(existing_file)
            existing_records = existing_data.to_dict('records')
            final_records.extend(existing_records)
            print(f"✅ Combined: {len(final_records):,} total records")
        except Exception as e:
            print(f"⚠️  Could not combine with existing data: {e}")
    
    # Save complete dataset
    if final_records:
        print(f"\\n💾 Saving complete Tennis Abstract dataset...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all records
        records_df = pd.DataFrame(final_records)
        records_file = output_dir / f'all_196_tennis_abstract_records_{timestamp}.parquet'
        records_df.to_parquet(records_file, index=False)
        print(f"📄 Records: {records_file}")
        
        # Save match summaries
        if final_matches:
            matches_df = pd.DataFrame(final_matches)
            matches_file = output_dir / f'all_196_tennis_abstract_matches_{timestamp}.parquet'
            matches_df.to_parquet(matches_file, index=False)
            print(f"📄 Matches: {matches_file}")
        
        # Summary
        unique_urls = len(set(record.get('url', '') for record in final_records))
        
        print(f"\\n🎉 TENNIS ABSTRACT SCRAPING COMPLETE!")
        print(f"✅ Total records: {len(final_records):,}")
        print(f"✅ Unique matches: {unique_urls}")
        print(f"✅ New matches scraped: {len(successful_urls)}")
        print(f"🎯 Coverage: {unique_urls}/196 URLs ({unique_urls/196*100:.1f}%)")
        
        if unique_urls == 196:
            print(f"🏆 PERFECT! All 196 Tennis Abstract matches captured!")
        else:
            print(f"⚠️  Still missing {196-unique_urls} matches")
        
        return pd.DataFrame(final_matches) if final_matches else pd.DataFrame(), len(final_records)
    
    else:
        print(f"\\n❌ No data scraped")
        return pd.DataFrame(), 0

if __name__ == "__main__":
    matches_df, total_records = scrape_all_196_tennis_abstract()
    
    if not matches_df.empty or total_records > 0:
        print(f"\\n🚀 SUCCESS!")
        print(f"Complete Tennis Abstract dataset ready for integration!")
    else:
        print(f"\\n❌ Scraping failed")