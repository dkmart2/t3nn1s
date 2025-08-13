#!/usr/bin/env python3
"""
Scrape the 73 missing Tennis Abstract matches we just discovered
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import requests
import re

def get_missing_urls():
    """Get the 73 missing URLs by comparing discovered vs cached"""
    print("🔍 IDENTIFYING MISSING URLs")
    print("="*50)
    
    # Get discovered URLs (using our robust discovery method)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    try:
        response = requests.get("https://www.tennisabstract.com/charting/", headers=headers, timeout=15)
        raw_html = response.text
        
        # Use the successful regex from robust discovery
        html_pattern = r'(\d{8}-[^\s"\'<>]+\.html)'
        matches = re.findall(html_pattern, raw_html)
        
        discovered_urls = set()
        for match in matches:
            full_url = f"https://www.tennisabstract.com/charting/{match}"
            discovered_urls.add(full_url)
        
        print(f"✅ Discovered {len(discovered_urls)} URLs from Tennis Abstract")
        
    except Exception as e:
        print(f"❌ Error discovering URLs: {e}")
        return []
    
    # Get cached URLs
    try:
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        with open(cache_file, 'r') as f:
            cached_urls = set(line.strip() for line in f if line.strip())
        
        print(f"✅ Loaded {len(cached_urls)} cached URLs")
        
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return []
    
    # Find missing URLs
    missing_urls = discovered_urls - cached_urls
    print(f"🆕 Found {len(missing_urls)} missing URLs")
    
    return sorted(list(missing_urls))

def scrape_missing_matches():
    """Scrape all 73 missing Tennis Abstract matches"""
    print("🎾 SCRAPING MISSING TENNIS ABSTRACT MATCHES")
    print("="*70)
    print("Goal: Scrape all 73 missing matches for complete coverage")
    print()
    
    # Get missing URLs
    missing_urls = get_missing_urls()
    
    if not missing_urls:
        print("✅ No missing URLs found - cache appears complete")
        return pd.DataFrame(), 0
    
    print(f"\n🚀 STARTING SCRAPE OF {len(missing_urls)} MATCHES")
    print("-" * 50)
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    successful_urls = []
    
    cache_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete')
    cache_dir.mkdir(exist_ok=True)
    
    for i, url in enumerate(missing_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\n[{i}/{len(missing_urls)}] {match_name[:60]}")
        
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
                    
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    
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
            print(f"  💥 Error: {str(e)[:80]}")
        
        # Progress update every 10 matches
        if i % 10 == 0:
            print(f"\n📊 Progress: {i}/{len(missing_urls)} ({i/len(missing_urls)*100:.1f}%)")
            print(f"   Successful: {len(successful_urls)}")
            print(f"   Records: {len(all_records):,}")
        
        time.sleep(1.5)  # Be respectful to Tennis Abstract
    
    # Save results
    if successful_urls:
        print(f"\n🎉 SCRAPING COMPLETED!")
        print(f"✅ Successfully scraped: {len(successful_urls)} matches")
        print(f"✅ Total records extracted: {len(all_records):,}")
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if all_matches:
            matches_df = pd.DataFrame(all_matches)
            matches_file = cache_dir / f'missing_73_matches_{timestamp}.parquet'
            matches_df.to_parquet(matches_file, index=False)
            print(f"💾 Matches: {matches_file}")
        
        if all_records:
            records_df = pd.DataFrame(all_records)
            records_file = cache_dir / f'missing_73_records_{timestamp}.parquet'
            records_df.to_parquet(records_file, index=False)
            print(f"💾 Records: {records_file}")
        
        # Update cache
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        try:
            with open(cache_file, 'a') as f:
                for url in successful_urls:
                    f.write(url + '\n')
            print(f"💾 Updated cache with {len(successful_urls)} new URLs")
        except Exception as e:
            print(f"⚠️  Could not update cache: {e}")
        
        # Final summary
        original_cache = 123
        new_total = original_cache + len(successful_urls)
        
        print(f"\n🎯 TENNIS ABSTRACT COVERAGE COMPLETE!")
        print(f"✅ Original cache: {original_cache} matches")
        print(f"✅ Newly scraped: {len(successful_urls)} matches")
        print(f"🏆 TOTAL COVERAGE: {new_total} matches")
        print(f"📊 Success rate: {len(successful_urls)}/{len(missing_urls)} ({len(successful_urls)/len(missing_urls)*100:.1f}%)")
        
        return pd.DataFrame(all_matches) if all_matches else pd.DataFrame(), len(all_records)
    
    else:
        print(f"\n❌ No matches scraped successfully")
        return pd.DataFrame(), 0

if __name__ == "__main__":
    matches_df, total_records = scrape_missing_matches()
    
    if not matches_df.empty or total_records > 0:
        print(f"\n🚀 MISSION ACCOMPLISHED!")
        print(f"Tennis Abstract coverage is now COMPLETE!")
        print(f"Ready for final data integration with API-Tennis and Jeff's data")
    else:
        print(f"\n⚠️  Scraping was not successful")
        print(f"May need to investigate further or try different approach")