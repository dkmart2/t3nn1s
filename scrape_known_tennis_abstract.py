#!/usr/bin/env python3
"""
Scrape known working Tennis Abstract URLs from previous work
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

def scrape_known_tennis_abstract_matches():
    """Scrape known working Tennis Abstract URLs"""
    print("🎾 SCRAPING KNOWN TENNIS ABSTRACT MATCHES")
    print("="*70)
    
    # Use known working URLs from recent tournament results
    known_urls = [
        # Wimbledon 2025 (major matches)
        "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html",
        "https://www.tennisabstract.com/charting/20250712-W-Wimbledon-F-Amanda_Anisimova-Iga_Swiatek.html", 
        "https://www.tennisabstract.com/charting/20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner.html",
        "https://www.tennisabstract.com/charting/20250710-W-Wimbledon-SF-Iga_Swiatek-Belinda_Bencic.html",
        
        # Recent grass court season
        "https://www.tennisabstract.com/charting/20250628-W-Bad_Homburg-F-Iga_Swiatek-Jessica_Pegula.html",
        "https://www.tennisabstract.com/charting/20250622-M-Queens_Club-F-Jiri_Lehecka-Carlos_Alcaraz.html",
        "https://www.tennisabstract.com/charting/20250615-M-Stuttgart-F-Taylor_Fritz-Alexander_Zverev.html",
        
        # Recent other tournaments
        "https://www.tennisabstract.com/charting/20250628-M-Mallorca-F-Tallon_Griekspoor-Corentin_Moutet.html",
        "https://www.tennisabstract.com/charting/20250622-W-Berlin-F-Marketa_Vondrousova-Xin_Yu_Wang.html",
        "https://www.tennisabstract.com/charting/20250622-M-Halle-F-Alexander_Bublik-Daniil_Medvedev.html",
    ]
    
    print(f"Testing {len(known_urls)} known Tennis Abstract URLs...")
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    successful_urls = []
    
    cache_dir = Path('/Users/danielkim/tennis_data/cache') / 'tennis_abstract_recent'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, url in enumerate(known_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\n[{i}/{len(known_urls)}] Scraping: {match_name}")
        
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
                    players = list(set(record.get('Player_canonical', 'Unknown') for record in records[:10]))
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
            print(f"  💥 Error: {e}")
        
        time.sleep(2)  # Be respectful to Tennis Abstract
    
    # Save results
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        records_df = pd.DataFrame(all_records)
        
        print(f"\n✅ TENNIS ABSTRACT SCRAPING COMPLETED!")
        print(f"   Successful URLs: {len(successful_urls)}")
        print(f"   Matches: {len(matches_df)}")
        print(f"   Detailed records: {len(records_df)}")
        
        # Save both datasets
        matches_file = cache_dir / f'ta_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        records_file = cache_dir / f'ta_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        
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
        
        return matches_df, successful_urls
    else:
        print(f"\n❌ No Tennis Abstract matches scraped successfully")
        return pd.DataFrame(), []

if __name__ == "__main__":
    matches_df, urls = scrape_known_tennis_abstract_matches()
    
    if not matches_df.empty:
        print(f"\n🎯 Ready to integrate {len(matches_df)} Tennis Abstract matches!")
        print(f"URLs successfully scraped: {len(urls)}")
    else:
        print(f"\n⚠️  No successful Tennis Abstract scraping")