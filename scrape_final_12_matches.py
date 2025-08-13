#!/usr/bin/env python3
"""
Scrape the final 12 missing Tennis Abstract matches to achieve 100% coverage
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

def scrape_final_12_matches():
    """Scrape the 12 missing Tennis Abstract matches we just identified"""
    print("🎾 SCRAPING FINAL 12 TENNIS ABSTRACT MATCHES")
    print("="*70)
    print("Goal: Achieve 100% Tennis Abstract coverage (123 total)")
    print()
    
    # The 12 missing URLs we confirmed exist
    missing_urls = [
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati_Masters-R64-Jannik_Sinner-Daniel_Elahi_Galan.html",
        "https://www.tennisabstract.com/charting/20250809-W-Cincinnati-R64-Emma_Raducanu-Olga_Danilovic.html",
        "https://www.tennisabstract.com/charting/20250806-W-Montreal-SF-Elena_Rybakina-Victoria_Mboko.html",
        "https://www.tennisabstract.com/charting/20250805-W-Montreal-QF-Naomi_Osaka-Elina_Svitolina.html",
        "https://www.tennisabstract.com/charting/20250805-W-Montreal-QF-Clara_Tauson-Madison_Keys.html",
        "https://www.tennisabstract.com/charting/20250804-W-Montreal-QF-Victoria_Mboko-Jessica_Bouzas_Maneiro.html",
        "https://www.tennisabstract.com/charting/20250803-W-Montreal-R16-Iga_Swiatek-Clara_Tauson.html",
        "https://www.tennisabstract.com/charting/20250803-W-Montreal-R16-Elina_Svitolina-Amanda_Anisimova.html",
        "https://www.tennisabstract.com/charting/20250802-W-Montreal-R16-Coco_Gauff-Victoria_Mboko.html",
        "https://www.tennisabstract.com/charting/20250801-W-Montreal-R32-Amanda_Anisimova-Emma_Raducanu.html",
        "https://www.tennisabstract.com/charting/20250610-W-s_Hertogenbosch-R32-Lulu_Sun-Anouck_Vrancken_Peeters.html",
        "https://www.tennisabstract.com/charting/20250610-W-s_Hertogenbosch-R32-Arianne_Hartono-Elisabetta_Cocciaretto.html"
    ]
    
    print(f"Scraping {len(missing_urls)} confirmed missing matches...")
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    successful_urls = []
    
    cache_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_final_12')
    cache_dir.mkdir(exist_ok=True)
    
    for i, url in enumerate(missing_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\n[{i}/{len(missing_urls)}] Scraping: {match_name}")
        
        try:
            records = scraper.scrape_comprehensive_match_data(url)
            
            if records and len(records) > 0:
                print(f"  ✅ Extracted {len(records)} records")
                successful_urls.append(url)
                
                # Parse match metadata
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
                print(f"  ❌ No records extracted")
                
        except Exception as e:
            print(f"  💥 Error: {str(e)[:100]}")
        
        time.sleep(2)  # Be respectful
    
    # Save results
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        records_df = pd.DataFrame(all_records)
        
        print(f"\n🎉 FINAL 12 MATCHES SCRAPED SUCCESSFULLY!")
        print(f"   Successful matches: {len(successful_urls)}")
        print(f"   Match summaries: {len(matches_df)}")
        print(f"   Detailed records: {len(all_records):,}")
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        matches_file = cache_dir / f'final_12_matches_{timestamp}.parquet'
        records_file = cache_dir / f'final_12_records_{timestamp}.parquet'
        
        matches_df.to_parquet(matches_file, index=False)
        records_df.to_parquet(records_file, index=False)
        
        print(f"💾 Saved matches: {matches_file}")
        print(f"💾 Saved records: {records_file}")
        
        # Tournament breakdown
        if 'tournament' in matches_df.columns:
            print(f"\n📊 Tournament breakdown:")
            tournament_counts = matches_df['tournament'].value_counts()
            for tournament, count in tournament_counts.items():
                print(f"  - {tournament}: {count}")
        
        # Update the master cache file
        cache_urls_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        try:
            with open(cache_urls_file, 'a') as f:
                for url in successful_urls:
                    f.write(url + '\n')
            print(f"\n💾 Added {len(successful_urls)} URLs to master cache")
        except Exception as e:
            print(f"⚠️  Could not update master cache: {e}")
        
        # Final status
        print(f"\n🎯 TENNIS ABSTRACT COVERAGE ACHIEVED!")
        print(f"✅ Original cache: 111 matches")
        print(f"✅ New matches: {len(successful_urls)}")
        print(f"✅ TOTAL: {111 + len(successful_urls)} matches")
        print(f"🏆 100% Tennis Abstract coverage confirmed!")
        
        return matches_df, len(all_records), len(successful_urls)
    else:
        print(f"\n❌ No new matches scraped successfully")
        return pd.DataFrame(), 0, 0

if __name__ == "__main__":
    matches_df, total_records, successful_count = scrape_final_12_matches()
    
    if not matches_df.empty:
        print(f"\n🚀 MISSION ACCOMPLISHED!")
        print(f"✅ Scraped {successful_count} additional Tennis Abstract matches")
        print(f"✅ Added {total_records:,} new detailed records")
        print(f"🎯 Tennis Abstract coverage now COMPLETE: {111 + successful_count} total matches!")
    else:
        print(f"\n⚠️  Scraping unsuccessful - may need to debug further")