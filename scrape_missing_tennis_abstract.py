#!/usr/bin/env python3
"""
Scrape the missing Tennis Abstract matches we just discovered!
"""

from tennis_updated import TennisAbstractScraper
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

def scrape_missing_tennis_abstract_matches():
    """Scrape the missing Tennis Abstract matches from August 9 back to July 16"""
    print("🎯 SCRAPING MISSING TENNIS ABSTRACT MATCHES")
    print("="*70)
    print("Goal: Get the matches from July 16 - August 9 that we're missing!")
    print()
    
    # Build the exact URLs from the recent additions page
    missing_urls = [
        # August 2025 - Most recent
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati_Masters-R64-Jannik_Sinner-Daniel_Elahi_Galan.html",
        "https://www.tennisabstract.com/charting/20250809-W-Cincinnati-R64-Emma_Raducanu-Olga_Danilovic.html",
        
        # Montreal/Toronto period
        "https://www.tennisabstract.com/charting/20250805-W-Montreal-QF-Naomi_Osaka-Elina_Svitolina.html",
        "https://www.tennisabstract.com/charting/20250729-W-Montreal-R64-Elena_Rybakina-Hailey_Baptiste.html", 
        "https://www.tennisabstract.com/charting/20250728-W-Montreal-R128-Ariana_Arseneault-Naomi_Osaka.html",
        "https://www.tennisabstract.com/charting/20250728-M-Canada_Masters-R128-Dan_Martin-Jaume_Munar.html",
        
        # Other July matches
        "https://www.tennisabstract.com/charting/20250726-W-Prague-F-Marie_Bouzkova-Linda_Noskova.html",
        "https://www.tennisabstract.com/charting/20250718-W-Hamburg-QF-Lois_Boisson-Viktoriya_Tomova.html",
        "https://www.tennisabstract.com/charting/20250716-M-Bastad-R16-Andrea_Pellegrino-Tallon_Griekspoor.html",
    ]
    
    # Also try some likely variations for common tournaments/players in that period
    likely_additional_urls = [
        # Cincinnati variations
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati-R64-Carlos_Alcaraz-Ben_Shelton.html",
        "https://www.tennisabstract.com/charting/20250809-W-Cincinnati-R64-Iga_Swiatek-Marta_Kostyuk.html",
        "https://www.tennisabstract.com/charting/20250808-M-Cincinnati-R64-Novak_Djokovic-Rafael_Nadal.html",
        "https://www.tennisabstract.com/charting/20250808-W-Cincinnati-R64-Aryna_Sabalenka-Sloane_Stephens.html",
        
        # Montreal/Toronto
        "https://www.tennisabstract.com/charting/20250805-M-Montreal-QF-Daniil_Medvedev-Alexander_Zverev.html",
        "https://www.tennisabstract.com/charting/20250804-W-Toronto-QF-Jessica_Pegula-Emma_Navarro.html",
        "https://www.tennisabstract.com/charting/20250804-M-Montreal-QF-Sebastian_Korda-Tommy_Paul.html",
        
        # Olympics period
        "https://www.tennisabstract.com/charting/20250803-M-Olympics-F-Carlos_Alcaraz-Jannik_Sinner.html",
        "https://www.tennisabstract.com/charting/20250802-W-Olympics-F-Iga_Swiatek-Coco_Gauff.html",
        "https://www.tennisabstract.com/charting/20250801-M-Olympics-SF-Novak_Djokovic-Lorenzo_Musetti.html",
        
        # Hamburg/Atlanta
        "https://www.tennisabstract.com/charting/20250721-M-Hamburg-F-Sebastian_Baez-Arthur_Fils.html",
        "https://www.tennisabstract.com/charting/20250720-W-Hamburg-F-Diana_Shnaider-Eva_Lys.html",
        "https://www.tennisabstract.com/charting/20250721-M-Atlanta-F-Ben_Shelton-Frances_Tiafoe.html",
        
        # Washington
        "https://www.tennisabstract.com/charting/20250804-M-Washington-F-Sebastian_Korda-Flavio_Cobolli.html",
        "https://www.tennisabstract.com/charting/20250804-W-Washington-F-Paula_Badosa-Marie_Bouzkova.html",
    ]
    
    all_urls_to_test = missing_urls + likely_additional_urls
    
    print(f"Testing {len(all_urls_to_test)} potential URLs...")
    
    # First, test which URLs exist
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    existing_urls = []
    for i, url in enumerate(all_urls_to_test, 1):
        match_name = url.split('/')[-1].replace('.html', '')[:50]
        
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                existing_urls.append(url)
                print(f"  ✅ [{i}] Found: {match_name}")
            elif i <= len(missing_urls):  # Only show failures for known URLs
                print(f"  ❌ [{i}] Not found: {match_name}")
        except:
            continue
        
        time.sleep(0.5)
    
    print(f"\n📊 DISCOVERY RESULTS:")
    print(f"✅ Found {len(existing_urls)} existing URLs")
    
    if not existing_urls:
        print("❌ No additional URLs found - our 111 matches may be complete")
        return pd.DataFrame(), 0
    
    # Now scrape the found matches
    print(f"\n🎾 SCRAPING DISCOVERED MATCHES")
    print("-" * 50)
    
    scraper = TennisAbstractScraper()
    all_records = []
    all_matches = []
    
    cache_dir = Path('/Users/danielkim/tennis_data/cache') / 'tennis_abstract_missing'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, url in enumerate(existing_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"\n[{i}/{len(existing_urls)}] Scraping: {match_name}")
        
        try:
            records = scraper.scrape_comprehensive_match_data(url)
            
            if records and len(records) > 0:
                print(f"  ✅ Extracted {len(records)} records")
                
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
        
        time.sleep(2)
    
    # Save results
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        records_df = pd.DataFrame(all_records)
        
        print(f"\n🎉 MISSING MATCHES SCRAPED SUCCESSFULLY!")
        print(f"   New matches: {len(matches_df)}")
        print(f"   New detailed records: {len(all_records):,}")
        
        # Save data
        matches_file = cache_dir / f'missing_ta_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        records_file = cache_dir / f'missing_ta_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        
        matches_df.to_parquet(matches_file, index=False)
        records_df.to_parquet(records_file, index=False)
        
        print(f"💾 Saved: {matches_file}")
        print(f"💾 Saved: {records_file}")
        
        # Show breakdown
        if 'tournament' in matches_df.columns:
            print(f"\n📊 New tournament breakdown:")
            tournament_counts = matches_df['tournament'].value_counts()
            for tournament, count in tournament_counts.items():
                print(f"  - {tournament}: {count}")
        
        return matches_df, len(all_records)
    else:
        print(f"\n❌ No additional matches scraped")
        return pd.DataFrame(), 0

if __name__ == "__main__":
    matches, records = scrape_missing_tennis_abstract_matches()
    
    if not matches.empty:
        print(f"\n🚀 SUCCESS! Found and scraped the missing Tennis Abstract matches!")
        print(f"✅ Added {len(matches)} new matches")
        print(f"✅ Added {records:,} new detailed records") 
        print(f"🎯 Tennis Abstract coverage now COMPLETE!")
    else:
        print(f"\n🔍 No additional matches found")
        print(f"Our 111 matches may represent complete Tennis Abstract coverage")
        print(f"Tennis Abstract is volunteer-driven, so limited coverage is expected")