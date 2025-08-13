#!/usr/bin/env python3
"""
Fetch recent tennis data using Tennis Abstract scraper and other sources
Complete the data pipeline for post-6/10/2025 matches
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
from settings import TENNIS_CACHE_DIR
import logging

# Import the existing Tennis Abstract scraper from tennis_updated
from tennis_updated import TennisAbstractScraper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_recent_tennis_abstract_matches(start_date=None, end_date=None):
    """Get recent Tennis Abstract match URLs"""
    if not start_date:
        start_date = date(2025, 6, 10)
    if not end_date:
        end_date = date.today()
    
    print(f"\n🔍 FINDING TENNIS ABSTRACT MATCHES")
    print(f"Date range: {start_date} to {end_date}")
    print("-"*50)
    
    match_urls = []
    
    # Tennis Abstract URL pattern for recent matches
    # Format: https://www.tennisabstract.com/charting/YYYYMMDD-G-Tournament-Round-Player1-Player2.html
    # Where G is M for men, W for women
    
    # Try to get recent match listings from known tournaments
    recent_tournaments = [
        ('20250610', '20250616', 'Stuttgart', 'M'),  # Stuttgart Open
        ('20250610', '20250616', 'Hertogenbosch', 'M'),  # Libema Open
        ('20250617', '20250623', 'Queens', 'M'),  # Queen's Club
        ('20250617', '20250623', 'Halle', 'M'),  # Halle Open
        ('20250624', '20250707', 'Wimbledon', 'M'),  # Wimbledon
        ('20250624', '20250707', 'Wimbledon', 'W'),
        ('20250708', '20250714', 'Hamburg', 'M'),  # Hamburg Open
        ('20250708', '20250714', 'Newport', 'M'),  # Hall of Fame Open
        ('20250715', '20250721', 'Bastad', 'M'),  # Swedish Open
        ('20250715', '20250721', 'Gstaad', 'M'),  # Swiss Open
        ('20250722', '20250728', 'Umag', 'M'),  # Croatia Open
        ('20250722', '20250728', 'Atlanta', 'M'),  # Atlanta Open
        ('20250729', '20250804', 'Kitzbuhel', 'M'),  # Austrian Open
        ('20250729', '20250804', 'Washington', 'M'),  # Citi Open
        ('20250805', '20250811', 'Montreal', 'M'),  # Canadian Open
        ('20250805', '20250811', 'Toronto', 'W'),  # Canadian Open
    ]
    
    # Check for specific known matches
    known_matches = [
        "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html",
        "https://www.tennisabstract.com/charting/20250707-M-Wimbledon-SF-Carlos_Alcaraz-Daniil_Medvedev.html",
        "https://www.tennisabstract.com/charting/20250706-M-Wimbledon-SF-Jannik_Sinner-Novak_Djokovic.html",
    ]
    
    # Test which URLs are accessible
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    for url in known_matches:
        try:
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                match_urls.append(url)
                print(f"  ✓ Found: {url.split('/')[-1][:30]}...")
            else:
                print(f"  ✗ {response.status_code}: {url.split('/')[-1][:30]}...")
        except Exception as e:
            print(f"  ✗ Error: {url.split('/')[-1][:30]}...")
        
        time.sleep(1)  # Be respectful
    
    return match_urls

def scrape_tennis_abstract_matches(match_urls):
    """Scrape Tennis Abstract matches using existing scraper"""
    print(f"\n📊 SCRAPING TENNIS ABSTRACT DATA")
    print("-"*50)
    
    if not match_urls:
        print("❌ No match URLs to scrape")
        return pd.DataFrame()
    
    scraper = TennisAbstractScraper()
    all_match_data = []
    
    for i, url in enumerate(match_urls, 1):
        print(f"\n[{i}/{len(match_urls)}] Scraping: {url.split('/')[-1]}")
        
        try:
            # Use the comprehensive scraping method
            records = scraper.scrape_comprehensive_match_data(url)
            
            if records:
                print(f"  ✓ Extracted {len(records)} records")
                
                # Parse match metadata from URL
                url_parts = url.split('/')[-1].replace('.html', '').split('-')
                if len(url_parts) >= 5:
                    date_str = url_parts[0]
                    gender = url_parts[1]
                    tournament = url_parts[2]
                    round_str = url_parts[3]
                    players = url_parts[4:]
                    
                    # Add metadata to records
                    for record in records:
                        record['match_date'] = pd.to_datetime(date_str, format='%Y%m%d')
                        record['gender'] = gender
                        record['tournament'] = tournament
                        record['round'] = round_str
                        record['source'] = 'tennis_abstract'
                        record['source_rank'] = 1
                        record['has_detailed_stats'] = True
                        record['url'] = url
                    
                    all_match_data.extend(records)
            else:
                print(f"  ⚠️  No data extracted")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Rate limiting
        time.sleep(2)
    
    if all_match_data:
        df = pd.DataFrame(all_match_data)
        print(f"\n✅ Total scraped: {len(df)} records from {len(match_urls)} matches")
        
        # Save to cache
        cache_dir = Path(TENNIS_CACHE_DIR) / 'tennis_abstract'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f'ta_scraped_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        df.to_parquet(cache_file, index=False)
        print(f"💾 Cached to: {cache_file}")
        
        return df
    else:
        print("\n❌ No data scraped")
        return pd.DataFrame()

def fetch_alternative_sources():
    """Fetch data from alternative sources"""
    print(f"\n🌐 CHECKING ALTERNATIVE SOURCES")
    print("-"*50)
    
    # Check for Jeff's GitHub updates
    jeff_github_url = "https://github.com/JeffSackmann/tennis_atp"
    print(f"  • Jeff Sackmann's GitHub: {jeff_github_url}")
    print(f"    → Check for CSV updates post-6/10/2025")
    
    # FlashScore API (if available)
    print(f"  • FlashScore/LiveScore APIs")
    print(f"    → Real-time match data and results")
    
    # ATP/WTA official sites
    print(f"  • ATP/WTA official websites")
    print(f"    → Official match results and statistics")
    
    return pd.DataFrame()

def integrate_recent_data():
    """Main function to integrate all recent data sources"""
    print("🎾 FETCHING RECENT TENNIS DATA")
    print("="*60)
    print("Goal: Complete coverage from 6/10/2025 to today")
    print()
    
    all_recent_data = []
    
    # Step 1: Tennis Abstract scraping
    print("📌 STEP 1: Tennis Abstract")
    ta_urls = get_recent_tennis_abstract_matches()
    
    if ta_urls:
        ta_data = scrape_tennis_abstract_matches(ta_urls)
        if not ta_data.empty:
            all_recent_data.append(ta_data)
            print(f"  ✓ Tennis Abstract: {len(ta_data)} records")
    else:
        print("  ⚠️  No Tennis Abstract URLs found")
    
    # Step 2: API-Tennis (if working)
    print(f"\n📌 STEP 2: API-Tennis")
    print("  ⚠️  API currently not working (rate limits/invalid key)")
    print("  → Need valid subscription or alternative API")
    
    # Step 3: Alternative sources
    print(f"\n📌 STEP 3: Alternative Sources")
    alt_data = fetch_alternative_sources()
    if not alt_data.empty:
        all_recent_data.append(alt_data)
    
    # Combine all data
    if all_recent_data:
        combined_df = pd.concat(all_recent_data, ignore_index=True)
        print(f"\n✅ TOTAL RECENT DATA: {len(combined_df)} records")
        
        # Save combined data
        cache_file = os.path.join(TENNIS_CACHE_DIR, 'recent_matches_combined.parquet')
        combined_df.to_parquet(cache_file, index=False)
        print(f"💾 Saved to: {cache_file}")
        
        return combined_df
    else:
        print(f"\n❌ No recent data fetched")
        return pd.DataFrame()

def main():
    """Main execution"""
    try:
        # Fetch recent data
        recent_data = integrate_recent_data()
        
        if not recent_data.empty:
            print(f"\n📊 SUMMARY")
            print(f"="*50)
            print(f"Total records: {len(recent_data)}")
            
            if 'match_date' in recent_data.columns:
                print(f"Date range: {recent_data['match_date'].min()} to {recent_data['match_date'].max()}")
            
            if 'data_type' in recent_data.columns:
                print(f"\nData types:")
                for dtype, count in recent_data['data_type'].value_counts().items():
                    print(f"  - {dtype}: {count}")
            
            if 'Player_canonical' in recent_data.columns:
                players = recent_data['Player_canonical'].unique()
                print(f"\nUnique players: {len(players)}")
                print(f"Sample players: {list(players[:5])}")
            
            print(f"\n✅ Recent data successfully fetched!")
        else:
            print(f"\n⚠️  No recent data available")
            print(f"\nRECOMMENDATIONS:")
            print(f"1. Check Tennis Abstract website for new match URLs")
            print(f"2. Update Jeff's CSV files from GitHub")
            print(f"3. Get valid API-Tennis subscription")
            print(f"4. Consider manual data entry for critical matches")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()