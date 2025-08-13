#!/usr/bin/env python3
"""
Focused Tennis Abstract scraper - scrape remaining URLs without loading Jeff data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path
import re

class SimpleTennisAbstractScraper:
    """Lightweight Tennis Abstract scraper without dependencies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def scrape_match(self, url):
        """Scrape a single Tennis Abstract match"""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all tables with match data
            tables = soup.find_all('table')
            all_records = []
            
            for table in tables:
                # Skip empty tables
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                # Get headers
                header_row = rows[0]
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                
                if not headers:
                    continue
                
                # Process data rows
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) != len(headers):
                        continue
                    
                    record = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            record[headers[i]] = cell.get_text().strip()
                    
                    if record:
                        all_records.append(record)
            
            return all_records
            
        except Exception as e:
            print(f"    Error scraping {url}: {str(e)[:50]}")
            return []

def scrape_remaining_tennis_abstract():
    """Scrape remaining Tennis Abstract matches"""
    print("🎾 FOCUSED TENNIS ABSTRACT SCRAPING")
    print("=" * 60)
    
    # Load all cached URLs
    cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
    
    try:
        with open(cache_file, 'r') as f:
            all_urls = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(all_urls)} cached URLs")
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return None
    
    # Check what we already have  
    existing_file = '/Users/danielkim/Desktop/t3nn1s/tennis_abstract_recent/recent_missing_records_20250811_223651.parquet'
    existing_urls = set()
    
    try:
        if Path(existing_file).exists():
            existing_data = pd.read_parquet(existing_file)
            if 'url' in existing_data.columns:
                existing_urls = set(existing_data['url'].unique())
                print(f"✅ Found {len(existing_urls)} already scraped URLs")
    except Exception as e:
        print(f"⚠️  Could not load existing: {e}")
    
    # Find remaining URLs
    remaining_urls = [url for url in all_urls if url not in existing_urls]
    print(f"🎯 Need to scrape {len(remaining_urls)} remaining URLs")
    
    if not remaining_urls:
        print("✅ All URLs already scraped!")
        return existing_data if 'existing_data' in locals() else pd.DataFrame()
    
    print(f"\\n🚀 Scraping {len(remaining_urls)} matches...")
    
    scraper = SimpleTennisAbstractScraper()
    all_new_records = []
    successful_count = 0
    
    for i, url in enumerate(remaining_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"[{i}/{len(remaining_urls)}] {match_name[:60]}")
        
        records = scraper.scrape_match(url)
        
        if records:
            # Add metadata
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
                
                all_new_records.extend(records)
                successful_count += 1
                print(f"  ✅ {len(records)} records")
            else:
                print(f"  ⚠️  Could not parse URL structure")
        else:
            print(f"  ❌ No records")
        
        if i % 20 == 0:
            print(f"\\n📊 Progress: {successful_count}/{i} successful")
        
        time.sleep(1.2)  # Be respectful
    
    # Combine with existing data
    print(f"\\n🔄 Combining data...")
    all_records = all_new_records.copy()
    
    if existing_urls and Path(existing_file).exists():
        try:
            existing_data = pd.read_parquet(existing_file)
            existing_records = existing_data.to_dict('records')
            all_records.extend(existing_records)
            print(f"✅ Combined: {len(all_records):,} total records")
        except Exception as e:
            print(f"⚠️  Could not combine: {e}")
    
    # Save complete dataset
    if all_records:
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        complete_file = output_dir / f'complete_tennis_abstract_{timestamp}.parquet'
        
        complete_df = pd.DataFrame(all_records)
        complete_df.to_parquet(complete_file, index=False)
        
        unique_urls = len(set(record.get('url', '') for record in all_records))
        
        print(f"\\n💾 Saved: {complete_file}")
        print(f"✅ Total records: {len(all_records):,}")
        print(f"✅ Unique matches: {unique_urls}")
        print(f"🎯 Coverage: {unique_urls}/196 URLs ({unique_urls/196*100:.1f}%)")
        
        if unique_urls >= 190:
            print(f"🏆 EXCELLENT! Nearly complete Tennis Abstract coverage!")
        
        return complete_df
    else:
        print("❌ No records scraped")
        return pd.DataFrame()

if __name__ == "__main__":
    result = scrape_remaining_tennis_abstract()
    if not result.empty:
        print("\\n🚀 Tennis Abstract scraping complete!")
    else:
        print("\\n❌ Scraping failed")