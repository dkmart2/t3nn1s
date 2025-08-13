#!/usr/bin/env python3
"""
Scrape the remaining 123 Tennis Abstract URLs we haven't scraped yet
This should get us to the full 196 matches
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

def get_remaining_urls():
    """Get the 123 URLs we haven't scraped yet"""
    print("🔍 Finding remaining URLs to scrape...")
    
    # Get all cached URLs
    with open('/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt', 'r') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    # Get already scraped URLs
    ta_file = '/Users/danielkim/Desktop/t3nn1s/tennis_abstract_recent/recent_missing_records_20250811_223651.parquet'
    scraped_urls = set()
    if Path(ta_file).exists():
        df = pd.read_parquet(ta_file)
        if 'url' in df.columns:
            scraped_urls = set(df['url'].unique())
    
    # Find remaining URLs
    remaining_urls = [url for url in all_urls if url not in scraped_urls]
    
    print(f"✅ Total URLs: {len(all_urls)}")
    print(f"✅ Already scraped: {len(scraped_urls)}")  
    print(f"🎯 Remaining to scrape: {len(remaining_urls)}")
    
    return remaining_urls

def simple_scrape_match(url, session):
    """Simple scraper for Tennis Abstract matches"""
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables with match data
        tables = soup.find_all('table')
        all_records = []
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Get headers from first row
            header_row = rows[0]
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            if not headers or len(headers) < 2:
                continue
            
            # Process data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) == len(headers):
                    record = {}
                    for i, cell in enumerate(cells):
                        record[headers[i]] = cell.get_text().strip()
                    
                    # Only add records with meaningful data
                    if record and any(value for value in record.values() if value and value != '-'):
                        all_records.append(record)
        
        return all_records
        
    except Exception as e:
        return []

def scrape_remaining_tennis_abstract():
    """Scrape the remaining 123 Tennis Abstract matches"""
    print("🎾 SCRAPING REMAINING 123 TENNIS ABSTRACT MATCHES")
    print("=" * 70)
    
    remaining_urls = get_remaining_urls()
    
    if not remaining_urls:
        print("✅ No remaining URLs to scrape!")
        return
    
    print(f"\\n🚀 Starting scrape of {len(remaining_urls)} matches...")
    print("=" * 50)
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    all_new_records = []
    successful_count = 0
    failed_count = 0
    
    for i, url in enumerate(remaining_urls, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"[{i:3d}/{len(remaining_urls)}] {match_name[:65]}")
        
        # Scrape the match
        records = simple_scrape_match(url, session)
        
        if records:
            # Add metadata to all records
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
                print(f"    ✅ {len(records):,} records")
            else:
                failed_count += 1
                print(f"    ❌ Could not parse URL")
        else:
            failed_count += 1
            print(f"    ❌ No records")
        
        # Progress updates
        if i % 20 == 0:
            success_rate = (successful_count / i) * 100
            print(f"\\n📊 Progress: {i}/{len(remaining_urls)} ({i/len(remaining_urls)*100:.1f}%)")
            print(f"   Success: {successful_count} ({success_rate:.1f}%)")
            print(f"   Records: {len(all_new_records):,}")
            print()
        
        # Be respectful to Tennis Abstract
        time.sleep(1.0)
    
    # Combine with existing data
    print(f"\\n🔄 Combining with existing scraped data...")
    
    # Load existing data
    existing_file = '/Users/danielkim/Desktop/t3nn1s/tennis_abstract_recent/recent_missing_records_20250811_223651.parquet'
    existing_records = []
    
    if Path(existing_file).exists():
        try:
            existing_df = pd.read_parquet(existing_file)
            existing_records = existing_df.to_dict('records')
            print(f"✅ Loaded {len(existing_records):,} existing records")
        except Exception as e:
            print(f"⚠️  Could not load existing data: {e}")
    
    # Combine all records
    all_records = existing_records + all_new_records
    total_unique_urls = len(set(record.get('url', '') for record in all_records))
    
    # Save complete dataset
    if all_records:
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_complete')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        complete_file = output_dir / f'complete_tennis_abstract_{timestamp}.parquet'
        
        complete_df = pd.DataFrame(all_records)
        complete_df.to_parquet(complete_file, index=False)
        
        print(f"\\n💾 Saved complete dataset: {complete_file}")
        print(f"✅ Total records: {len(all_records):,}")
        print(f"✅ Unique matches: {total_unique_urls}")
        print(f"✅ New matches scraped: {successful_count}")
        print(f"✅ Failed attempts: {failed_count}")
        
        coverage = (total_unique_urls / 196) * 100
        print(f"🎯 Coverage: {total_unique_urls}/196 URLs ({coverage:.1f}%)")
        
        if total_unique_urls >= 190:
            print(f"\\n🏆 EXCELLENT! Nearly complete Tennis Abstract integration!")
        elif total_unique_urls >= 180:
            print(f"\\n🎉 GREAT! Substantial Tennis Abstract coverage achieved!")
        else:
            print(f"\\n📊 Good progress, but room for improvement in coverage")
        
        return complete_df
    
    else:
        print("\\n❌ No data to save")
        return pd.DataFrame()

if __name__ == "__main__":
    result = scrape_remaining_tennis_abstract()
    
    if not result.empty:
        print("\\n🚀 Tennis Abstract scraping mission accomplished!")
        print("Ready for complete data integration!")
    else:
        print("\\n❌ Scraping mission failed")