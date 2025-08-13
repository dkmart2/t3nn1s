#!/usr/bin/env python3
"""
Complete Tennis Abstract discovery - get ALL 127 matches WebFetch confirmed exist
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import time
import re
from pathlib import Path

def discover_complete_tennis_abstract():
    """Use comprehensive parsing to find all 127 Tennis Abstract matches"""
    print("🔍 COMPLETE TENNIS ABSTRACT DISCOVERY")
    print("="*70)
    print("Goal: Find ALL 127 matches that WebFetch confirmed exist!")
    print()
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # URLs to check comprehensively
    urls_to_check = [
        ("Main Charting", "https://www.tennisabstract.com/charting/"),
        ("Recent Additions", "https://www.tennisabstract.com/charting/recent.html"),
        ("Meta Page", "https://www.tennisabstract.com/charting/meta.html")
    ]
    
    all_found_urls = set()
    cutoff_date = date(2025, 6, 10)
    
    for page_name, url in urls_to_check:
        print(f"\n📋 PARSING: {page_name}")
        print(f"URL: {url}")
        print("-" * 50)
        
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code != 200:
                print(f"❌ Failed: HTTP {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            page_urls = set()
            
            # Method 1: Look for direct links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                
                # Handle relative URLs
                if href.startswith('/charting/') and href.endswith('.html'):
                    full_url = f"https://www.tennisabstract.com{href}"
                elif 'tennisabstract.com/charting/' in href and href.endswith('.html'):
                    full_url = href
                else:
                    continue
                
                # Extract filename and check date
                filename = full_url.split('/')[-1]
                if len(filename) >= 8 and filename[:8].isdigit():
                    try:
                        date_str = filename[:8]
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                        if match_date >= cutoff_date:
                            page_urls.add(full_url)
                    except:
                        continue
            
            # Method 2: Look for URLs in JavaScript or text content
            page_text = response.text
            js_urls = re.findall(r'https?://(?:www\.)?tennisabstract\.com/charting/(\d{8}-[^"\'>\s]+\.html)', page_text)
            
            for match in js_urls:
                try:
                    date_str = match[:8]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    if match_date >= cutoff_date:
                        full_url = f"https://www.tennisabstract.com/charting/{match}"
                        page_urls.add(full_url)
                except:
                    continue
            
            # Method 3: Look for relative URLs in text
            relative_urls = re.findall(r'/charting/(\d{8}-[^"\'>\s]+\.html)', page_text)
            for match in relative_urls:
                try:
                    date_str = match[:8]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    if match_date >= cutoff_date:
                        full_url = f"https://www.tennisabstract.com/charting/{match}"
                        page_urls.add(full_url)
                except:
                    continue
            
            print(f"✅ Found {len(page_urls)} matches on {page_name}")
            all_found_urls.update(page_urls)
            
        except Exception as e:
            print(f"❌ Error parsing {page_name}: {e}")
        
        time.sleep(2)  # Be respectful
    
    print(f"\n📊 DISCOVERY SUMMARY")
    print("="*50)
    print(f"Total unique URLs found: {len(all_found_urls)}")
    
    if len(all_found_urls) == 0:
        print("❌ No matches found - this suggests parsing issues")
        return []
    
    # Sort by date
    sorted_urls = []
    for url in all_found_urls:
        try:
            filename = url.split('/')[-1]
            date_str = filename[:8]
            match_date = datetime.strptime(date_str, '%Y%m%d').date()
            sorted_urls.append((match_date, url))
        except:
            continue
    
    sorted_urls.sort(reverse=True)  # Newest first
    
    # Analyze what we found
    print(f"\n📅 DATE RANGE:")
    if sorted_urls:
        print(f"Latest: {sorted_urls[0][0]}")
        print(f"Earliest: {sorted_urls[-1][0]}")
    
    # Tournament breakdown
    tournaments = {}
    wimbledon_matches = []
    
    for match_date, url in sorted_urls:
        filename = url.split('/')[-1].replace('.html', '')
        parts = filename.split('-')
        if len(parts) >= 3:
            tournament = parts[2]
            tournaments[tournament] = tournaments.get(tournament, 0) + 1
            if tournament == 'Wimbledon':
                wimbledon_matches.append((match_date, url))
    
    print(f"\n🏆 TOP TOURNAMENTS:")
    for tournament, count in sorted(tournaments.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {tournament}: {count}")
    
    print(f"\n🎾 WIMBLEDON ANALYSIS:")
    print(f"Wimbledon matches found: {len(wimbledon_matches)}")
    if wimbledon_matches:
        print(f"Wimbledon date range: {min(w[0] for w in wimbledon_matches)} to {max(w[0] for w in wimbledon_matches)}")
    
    # Compare with our cache
    try:
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        with open(cache_file, 'r') as f:
            cached_urls = set(line.strip() for line in f if line.strip())
        
        print(f"\n🔍 CACHE COMPARISON:")
        print(f"Cached URLs: {len(cached_urls)}")
        print(f"Discovered URLs: {len(all_found_urls)}")
        
        missing_from_cache = all_found_urls - cached_urls
        print(f"Missing from cache: {len(missing_from_cache)}")
        
        if missing_from_cache:
            print(f"\n🆕 MISSING MATCHES (showing first 10):")
            missing_sorted = []
            for url in missing_from_cache:
                try:
                    filename = url.split('/')[-1]
                    date_str = filename[:8]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    missing_sorted.append((match_date, url))
                except:
                    continue
            
            missing_sorted.sort(reverse=True)
            for i, (match_date, url) in enumerate(missing_sorted[:10]):
                match_name = url.split('/')[-1].replace('.html', '')
                print(f"  {i+1}. {match_date}: {match_name[:60]}")
    
    except FileNotFoundError:
        print(f"\n⚠️  No cache file found")
        missing_from_cache = all_found_urls
    
    # Save all discovered URLs
    output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_discovery')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'all_tennis_abstract_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(output_file, 'w') as f:
        for match_date, url in sorted_urls:
            f.write(f"{url}\n")
    
    print(f"\n💾 Saved all URLs to: {output_file}")
    
    return [url for match_date, url in sorted_urls], missing_from_cache

if __name__ == "__main__":
    all_urls, missing_urls = discover_complete_tennis_abstract()
    
    print(f"\n🎯 DISCOVERY COMPLETE")
    print(f"✅ Total URLs: {len(all_urls)}")
    print(f"🆕 Missing from cache: {len(missing_urls)}")
    
    if len(all_urls) >= 120:
        print(f"🎉 SUCCESS! Found comprehensive Tennis Abstract coverage")
        if len(missing_urls) > 0:
            print(f"⚠️  Need to scrape {len(missing_urls)} additional matches")
    else:
        print(f"⚠️  Found fewer URLs than expected - may need different parsing approach")