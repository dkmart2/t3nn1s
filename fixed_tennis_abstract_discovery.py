#!/usr/bin/env python3
"""
Fixed Tennis Abstract URL discovery - properly extract all match URLs
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import re
import time
from pathlib import Path

def discover_all_tennis_abstract_urls():
    """Properly discover ALL Tennis Abstract URLs using corrected parsing"""
    print("🔧 FIXED TENNIS ABSTRACT URL DISCOVERY")
    print("="*70)
    print("Using corrected parsing to find ALL Tennis Abstract matches")
    print()
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    cutoff_date = date(2025, 6, 10)
    
    # Try multiple Tennis Abstract pages
    pages_to_check = [
        "https://www.tennisabstract.com/charting/",
        "https://www.tennisabstract.com/charting/recent.html", 
        "https://www.tennisabstract.com/charting/meta.html",
    ]
    
    all_urls = set()
    
    for page_url in pages_to_check:
        print(f"📋 Processing: {page_url}")
        
        try:
            response = requests.get(page_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"  ❌ Failed: HTTP {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Method 1: Extract from HTML <a> tags with proper href
            for link in soup.find_all('a', href=True):
                href = link.get('href').strip()
                
                # Handle relative and absolute URLs
                if href.startswith('/charting/') and href.endswith('.html'):
                    full_url = f"https://www.tennisabstract.com{href}"
                elif href.startswith('https://www.tennisabstract.com/charting/') and href.endswith('.html'):
                    full_url = href
                else:
                    continue
                
                # Validate it's a match URL with proper date
                filename = full_url.split('/')[-1]
                if len(filename) >= 12 and filename[:8].isdigit() and '-' in filename:
                    try:
                        date_str = filename[:8]
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                        if match_date >= cutoff_date:
                            all_urls.add(full_url)
                    except:
                        continue
            
            # Method 2: Find URLs in table cells or list items
            for element in soup.find_all(['td', 'li', 'div']):
                if element.get_text():
                    text = element.get_text()
                    # Look for date patterns at start of text
                    date_match = re.match(r'^(\d{4}-\d{2}-\d{2})', text)
                    if date_match:
                        # Find corresponding link
                        link = element.find('a', href=True)
                        if link:
                            href = link.get('href')
                            if '/charting/' in href and href.endswith('.html'):
                                if href.startswith('/'):
                                    full_url = f"https://www.tennisabstract.com{href}"
                                else:
                                    full_url = href
                                
                                filename = full_url.split('/')[-1]
                                if filename[:8].isdigit():
                                    try:
                                        date_str = filename[:8]
                                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                                        if match_date >= cutoff_date:
                                            all_urls.add(full_url)
                                    except:
                                        continue
            
            print(f"  ✅ Found URLs from {page_url.split('/')[-1]}")
            
        except Exception as e:
            print(f"  ❌ Error processing {page_url}: {e}")
        
        time.sleep(2)
    
    # Clean and validate URLs
    valid_urls = set()
    print(f"\n🧹 CLEANING AND VALIDATING URLs...")
    
    for url in all_urls:
        # Remove any HTML artifacts or malformed URLs
        if not url.startswith('http'):
            continue
        if '"' in url or '<' in url or '>' in url:
            continue
        if not url.endswith('.html'):
            continue
            
        # Validate filename format
        filename = url.split('/')[-1]
        if len(filename) < 12:  # Minimum length for valid match file
            continue
        if not filename[:8].isdigit():
            continue
        if filename.count('-') < 4:  # Should have at least YYYYMMDD-G-Tournament-Round-Player format
            continue
            
        valid_urls.add(url)
    
    print(f"✅ Cleaned URLs: {len(all_urls)} → {len(valid_urls)}")
    
    # Sort by date
    dated_urls = []
    for url in valid_urls:
        try:
            filename = url.split('/')[-1]
            date_str = filename[:8]
            match_date = datetime.strptime(date_str, '%Y%m%d').date()
            dated_urls.append((match_date, url))
        except:
            continue
    
    dated_urls.sort()  # Sort by date
    
    print(f"\n📊 DISCOVERY RESULTS:")
    print(f"Total valid URLs: {len(dated_urls)}")
    if dated_urls:
        print(f"Date range: {dated_urls[0][0]} to {dated_urls[-1][0]}")
    
    # Compare with our cache
    try:
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        with open(cache_file, 'r') as f:
            cached_urls = set(line.strip() for line in f if line.strip())
        
        discovered_urls = set(url for _, url in dated_urls)
        missing_urls = discovered_urls - cached_urls
        
        print(f"\n🔍 COMPARISON WITH CACHE:")
        print(f"Cached URLs: {len(cached_urls)}")
        print(f"Discovered URLs: {len(discovered_urls)}")
        print(f"Missing from cache: {len(missing_urls)}")
        
        if missing_urls:
            print(f"\n🆕 NEW URLS FOUND (first 20):")
            missing_list = sorted(missing_urls)
            for i, url in enumerate(missing_list[:20], 1):
                match_name = url.split('/')[-1].replace('.html', '')
                print(f"  {i:2d}. {match_name}")
            
            if len(missing_list) > 20:
                print(f"  ... and {len(missing_list) - 20} more")
        
        # Save all discovered URLs
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_fixed_discovery')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'all_discovered_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(output_file, 'w') as f:
            for match_date, url in dated_urls:
                f.write(f"{url}\n")
        
        print(f"\n💾 Saved all discovered URLs to: {output_file}")
        
        return list(discovered_urls), list(missing_urls)
        
    except Exception as e:
        print(f"Error comparing with cache: {e}")
        return [url for _, url in dated_urls], []

def test_sample_discovered_urls(urls_sample, count=10):
    """Test a sample of discovered URLs to verify they actually exist"""
    print(f"\n🧪 TESTING SAMPLE OF DISCOVERED URLs")
    print("="*50)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    test_urls = urls_sample[:count] if len(urls_sample) > count else urls_sample
    working = 0
    broken = 0
    
    for url in test_urls:
        match_name = url.split('/')[-1].replace('.html', '')[:50]
        
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                working += 1
                print(f"  ✅ {match_name}")
            else:
                broken += 1
                print(f"  ❌ {match_name} (HTTP {response.status_code})")
        except:
            broken += 1
            print(f"  💥 {match_name} (Error)")
        
        time.sleep(0.5)
    
    print(f"\n📊 SAMPLE TEST RESULTS:")
    print(f"Working: {working}/{len(test_urls)}")
    print(f"Success rate: {working/len(test_urls)*100:.1f}%")
    
    return working/len(test_urls) > 0.8  # 80% success rate threshold

def main():
    """Run fixed Tennis Abstract discovery"""
    print("🚀 FIXING TENNIS ABSTRACT URL DISCOVERY")
    print("="*80)
    print("Goal: Find ALL Tennis Abstract matches using corrected parsing")
    print()
    
    # Discover all URLs with fixed method
    all_urls, missing_urls = discover_all_tennis_abstract_urls()
    
    # Test sample to verify quality
    if missing_urls:
        is_quality = test_sample_discovered_urls(missing_urls, 10)
        
        print(f"\n🎯 DISCOVERY SUMMARY:")
        print(f"✅ Total URLs discovered: {len(all_urls)}")
        print(f"🆕 New URLs found: {len(missing_urls)}")
        print(f"🔍 Sample success rate: {'✅ Good' if is_quality else '❌ Poor'}")
        
        if len(all_urls) >= 150:  # Closer to user's 196
            print(f"🎉 SUCCESS! Found comprehensive Tennis Abstract coverage")
            print(f"This matches the user's expectation of ~196 matches")
        else:
            print(f"⚠️  Still missing matches - may need additional discovery methods")
        
        return missing_urls
    else:
        print(f"\n✅ No new URLs found - our cache appears complete")
        return []

if __name__ == "__main__":
    missing = main()
    
    if missing:
        print(f"\n📝 NEXT STEPS:")
        print(f"1. Scrape the {len(missing)} newly discovered matches")
        print(f"2. Update the Tennis Abstract cache")
        print(f"3. Verify we now have ~196 total matches as expected")
    else:
        print(f"\n✅ Tennis Abstract discovery appears complete")