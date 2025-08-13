#!/usr/bin/env python3
"""
Diagnose why our Tennis Abstract URL discovery is missing ~73 matches
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import re
import time
from pathlib import Path

def test_discovery_methods():
    """Test different methods to discover Tennis Abstract URLs"""
    print("🔍 DIAGNOSING TENNIS ABSTRACT URL DISCOVERY METHODS")
    print("="*70)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Test different Tennis Abstract pages
    pages_to_test = [
        ("Main Page", "https://www.tennisabstract.com/charting/"),
        ("Recent Page", "https://www.tennisabstract.com/charting/recent.html"),
        ("Meta Page", "https://www.tennisabstract.com/charting/meta.html"),
        ("Index Page", "https://www.tennisabstract.com/charting/index.html"),
    ]
    
    all_discovered_urls = set()
    
    for page_name, url in pages_to_test:
        print(f"\n📋 TESTING: {page_name}")
        print(f"URL: {url}")
        print("-" * 50)
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                continue
                
            page_urls = set()
            
            # Method 1: Parse HTML links
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                
                # Handle different URL formats
                if href.startswith('/charting/') and href.endswith('.html'):
                    full_url = f"https://www.tennisabstract.com{href}"
                elif 'tennisabstract.com/charting/' in href and href.endswith('.html'):
                    full_url = href
                else:
                    continue
                
                # Check if it's a match URL with date
                filename = full_url.split('/')[-1]
                if len(filename) >= 8 and filename[:8].isdigit():
                    try:
                        date_str = filename[:8]
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                        if match_date >= date(2025, 6, 10):
                            page_urls.add(full_url)
                    except:
                        continue
            
            # Method 2: Parse JavaScript/text content for URLs
            page_text = response.text
            
            # Look for URLs in JavaScript
            js_matches = re.findall(r'(?:https?://(?:www\.)?tennisabstract\.com)?/charting/(\d{8}-[^"\'>\s]+\.html)', page_text)
            for match in js_matches:
                try:
                    date_str = match[:8]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    if match_date >= date(2025, 6, 10):
                        full_url = f"https://www.tennisabstract.com/charting/{match}"
                        page_urls.add(full_url)
                except:
                    continue
            
            # Method 3: Look for patterns in plain text
            text_matches = re.findall(r'(\d{8})-([MW])-([^-]+)-([^-]+)-([^-]+)-([^-\s]+)', page_text)
            for match in text_matches:
                try:
                    date_str = match[0]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    if match_date >= date(2025, 6, 10):
                        # Reconstruct URL
                        url_components = '-'.join(match)
                        full_url = f"https://www.tennisabstract.com/charting/{url_components}.html"
                        page_urls.add(full_url)
                except:
                    continue
            
            print(f"Found {len(page_urls)} URLs on {page_name}")
            all_discovered_urls.update(page_urls)
            
        except Exception as e:
            print(f"Error testing {page_name}: {e}")
        
        time.sleep(2)
    
    print(f"\n📊 DISCOVERY SUMMARY:")
    print(f"Total unique URLs discovered: {len(all_discovered_urls)}")
    
    # Compare with our cache
    try:
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        with open(cache_file, 'r') as f:
            cached_urls = set(line.strip() for line in f if line.strip())
        
        print(f"URLs in our cache: {len(cached_urls)}")
        
        missing_from_cache = all_discovered_urls - cached_urls
        missing_from_discovery = cached_urls - all_discovered_urls
        
        print(f"Missing from cache: {len(missing_from_cache)}")
        print(f"Missing from discovery: {len(missing_from_discovery)}")
        
        if missing_from_cache:
            print(f"\n🆕 URLs we missed (showing first 10):")
            for i, url in enumerate(sorted(missing_from_cache)[:10], 1):
                match_name = url.split('/')[-1].replace('.html', '')
                print(f"  {i:2d}. {match_name}")
    
    except:
        print(f"Could not compare with cache")
    
    return all_discovered_urls

def test_specific_url_patterns():
    """Test if specific URL patterns from user's list exist"""
    print(f"\n🧪 TESTING USER'S URL PATTERNS")
    print("="*50)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Build URLs from user's comprehensive list
    sample_matches = [
        "20250813-M-Cincinnati_Masters-F-Jannik_Sinner-Frances_Tiafoe",
        "20250812-W-Cincinnati-F-Aryna_Sabalenka-Jessica_Pegula", 
        "20250811-M-Cincinnati_Masters-SF-Alexander_Zverev-Ben_Shelton",
        "20250810-W-Cincinnati-SF-Paula_Badosa-Iga_Swiatek",
        "20250806-W-Montreal-F-Jessica_Pegula-Amanda_Anisimova",
        "20250805-M-Canada_Masters-F-Alexei_Popyrin-Andrey_Rublev",
        "20250801-M-Canada_Masters-QF-Tommy_Paul-Sebastian_Korda",
        "20250731-W-Montreal-QF-Diana_Shnaider-Peyton_Stearns",
    ]
    
    existing_count = 0
    not_found_count = 0
    
    for match in sample_matches:
        url = f"https://www.tennisabstract.com/charting/{match}.html"
        
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                existing_count += 1
                print(f"  ✅ EXISTS: {match}")
            else:
                not_found_count += 1
                print(f"  ❌ NOT FOUND: {match}")
        except:
            not_found_count += 1
            print(f"  💥 ERROR: {match}")
        
        time.sleep(0.5)
    
    print(f"\n📊 URL PATTERN TEST:")
    print(f"Existing: {existing_count}")
    print(f"Not found: {not_found_count}")
    
def check_site_structure():
    """Check if Tennis Abstract has a different site structure we're missing"""
    print(f"\n🏗️  ANALYZING SITE STRUCTURE")
    print("="*50)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    try:
        # Check main page source for clues
        response = requests.get("https://www.tennisabstract.com/charting/", headers=headers, timeout=15)
        
        if response.status_code == 200:
            content = response.text
            
            # Look for pagination
            if 'pagination' in content.lower() or 'page' in content.lower():
                print("⚠️  Possible pagination detected")
            
            # Look for JavaScript loading
            if 'ajax' in content.lower() or 'fetch(' in content or 'XMLHttpRequest' in content:
                print("⚠️  Dynamic content loading detected")
            
            # Look for different sections
            sections = re.findall(r'<h[1-6][^>]*>([^<]+)</h[1-6]>', content, re.IGNORECASE)
            if sections:
                print(f"📋 Page sections found:")
                for section in sections[:10]:
                    print(f"  - {section.strip()}")
            
            # Look for links to other listing pages
            other_pages = re.findall(r'href="(/[^"]*\.html)"', content)
            unique_pages = list(set([p for p in other_pages if 'charting' in p and p != '/charting/']))
            
            if unique_pages:
                print(f"🔗 Other charting pages found:")
                for page in unique_pages[:10]:
                    print(f"  - https://www.tennisabstract.com{page}")
        
    except Exception as e:
        print(f"Error analyzing site structure: {e}")

def main():
    """Run comprehensive diagnosis"""
    print("🚀 COMPREHENSIVE TENNIS ABSTRACT DISCOVERY DIAGNOSIS")
    print("="*80)
    print("Goal: Find out why we're missing ~73 matches from user's list")
    print()
    
    # Test current discovery methods
    discovered_urls = test_discovery_methods()
    
    # Test user's URL patterns
    test_specific_url_patterns()
    
    # Check site structure
    check_site_structure()
    
    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    print(f"Our discovery found: {len(discovered_urls)} URLs")
    print(f"User's list shows: 196 matches") 
    print(f"Gap: {196 - len(discovered_urls)} matches")
    
    print(f"\n💡 POTENTIAL ISSUES:")
    print(f"1. Dynamic content loading (JavaScript)")
    print(f"2. Pagination or multiple index pages")
    print(f"3. URLs in different formats/locations")
    print(f"4. Time-based or rolling updates")
    print(f"5. Regional or access-based variations")

if __name__ == "__main__":
    main()