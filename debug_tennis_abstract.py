#!/usr/bin/env python3
"""
Debug Tennis Abstract site to find ALL matches - no compromises!
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, date, timedelta
import time

def debug_tennis_abstract_structure():
    """Debug the Tennis Abstract site structure to understand how matches are organized"""
    print("🔍 DEBUGGING TENNIS ABSTRACT SITE STRUCTURE")
    print("="*80)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Test multiple entry points
    urls_to_test = [
        ("Main Charting Page", "https://www.tennisabstract.com/charting/"),
        ("Recent Additions", "https://www.tennisabstract.com/charting/recent.html"),
        ("Meta Page", "https://www.tennisabstract.com/charting/meta.html"),
    ]
    
    for name, url in urls_to_test:
        print(f"\n📋 TESTING: {name}")
        print(f"URL: {url}")
        print("-" * 60)
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for match links
                match_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if '/charting/' in href and href.endswith('.html'):
                        # Check if this looks like a match (YYYYMMDD pattern)
                        filename = href.split('/')[-1]
                        if len(filename) >= 8 and filename[:8].isdigit():
                            match_links.append(href)
                
                print(f"Match links found: {len(match_links)}")
                
                # Show examples
                if match_links:
                    print(f"Sample links:")
                    for i, link in enumerate(match_links[:5]):
                        filename = link.split('/')[-1]
                        date_str = filename[:8]
                        try:
                            match_date = datetime.strptime(date_str, '%Y%m%d').date()
                            if match_date >= date(2025, 6, 10):
                                print(f"  ✅ {match_date}: {filename}")
                        except:
                            print(f"  ❓ {filename}")
                
                # Check for any pagination or "load more" elements
                load_more = soup.find_all(string=re.compile(r"(?i)(more|next|load|page)"))
                if load_more:
                    print(f"Potential pagination found: {len(load_more)} elements")
                    for element in load_more[:3]:
                        print(f"  - '{element.strip()}'")
                
                # Look for JavaScript or dynamic loading
                scripts = soup.find_all('script')
                js_content = ' '.join([script.get_text() for script in scripts])
                if 'ajax' in js_content.lower() or 'fetch' in js_content.lower():
                    print(f"⚠️  Potential dynamic content detected")
                
            else:
                print(f"❌ Failed to fetch: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)  # Be respectful

def test_specific_date_ranges():
    """Test if specific recent dates have matches"""
    print(f"\n🎯 TESTING SPECIFIC RECENT DATES")
    print("="*80)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Test recent dates that should have matches
    test_dates = [
        date(2025, 8, 11),  # Today
        date(2025, 8, 10),  # Yesterday
        date(2025, 8, 9),   # Few days ago
        date(2025, 8, 5),   # Week ago
        date(2025, 8, 1),   # Month start
        date(2025, 7, 28),  # Late July
        date(2025, 7, 20),  # Mid July
        date(2025, 7, 15),  # Last known good date
    ]
    
    # Common tournaments and players for URL construction
    tournaments = ['Cincinnati', 'Montreal', 'Toronto', 'Olympics', 'Washington', 'Hamburg', 'Atlanta']
    top_players = {
        'M': ['Carlos_Alcaraz', 'Jannik_Sinner', 'Novak_Djokovic', 'Daniil_Medvedev', 'Alexander_Zverev'],
        'W': ['Iga_Swiatek', 'Aryna_Sabalenka', 'Coco_Gauff', 'Elena_Rybakina', 'Jessica_Pegula']
    }
    rounds = ['F', 'SF', 'QF', 'R16', 'R32']
    
    found_matches = []
    
    for test_date in test_dates:
        print(f"\n📅 Testing {test_date}...")
        date_str = test_date.strftime('%Y%m%d')
        
        # Test some likely URL combinations
        test_urls = []
        
        for tournament in tournaments:
            for gender in ['M', 'W']:
                for round_name in rounds:
                    players = top_players[gender]
                    for i in range(min(2, len(players))):
                        for j in range(i+1, min(3, len(players))):
                            url = f"https://www.tennisabstract.com/charting/{date_str}-{gender}-{tournament}-{round_name}-{players[i]}-{players[j]}.html"
                            test_urls.append(url)
        
        # Test a sample of URLs for this date
        tested = 0
        for url in test_urls[:10]:  # Test first 10 combinations
            try:
                response = requests.head(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    found_matches.append(url)
                    match_name = url.split('/')[-1]
                    print(f"  ✅ FOUND: {match_name}")
                tested += 1
            except:
                continue
            
            time.sleep(0.5)  # Be respectful
        
        if tested > 0 and not any(test_date.strftime('%Y%m%d') in url for url in found_matches):
            print(f"  ❌ No matches found for {test_date}")
    
    print(f"\n📊 DISCOVERY SUMMARY")
    print(f"Found {len(found_matches)} additional matches")
    
    if found_matches:
        print(f"\nNew matches found:")
        for url in found_matches:
            match_name = url.split('/')[-1][:60]
            print(f"  - {match_name}")
    
    return found_matches

def check_if_site_changed():
    """Check if the Tennis Abstract site structure has changed"""
    print(f"\n🔍 CHECKING IF SITE STRUCTURE CHANGED")
    print("="*80)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Test a known working URL from our cache
    test_url = "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html"
    
    print(f"Testing known working URL: {test_url}")
    
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Known URL still works")
            
            # Check if the content structure is still the same
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for table structure
            tables = soup.find_all('table')
            print(f"Tables found: {len(tables)}")
            
            # Look for JavaScript tables
            js_tables = re.findall(r'var\s+(\w+)\s*=\s*\[', response.text)
            print(f"JavaScript tables found: {js_tables}")
            
            if 'pointlog' in response.text:
                print(f"✅ Point-by-point data still available")
            else:
                print(f"⚠️  Point-by-point data structure may have changed")
                
        else:
            print(f"❌ Known URL failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing known URL: {e}")

def main():
    """Debug Tennis Abstract comprehensively"""
    print("🚀 COMPREHENSIVE TENNIS ABSTRACT DEBUG")
    print("="*80)
    print("Goal: Find out why we're missing recent matches and fix it!")
    print()
    
    # Step 1: Debug site structure
    debug_tennis_abstract_structure()
    
    # Step 2: Test specific dates
    additional_matches = test_specific_date_ranges()
    
    # Step 3: Check if site changed
    check_if_site_changed()
    
    # Summary
    print(f"\n🎯 DEBUG SUMMARY")
    print("="*80)
    
    if additional_matches:
        print(f"✅ Found {len(additional_matches)} additional matches")
        print(f"📝 Recommendation: Update URL discovery method")
    else:
        print(f"⚠️  No additional matches found with current method")
        print(f"📝 Recommendation: Tennis Abstract may have limited recent coverage")
    
    print(f"\nNext steps:")
    print(f"1. If additional matches found: Scrape them")
    print(f"2. If no additional matches: Verify our 111 matches are complete coverage")
    print(f"3. Focus on maximizing the data we have")

if __name__ == "__main__":
    main()