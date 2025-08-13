#!/usr/bin/env python3
"""
Robust Tennis Abstract URL discovery - handle all possible HTML formats
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import re
import time

def robust_tennis_abstract_discovery():
    """Ultra-robust Tennis Abstract URL discovery using multiple methods"""
    print("💪 ROBUST TENNIS ABSTRACT URL DISCOVERY")
    print("="*70)
    print("Using every possible method to find Tennis Abstract URLs")
    print()
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    cutoff_date = date(2025, 6, 10)
    
    main_url = "https://www.tennisabstract.com/charting/"
    
    try:
        response = requests.get(main_url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"❌ Failed to fetch main page: HTTP {response.status_code}")
            return []
        
        print(f"✅ Fetched Tennis Abstract main page")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        raw_html = response.text
        
        all_urls = set()
        
        # Method 1: Standard <a href> links
        print(f"\n🔍 Method 1: Standard HTML links")
        method1_count = 0
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/charting/' in href and href.endswith('.html'):
                if href.startswith('/'):
                    full_url = f"https://www.tennisabstract.com{href}"
                elif not href.startswith('http'):
                    full_url = f"https://www.tennisabstract.com/charting/{href}"
                else:
                    full_url = href
                
                if validate_match_url(full_url, cutoff_date):
                    all_urls.add(full_url)
                    method1_count += 1
        
        print(f"Found {method1_count} URLs with method 1")
        
        # Method 2: Regex search for .html files
        print(f"\n🔍 Method 2: Regex search for HTML files")
        method2_count = 0
        html_pattern = r'(\d{8}-[^\s"\'<>]+\.html)'
        matches = re.findall(html_pattern, raw_html)
        
        for match in matches:
            full_url = f"https://www.tennisabstract.com/charting/{match}"
            if validate_match_url(full_url, cutoff_date):
                if full_url not in all_urls:
                    method2_count += 1
                all_urls.add(full_url)
        
        print(f"Found {method2_count} additional URLs with method 2")
        
        # Method 3: Look for onclick or JavaScript events
        print(f"\n🔍 Method 3: JavaScript/onclick events") 
        method3_count = 0
        onclick_elements = soup.find_all(attrs={"onclick": True})
        for element in onclick_elements:
            onclick = element.get('onclick', '')
            if '.html' in onclick:
                html_match = re.search(r'(\d{8}-[^\s"\'<>]+\.html)', onclick)
                if html_match:
                    filename = html_match.group(1)
                    full_url = f"https://www.tennisabstract.com/charting/{filename}"
                    if validate_match_url(full_url, cutoff_date):
                        if full_url not in all_urls:
                            method3_count += 1
                        all_urls.add(full_url)
        
        print(f"Found {method3_count} additional URLs with method 3")
        
        # Method 4: Parse table data
        print(f"\n🔍 Method 4: Table data parsing")
        method4_count = 0
        for table in soup.find_all('table'):
            for cell in table.find_all(['td', 'th']):
                cell_text = cell.get_text()
                # Look for date patterns
                if re.match(r'^\d{4}-\d{2}-\d{2}', cell_text):
                    # Look for corresponding link in same row
                    row = cell.find_parent('tr')
                    if row:
                        for link in row.find_all('a', href=True):
                            href = link.get('href')
                            if href and '.html' in href:
                                if href.startswith('/'):
                                    full_url = f"https://www.tennisabstract.com{href}"
                                else:
                                    full_url = href
                                if validate_match_url(full_url, cutoff_date):
                                    if full_url not in all_urls:
                                        method4_count += 1
                                    all_urls.add(full_url)
        
        print(f"Found {method4_count} additional URLs with method 4")
        
        # Method 5: Extract from any text that looks like match filenames
        print(f"\n🔍 Method 5: Text pattern extraction")
        method5_count = 0
        
        # Look for patterns like: YYYYMMDD-X-Tournament-Round-Player1-Player2
        text_pattern = r'\b(\d{8}-[MW]-[^\s<>"\']{10,}\.html)\b'
        text_matches = re.findall(text_pattern, raw_html)
        
        for match in text_matches:
            full_url = f"https://www.tennisabstract.com/charting/{match}"
            if validate_match_url(full_url, cutoff_date):
                if full_url not in all_urls:
                    method5_count += 1
                all_urls.add(full_url)
        
        print(f"Found {method5_count} additional URLs with method 5")
        
        # Method 6: Alternative regex patterns
        print(f"\n🔍 Method 6: Alternative patterns")
        method6_count = 0
        
        alternative_patterns = [
            r'"(\d{8}-[^"]+\.html)"',  # Quoted filenames
            r"'(\d{8}-[^']+\.html)'",  # Single-quoted filenames
            r'>(\d{8}-[^<]+\.html)<',  # Between HTML tags
            r'href="([^"]*\d{8}[^"]*\.html)"',  # In href attributes
        ]
        
        for pattern in alternative_patterns:
            matches = re.findall(pattern, raw_html)
            for match in matches:
                if not match.startswith('http'):
                    full_url = f"https://www.tennisabstract.com/charting/{match}"
                else:
                    full_url = match
                    
                if validate_match_url(full_url, cutoff_date):
                    if full_url not in all_urls:
                        method6_count += 1
                    all_urls.add(full_url)
        
        print(f"Found {method6_count} additional URLs with method 6")
        
        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"Total unique URLs found: {len(all_urls)}")
        
        if len(all_urls) > 0:
            # Sort by date
            dated_urls = []
            for url in all_urls:
                try:
                    filename = url.split('/')[-1]
                    date_str = filename[:8]
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    dated_urls.append((match_date, url))
                except:
                    continue
            
            dated_urls.sort()
            print(f"Date range: {dated_urls[0][0]} to {dated_urls[-1][0]}")
            
            # Show samples
            print(f"\n📋 Sample URLs found:")
            for i, (match_date, url) in enumerate(dated_urls[:10]):
                filename = url.split('/')[-1].replace('.html', '')
                print(f"  {i+1:2d}. {match_date}: {filename[:60]}")
            
            return [url for _, url in dated_urls]
        else:
            print(f"❌ No URLs found with any method!")
            
            # Debug: show raw HTML structure
            print(f"\n🔍 DEBUG: Raw HTML analysis")
            print(f"HTML length: {len(raw_html)} characters")
            print(f"Number of <a> tags: {len(soup.find_all('a'))}")
            print(f"Links with href: {len(soup.find_all('a', href=True))}")
            
            # Show first few links for debugging
            all_links = soup.find_all('a', href=True)[:5]
            for i, link in enumerate(all_links):
                print(f"  Link {i+1}: href='{link.get('href')}' text='{link.get_text()[:30]}'")
            
            return []
            
    except Exception as e:
        print(f"❌ Error in robust discovery: {e}")
        return []

def validate_match_url(url, cutoff_date):
    """Validate that a URL is a proper match URL after cutoff date"""
    try:
        if not url.startswith('http'):
            return False
        if not url.endswith('.html'):
            return False
        
        filename = url.split('/')[-1]
        if len(filename) < 12:
            return False
        if not filename[:8].isdigit():
            return False
        
        date_str = filename[:8]
        match_date = datetime.strptime(date_str, '%Y%m%d').date()
        
        return match_date >= cutoff_date
        
    except:
        return False

if __name__ == "__main__":
    urls = robust_tennis_abstract_discovery()
    
    print(f"\n🎯 FINAL RESULT:")
    if len(urls) >= 150:  # Close to user's 196
        print(f"🎉 SUCCESS! Found {len(urls)} Tennis Abstract URLs")
        print(f"This matches user expectations of comprehensive coverage")
    elif len(urls) > 50:
        print(f"⚠️  Partial success: Found {len(urls)} URLs")
        print(f"Still missing some matches compared to user's 196")
    else:
        print(f"❌ Discovery failed: Only found {len(urls)} URLs")
        print(f"Need to investigate Tennis Abstract site structure further")