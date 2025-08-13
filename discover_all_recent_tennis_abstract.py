#!/usr/bin/env python3
"""
Discover ALL recent Tennis Abstract matches by parsing the main page
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, date
import pandas as pd

def discover_all_recent_tennis_abstract():
    """Parse Tennis Abstract main page to find all recent matches"""
    print("🔍 DISCOVERING ALL RECENT TENNIS ABSTRACT MATCHES")
    print("="*70)
    
    url = "https://www.tennisabstract.com/charting/"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"❌ Failed to fetch Tennis Abstract: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links to match pages
        match_urls = []
        cutoff_date = date(2025, 6, 10)
        
        # Look for links with the pattern YYYYMMDD-X-Tournament-Round-Player1-Player2.html
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            
            # Check if this looks like a match URL
            if href.startswith('/charting/') and href.endswith('.html'):
                # Extract the filename
                filename = href.split('/')[-1]
                
                # Check if it matches the date pattern
                if len(filename) >= 8 and filename[:8].isdigit():
                    try:
                        date_str = filename[:8]
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                        
                        if match_date >= cutoff_date:
                            full_url = f"https://www.tennisabstract.com{href}"
                            match_urls.append((match_date, full_url))
                    except:
                        continue
        
        # Sort by date (newest first)
        match_urls.sort(key=lambda x: x[0], reverse=True)
        
        print(f"✅ Found {len(match_urls)} matches from {cutoff_date} onwards")
        
        if match_urls:
            print(f"\nDate range:")
            print(f"  Latest: {match_urls[0][0]}")
            print(f"  Earliest: {match_urls[-1][0]}")
            
            # Show some examples
            print(f"\nFirst 10 matches:")
            for i, (match_date, url) in enumerate(match_urls[:10]):
                match_name = url.split('/')[-1].replace('.html', '')
                print(f"  {i+1}. {match_date}: {match_name}")
        
        # Compare with our cached list
        print(f"\n🔍 COMPARING WITH CACHED LIST")
        print("-" * 50)
        
        cached_urls_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        try:
            with open(cached_urls_file, 'r') as f:
                cached_urls = set(line.strip() for line in f if line.strip())
            
            discovered_urls = set(url for _, url in match_urls)
            
            print(f"Cached URLs: {len(cached_urls)}")
            print(f"Discovered URLs: {len(discovered_urls)}")
            
            missing_from_cache = discovered_urls - cached_urls
            missing_from_discovery = cached_urls - discovered_urls
            
            print(f"Missing from cache: {len(missing_from_cache)}")
            print(f"Missing from discovery: {len(missing_from_discovery)}")
            
            if missing_from_cache:
                print(f"\n🆕 NEW MATCHES NOT IN CACHE:")
                for url in sorted(missing_from_cache):
                    match_name = url.split('/')[-1].replace('.html', '')[:50]
                    print(f"  - {match_name}")
                
                # Save all discovered URLs
                output_file = '/Users/danielkim/Desktop/t3nn1s/all_recent_tennis_abstract_urls.txt'
                with open(output_file, 'w') as f:
                    for _, url in match_urls:
                        f.write(url + '\n')
                
                print(f"\n💾 Saved all {len(match_urls)} URLs to: {output_file}")
                
                return [url for _, url in match_urls]
        
        except FileNotFoundError:
            print("❌ Cached URLs file not found")
            return [url for _, url in match_urls]
        
        return [url for _, url in match_urls]
        
    except Exception as e:
        print(f"❌ Error discovering matches: {e}")
        return []

def analyze_tennis_abstract_coverage():
    """Analyze the complete Tennis Abstract coverage"""
    urls = discover_all_recent_tennis_abstract()
    
    if not urls:
        return
    
    print(f"\n📊 TENNIS ABSTRACT COVERAGE ANALYSIS")
    print("="*70)
    
    # Parse tournament and date info
    matches_data = []
    
    for url in urls:
        filename = url.split('/')[-1].replace('.html', '')
        parts = filename.split('-')
        
        if len(parts) >= 5:
            try:
                date_str = parts[0]
                gender = parts[1]
                tournament = parts[2]
                round_name = parts[3]
                
                match_date = datetime.strptime(date_str, '%Y%m%d').date()
                
                matches_data.append({
                    'date': match_date,
                    'gender': gender,
                    'tournament': tournament,
                    'round': round_name,
                    'url': url,
                    'filename': filename
                })
            except:
                continue
    
    if matches_data:
        df = pd.DataFrame(matches_data)
        
        print(f"✅ Total matches analyzed: {len(df)}")
        
        # Tournament breakdown
        print(f"\n🏆 Tournament breakdown:")
        tournament_counts = df['tournament'].value_counts().head(10)
        for tournament, count in tournament_counts.items():
            print(f"  - {tournament}: {count}")
        
        # Gender breakdown
        print(f"\n⚥ Gender breakdown:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"  - {gender}: {count}")
        
        # Date distribution
        print(f"\n📅 Monthly breakdown:")
        df['month'] = df['date'].dt.to_period('M')
        monthly_counts = df['month'].value_counts().sort_index()
        for month, count in monthly_counts.items():
            print(f"  - {month}: {count}")
        
        # Major tournaments
        major_tournaments = ['Wimbledon', 'Cincinnati', 'Montreal', 'Toronto', 'Olympics']
        major_matches = df[df['tournament'].isin(major_tournaments)]
        
        if not major_matches.empty:
            print(f"\n🎾 Major tournament matches:")
            for tournament in major_tournaments:
                count = len(df[df['tournament'] == tournament])
                if count > 0:
                    print(f"  - {tournament}: {count}")

if __name__ == "__main__":
    analyze_tennis_abstract_coverage()