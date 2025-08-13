#!/usr/bin/env python3
"""
Discover Tennis Abstract matches from their recent additions page
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, date, timedelta
import pandas as pd

def discover_tennis_abstract_recent():
    """Discover Tennis Abstract matches from their recent page"""
    print("🔍 DISCOVERING TENNIS ABSTRACT RECENT MATCHES")
    print("="*70)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Check the recent additions page
    recent_url = "https://www.tennisabstract.com/charting/recent.html"
    print(f"Fetching recent additions: {recent_url}")
    
    try:
        response = requests.get(recent_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to match pages
            match_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if href.startswith('/charting/') and href.endswith('.html') and len(href.split('-')) >= 5:
                    full_url = f"https://www.tennisabstract.com{href}"
                    match_links.append(full_url)
            
            print(f"✅ Found {len(match_links)} potential matches from recent page")
            return match_links
        else:
            print(f"❌ Failed to fetch recent page: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching recent page: {e}")
        return []

def discover_tennis_abstract_meta():
    """Discover Tennis Abstract matches from meta page"""
    print(f"\n🔍 DISCOVERING FROM META PAGE")
    print("-" * 50)
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    meta_url = "https://www.tennisabstract.com/charting/meta.html"
    print(f"Fetching meta page: {meta_url}")
    
    try:
        response = requests.get(meta_url, headers=headers, timeout=15)
        if response.status_code == 200:
            content = response.text
            
            # Find all match URLs in the content
            match_pattern = r'href="/charting/(\d{8}-[MW]-[^-]+-[^-]+-[^-]+-[^"]+\.html)"'
            matches = re.findall(match_pattern, content)
            
            full_urls = [f"https://www.tennisabstract.com/charting/{match}" for match in matches]
            
            # Filter for recent matches (post 6/10/2025)
            recent_urls = []
            cutoff_date = date(2025, 6, 10)
            
            for url in full_urls:
                match_file = url.split('/')[-1]
                try:
                    date_str = match_file[:8]  # First 8 chars should be YYYYMMDD
                    match_date = datetime.strptime(date_str, '%Y%m%d').date()
                    if match_date >= cutoff_date:
                        recent_urls.append(url)
                except:
                    continue
            
            print(f"✅ Found {len(full_urls)} total matches, {len(recent_urls)} recent")
            return recent_urls
        else:
            print(f"❌ Failed to fetch meta page: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching meta page: {e}")
        return []

def generate_systematic_urls():
    """Generate URLs systematically for major tournaments"""
    print(f"\n🔍 GENERATING SYSTEMATIC TOURNAMENT URLS")
    print("-" * 50)
    
    urls_to_test = []
    
    # Major tournaments with known patterns
    tournaments = [
        # Wimbledon 2025
        ('20250630', '20250713', ['Wimbledon'], ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']),
        
        # French Open lead-up
        ('20250615', '20250622', ['Queens_Club', 's_Hertogenbosch', 'Halle'], ['R32', 'R16', 'QF', 'SF', 'F']),
        ('20250622', '20250629', ['Mallorca', 'Bad_Homburg', 'Eastbourne'], ['R32', 'R16', 'QF', 'SF', 'F']),
        
        # Summer hard court
        ('20250714', '20250721', ['Hamburg', 'Newport', 'Bastad', 'Palermo'], ['R32', 'R16', 'QF', 'SF', 'F']),
        ('20250722', '20250728', ['Umag', 'Atlanta', 'Washington'], ['R32', 'R16', 'QF', 'SF', 'F']),
        ('20250805', '20250811', ['Montreal', 'Toronto', 'Cincinnati'], ['R32', 'R16', 'QF', 'SF', 'F']),
        
        # Olympics period
        ('20250729', '20250803', ['Olympics', 'Kitzbuhel'], ['R32', 'R16', 'QF', 'SF', 'F']),
    ]
    
    # Top players for URL generation
    top_players = {
        'M': [
            'Carlos_Alcaraz', 'Jannik_Sinner', 'Novak_Djokovic', 'Daniil_Medvedev',
            'Alexander_Zverev', 'Andrey_Rublev', 'Taylor_Fritz', 'Stefanos_Tsitsipas',
            'Holger_Rune', 'Casper_Ruud', 'Alex_De_Minaur', 'Tommy_Paul',
            'Lorenzo_Musetti', 'Ben_Shelton', 'Frances_Tiafoe', 'Sebastian_Korda'
        ],
        'W': [
            'Iga_Swiatek', 'Aryna_Sabalenka', 'Coco_Gauff', 'Elena_Rybakina',
            'Jessica_Pegula', 'Ons_Jabeur', 'Marketa_Vondrousova', 'Qinwen_Zheng',
            'Maria_Sakkari', 'Barbora_Krejcikova', 'Danielle_Collins', 'Madison_Keys',
            'Belinda_Bencic', 'Petra_Kvitova', 'Victoria_Azarenka', 'Amanda_Anisimova'
        ]
    }
    
    for start_str, end_str, tournament_list, rounds in tournaments:
        start_date = datetime.strptime(start_str, '%Y%m%d').date()
        end_date = datetime.strptime(end_str, '%Y%m%d').date()
        
        current_date = start_date
        while current_date <= end_date:
            for tournament in tournament_list:
                for round_name in rounds:
                    for gender in ['M', 'W']:
                        players = top_players[gender]
                        
                        # Try combinations of top players
                        for i in range(min(6, len(players))):
                            for j in range(i+1, min(8, len(players))):
                                player1 = players[i]
                                player2 = players[j]
                                
                                url = f"https://www.tennisabstract.com/charting/{current_date.strftime('%Y%m%d')}-{gender}-{tournament}-{round_name}-{player1}-{player2}.html"
                                urls_to_test.append(url)
            
            current_date += timedelta(days=1)
    
    print(f"✅ Generated {len(urls_to_test)} systematic URLs to test")
    return urls_to_test

def test_tennis_abstract_urls(urls, max_test=500):
    """Test which Tennis Abstract URLs exist"""
    print(f"\n🔍 TESTING TENNIS ABSTRACT URLS")
    print("-" * 50)
    print(f"Testing {min(max_test, len(urls))} URLs...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    existing_urls = []
    
    for i, url in enumerate(urls[:max_test], 1):
        if i % 50 == 0:
            print(f"  📊 Progress: {i}/{min(max_test, len(urls))} ({i/min(max_test, len(urls))*100:.1f}%)")
        
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                existing_urls.append(url)
                match_name = url.split('/')[-1][:40]
                print(f"  ✅ Found: {match_name}...")
        except:
            continue
        
        time.sleep(0.3)  # Be respectful
    
    print(f"\n✅ Found {len(existing_urls)} existing Tennis Abstract matches")
    return existing_urls

def main():
    """Main discovery process"""
    print("🚀 COMPREHENSIVE TENNIS ABSTRACT DISCOVERY")
    print("="*80)
    print("Goal: Find hundreds of Tennis Abstract matches from 6/10 to today")
    print()
    
    all_discovered_urls = set()
    
    # Method 1: Recent additions page
    recent_urls = discover_tennis_abstract_recent()
    all_discovered_urls.update(recent_urls)
    
    # Method 2: Meta page parsing
    meta_urls = discover_tennis_abstract_meta()
    all_discovered_urls.update(meta_urls)
    
    # Method 3: Systematic generation
    systematic_urls = generate_systematic_urls()
    
    # Combine all discovery methods
    all_urls = list(all_discovered_urls) + systematic_urls
    unique_urls = list(set(all_urls))  # Remove duplicates
    
    print(f"\n📊 DISCOVERY SUMMARY")
    print("-" * 50)
    print(f"Recent page: {len(recent_urls)} URLs")
    print(f"Meta page: {len(meta_urls)} URLs")
    print(f"Systematic: {len(systematic_urls)} URLs")
    print(f"Total unique: {len(unique_urls)} URLs")
    
    # Test URLs to find existing ones
    existing_urls = test_tennis_abstract_urls(unique_urls, max_test=800)
    
    if existing_urls:
        # Save discovered URLs
        df = pd.DataFrame({'url': existing_urls})
        df['match_name'] = df['url'].apply(lambda x: x.split('/')[-1].replace('.html', ''))
        df['date_str'] = df['match_name'].apply(lambda x: x[:8])
        
        try:
            df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
            df = df.sort_values('date', ascending=False)  # Most recent first
            
            print(f"\n📅 DATE RANGE OF DISCOVERED MATCHES:")
            print(f"   Earliest: {df['date'].min().date()}")
            print(f"   Latest: {df['date'].max().date()}")
            
            # Show tournament distribution
            df['tournament'] = df['match_name'].apply(lambda x: x.split('-')[2] if len(x.split('-')) >= 3 else 'Unknown')
            print(f"\n🏆 TOURNAMENT BREAKDOWN:")
            tournament_counts = df['tournament'].value_counts().head(10)
            for tournament, count in tournament_counts.items():
                print(f"   {tournament}: {count}")
            
        except:
            pass
        
        # Save results
        output_file = '/Users/danielkim/Desktop/t3nn1s/discovered_tennis_abstract_urls.txt'
        with open(output_file, 'w') as f:
            for url in existing_urls:
                f.write(url + '\n')
        
        print(f"\n💾 Saved {len(existing_urls)} URLs to: {output_file}")
        return existing_urls
    else:
        print(f"\n❌ No Tennis Abstract matches discovered")
        return []

if __name__ == "__main__":
    urls = main()
    
    if urls:
        print(f"\n🎉 DISCOVERY COMPLETED!")
        print(f"Found {len(urls)} Tennis Abstract matches ready for scraping!")
    else:
        print(f"\n⚠️  No matches discovered - need to investigate further")