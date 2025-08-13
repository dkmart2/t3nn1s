#!/usr/bin/env python3
"""
Tennis Abstract Scraper
Scrapes match data from Tennis Abstract for recent matches (post-6/10/2025)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, date, timedelta
import time
import logging
from pathlib import Path
import json
import os
from settings import TENNIS_CACHE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TennisAbstractScraper:
    """Scraper for Tennis Abstract match data"""
    
    def __init__(self):
        self.base_url = "https://www.tennisabstract.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cache_dir = Path(TENNIS_CACHE_DIR) / 'tennis_abstract'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_recent_matches_urls(self, days_back=60):
        """Get URLs for recent match charts"""
        print(f"🔍 Finding Tennis Abstract matches from last {days_back} days")
        
        # Tennis Abstract uses specific URL patterns
        # Men's matches: /charting-m-{date}-{players}.html
        # Women's matches: /charting-w-{date}-{players}.html
        
        match_urls = []
        
        # Try to get the index/recent matches page
        try:
            # Check for recent match listings
            for gender in ['m', 'w']:
                gender_name = 'men' if gender == 'm' else 'women'
                print(f"\nChecking {gender_name}'s recent matches...")
                
                # Try different potential URLs
                potential_urls = [
                    f"{self.base_url}/cgi-bin/player-classic.cgi?p=Recent&f=A{datetime.now().year}qq",
                    f"{self.base_url}/recent-{gender}-matches.html",
                    f"{self.base_url}/charting-{gender}-recent.html"
                ]
                
                for url in potential_urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            # Find links to match charts
                            chart_links = soup.find_all('a', href=lambda x: x and f'charting-{gender}' in x)
                            
                            for link in chart_links:
                                href = link.get('href')
                                if href:
                                    full_url = href if href.startswith('http') else f"{self.base_url}/{href}"
                                    match_urls.append(full_url)
                                    
                            if chart_links:
                                print(f"  ✓ Found {len(chart_links)} {gender_name}'s match charts")
                                break
                    except Exception as e:
                        continue
                        
        except Exception as e:
            logger.error(f"Error getting match URLs: {e}")
        
        return match_urls
    
    def scrape_match_data(self, url):
        """Scrape data from a single match page"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match details from the page
            match_data = {}
            
            # Try to extract from URL (format: charting-m-20250610-...)
            url_parts = url.split('/')[-1].split('-')
            if len(url_parts) >= 3:
                gender_code = url_parts[1]  # 'm' or 'w'
                date_str = url_parts[2]  # YYYYMMDD
                
                try:
                    match_data['date'] = datetime.strptime(date_str, '%Y%m%d')
                    match_data['gender'] = gender_code.upper()
                except:
                    pass
            
            # Extract player names
            title = soup.find('title')
            if title:
                title_text = title.text
                # Usually format: "Player1 vs Player2 - Tennis Abstract"
                if ' vs ' in title_text:
                    players = title_text.split(' vs ')
                    if len(players) >= 2:
                        match_data['player1'] = players[0].strip()
                        match_data['player2'] = players[1].split('-')[0].strip()
            
            # Extract match stats from tables
            tables = soup.find_all('table')
            for table in tables:
                # Look for stats tables
                headers = [th.text.strip() for th in table.find_all('th')]
                if 'Aces' in headers or 'Winners' in headers:
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows:
                        cells = row.find_all('td')
                        if cells:
                            # Extract stats here
                            pass
            
            # Extract point-by-point data if available
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'pointdata' in script.string.lower():
                    # Parse JavaScript point data
                    match_data['has_point_data'] = True
                    
            return match_data if match_data else None
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_recent_matches(self, start_date=None, end_date=None):
        """Main method to scrape recent matches"""
        print("\n🎾 TENNIS ABSTRACT SCRAPER")
        print("="*50)
        
        if not start_date:
            start_date = date(2025, 6, 10)  # Jeff data cutoff
        if not end_date:
            end_date = date.today()
            
        print(f"Target date range: {start_date} to {end_date}")
        
        # Get match URLs
        match_urls = self.get_recent_matches_urls(days_back=(end_date - start_date).days)
        
        if not match_urls:
            print("⚠️  No Tennis Abstract match URLs found")
            print("Note: Tennis Abstract may require different access method")
            return pd.DataFrame()
        
        print(f"\n📊 Found {len(match_urls)} potential matches to scrape")
        
        # Scrape each match
        all_matches = []
        for i, url in enumerate(match_urls, 1):
            print(f"  Scraping {i}/{len(match_urls)}: {url}")
            
            match_data = self.scrape_match_data(url)
            if match_data:
                match_data['source'] = 'tennis_abstract'
                match_data['source_rank'] = 1  # Highest priority
                match_data['url'] = url
                all_matches.append(match_data)
                
            # Be respectful with rate limiting
            time.sleep(1)
            
            # Limit for testing
            if i >= 10:
                print("  (Limited to 10 matches for testing)")
                break
        
        if all_matches:
            df = pd.DataFrame(all_matches)
            print(f"\n✅ Scraped {len(df)} matches from Tennis Abstract")
            
            # Save to cache
            cache_file = self.cache_dir / f'ta_matches_{datetime.now().strftime("%Y%m%d")}.json'
            df.to_json(cache_file, orient='records', date_format='iso')
            print(f"💾 Cached to: {cache_file}")
            
            return df
        else:
            print("\n❌ No matches successfully scraped")
            return pd.DataFrame()

def main():
    """Test the Tennis Abstract scraper"""
    scraper = TennisAbstractScraper()
    
    # Scrape recent matches
    matches_df = scraper.scrape_recent_matches()
    
    if not matches_df.empty:
        print(f"\n📊 SCRAPING RESULTS")
        print(f"Total matches: {len(matches_df)}")
        if 'date' in matches_df.columns:
            print(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
        if 'gender' in matches_df.columns:
            print(f"Gender breakdown: {matches_df['gender'].value_counts().to_dict()}")
        
        return matches_df
    else:
        print("\n⚠️  Tennis Abstract scraping needs alternative approach")
        print("Options:")
        print("1. Use Jeff's GitHub for updated CSVs")
        print("2. Use API-Tennis for recent matches")
        print("3. Manual download from Tennis Abstract")
        
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()