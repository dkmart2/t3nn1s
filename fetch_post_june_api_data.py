#!/usr/bin/env python3
"""
Fetch API-Tennis data for post-June 2025 matches
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import gzip
import hashlib
from typing import Dict, List, Optional
import time

# API configuration
API_TENNIS_KEY = "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb"
BASE_API_URL = "https://api.api-tennis.com/tennis/"
BASE_CUTOFF_DATE = datetime(2025, 6, 10)

class APITennisFetcher:
    def __init__(self):
        self.api_key = API_TENNIS_KEY
        self.base_url = BASE_API_URL
        self.cache_dir = Path("/Users/danielkim/Desktop/t3nn1s/api_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_cache_key(self, url: str, params: Dict) -> str:
        """Generate cache key for request"""
        cache_str = f"{url}_{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.json.gz"
        if cache_file.exists():
            with gzip.open(cache_file, 'rt') as f:
                return json.load(f)
        return None
    
    def save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json.gz"
        with gzip.open(cache_file, 'wt') as f:
            json.dump(data, f)
    
    async def fetch_fixtures_for_date(self, date: datetime) -> List[Dict]:
        """Fetch all fixtures for a specific date"""
        url = f"{self.base_url}?method=get_fixtures"
        params = {
            "APIkey": self.api_key,
            "date_start": date.strftime("%Y-%m-%d"),
            "date_stop": date.strftime("%Y-%m-%d")
        }
        
        cache_key = self.get_cache_key(url, params)
        cached = self.load_from_cache(cache_key)
        if cached:
            print(f"  Using cached data for {date.strftime('%Y-%m-%d')}")
            return cached.get('result', [])
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.save_to_cache(cache_key, data)
                    fixtures = data.get('result', [])
                    print(f"  Fetched {len(fixtures)} fixtures for {date.strftime('%Y-%m-%d')}")
                    return fixtures
                else:
                    print(f"  API error {response.status} for {date.strftime('%Y-%m-%d')}")
                    return []
        except Exception as e:
            print(f"  Error fetching {date.strftime('%Y-%m-%d')}: {e}")
            return []
    
    async def fetch_fixture_details(self, fixture_id: str) -> Optional[Dict]:
        """Fetch detailed statistics for a fixture"""
        url = f"{self.base_url}?method=get_fixture"
        params = {
            "APIkey": self.api_key,
            "event_key": fixture_id
        }
        
        cache_key = self.get_cache_key(url, params)
        cached = self.load_from_cache(cache_key)
        if cached:
            return cached.get('result')
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.save_to_cache(cache_key, data)
                    return data.get('result')
                else:
                    return None
        except Exception as e:
            print(f"    Error fetching fixture {fixture_id}: {e}")
            return None
    
    async def fetch_all_post_june_matches(self):
        """Fetch all matches after June 10, 2025"""
        print("\n=== FETCHING POST-JUNE 2025 API-TENNIS DATA ===")
        
        start_date = BASE_CUTOFF_DATE + timedelta(days=1)
        end_date = datetime.now()
        
        all_fixtures = []
        current_date = start_date
        
        # Fetch fixtures day by day
        while current_date <= end_date:
            fixtures = await self.fetch_fixtures_for_date(current_date)
            
            # Filter for completed matches only
            completed = [f for f in fixtures if f.get('event_final_result') != '-']
            all_fixtures.extend(completed)
            
            current_date += timedelta(days=1)
            
            # Rate limiting
            if len(all_fixtures) % 100 == 0 and len(all_fixtures) > 0:
                print(f"  Progress: {len(all_fixtures)} matches fetched...")
                await asyncio.sleep(1)  # Rate limit
        
        print(f"\nTotal fixtures found: {len(all_fixtures)}")
        
        # Now fetch detailed statistics for each fixture
        detailed_fixtures = []
        
        for i, fixture in enumerate(all_fixtures[:100]):  # Limit to 100 for testing
            fixture_id = fixture.get('event_key')
            if fixture_id:
                details = await self.fetch_fixture_details(fixture_id)
                if details:
                    # Combine basic and detailed data
                    combined = {**fixture, **details}
                    detailed_fixtures.append(combined)
                
                if (i + 1) % 10 == 0:
                    print(f"  Detailed stats fetched: {i + 1}/{min(100, len(all_fixtures))}")
                    await asyncio.sleep(1)  # Rate limit
        
        return detailed_fixtures

def process_api_fixtures_to_dataframe(fixtures: List[Dict]) -> pd.DataFrame:
    """Convert API fixtures to DataFrame format"""
    records = []
    
    for fixture in fixtures:
        try:
            # Extract basic match info
            record = {
                'event_key': fixture.get('event_key'),
                'date': fixture.get('event_date'),
                'tournament': fixture.get('tournament_name'),
                'round': fixture.get('tournament_round'),
                'surface': None,  # Not provided in API
                'player1': fixture.get('event_first_player'),
                'player2': fixture.get('event_second_player'),
                'final_result': fixture.get('event_final_result'),
                'winner': fixture.get('event_winner'),
                'status': fixture.get('event_status'),
                'match_type': fixture.get('event_type_type'),
                'home_odds': None,
                'away_odds': None
            }
            
            # Extract odds if available
            if 'odds' in fixture:
                for odd in fixture.get('odds', []):
                    if odd.get('odd_bookmakers') == 'bet365':
                        record['home_odds'] = float(odd.get('home_od', 0))
                        record['away_odds'] = float(odd.get('away_od', 0))
                        break
            
            # Extract statistics if available
            if 'statistics' in fixture:
                for stat in fixture.get('statistics', []):
                    stat_name = stat.get('type', '').lower().replace(' ', '_')
                    record[f'{stat_name}_home'] = stat.get('home')
                    record[f'{stat_name}_away'] = stat.get('away')
            
            # Extract point progression if available
            if 'pointbypoint' in fixture:
                record['has_point_progression'] = True
                record['total_points'] = len(fixture.get('pointbypoint', []))
            else:
                record['has_point_progression'] = False
                record['total_points'] = 0
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing fixture {fixture.get('event_key')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"\nProcessed {len(df)} matches into DataFrame")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {len(df.columns)}")
    
    return df

async def main():
    """Main execution"""
    async with APITennisFetcher() as fetcher:
        # Fetch all post-June fixtures
        fixtures = await fetcher.fetch_all_post_june_matches()
        
        # Convert to DataFrame
        df = process_api_fixtures_to_dataframe(fixtures)
        
        # Save to CSV
        output_file = Path("/Users/danielkim/Desktop/t3nn1s/api_tennis_post_june_2025.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
        # Display sample
        print("\n=== SAMPLE DATA ===")
        print(df[['date', 'tournament', 'player1', 'player2', 'final_result']].head())
        
        # Statistics summary
        print("\n=== DATA SUMMARY ===")
        print(f"Total matches: {len(df)}")
        print(f"Matches with odds: {df['home_odds'].notna().sum()}")
        print(f"Matches with point progression: {df['has_point_progression'].sum()}")
        print(f"Unique tournaments: {df['tournament'].nunique()}")
        
        return df

if __name__ == "__main__":
    df = asyncio.run(main())
    print("\n✅ API-Tennis data fetch complete!")