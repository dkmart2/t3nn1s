#!/usr/bin/env python3
"""
API-Tennis Integration
Fetches recent tennis match data from API-Tennis for post-6/10/2025 matches
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, date, timedelta
import json
import os
from pathlib import Path
from settings import TENNIS_CACHE_DIR, API_TENNIS_KEY, BASE_API_URL
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITennisClient:
    """Client for API-Tennis data fetching"""
    
    def __init__(self):
        self.api_key = API_TENNIS_KEY
        self.base_url = BASE_API_URL or "https://api.api-tennis.com/tennis/"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'api-tennis.p.rapidapi.com'
        }
        self.cache_dir = Path(TENNIS_CACHE_DIR) / 'api_tennis'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def fetch_matches_by_date(self, session, date_str):
        """Fetch matches for a specific date"""
        url = f"{self.base_url}matches"
        params = {
            'date': date_str,
            'timezone': 'UTC'
        }
        
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    logger.warning(f"API returned status {response.status} for date {date_str}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching matches for {date_str}: {e}")
            return []
    
    async def fetch_match_statistics(self, session, match_id):
        """Fetch detailed statistics for a specific match"""
        url = f"{self.base_url}match/statistics"
        params = {'id': match_id}
        
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', {})
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error fetching stats for match {match_id}: {e}")
            return {}
    
    async def fetch_recent_matches(self, start_date, end_date):
        """Fetch all matches between start_date and end_date"""
        print(f"\n🌐 FETCHING API-TENNIS DATA")
        print(f"Date range: {start_date} to {end_date}")
        
        if not self.api_key:
            print("❌ No API-Tennis key configured")
            return pd.DataFrame()
        
        all_matches = []
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for all dates
            current_date = start_date
            tasks = []
            dates = []
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                dates.append(date_str)
                tasks.append(self.fetch_matches_by_date(session, date_str))
                current_date += timedelta(days=1)
            
            print(f"📡 Fetching {len(tasks)} days of match data...")
            
            # Fetch all dates concurrently
            results = await asyncio.gather(*tasks)
            
            # Process results
            for date_str, matches in zip(dates, results):
                for match in matches:
                    try:
                        # Extract match data
                        match_data = {
                            'date': datetime.strptime(date_str, '%Y-%m-%d'),
                            'match_id': match.get('id'),
                            'tournament': match.get('league', {}).get('name'),
                            'round': match.get('round'),
                            'status': match.get('status', {}).get('long'),
                            'source': 'api_tennis',
                            'source_rank': 2
                        }
                        
                        # Extract teams/players
                        teams = match.get('teams', {})
                        if teams:
                            home = teams.get('home', {})
                            away = teams.get('away', {})
                            
                            match_data['player1'] = home.get('name', 'Unknown')
                            match_data['player2'] = away.get('name', 'Unknown')
                            
                            # Determine winner if match is finished
                            if match_data['status'] == 'Match Finished':
                                if home.get('winner'):
                                    match_data['Winner'] = match_data['player1']
                                    match_data['Loser'] = match_data['player2']
                                elif away.get('winner'):
                                    match_data['Winner'] = match_data['player2']
                                    match_data['Loser'] = match_data['player1']
                        
                        # Extract scores
                        scores = match.get('scores', {})
                        if scores:
                            sets = scores.get('sets', [])
                            match_data['sets_played'] = len(sets)
                            
                            # Extract set scores
                            for i, set_score in enumerate(sets, 1):
                                match_data[f'set{i}_p1'] = set_score.get('home')
                                match_data[f'set{i}_p2'] = set_score.get('away')
                        
                        all_matches.append(match_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing match: {e}")
                        continue
            
            print(f"✅ Retrieved {len(all_matches)} matches from API-Tennis")
            
            # Fetch detailed statistics for finished matches (limit for demo)
            finished_matches = [m for m in all_matches if m.get('status') == 'Match Finished'][:5]
            
            if finished_matches:
                print(f"📊 Fetching detailed stats for {len(finished_matches)} matches...")
                
                stat_tasks = [self.fetch_match_statistics(session, m['match_id']) 
                            for m in finished_matches if m.get('match_id')]
                stat_results = await asyncio.gather(*stat_tasks)
                
                # Add stats to matches
                for match, stats in zip(finished_matches, stat_results):
                    if stats:
                        # Extract relevant statistics
                        for group in stats.get('statistics', []):
                            for stat in group.get('data', []):
                                stat_name = stat.get('name', '').lower().replace(' ', '_')
                                match[f'stat_{stat_name}_p1'] = stat.get('home')
                                match[f'stat_{stat_name}_p2'] = stat.get('away')
        
        if all_matches:
            df = pd.DataFrame(all_matches)
            
            # Add composite_id for deduplication
            if 'Winner' in df.columns and 'Loser' in df.columns:
                df['composite_id'] = (
                    df['Winner'].astype(str) + "_" + 
                    df['Loser'].astype(str) + "_" + 
                    df['date'].dt.strftime('%Y%m%d')
                )
            
            # Cache the results
            cache_file = self.cache_dir / f'api_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            df.to_parquet(cache_file, index=False)
            print(f"💾 Cached {len(df)} matches to: {cache_file}")
            
            return df
        else:
            print("❌ No matches retrieved from API-Tennis")
            return pd.DataFrame()
    
    def get_cached_matches(self):
        """Load previously cached API-Tennis matches"""
        cache_files = list(self.cache_dir.glob('api_matches_*.parquet'))
        
        if not cache_files:
            return pd.DataFrame()
        
        # Load most recent cache
        latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_cache)
        print(f"📂 Loaded {len(df)} cached API-Tennis matches from {latest_cache.name}")
        return df

async def fetch_recent_api_tennis_data(start_date=None, end_date=None):
    """Main function to fetch API-Tennis data"""
    if not start_date:
        start_date = date(2025, 6, 10)  # Jeff data cutoff
    if not end_date:
        end_date = date.today()
    
    client = APITennisClient()
    
    # Try to fetch new data
    df = await client.fetch_recent_matches(start_date, end_date)
    
    # If no new data, try cache
    if df.empty:
        df = client.get_cached_matches()
    
    return df

def main():
    """Test API-Tennis integration"""
    print("🎾 API-TENNIS INTEGRATION TEST")
    print("="*50)
    
    # Test with a small date range
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 11)
    
    # Run async fetch
    df = asyncio.run(fetch_recent_api_tennis_data(start_date, end_date))
    
    if not df.empty:
        print(f"\n📊 API-TENNIS RESULTS")
        print(f"Total matches: {len(df)}")
        
        if 'date' in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            print(f"Match status breakdown:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count}")
        
        if 'tournament' in df.columns:
            tournaments = df['tournament'].value_counts().head(5)
            print(f"Top tournaments:")
            for tournament, count in tournaments.items():
                print(f"  - {tournament}: {count}")
        
        return df
    else:
        print("\n⚠️  No API-Tennis data retrieved")
        print("Possible issues:")
        print("1. API key may be invalid")
        print("2. API limits may be reached")
        print("3. No matches in date range")
        
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()