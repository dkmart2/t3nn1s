#!/usr/bin/env python3
"""
Fetch ALL API-Tennis data for post-June 2025 matches
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
        self.cache_dir = Path("/Users/danielkim/Desktop/t3nn1s/api_cache_full")
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
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
            try:
                with gzip.open(cache_file, 'rt') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json.gz"
        try:
            with gzip.open(cache_file, 'wt') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Cache save error: {e}")
    
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
            result = cached.get('result', [])
            if result and result != [{"param": "date_start", "msg": "Required parameter missing", "cod": 201}]:
                return result
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.save_to_cache(cache_key, data)
                    fixtures = data.get('result', [])
                    
                    # Filter out error responses
                    if fixtures and isinstance(fixtures, list) and len(fixtures) > 0:
                        if isinstance(fixtures[0], dict) and 'event_key' in fixtures[0]:
                            return fixtures
                    return []
                else:
                    return []
        except Exception as e:
            print(f"  Error fetching {date.strftime('%Y-%m-%d')}: {e}")
            return []
    
    async def fetch_all_post_june_matches(self):
        """Fetch ALL matches after June 10, 2025"""
        print("\n=== FETCHING ALL POST-JUNE 2025 API-TENNIS DATA ===")
        
        start_date = BASE_CUTOFF_DATE + timedelta(days=1)
        end_date = datetime.now()
        
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Total days to fetch: {(end_date - start_date).days}")
        
        all_fixtures = []
        current_date = start_date
        days_processed = 0
        days_with_data = 0
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        batch_fixtures = []
        
        while current_date <= end_date:
            # Fetch fixtures for current date
            fixtures = await self.fetch_fixtures_for_date(current_date)
            
            if fixtures:
                # Filter for completed matches only
                completed = [f for f in fixtures if 
                           f.get('event_status') == 'Finished' and 
                           f.get('event_final_result') and
                           f.get('event_final_result') != '-']
                
                if completed:
                    batch_fixtures.extend(completed)
                    days_with_data += 1
            
            days_processed += 1
            
            # Progress update
            if days_processed % batch_size == 0:
                all_fixtures.extend(batch_fixtures)
                print(f"  Progress: {days_processed} days processed, {len(all_fixtures)} matches found")
                batch_fixtures = []
                await asyncio.sleep(0.5)  # Rate limiting
            
            current_date += timedelta(days=1)
        
        # Add remaining fixtures
        all_fixtures.extend(batch_fixtures)
        
        print(f"\n=== FETCH COMPLETE ===")
        print(f"Days processed: {days_processed}")
        print(f"Days with matches: {days_with_data}")
        print(f"Total matches found: {len(all_fixtures)}")
        
        return all_fixtures

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
                'player1_key': fixture.get('first_player_key'),
                'player2_key': fixture.get('second_player_key'),
                'final_result': fixture.get('event_final_result'),
                'winner': fixture.get('event_winner'),
                'status': fixture.get('event_status'),
                'match_type': fixture.get('event_type_type'),
                'tournament_key': fixture.get('tournament_key'),
                'tournament_season': fixture.get('tournament_season'),
            }
            
            # Extract scores if available
            if 'scores' in fixture:
                scores = fixture.get('scores', {})
                for set_num in range(1, 6):
                    set_key = f'{set_num}_set'
                    if set_key in scores:
                        record[f'set{set_num}_p1'] = scores[set_key].get('1_player')
                        record[f'set{set_num}_p2'] = scores[set_key].get('2_player')
            
            # FIXED: Properly extract statistics
            if 'statistics' in fixture:
                # First, get player keys for mapping
                p1_key = fixture.get('first_player_key')
                p2_key = fixture.get('second_player_key')
                
                # Process each statistic
                for stat in fixture.get('statistics', []):
                    player_key = stat.get('player_key')
                    stat_name = stat.get('stat_name', '').lower().replace(' ', '_').replace('%', 'pct')
                    stat_value = stat.get('stat_value', '')
                    
                    # Remove percentage sign if present
                    if isinstance(stat_value, str) and stat_value.endswith('%'):
                        try:
                            stat_value = float(stat_value.rstrip('%'))
                        except:
                            pass
                    
                    # Determine which player this stat belongs to
                    if player_key == p1_key:
                        # Player 1 statistic
                        record[f'{stat_name}_p1'] = stat_value
                        
                        # Add won/total if available
                        if stat.get('stat_won') is not None:
                            record[f'{stat_name}_won_p1'] = stat.get('stat_won')
                        if stat.get('stat_total') is not None:
                            record[f'{stat_name}_total_p1'] = stat.get('stat_total')
                            
                    elif player_key == p2_key:
                        # Player 2 statistic
                        record[f'{stat_name}_p2'] = stat_value
                        
                        # Add won/total if available
                        if stat.get('stat_won') is not None:
                            record[f'{stat_name}_won_p2'] = stat.get('stat_won')
                        if stat.get('stat_total') is not None:
                            record[f'{stat_name}_total_p2'] = stat.get('stat_total')
            
            # Extract point progression if available
            if 'pointbypoint' in fixture:
                points = fixture.get('pointbypoint', [])
                record['has_point_progression'] = True
                record['total_points'] = len(points)
            else:
                record['has_point_progression'] = False
                record['total_points'] = 0
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing fixture {fixture.get('event_key')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} matches into DataFrame")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Columns: {len(df.columns)}")
        
        # Filter for singles matches
        singles_mask = df['match_type'].str.contains('Singles', na=False)
        singles_count = singles_mask.sum()
        doubles_count = (~singles_mask).sum()
        
        print(f"Singles matches: {singles_count}")
        print(f"Doubles matches: {doubles_count}")
    
    return df

async def main():
    """Main execution"""
    async with APITennisFetcher() as fetcher:
        # Fetch all post-June fixtures
        fixtures = await fetcher.fetch_all_post_june_matches()
        
        if not fixtures:
            print("No fixtures fetched!")
            return None
        
        # Convert to DataFrame
        df = process_api_fixtures_to_dataframe(fixtures)
        
        if len(df) == 0:
            print("No data to save!")
            return None
        
        # Save full dataset
        output_file = Path("/Users/danielkim/Desktop/t3nn1s/api_tennis_post_june_2025_FULL.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
        # Also save as Parquet for faster loading
        parquet_file = Path("/Users/danielkim/Desktop/t3nn1s/api_tennis_post_june_2025_FULL.parquet")
        df.to_parquet(parquet_file)
        print(f"Saved Parquet: {parquet_file}")
        
        # Display summary
        print("\n=== DATA SUMMARY ===")
        print(f"Total matches: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Singles only summary
        singles_df = df[df['match_type'].str.contains('Singles', na=False)]
        print(f"\nSingles matches: {len(singles_df):,}")
        print(f"Unique tournaments: {singles_df['tournament'].nunique()}")
        
        # Tournament breakdown
        print("\nTop tournaments by match count:")
        top_tournaments = singles_df['tournament'].value_counts().head(10)
        for tourney, count in top_tournaments.items():
            print(f"  {tourney}: {count}")
        
        # Monthly breakdown
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_counts = df.groupby('month').size()
        
        print("\nMonthly match counts:")
        for month, count in monthly_counts.items():
            print(f"  {month}: {count:,}")
        
        return df

if __name__ == "__main__":
    df = asyncio.run(main())
    if df is not None:
        print("\n✅ Full API-Tennis data fetch complete!")