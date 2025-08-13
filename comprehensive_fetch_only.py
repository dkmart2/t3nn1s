#!/usr/bin/env python3
"""
Comprehensive Recent Data Fetch - API-Tennis + Tennis Abstract Only
Focused on fetching just the recent data without loading existing datasets
"""

import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataFetcher:
    """Comprehensive fetcher for recent tennis data only"""
    
    def __init__(self):
        self.api_key = os.getenv("API_TENNIS_KEY", "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb")
        self.base_url = "https://api.api-tennis.com/tennis/"
        self.session = requests.Session()
        self.cache_dir = Path('/Users/danielkim/tennis_data/cache') / 'comprehensive_recent'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def api_call(self, method: str, **params):
        """API wrapper using working format"""
        url = self.base_url
        params["method"] = method
        params["APIkey"] = self.api_key
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") == 1:
                    return data.get("result", [])
            return []
        except Exception as e:
            logger.error(f"API error for {method}: {e}")
            return []
    
    def fetch_comprehensive_api_tennis(self, start_date, end_date):
        """Fetch comprehensive API-Tennis data for date range"""
        print(f"\n🌐 COMPREHENSIVE API-TENNIS FETCH")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 60)
        
        # Get event types for both ATP and WTA
        events = self.api_call("get_events")
        
        event_types = {}
        for event in events:
            event_type = event.get("event_type_type")
            if event_type in ["Atp Singles", "Wta Singles", "Atp Doubles", "Wta Doubles"]:
                event_types[event_type] = event.get("event_type_key")
        
        print(f"Event types found: {list(event_types.keys())}")
        
        all_matches = []
        current_date = start_date
        
        while current_date <= end_date:
            print(f"\n📅 Fetching {current_date}...")
            
            day_matches = []
            
            # Fetch for each event type
            for event_name, event_key in event_types.items():
                if event_key:
                    fixtures = self.api_call(
                        "get_fixtures",
                        date_start=current_date.isoformat(),
                        date_stop=current_date.isoformat(),
                        event_type_key=event_key,
                        timezone="UTC"
                    )
                    
                    if fixtures:
                        print(f"  {event_name}: {len(fixtures)} fixtures")
                        
                        for fixture in fixtures:
                            match_data = self.parse_api_fixture(fixture, current_date, event_name)
                            if match_data:
                                day_matches.append(match_data)
            
            if day_matches:
                all_matches.extend(day_matches)
                print(f"  ✅ Total day: {len(day_matches)} matches")
            else:
                print(f"  ❌ No matches")
            
            current_date += timedelta(days=1)
            time.sleep(0.1)  # Rate limiting
        
        if all_matches:
            df = pd.DataFrame(all_matches)
            print(f"\n✅ API-Tennis comprehensive: {len(df)} matches")
            
            # Save comprehensive API data
            api_file = self.cache_dir / f'api_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            df.to_parquet(api_file, index=False)
            print(f"💾 Cached API data: {api_file}")
            
            return df
        else:
            print(f"\n❌ No API matches retrieved")
            return pd.DataFrame()
    
    def parse_api_fixture(self, fixture, match_date, event_type):
        """Parse API fixture to match format"""
        try:
            home_team = fixture.get('event_home_team', 'Unknown')
            away_team = fixture.get('event_away_team', 'Unknown')
            status = fixture.get('event_status', 'Unknown')
            
            match_data = {
                'date': match_date,
                'Player_1': home_team,
                'Player_2': away_team,
                'event_name': fixture.get('event_name', 'Unknown'),
                'tournament': fixture.get('event_country_name', 'Unknown'),
                'status': status,
                'source': 'api_tennis',
                'source_rank': 2,
                'event_type': event_type,
                'fixture_id': fixture.get('event_key'),
                'has_detailed_stats': False
            }
            
            # Determine winner if finished
            if status == 'Finished':
                home_result = fixture.get('event_home_final_result')
                away_result = fixture.get('event_away_final_result')
                
                if home_result and away_result:
                    try:
                        home_score = float(home_result)
                        away_score = float(away_result)
                        
                        if home_score > away_score:
                            match_data['Winner'] = home_team
                            match_data['Loser'] = away_team
                        else:
                            match_data['Winner'] = away_team
                            match_data['Loser'] = home_team
                    except:
                        pass
            
            # Add composite_id
            if 'Winner' in match_data and 'Loser' in match_data:
                match_data['composite_id'] = (
                    f"{match_data['Winner']}_{match_data['Loser']}_"
                    f"{match_date.strftime('%Y%m%d')}"
                )
            else:
                match_data['composite_id'] = (
                    f"{home_team}_{away_team}_{match_date.strftime('%Y%m%d')}"
                )
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error parsing fixture: {e}")
            return None

def main():
    """Execute comprehensive recent data fetch"""
    print("🚀 COMPREHENSIVE RECENT TENNIS DATA FETCH")
    print("="*80)
    print("Goal: Complete API-Tennis fetch for post-6/10/2025")
    print()
    
    fetcher = ComprehensiveDataFetcher()
    
    # Date range for comprehensive fetch - full period as requested
    start_date = date(2025, 6, 10)  # Jeff data cutoff as originally requested
    end_date = date.today()
    
    # Comprehensive API-Tennis fetch
    print("🎯 COMPREHENSIVE API-TENNIS FETCH")
    api_data = fetcher.fetch_comprehensive_api_tennis(start_date, end_date)
    
    if not api_data.empty:
        print(f"\n🎉 COMPREHENSIVE API FETCH COMPLETED!")
        print(f"API-Tennis matches: {len(api_data)}")
        
        # Show breakdown
        if 'source' in api_data.columns:
            print(f"\nSource breakdown:")
            source_counts = api_data['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  - {source}: {count}")
        
        if 'event_type' in api_data.columns:
            print(f"\nEvent type breakdown:")
            event_counts = api_data['event_type'].value_counts()
            for event_type, count in event_counts.items():
                print(f"  - {event_type}: {count}")
        
        # Show status breakdown
        if 'status' in api_data.columns:
            print(f"\nMatch status breakdown:")
            status_counts = api_data['status'].value_counts()
            for status, count in status_counts.items():
                print(f"  - {status}: {count}")
        
        return api_data
    else:
        print(f"\n❌ No comprehensive data fetched")
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()