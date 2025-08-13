#!/usr/bin/env python3
"""
Fixed API-Tennis Integration
Using the correct working format from tennis_updated.py
"""

import requests
import pandas as pd
from datetime import datetime, date, timedelta
import os
import json
from pathlib import Path
from settings import TENNIS_CACHE_DIR
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingAPITennisClient:
    """Corrected API-Tennis client using the working format"""
    
    def __init__(self):
        self.api_key = os.getenv("API_TENNIS_KEY", "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb")
        self.base_url = "https://api.api-tennis.com/tennis/"
        self.session = requests.Session()
        self.cache_dir = Path(TENNIS_CACHE_DIR) / 'api_tennis_working'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def api_call(self, method: str, **params):
        """Working API wrapper using query parameters"""
        url = self.base_url
        params["method"] = method
        params["APIkey"] = self.api_key
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") == 1:
                return data.get("result", [])
            else:
                logger.error(f"API returned unsuccessful response for {method}: {data}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for method {method}: {e}")
            return []
    
    def get_events(self):
        """Get available tennis events/tournaments"""
        return self.api_call("get_events")
    
    def get_atp_singles_event_key(self):
        """Get the event type key for ATP Singles"""
        events = self.get_events()
        
        atp_event = next(
            (e for e in events if e.get("event_type_type") == "Atp Singles"),
            None
        )
        
        if atp_event:
            return atp_event.get('event_type_key')
        else:
            return None
    
    def get_fixtures_for_date(self, target_date, event_type_key=None):
        """Get all fixtures for a specific date"""
        params = {
            "date_start": target_date.isoformat(),
            "date_stop": target_date.isoformat(),
            "timezone": "UTC"
        }
        
        if event_type_key:
            params["event_type_key"] = event_type_key
        
        return self.api_call("get_fixtures", **params)
    
    def parse_fixture_to_match(self, fixture, target_date):
        """Parse fixture data to match format"""
        try:
            match_data = {
                'date': target_date,
                'source': 'api_tennis',
                'source_rank': 2,
                'fixture_id': fixture.get('event_key'),
                'event_name': fixture.get('event_name'),
                'tournament': fixture.get('event_country_name'),
                'status': fixture.get('event_status'),
                'event_date': fixture.get('event_date_start')
            }
            
            # Extract team names
            home_team = fixture.get('event_home_team', 'Unknown')
            away_team = fixture.get('event_away_team', 'Unknown')
            
            match_data['player1'] = home_team
            match_data['player2'] = away_team
            
            # Determine winner if match is finished
            if fixture.get('event_status') == 'Finished':
                # Try to determine winner from score data
                scores = fixture.get('scores', [])
                if scores:
                    # This would need more sophisticated parsing
                    # For now, just mark that we have the data
                    match_data['has_scores'] = True
                    
                    # Check for final result
                    final_score = fixture.get('event_final_result')
                    if final_score:
                        # Would need to parse this properly
                        match_data['final_score'] = final_score
                        
                        # Simple winner detection (would need improvement)
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
                    f"{target_date.strftime('%Y%m%d')}"
                )
            else:
                match_data['composite_id'] = (
                    f"{home_team}_{away_team}_{target_date.strftime('%Y%m%d')}"
                )
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error parsing fixture: {e}")
            return None
    
    def fetch_recent_matches(self, start_date, end_date):
        """Fetch recent matches using working API format"""
        print(f"\n🌐 FETCHING API-TENNIS DATA (CORRECTED)")
        print(f"Using working format: query parameters with method/APIkey")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 60)
        
        # Get ATP Singles event key
        atp_key = self.get_atp_singles_event_key()
        print(f"ATP Singles event key: {atp_key}")
        
        all_matches = []
        current_date = start_date
        
        while current_date <= end_date:
            print(f"  📅 Fetching {current_date}...")
            
            fixtures = self.get_fixtures_for_date(current_date, atp_key)
            
            if fixtures:
                print(f"     ✓ Found {len(fixtures)} fixtures")
                
                for fixture in fixtures:
                    match_data = self.parse_fixture_to_match(fixture, current_date)
                    if match_data:
                        all_matches.append(match_data)
                        
            else:
                print(f"     ✗ No fixtures")
            
            current_date += timedelta(days=1)
            
            # Limit for testing
            if len(all_matches) >= 100:
                print(f"  (Limited to 100 matches for testing)")
                break
        
        if all_matches:
            df = pd.DataFrame(all_matches)
            print(f"\n✅ Retrieved {len(df)} matches from API-Tennis")
            
            # Save to cache
            cache_file = self.cache_dir / f'working_api_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            df.to_parquet(cache_file, index=False)
            print(f"💾 Cached to: {cache_file}")
            
            # Show sample data
            print(f"\n📊 Sample data:")
            if 'status' in df.columns:
                print(f"   Status breakdown: {df['status'].value_counts().to_dict()}")
            
            if 'Winner' in df.columns:
                finished_with_winner = df['Winner'].notna().sum()
                print(f"   Finished matches with winner: {finished_with_winner}")
            
            return df
        else:
            print(f"\n❌ No matches retrieved")
            return pd.DataFrame()

def test_corrected_api():
    """Test the corrected API implementation"""
    print("🔧 TESTING CORRECTED API-TENNIS INTEGRATION")
    print("="*70)
    
    client = WorkingAPITennisClient()
    
    # Test with recent date range
    start_date = date(2025, 8, 5)
    end_date = date(2025, 8, 11)
    
    matches_df = client.fetch_recent_matches(start_date, end_date)
    
    if not matches_df.empty:
        print(f"\n🎯 SUCCESS! API-TENNIS IS WORKING")
        print(f"="*50)
        print(f"Total matches: {len(matches_df)}")
        
        if 'date' in matches_df.columns:
            print(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
        
        if 'event_name' in matches_df.columns:
            tournaments = matches_df['event_name'].value_counts().head(5)
            print(f"\nTop tournaments:")
            for tournament, count in tournaments.items():
                print(f"  - {tournament}: {count}")
        
        if 'status' in matches_df.columns:
            status_counts = matches_df['status'].value_counts()
            print(f"\nMatch statuses:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count}")
        
        return matches_df
    else:
        print(f"\n❌ No matches retrieved")
        return pd.DataFrame()

def integrate_with_existing_pipeline(api_matches_df):
    """Integrate API-Tennis data with existing pipeline"""
    if api_matches_df.empty:
        return pd.DataFrame()
    
    print(f"\n🔧 INTEGRATING WITH EXISTING PIPELINE")
    print("="*50)
    
    # Load existing dataset
    try:
        existing_file = os.path.join(TENNIS_CACHE_DIR, 'final_complete_tennis_2020s.parquet')
        existing_df = pd.read_parquet(existing_file)
        print(f"✅ Loaded existing dataset: {len(existing_df)} matches")
        
        # Combine datasets
        combined_df = pd.concat([existing_df, api_matches_df], ignore_index=True)
        print(f"✅ Combined dataset: {len(combined_df)} matches")
        
        # Deduplicate
        if 'composite_id' in combined_df.columns:
            initial_count = len(combined_df)
            combined_df = combined_df.sort_values('source_rank').drop_duplicates(
                subset='composite_id', keep='first'
            ).reset_index(drop=True)
            
            duplicates_removed = initial_count - len(combined_df)
            print(f"✅ After deduplication: {len(combined_df)} matches ({duplicates_removed} duplicates removed)")
        
        # Update historical_data.parquet
        historical_file = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
        combined_df.to_parquet(historical_file, index=False)
        print(f"✅ Updated historical_data.parquet")
        
        # Show final source breakdown
        if 'source' in combined_df.columns:
            print(f"\nFinal source breakdown:")
            source_counts = combined_df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  - {source}: {count}")
        
        return combined_df
        
    except Exception as e:
        print(f"❌ Error integrating: {e}")
        return pd.DataFrame()

def main():
    """Main execution"""
    print("🚀 EXECUTING CORRECTED API-TENNIS INTEGRATION")
    print("="*70)
    
    # Test corrected API
    api_matches = test_corrected_api()
    
    if not api_matches.empty:
        # Integrate with existing pipeline
        final_dataset = integrate_with_existing_pipeline(api_matches)
        
        if not final_dataset.empty:
            print(f"\n🎯 INTEGRATION COMPLETED SUCCESSFULLY!")
            print(f"Final dataset size: {len(final_dataset)} matches")
            print(f"API-Tennis matches added: {len(api_matches)}")
            
            return final_dataset
    
    print(f"\n⚠️  Integration incomplete - check API configuration")
    return pd.DataFrame()

if __name__ == "__main__":
    result = main()