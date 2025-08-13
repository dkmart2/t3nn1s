#!/usr/bin/env python3
"""
Comprehensive Recent Data Fetch
Complete integration of API-Tennis and Tennis Abstract for post-6/10/2025 period
"""

import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time
import os
from pathlib import Path
from settings import TENNIS_CACHE_DIR
from tennis_updated import TennisAbstractScraper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataFetcher:
    """Comprehensive fetcher for recent tennis data"""
    
    def __init__(self):
        self.api_key = os.getenv("API_TENNIS_KEY", "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb")
        self.base_url = "https://api.api-tennis.com/tennis/"
        self.session = requests.Session()
        self.cache_dir = Path(TENNIS_CACHE_DIR) / 'comprehensive_recent'
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
    
    def get_comprehensive_tennis_abstract_urls(self, start_date, end_date):
        """Get comprehensive Tennis Abstract URLs for the date range"""
        print(f"\n🔍 FINDING COMPREHENSIVE TENNIS ABSTRACT MATCHES")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 60)
        
        # Extended list of potential Tennis Abstract matches
        potential_urls = []
        
        # Known tournament patterns and dates
        tournament_patterns = [
            # June 2025
            ('20250610', '20250615', ['Stuttgart', 'Hertogenbosch', 'Nottingham']),
            ('20250617', '20250622', ['Queens', 'Halle', 'Birmingham']),
            ('20250624', '20250706', ['Wimbledon']),  # Wimbledon
            
            # July 2025  
            ('20250708', '20250713', ['Hamburg', 'Newport', 'Bastad']),
            ('20250715', '20250720', ['Gstaad', 'Bucharest', 'Palermo']),
            ('20250722', '20250727', ['Umag', 'Atlanta', 'Washington']),
            ('20250729', '20250803', ['Kitzbuhel', 'Washington']),
            
            # August 2025
            ('20250804', '20250810', ['Montreal', 'Toronto', 'Cincinnati']),
        ]
        
        # Generate potential URLs
        for start_str, end_str, tournaments in tournament_patterns:
            for tournament in tournaments:
                for gender in ['M', 'W']:
                    # Try different rounds
                    rounds = ['F', 'SF', 'QF', 'R16', 'R32']
                    
                    for round_name in rounds:
                        # Try different date combinations
                        start_date_obj = datetime.strptime(start_str, '%Y%m%d').date()
                        end_date_obj = datetime.strptime(end_str, '%Y%m%d').date()
                        
                        current = start_date_obj
                        while current <= end_date_obj:
                            # Generate potential player combinations (top players)
                            top_players = {
                                'M': ['Carlos_Alcaraz', 'Jannik_Sinner', 'Novak_Djokovic', 'Daniil_Medvedev', 
                                      'Alexander_Zverev', 'Andrey_Rublev', 'Stefanos_Tsitsipas', 'Holger_Rune'],
                                'W': ['Iga_Swiatek', 'Coco_Gauff', 'Aryna_Sabalenka', 'Elena_Rybakina',
                                      'Jessica_Pegula', 'Ons_Jabeur', 'Marketa_Vondrousova', 'Qinwen_Zheng']
                            }
                            
                            players = top_players[gender]
                            
                            # Try combinations of top players
                            for i in range(min(4, len(players))):
                                for j in range(i+1, min(4, len(players))):
                                    player1 = players[i]
                                    player2 = players[j]
                                    
                                    url = f"https://www.tennisabstract.com/charting/{current.strftime('%Y%m%d')}-{gender}-{tournament}-{round_name}-{player1}-{player2}.html"
                                    potential_urls.append(url)
                            
                            current += timedelta(days=1)
        
        # Test which URLs exist
        print(f"Testing {len(potential_urls)} potential URLs...")
        existing_urls = []
        
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        
        for i, url in enumerate(potential_urls[:50], 1):  # Limit for testing
            try:
                response = requests.head(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    existing_urls.append(url)
                    match_name = url.split('/')[-1][:50]
                    print(f"  ✅ Found: {match_name}...")
                elif i % 10 == 0:
                    print(f"  📊 Tested {i}/{min(50, len(potential_urls))} URLs...")
            except:
                continue
            
            time.sleep(0.5)  # Be respectful
        
        print(f"✅ Found {len(existing_urls)} existing Tennis Abstract matches")
        return existing_urls
    
    def scrape_comprehensive_tennis_abstract(self, urls):
        """Scrape comprehensive Tennis Abstract data"""
        if not urls:
            return pd.DataFrame()
        
        print(f"\n📊 SCRAPING COMPREHENSIVE TENNIS ABSTRACT DATA")
        print(f"URLs to scrape: {len(urls)}")
        print("-" * 60)
        
        scraper = TennisAbstractScraper()
        all_records = []
        all_matches = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Scraping: {url.split('/')[-1]}")
            
            try:
                # Get comprehensive match data
                records = scraper.scrape_comprehensive_match_data(url)
                
                if records:
                    print(f"  ✅ Extracted {len(records)} records")
                    
                    # Parse match metadata
                    url_parts = url.split('/')[-1].replace('.html', '').split('-')
                    if len(url_parts) >= 5:
                        date_str = url_parts[0]
                        gender = url_parts[1]
                        tournament = url_parts[2]
                        round_str = url_parts[3]
                        
                        match_date = datetime.strptime(date_str, '%Y%m%d').date()
                        
                        # Add metadata to all records
                        for record in records:
                            record['match_date'] = match_date
                            record['gender'] = gender
                            record['tournament'] = tournament
                            record['round'] = round_str
                            record['source'] = 'tennis_abstract'
                            record['source_rank'] = 1
                            record['has_detailed_stats'] = True
                            record['has_point_data'] = True
                            record['url'] = url
                        
                        all_records.extend(records)
                        
                        # Create match record
                        players = list(set(record.get('Player_canonical', 'Unknown') for record in records))
                        if len(players) >= 2:
                            match_record = {
                                'date': match_date,
                                'Player_1': players[0],
                                'Player_2': players[1],
                                'gender': gender,
                                'tournament': tournament,
                                'round': round_str,
                                'source': 'tennis_abstract',
                                'source_rank': 1,
                                'has_detailed_stats': True,
                                'has_point_data': True,
                                'comprehensive_records': len(records),
                                'url': url,
                                'composite_id': f"{players[0]}_{players[1]}_{date_str}"
                            }
                            all_matches.append(match_record)
                else:
                    print(f"  ⚠️  No records extracted")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            time.sleep(2)  # Rate limiting
        
        if all_matches:
            matches_df = pd.DataFrame(all_matches)
            records_df = pd.DataFrame(all_records)
            
            print(f"\n✅ Tennis Abstract comprehensive:")
            print(f"   Matches: {len(matches_df)}")
            print(f"   Detailed records: {len(records_df)}")
            
            # Save both datasets
            matches_file = self.cache_dir / f'ta_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            records_file = self.cache_dir / f'ta_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            
            matches_df.to_parquet(matches_file, index=False)
            records_df.to_parquet(records_file, index=False)
            
            print(f"💾 Cached TA matches: {matches_file}")
            print(f"💾 Cached TA records: {records_file}")
            
            return matches_df
        else:
            print(f"\n❌ No Tennis Abstract matches scraped")
            return pd.DataFrame()

def main():
    """Execute comprehensive recent data fetch"""
    print("🚀 COMPREHENSIVE RECENT TENNIS DATA FETCH")
    print("="*80)
    print("Goal: Complete API-Tennis + Tennis Abstract integration for post-6/10/2025")
    print()
    
    fetcher = ComprehensiveDataFetcher()
    
    # Date range for comprehensive fetch
    start_date = date(2025, 6, 10)  # Jeff data cutoff
    end_date = date.today()
    
    all_recent_data = []
    
    # Step 1: Comprehensive API-Tennis fetch
    print("🎯 STEP 1: COMPREHENSIVE API-TENNIS")
    api_data = fetcher.fetch_comprehensive_api_tennis(start_date, end_date)
    if not api_data.empty:
        all_recent_data.append(api_data)
    
    # Step 2: Comprehensive Tennis Abstract scraping
    print("\n🎯 STEP 2: COMPREHENSIVE TENNIS ABSTRACT")
    ta_urls = fetcher.get_comprehensive_tennis_abstract_urls(start_date, end_date)
    if ta_urls:
        ta_data = fetcher.scrape_comprehensive_tennis_abstract(ta_urls)
        if not ta_data.empty:
            all_recent_data.append(ta_data)
    
    # Step 3: Combine and integrate
    if all_recent_data:
        print(f"\n🎯 STEP 3: COMBINING COMPREHENSIVE DATA")
        print("-" * 60)
        
        combined_recent = pd.concat(all_recent_data, ignore_index=True)
        print(f"✅ Combined recent data: {len(combined_recent)} matches")
        
        # Show breakdown
        if 'source' in combined_recent.columns:
            print(f"\nRecent source breakdown:")
            source_counts = combined_recent['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  - {source}: {count}")
        
        # Save comprehensive recent data
        comprehensive_file = os.path.join(TENNIS_CACHE_DIR, 'comprehensive_recent_matches.parquet')
        combined_recent.to_parquet(comprehensive_file, index=False)
        print(f"\n💾 Saved comprehensive recent data: {comprehensive_file}")
        
        return combined_recent
    else:
        print(f"\n❌ No comprehensive data fetched")
        return pd.DataFrame()

if __name__ == "__main__":
    result = main()
    
    if not result.empty:
        print(f"\n🎉 COMPREHENSIVE FETCH COMPLETED!")
        print(f"Recent matches: {len(result)}")
        print(f"Ready for final integration with main dataset")
    else:
        print(f"\n⚠️  Comprehensive fetch incomplete")
        print(f"Check API keys and network connectivity")