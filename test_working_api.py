#!/usr/bin/env python3
"""
Test the working API-Tennis implementation from tennis_updated.py
"""

import requests
import os
from datetime import datetime, date, timedelta
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration from tennis_updated.py
API_KEY = os.getenv("API_TENNIS_KEY", "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb")
BASE_API_URL = "https://api.api-tennis.com/tennis/"
SESSION = requests.Session()

def api_call(method: str, **params):
    """Wrapper for API-Tennis endpoints. Uses query param "method" rather than path."""
    url = BASE_API_URL
    params["method"] = method
    params["APIkey"] = API_KEY
    
    print(f"🔍 API Call: {method}")
    print(f"   URL: {url}")
    print(f"   Params: {dict(params)}")
    
    try:
        response = SESSION.get(url, params=params)
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
            
            if data.get("success") == 1:
                result = data.get("result", [])
                print(f"   ✅ Success: {len(result)} items returned")
                return result
            else:
                print(f"   ❌ API returned unsuccessful: {data}")
                return []
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return []
            
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return []

def test_get_events():
    """Test getting events (tournaments)"""
    print(f"\n🎾 TESTING GET_EVENTS")
    print("="*50)
    
    events = api_call("get_events")
    
    if events:
        print(f"✅ Found {len(events)} events")
        
        # Show sample events
        print(f"\nSample events:")
        for i, event in enumerate(events[:5], 1):
            print(f"  {i}. {event.get('event_name', 'Unknown')} - Type: {event.get('event_type_type', 'Unknown')}")
            print(f"     Key: {event.get('event_type_key')}")
        
        # Find ATP Singles event
        atp_event = next(
            (e for e in events if e.get("event_type_type") == "Atp Singles"),
            None
        )
        
        if atp_event:
            print(f"\n✅ Found ATP Singles event: {atp_event.get('event_name')}")
            print(f"   Event type key: {atp_event.get('event_type_key')}")
            return atp_event.get('event_type_key')
        else:
            print(f"\n⚠️  No ATP Singles event found")
            return None
    else:
        print(f"❌ No events returned")
        return None

def test_get_fixtures(event_type_key=None, target_date=None):
    """Test getting fixtures for a specific date"""
    print(f"\n🎾 TESTING GET_FIXTURES")
    print("="*50)
    
    if not target_date:
        target_date = date.today() - timedelta(days=1)  # Yesterday to ensure matches exist
    
    print(f"Target date: {target_date}")
    print(f"Event type key: {event_type_key}")
    
    params = {
        "date_start": target_date.isoformat(),
        "date_stop": target_date.isoformat(),
        "timezone": "UTC"
    }
    
    if event_type_key:
        params["event_type_key"] = event_type_key
    
    fixtures = api_call("get_fixtures", **params)
    
    if fixtures:
        print(f"✅ Found {len(fixtures)} fixtures")
        
        # Show sample fixtures
        print(f"\nSample fixtures:")
        for i, fixture in enumerate(fixtures[:3], 1):
            print(f"  {i}. {fixture.get('event_name', 'Unknown Event')}")
            print(f"     Home: {fixture.get('event_home_team', 'Unknown')}")
            print(f"     Away: {fixture.get('event_away_team', 'Unknown')}")
            print(f"     Status: {fixture.get('event_status', 'Unknown')}")
            print(f"     Date: {fixture.get('event_date_start', 'Unknown')}")
        
        return fixtures
    else:
        print(f"❌ No fixtures returned")
        return []

def test_different_dates():
    """Test with different dates to find when API has data"""
    print(f"\n🎾 TESTING DIFFERENT DATES")
    print("="*50)
    
    test_dates = [
        date.today(),
        date.today() - timedelta(days=1),
        date.today() - timedelta(days=7),
        date(2024, 8, 10),  # Known tennis date
        date(2024, 7, 14),  # Wimbledon final
        date(2024, 6, 10),  # Around French Open
    ]
    
    for test_date in test_dates:
        print(f"\n📅 Testing date: {test_date}")
        fixtures = api_call(
            "get_fixtures",
            date_start=test_date.isoformat(),
            date_stop=test_date.isoformat(),
            timezone="UTC"
        )
        
        if fixtures:
            print(f"   ✅ Found {len(fixtures)} fixtures")
            return test_date, fixtures
        else:
            print(f"   ❌ No fixtures")
    
    return None, []

def main():
    """Test the working API implementation"""
    print("🔧 TESTING WORKING API-TENNIS IMPLEMENTATION")
    print("="*70)
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"Base URL: {BASE_API_URL}")
    print()
    
    # Test 1: Get events
    event_type_key = test_get_events()
    
    # Test 2: Get fixtures with known event type
    fixtures = test_get_fixtures(event_type_key)
    
    # Test 3: If no fixtures, try different dates
    if not fixtures:
        print(f"\n🔍 Trying different dates...")
        working_date, fixtures = test_different_dates()
        
        if fixtures:
            print(f"\n✅ Found working date: {working_date}")
        else:
            print(f"\n❌ No working dates found")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("="*50)
    
    if fixtures:
        print(f"✅ API-Tennis is working!")
        print(f"   Found fixtures: {len(fixtures)}")
        print(f"   Sample match: {fixtures[0].get('event_home_team')} vs {fixtures[0].get('event_away_team')}")
        
        # Test with 2025 dates specifically
        print(f"\n🎯 Testing 2025 dates...")
        recent_dates = [
            date(2025, 8, 10),
            date(2025, 8, 5),
            date(2025, 7, 15),
            date(2025, 7, 1),
            date(2025, 6, 15)
        ]
        
        for test_date in recent_dates:
            fixtures_2025 = api_call(
                "get_fixtures",
                date_start=test_date.isoformat(),
                date_stop=test_date.isoformat(),
                timezone="UTC"
            )
            
            if fixtures_2025:
                print(f"   ✅ {test_date}: {len(fixtures_2025)} matches")
            else:
                print(f"   ❌ {test_date}: No matches")
        
        return True
    else:
        print(f"❌ API-Tennis is not working")
        print(f"Possible issues:")
        print(f"1. API key invalid or expired")
        print(f"2. API service changed")
        print(f"3. Rate limits exceeded")
        print(f"4. Different API endpoint needed")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print(f"\n💡 DEBUGGING SUGGESTIONS:")
        print(f"1. Try different BASE_API_URL")
        print(f"2. Check API key validity")
        print(f"3. Test with curl command")
        print(f"4. Check API documentation")