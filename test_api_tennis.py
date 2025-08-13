#!/usr/bin/env python3
"""
Test API-Tennis connection and find correct endpoints
"""

import requests
import json
from datetime import datetime, date
import os

# API configuration
API_KEY = os.getenv('API_TENNIS_KEY', 'adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb')

def test_api_endpoints():
    """Test different API-Tennis endpoints to find the working one"""
    
    print("🔍 TESTING API-TENNIS ENDPOINTS")
    print("="*60)
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print()
    
    # Different possible API configurations
    test_configs = [
        {
            'name': 'API-Tennis (Original)',
            'url': 'https://api.api-tennis.com/tennis/',
            'headers': {
                'x-rapidapi-key': API_KEY,
                'x-rapidapi-host': 'api-tennis.p.rapidapi.com'
            }
        },
        {
            'name': 'Tennis Live Data API',
            'url': 'https://tennis-live-data.p.rapidapi.com/',
            'headers': {
                'x-rapidapi-key': API_KEY,
                'x-rapidapi-host': 'tennis-live-data.p.rapidapi.com'
            }
        },
        {
            'name': 'API-Tennis Direct',
            'url': 'https://api-tennis.com/api/',
            'headers': {
                'Authorization': f'Bearer {API_KEY}'
            }
        },
        {
            'name': 'Ultimate Tennis API',
            'url': 'https://ultimate-tennis1.p.rapidapi.com/',
            'headers': {
                'x-rapidapi-key': API_KEY,
                'x-rapidapi-host': 'ultimate-tennis1.p.rapidapi.com'
            }
        }
    ]
    
    # Test endpoints for each configuration
    test_endpoints = [
        'matches?date=2025-08-10',
        'games/list-live',
        'fixtures?date=2025-08-10',
        'matches/date/2025-08-10',
        'events?date=2025-08-10',
        'matches/live',
        'schedule?date=2025-08-10'
    ]
    
    working_configs = []
    
    for config in test_configs:
        print(f"\n📡 Testing: {config['name']}")
        print(f"   Base URL: {config['url']}")
        
        success = False
        for endpoint in test_endpoints:
            url = config['url'] + endpoint
            
            try:
                response = requests.get(url, headers=config['headers'], timeout=5)
                
                if response.status_code == 200:
                    print(f"   ✅ SUCCESS: {endpoint}")
                    print(f"      Response size: {len(response.content)} bytes")
                    
                    # Try to parse JSON
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            print(f"      Keys: {list(data.keys())[:5]}")
                            if 'response' in data:
                                print(f"      Response items: {len(data.get('response', []))}")
                    except:
                        print(f"      (Non-JSON response)")
                    
                    working_configs.append({
                        'config': config,
                        'endpoint': endpoint,
                        'response': response
                    })
                    success = True
                    break
                    
                elif response.status_code == 401:
                    print(f"   ❌ 401 Unauthorized: {endpoint}")
                elif response.status_code == 403:
                    print(f"   ❌ 403 Forbidden: {endpoint}")
                elif response.status_code == 404:
                    # Don't print 404s, too noisy
                    pass
                else:
                    print(f"   ⚠️  {response.status_code}: {endpoint}")
                    
            except requests.exceptions.Timeout:
                print(f"   ⏱️  Timeout: {endpoint}")
            except requests.exceptions.ConnectionError:
                print(f"   🔌 Connection error: {endpoint}")
            except Exception as e:
                print(f"   💥 Error: {str(e)[:50]}")
        
        if not success:
            print(f"   ❌ No working endpoints found")
    
    return working_configs

def test_rapidapi_specific():
    """Test RapidAPI specific tennis endpoints"""
    print(f"\n🎾 TESTING RAPIDAPI TENNIS APIS")
    print("="*60)
    
    # Common RapidAPI tennis endpoints
    rapidapi_tests = [
        {
            'name': 'Tennis API - api-sports',
            'url': 'https://tennis-api-api-sports.p.rapidapi.com/',
            'endpoints': ['games', 'fixtures', 'standings'],
            'host': 'tennis-api-api-sports.p.rapidapi.com'
        },
        {
            'name': 'AllSportsAPI',
            'url': 'https://allsportsapi2.p.rapidapi.com/',
            'endpoints': ['tennis/?met=Fixtures&APIkey=' + API_KEY],
            'host': 'allsportsapi2.p.rapidapi.com'
        }
    ]
    
    for api_test in rapidapi_tests:
        print(f"\n🔍 Testing: {api_test['name']}")
        
        headers = {
            'x-rapidapi-key': API_KEY,
            'x-rapidapi-host': api_test['host']
        }
        
        for endpoint in api_test['endpoints']:
            url = api_test['url'] + endpoint
            
            try:
                response = requests.get(url, headers=headers, timeout=5)
                print(f"   {endpoint}: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"   ✅ Working endpoint found!")
                    return {
                        'name': api_test['name'],
                        'url': url,
                        'headers': headers,
                        'response': response
                    }
            except Exception as e:
                print(f"   Error: {str(e)[:50]}")
    
    return None

def main():
    """Run API tests"""
    
    # Test various endpoints
    working_configs = test_api_endpoints()
    
    if working_configs:
        print(f"\n✅ FOUND {len(working_configs)} WORKING CONFIGURATION(S)")
        
        for i, config in enumerate(working_configs, 1):
            print(f"\n{i}. {config['config']['name']}")
            print(f"   Endpoint: {config['endpoint']}")
            print(f"   Full URL: {config['config']['url']}{config['endpoint']}")
            print(f"   Headers: {config['config']['headers']}")
    else:
        print(f"\n❌ No working configurations found with standard endpoints")
        
        # Try RapidAPI specific
        rapidapi_result = test_rapidapi_specific()
        
        if rapidapi_result:
            print(f"\n✅ Found working RapidAPI endpoint:")
            print(f"   Name: {rapidapi_result['name']}")
            print(f"   URL: {rapidapi_result['url']}")
        else:
            print(f"\n❌ No working RapidAPI endpoints found either")
            print(f"\nPossible issues:")
            print(f"1. API key may be invalid or expired")
            print(f"2. API service may have changed")
            print(f"3. Rate limits may be exceeded")
            print(f"4. Need different API subscription")

if __name__ == "__main__":
    main()