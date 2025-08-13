"""
API-TENNIS COMPLETE DATA EXPLORATION
Deep dive into all available data from API-Tennis with working format
"""

import requests
import json
from datetime import datetime, date, timedelta
from collections import defaultdict

API_KEY = "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb"
BASE_API_URL = "https://api.api-tennis.com/tennis/"
SESSION = requests.Session()

def api_call(method: str, **params):
    """Make API call with correct format"""
    url = BASE_API_URL
    params["method"] = method
    params["APIkey"] = API_KEY
    
    try:
        response = SESSION.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") == 1:
                return data.get("result", [])
        return []
    except:
        return []

def explore_complete_fixture_data():
    """Comprehensive fixture data analysis"""
    print("="*80)
    print("COMPLETE API-TENNIS DATA EXPLORATION")
    print("="*80)
    
    # Get recent fixtures with rich data
    fixtures = api_call("get_fixtures",
                       date_start="2025-08-05",
                       date_stop="2025-08-05",
                       timezone="UTC")
    
    if not fixtures:
        print("❌ No fixtures found")
        return {}
    
    print(f"✅ Found {len(fixtures)} fixtures for analysis")
    
    # Analyze first fixture in detail
    sample_fixture = fixtures[0]
    
    print(f"\n📊 FIXTURE DATA STRUCTURE:")
    print("="*60)
    
    all_fields = list(sample_fixture.keys())
    print(f"Total fields available: {len(all_fields)}")
    
    # Categorize fields
    categories = {
        "basic_info": [],
        "player_data": [],
        "match_data": [],
        "statistics": [],
        "odds": [],
        "temporal": [],
        "special": []
    }
    
    for field in all_fields:
        field_lower = field.lower()
        if any(x in field_lower for x in ["player", "team", "home", "away"]):
            categories["player_data"].append(field)
        elif any(x in field_lower for x in ["odd", "bet", "probability"]):
            categories["odds"].append(field)
        elif any(x in field_lower for x in ["stat", "score", "result", "winner"]):
            categories["statistics"].append(field)
        elif any(x in field_lower for x in ["date", "time", "minute"]):
            categories["temporal"].append(field)
        elif any(x in field_lower for x in ["event", "tournament", "league"]):
            categories["basic_info"].append(field)
        elif any(x in field_lower for x in ["status", "round", "stage"]):
            categories["match_data"].append(field)
        else:
            categories["special"].append(field)
    
    for category, fields in categories.items():
        if fields:
            print(f"\n{category.upper().replace('_', ' ')} ({len(fields)} fields):")
            for field in fields[:10]:  # Show first 10
                value = sample_fixture.get(field)
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"  {field}: {value}")
            if len(fields) > 10:
                print(f"  ... and {len(fields)-10} more")
    
    # Check for rich data fields
    print(f"\n🔍 RICH DATA ANALYSIS:")
    print("="*60)
    
    rich_data_found = {}
    
    # 1. Check for detailed statistics
    if "statistics" in sample_fixture or any("stat" in k.lower() for k in all_fields):
        stats_fields = [k for k in all_fields if "stat" in k.lower()]
        rich_data_found["statistics"] = {
            "available": True,
            "fields": stats_fields,
            "sample": {k: sample_fixture.get(k) for k in stats_fields[:5]}
        }
        print(f"✅ STATISTICS: {len(stats_fields)} stat fields found")
        for field in stats_fields[:5]:
            print(f"   {field}: {sample_fixture.get(field)}")
    
    # 2. Check for point-by-point data
    if "pointbypoint" in sample_fixture or any("point" in k.lower() for k in all_fields):
        point_fields = [k for k in all_fields if "point" in k.lower()]
        rich_data_found["point_by_point"] = {
            "available": True,
            "fields": point_fields
        }
        print(f"✅ POINT DATA: {len(point_fields)} point-related fields")
        for field in point_fields:
            print(f"   {field}: {type(sample_fixture.get(field))}")
    
    # 3. Check for odds data
    odds_fields = [k for k in all_fields if any(x in k.lower() for x in ["odd", "bet", "probability"])]
    if odds_fields:
        rich_data_found["odds"] = {
            "available": True,
            "fields": odds_fields,
            "sample": {k: sample_fixture.get(k) for k in odds_fields[:5]}
        }
        print(f"✅ ODDS: {len(odds_fields)} odds fields found")
        for field in odds_fields[:5]:
            print(f"   {field}: {sample_fixture.get(field)}")
    
    # 4. Check for live data
    live_fields = [k for k in all_fields if any(x in k.lower() for x in ["live", "minute", "current"])]
    if live_fields:
        rich_data_found["live_data"] = {
            "available": True,
            "fields": live_fields
        }
        print(f"✅ LIVE DATA: {len(live_fields)} live fields found")
        for field in live_fields:
            print(f"   {field}: {sample_fixture.get(field)}")
    
    # 5. Analyze specific match data
    print(f"\n📋 SAMPLE MATCH ANALYSIS:")
    print("="*60)
    
    match_info = {
        "home_player": sample_fixture.get("event_home_team", "Unknown"),
        "away_player": sample_fixture.get("event_away_team", "Unknown"),
        "status": sample_fixture.get("event_status", "Unknown"),
        "date": sample_fixture.get("event_date_start", "Unknown"),
        "tournament": sample_fixture.get("league_name", "Unknown"),
        "result": sample_fixture.get("event_result", "Unknown")
    }
    
    for key, value in match_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check multiple fixtures for data consistency
    print(f"\n🔄 MULTI-FIXTURE ANALYSIS:")
    print("="*60)
    
    field_coverage = defaultdict(int)
    data_types = defaultdict(set)
    
    for fixture in fixtures[:20]:  # Analyze first 20 fixtures
        for field, value in fixture.items():
            if value is not None and value != "":
                field_coverage[field] += 1
                data_types[field].add(type(value).__name__)
    
    # Fields with highest coverage
    top_fields = sorted(field_coverage.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("Most populated fields (coverage %):")
    for field, count in top_fields:
        coverage_pct = (count / 20) * 100
        types = ", ".join(data_types[field])
        print(f"  {field}: {coverage_pct:.0f}% ({types})")
    
    return {
        "total_fixtures": len(fixtures),
        "total_fields": len(all_fields),
        "field_categories": {k: len(v) for k, v in categories.items()},
        "rich_data_found": rich_data_found,
        "top_fields": top_fields[:10],
        "sample_match": match_info,
        "all_fields": all_fields
    }

def test_all_endpoints():
    """Test all available endpoints"""
    print(f"\n🔧 TESTING ALL ENDPOINTS:")
    print("="*60)

    endpoints_to_test = [
        ("get_events", {}),
        ("get_tournaments", {}),
        ("get_livescore", {}),
        ("get_standings", {"event_type_key": 265}),
        ("get_H2H", {"first_player_key": "1", "second_player_key": "2"}),
        ("get_odds", {"date_start": "2025-08-05", "date_stop": "2025-08-05"}),
        ("get_players", {"player_key": "1"})
    ]

    endpoint_results = {}

    for method, params in endpoints_to_test:
        print(f"\nTesting {method}...")
        data = api_call(method, **params)
        
        if data:
            print(f"   ✅ Success: {len(data)} items")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                fields = list(data[0].keys())
                print(f"   Fields: {fields[:5]}...")
                endpoint_results[method] = {
                    "success": True,
                    "count": len(data),
                    "fields": fields
                }
                
                # Show sample data for key endpoints
                if method in ["get_events", "get_tournaments"]:
                    sample = data[0]
                    print(f"   Sample: {sample.get('event_name', sample.get('tournament_name', 'Unknown'))}")
                    
            else:
                endpoint_results[method] = {
                    "success": True,
                    "count": len(data),
                    "data_type": type(data).__name__
                }
        else:
            print(f"   ❌ No data")
            endpoint_results[method] = {"success": False}
    
    return endpoint_results

def analyze_specific_match_data():
    """Deep dive into a specific match's data"""
    print(f"\n🎾 SPECIFIC MATCH DEEP DIVE:")
    print("="*60)
    
    # Get finished matches
    fixtures = api_call("get_fixtures",
                       date_start="2025-08-05",
                       date_stop="2025-08-05",
                       timezone="UTC")
    
    if not fixtures:
        return {}
    
    # Find a finished match
    finished_match = None
    for fixture in fixtures:
        if fixture.get("event_status") == "Finished":
            finished_match = fixture
            break
    
    if not finished_match:
        finished_match = fixtures[0]  # Use first match
    
    print(f"Analyzing match: {finished_match.get('event_home_team')} vs {finished_match.get('event_away_team')}")
    
    # Extract all data types
    match_analysis = {
        "basic_info": {},
        "scores": {},
        "statistics": {},
        "odds": {},
        "meta_data": {}
    }
    
    for field, value in finished_match.items():
        field_lower = field.lower()
        
        if any(x in field_lower for x in ["score", "result", "set"]):
            match_analysis["scores"][field] = value
        elif any(x in field_lower for x in ["stat", "ace", "fault", "break"]):
            match_analysis["statistics"][field] = value
        elif any(x in field_lower for x in ["odd", "bet", "probability"]):
            match_analysis["odds"][field] = value
        elif any(x in field_lower for x in ["event", "team", "player", "league"]):
            match_analysis["basic_info"][field] = value
        else:
            match_analysis["meta_data"][field] = value
    
    # Print analysis
    for category, data in match_analysis.items():
        if data:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for field, value in list(data.items())[:5]:
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {field}: {value}")
    
    return match_analysis

def main():
    """Run complete exploration"""
    
    # 1. Main fixture exploration
    fixture_results = explore_complete_fixture_data()
    
    # 2. Test all endpoints
    endpoint_results = test_all_endpoints()
    
    # 3. Deep dive into specific match
    match_analysis = analyze_specific_match_data()
    
    # 4. Final summary
    print(f"\n📈 COMPREHENSIVE SUMMARY:")
    print("="*80)
    
    print(f"✅ API-Tennis Complete Data Capabilities:")
    print(f"   📊 Fixtures: {fixture_results.get('total_fixtures', 0)} available")
    print(f"   🏷️  Fields per fixture: {fixture_results.get('total_fields', 0)}")
    print(f"   🎯 Rich data types: {len(fixture_results.get('rich_data_found', {}))}")
    
    for data_type, info in fixture_results.get("rich_data_found", {}).items():
        print(f"     - {data_type.replace('_', ' ').title()}: {len(info['fields'])} fields")
    
    print(f"\n✅ Working endpoints: {sum(1 for r in endpoint_results.values() if r['success'])}/{len(endpoint_results)}")
    
    for endpoint, result in endpoint_results.items():
        if result["success"]:
            print(f"   ✅ {endpoint}: {result.get('count', 'N/A')} items")
        else:
            print(f"   ❌ {endpoint}: No data")
    
    # Key insights for tennis prediction
    print(f"\n🎾 KEY INSIGHTS FOR TENNIS PREDICTION:")
    print("="*50)
    
    insights = []
    
    if "statistics" in fixture_results.get("rich_data_found", {}):
        insights.append("✅ Match statistics available - can extract serve patterns, break points")
    
    if "odds" in fixture_results.get("rich_data_found", {}):
        insights.append("✅ Betting odds available - market sentiment data")
        
    if "point_by_point" in fixture_results.get("rich_data_found", {}):
        insights.append("✅ Point-by-point data - momentum analysis possible")
    
    if endpoint_results.get("get_H2H", {}).get("success"):
        insights.append("✅ Head-to-head data - historical matchup analysis")
        
    if endpoint_results.get("get_standings", {}).get("success"):
        insights.append("✅ Rankings data - current form assessment")
    
    for insight in insights:
        print(f"   {insight}")
    
    if not insights:
        print("   ⚠️  Limited rich data - mainly basic match results")
    
    return {
        "fixture_analysis": fixture_results,
        "endpoint_results": endpoint_results,
        "match_deep_dive": match_analysis
    }

if __name__ == "__main__":
    results = main()
    
    # Save results
    with open("api_tennis_complete_exploration.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to api_tennis_complete_exploration.json")