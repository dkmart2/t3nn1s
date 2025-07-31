#!/usr/bin/env python3
import os

os.environ["API_TENNIS_KEY"] = "adfc70491c47895e5fffdc6428bbf36a561989d4bffcfa9ecfba8d91e947b4fb"

import pandas as pd
from tennis_updated import TennisAbstractScraper
from pathlib import Path


def find_common_match():
    """Find a match that exists in both Jeff CSV and TA website"""

    # Load Jeff's matches
    jeff_matches = pd.read_csv('charting-m-matches.csv')
    print("Jeff matches sample:")
    print(jeff_matches[['match_id', 'Date', 'Player1', 'Player2']].head())

    # Check what match_ids look like in overview
    jeff_overview = pd.read_csv('charting-m-stats-Overview.csv')
    print("Jeff overview match_ids sample:")
    print(jeff_overview['match_id'].head())

    # Compare to scraped TA URLs
    scraped_urls = [
        "20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner.html",
        "20250709-M-Wimbledon-QF-Novak_Djokovic-Flavio_Cobolli.html"
    ]

    # Look for overlap
    for url in scraped_urls:
        match_id = url.replace('.html', '')
        print(f"Checking if {match_id} exists in Jeff CSV...")

        jeff_match = jeff_overview[jeff_overview['match_id'].str.contains(match_id, na=False)]
        if not jeff_match.empty:
            print(f"FOUND MATCH: {match_id}")
            return match_id, url

    return None, None


def compare_match_data_sources(match_id, ta_url):
    """Compare Jeff CSV vs TA website data for same match"""

    # Jeff CSV data for this match
    print("=== JEFF CSV DATA ===")
    csv_files = [
        'charting-m-stats-Overview.csv',
        'charting-m-stats-ServeBasics.csv',
        'charting-m-stats-ReturnOutcomes.csv',
        'charting-m-stats-KeyPointsServe.csv'
    ]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            match_data = df[df['match_id'] == match_id]
            print(f"{csv_file}: {len(match_data)} records, {len(df.columns)} columns")
            if len(match_data) > 0:
                print(f"  Columns: {list(df.columns)}")

    # TA website data for same match
    print("\n=== TENNIS ABSTRACT WEBSITE DATA ===")
    scraper = TennisAbstractScraper()
    scraped_data = scraper.scrape_comprehensive_match_data(ta_url)

    # Count by data_type
    data_types = {}
    for record in scraped_data:
        dt = record.get('data_type', 'unknown')
        data_types[dt] = data_types.get(dt, 0) + 1

    for data_type, count in sorted(data_types.items()):
        print(f"{data_type}: {count} records")


if __name__ == "__main__":
    match_id, ta_url = find_common_match()
    if match_id:
        compare_match_data_sources(match_id, ta_url)
    else:
        print("No common match found between Jeff CSV and scraped TA URLs")