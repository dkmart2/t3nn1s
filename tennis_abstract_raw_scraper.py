#!/usr/bin/env python3
"""
Tennis Abstract scraper that extracts RAW statistics instead of percentages
Based on discovered HTML structure: "20  (27%)" where 20 is the raw count
"""

import requests
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import time
from bs4 import BeautifulSoup

def extract_raw_stat(stat_text):
    """Extract raw count from Tennis Abstract format like '20  (27%)'"""
    if not stat_text:
        return 0
    
    # Pattern: number followed by optional spaces and parentheses with percentage
    match = re.search(r'(\d+)\s*\(\d+%\)', stat_text.strip())
    if match:
        return int(match.group(1))
    
    # Fallback: just extract first number
    match = re.search(r'(\d+)', stat_text.strip())
    if match:
        return int(match.group(1))
    
    return 0

def extract_player_stats_from_html(html, match_url):
    """Extract raw statistics for both players from Tennis Abstract HTML"""
    
    # Parse match info from title/URL
    url_parts = match_url.split('/')[-1].replace('.html', '').split('-')
    if len(url_parts) >= 6:
        date_str = url_parts[0]
        gender = url_parts[1] 
        tournament = url_parts[2]
        round_name = url_parts[3]
        player1 = url_parts[4].replace('_', ' ')
        player2 = url_parts[5].replace('_', ' ')
    else:
        return []
    
    match_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # The statistics are embedded in JavaScript variables
    # Look for the serve table in JavaScript
    serve_table_match = re.search(r"var serve = '(<table.*?</table>)", html, re.DOTALL)
    if not serve_table_match:
        print(f"No serve stats found in JavaScript for {match_url}")
        return []
    
    serve_table_html = serve_table_match.group(1).replace('\\n', '\n').replace("\\'", "'")
    soup = BeautifulSoup(serve_table_html, 'html.parser')
    
    serve_table = soup.find('table')
    if not serve_table:
        print(f"Could not parse serve table for {match_url}")
        return []
        
    rows = serve_table.find_all('tr')
    
    player_stats = []
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 9:
            continue
            
        player_label = cells[0].get_text(strip=True)
        if not player_label or 'Total' not in player_label:
            continue
            
        # Extract player code (TF, AR, etc.)
        player_code = player_label.split()[0]
        
        # Map player code to actual name based on position
        # Tennis Abstract uses player initials - first player gets first code alphabetically
        if player_code in ['TF', 'TC', 'TA', 'TB']:
            player_name = player1  # Likely Taylor Fritz or first player
        elif player_code in ['AR', 'AS', 'AD', 'AB']:
            player_name = player2  # Likely Andrey Rublev or second player
        else:
            # Generic mapping - assume first occurrence = first player
            player_name = player1
        
        # Extract raw statistics
        serve_pts = extract_raw_stat(cells[1].get_text(strip=True))
        points_won_raw = extract_raw_stat(cells[2].find('span').get_text(strip=True) if cells[2].find('span') else cells[2].get_text(strip=True))
        aces = extract_raw_stat(cells[3].find('span').get_text(strip=True) if cells[3].find('span') else cells[3].get_text(strip=True))
        unreturnables = extract_raw_stat(cells[4].find('span').get_text(strip=True) if cells[4].find('span') else cells[4].get_text(strip=True))
        forced_errors = extract_raw_stat(cells[5].find('span').get_text(strip=True) if cells[5].find('span') else cells[5].get_text(strip=True))
        three_stroke_wins = extract_raw_stat(cells[6].find('span').get_text(strip=True) if cells[6].find('span') else cells[6].get_text(strip=True))
        wide_serves = extract_raw_stat(cells[7].find('span').get_text(strip=True) if cells[7].find('span') else cells[7].get_text(strip=True))
        body_serves = extract_raw_stat(cells[8].find('span').get_text(strip=True) if cells[8].find('span') else cells[8].get_text(strip=True))
        t_serves = extract_raw_stat(cells[9].find('span').get_text(strip=True) if cells[9].find('span') else cells[9].get_text(strip=True))
        
        # Look for first serve and second serve data in breakdown tables
        first_serve_pts = 0
        first_serve_won = 0
        second_serve_pts = 0 
        second_serve_won = 0
        double_faults = 0
        
        # Find breakdown table for this player
        breakdown_tables = soup.find_all('table', {'class': 'tablesorter'})
        for table in breakdown_tables:
            table_rows = table.find_all('tr')
            for table_row in table_rows:
                table_cells = table_row.find_all('td')
                if len(table_cells) >= 8:
                    row_label = table_cells[0].get_text(strip=True)
                    if f"{player_code} 1st" in row_label:
                        first_serve_pts = extract_raw_stat(table_cells[1].get_text(strip=True))
                        first_serve_won = extract_raw_stat(table_cells[2].find('span').get_text(strip=True) if table_cells[2].find('span') else table_cells[2].get_text(strip=True))
                    elif f"{player_code} 2nd" in row_label:
                        second_serve_pts = extract_raw_stat(table_cells[1].get_text(strip=True))
                        second_serve_won = extract_raw_stat(table_cells[2].find('span').get_text(strip=True) if table_cells[2].find('span') else table_cells[2].get_text(strip=True))
                        if len(table_cells) >= 8:
                            double_faults = extract_raw_stat(table_cells[7].get_text(strip=True))
        
        # Create Jeff-format record
        record = {
            'match_id': match_url.split('/')[-1].replace('.html', ''),
            'date': match_date,
            'tournament': tournament.replace('_', ' '),
            'round': round_name,
            'gender': gender,
            'player': player_name.lower().replace(' ', '_'),
            'set': 'Total',
            'serve_pts': serve_pts,
            'aces': aces,
            'dfs': double_faults,
            'first_in': first_serve_pts,
            'first_won': first_serve_won, 
            'second_in': second_serve_pts,
            'second_won': second_serve_won,
            'winners': three_stroke_wins,  # Approximate
            'unforced': 0,  # Not directly available
            'bk_pts': 0,    # Not in this table
            'bp_saved': 0,  # Not in this table
            'return_pts': 0, # Need to calculate from opponent serve_pts
            'return_pts_won': 0,
            'winners_fh': 0,
            'winners_bh': 0,
            'unforced_fh': 0,
            'unforced_bh': 0,
            'url': match_url
        }
        
        player_stats.append(record)
    
    return player_stats

def scrape_tennis_abstract_raw():
    """Scrape Tennis Abstract with proper raw statistic extraction"""
    
    # Load URLs to scrape
    urls_file = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_196_urls.txt')
    if not urls_file.exists():
        print(f"URLs file not found: {urls_file}")
        return
        
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Scraping {len(urls)} Tennis Abstract matches for RAW statistics...")
    
    all_records = []
    failed_urls = []
    
    for i, url in enumerate(urls):
        print(f"Scraping {i+1}/{len(urls)}: {url}", flush=True)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract statistics from HTML
            match_stats = extract_player_stats_from_html(response.text, url)
            
            if match_stats:
                all_records.extend(match_stats)
                print(f"  ✓ Extracted {len(match_stats)} player records", flush=True)
            else:
                print(f"  ✗ No stats extracted", flush=True)
                failed_urls.append(url)
                
        except Exception as e:
            print(f"  ✗ Error: {e}", flush=True)
            failed_urls.append(url)
            
        # Rate limiting
        time.sleep(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    print(f"\n=== SCRAPING COMPLETE ===")
    print(f"Successfully scraped: {len(all_records)} player records")
    print(f"Failed URLs: {len(failed_urls)}")
    print(f"Unique matches: {df['match_id'].nunique() if len(df) > 0 else 0}")
    
    if len(df) > 0:
        # Save results
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_raw_stats')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        raw_file = output_dir / f'tennis_abstract_raw_{timestamp}.parquet'
        df.to_parquet(raw_file, index=False)
        
        csv_file = output_dir / f'tennis_abstract_raw_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\nSaved raw statistics:")
        print(f"  Parquet: {raw_file}")
        print(f"  CSV: {csv_file}")
        
        # Show sample
        print(f"\nSample records:")
        print(df[['match_id', 'player', 'serve_pts', 'aces', 'first_in', 'winners']].head(10))
        
        # Statistics
        print(f"\n=== STATISTICS ===")
        print(f"Non-zero aces: {(df['aces'] > 0).sum()} records")
        print(f"Non-zero serve points: {(df['serve_pts'] > 0).sum()} records")
        print(f"Average aces per match: {df['aces'].mean():.1f}")
        print(f"Average serve points: {df['serve_pts'].mean():.1f}")
    
    if failed_urls:
        print(f"\nFailed URLs ({len(failed_urls)}):")
        for url in failed_urls[:10]:  # Show first 10
            print(f"  {url}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")

if __name__ == "__main__":
    scrape_tennis_abstract_raw()