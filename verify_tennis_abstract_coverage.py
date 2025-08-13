#!/usr/bin/env python3
"""
Verify Tennis Abstract coverage - find the gap between 127 actual and 111 cached
"""

import requests
import time
from datetime import datetime, date
from pathlib import Path

def test_specific_tennis_abstract_matches():
    """Test specific high-profile matches that should definitely exist"""
    print("🔍 TESTING SPECIFIC HIGH-PROFILE TENNIS ABSTRACT MATCHES")
    print("="*70)
    print("Goal: Find the missing matches from our cache")
    print()
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    # Load our existing cache
    try:
        cache_file = '/Users/danielkim/Desktop/t3nn1s/scripts/cache/tennis_abstract_cache/scraped_urls.txt'
        with open(cache_file, 'r') as f:
            cached_urls = set(line.strip() for line in f if line.strip())
        print(f"✅ Loaded {len(cached_urls)} cached URLs")
    except:
        cached_urls = set()
        print("❌ No cache found")
    
    # Test key matches that should exist based on WebFetch findings
    key_matches_to_test = [
        # Recent Cincinnati/Montreal matches from WebFetch sample
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati_Masters-R64-Jannik_Sinner-Daniel_Elahi_Galan.html",
        "https://www.tennisabstract.com/charting/20250809-W-Cincinnati-R64-Emma_Raducanu-Olga_Danilovic.html",
        "https://www.tennisabstract.com/charting/20250806-W-Montreal-SF-Elena_Rybakina-Victoria_Mboko.html",
        "https://www.tennisabstract.com/charting/20250805-W-Montreal-QF-Naomi_Osaka-Elina_Svitolina.html",
        "https://www.tennisabstract.com/charting/20250805-W-Montreal-QF-Clara_Tauson-Madison_Keys.html",
        "https://www.tennisabstract.com/charting/20250804-W-Montreal-QF-Victoria_Mboko-Jessica_Bouzas_Maneiro.html",
        "https://www.tennisabstract.com/charting/20250803-W-Montreal-R16-Iga_Swiatek-Clara_Tauson.html",
        "https://www.tennisabstract.com/charting/20250803-W-Montreal-R16-Elina_Svitolina-Amanda_Anisimova.html",
        "https://www.tennisabstract.com/charting/20250802-W-Montreal-R16-Coco_Gauff-Victoria_Mboko.html",
        "https://www.tennisabstract.com/charting/20250801-W-Montreal-R32-Amanda_Anisimova-Emma_Raducanu.html",
        
        # More Wimbledon matches (we only have 37 out of 54)
        "https://www.tennisabstract.com/charting/20250714-M-Wimbledon-F-Carlos_Alcaraz-Novak_Djokovic.html",
        "https://www.tennisabstract.com/charting/20250713-W-Wimbledon-F-Barbora_Krejcikova-Jasmine_Paolini.html", 
        "https://www.tennisabstract.com/charting/20250712-M-Wimbledon-SF-Novak_Djokovic-Lorenzo_Musetti.html",
        "https://www.tennisabstract.com/charting/20250712-M-Wimbledon-SF-Carlos_Alcaraz-Daniil_Medvedev.html",
        "https://www.tennisabstract.com/charting/20250711-W-Wimbledon-SF-Barbora_Krejcikova-Elena_Rybakina.html",
        "https://www.tennisabstract.com/charting/20250711-W-Wimbledon-SF-Jasmine_Paolini-Donna_Vekic.html",
        "https://www.tennisabstract.com/charting/20250710-M-Wimbledon-QF-Carlos_Alcaraz-Tommy_Paul.html",
        "https://www.tennisabstract.com/charting/20250710-M-Wimbledon-QF-Novak_Djokovic-Holger_Rune.html",
        "https://www.tennisabstract.com/charting/20250710-W-Wimbledon-QF-Barbora_Krejcikova-Jelena_Ostapenko.html",
        "https://www.tennisabstract.com/charting/20250710-W-Wimbledon-QF-Jasmine_Paolini-Emma_Navarro.html",
        
        # More s-Hertogenbosch matches
        "https://www.tennisabstract.com/charting/20250615-M-s_Hertogenbosch-F-Gabriel_Diallo-Zizou_Bergs.html",
        "https://www.tennisabstract.com/charting/20250615-W-s_Hertogenbosch-F-Elise_Mertens-Elena_Gabriela_Ruse.html",
        "https://www.tennisabstract.com/charting/20250614-M-s_Hertogenbosch-SF-Zizou_Bergs-Reilly_Opelka.html",
        "https://www.tennisabstract.com/charting/20250614-W-s_Hertogenbosch-SF-Ekaterina_Alexandrova-Elise_Mertens.html",
        "https://www.tennisabstract.com/charting/20250613-W-s_Hertogenbosch-QF-Bianca_Andreescu-Elena_Gabriela_Ruse.html",
        "https://www.tennisabstract.com/charting/20250612-M-s_Hertogenbosch-R16-Giovanni_Mpetshi_Perricard-Felix_Auger_Aliassime.html",
        "https://www.tennisabstract.com/charting/20250611-M-s_Hertogenbosch-R32-Hubert_Hurkacz-Roberto_Bautista_Agut.html",
        "https://www.tennisabstract.com/charting/20250610-W-s_Hertogenbosch-R32-Lulu_Sun-Anouck_Vrancken_Peeters.html",
        "https://www.tennisabstract.com/charting/20250610-W-s_Hertogenbosch-R32-Joanna_Garland-Bianca_Andreescu.html",
        "https://www.tennisabstract.com/charting/20250610-W-s_Hertogenbosch-R32-Arianne_Hartono-Elisabetta_Cocciaretto.html",
        
        # Additional key tournaments that should have matches
        "https://www.tennisabstract.com/charting/20250721-M-Hamburg-F-Sebastian_Baez-Alexander_Zverev.html",
        "https://www.tennisabstract.com/charting/20250720-W-Hamburg-F-Diana_Shnaider-Eva_Lys.html",
        "https://www.tennisabstract.com/charting/20250728-M-Atlanta-F-Ben_Shelton-Frances_Tiafoe.html",
        "https://www.tennisabstract.com/charting/20250804-M-Washington-F-Sebastian_Korda-Flavio_Cobolli.html",
    ]
    
    existing_matches = []
    missing_matches = []
    
    print(f"Testing {len(key_matches_to_test)} high-profile matches...")
    
    for i, url in enumerate(key_matches_to_test, 1):
        match_name = url.split('/')[-1].replace('.html', '')
        
        # Check if we already have it cached
        if url in cached_urls:
            existing_matches.append(url)
            print(f"  ✅ [{i:2d}] CACHED: {match_name[:60]}")
            continue
        
        # Test if URL exists on Tennis Abstract
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                missing_matches.append(url)
                print(f"  🆕 [{i:2d}] MISSING: {match_name[:60]}")
            else:
                print(f"  ❌ [{i:2d}] NOT FOUND: {match_name[:60]}")
        except:
            print(f"  ❓ [{i:2d}] ERROR: {match_name[:60]}")
        
        time.sleep(0.3)  # Be respectful
    
    print(f"\n📊 TESTING RESULTS:")
    print(f"✅ Already cached: {len(existing_matches)}")
    print(f"🆕 Missing from cache (but exist): {len(missing_matches)}")
    print(f"❌ Not found on Tennis Abstract: {len(key_matches_to_test) - len(existing_matches) - len(missing_matches)}")
    
    if missing_matches:
        print(f"\n🎯 MISSING MATCHES IDENTIFIED:")
        for i, url in enumerate(missing_matches, 1):
            match_name = url.split('/')[-1].replace('.html', '')
            date_str = match_name[:8]
            try:
                match_date = datetime.strptime(date_str, '%Y%m%d').date()
                print(f"  {i:2d}. {match_date}: {match_name}")
            except:
                print(f"  {i:2d}. {match_name}")
        
        # Save missing URLs for scraping
        output_dir = Path('/Users/danielkim/Desktop/t3nn1s/tennis_abstract_missing')
        output_dir.mkdir(exist_ok=True)
        
        missing_file = output_dir / f'missing_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(missing_file, 'w') as f:
            for url in missing_matches:
                f.write(url + '\n')
        
        print(f"\n💾 Saved {len(missing_matches)} missing URLs to: {missing_file}")
        
        return missing_matches
    else:
        print(f"\n✅ All tested matches are already in cache")
        return []

if __name__ == "__main__":
    missing = test_specific_tennis_abstract_matches()
    
    if missing:
        print(f"\n🚨 COVERAGE GAP CONFIRMED!")
        print(f"Found {len(missing)} missing Tennis Abstract matches")
        print(f"These need to be scraped to achieve complete coverage")
    else:
        print(f"\n🎯 Our cache appears complete for major matches")
        print(f"The 127 vs 111 discrepancy may be due to minor/ITF matches")