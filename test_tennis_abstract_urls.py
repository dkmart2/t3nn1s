#!/usr/bin/env python3
"""
Test Tennis Abstract URLs to find recent matches
"""

import requests
import time
from datetime import datetime, date

def test_tennis_abstract_urls():
    """Test known Tennis Abstract match URLs"""
    print("🔍 TESTING TENNIS ABSTRACT URLS")
    print("="*60)
    
    # Known recent match patterns
    test_urls = [
        # August 2025 - Cincinnati/Montreal
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati-SF-Jannik_Sinner-Alexander_Zverev.html",
        "https://www.tennisabstract.com/charting/20250810-M-Cincinnati-F-Jannik_Sinner-Daniil_Medvedev.html",
        "https://www.tennisabstract.com/charting/20250809-W-Toronto-SF-Jessica_Pegula-Amanda_Anisimova.html",
        "https://www.tennisabstract.com/charting/20250810-W-Toronto-F-Jessica_Pegula-Liudmila_Samsonova.html",
        
        # July 2025 - Wimbledon
        "https://www.tennisabstract.com/charting/20250706-M-Wimbledon-F-Carlos_Alcaraz-Novak_Djokovic.html",
        "https://www.tennisabstract.com/charting/20250706-W-Wimbledon-F-Barbora_Krejcikova-Jasmine_Paolini.html",
        "https://www.tennisabstract.com/charting/20250705-M-Wimbledon-SF-Carlos_Alcaraz-Daniil_Medvedev.html",
        "https://www.tennisabstract.com/charting/20250705-W-Wimbledon-SF-Barbora_Krejcikova-Elena_Rybakina.html",
        
        # Earlier tournaments
        "https://www.tennisabstract.com/charting/20250728-M-Hamburg-F-Sebastian_Baez-Arthur_Fils.html",
        "https://www.tennisabstract.com/charting/20250721-M-Gstaad-F-Matteo_Berrettini-Quentin_Halys.html",
        "https://www.tennisabstract.com/charting/20250720-W-Palermo-F-Qinwen_Zheng-Karolina_Muchova.html",
        
        # June 2025 - Early tournaments  
        "https://www.tennisabstract.com/charting/20250615-M-Stuttgart-F-Matteo_Berrettini-Lorenzo_Musetti.html",
        "https://www.tennisabstract.com/charting/20250622-M-Queens-F-Tommy_Paul-Lorenzo_Musetti.html",
        
        # Try some variations
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati-QF-Jannik_Sinner-Andrey_Rublev.html",
        "https://www.tennisabstract.com/charting/20250808-M-Cincinnati-R16-Jannik_Sinner-Sebastian_Korda.html",
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    existing_urls = []
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n[{i}/{len(test_urls)}] Testing: {url.split('/')[-1]}")
        
        try:
            response = requests.head(url, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                existing_urls.append(url)
                print(f"   ✅ FOUND!")
            else:
                print(f"   ❌ Not found")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        time.sleep(1)  # Be respectful
    
    print(f"\n📊 RESULTS")
    print("="*60)
    print(f"✅ Found {len(existing_urls)} existing Tennis Abstract matches:")
    
    for url in existing_urls:
        match_name = url.split('/')[-1].replace('.html', '')
        print(f"  - {match_name}")
    
    return existing_urls

if __name__ == "__main__":
    found_urls = test_tennis_abstract_urls()
    
    if found_urls:
        print(f"\n🎯 Ready to scrape {len(found_urls)} Tennis Abstract matches!")
    else:
        print(f"\n⚠️  No Tennis Abstract matches found - may need different URL patterns")