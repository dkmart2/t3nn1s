#!/usr/bin/env python3
"""
Verify if the user's matches actually exist on Tennis Abstract
"""

import requests
import time
from datetime import datetime

def verify_user_matches():
    """Test if user's comprehensive list matches actually exist"""
    print("🔍 VERIFYING USER'S TENNIS ABSTRACT MATCHES")
    print("="*70)
    print("Testing if user's 196 matches actually exist on Tennis Abstract")
    print()
    
    # Sample of user's matches in different URL formats
    test_matches = [
        # Recent Cincinnati/Montreal (exact matches from user's list)
        ("Cincinnati Sinner", "20250809-M-Cincinnati_Masters-R64-Jannik_Sinner-Daniel_Elahi_Galan"),
        ("Cincinnati Raducanu", "20250809-W-Cincinnati-R64-Emma_Raducanu-Olga_Danilovic"),
        ("Montreal Rybakina", "20250806-W-Montreal-SF-Elena_Rybakina-Victoria_Mboko"),
        ("Montreal Osaka", "20250805-W-Montreal-QF-Naomi_Osaka-Elina_Svitolina"),
        
        # Wimbledon (user shows 55 matches)
        ("Wimbledon Final", "20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner"),
        ("Wimbledon WTA Final", "20250712-W-Wimbledon-F-Amanda_Anisimova-Iga_Swiatek"),
        ("Wimbledon SF", "20250711-M-Wimbledon-SF-Taylor_Fritz-Carlos_Alcaraz"),
        ("Wimbledon SF 2", "20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner"),
        
        # ITF matches (user shows 38)
        ("ITF Granby", "20250720-W-ITF_Granby-F-Talia_Gibson-Fiona_Crawley"),
        ("ITF Palma", "20250629-W-ITF_Palma_Del_Rio-F-Clervie_Ngounoue-Eva_Vedder"),
        
        # Queens Club (user shows 8) 
        ("Queens Final", "20250622-M-Queens_Club-F-Jiri_Lehecka-Carlos_Alcaraz"),
        
        # Stuttgart (user shows 5)
        ("Stuttgart Final", "20250615-M-Stuttgart-F-Taylor_Fritz-Alexander_Zverev"),
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    existing = 0
    not_found = 0
    
    for match_name, match_pattern in test_matches:
        # Test different URL variations
        url_variations = [
            f"https://www.tennisabstract.com/charting/{match_pattern}.html",
            # Try with different naming conventions
            f"https://www.tennisabstract.com/charting/{match_pattern.replace('_Masters', '').replace('_Club', '')}.html",
            # Try replacing underscores
            f"https://www.tennisabstract.com/charting/{match_pattern.replace('_', '-')}.html",
        ]
        
        found = False
        for url in url_variations:
            try:
                response = requests.head(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    existing += 1
                    print(f"  ✅ {match_name}: FOUND")
                    found = True
                    break
            except:
                continue
            time.sleep(0.3)
        
        if not found:
            not_found += 1
            print(f"  ❌ {match_name}: NOT FOUND")
    
    print(f"\n📊 VERIFICATION RESULTS:")
    print(f"Found: {existing}/{len(test_matches)}")
    print(f"Not found: {not_found}/{len(test_matches)}")
    print(f"Success rate: {existing/len(test_matches)*100:.1f}%")
    
    # Test our known working URLs for comparison
    print(f"\n🧪 TESTING OUR KNOWN URLS:")
    known_urls = [
        "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html",
        "https://www.tennisabstract.com/charting/20250615-M-s_Hertogenbosch-F-Gabriel_Diallo-Zizou_Bergs.html",
        "https://www.tennisabstract.com/charting/20250809-M-Cincinnati_Masters-R64-Jannik_Sinner-Daniel_Elahi_Galan.html",
    ]
    
    working = 0
    for url in known_urls:
        try:
            response = requests.head(url, headers=headers, timeout=5)
            if response.status_code == 200:
                working += 1
                match_name = url.split('/')[-1][:50]
                print(f"  ✅ {match_name}")
            else:
                match_name = url.split('/')[-1][:50]
                print(f"  ❌ {match_name}")
        except:
            match_name = url.split('/')[-1][:50]
            print(f"  💥 {match_name}")
        time.sleep(0.5)
    
    print(f"\nOur known URLs working: {working}/{len(known_urls)}")
    
    return existing, not_found

def analyze_user_list_source():
    """Analyze where the user's comprehensive list might be coming from"""
    print(f"\n🤔 ANALYZING USER'S LIST SOURCE")
    print("="*50)
    
    print(f"User's list characteristics:")
    print(f"- 196 matches from June 10+ 2025")
    print(f"- Very comprehensive tournament coverage")
    print(f"- Includes finals, semifinals for major tournaments")
    print(f"- Covers ITF, ATP, WTA across multiple countries")
    print(f"- Systematic round-by-round coverage")
    
    print(f"\nPossible sources:")
    print(f"1. Official ATP/WTA tournament schedules")
    print(f"2. Another tennis data site (not Tennis Abstract)")
    print(f"3. Tennis Abstract's full database (not just public pages)")
    print(f"4. Expected/planned matches (not all actually charted)")
    print(f"5. Multiple tennis data sources combined")
    
    print(f"\nKey insight:")
    print(f"Tennis Abstract is VOLUNTEER-driven charting.")
    print(f"Not every match gets charted - only what volunteers choose to analyze.")
    print(f"The user's list may represent POTENTIAL matches, not ACTUAL Tennis Abstract uploads.")

if __name__ == "__main__":
    existing, not_found = verify_user_matches()
    analyze_user_list_source()
    
    print(f"\n🎯 CONCLUSION:")
    if not_found > existing:
        print(f"❌ Most user matches DON'T exist on Tennis Abstract")
        print(f"The user's list likely comes from tournament schedules, not Tennis Abstract uploads")
        print(f"Our 123 cached matches may represent the actual Tennis Abstract coverage")
    else:
        print(f"✅ User matches exist - our discovery is broken")
        print(f"We need to fix our URL discovery methodology")