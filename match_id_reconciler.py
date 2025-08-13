#!/usr/bin/env python3
"""
Match ID Reconciliation System
Reconciles Jeff, Tennis Abstract, and composite ID formats
"""

import pandas as pd
from datetime import datetime
import os
from pathlib import Path

class MatchIDReconciler:
    """Reconcile Jeff, TA, and composite IDs"""
    
    def __init__(self):
        self.jeff_to_composite = {}
        self.composite_to_jeff = {}
        self.ta_to_jeff = {}
        
    def parse_jeff_id(self, jeff_id):
        """Parse Jeff's match_id: 20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner"""
        parts = jeff_id.split('-', 4)  # Limit split to handle hyphenated names
        
        if len(parts) < 5:
            return None
            
        return {
            'date': parts[0],
            'gender': parts[1],
            'tournament': parts[2],
            'round': parts[3],
            'players_str': parts[4]  # "Novak_Djokovic-Jannik_Sinner"
        }
    
    def jeff_to_standard_composite(self, jeff_id):
        """Convert Jeff ID to composite format"""
        parsed = self.parse_jeff_id(jeff_id)
        if not parsed:
            return None
        
        # Extract player names from last part
        players_str = parsed['players_str']
        # Handle case where players have hyphens in names
        player_parts = players_str.split('-')
        
        if len(player_parts) == 2:
            p1_jeff = player_parts[0]  # "Novak_Djokovic"
            p2_jeff = player_parts[1]  # "Jannik_Sinner"
        else:
            # Complex case - try to split intelligently
            return None
        
        # Convert Jeff player format to canonical
        p1_canonical = self.jeff_name_to_canonical(p1_jeff)
        p2_canonical = self.jeff_name_to_canonical(p2_jeff)
        
        # Build composite using existing logic
        date_obj = datetime.strptime(parsed['date'], '%Y%m%d').date()
        tournament_norm = parsed['tournament'].lower()
        
        return self.build_composite_id(date_obj, tournament_norm, p1_canonical, p2_canonical)
    
    def jeff_name_to_canonical(self, jeff_name):
        """Convert Jeff's Novak_Djokovic to canonical djokovic_n"""
        # Jeff format: FirstName_LastName
        parts = jeff_name.split('_')
        if len(parts) >= 2:
            first = parts[0].lower()
            last = parts[-1].lower()
            return f"{last}_{first[0]}"
        return jeff_name.lower()
    
    def build_composite_id(self, match_date, tourney_slug, p1_slug, p2_slug):
        """Build composite ID with consistent ordering"""
        # Sort players alphabetically for consistency
        players = sorted([p1_slug, p2_slug])
        ymd = match_date.strftime("%Y%m%d")
        return f"{ymd}-{tourney_slug}-{players[0]}-{players[1]}"
    
    def build_crosswalk_from_jeff_data(self, jeff_files_dir):
        """Build crosswalk from Jeff CSV files"""
        jeff_files = [
            'charting-m-stats-ReturnOutcomes.csv',
            'charting-m-stats-ShotTypes.csv',
            'charting-m-points-2020s.csv',
            'charting-m-points-2010s.csv'
        ]
        
        processed_jeff_ids = set()
        
        for file in jeff_files:
            file_path = Path(jeff_files_dir) / file
            if not file_path.exists():
                continue
                
            df = pd.read_csv(file_path)
            if 'match_id' not in df.columns:
                continue
            
            for jeff_id in df['match_id'].dropna().unique():
                if jeff_id in processed_jeff_ids:
                    continue
                    
                composite = self.jeff_to_standard_composite(jeff_id)
                if composite:
                    self.jeff_to_composite[jeff_id] = composite
                    self.composite_to_jeff[composite] = jeff_id
                    processed_jeff_ids.add(jeff_id)
        
        print(f"Crosswalk built: {len(self.jeff_to_composite)} Jeff IDs mapped")
        return self
    
    def add_ta_urls(self, ta_urls):
        """Map TA URLs to Jeff IDs"""
        for ta_url in ta_urls:
            # TA format: 20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner.html
            jeff_format = ta_url.replace('.html', '').split('/')[-1]
            
            if jeff_format in self.jeff_to_composite:
                self.ta_to_jeff[ta_url] = jeff_format
        
        print(f"TA URLs mapped: {len(self.ta_to_jeff)}")
        return self
    
    def get_composite_for_jeff(self, jeff_id):
        """Get composite ID for Jeff match ID"""
        return self.jeff_to_composite.get(jeff_id)
    
    def get_jeff_for_composite(self, composite_id):
        """Get Jeff ID for composite ID"""
        return self.composite_to_jeff.get(composite_id)
    
    def get_jeff_for_ta_url(self, ta_url):
        """Get Jeff ID for TA URL"""
        return self.ta_to_jeff.get(ta_url)
    
    def save_crosswalk(self, filepath):
        """Save crosswalk mappings"""
        crosswalk_data = {
            'jeff_to_composite': self.jeff_to_composite,
            'composite_to_jeff': self.composite_to_jeff,
            'ta_to_jeff': self.ta_to_jeff
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(crosswalk_data, f, indent=2)
        
        print(f"Crosswalk saved to {filepath}")
    
    def load_crosswalk(self, filepath):
        """Load crosswalk mappings"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.jeff_to_composite = data.get('jeff_to_composite', {})
        self.composite_to_jeff = data.get('composite_to_jeff', {})
        self.ta_to_jeff = data.get('ta_to_jeff', {})
        
        print(f"Crosswalk loaded: {len(self.jeff_to_composite)} mappings")
        return self


def test_id_reconciliation():
    """Test that ID reconciliation works"""
    
    # Sample Jeff ID
    jeff_id = "20250711-M-Wimbledon-SF-Novak_Djokovic-Jannik_Sinner"
    
    reconciler = MatchIDReconciler()
    composite = reconciler.jeff_to_standard_composite(jeff_id)
    
    print(f"Jeff ID: {jeff_id}")
    print(f"Composite: {composite}")
    print(f"Expected: 20250711-wimbledon-djokovic_n-sinner_j")
    
    # Test parsing
    parsed = reconciler.parse_jeff_id(jeff_id)
    print(f"\nParsed components:")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    return reconciler


if __name__ == "__main__":
    # Test the reconciliation
    reconciler = test_id_reconciliation()
    
    # Build actual crosswalk
    print("\nBuilding crosswalk from actual data...")
    reconciler.build_crosswalk_from_jeff_data('/Users/danielkim/Desktop/t3nn1s')
    
    # Save crosswalk
    reconciler.save_crosswalk('/Users/danielkim/Desktop/t3nn1s/match_id_crosswalk.json')