#!/usr/bin/env python3
"""
Intelligent Data Merger - API-Tennis + Tennis Abstract
Implements selective override merge strategy for post-June 2025 matches
"""

import pandas as pd
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime

class IntelligentDataMerger:
    """Merges API-Tennis (base) with Tennis Abstract (enhancement)"""
    
    def __init__(self):
        self.data_source_authority = {
            # API is authoritative for these
            'api_authoritative': [
                'event_key', 'player_keys', 'live_odds', 'betting_lines',
                'h2h_record', 'tournament_info', 'match_status', 'point_progression',
                'match_id', 'player_ids', 'odds', 'h2h'
            ],
            
            # TA is authoritative for these (when available)
            'ta_authoritative': [
                'serve_placement', 'serve_wide_pct', 'serve_body_pct', 'serve_t_pct',
                'return_depth', 'return_depth_deep', 'return_depth_short',
                'rally_patterns', 'avg_rally_length', 'rally_length_distribution',
                'approach_shots', 'net_approach_success', 'shot_directions',
                'pressure_details', 'break_point_conversion_detail'
            ],
            
            # Both provide, TA more accurate when available
            'ta_override_when_available': [
                'aces', 'double_faults', 'winners', 'unforced_errors',
                'break_points_saved', 'break_points_faced', 'net_points',
                'first_serve_in', 'first_serve_won', 'second_serve_won'
            ],
            
            # Both provide, use average or API as fallback
            'average_when_both': [
                'first_serve_pct', 'points_won', 'service_games',
                'total_points', 'games_won'
            ]
        }
    
    def merge_api_and_ta_intelligently(self, api_fixture: Dict, ta_scraped: Optional[Dict]) -> Dict:
        """
        Start with complete API data, enhance with TA, 
        override specific fields where TA is superior
        """
        
        # 1. START WITH COMPLETE API DATA (base layer)
        merged = {
            # API exclusive data (TA doesn't have)
            'event_key': api_fixture.get('event_key'),
            'player_keys': api_fixture.get('player_keys'),  
            'live_odds': api_fixture.get('odds'),
            'h2h_history': api_fixture.get('h2h'),
            'tournament_structure': api_fixture.get('tournament_info'),
            'point_progression': api_fixture.get('pointbypoint'),
            'match_status': api_fixture.get('status'),
            
            # API statistics (will selectively override)
            'statistics': api_fixture.get('statistics', {}),
        }
        
        # Copy all basic match info from API
        for field in ['date', 'tournament', 'round', 'surface']:
            if field in api_fixture:
                merged[field] = api_fixture[field]
        
        # 2. ADD TA EXCLUSIVE DATA (when available)
        if ta_scraped:
            # TA exclusive features that API doesn't provide
            ta_exclusive_fields = {
                'serve_direction_patterns': ta_scraped.get('serve_dirs'),
                'serve_wide_pct': ta_scraped.get('serve_wide_pct'),
                'serve_body_pct': ta_scraped.get('serve_body_pct'), 
                'serve_t_pct': ta_scraped.get('serve_t_pct'),
                'return_depth': ta_scraped.get('return_depth'),
                'return_depth_deep': ta_scraped.get('return_deep_pct'),
                'return_depth_short': ta_scraped.get('return_short_pct'),
                'rally_length_distribution': ta_scraped.get('rally_stats'),
                'avg_rally_length': ta_scraped.get('avg_rally_length'),
                'shot_direction_outcomes': ta_scraped.get('shot_dirs'),
                'net_approach_details': ta_scraped.get('net_patterns'),
                'pressure_shot_selection': ta_scraped.get('pressure_stats'),
            }
            
            # Add non-null TA exclusive data
            for field, value in ta_exclusive_fields.items():
                if value is not None:
                    merged[field] = value
        
        # 3. SELECTIVE OVERRIDE (TA better quality for these)
        if ta_scraped:
            override_mappings = {
                # TA field name -> merged field name
                'aces': 'aces',
                'dfs': 'double_faults', 
                '1st_in': 'first_serve_in',
                '1st_won': 'first_serve_won',
                '2nd_won': 'second_serve_won',
                'bp_saved': 'break_points_saved',
                'bp_faced': 'break_points_faced',
                'winners': 'winners',
                'ues': 'unforced_errors',
                'net_pts': 'net_points',
                'net_won': 'net_points_won',
            }
            
            for ta_field, merged_field in override_mappings.items():
                ta_value = ta_scraped.get(ta_field)
                api_value = self._extract_api_stat(api_fixture, merged_field)
                
                if ta_value is not None:
                    merged[merged_field] = ta_value
                    merged[f'{merged_field}_source'] = 'TA'
                elif api_value is not None:
                    merged[merged_field] = api_value  
                    merged[f'{merged_field}_source'] = 'API'
        else:
            # No TA data - use API only
            api_stat_mappings = {
                'aces': 'aces',
                'double_faults': 'double_faults',
                'winners': 'winners', 
                'unforced_errors': 'unforced_errors',
                'first_serve_percentage': 'first_serve_pct',
                'break_points_saved': 'break_points_saved'
            }
            
            for api_field, merged_field in api_stat_mappings.items():
                value = self._extract_api_stat(api_fixture, api_field)
                if value is not None:
                    merged[merged_field] = value
                    merged[f'{merged_field}_source'] = 'API'
        
        # 4. DATA QUALITY TRACKING
        merged['data_completeness'] = {
            'has_api_base': True,
            'has_ta_enhancement': bool(ta_scraped),
            'has_odds': bool(api_fixture.get('odds')),
            'has_point_progression': bool(api_fixture.get('pointbypoint')),
            'has_serve_patterns': bool(ta_scraped and ta_scraped.get('serve_dirs')) if ta_scraped else False,
            'quality_score': self._calculate_quality_score(merged, ta_scraped)
        }
        
        # 5. SOURCE RANKING
        if ta_scraped:
            merged['source_rank'] = 1  # Highest quality
            merged['source_detail'] = 'api_base_ta_enhanced'
        else:
            merged['source_rank'] = 2
            merged['source_detail'] = 'api_only'
        
        merged['available_features'] = self._count_available_features(merged)
        
        return merged
    
    def _extract_api_stat(self, api_fixture: Dict, stat_name: str) -> Optional[float]:
        """Extract a specific statistic from API fixture"""
        statistics = api_fixture.get('statistics', [])
        if not statistics:
            return None
        
        # Handle both list and dict formats
        if isinstance(statistics, list):
            for stat in statistics:
                if isinstance(stat, dict):
                    stat_type = stat.get('type', '').lower().replace(' ', '_')
                    if stat_type == stat_name.lower().replace(' ', '_'):
                        # Return home/away based on context
                        return stat.get('home') or stat.get('away')
        
        elif isinstance(statistics, dict):
            return statistics.get(stat_name)
        
        return None
    
    def _calculate_quality_score(self, merged: Dict, ta_scraped: Optional[Dict]) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.6  # Base API score
        
        if ta_scraped:
            score += 0.3  # TA enhancement bonus
            
            # Bonus for TA exclusive features
            ta_exclusives = ['serve_wide_pct', 'return_depth', 'avg_rally_length']
            for field in ta_exclusives:
                if merged.get(field) is not None:
                    score += 0.03
        
        # Bonus for having odds
        if merged.get('live_odds'):
            score += 0.05
            
        # Bonus for point progression
        if merged.get('point_progression'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _count_available_features(self, merged: Dict) -> int:
        """Count non-null features in merged data"""
        count = 0
        for key, value in merged.items():
            if value is not None and key not in ['data_completeness', 'source_rank', 'source_detail']:
                if isinstance(value, (str, int, float)) and value != '':
                    count += 1
                elif isinstance(value, (list, dict)) and len(value) > 0:
                    count += 1
        return count

    def merge_player_records(self, p1_api: Dict, p1_ta: Optional[Dict], 
                           p2_api: Dict, p2_ta: Optional[Dict]) -> Dict:
        """Merge data for both players in a match"""
        
        # Merge each player's data
        p1_merged = self.merge_api_and_ta_intelligently(p1_api, p1_ta)
        p2_merged = self.merge_api_and_ta_intelligently(p2_api, p2_ta)
        
        # Combine into match record with player prefixes
        match_record = {}
        
        # Add match-level fields from p1 (assumes same for both)
        match_fields = ['date', 'tournament', 'round', 'surface', 'event_key']
        for field in match_fields:
            if field in p1_merged:
                match_record[field] = p1_merged[field]
        
        # Add player-specific fields with prefixes
        for prefix, player_data in [('p1', p1_merged), ('p2', p2_merged)]:
            for field, value in player_data.items():
                if field not in match_fields:
                    match_record[f'{prefix}_{field}'] = value
        
        # Overall match quality
        p1_quality = p1_merged.get('data_completeness', {}).get('quality_score', 0.6)
        p2_quality = p2_merged.get('data_completeness', {}).get('quality_score', 0.6)
        match_record['overall_quality_score'] = (p1_quality + p2_quality) / 2
        
        return match_record

def test_intelligent_merger():
    """Test the intelligent merge functionality"""
    
    # Sample API fixture
    api_fixture = {
        'event_key': 'atp_2025_wimbledon_123',
        'odds': {'home': 1.85, 'away': 1.95},
        'h2h': '3-2',
        'pointbypoint': ['0-0', '15-0', '15-15', '30-15'],
        'statistics': [
            {'type': 'aces', 'home': 10, 'away': 8},
            {'type': 'winners', 'home': 25, 'away': 30},
            {'type': 'first serve percentage', 'home': 65, 'away': 72}
        ]
    }
    
    # Sample TA scraped data
    ta_scraped = {
        'aces': 12,  # More accurate count
        'winners': 28,  # More accurate count
        'serve_wide_pct': 0.43,  # TA exclusive
        'serve_body_pct': 0.31,  # TA exclusive
        'serve_t_pct': 0.26,  # TA exclusive
        'return_deep_pct': 0.68,  # TA exclusive
        'avg_rally_length': 4.2,  # TA exclusive
    }
    
    merger = IntelligentDataMerger()
    
    # Test with TA enhancement
    print("=== MERGE WITH TA ENHANCEMENT ===")
    enhanced = merger.merge_api_and_ta_intelligently(api_fixture, ta_scraped)
    
    print(f"Quality Score: {enhanced['data_completeness']['quality_score']:.2f}")
    print(f"Source Rank: {enhanced['source_rank']} ({enhanced['source_detail']})")
    print(f"Available Features: {enhanced['available_features']}")
    print(f"Aces: {enhanced['aces']} (source: {enhanced['aces_source']})")
    print(f"Serve Wide %: {enhanced.get('serve_wide_pct', 'N/A')}")
    
    # Test without TA enhancement
    print("\n=== MERGE API ONLY ===")
    api_only = merger.merge_api_and_ta_intelligently(api_fixture, None)
    
    print(f"Quality Score: {api_only['data_completeness']['quality_score']:.2f}")
    print(f"Source Rank: {api_only['source_rank']} ({api_only['source_detail']})")
    print(f"Available Features: {api_only['available_features']}")
    print(f"Aces: {api_only.get('aces', 'N/A')} (source: {api_only.get('aces_source', 'N/A')})")
    
    return enhanced, api_only

if __name__ == "__main__":
    enhanced, api_only = test_intelligent_merger()
    print("\n✅ Intelligent merger test complete!")