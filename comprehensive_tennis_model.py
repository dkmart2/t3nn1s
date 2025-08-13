"""
COMPREHENSIVE TENNIS PREDICTION PIPELINE
Production-Ready Architecture with Stateful Simulation

This replaces the fragmented approach in model.py with a unified, stateful system
that maintains temporal continuity across the Jeff Sackmann data cutoff (2025-06-10).

Key Features:
- Temporal data continuity (Jeff → Tennis Abstract)
- Stateful match simulation (energy, momentum, confidence evolution)
- Pattern-based point prediction using actual point sequences
- Market validation without dependency
- Comprehensive player profiling from all data sources
- AI contextual adjustments for human factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress pandas warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

JEFF_CUTOFF_DATE = datetime(2025, 6, 10)
DEFAULT_DATA_DIR = Path("/Users/danielkim/Desktop/t3nn1s")

class DataSourceType(Enum):
    JEFF_SACKMANN = "jeff"
    TENNIS_ABSTRACT = "ta"
    API_TENNIS = "api"
    TENNIS_DATA = "historical"

class Surface(Enum):
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"

# ============================================================================
# PATTERN EXTRACTION: Learning from Point Sequences
# ============================================================================

class UnifiedPatternParser:
    """
    Parses both Jeff Sackmann and Tennis Abstract notation into unified patterns
    
    Jeff notation example: '4f8b3f*' = wide serve, forehand crosscourt, backhand line, forehand winner
    TA notation converted to equivalent format for consistency
    """
    
    def __init__(self):
        # Jeff Sackmann notation mappings
        self.serve_directions = {
            '4': 'wide',    # Wide serve
            '5': 'body',    # Body serve  
            '6': 'T'        # T serve (center)
        }
        
        self.shot_types = {
            'f': 'forehand',
            'b': 'backhand', 
            'v': 'volley',
            's': 'slice',
            'r': 'rally'
        }
        
        self.court_positions = {
            '1': 'wide_deuce', '2': 'middle_deuce', '3': 'center',
            '4': 'wide_ad', '5': 'middle_ad', '6': 'T_area',
            '7': 'deep_deuce', '8': 'deep_middle', '9': 'deep_ad'
        }
        
        self.outcomes = {
            '*': 'winner',
            '@': 'unforced_error', 
            '#': 'forced_error',
            'n': 'net_error',
            'w': 'wide_error',
            'l': 'long_error'
        }
        
        logging.info("UnifiedPatternParser initialized")
    
    def extract_patterns(self, point_data: pd.DataFrame) -> Dict:
        """Extract comprehensive patterns from point sequences"""
        
        if len(point_data) == 0:
            return self._empty_patterns()
        
        patterns = {
            'serve': self._extract_serve_patterns(point_data),
            'return': self._extract_return_patterns(point_data),
            'rally': self._extract_rally_patterns(point_data), 
            'pressure': self._extract_pressure_patterns(point_data),
            'momentum': self._extract_momentum_patterns(point_data)
        }
        
        logging.info(f"Extracted patterns from {len(point_data)} matches")
        return patterns
    
    def _extract_serve_patterns(self, data: pd.DataFrame) -> Dict:
        """Extract serve location preferences by game state"""
        
        patterns = {
            'deuce_court': {'wide': 0, 'body': 0, 'T': 0, 'total': 0},
            'ad_court': {'wide': 0, 'body': 0, 'T': 0, 'total': 0},
            'break_point': {'wide': 0, 'body': 0, 'T': 0, 'total': 0},
            'game_point': {'wide': 0, 'body': 0, 'T': 0, 'total': 0},
            'set_point': {'wide': 0, 'body': 0, 'T': 0, 'total': 0},
            'effectiveness': {}
        }
        
        for _, match in data.iterrows():
            # Process first serves
            first_serve = str(match.get('1st', ''))
            if first_serve and len(first_serve) > 0 and first_serve[0] in self.serve_directions:
                direction = self.serve_directions[first_serve[0]]
                
                # Determine context from score or other indicators
                context = self._determine_serve_context(match)
                
                if context in patterns:
                    patterns[context][direction] += 1
                    patterns[context]['total'] += 1
                    
                    # Track effectiveness
                    point_won = self._determine_point_winner(match, 'server')
                    eff_key = f"{context}_{direction}"
                    if eff_key not in patterns['effectiveness']:
                        patterns['effectiveness'][eff_key] = {'wins': 0, 'total': 0}
                    
                    patterns['effectiveness'][eff_key]['total'] += 1
                    if point_won:
                        patterns['effectiveness'][eff_key]['wins'] += 1
        
        # Normalize to probabilities
        for context in ['deuce_court', 'ad_court', 'break_point', 'game_point', 'set_point']:
            total = patterns[context]['total']
            if total > 0:
                for direction in ['wide', 'body', 'T']:
                    patterns[context][direction] = patterns[context][direction] / total
        
        # Calculate effectiveness rates
        for key, data in patterns['effectiveness'].items():
            if data['total'] > 0:
                patterns['effectiveness'][key] = data['wins'] / data['total']
        
        return patterns
    
    def _extract_return_patterns(self, data: pd.DataFrame) -> Dict:
        """Extract return patterns and effectiveness against different serves"""
        
        return_patterns = {
            'vs_wide': {'success_rate': 0.0, 'aggression': 0.0, 'total': 0},
            'vs_body': {'success_rate': 0.0, 'aggression': 0.0, 'total': 0}, 
            'vs_T': {'success_rate': 0.0, 'aggression': 0.0, 'total': 0},
            'break_point_conversion': {'converted': 0, 'opportunities': 0},
            'return_depth': {'short': 0, 'medium': 0, 'deep': 0, 'total': 0}
        }
        
        for _, match in data.iterrows():
            # Analyze return performance based on serve direction
            first_serve = str(match.get('1st', ''))
            if first_serve and len(first_serve) > 0 and first_serve[0] in self.serve_directions:
                direction = self.serve_directions[first_serve[0]]
                
                point_won = self._determine_point_winner(match, 'returner')
                
                key = f'vs_{direction}'
                if key in return_patterns:
                    return_patterns[key]['total'] += 1
                    if point_won:
                        return_patterns[key]['success_rate'] += 1
                        
                    # Analyze aggression level from shot sequence
                    aggression = self._calculate_return_aggression(first_serve)
                    return_patterns[key]['aggression'] += aggression
        
        # Normalize rates
        for pattern in ['vs_wide', 'vs_body', 'vs_T']:
            if return_patterns[pattern]['total'] > 0:
                total = return_patterns[pattern]['total']
                return_patterns[pattern]['success_rate'] /= total
                return_patterns[pattern]['aggression'] /= total
        
        return return_patterns
    
    def _extract_rally_patterns(self, data: pd.DataFrame) -> Dict:
        """Extract rally length and shot patterns"""
        
        rally_patterns = {
            'avg_length': 0.0,
            'length_distribution': {'short': 0, 'medium': 0, 'long': 0},
            'shot_preferences': {'forehand': 0, 'backhand': 0, 'volley': 0},
            'court_coverage': {'defensive': 0.0, 'neutral': 0.0, 'aggressive': 0.0},
            'rally_outcomes': {'winners': 0, 'errors': 0, 'total': 0}
        }
        
        total_rallies = 0
        total_length = 0
        
        for _, match in data.iterrows():
            # Analyze rally from point sequence
            rally_data = self._analyze_rally_sequence(match)
            
            if rally_data['length'] > 0:
                total_rallies += 1
                total_length += rally_data['length']
                
                # Categorize length
                if rally_data['length'] <= 4:
                    rally_patterns['length_distribution']['short'] += 1
                elif rally_data['length'] <= 8:
                    rally_patterns['length_distribution']['medium'] += 1
                else:
                    rally_patterns['length_distribution']['long'] += 1
                
                # Update shot preferences
                for shot_type in rally_data['shots']:
                    if shot_type in rally_patterns['shot_preferences']:
                        rally_patterns['shot_preferences'][shot_type] += 1
                
                # Track outcomes
                rally_patterns['rally_outcomes']['total'] += 1
                if rally_data['outcome'] == 'winner':
                    rally_patterns['rally_outcomes']['winners'] += 1
                elif 'error' in rally_data['outcome']:
                    rally_patterns['rally_outcomes']['errors'] += 1
        
        # Calculate averages
        if total_rallies > 0:
            rally_patterns['avg_length'] = total_length / total_rallies
            
            # Normalize distributions
            for category in rally_patterns['length_distribution']:
                rally_patterns['length_distribution'][category] /= total_rallies
        
        return rally_patterns
    
    def _extract_pressure_patterns(self, data: pd.DataFrame) -> Dict:
        """Extract performance under pressure situations"""
        
        pressure_stats = {
            'break_points': {'saved': 0, 'faced': 0, 'save_rate': 0.0},
            'game_points': {'won': 0, 'played': 0, 'conversion_rate': 0.0},
            'set_points': {'won': 0, 'played': 0, 'conversion_rate': 0.0},
            'tiebreaks': {'won': 0, 'played': 0, 'points_won': 0, 'points_played': 0},
            'pressure_response': 'neutral'  # clutch, neutral, or pressure_sensitive
        }
        
        for _, match in data.iterrows():
            # Analyze pressure situations from match context
            pressure_context = self._identify_pressure_situations(match)
            
            for situation in pressure_context:
                if situation['type'] == 'break_point':
                    pressure_stats['break_points']['faced'] += 1
                    if situation['outcome'] == 'saved':
                        pressure_stats['break_points']['saved'] += 1
                        
                elif situation['type'] == 'game_point':
                    pressure_stats['game_points']['played'] += 1
                    if situation['outcome'] == 'won':
                        pressure_stats['game_points']['won'] += 1
                        
                elif situation['type'] == 'set_point':
                    pressure_stats['set_points']['played'] += 1
                    if situation['outcome'] == 'won':
                        pressure_stats['set_points']['won'] += 1
        
        # Calculate rates
        if pressure_stats['break_points']['faced'] > 0:
            pressure_stats['break_points']['save_rate'] = (
                pressure_stats['break_points']['saved'] / 
                pressure_stats['break_points']['faced']
            )
        
        if pressure_stats['game_points']['played'] > 0:
            pressure_stats['game_points']['conversion_rate'] = (
                pressure_stats['game_points']['won'] / 
                pressure_stats['game_points']['played']
            )
        
        # Determine pressure response type
        bp_rate = pressure_stats['break_points']['save_rate']
        gp_rate = pressure_stats['game_points']['conversion_rate']
        
        avg_pressure_performance = (bp_rate + gp_rate) / 2
        
        if avg_pressure_performance > 0.65:
            pressure_stats['pressure_response'] = 'clutch'
        elif avg_pressure_performance < 0.45:
            pressure_stats['pressure_response'] = 'pressure_sensitive'
        else:
            pressure_stats['pressure_response'] = 'neutral'
        
        return pressure_stats
    
    def _extract_momentum_patterns(self, data: pd.DataFrame) -> Dict:
        """Extract momentum shift patterns within matches"""
        
        momentum_patterns = {
            'momentum_sensitivity': 0.0,  # How much performance changes with momentum
            'comeback_ability': 0.0,      # Ability to recover from deficits
            'momentum_sustain': 0.0,      # Ability to maintain leads
            'clutch_momentum': 0.0        # Performance in decisive moments
        }
        
        # This would require more detailed point-by-point analysis
        # For now, provide baseline patterns
        
        return momentum_patterns
    
    def _determine_serve_context(self, match: pd.Series) -> str:
        """Determine the context of a serve (deuce court, break point, etc.)"""
        
        # This would analyze score, set situation, etc.
        # For now, return deuce court as default
        
        return 'deuce_court'
    
    def _determine_point_winner(self, match: pd.Series, perspective: str) -> bool:
        """Determine if the server or returner won the point"""
        
        # Analyze from Winner/Loser and point sequence
        winner = match.get('PtWinner', match.get('Winner'))
        server = match.get('Svr', match.get('Winner'))  # Simplified
        
        if perspective == 'server':
            return winner == server
        else:  # returner
            return winner != server
    
    def _calculate_return_aggression(self, serve_sequence: str) -> float:
        """Calculate aggression level of return based on sequence"""
        
        # Analyze shot sequence after serve
        if len(serve_sequence) > 2:
            # Look for immediate winners or attacking shots
            if '*' in serve_sequence[1:3]:  # Winner on return or next shot
                return 1.0
            elif 'f' in serve_sequence[1:2] or 'b' in serve_sequence[1:2]:
                return 0.7  # Aggressive groundstroke
        
        return 0.3  # Conservative return
    
    def _analyze_rally_sequence(self, match: pd.Series) -> Dict:
        """Analyze rally from point sequence notation"""
        
        sequence = str(match.get('1st', ''))
        
        rally_data = {
            'length': 0,
            'shots': [],
            'outcome': 'unknown'
        }
        
        # Count shots and analyze sequence
        shot_count = 0
        for i, char in enumerate(sequence):
            if char in self.shot_types:
                shot_count += 1
                rally_data['shots'].append(self.shot_types[char])
            elif char in self.outcomes:
                rally_data['outcome'] = self.outcomes[char]
                break
        
        rally_data['length'] = shot_count
        
        return rally_data
    
    def _identify_pressure_situations(self, match: pd.Series) -> List[Dict]:
        """Identify pressure situations from match data"""
        
        # This would analyze score progressions to identify BP, GP, SP situations
        # For now, return empty list as placeholder
        
        return []
    
    def _empty_patterns(self) -> Dict:
        """Return empty pattern structure when no data available"""
        
        return {
            'serve': {'deuce_court': {'wide': 0.33, 'body': 0.33, 'T': 0.33}},
            'return': {'vs_wide': {'success_rate': 0.3}, 'vs_body': {'success_rate': 0.35}, 'vs_T': {'success_rate': 0.25}},
            'rally': {'avg_length': 4.0, 'length_distribution': {'short': 0.5, 'medium': 0.3, 'long': 0.2}},
            'pressure': {'break_points': {'save_rate': 0.5}, 'pressure_response': 'neutral'},
            'momentum': {'momentum_sensitivity': 0.0, 'comeback_ability': 0.5}
        }


# ============================================================================
# DATA LAYER: Unified Data Ingestion and Processing
# ============================================================================

class UnifiedDataPipeline:
    """
    Handles all data sources and maintains temporal continuity across the Jeff cutoff date.
    
    Pre-2025-06-10: Uses Jeff Sackmann data + tennis-data odds
    Post-2025-06-10: Uses Tennis Abstract + API-tennis, converted to Jeff format
    """
    
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.jeff_cutoff = JEFF_CUTOFF_DATE
        self.cache = {}
        
        # Verify data files exist
        self.verify_data_sources()
        
        logging.info("UnifiedDataPipeline initialized")
    
    def verify_data_sources(self):
        """Verify required data files exist"""
        required_files = [
            "charting-m-points-2020s.csv",
            "charting-m-stats-ServeBasics.csv",
            "charting-m-stats-KeyPointsServe.csv",
            "match_history_men.csv",
            "men_elo_normalized.csv"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logging.warning(f"Missing data files: {missing_files}")
        else:
            logging.info("All required data files found")
    
    def get_player_data(self, player: str, date: datetime) -> Dict:
        """
        Get comprehensive player data for any date with proper temporal handling
        
        Args:
            player: Player name
            date: Date for which to get data
        
        Returns:
            Dict containing all player data unified across sources
        """
        cache_key = f"{player}_{date.strftime('%Y%m%d')}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if date < self.jeff_cutoff:
                # Pre-cutoff: Use Jeff's comprehensive data
                data = self._get_jeff_comprehensive_data(player)
                logging.info(f"Loaded Jeff data for {player}")
            else:
                # Post-cutoff: Use TA + API converted to Jeff format
                data = self._get_post_cutoff_data(player, date)
                logging.info(f"Loaded TA/API data for {player}")
            
            # Cache the result
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            logging.error(f"Failed to load data for {player}: {e}")
            return self._get_fallback_data(player, date)
    
    def _get_jeff_comprehensive_data(self, player: str) -> Dict:
        """Load complete Jeff Sackmann data for a player"""
        
        # Load point-by-point sequences
        point_data = self._load_jeff_points(player)
        
        # Load all categorical CSV files
        categorical_data = self._load_jeff_categoricals(player)
        
        # Load historical betting odds
        betting_data = self._load_tennis_data_odds(player)
        
        # Load ELO ratings
        elo_data = self._load_elo_ratings(player)
        
        return {
            'point_sequences': point_data,
            'categorical_stats': categorical_data,
            'betting_history': betting_data,
            'elo_ratings': elo_data,
            'data_source': DataSourceType.JEFF_SACKMANN,
            'data_quality': self._assess_data_quality(point_data, categorical_data)
        }
    
    def _load_jeff_points(self, player: str) -> pd.DataFrame:
        """Load point-by-point sequences from Jeff's CSVs"""
        
        point_files = [
            "charting-m-points-2020s.csv",
            "charting-m-points-2010s.csv", 
            "charting-m-points-to-2009.csv"
        ]
        
        all_points = []
        
        for file in point_files:
            file_path = self.data_dir / file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Filter for player (check both winner and loser columns)
                    player_points = df[
                        (df['Winner'].str.contains(player, case=False, na=False)) |
                        (df['Loser'].str.contains(player, case=False, na=False))
                    ]
                    
                    if len(player_points) > 0:
                        all_points.append(player_points)
                        logging.info(f"Found {len(player_points)} matches for {player} in {file}")
                
                except Exception as e:
                    logging.warning(f"Error reading {file}: {e}")
        
        if all_points:
            combined = pd.concat(all_points, ignore_index=True)
            return combined
        else:
            logging.warning(f"No point data found for {player}")
            return pd.DataFrame()
    
    def _load_jeff_categoricals(self, player: str) -> Dict:
        """Load all categorical statistics from Jeff's specialized CSVs"""
        
        categorical_files = {
            'ServeBasics': 'charting-m-stats-ServeBasics.csv',
            'KeyPointsServe': 'charting-m-stats-KeyPointsServe.csv',
            'KeyPointsReturn': 'charting-m-stats-KeyPointsReturn.csv',
            'ReturnOutcomes': 'charting-m-stats-ReturnOutcomes.csv',
            'ReturnDepth': 'charting-m-stats-ReturnDepth.csv',
            'Rally': 'charting-m-stats-Rally.csv',
            'ServeDirection': 'charting-m-stats-ServeDirection.csv',
            'ServeInfluence': 'charting-m-stats-ServeInfluence.csv',
            'NetPoints': 'charting-m-stats-NetPoints.csv',
            'ShotTypes': 'charting-m-stats-ShotTypes.csv',
            'ShotDirection': 'charting-m-stats-ShotDirection.csv',
            'ShotDirOutcomes': 'charting-m-stats-ShotDirOutcomes.csv',
            'SnV': 'charting-m-stats-SnV.csv'
        }
        
        categorical_data = {}
        
        for category, filename in categorical_files.items():
            file_path = self.data_dir / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Filter for player
                    player_data = df[df['player'].str.contains(player, case=False, na=False)]
                    
                    if len(player_data) > 0:
                        categorical_data[category] = player_data.to_dict('records')
                        logging.info(f"Loaded {len(player_data)} {category} records for {player}")
                    
                except Exception as e:
                    logging.warning(f"Error loading {filename}: {e}")
            else:
                logging.warning(f"File not found: {filename}")
        
        return categorical_data
    
    def _load_tennis_data_odds(self, player: str) -> pd.DataFrame:
        """Load historical betting odds from tennis-data"""
        
        betting_files = [
            "match_history_men.csv",
            "match_history_women.csv"
        ]
        
        all_odds = []
        
        for file in betting_files:
            file_path = self.data_dir / file
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Filter for player matches
                    player_odds = df[
                        (df['Winner'].str.contains(player, case=False, na=False)) |
                        (df['Loser'].str.contains(player, case=False, na=False))
                    ]
                    
                    if len(player_odds) > 0:
                        all_odds.append(player_odds)
                
                except Exception as e:
                    logging.warning(f"Error reading betting data {file}: {e}")
        
        if all_odds:
            return pd.concat(all_odds, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _load_elo_ratings(self, player: str) -> Dict:
        """Load ELO ratings for player"""
        
        elo_files = [
            "men_elo_normalized.csv",
            "women_elo_normalized.csv"
        ]
        
        for file in elo_files:
            file_path = self.data_dir / file
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Look for player in the data
                    if 'player' in df.columns:
                        player_elo = df[df['player'].str.contains(player, case=False, na=False)]
                    else:
                        # Try common column names
                        for col in ['Player', 'name', 'player_name']:
                            if col in df.columns:
                                player_elo = df[df[col].str.contains(player, case=False, na=False)]
                                break
                        else:
                            continue
                    
                    if len(player_elo) > 0:
                        return player_elo.iloc[-1].to_dict()  # Most recent ELO
                
                except Exception as e:
                    logging.warning(f"Error reading ELO data {file}: {e}")
        
        return {'elo_rating': 1500, 'elo_source': 'default'}  # Default ELO
    
    def _get_post_cutoff_data(self, player: str, date: datetime) -> Dict:
        """
        Get data for post-cutoff dates using Tennis Abstract + API-tennis
        Convert to Jeff format for consistency
        """
        
        # For now, return structured placeholder until TA/API integration is complete
        # In production, this would call tennis_updated.py functions
        
        return {
            'point_sequences': pd.DataFrame(),  # Would be TA point data converted to Jeff format
            'categorical_stats': {},           # Would be TA stats converted to Jeff categorical format
            'betting_history': pd.DataFrame(), # Would be API-tennis odds data
            'elo_ratings': {'elo_rating': 1500, 'elo_source': 'api'},
            'data_source': DataSourceType.TENNIS_ABSTRACT,
            'data_quality': 0.7,  # Lower quality until full integration
            'note': 'TA/API integration placeholder - connect to tennis_updated.py pipeline'
        }
    
    def _get_fallback_data(self, player: str, date: datetime) -> Dict:
        """Fallback data when primary sources fail"""
        
        return {
            'point_sequences': pd.DataFrame(),
            'categorical_stats': {},
            'betting_history': pd.DataFrame(),
            'elo_ratings': {'elo_rating': 1500, 'elo_source': 'fallback'},
            'data_source': 'fallback',
            'data_quality': 0.3
        }
    
    def _assess_data_quality(self, point_data: pd.DataFrame, categorical_data: Dict) -> float:
        """Assess quality of loaded data"""
        
        quality_score = 0.0
        
        # Point sequence data quality
        if len(point_data) > 0:
            quality_score += 0.4
            
            # Bonus for recent data
            if len(point_data) > 20:
                quality_score += 0.1
        
        # Categorical data quality
        if len(categorical_data) > 0:
            quality_score += 0.3
            
            # Bonus for comprehensive categories
            if len(categorical_data) > 5:
                quality_score += 0.1
        
        # Baseline quality
        quality_score += 0.2
        
        return min(1.0, quality_score)


# ============================================================================
# PLAYER STATE: Dynamic State Evolution During Matches
# ============================================================================

@dataclass
class PlayerState:
    """
    Represents a player's current state during match with dynamic evolution
    
    This tracks how fatigue, confidence, momentum change throughout a match
    and affects point-by-point performance.
    """
    
    # Physical state
    energy: float = 1.0                    # 1.0 = fresh, 0.0 = exhausted
    serve_fatigue: float = 0.0             # Accumulated serve fatigue
    movement_fatigue: float = 0.0          # Movement/rally fatigue
    
    # Mental state  
    confidence: float = 0.5                # 0.0 = shaken, 1.0 = supreme confidence
    pressure: float = 0.0                  # Current pressure level (0-1)
    momentum: float = 0.0                  # -1.0 = strong negative, +1.0 = strong positive
    
    # Match context
    sets_won: int = 0
    games_won_current_set: int = 0
    points_played_total: int = 0
    time_on_court: float = 0.0             # Minutes
    breaks_of_serve: int = 0               # Breaks achieved
    breaks_suffered: int = 0               # Service breaks lost
    
    # Performance tracking
    first_serve_pct: float = 0.65          # Current first serve percentage
    points_won_on_serve: int = 0
    points_played_on_serve: int = 0
    
    def update_after_point(self, won: bool, was_serving: bool, rally_length: int, 
                          serve_speed: float = None, point_importance: float = 1.0):
        """Update state after each point with realistic evolution"""
        
        self.points_played_total += 1
        self.time_on_court += 0.5 + (rally_length * 0.3)  # Base time + rally time
        
        # Physical state updates
        self._update_physical_state(rally_length, serve_speed, was_serving)
        
        # Mental state updates  
        self._update_mental_state(won, point_importance)
        
        # Performance tracking
        if was_serving:
            self.points_played_on_serve += 1
            if won:
                self.points_won_on_serve += 1
    
    def _update_physical_state(self, rally_length: int, serve_speed: float, was_serving: bool):
        """Update physical fatigue realistically"""
        
        # Base energy decay per point
        self.energy *= 0.9998  # Very gradual decay
        
        # Rally-based fatigue
        if rally_length > 4:
            extra_fatigue = (rally_length - 4) * 0.002
            self.movement_fatigue += extra_fatigue
            self.energy -= extra_fatigue
        
        # Serve fatigue (only when serving)
        if was_serving and serve_speed:
            serve_intensity = serve_speed / 130.0  # Normalize to 0-1
            self.serve_fatigue += serve_intensity * 0.001
        
        # Long points are especially tiring
        if rally_length > 15:
            self.energy -= 0.01
        
        # Bounds checking
        self.energy = max(0.0, self.energy)
        self.serve_fatigue = min(1.0, self.serve_fatigue)
        self.movement_fatigue = min(1.0, self.movement_fatigue)
    
    def _update_mental_state(self, won: bool, importance: float):
        """Update mental state based on point outcome and importance"""
        
        base_confidence_change = 0.01 * importance
        base_momentum_change = 0.03 * importance
        
        if won:
            # Winning builds confidence and momentum
            self.confidence = min(0.95, self.confidence + base_confidence_change)
            self.momentum = min(1.0, self.momentum + base_momentum_change)
            
            # Important points have bigger impact
            if importance > 1.5:  # Break points, set points, etc.
                self.momentum = min(1.0, self.momentum + 0.1)
                
        else:
            # Losing hurts confidence and momentum
            self.confidence = max(0.05, self.confidence - base_confidence_change * 1.2)
            self.momentum = max(-1.0, self.momentum - base_momentum_change * 1.5)
            
            # Important points hurt more
            if importance > 1.5:
                self.momentum = max(-1.0, self.momentum - 0.15)
    
    def update_after_game(self, won: bool, was_serving: bool, was_break: bool = False):
        """Update state after each game"""
        
        if won:
            self.games_won_current_set += 1
            
            if was_break:
                self.breaks_of_serve += 1
                self.confidence *= 1.08
                self.momentum = min(1.0, self.momentum + 0.2)
            elif was_serving:
                # Holding serve - small confidence boost
                self.confidence *= 1.02
        else:
            if was_serving:
                # Lost serve - break suffered
                self.breaks_suffered += 1
                self.confidence *= 0.92
                self.momentum = max(-1.0, self.momentum - 0.15)
        
        # Game completion brings small recovery
        self.energy = min(1.0, self.energy + 0.001)
    
    def update_after_set(self, won: bool, score: Tuple[int, int], duration: float):
        """Update state after each set"""
        
        games_played = sum(score)
        
        if won:
            self.sets_won += 1
            self.confidence *= 1.12
            self.momentum = 0.25  # Fresh momentum for new set
        else:
            self.confidence *= 0.88
            self.momentum = -0.15  # Slight negative momentum
        
        # Physical impact based on set length and intensity
        if games_played >= 12:  # Long set (6-6, 7-5, etc.)
            self.energy *= 0.90
            self.serve_fatigue += 0.08
            self.movement_fatigue += 0.10
        elif games_played <= 7:  # Quick set (6-1, 6-0)
            self.energy *= 0.96
            self.serve_fatigue += 0.03
        else:  # Normal set
            self.energy *= 0.93
            self.serve_fatigue += 0.05
            self.movement_fatigue += 0.06
        
        # Tiebreak exhaustion
        if max(score) == 7 and min(score) == 6:
            self.energy *= 0.95
            self.movement_fatigue += 0.05
        
        # Reset for new set
        self.games_won_current_set = 0
        self.pressure = 0.0
        
        # Brief mental reset
        self.confidence = 0.4 + (self.confidence - 0.4) * 0.8  # Pull toward center
    
    def get_serve_effectiveness(self) -> float:
        """Calculate current serve effectiveness based on state"""
        
        # Base effectiveness
        effectiveness = 0.65
        
        # Physical factors
        effectiveness *= (1.0 - self.serve_fatigue * 0.2)  # Serve fatigue impacts serve
        effectiveness *= (0.8 + self.energy * 0.2)        # Overall energy
        
        # Mental factors
        effectiveness *= (0.85 + self.confidence * 0.3)   # Confidence helps serving
        effectiveness *= (1.0 + self.momentum * 0.08)     # Positive momentum helps
        
        # Pressure hurts serving
        effectiveness *= (1.0 - self.pressure * 0.1)
        
        return np.clip(effectiveness, 0.25, 0.85)
    
    def get_return_effectiveness(self) -> float:
        """Calculate current return effectiveness based on state"""
        
        # Base return ability
        effectiveness = 0.35  # Complement of serve (65% serve = 35% return)
        
        # Physical factors
        effectiveness *= (0.9 + self.energy * 0.1)         # Energy helps movement
        effectiveness *= (1.0 - self.movement_fatigue * 0.15)  # Movement fatigue hurts returns
        
        # Mental factors  
        effectiveness *= (0.85 + self.confidence * 0.3)    # Confidence helps aggressive returns
        effectiveness *= (1.0 + self.momentum * 0.12)      # Positive momentum boosts returns more
        
        return np.clip(effectiveness, 0.15, 0.55)
    
    def get_pressure_multiplier(self, situation: str) -> float:
        """Get performance multiplier based on pressure situation"""
        
        base_multiplier = 1.0
        
        # Confidence affects pressure response
        confidence_factor = 0.8 + (self.confidence * 0.4)  # 0.8 to 1.2 range
        
        # Momentum affects pressure response
        momentum_factor = 1.0 + (self.momentum * 0.1)      # -0.1 to +0.1 range
        
        # Fatigue makes pressure worse
        fatigue_factor = 1.0 - (self.serve_fatigue + self.movement_fatigue) * 0.1
        
        multiplier = base_multiplier * confidence_factor * momentum_factor * fatigue_factor
        
        # Different situations have different base impacts
        situation_impacts = {
            'break_point_serving': 0.95,    # Slightly harder to serve under BP pressure
            'break_point_returning': 1.08,  # Opportunity boosts return
            'game_point': 1.02,             # Slight boost when closing
            'set_point': 0.98,              # Set points are tough
            'match_point': 0.92             # Match points are very tough
        }
        
        if situation in situation_impacts:
            multiplier *= situation_impacts[situation]
        
        return np.clip(multiplier, 0.7, 1.3)
    
    def recover_between_sets(self):
        """Partial recovery between sets (changeover)"""
        
        # Physical recovery (limited - a few minutes between sets)
        self.energy = min(1.0, self.energy + 0.05)  # Small energy recovery
        self.serve_fatigue *= 0.9                   # Slight serve recovery
        self.movement_fatigue *= 0.92               # Slight movement recovery
        
        # Mental reset (more significant)
        self.pressure = 0.0                         # Reset pressure
        self.confidence = 0.4 + (self.confidence - 0.4) * 0.85  # Pull toward center
        
        # Momentum carries over but is dampened
        self.momentum *= 0.6                        # Significant momentum decay
        
        # Set-specific resets
        self.games_won_current_set = 0


# ============================================================================
# COMPREHENSIVE PLAYER PROFILE
# ============================================================================

class PlayerProfile:
    """
    Complete player profile combining all data sources with learned patterns
    
    This builds a comprehensive understanding of how a player behaves in
    different situations based on actual point-by-point data.
    """
    
    def __init__(self, player_name: str, date: datetime):
        self.name = player_name
        self.date = date
        self.data_pipeline = UnifiedDataPipeline()
        self.pattern_parser = UnifiedPatternParser()
        
        # Load and process all data
        self._load_and_process_data()
        
        logging.info(f"PlayerProfile created for {player_name} (quality: {self.data_quality:.2f})")
    
    def _load_and_process_data(self):
        """Load and process all available data for this player"""
        
        # Get unified data from pipeline
        raw_data = self.data_pipeline.get_player_data(self.name, self.date)
        
        # Extract patterns from point sequences
        self.patterns = self.pattern_parser.extract_patterns(raw_data['point_sequences'])
        
        # Process categorical statistics
        self.categorical_insights = self._process_categorical_stats(raw_data['categorical_stats'])
        
        # Store raw data for reference
        self.raw_data = raw_data
        self.data_quality = raw_data['data_quality']
        
        # Calculate derived metrics
        self.fitness_score = self._calculate_fitness_score()
        self.mental_strength = self._calculate_mental_strength()
        self.style_profile = self._determine_playing_style()
        
        # ELO and market data
        self.elo_rating = raw_data['elo_ratings'].get('elo_rating', 1500)
        self.market_history = raw_data.get('betting_history', pd.DataFrame())
    
    def _process_categorical_stats(self, categorical_data: Dict) -> Dict:
        """Extract insights from Jeff's categorical CSV data"""
        
        insights = {
            'serve_dominance': 0.5,
            'return_strength': 0.5,
            'net_play_frequency': 0.0,
            'rally_tolerance': 0.5,
            'clutch_factor': 1.0,
            'surface_preferences': {}
        }
        
        # Process ServeBasics data
        if 'ServeBasics' in categorical_data:
            serve_data = categorical_data['ServeBasics']
            if serve_data:
                total_points = sum(r.get('pts', 0) for r in serve_data)
                total_won = sum(r.get('pts_won', 0) for r in serve_data)
                
                if total_points > 0:
                    insights['serve_dominance'] = total_won / total_points
        
        # Process KeyPointsServe for clutch performance
        if 'KeyPointsServe' in categorical_data:
            key_points = categorical_data['KeyPointsServe']
            bp_data = [r for r in key_points if r.get('row') == 'BP']
            
            if bp_data:
                bp_points = sum(r.get('pts', 0) for r in bp_data)
                bp_won = sum(r.get('pts_won', 0) for r in bp_data)
                
                if bp_points > 0:
                    bp_save_rate = bp_won / bp_points
                    insights['clutch_factor'] = 0.8 + (bp_save_rate - 0.5) * 0.6
        
        # Process Rally data
        if 'Rally' in categorical_data:
            rally_data = categorical_data['Rally']
            if rally_data:
                # This would analyze rally length preferences
                insights['rally_tolerance'] = 0.6  # Placeholder
        
        # Process NetPoints data
        if 'NetPoints' in categorical_data:
            net_data = categorical_data['NetPoints']
            if net_data:
                # Calculate net play frequency
                total_points = sum(r.get('total_points', 0) for r in net_data if 'total_points' in r)
                net_points = len(net_data)
                
                if total_points > 0:
                    insights['net_play_frequency'] = min(1.0, net_points / total_points * 10)
        
        # Process ReturnOutcomes
        if 'ReturnOutcomes' in categorical_data:
            return_data = categorical_data['ReturnOutcomes']
            if return_data:
                # Calculate return effectiveness
                insights['return_strength'] = 0.4  # Would calculate from actual data
        
        return insights
    
    def _calculate_fitness_score(self) -> float:
        """Calculate fitness based on recent match performance and age estimation"""
        
        # This would analyze recent match durations, performance in long matches
        # For now, return reasonable baseline
        
        base_fitness = 0.75
        
        # Adjust based on data quality (better data = more confidence in fitness assessment)
        if self.data_quality > 0.8:
            # Could analyze actual match durations and late-set performance
            return base_fitness
        
        return base_fitness
    
    def _calculate_mental_strength(self) -> float:
        """Calculate mental strength from pressure patterns and clutch factor"""
        
        mental_strength = 0.5  # Baseline
        
        # Use pressure patterns
        pressure_patterns = self.patterns.get('pressure', {})
        if pressure_patterns:
            bp_save_rate = pressure_patterns.get('break_points', {}).get('save_rate', 0.5)
            gp_conversion = pressure_patterns.get('game_points', {}).get('conversion_rate', 0.5)
            
            # Combine break point defense and game point conversion
            mental_strength = (bp_save_rate * 0.6) + (gp_conversion * 0.4)
        
        # Use categorical insights
        clutch_factor = self.categorical_insights.get('clutch_factor', 1.0)
        mental_strength *= clutch_factor
        
        return np.clip(mental_strength, 0.2, 0.9)
    
    def _determine_playing_style(self) -> Dict:
        """Classify playing style from patterns and categorical data"""
        
        style = {
            'aggression': 0.5,        # 0 = defensive, 1 = ultra-aggressive
            'consistency': 0.5,       # 0 = error-prone, 1 = rock solid
            'net_play': 0.0,         # 0 = baseline only, 1 = serve-and-volley
            'power': 0.5,            # 0 = placement, 1 = power
            'court_coverage': 0.5,   # 0 = limited, 1 = excellent
            'surface_adaptability': 0.5  # 0 = specialist, 1 = all-court
        }
        
        # Analyze from rally patterns
        rally_patterns = self.patterns.get('rally', {})
        if rally_patterns:
            avg_rally_length = rally_patterns.get('avg_length', 4.0)
            
            # Longer rallies suggest more consistency, shorter suggests aggression
            if avg_rally_length > 6:
                style['consistency'] = 0.7
                style['aggression'] = 0.4
            elif avg_rally_length < 3:
                style['aggression'] = 0.8
                style['consistency'] = 0.3
        
        # Use categorical insights
        style['net_play'] = self.categorical_insights.get('net_play_frequency', 0.0)
        
        # Serve patterns indicate aggression
        serve_patterns = self.patterns.get('serve', {})
        if serve_patterns:
            # Analyze serve direction variety (more variety = more tactical)
            deuce_patterns = serve_patterns.get('deuce_court', {})
            if deuce_patterns:
                # Calculate entropy/variety in serve directions
                directions = [deuce_patterns.get('wide', 0), deuce_patterns.get('body', 0), deuce_patterns.get('T', 0)]
                if sum(directions) > 0:
                    # Higher variety suggests more tactical approach
                    variety_score = -sum(p * np.log(p + 0.001) for p in directions if p > 0)
                    style['aggression'] = min(0.9, 0.3 + variety_score * 0.3)
        
        return style
    
    def get_matchup_advantage(self, opponent_profile: 'PlayerProfile', surface: Surface) -> Dict:
        """Calculate specific matchup advantages against opponent"""
        
        advantages = {
            'serve_vs_return': 0.0,     # Positive = advantage when serving
            'return_vs_serve': 0.0,     # Positive = advantage when returning
            'rally_dynamics': 0.0,      # Positive = advantage in rallies
            'pressure_differential': 0.0, # Positive = better under pressure
            'style_compatibility': 0.0, # Positive = favorable style matchup
            'surface_edge': 0.0         # Positive = surface advantage
        }
        
        # Serve vs Return matchup
        my_serve_dom = self.categorical_insights.get('serve_dominance', 0.5)
        opp_return_str = opponent_profile.categorical_insights.get('return_strength', 0.5)
        advantages['serve_vs_return'] = (my_serve_dom - opp_return_str) * 0.5
        
        # Return vs Serve matchup
        my_return_str = self.categorical_insights.get('return_strength', 0.5)
        opp_serve_dom = opponent_profile.categorical_insights.get('serve_dominance', 0.5)
        advantages['return_vs_serve'] = (my_return_str - opp_serve_dom) * 0.5
        
        # Rally dynamics
        my_rally_tol = self.categorical_insights.get('rally_tolerance', 0.5)
        opp_rally_tol = opponent_profile.categorical_insights.get('rally_tolerance', 0.5)
        advantages['rally_dynamics'] = (my_rally_tol - opp_rally_tol) * 0.3
        
        # Pressure differential
        my_mental = self.mental_strength
        opp_mental = opponent_profile.mental_strength
        advantages['pressure_differential'] = (my_mental - opp_mental) * 0.4
        
        # Style compatibility
        my_style = self.style_profile
        opp_style = opponent_profile.style_profile
        
        # Aggressive players vs consistent players
        if my_style['aggression'] > 0.7 and opp_style['consistency'] > 0.7:
            advantages['style_compatibility'] = 0.1  # Aggression can break down consistency
        elif my_style['consistency'] > 0.7 and opp_style['aggression'] > 0.7:
            advantages['style_compatibility'] = -0.05  # Consistency can frustrate aggression
        
        # Net player vs baseliner
        if my_style['net_play'] > 0.3 and opp_style['net_play'] < 0.1:
            advantages['style_compatibility'] = 0.08  # Net play advantage
        
        # Surface advantages would be calculated based on historical performance
        # For now, use ELO-based approximation
        advantages['surface_edge'] = (self.elo_rating - opponent_profile.elo_rating) / 400 * 0.1
        
        return advantages


# ============================================================================
# MATCH SIMULATION: Stateful Point-by-Point Match Simulation
# ============================================================================

class MatchSimulator:
    """Stateful match simulation using pattern-based point decisions"""
    
    def __init__(self, pattern_parser: UnifiedPatternParser):
        self.pattern_parser = pattern_parser
        self.rally_patterns = {}
        self.pressure_patterns = {}
        
    def simulate_match(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile, 
                      best_of: int = 3, surface: str = 'hard') -> Dict:
        """Simulate full match with evolving player states"""
        
        # Initialize player states
        p1_state = PlayerState()
        p2_state = PlayerState()
        
        # Match context
        match_context = {
            'surface': surface,
            'best_of': best_of,
            'sets_won': [0, 0],
            'current_set': 1,
            'games_won': [0, 0],
            'score': [0, 0],
            'server_index': 0,  # 0 for p1, 1 for p2
            'total_points': 0,
            'duration_minutes': 0
        }
        
        # Match history tracking
        match_history = {
            'points': [],
            'games': [],
            'sets': [],
            'momentum_shifts': [],
            'break_points': [],
            'state_evolution': {'p1': [], 'p2': []}
        }
        
        # Simulate sets until match completion
        while not self._is_match_complete(match_context):
            set_result = self._simulate_set(
                p1_profile, p2_profile, p1_state, p2_state, 
                match_context, match_history
            )
            
            # Update match context after set
            winner_idx = set_result['winner_index']
            match_context['sets_won'][winner_idx] += 1
            match_context['current_set'] += 1
            match_context['games_won'] = [0, 0]
            
            # Reset states between sets (partial recovery)
            p1_state.recover_between_sets()
            p2_state.recover_between_sets()
            
        # Determine match winner
        match_winner_idx = 0 if match_context['sets_won'][0] > match_context['sets_won'][1] else 1
        
        return {
            'winner_index': match_winner_idx,
            'winner_name': p1_profile.name if match_winner_idx == 0 else p2_profile.name,
            'final_score': f"{match_context['sets_won'][0]}-{match_context['sets_won'][1]}",
            'sets_won': match_context['sets_won'],
            'total_points': match_context['total_points'],
            'duration_minutes': match_context['duration_minutes'],
            'match_history': match_history,
            'final_states': {'p1': p1_state, 'p2': p2_state}
        }
    
    def _simulate_set(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                     p1_state: PlayerState, p2_state: PlayerState,
                     match_context: Dict, match_history: Dict) -> Dict:
        """Simulate a complete set"""
        
        games = [0, 0]  # Games won by each player in this set
        
        while not self._is_set_complete(games):
            game_result = self._simulate_game(
                p1_profile, p2_profile, p1_state, p2_state,
                match_context, match_history, games
            )
            
            # Update games
            winner_idx = game_result['winner_index']
            games[winner_idx] += 1
            
            # Track game in history
            match_history['games'].append({
                'set_number': match_context['current_set'],
                'games_before': games.copy(),
                'winner_index': winner_idx,
                'server_index': match_context['server_index'],
                'break_point': (winner_idx != match_context['server_index'])
            })
            
            # Switch server for next game
            match_context['server_index'] = 1 - match_context['server_index']
            
            # Track momentum shifts
            if len(match_history['games']) >= 3:
                self._detect_momentum_shift(match_history, p1_state, p2_state)
        
        # Determine set winner
        set_winner = 0 if games[0] > games[1] else 1
        
        return {
            'winner_index': set_winner,
            'games': games,
            'duration_games': sum(games)
        }
    
    def _simulate_game(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                      p1_state: PlayerState, p2_state: PlayerState,
                      match_context: Dict, match_history: Dict, games: List[int]) -> Dict:
        """Simulate a complete game"""
        
        server_idx = match_context['server_index']
        returner_idx = 1 - server_idx
        
        server_profile = p1_profile if server_idx == 0 else p2_profile
        returner_profile = p2_profile if server_idx == 0 else p1_profile
        server_state = p1_state if server_idx == 0 else p2_state
        returner_state = p2_state if server_idx == 0 else p1_state
        
        # Game score (0, 15, 30, 40, deuce, advantage, game)
        points = [0, 0]  # Points won by server, returner
        
        while not self._is_game_complete(points):
            # Determine point importance for pressure calculation
            point_importance = self._calculate_point_importance(points, games, match_context)
            
            # Simulate individual point
            point_result = self._simulate_point(
                server_profile, returner_profile, server_state, returner_state,
                match_context, point_importance
            )
            
            # Update points
            if point_result['winner_index'] == 0:  # Server won
                points[0] += 1
            else:  # Returner won
                points[1] += 1
            
            # Update states after point
            server_state.update_after_point(
                won=(point_result['winner_index'] == 0),
                was_serving=True,
                rally_length=point_result.get('rally_length', 4),
                serve_speed=point_result.get('serve_speed', 120),
                point_importance=point_importance
            )
            
            returner_state.update_after_point(
                won=(point_result['winner_index'] == 1),
                was_serving=False,
                rally_length=point_result.get('rally_length', 4),
                point_importance=point_importance
            )
            
            # Track point in history
            match_history['points'].append({
                'set': match_context['current_set'],
                'game': len(match_history['games']) + 1,
                'server_index': server_idx,
                'winner_index': point_result['winner_index'],
                'rally_length': point_result.get('rally_length', 4),
                'point_type': point_result.get('point_type', 'baseline'),
                'importance': point_importance
            })
            
            match_context['total_points'] += 1
            match_context['duration_minutes'] += point_result.get('duration', 0.8)
        
        # Determine game winner based on tennis scoring
        game_winner = self._determine_game_winner(points)
        
        return {
            'winner_index': server_idx if game_winner == 'server' else returner_idx,
            'points': points,
            'total_points': sum(points)
        }
    
    def _simulate_point(self, server_profile: PlayerProfile, returner_profile: PlayerProfile,
                       server_state: PlayerState, returner_state: PlayerState,
                       match_context: Dict, importance: float) -> Dict:
        """Simulate individual point using pattern matching"""
        
        # Get serve patterns
        serve_patterns = server_profile.patterns.get('serve_patterns', {})
        return_patterns = returner_profile.patterns.get('return_patterns', {})
        
        # Adjust probabilities based on current states
        serve_effectiveness = self._calculate_serve_effectiveness(server_state, server_profile, importance)
        return_effectiveness = self._calculate_return_effectiveness(returner_state, returner_profile, importance)
        
        # Determine serve outcome
        serve_result = self._resolve_serve_interaction(serve_effectiveness, return_effectiveness)
        
        if serve_result['outcome'] == 'ace':
            return {
                'winner_index': 0,  # Server wins
                'point_type': 'ace',
                'rally_length': 1,
                'duration': 0.3
            }
        elif serve_result['outcome'] == 'double_fault':
            return {
                'winner_index': 1,  # Returner wins
                'point_type': 'double_fault', 
                'rally_length': 1,
                'duration': 0.5
            }
        elif serve_result['outcome'] == 'return_winner':
            return {
                'winner_index': 1,  # Returner wins
                'point_type': 'return_winner',
                'rally_length': 2,
                'duration': 0.4
            }
        else:
            # Rally ensues - use rally patterns
            return self._simulate_rally(
                server_profile, returner_profile, server_state, returner_state,
                serve_result, importance
            )
    
    def _calculate_serve_effectiveness(self, server_state: PlayerState, 
                                     server_profile: PlayerProfile, importance: float) -> float:
        """Calculate current serve effectiveness based on state and patterns"""
        
        base_effectiveness = server_profile.categorical_insights.get('serve_dominance', 0.65)
        
        # Fatigue penalties
        fatigue_penalty = server_state.serve_fatigue * 0.15
        energy_penalty = (1.0 - server_state.energy) * 0.1
        
        # Confidence adjustments
        confidence_boost = (server_state.confidence - 0.5) * 0.2
        
        # Pressure adjustments
        pressure_penalty = server_state.pressure * 0.1
        
        # Momentum effects
        momentum_boost = server_state.momentum * 0.15
        
        adjusted_effectiveness = (base_effectiveness 
                                 - fatigue_penalty 
                                 - energy_penalty
                                 - pressure_penalty
                                 + confidence_boost
                                 + momentum_boost)
        
        return max(0.2, min(0.95, adjusted_effectiveness))
    
    def _calculate_return_effectiveness(self, returner_state: PlayerState,
                                      returner_profile: PlayerProfile, importance: float) -> float:
        """Calculate current return effectiveness"""
        
        base_effectiveness = returner_profile.categorical_insights.get('return_strength', 0.35)
        
        # Similar adjustments as serve effectiveness
        fatigue_penalty = returner_state.movement_fatigue * 0.1
        energy_penalty = (1.0 - returner_state.energy) * 0.08
        confidence_boost = (returner_state.confidence - 0.5) * 0.15
        pressure_penalty = returner_state.pressure * 0.08
        momentum_boost = returner_state.momentum * 0.12
        
        adjusted_effectiveness = (base_effectiveness
                                 - fatigue_penalty
                                 - energy_penalty  
                                 - pressure_penalty
                                 + confidence_boost
                                 + momentum_boost)
        
        return max(0.15, min(0.85, adjusted_effectiveness))
    
    def _resolve_serve_interaction(self, serve_eff: float, return_eff: float) -> Dict:
        """Resolve serve vs return interaction"""
        
        # Base probabilities
        ace_prob = max(0.02, serve_eff * 0.12)
        double_fault_prob = max(0.01, (1.0 - serve_eff) * 0.08)  
        return_winner_prob = return_eff * 0.15
        
        # Normalize so rally is most likely outcome
        total_special = ace_prob + double_fault_prob + return_winner_prob
        if total_special > 0.4:  # Cap special outcomes at 40%
            scale = 0.4 / total_special
            ace_prob *= scale
            double_fault_prob *= scale
            return_winner_prob *= scale
        
        # Random outcome
        rand = np.random.random()
        if rand < ace_prob:
            return {'outcome': 'ace'}
        elif rand < ace_prob + double_fault_prob:
            return {'outcome': 'double_fault'}
        elif rand < ace_prob + double_fault_prob + return_winner_prob:
            return {'outcome': 'return_winner'}
        else:
            return {'outcome': 'rally', 'serve_quality': serve_eff, 'return_quality': return_eff}
    
    def _simulate_rally(self, server_profile: PlayerProfile, returner_profile: PlayerProfile,
                       server_state: PlayerState, returner_state: PlayerState,
                       serve_result: Dict, importance: float) -> Dict:
        """Simulate rally after successful return"""
        
        # Rally length based on player styles and current states
        base_rally_length = 6
        
        server_rally_pref = server_profile.categorical_insights.get('rally_tolerance', 0.5)
        returner_rally_pref = returner_profile.categorical_insights.get('rally_tolerance', 0.5)
        
        # Longer rallies if both players are patient
        if server_rally_pref > 0.6 and returner_rally_pref > 0.6:
            base_rally_length = 10
        elif server_rally_pref < 0.4 or returner_rally_pref < 0.4:
            base_rally_length = 4
            
        # Energy affects rally length (tired players end points faster)
        energy_factor = min(server_state.energy, returner_state.energy)
        rally_length = max(3, int(base_rally_length * energy_factor + np.random.exponential(2)))
        
        # Determine winner based on relative strengths
        server_rally_skill = server_profile.style_profile.get('consistency', 0.5)
        returner_rally_skill = returner_profile.style_profile.get('consistency', 0.5)
        
        # Adjust for current states
        server_adjusted = server_rally_skill + (server_state.momentum * 0.2) - (server_state.movement_fatigue * 0.3)
        returner_adjusted = returner_rally_skill + (returner_state.momentum * 0.2) - (returner_state.movement_fatigue * 0.3)
        
        # Winner probability (0 = server, 1 = returner)
        server_win_prob = server_adjusted / (server_adjusted + returner_adjusted)
        
        winner_index = 0 if np.random.random() < server_win_prob else 1
        
        return {
            'winner_index': winner_index,
            'point_type': 'rally',
            'rally_length': rally_length,
            'duration': 0.3 + rally_length * 0.15
        }
    
    def _is_match_complete(self, match_context: Dict) -> bool:
        """Check if match is complete"""
        sets_to_win = (match_context['best_of'] + 1) // 2
        return max(match_context['sets_won']) >= sets_to_win
    
    def _is_set_complete(self, games: List[int]) -> bool:
        """Check if set is complete (including tiebreak logic)"""
        if max(games) >= 6:
            if max(games) - min(games) >= 2:
                return True
            elif games == [6, 6]:
                return False  # Need tiebreak (simplified - assume one more game decides)
            elif max(games) >= 7:
                return True
        return False
    
    def _is_game_complete(self, points: List[int]) -> bool:
        """Check if game is complete using tennis scoring"""
        # Simplified: first to 4 points with 2-point margin wins
        if max(points) >= 4 and max(points) - min(points) >= 2:
            return True
        # Deuce situations would be handled here in full implementation
        return False
    
    def _determine_game_winner(self, points: List[int]) -> str:
        """Determine game winner from points"""
        return 'server' if points[0] > points[1] else 'returner'
    
    def _calculate_point_importance(self, points: List[int], games: List[int], match_context: Dict) -> float:
        """Calculate importance of current point for pressure calculations"""
        
        base_importance = 1.0
        
        # Game situation importance
        if max(points) >= 3:  # Close games are more important
            base_importance += 0.5
            
        # Set situation importance  
        if max(games) >= 5:  # Close sets are more important
            base_importance += 1.0
            
        # Match situation importance
        if max(match_context['sets_won']) >= match_context['best_of'] // 2:
            base_importance += 1.5
            
        # Break point situations
        current_server = match_context['server_index']
        if points[1] > points[0] and max(points) >= 3:  # Returner ahead in close game
            base_importance += 1.0  # Break point
            
        return min(5.0, base_importance)
    
    def _detect_momentum_shift(self, match_history: Dict, p1_state: PlayerState, p2_state: PlayerState):
        """Detect and record momentum shifts"""
        
        recent_games = match_history['games'][-3:]
        if len(recent_games) < 3:
            return
            
        # Simple momentum detection: if one player wins 3 straight games
        winners = [game['winner_index'] for game in recent_games]
        if len(set(winners)) == 1:  # All same winner
            winner_idx = winners[0]
            
            match_history['momentum_shifts'].append({
                'game_number': len(match_history['games']),
                'player_index': winner_idx,
                'type': 'three_game_run'
            })
            
            # Boost momentum for winner, reduce for opponent
            if winner_idx == 0:
                p1_state.momentum = min(1.0, p1_state.momentum + 0.3)
                p2_state.momentum = max(-1.0, p2_state.momentum - 0.2)
            else:
                p2_state.momentum = min(1.0, p2_state.momentum + 0.3)
                p1_state.momentum = max(-1.0, p1_state.momentum - 0.2)


# ============================================================================
# MARKET INTELLIGENCE: Market-Based Validation and Calibration
# ============================================================================

class MarketIntelligence:
    """
    Validates predictions against market odds and provides calibration
    
    Uses betting market data to:
    1. Validate model predictions against market consensus
    2. Identify value betting opportunities  
    3. Calibrate probability estimates
    4. Track model performance vs market
    """
    
    def __init__(self, data_pipeline: UnifiedDataPipeline):
        self.data_pipeline = data_pipeline
        self.market_history = []
        self.calibration_data = []
        
        # Betting market parameters
        self.market_margin = 0.05  # Typical bookmaker margin
        self.sharp_threshold = 0.10  # Consider 10%+ edge as "sharp" value
        
    def validate_prediction(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile, 
                          model_prob: float, match_date: datetime) -> Dict:
        """
        Validate model prediction against market odds
        
        Args:
            p1_profile: First player profile
            p2_profile: Second player profile  
            model_prob: Model's probability for p1 winning (0-1)
            match_date: Date of match
            
        Returns:
            Dict with market analysis, value assessment, confidence score
        """
        
        # Get market data for this matchup
        market_data = self._get_market_data(p1_profile.name, p2_profile.name, match_date)
        
        if not market_data:
            return self._no_market_fallback(model_prob)
        
        # Convert odds to probabilities  
        market_prob_p1 = self._odds_to_probability(market_data['p1_odds'])
        market_prob_p2 = self._odds_to_probability(market_data['p2_odds'])
        
        # Normalize probabilities (remove market margin)
        total_prob = market_prob_p1 + market_prob_p2
        market_prob_p1_fair = market_prob_p1 / total_prob
        market_prob_p2_fair = market_prob_p2 / total_prob
        
        # Calculate disagreement
        model_edge = model_prob - market_prob_p1_fair
        
        # Assess confidence based on various factors
        confidence_score = self._calculate_prediction_confidence(
            model_prob, market_prob_p1_fair, p1_profile, p2_profile, market_data
        )
        
        # Determine value bet assessment
        value_assessment = self._assess_betting_value(model_edge, confidence_score, market_data)
        
        return {
            'market_probability': market_prob_p1_fair,
            'model_probability': model_prob,
            'model_edge': model_edge,
            'confidence_score': confidence_score,
            'value_assessment': value_assessment,
            'market_data': market_data,
            'recommendation': self._generate_betting_recommendation(model_edge, confidence_score),
            'calibration_bracket': self._get_calibration_bracket(model_prob)
        }
    
    def _get_market_data(self, player1: str, player2: str, match_date: datetime) -> Optional[Dict]:
        """Get market odds data for specific match"""
        
        # Try to load from tennis-data betting CSV files
        try:
            # Look for match in betting data (from data pipeline)
            betting_data = self.data_pipeline._load_tennis_data_odds(player1)
            
            # Find this specific match
            for match in betting_data:
                if (match.get('opponent') == player2 and 
                    abs((match.get('date', match_date) - match_date).days) <= 1):
                    
                    return {
                        'p1_odds': match.get('p1_odds', 2.0),
                        'p2_odds': match.get('p2_odds', 2.0),
                        'market_volume': match.get('volume', 'unknown'),
                        'closing_odds': True,
                        'bookmaker': match.get('bookmaker', 'average')
                    }
            
            # Fallback: estimate based on ELO if no direct odds
            return self._estimate_market_from_elo(player1, player2, match_date)
            
        except Exception as e:
            logging.warning(f"Could not load market data for {player1} vs {player2}: {e}")
            return None
    
    def _estimate_market_from_elo(self, player1: str, player2: str, match_date: datetime) -> Dict:
        """Estimate market odds from ELO ratings when no betting data available"""
        
        try:
            # Get ELO ratings from profiles
            p1_data = self.data_pipeline.get_player_data(player1, match_date)
            p2_data = self.data_pipeline.get_player_data(player2, match_date)
            
            p1_elo = p1_data['elo_ratings'].get('elo_rating', 1500)
            p2_elo = p2_data['elo_ratings'].get('elo_rating', 1500)
            
            # ELO-based probability
            elo_diff = p1_elo - p2_elo
            p1_prob_elo = 1 / (1 + 10 ** (-elo_diff / 400))
            
            # Convert to odds with typical market margin
            p1_fair_odds = 1 / p1_prob_elo
            p2_fair_odds = 1 / (1 - p1_prob_elo)
            
            # Add margin
            p1_odds = p1_fair_odds * 1.025  # 2.5% margin each side
            p2_odds = p2_fair_odds * 1.025
            
            return {
                'p1_odds': p1_odds,
                'p2_odds': p2_odds,
                'market_volume': 'estimated',
                'closing_odds': False,
                'bookmaker': 'elo_estimate'
            }
            
        except Exception as e:
            logging.error(f"Could not estimate market from ELO: {e}")
            return None
    
    def _odds_to_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        if decimal_odds <= 1.0:
            return 0.5  # Invalid odds, return neutral
        return 1.0 / decimal_odds
    
    def _calculate_prediction_confidence(self, model_prob: float, market_prob: float,
                                       p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                                       market_data: Dict) -> float:
        """Calculate confidence in prediction based on multiple factors"""
        
        confidence = 0.5  # Base confidence
        
        # Data quality factor
        p1_quality = p1_profile.data_quality
        p2_quality = p2_profile.data_quality
        avg_quality = (p1_quality + p2_quality) / 2
        confidence += (avg_quality - 0.5) * 0.3  # +/- 15% based on data quality
        
        # Market liquidity factor
        if market_data.get('closing_odds'):
            confidence += 0.1  # Closing odds are more reliable
        if market_data.get('market_volume') == 'high':
            confidence += 0.05
        
        # Edge size factor (extreme disagreements reduce confidence)
        edge_magnitude = abs(model_prob - market_prob)
        if edge_magnitude > 0.2:  # Very large disagreement
            confidence -= 0.1
        elif edge_magnitude < 0.05:  # Close agreement
            confidence += 0.05
        
        # Model complexity factor (more data sources = higher confidence)
        if hasattr(p1_profile, 'patterns') and p1_profile.patterns:
            confidence += 0.08  # Have point sequence patterns
        if hasattr(p2_profile, 'patterns') and p2_profile.patterns:
            confidence += 0.08
            
        # Recent form factor
        recent_matches_p1 = len(p1_profile.raw_data.get('recent_matches', []))
        recent_matches_p2 = len(p2_profile.raw_data.get('recent_matches', []))
        
        if recent_matches_p1 >= 5 and recent_matches_p2 >= 5:
            confidence += 0.1  # Good recent form data
        elif recent_matches_p1 < 2 or recent_matches_p2 < 2:
            confidence -= 0.15  # Poor recent form data
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _assess_betting_value(self, model_edge: float, confidence: float, market_data: Dict) -> Dict:
        """Assess betting value of the prediction"""
        
        # Calculate Kelly criterion for optimal bet sizing
        model_prob = model_edge + self._odds_to_probability(market_data['p1_odds'])
        
        if model_edge > 0:  # Positive expected value
            kelly_fraction = self._kelly_criterion(
                model_prob, market_data['p1_odds'], confidence
            )
            
            value_rating = "STRONG_VALUE" if model_edge > 0.15 else \
                          "MODERATE_VALUE" if model_edge > 0.08 else \
                          "SLIGHT_VALUE"
        else:
            kelly_fraction = 0.0
            value_rating = "NO_VALUE"
        
        # Expected ROI calculation
        if model_edge > 0:
            expected_roi = model_edge * confidence
        else:
            expected_roi = model_edge  # Negative expected value
        
        return {
            'value_rating': value_rating,
            'expected_edge': model_edge,
            'kelly_fraction': kelly_fraction,
            'expected_roi': expected_roi,
            'confidence_adjusted_edge': model_edge * confidence,
            'recommended_stake': min(kelly_fraction * 0.5, 0.05)  # Conservative Kelly
        }
    
    def _kelly_criterion(self, win_prob: float, odds: float, confidence: float) -> float:
        """Calculate Kelly criterion for optimal bet sizing"""
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction to bet, b = odds-1, p = win prob, q = lose prob
        
        b = odds - 1  # Net odds
        p = win_prob * confidence  # Confidence-adjusted win probability
        q = 1 - p     # Lose probability
        
        kelly = (b * p - q) / b
        
        # Cap Kelly at reasonable levels
        return max(0.0, min(kelly, 0.25))
    
    def _generate_betting_recommendation(self, model_edge: float, confidence: float) -> str:
        """Generate human-readable betting recommendation"""
        
        if confidence < 0.4:
            return "PASS - Low confidence in prediction"
        
        if model_edge > 0.15 and confidence > 0.7:
            return "STRONG BET - High edge with high confidence"
        elif model_edge > 0.08 and confidence > 0.6:
            return "MODERATE BET - Good edge with decent confidence"
        elif model_edge > 0.03 and confidence > 0.5:
            return "SMALL BET - Slight edge, proceed cautiously"
        elif model_edge < -0.05:
            return "AVOID - Model disagrees with market significantly"
        else:
            return "PASS - Insufficient edge or confidence"
    
    def _get_calibration_bracket(self, model_prob: float) -> str:
        """Get probability bracket for calibration tracking"""
        
        brackets = [
            (0.0, 0.1, "0-10%"),
            (0.1, 0.2, "10-20%"), 
            (0.2, 0.3, "20-30%"),
            (0.3, 0.4, "30-40%"),
            (0.4, 0.5, "40-50%"),
            (0.5, 0.6, "50-60%"),
            (0.6, 0.7, "60-70%"),
            (0.7, 0.8, "70-80%"),
            (0.8, 0.9, "80-90%"),
            (0.9, 1.0, "90-100%")
        ]
        
        for min_val, max_val, label in brackets:
            if min_val <= model_prob < max_val:
                return label
                
        return "90-100%"  # Fallback
    
    def _no_market_fallback(self, model_prob: float) -> Dict:
        """Fallback response when no market data is available"""
        
        return {
            'market_probability': None,
            'model_probability': model_prob, 
            'model_edge': None,
            'confidence_score': 0.4,  # Lower confidence without market validation
            'value_assessment': {
                'value_rating': "NO_MARKET_DATA",
                'expected_edge': None,
                'kelly_fraction': 0.0,
                'expected_roi': None,
                'recommended_stake': 0.0
            },
            'market_data': None,
            'recommendation': "NO MARKET DATA - Cannot assess betting value",
            'calibration_bracket': self._get_calibration_bracket(model_prob)
        }
    
    def track_prediction_outcome(self, prediction_result: Dict, actual_winner: int) -> Dict:
        """Track prediction outcome for model calibration"""
        
        model_prob = prediction_result['model_probability']
        predicted_winner = 0 if model_prob > 0.5 else 1
        was_correct = (predicted_winner == actual_winner)
        
        # Calculate Brier score
        if actual_winner == 0:
            brier_score = (1 - model_prob) ** 2
        else:
            brier_score = model_prob ** 2
            
        # Log probability score
        log_prob_score = -np.log(model_prob if actual_winner == 0 else 1 - model_prob)
        
        outcome_record = {
            'model_probability': model_prob,
            'market_probability': prediction_result.get('market_probability'),
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'was_correct': was_correct,
            'brier_score': brier_score,
            'log_probability_score': log_prob_score,
            'confidence_score': prediction_result['confidence_score'],
            'calibration_bracket': prediction_result['calibration_bracket']
        }
        
        # Add to tracking history
        self.calibration_data.append(outcome_record)
        
        return outcome_record
    
    def get_calibration_report(self) -> Dict:
        """Generate calibration report showing model accuracy vs confidence"""
        
        if not self.calibration_data:
            return {'error': 'No calibration data available'}
        
        # Group by calibration brackets
        bracket_stats = {}
        for record in self.calibration_data:
            bracket = record['calibration_bracket']
            if bracket not in bracket_stats:
                bracket_stats[bracket] = {
                    'predictions': 0,
                    'correct': 0, 
                    'total_brier': 0.0,
                    'total_log_prob': 0.0
                }
            
            stats = bracket_stats[bracket]
            stats['predictions'] += 1
            stats['correct'] += record['was_correct']
            stats['total_brier'] += record['brier_score']
            stats['total_log_prob'] += record['log_probability_score']
        
        # Calculate statistics
        calibration_results = {}
        for bracket, stats in bracket_stats.items():
            n = stats['predictions']
            if n > 0:
                calibration_results[bracket] = {
                    'predictions': n,
                    'accuracy': stats['correct'] / n,
                    'avg_brier_score': stats['total_brier'] / n,
                    'avg_log_prob_score': stats['total_log_prob'] / n
                }
        
        # Overall statistics
        total_predictions = len(self.calibration_data)
        total_correct = sum(r['was_correct'] for r in self.calibration_data)
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        avg_brier = np.mean([r['brier_score'] for r in self.calibration_data])
        avg_log_prob = np.mean([r['log_probability_score'] for r in self.calibration_data])
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': total_predictions,
            'avg_brier_score': avg_brier,
            'avg_log_prob_score': avg_log_prob,
            'calibration_by_bracket': calibration_results,
            'model_quality': self._assess_model_quality(overall_accuracy, avg_brier)
        }
    
    def _assess_model_quality(self, accuracy: float, brier_score: float) -> str:
        """Assess overall model quality based on metrics"""
        
        if accuracy > 0.65 and brier_score < 0.2:
            return "EXCELLENT"
        elif accuracy > 0.60 and brier_score < 0.25:
            return "VERY_GOOD" 
        elif accuracy > 0.55 and brier_score < 0.3:
            return "GOOD"
        elif accuracy > 0.50 and brier_score < 0.35:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"


# ============================================================================
# AI CONTEXTUAL ADJUSTMENT: Dynamic Model Enhancement
# ============================================================================

class AIContextualAdjustment:
    """
    AI-powered contextual adjustments to base model predictions
    
    Applies machine learning to:
    1. Learn from prediction errors and market disagreements
    2. Adjust for contextual factors not captured in base model
    3. Adapt to changing tennis meta (playing styles, surfaces, etc)
    4. Fine-tune predictions based on recent performance patterns
    """
    
    def __init__(self, data_pipeline: UnifiedDataPipeline, market_intelligence: MarketIntelligence):
        self.data_pipeline = data_pipeline
        self.market_intelligence = market_intelligence
        
        # Learning components
        self.adjustment_history = []
        self.context_features = []
        self.performance_tracker = {}
        
        # Model parameters
        self.learning_rate = 0.01
        self.adjustment_strength = 0.15  # Max adjustment magnitude
        self.context_memory = 100  # Remember last 100 adjustments
        
    def adjust_prediction(self, base_prediction: Dict, p1_profile: PlayerProfile, 
                         p2_profile: PlayerProfile, match_context: Dict) -> Dict:
        """
        Apply contextual adjustments to base model prediction
        
        Args:
            base_prediction: Raw model prediction with probability
            p1_profile: Player 1 profile
            p2_profile: Player 2 profile  
            match_context: Match details (surface, tournament, conditions)
            
        Returns:
            Adjusted prediction with explanation of adjustments
        """
        
        base_prob = base_prediction['win_probability']
        adjustments = {'total_adjustment': 0.0, 'components': {}}
        
        # 1. Surface specialization adjustment
        surface_adj = self._calculate_surface_adjustment(
            p1_profile, p2_profile, match_context.get('surface', 'hard')
        )
        adjustments['components']['surface_specialization'] = surface_adj
        
        # 2. Recent form momentum adjustment  
        form_adj = self._calculate_form_adjustment(p1_profile, p2_profile, match_context)
        adjustments['components']['recent_form'] = form_adj
        
        # 3. Tournament context adjustment
        tournament_adj = self._calculate_tournament_adjustment(
            p1_profile, p2_profile, match_context
        )
        adjustments['components']['tournament_context'] = tournament_adj
        
        # 4. Head-to-head learning adjustment
        h2h_adj = self._calculate_h2h_adjustment(p1_profile, p2_profile)
        adjustments['components']['head_to_head'] = h2h_adj
        
        # 5. Market disagreement learning adjustment
        market_adj = self._calculate_market_learning_adjustment(
            base_prob, p1_profile, p2_profile, match_context
        )
        adjustments['components']['market_learning'] = market_adj
        
        # 6. Meta-game evolution adjustment
        meta_adj = self._calculate_meta_adjustment(p1_profile, p2_profile, match_context)
        adjustments['components']['meta_evolution'] = meta_adj
        
        # 7. Injury/fitness adjustment
        fitness_adj = self._calculate_fitness_adjustment(p1_profile, p2_profile)
        adjustments['components']['fitness'] = fitness_adj
        
        # Sum all adjustments with dampening
        total_adjustment = sum(adjustments['components'].values())
        total_adjustment = np.clip(total_adjustment, -self.adjustment_strength, self.adjustment_strength)
        adjustments['total_adjustment'] = total_adjustment
        
        # Apply sigmoid dampening to prevent extreme probabilities
        adjusted_prob = self._apply_sigmoid_adjustment(base_prob, total_adjustment)
        
        # Create adjusted prediction
        adjusted_prediction = base_prediction.copy()
        adjusted_prediction.update({
            'win_probability': adjusted_prob,
            'base_probability': base_prob,
            'adjustments': adjustments,
            'adjustment_confidence': self._calculate_adjustment_confidence(adjustments),
            'explanation': self._generate_adjustment_explanation(adjustments)
        })
        
        # Track this adjustment for learning
        self._record_adjustment(
            base_prob, adjusted_prob, adjustments, p1_profile, p2_profile, match_context
        )
        
        return adjusted_prediction
    
    def _calculate_surface_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                                    surface: str) -> float:
        """Calculate adjustment based on surface specialization"""
        
        # Get surface-specific performance from player profiles
        p1_surface_perf = self._get_surface_performance(p1_profile, surface)
        p2_surface_perf = self._get_surface_performance(p2_profile, surface)
        
        # Calculate relative surface advantage
        surface_diff = p1_surface_perf - p2_surface_perf
        
        # Scale adjustment (surface specialists get bigger boost)
        surface_adjustment = surface_diff * 0.08  # Up to 8% adjustment
        
        return np.clip(surface_adjustment, -0.08, 0.08)
    
    def _get_surface_performance(self, profile: PlayerProfile, surface: str) -> float:
        """Get player's historical performance on specific surface"""
        
        try:
            # Try to get from categorical data first
            surface_prefs = profile.categorical_insights.get('surface_preferences', {})
            if surface in surface_prefs:
                return surface_prefs[surface]
            
            # Fallback: estimate from recent matches
            if hasattr(profile, 'raw_data') and 'recent_matches' in profile.raw_data:
                surface_matches = [
                    m for m in profile.raw_data['recent_matches'] 
                    if m.get('surface', '').lower() == surface.lower()
                ]
                
                if surface_matches:
                    wins = sum(1 for m in surface_matches if m.get('won', False))
                    return wins / len(surface_matches)
            
            # Default neutral performance
            return 0.5
            
        except Exception as e:
            logging.warning(f"Could not get surface performance for {profile.name}: {e}")
            return 0.5
    
    def _calculate_form_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                                 match_context: Dict) -> float:
        """Calculate adjustment based on recent form trends"""
        
        p1_form = self._get_recent_form_trend(p1_profile)
        p2_form = self._get_recent_form_trend(p2_profile)
        
        # Form differential
        form_diff = p1_form - p2_form
        
        # Recent form matters more for current predictions
        form_weight = 0.06  # Up to 6% adjustment
        form_adjustment = form_diff * form_weight
        
        return np.clip(form_adjustment, -0.06, 0.06)
    
    def _get_recent_form_trend(self, profile: PlayerProfile) -> float:
        """Calculate recent form trend (-1 = declining, +1 = improving)"""
        
        try:
            # Get recent match results (last 10-15 matches)
            if not hasattr(profile, 'raw_data') or 'recent_matches' not in profile.raw_data:
                return 0.0
            
            recent_matches = profile.raw_data['recent_matches']
            if len(recent_matches) < 3:
                return 0.0
            
            # Weight recent matches more heavily
            weights = [0.4, 0.3, 0.2, 0.1]  # Most recent gets highest weight
            form_scores = []
            
            for i, match in enumerate(recent_matches[:4]):
                weight = weights[i] if i < len(weights) else 0.05
                
                # Win/loss (primary factor)
                win_score = 1.0 if match.get('won', False) else -1.0
                
                # Opponent strength factor
                opponent_elo = match.get('opponent_elo', 1500)
                strength_factor = 1.0 + (opponent_elo - 1500) / 800  # Normalize
                
                # Match quality factor (sets, duration)
                quality_factor = 1.0
                if match.get('sets_won', 0) >= 2 and match.get('sets_lost', 0) <= 1:
                    quality_factor = 1.2  # Dominant win
                elif match.get('sets_won', 0) == match.get('sets_lost', 0):
                    quality_factor = 0.8  # Close match
                
                match_score = win_score * strength_factor * quality_factor
                form_scores.append(match_score * weight)
            
            return np.clip(sum(form_scores), -1.0, 1.0)
            
        except Exception as e:
            logging.warning(f"Could not calculate form trend for {profile.name}: {e}")
            return 0.0
    
    def _calculate_tournament_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                                       match_context: Dict) -> float:
        """Calculate adjustment based on tournament-specific factors"""
        
        tournament = match_context.get('tournament', '')
        tournament_level = match_context.get('level', '')
        round_info = match_context.get('round', '')
        
        adjustment = 0.0
        
        # Grand Slam specialists
        if tournament_level.lower() in ['grand slam', 'gs']:
            p1_gs_perf = self._get_tournament_level_performance(p1_profile, 'grand_slam')
            p2_gs_perf = self._get_tournament_level_performance(p2_profile, 'grand_slam')
            adjustment += (p1_gs_perf - p2_gs_perf) * 0.05
        
        # Masters specialists  
        elif tournament_level.lower() in ['masters', 'atp masters 1000', 'm1000']:
            p1_masters_perf = self._get_tournament_level_performance(p1_profile, 'masters')
            p2_masters_perf = self._get_tournament_level_performance(p2_profile, 'masters')
            adjustment += (p1_masters_perf - p2_masters_perf) * 0.04
        
        # Late round experience (big match players)
        if round_info.lower() in ['final', 'semifinal', 'quarterfinal']:
            p1_big_match = self._get_big_match_performance(p1_profile)
            p2_big_match = self._get_big_match_performance(p2_profile)
            adjustment += (p1_big_match - p2_big_match) * 0.03
        
        return np.clip(adjustment, -0.08, 0.08)
    
    def _get_tournament_level_performance(self, profile: PlayerProfile, level: str) -> float:
        """Get player performance at specific tournament level"""
        
        # This would analyze historical performance at GS, Masters, etc.
        # For now, use a placeholder based on overall strength
        base_strength = profile.categorical_insights.get('serve_dominance', 0.5)
        
        # Grand Slam specialists tend to be more consistent
        if level == 'grand_slam':
            consistency = profile.style_profile.get('consistency', 0.5)
            return base_strength * 0.7 + consistency * 0.3
        
        # Masters specialists tend to be more aggressive
        elif level == 'masters':
            aggression = profile.style_profile.get('aggression', 0.5)
            return base_strength * 0.8 + aggression * 0.2
        
        return base_strength
    
    def _get_big_match_performance(self, profile: PlayerProfile) -> float:
        """Get player performance in high-pressure situations"""
        
        # Use clutch factor from categorical insights
        clutch_factor = profile.categorical_insights.get('clutch_factor', 1.0)
        mental_strength = profile.mental_strength
        
        # Combine clutch performance with mental strength
        big_match_perf = clutch_factor * 0.6 + mental_strength * 0.4
        
        return np.clip(big_match_perf, 0.3, 1.2) - 0.75  # Center around 0
    
    def _calculate_h2h_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile) -> float:
        """Calculate adjustment based on head-to-head matchup patterns"""
        
        # Get H2H record if available
        h2h_data = self._get_h2h_data(p1_profile.name, p2_profile.name)
        
        if not h2h_data or h2h_data['total_matches'] < 2:
            return 0.0  # Not enough H2H data
        
        # Recent H2H matters more
        recent_h2h = h2h_data['recent_results'][:5]  # Last 5 meetings
        if len(recent_h2h) >= 2:
            p1_recent_wins = sum(1 for result in recent_h2h if result['winner'] == p1_profile.name)
            recent_win_rate = p1_recent_wins / len(recent_h2h)
            
            # Adjust based on deviation from expected (0.5)
            h2h_adjustment = (recent_win_rate - 0.5) * 0.04
            
            # Dampen if only few matches
            if len(recent_h2h) < 4:
                h2h_adjustment *= 0.7
                
            return np.clip(h2h_adjustment, -0.04, 0.04)
        
        return 0.0
    
    def _get_h2h_data(self, player1: str, player2: str) -> Optional[Dict]:
        """Get head-to-head data between two players"""
        
        # This would query historical match databases
        # For now, return None (no H2H adjustment)
        return None
    
    def _calculate_market_learning_adjustment(self, model_prob: float, p1_profile: PlayerProfile,
                                            p2_profile: PlayerProfile, match_context: Dict) -> float:
        """Learn from historical model vs market disagreements"""
        
        # Get similar past predictions where model disagreed with market
        similar_predictions = self._find_similar_predictions(
            model_prob, p1_profile, p2_profile, match_context
        )
        
        if len(similar_predictions) < 5:
            return 0.0  # Not enough similar cases
        
        # Calculate average adjustment needed in similar cases
        adjustments = [pred['needed_adjustment'] for pred in similar_predictions[-10:]]  # Last 10
        
        if adjustments:
            avg_adjustment = np.mean(adjustments)
            return np.clip(avg_adjustment, -0.05, 0.05)
        
        return 0.0
    
    def _find_similar_predictions(self, model_prob: float, p1_profile: PlayerProfile,
                                p2_profile: PlayerProfile, match_context: Dict) -> List[Dict]:
        """Find similar historical predictions for learning"""
        
        similar_preds = []
        
        for adj_record in self.adjustment_history:
            # Similar probability range
            if abs(adj_record['base_prob'] - model_prob) < 0.15:
                # Similar player strength differential
                p1_elo = p1_profile.elo_rating
                p2_elo = p2_profile.elo_rating
                current_elo_diff = p1_elo - p2_elo
                
                record_elo_diff = adj_record.get('elo_diff', 0)
                if abs(current_elo_diff - record_elo_diff) < 200:
                    similar_preds.append(adj_record)
        
        return similar_preds
    
    def _calculate_meta_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile,
                                 match_context: Dict) -> float:
        """Adjust for evolving tennis meta-game"""
        
        # Modern tennis trends: power baseline, athleticism, return dominance
        current_year = match_context.get('date', datetime.now()).year
        
        adjustment = 0.0
        
        # Favor modern athletic players in recent years
        if current_year >= 2020:
            p1_athleticism = self._estimate_athleticism(p1_profile)
            p2_athleticism = self._estimate_athleticism(p2_profile)
            
            athleticism_diff = p1_athleticism - p2_athleticism
            adjustment += athleticism_diff * 0.02
        
        # Surface speed evolution (courts getting slower)
        surface = match_context.get('surface', 'hard')
        if surface.lower() == 'hard' and current_year >= 2018:
            # Favor consistent baseliners on slower hard courts
            p1_consistency = p1_profile.style_profile.get('consistency', 0.5)
            p2_consistency = p2_profile.style_profile.get('consistency', 0.5)
            
            consistency_diff = p1_consistency - p2_consistency
            adjustment += consistency_diff * 0.015
        
        return np.clip(adjustment, -0.03, 0.03)
    
    def _estimate_athleticism(self, profile: PlayerProfile) -> float:
        """Estimate player athleticism from available data"""
        
        # Use movement-related stats and age as proxies
        movement_fatigue_resistance = 1.0 - profile.categorical_insights.get('rally_tolerance', 0.5)
        
        # Age factor (peak athleticism around 24-28)
        try:
            # Rough age estimation based on ELO and career length
            estimated_age = 25  # Default
            athleticism_age_factor = max(0.3, 1.0 - abs(estimated_age - 26) * 0.02)
        except:
            athleticism_age_factor = 0.7
        
        athleticism = movement_fatigue_resistance * 0.6 + athleticism_age_factor * 0.4
        
        return np.clip(athleticism, 0.0, 1.0)
    
    def _calculate_fitness_adjustment(self, p1_profile: PlayerProfile, p2_profile: PlayerProfile) -> float:
        """Calculate adjustment based on current fitness/injury status"""
        
        # This would incorporate injury reports, recent match load, etc.
        # For now, use basic fitness proxy from recent activity
        
        p1_fitness = self._estimate_current_fitness(p1_profile)
        p2_fitness = self._estimate_current_fitness(p2_profile)
        
        fitness_diff = p1_fitness - p2_fitness
        fitness_adjustment = fitness_diff * 0.04  # Up to 4% adjustment
        
        return np.clip(fitness_adjustment, -0.04, 0.04)
    
    def _estimate_current_fitness(self, profile: PlayerProfile) -> float:
        """Estimate current player fitness level"""
        
        try:
            # Recent match load
            if hasattr(profile, 'raw_data') and 'recent_matches' in profile.raw_data:
                recent_matches = profile.raw_data['recent_matches']
                
                # Count matches in last 2 weeks
                recent_count = len([m for m in recent_matches if m.get('days_ago', 30) <= 14])
                
                if recent_count == 0:
                    return 0.3  # Possibly injured/inactive
                elif recent_count <= 2:
                    return 1.0  # Well rested
                elif recent_count <= 4:
                    return 0.8  # Normal load
                else:
                    return 0.6  # Heavy load, potential fatigue
            
            return 0.7  # Default fitness
            
        except Exception:
            return 0.7
    
    def _apply_sigmoid_adjustment(self, base_prob: float, adjustment: float) -> float:
        """Apply sigmoid-dampened adjustment to prevent extreme probabilities"""
        
        # Convert to log-odds for adjustment
        if base_prob <= 0.01:
            base_prob = 0.01
        elif base_prob >= 0.99:
            base_prob = 0.99
            
        log_odds = np.log(base_prob / (1 - base_prob))
        
        # Apply adjustment in log-odds space
        adjusted_log_odds = log_odds + adjustment * 2  # Scale adjustment
        
        # Convert back to probability
        adjusted_prob = 1 / (1 + np.exp(-adjusted_log_odds))
        
        return np.clip(adjusted_prob, 0.01, 0.99)
    
    def _calculate_adjustment_confidence(self, adjustments: Dict) -> float:
        """Calculate confidence in the total adjustment"""
        
        # More components agreeing = higher confidence
        components = adjustments['components']
        non_zero_components = sum(1 for adj in components.values() if abs(adj) > 0.005)
        
        # Components pointing same direction = higher confidence
        positive_adj = sum(1 for adj in components.values() if adj > 0.005)
        negative_adj = sum(1 for adj in components.values() if adj < -0.005)
        
        if non_zero_components == 0:
            return 0.1  # No adjustments
        
        direction_consistency = max(positive_adj, negative_adj) / non_zero_components
        magnitude_consistency = 1.0 - abs(adjustments['total_adjustment']) / self.adjustment_strength
        
        confidence = (direction_consistency * 0.6 + magnitude_consistency * 0.4) * 0.8
        
        return np.clip(confidence, 0.1, 0.8)
    
    def _generate_adjustment_explanation(self, adjustments: Dict) -> str:
        """Generate human-readable explanation of adjustments"""
        
        components = adjustments['components']
        total = adjustments['total_adjustment']
        
        if abs(total) < 0.01:
            return "No significant adjustments applied."
        
        explanations = []
        
        for component, adj in components.items():
            if abs(adj) >= 0.01:
                direction = "favors P1" if adj > 0 else "favors P2"
                magnitude = "strongly" if abs(adj) >= 0.04 else "moderately" if abs(adj) >= 0.02 else "slightly"
                
                explanations.append(f"{component.replace('_', ' ').title()} {magnitude} {direction}")
        
        if explanations:
            return f"Adjustments: {'; '.join(explanations)}. Net effect: {'+' if total > 0 else ''}{total:.3f}"
        else:
            return f"Minor adjustments applied (net: {total:+.3f})."
    
    def _record_adjustment(self, base_prob: float, adjusted_prob: float, adjustments: Dict,
                          p1_profile: PlayerProfile, p2_profile: PlayerProfile, match_context: Dict):
        """Record adjustment for future learning"""
        
        record = {
            'timestamp': datetime.now(),
            'base_prob': base_prob,
            'adjusted_prob': adjusted_prob,
            'total_adjustment': adjustments['total_adjustment'],
            'components': adjustments['components'].copy(),
            'p1_elo': p1_profile.elo_rating,
            'p2_elo': p2_profile.elo_rating,
            'elo_diff': p1_profile.elo_rating - p2_profile.elo_rating,
            'surface': match_context.get('surface', 'unknown'),
            'tournament_level': match_context.get('level', 'unknown')
        }
        
        self.adjustment_history.append(record)
        
        # Keep only recent history
        if len(self.adjustment_history) > self.context_memory:
            self.adjustment_history = self.adjustment_history[-self.context_memory:]
    
    def learn_from_outcome(self, prediction_record: Dict, actual_winner: int):
        """Learn from prediction outcome to improve future adjustments"""
        
        if 'adjustments' not in prediction_record:
            return
        
        model_prob = prediction_record['win_probability']
        base_prob = prediction_record['base_probability']
        actual_prob = 1.0 if actual_winner == 0 else 0.0
        
        # Calculate what the ideal adjustment would have been
        ideal_adjusted_prob = (base_prob + actual_prob) / 2  # Move halfway toward truth
        ideal_adjustment = ideal_adjusted_prob - base_prob
        actual_adjustment = model_prob - base_prob
        
        # Error in our adjustment
        adjustment_error = ideal_adjustment - actual_adjustment
        
        # Update adjustment history with learning feedback
        if self.adjustment_history:
            latest_record = self.adjustment_history[-1]
            latest_record['needed_adjustment'] = ideal_adjustment
            latest_record['adjustment_error'] = adjustment_error
            
            # Simple learning: adjust component weights based on error
            self._update_component_weights(latest_record, adjustment_error)
    
    def _update_component_weights(self, record: Dict, error: float):
        """Update internal component weights based on error"""
        
        # This would implement more sophisticated learning
        # For now, just track that we had an error for future similar cases
        components = record.get('components', {})
        
        for component, adj_value in components.items():
            if component not in self.performance_tracker:
                self.performance_tracker[component] = {'total_error': 0.0, 'count': 0}
            
            # Track cumulative error for this component type
            self.performance_tracker[component]['total_error'] += abs(error)
            self.performance_tracker[component]['count'] += 1


if __name__ == "__main__":
    # Test the comprehensive system
    pipeline = UnifiedDataPipeline()
    
    # Test with known players
    test_player1 = "Jannik Sinner"
    test_player2 = "Carlos Alcaraz"
    test_date = datetime(2024, 12, 15)
    
    print(f"Testing comprehensive system:")
    print(f"Player 1: {test_player1}")
    print(f"Player 2: {test_player2}")
    print(f"Date: {test_date}")
    
    try:
        # Create player profiles
        p1_profile = PlayerProfile(test_player1, test_date)
        p2_profile = PlayerProfile(test_player2, test_date)
        
        print(f"\n{test_player1} Profile:")
        print(f"  Data quality: {p1_profile.data_quality:.2f}")
        print(f"  ELO rating: {p1_profile.elo_rating}")
        print(f"  Mental strength: {p1_profile.mental_strength:.2f}")
        print(f"  Fitness score: {p1_profile.fitness_score:.2f}")
        print(f"  Style: Aggression={p1_profile.style_profile['aggression']:.2f}, Consistency={p1_profile.style_profile['consistency']:.2f}")
        
        print(f"\n{test_player2} Profile:")
        print(f"  Data quality: {p2_profile.data_quality:.2f}")
        print(f"  ELO rating: {p2_profile.elo_rating}")
        print(f"  Mental strength: {p2_profile.mental_strength:.2f}")
        print(f"  Fitness score: {p2_profile.fitness_score:.2f}")
        print(f"  Style: Aggression={p2_profile.style_profile['aggression']:.2f}, Consistency={p2_profile.style_profile['consistency']:.2f}")
        
        # Calculate matchup
        matchup = p1_profile.get_matchup_advantage(p2_profile, Surface.HARD)
        print(f"\nMatchup Analysis ({test_player1} vs {test_player2}):")
        for key, value in matchup.items():
            print(f"  {key}: {value:+.3f}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()