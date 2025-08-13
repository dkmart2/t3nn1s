#!/usr/bin/env python3
"""
Live Odds Engine & Edge Detection System
Combines real-time probability updates with edge identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import logging
from collections import defaultdict, deque
import threading
import time
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

@dataclass
class MatchState:
    """Current match state for live updates"""
    match_id: str
    player1: str
    player2: str
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0
    games_p2: int = 0
    points_p1: int = 0
    points_p2: int = 0
    serving: int = 1  # 1 or 2
    surface: str = "Hard"
    best_of: int = 3
    momentum_state: str = "NEUTRAL"
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'sets_p1': self.sets_p1,
            'sets_p2': self.sets_p2,
            'games_p1': self.games_p1,
            'games_p2': self.games_p2,
            'points_p1': self.points_p1,
            'points_p2': self.points_p2,
            'serving': self.serving,
            'momentum': self.momentum_state
        }

@dataclass
class MarketOdds:
    """Market odds from various sources"""
    source: str
    p1_win: float
    p2_win: float
    timestamp: datetime
    volume: float = 0.0
    
    @property
    def implied_prob_p1(self) -> float:
        return 1 / self.p1_win if self.p1_win > 0 else 0.0
    
    @property
    def implied_prob_p2(self) -> float:
        return 1 / self.p2_win if self.p2_win > 0 else 0.0
    
    @property
    def overround(self) -> float:
        return self.implied_prob_p1 + self.implied_prob_p2

@dataclass
class EdgeOpportunity:
    """Identified betting edge"""
    match_id: str
    bet_type: str
    selection: str
    our_prob: float
    market_prob: float
    best_odds: float
    edge_percent: float
    confidence: float
    kelly_fraction: float
    expected_value: float
    market_source: str
    timestamp: datetime
    
    def __str__(self):
        return f"{self.bet_type}: {self.selection} @ {self.best_odds:.2f} (Edge: {self.edge_percent:.1f}%)"

class PathToVictoryCalculator:
    """Calculate probabilistic paths to match victory"""
    
    def __init__(self):
        self.point_transitions = self._build_point_transitions()
        self.game_transitions = self._build_game_transitions()
    
    def _build_point_transitions(self) -> Dict:
        """Point score transitions (0-0, 15-0, etc.)"""
        states = [
            (0, 0), (15, 0), (30, 0), (40, 0),  # P1 leading
            (0, 15), (15, 15), (30, 15), (40, 15),
            (0, 30), (15, 30), (30, 30), (40, 30),
            (0, 40), (15, 40), (30, 40), (40, 40),  # Deuce
            ('A', 40), (40, 'A')  # Advantage states
        ]
        
        transitions = {}
        for state in states:
            transitions[state] = self._get_next_point_states(state)
        
        return transitions
    
    def _get_next_point_states(self, state: Tuple) -> List[Tuple]:
        """Get possible next states from current point state"""
        p1_score, p2_score = state
        
        # Handle advantage/deuce
        if state == (40, 40):  # Deuce
            return [('A', 40), (40, 'A')]
        elif state == ('A', 40):  # P1 advantage
            return [(0, 0), (40, 40)]  # Win game or back to deuce
        elif state == (40, 'A'):  # P2 advantage
            return [(40, 40), (0, 0)]  # Back to deuce or win game
        
        # Normal progression
        score_map = {0: 15, 15: 30, 30: 40}
        
        next_states = []
        
        # P1 wins point
        if p1_score in score_map:
            next_states.append((score_map[p1_score], p2_score))
        elif p1_score == 40 and p2_score < 40:
            next_states.append((0, 0))  # P1 wins game
        
        # P2 wins point
        if p2_score in score_map:
            next_states.append((p1_score, score_map[p2_score]))
        elif p2_score == 40 and p1_score < 40:
            next_states.append((0, 0))  # P2 wins game
        
        return next_states
    
    def _build_game_transitions(self) -> Dict:
        """Game score transitions within sets"""
        return {}  # Implement based on current game score
    
    def calculate_match_win_probability(self, 
                                      match_state: MatchState,
                                      p1_serve_prob: float,
                                      p2_serve_prob: float,
                                      p1_return_prob: float = None,
                                      p2_return_prob: float = None) -> Dict[str, float]:
        """
        Calculate live match win probabilities using Monte Carlo simulation
        """
        if p1_return_prob is None:
            p1_return_prob = 1 - p2_serve_prob
        if p2_return_prob is None:
            p2_return_prob = 1 - p1_serve_prob
        
        # Run Monte Carlo simulation
        num_simulations = 10000
        p1_wins = 0
        
        for _ in range(num_simulations):
            if self._simulate_match_completion(match_state, p1_serve_prob, p2_serve_prob):
                p1_wins += 1
        
        p1_prob = p1_wins / num_simulations
        p2_prob = 1 - p1_prob
        
        # Add confidence intervals
        std_error = np.sqrt(p1_prob * (1 - p1_prob) / num_simulations)
        ci_lower = max(0, p1_prob - 1.96 * std_error)
        ci_upper = min(1, p1_prob + 1.96 * std_error)
        
        return {
            'p1_win_prob': p1_prob,
            'p2_win_prob': p2_prob,
            'confidence_interval': (ci_lower, ci_upper),
            'std_error': std_error
        }
    
    def _simulate_match_completion(self, 
                                 initial_state: MatchState,
                                 p1_serve_prob: float,
                                 p2_serve_prob: float) -> bool:
        """Simulate match completion, return True if P1 wins"""
        state = MatchState(**initial_state.__dict__)
        
        while not self._is_match_complete(state):
            # Simulate next point
            serving_player = state.serving
            point_win_prob = p1_serve_prob if serving_player == 1 else (1 - p2_serve_prob)
            
            if np.random.random() < point_win_prob:
                # Server wins point
                if serving_player == 1:
                    state.points_p1 += 1
                else:
                    state.points_p2 += 1
            else:
                # Returner wins point
                if serving_player == 1:
                    state.points_p2 += 1
                else:
                    state.points_p1 += 1
            
            # Update game/set scores
            self._update_scores(state)
        
        return state.sets_p1 > state.sets_p2
    
    def _is_match_complete(self, state: MatchState) -> bool:
        """Check if match is complete"""
        sets_to_win = (state.best_of + 1) // 2
        return state.sets_p1 >= sets_to_win or state.sets_p2 >= sets_to_win
    
    def _update_scores(self, state: MatchState):
        """Update game and set scores based on points"""
        # Simplified scoring logic
        if state.points_p1 >= 4 and state.points_p1 - state.points_p2 >= 2:
            # P1 wins game
            state.games_p1 += 1
            state.points_p1 = 0
            state.points_p2 = 0
            state.serving = 2 if state.serving == 1 else 1
            
            # Check for set win
            if state.games_p1 >= 6 and state.games_p1 - state.games_p2 >= 2:
                state.sets_p1 += 1
                state.games_p1 = 0
                state.games_p2 = 0
        
        elif state.points_p2 >= 4 and state.points_p2 - state.points_p1 >= 2:
            # P2 wins game
            state.games_p2 += 1
            state.points_p1 = 0
            state.points_p2 = 0
            state.serving = 2 if state.serving == 1 else 1
            
            # Check for set win
            if state.games_p2 >= 6 and state.games_p2 - state.games_p1 >= 2:
                state.sets_p2 += 1
                state.games_p1 = 0
                state.games_p2 = 0

class BayesianOddsUpdater:
    """Bayesian updating of match probabilities"""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.point_history = deque(maxlen=50)  # Recent point outcomes
    
    def update_with_point_outcome(self, 
                                p1_won_point: bool,
                                serving_player: int,
                                current_probs: Dict[str, float]) -> Dict[str, float]:
        """Update probabilities based on point outcome"""
        self.point_history.append({
            'p1_won': p1_won_point,
            'serving': serving_player,
            'timestamp': datetime.now()
        })
        
        # Calculate serve/return performance updates
        recent_serve_p1 = [p for p in self.point_history if p['serving'] == 1]
        recent_serve_p2 = [p for p in self.point_history if p['serving'] == 2]
        
        # Update serve probabilities using Bayesian updating
        if recent_serve_p1:
            p1_serve_wins = sum(1 for p in recent_serve_p1 if p['p1_won'])
            p1_serve_total = len(recent_serve_p1)
            
            # Beta-binomial update
            alpha_post = self.alpha_prior + p1_serve_wins
            beta_post = self.beta_prior + p1_serve_total - p1_serve_wins
            p1_serve_prob = alpha_post / (alpha_post + beta_post)
        else:
            p1_serve_prob = current_probs.get('p1_serve_prob', 0.65)
        
        if recent_serve_p2:
            p2_serve_wins = sum(1 for p in recent_serve_p2 if not p['p1_won'])
            p2_serve_total = len(recent_serve_p2)
            
            alpha_post = self.alpha_prior + p2_serve_wins
            beta_post = self.beta_prior + p2_serve_total - p2_serve_wins
            p2_serve_prob = alpha_post / (alpha_post + beta_post)
        else:
            p2_serve_prob = current_probs.get('p2_serve_prob', 0.65)
        
        return {
            'p1_serve_prob': p1_serve_prob,
            'p2_serve_prob': p2_serve_prob,
            'p1_return_prob': 1 - p2_serve_prob,
            'p2_return_prob': 1 - p1_serve_prob
        }

class EdgeDetectionEngine:
    """Identify betting edges by comparing our probabilities with market odds"""
    
    def __init__(self, min_edge_threshold: float = 0.05, min_confidence: float = 0.7):
        self.min_edge_threshold = min_edge_threshold
        self.min_confidence = min_confidence
        self.market_history = defaultdict(list)
        self.clv_tracker = {}  # Closing Line Value tracking
    
    def find_edges(self, 
                  match_id: str,
                  our_probabilities: Dict[str, float],
                  market_odds: List[MarketOdds],
                  confidence_level: float = 0.8) -> List[EdgeOpportunity]:
        """Find betting edges across all markets"""
        edges = []
        
        our_p1_prob = our_probabilities.get('p1_win_prob', 0.5)
        our_p2_prob = our_probabilities.get('p2_win_prob', 0.5)
        
        for odds in market_odds:
            # P1 to win edge
            edge_p1 = self._calculate_edge(our_p1_prob, odds.p1_win, confidence_level)
            if edge_p1['edge_percent'] >= self.min_edge_threshold * 100:
                edges.append(EdgeOpportunity(
                    match_id=match_id,
                    bet_type="Match Winner",
                    selection="Player 1",
                    our_prob=our_p1_prob,
                    market_prob=odds.implied_prob_p1,
                    best_odds=odds.p1_win,
                    edge_percent=edge_p1['edge_percent'],
                    confidence=confidence_level,
                    kelly_fraction=edge_p1['kelly_fraction'],
                    expected_value=edge_p1['expected_value'],
                    market_source=odds.source,
                    timestamp=datetime.now()
                ))
            
            # P2 to win edge
            edge_p2 = self._calculate_edge(our_p2_prob, odds.p2_win, confidence_level)
            if edge_p2['edge_percent'] >= self.min_edge_threshold * 100:
                edges.append(EdgeOpportunity(
                    match_id=match_id,
                    bet_type="Match Winner",
                    selection="Player 2",
                    our_prob=our_p2_prob,
                    market_prob=odds.implied_prob_p2,
                    best_odds=odds.p2_win,
                    edge_percent=edge_p2['edge_percent'],
                    confidence=confidence_level,
                    kelly_fraction=edge_p2['kelly_fraction'],
                    expected_value=edge_p2['expected_value'],
                    market_source=odds.source,
                    timestamp=datetime.now()
                ))
        
        return sorted(edges, key=lambda x: x.edge_percent, reverse=True)
    
    def _calculate_edge(self, our_prob: float, market_odds: float, confidence: float) -> Dict:
        """Calculate betting edge and Kelly fraction"""
        market_prob = 1 / market_odds if market_odds > 0 else 0
        
        # Adjust our probability for confidence
        confidence_adjusted_prob = our_prob * confidence + (1 - confidence) * 0.5
        
        edge_percent = ((confidence_adjusted_prob * market_odds - 1) / 1) * 100
        expected_value = confidence_adjusted_prob * (market_odds - 1) - (1 - confidence_adjusted_prob)
        
        # Kelly fraction
        if market_odds > 1:
            kelly_fraction = (confidence_adjusted_prob * market_odds - 1) / (market_odds - 1)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0
        
        return {
            'edge_percent': edge_percent,
            'expected_value': expected_value,
            'kelly_fraction': kelly_fraction
        }
    
    def track_closing_line_value(self, match_id: str, bet_odds: float, closing_odds: float) -> float:
        """Track CLV for our betting strategy"""
        clv = (closing_odds - bet_odds) / bet_odds * 100
        
        if match_id not in self.clv_tracker:
            self.clv_tracker[match_id] = []
        
        self.clv_tracker[match_id].append({
            'bet_odds': bet_odds,
            'closing_odds': closing_odds,
            'clv': clv,
            'timestamp': datetime.now()
        })
        
        return clv
    
    def get_clv_summary(self) -> Dict:
        """Get summary of CLV performance"""
        all_clv = []
        for match_clvs in self.clv_tracker.values():
            all_clv.extend([entry['clv'] for entry in match_clvs])
        
        if not all_clv:
            return {}
        
        return {
            'mean_clv': np.mean(all_clv),
            'median_clv': np.median(all_clv),
            'positive_clv_rate': sum(1 for clv in all_clv if clv > 0) / len(all_clv),
            'total_bets': len(all_clv)
        }

class PropBetGenerator:
    """Generate prop bet probabilities from point-by-point predictions"""
    
    def __init__(self):
        self.prop_types = [
            'total_games',
            'set_betting',
            'first_set_winner',
            'total_aces',
            'total_double_faults',
            'match_duration'
        ]
    
    def generate_prop_probabilities(self, 
                                  match_state: MatchState,
                                  player_stats: Dict,
                                  match_simulation_results: List[Dict]) -> Dict:
        """Generate prop bet probabilities from match simulations"""
        props = {}
        
        # Total games
        game_totals = [result['total_games'] for result in match_simulation_results]
        props['total_games'] = {
            'over_21_5': sum(1 for g in game_totals if g > 21.5) / len(game_totals),
            'under_21_5': sum(1 for g in game_totals if g < 21.5) / len(game_totals),
            'over_22_5': sum(1 for g in game_totals if g > 22.5) / len(game_totals),
            'under_22_5': sum(1 for g in game_totals if g < 22.5) / len(game_totals)
        }
        
        # Set betting (exact score)
        set_scores = defaultdict(int)
        for result in match_simulation_results:
            score = f"{result['sets_p1']}-{result['sets_p2']}"
            set_scores[score] += 1
        
        total_sims = len(match_simulation_results)
        props['set_betting'] = {score: count/total_sims for score, count in set_scores.items()}
        
        # First set winner (if match hasn't started)
        if match_state.sets_p1 == 0 and match_state.sets_p2 == 0:
            first_set_p1_wins = sum(1 for result in match_simulation_results if result['first_set_winner'] == 1)
            props['first_set_winner'] = {
                'p1': first_set_p1_wins / total_sims,
                'p2': (total_sims - first_set_p1_wins) / total_sims
            }
        
        return props

class LiveOddsEngine:
    """Main live odds engine coordinating all components"""
    
    def __init__(self, 
                 model=None,
                 update_frequency: int = 30,  # seconds
                 max_concurrent_matches: int = 10):
        
        self.model = model
        self.update_frequency = update_frequency
        self.max_concurrent_matches = max_concurrent_matches
        
        # Components
        self.path_calculator = PathToVictoryCalculator()
        self.bayesian_updater = BayesianOddsUpdater()
        self.edge_detector = EdgeDetectionEngine()
        self.prop_generator = PropBetGenerator()
        
        # State management
        self.active_matches = {}  # match_id -> MatchState
        self.current_odds = {}    # match_id -> our calculated odds
        self.market_odds = defaultdict(list)  # match_id -> List[MarketOdds]
        self.identified_edges = defaultdict(list)  # match_id -> List[EdgeOpportunity]
        
        # Threading
        self.running = False
        self.update_thread = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start_live_tracking(self, match_ids: List[str]):
        """Start live tracking for specified matches"""
        self.running = True
        
        # Initialize matches
        for match_id in match_ids:
            await self.initialize_match(match_id)
        
        # Start update loop
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info(f"Started live tracking for {len(match_ids)} matches")
    
    async def initialize_match(self, match_id: str):
        """Initialize match for live tracking"""
        # This would typically fetch initial match data from API
        # For now, create placeholder
        match_state = MatchState(
            match_id=match_id,
            player1=f"Player1_{match_id}",
            player2=f"Player2_{match_id}"
        )
        
        self.active_matches[match_id] = match_state
        self.logger.info(f"Initialized match {match_id}")
    
    def _update_loop(self):
        """Main update loop running in background thread"""
        while self.running:
            try:
                # Update all active matches
                for match_id in list(self.active_matches.keys()):
                    self._update_match_probabilities(match_id)
                    self._fetch_market_odds(match_id)
                    self._identify_edges(match_id)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def _update_match_probabilities(self, match_id: str):
        """Update probabilities for a specific match"""
        if match_id not in self.active_matches:
            return
        
        match_state = self.active_matches[match_id]
        
        try:
            # Get base probabilities from model if available
            if self.model:
                base_probs = self._get_model_probabilities(match_state)
            else:
                base_probs = {'p1_serve_prob': 0.65, 'p2_serve_prob': 0.65}
            
            # Calculate live match probabilities
            live_probs = self.path_calculator.calculate_match_win_probability(
                match_state=match_state,
                p1_serve_prob=base_probs['p1_serve_prob'],
                p2_serve_prob=base_probs['p2_serve_prob']
            )
            
            # Generate prop probabilities
            prop_probs = self._generate_prop_probabilities(match_state, base_probs)
            
            # Store updated probabilities
            self.current_odds[match_id] = {
                **live_probs,
                **prop_probs,
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating probabilities for {match_id}: {e}")
    
    def _get_model_probabilities(self, match_state: MatchState) -> Dict:
        """Get base probabilities from trained model"""
        # This would use the actual model to predict serve/return probabilities
        # For now, return reasonable defaults
        return {
            'p1_serve_prob': 0.65,
            'p2_serve_prob': 0.62,
            'p1_return_prob': 0.38,
            'p2_return_prob': 0.35
        }
    
    def _generate_prop_probabilities(self, match_state: MatchState, base_probs: Dict) -> Dict:
        """Generate prop bet probabilities"""
        # Simulate match multiple times to get prop distributions
        simulations = []
        for _ in range(1000):
            sim_result = self._simulate_match(match_state, base_probs)
            simulations.append(sim_result)
        
        return self.prop_generator.generate_prop_probabilities(
            match_state, {}, simulations
        )
    
    def _simulate_match(self, match_state: MatchState, probs: Dict) -> Dict:
        """Single match simulation for prop generation"""
        # Simplified simulation result
        return {
            'total_games': np.random.poisson(22),
            'sets_p1': np.random.choice([2, 1, 0], p=[0.6, 0.3, 0.1]),
            'sets_p2': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),
            'first_set_winner': np.random.choice([1, 2], p=[0.55, 0.45])
        }
    
    def _fetch_market_odds(self, match_id: str):
        """Fetch current market odds from bookmakers"""
        # This would make API calls to various bookmakers
        # For now, generate sample market odds
        
        sample_odds = [
            MarketOdds(
                source="Bookmaker1",
                p1_win=1.85,
                p2_win=1.95,
                timestamp=datetime.now()
            ),
            MarketOdds(
                source="Bookmaker2", 
                p1_win=1.90,
                p2_win=1.90,
                timestamp=datetime.now()
            )
        ]
        
        self.market_odds[match_id] = sample_odds
    
    def _identify_edges(self, match_id: str):
        """Identify betting edges for match"""
        if match_id not in self.current_odds or match_id not in self.market_odds:
            return
        
        our_probs = self.current_odds[match_id]
        market_odds = self.market_odds[match_id]
        
        edges = self.edge_detector.find_edges(
            match_id=match_id,
            our_probabilities=our_probs,
            market_odds=market_odds,
            confidence_level=0.8
        )
        
        self.identified_edges[match_id] = edges
        
        # Log significant edges
        for edge in edges:
            if edge.edge_percent > 10:  # 10%+ edge
                self.logger.info(f"SIGNIFICANT EDGE: {edge}")
    
    def update_match_state(self, match_id: str, new_state_data: Dict):
        """Update match state with live score data"""
        if match_id in self.active_matches:
            match_state = self.active_matches[match_id]
            
            # Update state
            for key, value in new_state_data.items():
                if hasattr(match_state, key):
                    setattr(match_state, key, value)
            
            match_state.last_update = datetime.now()
            
            # Trigger immediate probability update
            self._update_match_probabilities(match_id)
            self._identify_edges(match_id)
    
    def get_current_odds(self, match_id: str = None) -> Dict:
        """Get current odds for match(es)"""
        if match_id:
            return self.current_odds.get(match_id, {})
        return self.current_odds
    
    def get_identified_edges(self, match_id: str = None, min_edge: float = 5.0) -> List[EdgeOpportunity]:
        """Get identified edges above threshold"""
        if match_id:
            edges = self.identified_edges.get(match_id, [])
        else:
            edges = []
            for match_edges in self.identified_edges.values():
                edges.extend(match_edges)
        
        return [edge for edge in edges if edge.edge_percent >= min_edge]
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        active_edges = self.get_identified_edges(min_edge=3.0)
        
        return {
            'active_matches': len(self.active_matches),
            'total_edges_found': len(active_edges),
            'high_value_edges': len([e for e in active_edges if e.edge_percent > 10]),
            'avg_edge_size': np.mean([e.edge_percent for e in active_edges]) if active_edges else 0,
            'clv_summary': self.edge_detector.get_clv_summary(),
            'recent_edges': sorted(active_edges, key=lambda x: x.timestamp, reverse=True)[:10],
            'best_edges': sorted(active_edges, key=lambda x: x.edge_percent, reverse=True)[:5]
        }
    
    def stop_tracking(self):
        """Stop live tracking"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Stopped live tracking")

def demo_live_odds_engine():
    """Demonstration of the live odds engine"""
    print("🚀 Initializing Live Odds Engine...")
    
    engine = LiveOddsEngine(update_frequency=10)  # Update every 10 seconds for demo
    
    # Start tracking some matches
    match_ids = ["match_001", "match_002", "match_003"]
    
    print(f"📡 Starting live tracking for matches: {match_ids}")
    
    async def run_demo():
        await engine.start_live_tracking(match_ids)
        
        # Simulate some match updates
        for i in range(5):
            await asyncio.sleep(15)
            
            # Simulate score update
            engine.update_match_state("match_001", {
                'games_p1': i + 1,
                'points_p1': 30 if i % 2 == 0 else 0
            })
            
            print(f"\n📊 Dashboard Update #{i+1}:")
            dashboard = engine.get_dashboard_data()
            
            print(f"Active matches: {dashboard['active_matches']}")
            print(f"Edges found: {dashboard['total_edges_found']}")
            print(f"High-value edges: {dashboard['high_value_edges']}")
            
            if dashboard['recent_edges']:
                print("\n🎯 Recent edges:")
                for edge in dashboard['recent_edges'][:3]:
                    print(f"  {edge}")
        
        engine.stop_tracking()
    
    # Run the demo
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        engine.stop_tracking()

if __name__ == "__main__":
    demo_live_odds_engine()