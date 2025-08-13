#!/usr/bin/env python3
"""
Sportsbook-Grade Tennis Prediction System
Combines live odds engine with edge identification for profitable tennis betting

This system provides:
1. Live odds calculation with Bayesian updates
2. Edge identification across multiple bookmakers  
3. Market calibration and CLV tracking
4. Prop bet generation from point sequences
5. Information asymmetry exploitation
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from model import TennisModelPipeline, ModelConfig
    from live_odds_engine import LiveOddsEngine, EdgeOpportunity, MatchState, MarketOdds
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Model imports failed: {e}")
    MODEL_AVAILABLE = False

class SportsbookGradeSystem:
    """Complete sportsbook-grade tennis prediction system"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.pipeline = None
        self.live_engine = None
        self.performance_tracker = {}
        self.market_efficiency_tracker = {}
        
        print("🏆 Initializing Sportsbook-Grade System...")
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        if not MODEL_AVAILABLE:
            print("❌ Model components not available")
            return
        
        try:
            # Initialize main prediction pipeline
            self.pipeline = TennisModelPipeline(
                config=ModelConfig(),
                fast_mode=False,
                enable_live_odds=True
            )
            
            # Load trained model if available
            if self.model_path:
                try:
                    self.pipeline.load(self.model_path)
                    print("✅ Trained model loaded")
                except:
                    print("⚠️  No trained model found - using default parameters")
            
            print("✅ Sportsbook-grade system initialized")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
    
    async def run_live_edge_detection(self, 
                                    match_schedule: List[Dict],
                                    duration_hours: int = 8,
                                    min_edge_threshold: float = 3.0):
        """
        Run live edge detection for a full day of matches
        
        Args:
            match_schedule: List of matches with player names, start times, etc.
            duration_hours: How long to run detection
            min_edge_threshold: Minimum edge % to alert on
        """
        
        if not self.pipeline or not self.pipeline.live_odds_engine:
            print("❌ Live odds engine not available")
            return
        
        print(f"🚀 Starting live edge detection for {len(match_schedule)} matches")
        print(f"📊 Running for {duration_hours} hours with {min_edge_threshold}% minimum edge")
        
        # Initialize tracking for all matches
        match_ids = [f"match_{i}" for i in range(len(match_schedule))]
        await self.pipeline.start_live_tracking(match_ids)
        
        # Run detection loop
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        edges_found = 0
        high_value_edges = 0
        total_profit_potential = 0.0
        
        while datetime.now() < end_time:
            try:
                # Get dashboard data
                dashboard = self.pipeline.get_live_dashboard()
                
                # Check for new edges
                new_edges = self.pipeline.get_betting_edges(min_edge=min_edge_threshold)
                
                if new_edges:
                    print(f"\n⚡ {len(new_edges)} edges detected at {datetime.now().strftime('%H:%M:%S')}")
                    
                    for edge in new_edges:
                        edges_found += 1
                        
                        if edge.edge_percent > 10.0:  # High-value edge
                            high_value_edges += 1
                            print(f"🎯 HIGH VALUE: {edge}")
                        else:
                            print(f"📈 Edge: {edge}")
                        
                        # Track profit potential
                        total_profit_potential += edge.expected_value
                
                # Print periodic summary
                if edges_found > 0 and edges_found % 10 == 0:
                    self._print_session_summary(edges_found, high_value_edges, total_profit_potential)
                
                # Simulate some match score updates
                await self._simulate_match_updates(match_ids)
                
                # Wait before next check
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                await asyncio.sleep(5)
        
        print(f"\n🏁 Live edge detection session complete!")
        self._print_final_summary(edges_found, high_value_edges, total_profit_potential, duration_hours)
        
        # Stop tracking
        self.pipeline.stop_live_tracking()
    
    async def _simulate_match_updates(self, match_ids: List[str]):
        """Simulate live match score updates for demo"""
        
        # Pick random match to update
        if match_ids:
            match_id = np.random.choice(match_ids)
            
            # Generate realistic score update
            score_updates = [
                {'games_p1': 3, 'games_p2': 2, 'points_p1': 40, 'points_p2': 30},
                {'games_p1': 4, 'games_p2': 3, 'points_p1': 15, 'points_p2': 0},
                {'games_p1': 5, 'games_p2': 4, 'points_p1': 0, 'points_p2': 15},
                {'sets_p1': 1, 'sets_p2': 0, 'games_p1': 1, 'games_p2': 0}
            ]
            
            update = np.random.choice(score_updates)
            self.pipeline.update_live_match_state(match_id, update)
    
    def analyze_market_efficiency(self, 
                                historical_edges: List[EdgeOpportunity],
                                results: List[Dict]) -> Dict:
        """
        Analyze market efficiency and our edge identification performance
        
        Args:
            historical_edges: Previously identified edges
            results: Actual match results
            
        Returns:
            Market efficiency analysis
        """
        
        if not historical_edges or not results:
            return {'error': 'No data provided'}
        
        analysis = {
            'total_edges_identified': len(historical_edges),
            'profitable_edges': 0,
            'total_roi': 0.0,
            'win_rate': 0.0,
            'avg_edge_size': np.mean([e.edge_percent for e in historical_edges]),
            'clv_performance': 0.0,
            'market_efficiency_score': 0.0
        }
        
        profitable_count = 0
        total_invested = 0.0
        total_return = 0.0
        
        for edge in historical_edges:
            # Find corresponding result
            result = self._find_matching_result(edge, results)
            
            if result:
                # Calculate if edge was profitable
                stake = edge.kelly_fraction * 100  # Assume $100 base unit
                
                if result['winner'] == edge.selection:
                    # Win
                    profit = stake * (edge.best_odds - 1)
                    total_return += profit
                    profitable_count += 1
                else:
                    # Loss
                    total_return -= stake
                
                total_invested += stake
        
        # Calculate performance metrics
        if total_invested > 0:
            analysis['total_roi'] = (total_return / total_invested) * 100
            analysis['win_rate'] = profitable_count / len(historical_edges)
        
        # Market efficiency score (how hard to beat)
        analysis['market_efficiency_score'] = self._calculate_market_efficiency_score(historical_edges, results)
        
        return analysis
    
    def _find_matching_result(self, edge: EdgeOpportunity, results: List[Dict]) -> Optional[Dict]:
        """Find result matching an edge opportunity"""
        for result in results:
            if result.get('match_id') == edge.match_id:
                return result
        return None
    
    def _calculate_market_efficiency_score(self, edges: List[EdgeOpportunity], results: List[Dict]) -> float:
        """Calculate market efficiency score (0-100, higher = more efficient)"""
        
        # Simplified calculation - would be more sophisticated in reality
        avg_edge_accuracy = 0.0
        
        for edge in edges:
            result = self._find_matching_result(edge, results)
            if result:
                # Compare our probability vs market probability vs actual outcome
                our_prob = edge.our_prob
                market_prob = edge.market_prob
                actual = 1.0 if result.get('winner') == edge.selection else 0.0
                
                # Score based on how much better our estimate was than market
                our_error = abs(our_prob - actual)
                market_error = abs(market_prob - actual)
                
                if our_error < market_error:
                    avg_edge_accuracy += 1.0
        
        if edges:
            avg_edge_accuracy /= len(edges)
        
        # Market efficiency is inverse of our ability to find profitable edges
        return max(0, 100 - (avg_edge_accuracy * 100))
    
    def generate_betting_strategy(self, 
                                available_matches: List[Dict],
                                bankroll: float = 10000.0,
                                max_risk_per_bet: float = 0.05) -> Dict:
        """
        Generate optimal betting strategy for available matches
        
        Args:
            available_matches: List of upcoming matches
            bankroll: Available betting capital
            max_risk_per_bet: Maximum % of bankroll per bet
            
        Returns:
            Optimized betting strategy
        """
        
        if not self.pipeline:
            return {'error': 'Prediction system not available'}
        
        print(f"🎯 Generating betting strategy for {len(available_matches)} matches")
        print(f"💰 Bankroll: ${bankroll:,.2f} | Max risk per bet: {max_risk_per_bet:.1%}")
        
        strategy = {
            'recommended_bets': [],
            'total_stakes': 0.0,
            'expected_roi': 0.0,
            'risk_level': 'MODERATE',
            'diversification_score': 0.0
        }
        
        # Generate predictions for all matches
        match_predictions = []
        
        for i, match in enumerate(available_matches):
            try:
                # Create match context
                context = {
                    'surface': match.get('surface', 'Hard'),
                    'best_of': match.get('best_of', 3),
                    'tournament_level': match.get('tournament_level', 'ATP250'),
                    'data_quality_score': 0.8
                }
                
                # Get prediction
                prediction = self.pipeline.predict(context)
                prediction['match_id'] = f"strategy_match_{i}"
                prediction['match_info'] = match
                match_predictions.append(prediction)
                
            except Exception as e:
                print(f"⚠️  Failed to predict match {i}: {e}")
        
        # Simulate market odds and identify edges
        for prediction in match_predictions:
            # Simulate realistic market odds
            our_prob = prediction['win_probability']
            market_odds = self._simulate_market_odds(our_prob)
            
            # Check for edge
            edge = self._calculate_betting_edge(our_prob, market_odds, prediction['confidence_level'])
            
            if edge and edge > 3.0:  # 3% minimum edge
                # Calculate Kelly stake
                kelly_fraction = self._calculate_kelly_fraction(our_prob, market_odds)
                kelly_fraction = min(kelly_fraction, max_risk_per_bet)  # Risk management
                
                stake = bankroll * kelly_fraction
                
                if stake >= 25:  # Minimum bet size
                    bet_recommendation = {
                        'match': prediction['match_info'],
                        'selection': 'Player 1 Win',
                        'odds': market_odds,
                        'our_probability': our_prob,
                        'edge_percent': edge,
                        'recommended_stake': stake,
                        'expected_value': stake * (our_prob * (market_odds - 1) - (1 - our_prob)),
                        'confidence': prediction['confidence_level']
                    }
                    
                    strategy['recommended_bets'].append(bet_recommendation)
                    strategy['total_stakes'] += stake
        
        # Calculate strategy metrics
        if strategy['recommended_bets']:
            total_ev = sum(bet['expected_value'] for bet in strategy['recommended_bets'])
            strategy['expected_roi'] = (total_ev / strategy['total_stakes']) * 100 if strategy['total_stakes'] > 0 else 0
            
            # Risk assessment
            risk_ratio = strategy['total_stakes'] / bankroll
            if risk_ratio > 0.2:
                strategy['risk_level'] = 'HIGH'
            elif risk_ratio < 0.05:
                strategy['risk_level'] = 'LOW'
            
            # Diversification
            strategy['diversification_score'] = min(len(strategy['recommended_bets']) / 10, 1.0)
            
        # Sort by expected value
        strategy['recommended_bets'].sort(key=lambda x: x['expected_value'], reverse=True)
        
        return strategy
    
    def _simulate_market_odds(self, our_prob: float) -> float:
        """Simulate realistic market odds with some inefficiency"""
        
        # Add some noise to create market inefficiency
        market_prob = our_prob + np.random.normal(0, 0.05)  # 5% standard deviation
        market_prob = np.clip(market_prob, 0.1, 0.9)
        
        # Convert to odds with bookmaker margin
        margin = 0.05  # 5% bookmaker margin
        fair_odds = 1 / market_prob
        market_odds = fair_odds * (1 - margin)
        
        return max(1.1, market_odds)  # Minimum odds of 1.10
    
    def _calculate_betting_edge(self, our_prob: float, market_odds: float, confidence: str) -> float:
        """Calculate betting edge percentage"""
        
        market_prob = 1 / market_odds
        
        # Adjust our probability based on confidence
        confidence_multiplier = {'HIGH': 0.95, 'MEDIUM': 0.85, 'LOW': 0.75}.get(confidence, 0.8)
        adjusted_prob = our_prob * confidence_multiplier
        
        # Calculate edge
        edge = ((adjusted_prob * market_odds - 1) / 1) * 100
        
        return edge if edge > 0 else None
    
    def _calculate_kelly_fraction(self, prob: float, odds: float) -> float:
        """Calculate Kelly fraction for optimal bet sizing"""
        
        if odds <= 1:
            return 0
        
        kelly = (prob * odds - 1) / (odds - 1)
        
        # Apply fractional Kelly for risk management
        return max(0, min(kelly * 0.5, 0.1))  # Half-Kelly, max 10%
    
    def _print_session_summary(self, edges_found: int, high_value: int, profit_potential: float):
        """Print session summary"""
        
        print(f"\n📊 Session Summary:")
        print(f"   Edges found: {edges_found}")
        print(f"   High-value edges (>10%): {high_value}")
        print(f"   Profit potential: ${profit_potential:.2f}")
        print(f"   Hit rate: {(high_value / edges_found) * 100 if edges_found > 0 else 0:.1f}%")
    
    def _print_final_summary(self, edges_found: int, high_value: int, profit_potential: float, hours: int):
        """Print final session summary"""
        
        print(f"\n🎉 Final Summary ({hours} hours):")
        print(f"{'='*50}")
        print(f"Total edges identified: {edges_found}")
        print(f"High-value edges (>10%): {high_value}")
        print(f"Total profit potential: ${profit_potential:.2f}")
        print(f"Edges per hour: {edges_found / hours:.1f}")
        print(f"Avg profit per edge: ${profit_potential / edges_found if edges_found > 0 else 0:.2f}")
        print(f"Quality rate: {(high_value / edges_found) * 100 if edges_found > 0 else 0:.1f}% high-value")
        print(f"{'='*50}")

def demo_sportsbook_system():
    """Comprehensive demo of the sportsbook-grade system"""
    
    print("🚀 Tennis Sportsbook-Grade System Demo")
    print("=" * 60)
    
    # Initialize system
    system = SportsbookGradeSystem()
    
    if not system.pipeline:
        print("❌ System not available - check model and dependencies")
        return
    
    # Sample match schedule
    match_schedule = [
        {'player1': 'Novak Djokovic', 'player2': 'Rafael Nadal', 'surface': 'Clay', 'tournament_level': 'Grand Slam'},
        {'player1': 'Carlos Alcaraz', 'player2': 'Jannik Sinner', 'surface': 'Hard', 'tournament_level': 'ATP 1000'},
        {'player1': 'Alexander Zverev', 'player2': 'Stefanos Tsitsipas', 'surface': 'Hard', 'tournament_level': 'ATP 500'},
        {'player1': 'Daniil Medvedev', 'player2': 'Andrey Rublev', 'surface': 'Hard', 'tournament_level': 'ATP 250'},
    ]
    
    print(f"📅 Sample schedule: {len(match_schedule)} matches")
    
    # Demo 1: Betting Strategy Generation
    print("\n1️⃣  BETTING STRATEGY GENERATION")
    print("-" * 40)
    
    strategy = system.generate_betting_strategy(match_schedule, bankroll=10000.0)
    
    if strategy.get('recommended_bets'):
        print(f"💰 Strategy Overview:")
        print(f"   Recommended bets: {len(strategy['recommended_bets'])}")
        print(f"   Total stakes: ${strategy['total_stakes']:.2f}")
        print(f"   Expected ROI: {strategy['expected_roi']:.2f}%")
        print(f"   Risk level: {strategy['risk_level']}")
        
        print(f"\n🎯 Top Recommendations:")
        for i, bet in enumerate(strategy['recommended_bets'][:3]):
            print(f"   {i+1}. {bet['match']['player1']} vs {bet['match']['player2']}")
            print(f"      Edge: {bet['edge_percent']:.1f}% | Stake: ${bet['recommended_stake']:.2f}")
            print(f"      Expected Value: ${bet['expected_value']:.2f}")
    else:
        print("⚠️  No profitable opportunities found")
    
    # Demo 2: Market Efficiency Analysis
    print("\n2️⃣  MARKET EFFICIENCY ANALYSIS")
    print("-" * 40)
    
    # Simulate historical performance
    sample_edges = [
        EdgeOpportunity(
            match_id="hist_1",
            bet_type="Match Winner",
            selection="Player 1",
            our_prob=0.65,
            market_prob=0.55,
            best_odds=1.85,
            edge_percent=8.2,
            confidence=0.85,
            kelly_fraction=0.05,
            expected_value=12.5,
            market_source="BookmakerA",
            timestamp=datetime.now()
        )
    ]
    
    sample_results = [
        {'match_id': 'hist_1', 'winner': 'Player 1'}
    ]
    
    efficiency = system.analyze_market_efficiency(sample_edges, sample_results)
    
    print(f"📊 Market Analysis:")
    print(f"   Edges identified: {efficiency.get('total_edges_identified', 0)}")
    print(f"   Win rate: {efficiency.get('win_rate', 0):.1%}")
    print(f"   Total ROI: {efficiency.get('total_roi', 0):.2f}%")
    print(f"   Market efficiency score: {efficiency.get('market_efficiency_score', 0):.1f}/100")
    
    # Demo 3: Live Edge Detection (Short Demo)
    print("\n3️⃣  LIVE EDGE DETECTION (30-second demo)")
    print("-" * 40)
    
    async def run_short_demo():
        await system.run_live_edge_detection(
            match_schedule=match_schedule[:2],  # Just 2 matches
            duration_hours=0.01,  # 30 seconds
            min_edge_threshold=2.0  # Lower threshold for demo
        )
    
    try:
        # Run the live demo
        asyncio.run(run_short_demo())
    except Exception as e:
        print(f"⚠️  Live demo failed: {e}")
        print("This is normal if dependencies are missing")
    
    print("\n🏆 Demo Complete!")
    print("The system provides:")
    print("✅ Live odds calculation with Bayesian updates")
    print("✅ Real-time edge identification")
    print("✅ Market calibration and efficiency analysis")
    print("✅ Optimal betting strategy generation")
    print("✅ Risk management and position sizing")
    
if __name__ == "__main__":
    demo_sportsbook_system()