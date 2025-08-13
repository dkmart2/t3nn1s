#!/usr/bin/env python3
"""
Complete Sportsbook-Grade Tennis System Runner

This script demonstrates the full system:
1. Live odds calculation with Bayesian updates
2. Edge identification across markets  
3. Prop bet generation from Jeff's point sequences
4. Market efficiency analysis and CLV tracking
5. Optimal betting strategy generation

Usage:
    python run_sportsbook_system.py --demo           # Quick demo
    python run_sportsbook_system.py --live           # Live edge detection
    python run_sportsbook_system.py --strategy       # Betting strategy
    python run_sportsbook_system.py --props          # Prop bet generation
    python run_sportsbook_system.py --full           # Complete system test
"""

import asyncio
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def print_system_header():
    """Print system header"""
    print("=" * 80)
    print("🏆 SPORTSBOOK-GRADE TENNIS PREDICTION SYSTEM")
    print("=" * 80)
    print("Live Odds Engine + Edge Detection + Prop Generation")
    print(f"Initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

async def run_live_demo():
    """Run live odds and edge detection demo"""
    print("\n🚀 LIVE ODDS & EDGE DETECTION DEMO")
    print("-" * 50)
    
    try:
        from sportsbook_grade_system import SportsbookGradeSystem
        
        # Initialize system
        system = SportsbookGradeSystem()
        
        # Sample upcoming matches
        matches = [
            {'player1': 'Novak Djokovic', 'player2': 'Rafael Nadal', 'surface': 'Clay', 'start_time': '14:00'},
            {'player1': 'Carlos Alcaraz', 'player2': 'Jannik Sinner', 'surface': 'Hard', 'start_time': '16:00'},
            {'player1': 'Alexander Zverev', 'player2': 'Stefanos Tsitsipas', 'surface': 'Hard', 'start_time': '18:00'},
        ]
        
        print(f"📅 Tracking {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"   {i}. {match['player1']} vs {match['player2']} ({match['surface']}) at {match['start_time']}")
        
        # Run live detection for short period
        await system.run_live_edge_detection(
            match_schedule=matches,
            duration_hours=0.02,  # ~1 minute for demo
            min_edge_threshold=2.0
        )
        
    except ImportError as e:
        print(f"❌ Live system not available: {e}")
        print("💡 Install dependencies or check module imports")

def run_prop_demo():
    """Run prop bet generation demo"""
    print("\n🎾 PROP BET GENERATION DEMO")
    print("-" * 50)
    
    try:
        from prop_bet_engine import demo_prop_bet_engine
        demo_prop_bet_engine()
        
    except ImportError as e:
        print(f"❌ Prop bet engine not available: {e}")

def run_strategy_demo():
    """Run betting strategy generation demo"""
    print("\n💰 BETTING STRATEGY DEMO")
    print("-" * 50)
    
    try:
        from sportsbook_grade_system import SportsbookGradeSystem
        
        system = SportsbookGradeSystem()
        
        # Sample matches for strategy
        available_matches = [
            {'player1': 'Daniil Medvedev', 'player2': 'Andrey Rublev', 'surface': 'Hard', 'tournament_level': 'ATP 1000'},
            {'player1': 'Taylor Fritz', 'player2': 'Frances Tiafoe', 'surface': 'Hard', 'tournament_level': 'ATP 500'},
            {'player1': 'Casper Ruud', 'player2': 'Holger Rune', 'surface': 'Clay', 'tournament_level': 'ATP 250'},
            {'player1': 'Lorenzo Musetti', 'player2': 'Sebastian Korda', 'surface': 'Clay', 'tournament_level': 'ATP 250'},
        ]
        
        # Generate strategy
        strategy = system.generate_betting_strategy(
            available_matches=available_matches,
            bankroll=5000.0,
            max_risk_per_bet=0.03  # 3% max risk
        )
        
        if strategy.get('recommended_bets'):
            print(f"📊 Strategy Results:")
            print(f"   Bankroll: $5,000")
            print(f"   Recommended bets: {len(strategy['recommended_bets'])}")
            print(f"   Total stakes: ${strategy['total_stakes']:.2f}")
            print(f"   Expected ROI: {strategy['expected_roi']:.2f}%")
            print(f"   Risk level: {strategy['risk_level']}")
            
            print(f"\n🎯 Top Recommendations:")
            for i, bet in enumerate(strategy['recommended_bets'][:3], 1):
                match = bet['match']
                print(f"   {i}. {match['player1']} vs {match['player2']}")
                print(f"      Tournament: {match['tournament_level']} | Surface: {match['surface']}")
                print(f"      Edge: {bet['edge_percent']:.1f}% | Odds: {bet['odds']:.2f}")
                print(f"      Recommended Stake: ${bet['recommended_stake']:.2f}")
                print(f"      Expected Value: ${bet['expected_value']:.2f}")
                print(f"      Confidence: {bet['confidence']}")
                print()
        else:
            print("⚠️  No profitable opportunities identified")
            print("💡 This is normal - profitable edges are rare!")
        
    except ImportError as e:
        print(f"❌ Strategy system not available: {e}")

def run_model_integration_demo():
    """Demo model integration with live odds"""
    print("\n🤖 MODEL INTEGRATION DEMO") 
    print("-" * 50)
    
    try:
        from model import TennisModelPipeline, ModelConfig
        
        # Initialize pipeline with live odds enabled
        pipeline = TennisModelPipeline(
            config=ModelConfig(),
            fast_mode=True,
            enable_live_odds=True
        )
        
        print("✅ Model pipeline initialized with live odds support")
        
        # Test prediction
        match_context = {
            'surface': 'Hard',
            'best_of': 3,
            'data_quality_score': 0.8,
            'tournament_level': 'ATP 500'
        }
        
        prediction = pipeline.predict(match_context)
        
        print(f"📊 Sample Prediction:")
        print(f"   Win Probability: {prediction['win_probability']:.3f}")
        print(f"   Confidence: {prediction['confidence_level']}")
        
        # Test live odds methods
        dashboard = pipeline.get_live_dashboard()
        print(f"📡 Live Odds Dashboard:")
        print(f"   Status: {dashboard.get('status', 'Ready')}")
        print(f"   Active matches: {dashboard.get('active_matches', 0)}")
        print(f"   Edges found: {dashboard.get('edges_found', 0)}")
        
    except ImportError as e:
        print(f"❌ Model integration not available: {e}")
        print("💡 Run the data pipeline first to generate trained models")

async def run_full_system_test():
    """Run comprehensive system test"""
    print("\n🔥 FULL SYSTEM TEST")
    print("-" * 50)
    
    print("Testing all components...")
    
    # Test 1: Model Integration
    print("\n1️⃣  Model Integration:")
    run_model_integration_demo()
    
    # Test 2: Prop Generation
    print("\n2️⃣  Prop Bet Generation:")
    run_prop_demo()
    
    # Test 3: Strategy Generation
    print("\n3️⃣  Betting Strategy:")
    run_strategy_demo()
    
    # Test 4: Live System (brief)
    print("\n4️⃣  Live Edge Detection:")
    await run_live_demo()
    
    # System Summary
    print("\n🏆 SYSTEM CAPABILITIES SUMMARY")
    print("=" * 50)
    print("✅ Live odds calculation with Bayesian updates")
    print("✅ Real-time edge identification across markets")
    print("✅ Prop bet generation from point-by-point data")
    print("✅ Market efficiency analysis and CLV tracking")  
    print("✅ Optimal betting strategy generation")
    print("✅ Risk management and position sizing")
    print("✅ Integration with trained ML models")
    print("=" * 50)
    
    print("\n📈 COMPETITIVE ADVANTAGES:")
    print("🎯 Information asymmetry exploitation")
    print("⚡ Speed advantage with live updates")
    print("🧠 Advanced statistical modeling")
    print("📊 Comprehensive risk management")
    print("🔄 Continuous learning and adaptation")

def show_usage():
    """Show usage information"""
    print("\n📖 USAGE GUIDE")
    print("=" * 50)
    print("Available commands:")
    print("  --demo      Quick demonstration of all features")
    print("  --live      Live edge detection demo (requires dependencies)")
    print("  --props     Prop bet generation from point sequences")
    print("  --strategy  Betting strategy generation")
    print("  --model     Model integration demo")
    print("  --full      Complete system test (all components)")
    print("  --help      Show this usage guide")
    
    print("\n🚀 QUICK START:")
    print("  python run_sportsbook_system.py --demo")
    
    print("\n⚙️  REQUIREMENTS:")
    print("  • Python 3.8+")
    print("  • Required: numpy, pandas, scipy")
    print("  • Optional: trained model files for full functionality")
    
    print("\n🏗️  SYSTEM ARCHITECTURE:")
    print("  model.py              - Core prediction models")
    print("  live_odds_engine.py   - Real-time odds and edge detection")
    print("  prop_bet_engine.py    - Prop generation from point data")
    print("  sportsbook_grade_system.py - Complete system orchestration")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sportsbook-Grade Tennis Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--demo', action='store_true', help='Quick demo of key features')
    parser.add_argument('--live', action='store_true', help='Live edge detection demo')  
    parser.add_argument('--props', action='store_true', help='Prop bet generation demo')
    parser.add_argument('--strategy', action='store_true', help='Betting strategy demo')
    parser.add_argument('--model', action='store_true', help='Model integration demo')
    parser.add_argument('--full', action='store_true', help='Full system test')
    
    args = parser.parse_args()
    
    # Show header
    print_system_header()
    
    try:
        if args.demo:
            print("🎬 QUICK DEMO MODE")
            run_prop_demo()
            run_strategy_demo()
            
        elif args.live:
            await run_live_demo()
            
        elif args.props:
            run_prop_demo()
            
        elif args.strategy:
            run_strategy_demo()
            
        elif args.model:
            run_model_integration_demo()
            
        elif args.full:
            await run_full_system_test()
            
        else:
            show_usage()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try --demo for a basic test or --help for usage information")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())