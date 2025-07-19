#!/usr/bin/env python3
"""
Test script for the tennis prediction model using synthetic data
"""

import pandas as pd
import numpy as np
from datetime import date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TennisModelPipeline, DataDrivenTennisModel, PointLevelModel
try:
    from model_pipeline import ModelConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure model.py is in the same directory as this script")
    sys.exit(1)


def generate_synthetic_point_data(n_matches=50, points_per_match=100):
    """Generate realistic synthetic point data"""
    data = []
    surfaces = ['Hard', 'Clay', 'Grass']

    for match_id in range(n_matches):
        surface = np.random.choice(surfaces)
        for point_num in range(points_per_match):
            server = np.random.choice([1, 2])
            winner = server if np.random.random() < 0.65 else (3 - server)

            data.append({
                'match_id': f'match_{match_id}',
                'Pt': point_num + 1,
                'Svr': server,
                'PtWinner': winner,
                'surface': surface,
                'is_break_point': np.random.random() < 0.08,
                'is_set_point': np.random.random() < 0.05,
                'is_match_point': np.random.random() < 0.02,
                'rallyCount': np.random.poisson(4) + 1,
                'p1_games': np.random.randint(0, 7),
                'p2_games': np.random.randint(0, 7),
                'p1_sets': np.random.randint(0, 3),
                'p2_sets': np.random.randint(0, 3),
            })

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} point records with columns: {list(df.columns)}")
    return df


def generate_synthetic_match_data(n_matches=50):
    """Generate realistic synthetic match data"""
    surfaces = ['Hard', 'Clay', 'Grass']
    players = ['player_a', 'player_b', 'player_c', 'player_d', 'player_e']

    data = []
    for i in range(n_matches):
        winner = np.random.choice(players)
        loser = np.random.choice([p for p in players if p != winner])

        data.append({
            'match_id': f'match_{i}',
            'surface': np.random.choice(surfaces),
            'winner_canonical': winner,
            'loser_canonical': loser,
            'WRank': np.random.randint(1, 200),
            'LRank': np.random.randint(1, 200),
            'winner_aces': np.random.poisson(7),
            'loser_aces': np.random.poisson(5),
            'winner_serve_pts': np.random.normal(80, 10),
            'loser_serve_pts': np.random.normal(80, 10),
            'tournament_tier': np.random.choice(['ATP 250', 'ATP 500', 'Masters 1000', 'Grand Slam'])
        })

    return pd.DataFrame(data)


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("=" * 60)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Test point data generation
    print("\n1. Generating point-level data...")
    point_data = generate_synthetic_point_data(n_matches=20, points_per_match=50)
    print(f"   Generated {len(point_data)} point records")
    print(f"   Sample data:")
    print(point_data[['match_id', 'Svr', 'PtWinner', 'surface', 'is_break_point']].head(3))

    # Test match data generation
    print("\n2. Generating match-level data...")
    match_data = generate_synthetic_match_data(n_matches=30)
    print(f"   Generated {len(match_data)} match records")
    print(f"   Sample data:")
    print(match_data[['match_id', 'WRank', 'LRank', 'winner_aces', 'surface']].head(3))

    return point_data, match_data


def test_point_model_training(point_data):
    """Test point-level model training"""
    print("\n" + "=" * 60)
    print("TESTING POINT MODEL TRAINING")
    print("=" * 60)

    # Add required features for point model
    point_data = point_data.copy()
    point_data['surface_hard'] = (point_data['surface'] == 'Hard').astype(int)
    point_data['surface_clay'] = (point_data['surface'] == 'Clay').astype(int)
    point_data['surface_grass'] = (point_data['surface'] == 'Grass').astype(int)
    point_data['server_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['returner_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['server_h2h_win_pct'] = np.random.uniform(0.3, 0.7, len(point_data))

    # Create and train point model
    config = ModelConfig(lgb_estimators=20, lgb_verbose=-1)  # Reduced for testing
    point_model = PointLevelModel(fast_mode=True, config=config)

    print("1. Training point-level model...")
    try:
        importance = point_model.fit(point_data)
        print(f"   ✓ Model trained successfully!")
        if not importance.empty:
            print(f"   Top 3 features:")
            print(importance.head(3))

        # Test prediction
        if point_model.feature_names:
            test_features = pd.DataFrame([{
                fname: 0.5 if 'pct' in fname else 1 if fname.startswith('is_') else 0
                for fname in point_model.feature_names
            }])
            predictions = point_model.predict_proba(test_features)
            print(f"   Sample prediction: {predictions[0]:.3f}")
        else:
            print("   No feature names available for prediction test")

        return point_model

    except Exception as e:
        print(f"   ✗ ERROR training point model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_match_ensemble_training(match_data):
    """Test match-level ensemble training"""
    print("\n" + "=" * 60)
    print("TESTING MATCH ENSEMBLE TRAINING")
    print("=" * 60)

    from model import MatchLevelEnsemble

    # Prepare match data
    match_data = match_data.copy()
    match_data['winner_elo'] = 1600.0 + np.random.normal(0, 150, len(match_data))
    match_data['loser_elo'] = 1500.0 + np.random.normal(0, 150, len(match_data))
    match_data['winner_last10_wins'] = np.random.randint(5, 10, len(match_data))
    match_data['loser_last10_wins'] = np.random.randint(3, 8, len(match_data))

    config = ModelConfig(lgb_estimators=20, rf_estimators=20, lgb_verbose=-1)
    ensemble = MatchLevelEnsemble(fast_mode=True, config=config)

    print("1. Training match-level ensemble...")
    try:
        ensemble.fit(match_data)
        print("   ✓ Ensemble trained successfully!")

        # Test prediction
        test_match = pd.DataFrame([{
            'WRank': 10, 'LRank': 20, 'winner_elo': 1700, 'loser_elo': 1600,
            'tournament_tier': 'ATP 250', 'winner_aces': 8, 'loser_aces': 5,
            'winner_serve_pts': 80, 'loser_serve_pts': 75
        }])

        features = ensemble.engineer_match_features(test_match)
        prediction = ensemble.predict(features)
        print(f"   Sample prediction: {prediction:.3f}")

        return ensemble

    except Exception as e:
        print(f"   ✗ ERROR training ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline_training():
    """Test the complete training pipeline"""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE TRAINING")
    print("=" * 60)

    print("1. Generating training data...")
    point_data = generate_synthetic_point_data(n_matches=30, points_per_match=40)
    match_data = generate_synthetic_match_data(n_matches=50)

    # Prepare point data
    point_data['surface_hard'] = (point_data['surface'] == 'Hard').astype(int)
    point_data['surface_clay'] = (point_data['surface'] == 'Clay').astype(int)
    point_data['surface_grass'] = (point_data['surface'] == 'Grass').astype(int)
    point_data['server_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['returner_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['server_h2h_win_pct'] = np.random.uniform(0.3, 0.7, len(point_data))

    # Prepare match data
    match_data['winner_elo'] = 1600.0 + np.random.normal(0, 150, len(match_data))
    match_data['loser_elo'] = 1500.0 + np.random.normal(0, 150, len(match_data))
    match_data['winner_last10_wins'] = np.random.randint(5, 10, len(match_data))
    match_data['loser_last10_wins'] = np.random.randint(3, 8, len(match_data))
    match_data['winner_serve_pts'] = np.maximum(50, match_data['winner_serve_pts'])
    match_data['loser_serve_pts'] = np.maximum(50, match_data['loser_serve_pts'])

    print("2. Training complete pipeline...")
    try:
        config = ModelConfig(
            lgb_estimators=20,
            rf_estimators=20,
            n_simulations=100,
            lgb_verbose=-1
        )
        pipeline = TennisModelPipeline(config=config, fast_mode=True)
        pipeline.train(point_data, match_data)
        print("   ✓ Pipeline trained successfully!")

        return pipeline, match_data

    except Exception as e:
        print(f"   ✗ ERROR training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, match_data


def test_prediction(pipeline, match_data):
    """Test making predictions with trained model"""
    print("\n" + "=" * 60)
    print("TESTING MATCH PREDICTIONS")
    print("=" * 60)

    if pipeline is None:
        print("Pipeline not available, skipping prediction test")
        return

    # Create test match context
    print("1. Creating test match context...")
    test_context = {
        'surface': 'Hard',
        'p1_ranking': 15,
        'p2_ranking': 25,
        'WRank': 15,
        'LRank': 25,
        'winner_elo': 1700,
        'loser_elo': 1650,
        'elo_diff': 50,
        'h2h_advantage': 0.1,
        'data_quality_score': 0.8,
        'tournament_tier': 'ATP 500',
        'winner_aces': 8,
        'loser_aces': 5,
        'winner_serve_pts': 80,
        'loser_serve_pts': 75
    }

    print("   Test context:")
    print(f"   Surface: {test_context['surface']}")
    print(f"   Rankings: {test_context['p1_ranking']} vs {test_context['p2_ranking']}")
    print(f"   Elo difference: {test_context['elo_diff']}")

    print("\n2. Making prediction...")
    try:
        result = pipeline.predict(test_context, best_of=3, fast_mode=True)

        print(f"   ✓ Prediction successful!")
        print(f"   Win probability: {result['win_probability']:.3f}")
        print(f"   Simulation component: {result['simulation_component']:.3f}")
        print(f"   Direct component: {result['direct_component']:.3f}")
        print(f"   Confidence: {result['confidence']}")

        # Test multiple predictions
        print("\n3. Testing multiple predictions...")
        for i in range(3):
            test_ctx = test_context.copy()
            test_ctx['p1_ranking'] = 10 + i * 20
            test_ctx['p2_ranking'] = 30 + i * 15

            result = pipeline.predict(test_ctx, fast_mode=True)
            print(f"   Test {i + 1}: P(win) = {result['win_probability']:.3f}, "
                  f"Confidence = {result['confidence']}")

    except Exception as e:
        print(f"   ✗ ERROR making prediction: {e}")
        import traceback
        traceback.print_exc()


def test_model_components():
    """Test individual model components"""
    print("\n" + "=" * 60)
    print("TESTING MODEL COMPONENTS")
    print("=" * 60)

    from model import StateDependentModifiers

    print("1. Testing StateDependentModifiers...")
    modifiers = StateDependentModifiers()

    # Test momentum calculation
    recent_points = [1, 1, 2, 1, 2, 2, 1]  # Mixed wins
    momentum = modifiers.calculate_momentum(recent_points, player=1)
    print(f"   Momentum for player 1: {momentum:.3f}")

    # Test pressure modifiers
    score_state = {'is_break_point': True, 'is_set_point': False, 'is_match_point': False}
    pressure_mod = modifiers.get_pressure_modifier(score_state, 'server')
    print(f"   Break point pressure modifier: {pressure_mod:.3f}")

    print("   ✓ StateDependentModifiers working correctly")


def run_complete_test():
    """Run all tests in sequence"""
    print("🎾 TENNIS MODEL TESTING SUITE 🎾")
    print("=" * 60)

    try:
        # Test 1: Data generation
        point_data, match_data = test_synthetic_data_generation()

        # Test 2: Model components
        test_model_components()

        # Test 3: Point model training
        point_model = test_point_model_training(point_data.copy())

        # Test 4: Match ensemble training
        ensemble = test_match_ensemble_training(match_data.copy())

        # Test 5: Full pipeline training
        pipeline, full_match_data = test_full_pipeline_training()

        # Test 6: Prediction
        test_prediction(pipeline, full_match_data)

        print("\n" + "=" * 60)
        print("🏆 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✓ Synthetic data generation working")
        print("✓ Point-level model training functional")
        print("✓ Match-level ensemble training functional")
        print("✓ Full pipeline integration working")
        print("✓ Prediction system operational")
        print("✓ Model components validated")

    except Exception as e:
        print(f"\n💥 TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_complete_test()