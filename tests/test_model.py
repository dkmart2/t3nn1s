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

# Import your modules
from tennis_updated import (
    generate_synthetic_training_data_for_model,
    generate_synthetic_point_data,
    generate_synthetic_match_data,
    extract_unified_features_fixed,
    extract_unified_match_context_fixed
)
from model import TennisModelPipeline, DataDrivenTennisModel, PointLevelModel


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("=" * 60)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Test point data generation
    print("\n1. Generating point-level data...")
    point_data = generate_synthetic_point_data(n_matches=50, points_per_match=100)
    print(f"   Generated {len(point_data)} point records")
    print(f"   Columns: {list(point_data.columns)}")
    print(f"   Sample data:")
    print(point_data.head(3))

    # Test match data generation
    print("\n2. Generating match-level data...")
    match_data = generate_synthetic_match_data(n_matches=100)
    print(f"   Generated {len(match_data)} match records")
    print(f"   Columns: {list(match_data.columns)}")
    print(f"   Sample data:")
    print(match_data[['surface', 'winner_canonical', 'loser_canonical', 'winner_aces', 'loser_aces']].head(3))

    return point_data, match_data


def test_momentum_learning(point_data):
    """Test momentum parameter learning"""
    print("\n" + "=" * 60)
    print("TESTING MOMENTUM LEARNING")
    print("=" * 60)

    from model import StateDependentModifiers

    # Create and train momentum model
    modifiers = StateDependentModifiers()

    # Test pressure learning
    print("\n1. Testing pressure multiplier learning...")
    # Add required columns for pressure learning
    point_data['is_break_point'] = (point_data['p1_games'] >= 5) | (point_data['p2_games'] >= 5)
    point_data['is_set_point'] = (point_data['p1_sets'] >= 2) | (point_data['p2_sets'] >= 2)
    point_data['is_match_point'] = ((point_data['p1_sets'] >= 2) | (point_data['p2_sets'] >= 2)) & point_data[
        'is_set_point']

    modifiers.fit(point_data)
    print(f"   Learned pressure multipliers: {modifiers.pressure_multipliers}")

    # Test momentum learning
    print("\n2. Testing momentum decay learning...")
    modifiers.fit_momentum(point_data)
    print(f"   Learned momentum decay: {modifiers.momentum_decay:.3f}")

    return modifiers


def test_point_model_training(point_data):
    """Test point-level model training"""
    print("\n" + "=" * 60)
    print("TESTING POINT MODEL TRAINING")
    print("=" * 60)

    from model import PointLevelModel

    # Add required features for point model
    point_data['surface_hard'] = (point_data['surface'] == 'Hard').astype(int)
    point_data['surface_clay'] = (point_data['surface'] == 'Clay').astype(int)
    point_data['surface_grass'] = (point_data['surface'] == 'Grass').astype(int)
    point_data['rallyCount'] = np.random.poisson(4, len(point_data))
    point_data['server_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['returner_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['server_h2h_win_pct'] = np.random.uniform(0.3, 0.7, len(point_data))

    # Create and train point model
    point_model = PointLevelModel()

    print("1. Training point-level model...")
    try:
        importance = point_model.fit(point_data)
        print(f"   Model trained successfully!")
        print(f"   Top 5 features:")
        print(importance.head())

        # Test prediction
        test_features = point_data.iloc[:5][point_model.feature_names]
        predictions = point_model.predict_proba(test_features)
        print(f"   Sample predictions: {predictions}")

        return point_model

    except Exception as e:
        print(f"   ERROR training point model: {e}")
        return None


def test_full_pipeline_training():
    """Test the complete training pipeline"""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE TRAINING")
    print("=" * 60)

    print("1. Generating comprehensive training data...")
    point_data, match_data, jeff_data, defaults = generate_synthetic_training_data_for_model()

    print(f"   Point data: {len(point_data)} records")
    print(f"   Match data: {len(match_data)} records")

    # Prepare data for training
    print("\n2. Preparing data for model training...")

    # Add required point-level features
    point_data['surface_hard'] = (point_data['surface'] == 'Hard').astype(int)
    point_data['surface_clay'] = (point_data['surface'] == 'Clay').astype(int)
    point_data['surface_grass'] = (point_data['surface'] == 'Grass').astype(int)
    point_data['rallyCount'] = np.random.poisson(4, len(point_data))
    point_data['server_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['returner_elo'] = 1500 + np.random.normal(0, 200, len(point_data))
    point_data['server_h2h_win_pct'] = np.random.uniform(0.3, 0.7, len(point_data))

    # Add required match-level features with proper data types
    match_data = match_data.copy()
    match_data['actual_winner'] = 1  # Winner is always player 1 in our synthetic data
    match_data['winner_elo'] = 1600.0 + np.random.normal(0, 150, len(match_data))
    match_data['loser_elo'] = 1500.0 + np.random.normal(0, 150, len(match_data))
    match_data['winner_last10_wins'] = np.random.randint(5, 10, len(match_data)).astype(float)
    match_data['loser_last10_wins'] = np.random.randint(3, 8, len(match_data)).astype(float)
    match_data['p1_surface_h2h_wins'] = np.random.randint(0, 5, len(match_data)).astype(float)
    match_data['p2_surface_h2h_wins'] = np.random.randint(0, 5, len(match_data)).astype(float)

    # Add missing features for feature extraction
    match_data['WRank'] = np.random.randint(1, 100, len(match_data)).astype(float)
    match_data['LRank'] = np.random.randint(1, 100, len(match_data)).astype(float)
    match_data['p1_ranking'] = match_data['WRank']
    match_data['p2_ranking'] = match_data['LRank']

    # Add serve points won for serve effectiveness calculation
    match_data['winner_pts_won'] = (match_data['winner_serve_pts'] * 0.65).astype(int)
    match_data['loser_pts_won'] = (match_data['loser_serve_pts'] * 0.58).astype(int)

    # Add return points for return effectiveness
    match_data['winner_return_pts'] = 80
    match_data['loser_return_pts'] = 80
    match_data['winner_return_pts_won'] = (match_data['winner_return_pts'] * 0.38).astype(int)
    match_data['loser_return_pts_won'] = (match_data['loser_return_pts'] * 0.32).astype(int)

    print("3. Training complete pipeline...")
    try:
        pipeline = TennisModelPipeline()
        pipeline.train(point_data, match_data)
        print("   ✓ Pipeline trained successfully!")

        return pipeline, match_data

    except Exception as e:
        print(f"   ERROR training pipeline: {e}")
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
    sample_match = match_data.iloc[0].to_dict()

    # Extract features for prediction
    p1_features = extract_unified_features_fixed(sample_match, 'winner')
    p2_features = extract_unified_features_fixed(sample_match, 'loser')
    match_context = extract_unified_match_context_fixed(sample_match)

    print("1. Test match context:")
    print(f"   Surface: {match_context.get('surface')}")
    print(f"   Data quality: {match_context.get('data_quality_score')}")
    print(f"   Rankings: P1={match_context.get('p1_ranking')}, P2={match_context.get('p2_ranking')}")

    print("\n2. Player features:")
    print(f"   Player 1 serve effectiveness: {p1_features.get('serve_effectiveness'):.3f}")
    print(f"   Player 2 serve effectiveness: {p2_features.get('serve_effectiveness'):.3f}")
    print(f"   Player 1 return effectiveness: {p1_features.get('return_effectiveness'):.3f}")
    print(f"   Player 2 return effectiveness: {p2_features.get('return_effectiveness'):.3f}")

    print("\n3. Making prediction...")
    try:
        # Safe ranking calculation with defaults
        p1_rank = match_context.get('p1_ranking') or 50
        p2_rank = match_context.get('p2_ranking') or 50
        h2h_pct = match_context.get('p1_h2h_win_pct') or 0.5

        # Prepare context for pipeline prediction
        prediction_context = {
            **match_context,
            'elo_diff': p1_rank - p2_rank,
            'h2h_advantage': h2h_pct - 0.5,
            'data_quality_score': match_context.get('data_quality_score', 0.8)
        }

        result = pipeline.predict(prediction_context, best_of=3)

        print(f"   Win probability: {result['win_probability']:.3f}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Simulation component: {result['simulation_component']:.3f}")

    except Exception as e:
        print(f"   ERROR making prediction: {e}")
        import traceback
        traceback.print_exc()


def run_complete_test():
    """Run all tests in sequence"""
    print("🎾 TENNIS MODEL TESTING SUITE 🎾")
    print("=" * 60)

    # Test 1: Data generation
    point_data, match_data = test_synthetic_data_generation()

    # Test 2: Momentum learning
    modifiers = test_momentum_learning(point_data.copy())

    # Test 3: Point model training
    point_model = test_point_model_training(point_data.copy())

    # Test 4: Full pipeline training
    pipeline, full_match_data = test_full_pipeline_training()

    # Test 5: Prediction
    test_prediction(pipeline, full_match_data)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_complete_test()