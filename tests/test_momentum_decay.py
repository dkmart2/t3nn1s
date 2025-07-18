#!/usr/bin/env python3
"""
Test script to verify momentum learning fix
"""

import pandas as pd
import numpy as np
from model import StateDependentModifiers


def test_momentum_learning():
    """Test the fixed momentum learning implementation"""
    print("Testing momentum learning...")

    # Create synthetic point data with clear momentum patterns
    np.random.seed(42)

    # Match 1: Strong momentum pattern (streaky)
    match1_points = []
    for i in range(50):
        # Create streaks: server wins 5 in a row, then returner wins 5
        if (i // 5) % 2 == 0:
            server = 1
            winner = 1  # Server wins
        else:
            server = 1
            winner = 2  # Returner wins

        match1_points.append({
            'match_id': 'match1',
            'Pt': i + 1,
            'Svr': server,
            'PtWinner': winner
        })

    # Match 2: Random pattern (no momentum)
    match2_points = []
    for i in range(50):
        server = 1 if i % 2 == 0 else 2
        winner = np.random.choice([1, 2])

        match2_points.append({
            'match_id': 'match2',
            'Pt': i + 1,
            'Svr': server,
            'PtWinner': winner
        })

    # Combine data
    point_data = pd.DataFrame(match1_points + match2_points)

    # Test momentum learning
    modifiers = StateDependentModifiers()
    original_decay = modifiers.momentum_decay

    print(f"Original momentum_decay: {original_decay}")

    # Fit momentum
    modifiers.fit_momentum(point_data)

    print(f"Learned momentum_decay: {modifiers.momentum_decay}")

    # Test momentum calculation
    recent_points = [1, 1, 1, 2, 1]  # Mostly server wins
    momentum = modifiers.calculate_momentum(recent_points, player=1)
    print(f"Momentum for server (player 1): {momentum:.3f}")

    momentum_returner = modifiers.calculate_momentum(recent_points, player=2)
    print(f"Momentum for returner (player 2): {momentum_returner:.3f}")

    # Verify momentum is opposite for different players
    assert abs(momentum + momentum_returner) < 0.1, "Momentum should be opposite for different players"

    print("✅ Momentum learning test passed!")
    return True


def test_pressure_learning():
    """Test pressure effect learning"""
    print("\nTesting pressure learning...")

    # Create point data with pressure situations
    pressure_points = []

    # Normal points: 65% server win rate
    for i in range(100):
        pressure_points.append({
            'match_id': 'test',
            'Pt': i + 1,
            'Svr': 1,
            'PtWinner': 1 if np.random.random() < 0.65 else 2,
            'is_break_point': 0,
            'is_set_point': 0,
            'is_match_point': 0
        })

    # Break points: 45% server win rate (pressure hurts server)
    for i in range(50):
        pressure_points.append({
            'match_id': 'test',
            'Pt': i + 101,
            'Svr': 1,
            'PtWinner': 1 if np.random.random() < 0.45 else 2,
            'is_break_point': 1,
            'is_set_point': 0,
            'is_match_point': 0
        })

    point_data = pd.DataFrame(pressure_points)

    # Test pressure learning
    modifiers = StateDependentModifiers()
    modifiers.fit(point_data)

    # Check learned multipliers
    server_bp_mult = modifiers.pressure_multipliers['server'].get('break_point', 1.0)
    returner_bp_mult = modifiers.pressure_multipliers['returner'].get('break_point', 1.0)

    print(f"Server break point multiplier: {server_bp_mult:.3f}")
    print(f"Returner break point multiplier: {returner_bp_mult:.3f}")

    # Server multiplier should be < 1 (pressure hurts), returner > 1 (pressure helps)
    assert server_bp_mult < 1.0, f"Server break point multiplier should be < 1, got {server_bp_mult}"
    assert returner_bp_mult > 1.0, f"Returner break point multiplier should be > 1, got {returner_bp_mult}"

    print("✅ Pressure learning test passed!")
    return True


def test_point_model_integration():
    """Test point model feature engineering"""
    print("\nTesting point model integration...")

    from model import PointLevelModel

    # Create sample point data
    point_data = pd.DataFrame([
        {
            'Pt': 1,
            'Svr': 1,
            'PtWinner': 1,
            '1st': 'S4wf',  # Wide serve, winner
            '2nd': None,
            'Gm1': 0,
            'Gm2': 0,
            'Set1': 0,
            'Set2': 0,
            'Pts': '15-0',
            'TbSet': False,
            'surface': 'Hard'
        },
        {
            'Pt': 2,
            'Svr': 1,
            'PtWinner': 2,
            '1st': 'S6n',  # Serve into net
            '2nd': 'S4f',  # Second serve winner
            'Gm1': 0,
            'Gm2': 0,
            'Set1': 0,
            'Set2': 0,
            'Pts': '15-15',
            'TbSet': False,
            'surface': 'Hard'
        }
    ])

    # Test feature engineering
    model = PointLevelModel()
    features = model.engineer_point_features(point_data)

    print(f"Engineered {len(features.columns)} features: {list(features.columns)}")
    print("Sample feature values:")
    print(features.head())

    # Check that features are properly encoded
    assert 'is_first_serve' in features.columns
    assert 'serve_direction_wide' in features.columns
    assert 'surface_hard' in features.columns
    assert features['is_first_serve'].iloc[0] == 1  # First point is first serve
    assert features['is_first_serve'].iloc[1] == 0  # Second point had double fault

    print("✅ Point model integration test passed!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING ML MODEL ENHANCEMENTS")
    print("=" * 50)

    try:
        test_momentum_learning()
        test_pressure_learning()
        test_point_model_integration()

        print("\n" + "=" * 50)
        print("🎾 ALL TESTS PASSED! 🎾")
        print("The ML model enhancements are working correctly.")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()