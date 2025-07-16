import pytest
import pandas as pd
from datetime import date
from tennis_updated import (
    parse_match_statistics,
    extract_embedded_statistics,
    integrate_api_tennis_data,
    BayesianTennisModel,
    flatten_fixtures,
    get_fixtures_for_date,
)

def test_parse_match_statistics():
    fixture = {
        "first_player_key": "1",
        "second_player_key": "2",
        "p1_aces": 5,
        "p2_aces": 3
    }
    df = flatten_fixtures([fixture])
    assert "p1_aces" in df.columns
    stats = parse_match_statistics(fixture)
    assert 1 in stats and 2 in stats
    assert stats[1]["aces"] == 5
    assert stats[2]["aces"] == 3

def test_extract_embedded_statistics():
    fixture = {
        "scores": [
            {"score_first": "6", "score_second": "4"},
            {"score_first": "3", "score_second": "6"}
        ]
    }
    stats = extract_embedded_statistics(fixture)
    assert stats["sets_won_p1"] == 1
    assert stats["sets_won_p2"] == 1
    assert stats["total_sets"] == 2

def test_integrate_api_tennis_data(monkeypatch):
    import tennis_updated
    hist = pd.DataFrame({"composite_id": []})
    fixtures_day = [
        {"event_key": "A", "first_player_key": "1", "second_player_key": "2"},
        {"event_key": "B", "first_player_key": "3", "second_player_key": "4"},
    ]
    monkeypatch.setattr(tennis_updated, "get_fixtures_for_date", lambda d: fixtures_day)
    monkeypatch.setattr(tennis_updated, "parse_match_statistics", lambda f: {1: {"x":1}, 2: {"x":2}})
    monkeypatch.setattr(tennis_updated, "extract_embedded_statistics", lambda f: {"y":10})
    merged = integrate_api_tennis_data(hist, days_back=1)
    assert len(merged) == 2
    assert "x" in merged.columns and "y" in merged.columns

def test_simulation_methods():
    model = BayesianTennisModel(n_simulations=10)
    p1_stats = {"serve_pts": 10, "first_serve_won": 5, "return_pts_won": 5}
    p2_stats = {"serve_pts": 10, "first_serve_won": 5, "return_pts_won": 5}
    p = model._estimate_point_win_prob(p1_stats, p2_stats)
    assert 0.01 <= p <= 0.99
    tb = model.simulate_tiebreak(p1_stats, p2_stats)
    assert tb in (1, 2)
    st = model.simulate_set(p1_stats, p2_stats)
    assert st in (1, 2)
    prob = model.simulate_match(p1_stats, p2_stats, best_of=3)
    assert 0.0 <= prob <= 1.0
