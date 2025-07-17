import pytest
import pandas as pd
from tennis_updated import TennisAbstractScraper, integrate_scraped_data_hybrid
from tennis_updated import extract_unified_features_fixed, extract_comprehensive_jeff_features
from collections import Counter

URL = "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html"


@pytest.fixture(scope="module")
def scraped_records():
    return TennisAbstractScraper().scrape_comprehensive_match_data(URL)


@pytest.fixture(scope="module")
def historical_data(scraped_records):
    comp_id = scraped_records[0]["composite_id"]
    return pd.DataFrame([{
        "composite_id": comp_id,
        "winner_canonical": "jannik_sinner",
        "loser_canonical": "carlos_alcaraz",
        "source_rank": 2
    }])


def test_hybrid_integration(scraped_records, historical_data):
    integrated = integrate_scraped_data_hybrid(historical_data.copy(), scraped_records)

    # Test that TA columns were created correctly
    ta_columns = [col for col in integrated.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
    assert len(ta_columns) > 0, "Should have Tennis Abstract columns"

    # Test that values exist and are numeric
    winner_ta_cols = [col for col in ta_columns if col.startswith('winner_ta_')]
    loser_ta_cols = [col for col in ta_columns if col.startswith('loser_ta_')]

    assert len(winner_ta_cols) > 0, "Should have winner TA columns"
    assert len(loser_ta_cols) > 0, "Should have loser TA columns"

    # Check that at least some TA values are populated (not null)
    winner_populated = sum(1 for col in winner_ta_cols if pd.notna(integrated.loc[0, col]))
    loser_populated = sum(1 for col in loser_ta_cols if pd.notna(integrated.loc[0, col]))

    assert winner_populated > 0, "Should have populated winner TA values"
    assert loser_populated > 0, "Should have populated loser TA values"

    # Verify TA enhancement flag
    assert integrated.loc[0, 'ta_enhanced'] == True, "Should mark match as TA enhanced"

    # Verify data quality score updated
    assert 'data_quality_score' in integrated.columns, "Should have data quality score"
    assert integrated.loc[0, 'data_quality_score'] > 0, "Should have positive data quality score"


def test_feature_extraction_unified():
    md = {"winner_ta_serve_won_pct": 0.8, "winner_ta_return1_points_won_pct": 0.4}
    feats = extract_unified_features_fixed(md, "winner")
    keys = {"serve_effectiveness", "return_effectiveness", "winners_rate", "unforced_rate", "pressure_performance",
            "net_effectiveness"}
    assert keys.issubset(feats.keys())
    for v in feats.values():
        assert isinstance(v, float) and 0 <= v <= 1


def test_feature_extraction_comprehensive_jeff_defaults():
    feats = extract_comprehensive_jeff_features("any", "M", jeff_data={})
    assert "serve_pts" in feats and "aces" in feats and "return_pts_won" in feats