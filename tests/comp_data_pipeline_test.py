import os
import pickle
import pandas as pd
import pytest
from datetime import date, timedelta
from tennis_updated import (
    load_from_cache_with_scraping,
    CACHE_DIR,
    HD_PATH,
    JEFF_PATH,
    DEF_PATH,
    AutomatedTennisAbstractScraper
)

@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    # Redirect cache paths to a temporary directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr('tennis_updated.CACHE_DIR', str(cache_dir))
    monkeypatch.setattr('tennis_updated.HD_PATH', str(cache_dir / "historical_data.parquet"))
    monkeypatch.setattr('tennis_updated.JEFF_PATH', str(cache_dir / "jeff_data.pkl"))
    monkeypatch.setattr('tennis_updated.DEF_PATH', str(cache_dir / "weighted_defaults.pkl"))
    # Prevent real scraping by returning no records
    monkeypatch.setattr(
        AutomatedTennisAbstractScraper,
        "automated_scraping_session",
        lambda self, force=False: []
    )
    yield

def test_initial_backfill_branch(capsys):
    # No cache files exist → should trigger initial integration
    hist, jeff, defaults = load_from_cache_with_scraping()
    captured = capsys.readouterr()
    assert "No Tennis Abstract data found in cache" in captured.out
    assert isinstance(hist, pd.DataFrame)
    assert isinstance(jeff, dict)
    assert isinstance(defaults, dict)

def test_skip_backfill_when_ta_present(capsys):
    # Create a dummy historical_data.parquet with one TA column already
    data = pd.DataFrame({
        "composite_id": ["match1"],
        "date": [date.today() - timedelta(days=1)],
        "winner_ta_points": [42]
    })
    data["date"] = pd.to_datetime(data["date"]).dt.date
    os.makedirs(os.path.dirname(HD_PATH), exist_ok=True)
    data.to_parquet(HD_PATH, index=False)

    # Create dummy Jeff and defaults pickle files
    os.makedirs(os.path.dirname(JEFF_PATH), exist_ok=True)
    with open(JEFF_PATH, "wb") as f:
        pickle.dump({}, f)
    os.makedirs(os.path.dirname(DEF_PATH), exist_ok=True)
    with open(DEF_PATH, "wb") as f:
        pickle.dump({}, f)

    hist, jeff, defaults = load_from_cache_with_scraping()
    captured = capsys.readouterr()
    assert "Tennis Abstract data is current" in captured.out
    assert "winner_ta_points" in hist.columns
    assert hist.loc[0, "winner_ta_points"] == 42