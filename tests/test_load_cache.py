import os
import pickle
import pandas as pd
import pytest
from datetime import date, timedelta
from tennis_updated import (
    load_from_cache_with_scraping,
    run_automated_tennis_abstract_integration,
    CACHE_DIR,
    HD_PATH,
    JEFF_PATH,
    DEF_PATH
)

@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    # Redirect cache paths to a temporary directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HD_PATH", str(cache_dir / "historical_data.parquet"))
    monkeypatch.setenv("JEFF_PATH", str(cache_dir / "jeff_data.pkl"))
    monkeypatch.setenv("DEF_PATH", str(cache_dir / "weighted_defaults.pkl"))
    # Prevent any real Tennis Abstract scraping
    monkeypatch.setattr(
        "tennis_updated.run_automated_tennis_abstract_integration",
        lambda hist, days_back=None: hist
    )
    yield

def test_initial_backfill_branch(capsys):
    # No cache files exist → should trigger initial integration
    hist, jeff, defaults = load_from_cache_with_scraping()
    captured = capsys.readouterr()
    assert "No Tennis Abstract data found in cache" in captured.out
    assert isinstance(hist, pd.DataFrame)
    # On initial backfill jeff and defaults come through as empty or default dicts
    assert isinstance(jeff, dict) and isinstance(defaults, dict)

def test_skip_backfill_when_ta_present(capsys):
    # Create a dummy historical_data.parquet with one TA column already
    data = pd.DataFrame({
        "composite_id": ["match1"],
        "date": [date.today() - timedelta(days=1)],
        "winner_ta_points": [42]
    })
    # Ensure date col is correct type
    data["date"] = pd.to_datetime(data["date"]).dt.date
    os.makedirs(os.path.dirname(HD_PATH), exist_ok=True)
    data.to_parquet(HD_PATH, index=False)

    # Create dummy Jeff/defaults files
    with open(JEFF_PATH, "wb") as f:
        pickle.dump({}, f)
    with open(DEF_PATH, "wb") as f:
        pickle.dump({}, f)

    hist, jeff, defaults = load_from_cache_with_scraping()
    captured = capsys.readouterr()
    assert "Tennis Abstract data is current" in captured.out
    # The returned DataFrame should still have that TA column
    assert "winner_ta_points" in hist.columns
    # And no change in its value
    assert hist.loc[0, "winner_ta_points"] == 42