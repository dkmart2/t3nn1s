import pytest
from tennis_updated import TennisAbstractScraper
from model import StateDependentModifiers

@pytest.fixture
def raw_pointlog():
    url = "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html"
    return TennisAbstractScraper().get_raw_pointlog(url)

def test_momentum_decay_learns_from_raw_pointlog(raw_pointlog):
    mod = StateDependentModifiers()
    default_decay = mod.momentum_decay
    mod.fit_momentum(raw_pointlog)
    assert 0.5 <= mod.momentum_decay <= 0.99
    assert mod.momentum_decay != default_decay
