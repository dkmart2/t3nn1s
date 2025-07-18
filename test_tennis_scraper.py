import pytest
from tennis_updated import TennisAbstractScraper
from collections import Counter

TEST_CASES = [
    {
        "url": "https://www.tennisabstract.com/charting/20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner.html",
        "expected_prefix": {'CA': 'carlos_alcaraz', 'JS': 'jannik_sinner'},
        "tables": {
            'serve': 54, 'serveNeut': 44, 'keypoints': 72, 'rallyoutcomes': 72
        }
    },
]

@pytest.mark.parametrize("case", TEST_CASES)
def test_scraper_basic(case):
    scraper = TennisAbstractScraper()
    records = scraper.scrape_comprehensive_match_data(case["url"])
    assert scraper.prefix_map == case["expected_prefix"]
    canonicals = sorted({r["Player_canonical"] for r in records})
    assert sorted(case["expected_prefix"].values()) == canonicals
    counts = Counter(r["data_type"] for r in records)
    for tbl, exp in case["tables"].items():
        assert counts[tbl] == exp
