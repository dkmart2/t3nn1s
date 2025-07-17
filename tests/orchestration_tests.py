#!/usr/bin/env python3
"""
FIXED: Comprehensive orchestration tests for tennis data pipeline
Tests actual pipeline workflows with improved mocking
"""

import pytest
import os
import shutil
import tempfile
import pickle
import subprocess
import time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock, Mock, mock_open
from pathlib import Path

# Import your pipeline functions
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_updated import (
    load_from_cache_with_scraping,
    generate_comprehensive_historical_data,
    save_to_cache,
    load_from_cache,
    run_automated_tennis_abstract_integration,
    integrate_scraped_data_hybrid,
    process_tennis_abstract_scraped_data,
    AutomatedTennisAbstractScraper,
    TennisAbstractScraper,
    integrate_api_tennis_data_incremental
)


class TestPipelineOrchestrationFixed:
    """FIXED: Test actual pipeline orchestration and workflows"""

    @pytest.fixture(autouse=True)
    def setup_orchestration_environment(self):
        """Setup complete test environment for orchestration"""
        self.temp_cache_dir = tempfile.mkdtemp(prefix="tennis_orchestration_")
        self.hd_path = os.path.join(self.temp_cache_dir, "historical_data.parquet")
        self.jeff_path = os.path.join(self.temp_cache_dir, "jeff_data.pkl")
        self.def_path = os.path.join(self.temp_cache_dir, "weighted_defaults.pkl")

        # Patch all cache paths
        self.cache_patches = [
            patch('tennis_updated.CACHE_DIR', self.temp_cache_dir),
            patch('tennis_updated.HD_PATH', self.hd_path),
            patch('tennis_updated.JEFF_PATH', self.jeff_path),
            patch('tennis_updated.DEF_PATH', self.def_path)
        ]

        for p in self.cache_patches:
            p.start()

        yield

        # Cleanup
        for p in self.cache_patches:
            p.stop()
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def create_realistic_historical_data(self, n_matches=1000):
        """Create realistic historical data for testing"""
        dates = [date(2025, 6, 10) + timedelta(days=i // 10) for i in range(n_matches)]

        data = []
        for i, match_date in enumerate(dates):
            data.append({
                'composite_id': f"{match_date.strftime('%Y%m%d')}-tournament_{i % 5}-player_a_{i % 100}-player_b_{i % 100}",
                'date': match_date,
                'Winner': f'Player A {i % 100}',
                'Loser': f'Player B {i % 100}',
                'winner_canonical': f'player_a_{i % 100}',
                'loser_canonical': f'player_b_{i % 100}',
                'gender': 'M' if i % 2 == 0 else 'W',
                'Tournament': f'Tournament {i % 5}',
                'source_rank': 3,
                'winner_serve_pts': 60 + (i % 20),
                'loser_serve_pts': 55 + (i % 15),
                'winner_aces': 5 + (i % 10),
                'loser_aces': 3 + (i % 8),
                'PSW': 1.5 + (i % 10) * 0.1,
                'PSL': 2.5 + (i % 10) * 0.1,
                'ta_enhanced': False  # Initially no TA enhancement
            })

        return pd.DataFrame(data)

    def create_comprehensive_jeff_data(self):
        """Create comprehensive Jeff data structure"""
        return {
            'men': {
                'overview': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'set': 'Total', 'serve_pts': 60 + i, 'aces': 5 + i}
                    for i in range(50)
                ])
            },
            'women': {
                'overview': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'set': 'Total', 'serve_pts': 55 + i, 'aces': 4 + i}
                    for i in range(50)
                ])
            }
        }

    def create_weighted_defaults(self):
        """Create comprehensive weighted defaults"""
        return {
            'men': {'serve_pts': 65.0, 'aces': 6.2, 'return_pts_won': 25.8},
            'women': {'serve_pts': 60.0, 'aces': 4.8, 'return_pts_won': 23.2}
        }

    def create_tennis_abstract_scraped_data(self):
        """Create realistic Tennis Abstract scraped data"""
        return [
            {
                'composite_id': '20250715-wimbledon-jannik_sinner-carlos_alcaraz',
                'Player_canonical': 'jannik_sinner',
                'data_type': 'serve',
                'stat_name': 'points_won_pct',
                'stat_value': 75.5,
                'Date': '20250715',
                'tournament': 'Wimbledon',
                'player1': 'Jannik Sinner',
                'player2': 'Carlos Alcaraz'
            }
        ]

    # FIXED TEST 1: COLD START ORCHESTRATION
    @patch('tennis_updated.save_to_cache')
    @patch('tennis_updated.integrate_api_tennis_data_incremental')
    @patch('tennis_updated.run_automated_tennis_abstract_integration')
    @patch('tennis_updated.generate_comprehensive_historical_data')
    @patch('tennis_updated.load_from_cache')
    def test_1_cold_start_orchestration_fixed(self, mock_load_cache, mock_generate, mock_ta_integration,
                                              mock_api_integration, mock_save):
        """FIXED: Test complete cold start pipeline orchestration"""
        print("\n=== TESTING COLD START ORCHESTRATION (FIXED) ===")

        # Mock empty cache (triggers cold start)
        mock_load_cache.return_value = (None, None, None)

        # Mock data generation
        historical_data = self.create_realistic_historical_data(500)
        jeff_data = self.create_comprehensive_jeff_data()
        defaults = self.create_weighted_defaults()
        mock_generate.return_value = (historical_data, jeff_data, defaults)

        # Mock TA integration (returns enhanced data)
        enhanced_data = historical_data.copy()
        enhanced_data['ta_enhanced'] = True
        mock_ta_integration.return_value = enhanced_data

        # Mock save operation
        mock_save.return_value = True

        # Verify empty cache
        assert not os.path.exists(self.hd_path), "Cache should be empty initially"

        # Execute cold start orchestration
        hist, jeff_result, defaults_result = load_from_cache_with_scraping()

        # Verify orchestration results
        assert hist is not None, "Should generate historical data"
        assert len(hist) >= 500, "Should contain base tennis data"
        assert jeff_result is not None, "Should have Jeff data"
        assert defaults_result is not None, "Should have weighted defaults"

        # Verify call sequence
        mock_load_cache.assert_called_once()
        mock_generate.assert_called_once()
        mock_ta_integration.assert_called_once()
        mock_save.assert_called_once()

        print("✓ Cold start orchestration completed successfully")

    # TEST 2: INCREMENTAL UPDATE ORCHESTRATION (UNCHANGED - WORKING)
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    @patch('tennis_updated.api_call')
    @patch('tennis_updated.get_player_rankings')
    @patch('tennis_updated.get_tournaments_metadata')
    def test_2_incremental_update_orchestration(self, mock_tournaments, mock_rankings, mock_api, mock_scraper):
        """Test incremental update orchestration with existing cache"""
        print("\n=== TESTING INCREMENTAL UPDATE ORCHESTRATION ===")

        # Setup existing cache
        existing_data = self.create_realistic_historical_data(300)
        existing_data['date'] = existing_data['date'].apply(lambda x: x - timedelta(days=5))
        existing_jeff = self.create_comprehensive_jeff_data()
        existing_defaults = self.create_weighted_defaults()

        save_to_cache(existing_data, existing_jeff, existing_defaults)
        original_count = len(existing_data)

        # Mock API data
        mock_api.return_value = []
        mock_rankings.return_value = {}
        mock_tournaments.return_value = {}

        # Mock Tennis Abstract
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = self.create_tennis_abstract_scraped_data()
        mock_scraper.return_value = mock_scraper_instance

        # Execute incremental update
        hist, _, _ = load_from_cache_with_scraping()

        # Verify incremental updates
        assert len(hist) >= original_count, "Should maintain or increase data count"

        print("✓ Incremental update orchestration completed successfully")

    # TEST 3: TENNIS ABSTRACT INTEGRATION (UNCHANGED - WORKING)
    def test_3_tennis_abstract_backfill_orchestration(self):
        """Test Tennis Abstract backfill logic and integration"""
        print("\n=== TESTING TENNIS ABSTRACT BACKFILL ORCHESTRATION ===")

        hist_data = self.create_realistic_historical_data(200)
        hist_data['ta_enhanced'] = False
        scraped_records = self.create_tennis_abstract_scraped_data()

        enhanced_data = integrate_scraped_data_hybrid(hist_data, scraped_records)

        assert len(enhanced_data) >= len(hist_data), "Should maintain or increase data count"

        ta_columns = [col for col in enhanced_data.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
        assert len(ta_columns) > 0, "Should have Tennis Abstract feature columns"

        processed_matches = process_tennis_abstract_scraped_data(scraped_records)
        assert len(processed_matches) > 0, "Should process scraped records"

        print("✓ Tennis Abstract backfill orchestration completed successfully")

    # TEST 4: SCRAPER ORCHESTRATION (UNCHANGED - WORKING)
    @patch('requests.get')
    def test_4_tennis_abstract_scraper_orchestration(self, mock_get):
        """Test actual Tennis Abstract scraper orchestration"""
        print("\n=== TESTING TENNIS ABSTRACT SCRAPER ORCHESTRATION ===")

        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <head><title>Tennis Abstract: Jannik Sinner vs Carlos Alcaraz Detailed Stats</title></head>
        <body>
        <script>
        var serve = '<table><tr><th>Player</th><th>Pts Won</th></tr><tr><td>JS Total</td><td>45 (75%)</td></tr><tr><td>CA Total</td><td>32 (68%)</td></tr></table>';
        </script>
        </body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        scraper = TennisAbstractScraper()
        test_url = "https://www.tennisabstract.com/charting/20250715-M-Wimbledon-F-Jannik_Sinner-Carlos_Alcaraz.html"

        match_meta = scraper._parse_match_url(test_url)
        assert 'Date' in match_meta, "Should parse match metadata"
        assert match_meta['Date'] == '20250715', "Should extract correct date"

        scraped_data = scraper.scrape_comprehensive_match_data(test_url)
        assert isinstance(scraped_data, list), "Should return list of records"

        print("✓ Tennis Abstract scraper orchestration completed successfully")

    # TEST 5: SIMPLIFIED SCRIPT ORCHESTRATION
    def test_5_simplified_script_orchestration(self):
        """Test simplified script orchestration without subprocess"""
        print("\n=== TESTING SIMPLIFIED SCRIPT ORCHESTRATION ===")

        # Test that we can import and call functions without errors
        try:
            from tennis_updated import load_from_cache_with_scraping

            # Mock all external dependencies
            with patch('tennis_updated.api_call', return_value=[]), \
                    patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper, \
                    patch('tennis_updated.load_all_tennis_data',
                          return_value=self.create_realistic_historical_data(10)), \
                    patch('tennis_updated.load_jeff_comprehensive_data',
                          return_value=self.create_comprehensive_jeff_data()), \
                    patch('tennis_updated.calculate_comprehensive_weighted_defaults',
                          return_value=self.create_weighted_defaults()):

                mock_scraper_instance = MagicMock()
                mock_scraper_instance.automated_scraping_session.return_value = []
                mock_scraper.return_value = mock_scraper_instance

                # Test function call
                result = load_from_cache_with_scraping()

                # Should not crash
                assert result is not None, "Function should return something"
                assert len(result) == 3, "Should return tuple of (hist, jeff, defaults)"

        except Exception as e:
            pytest.fail(f"Script orchestration failed: {e}")

        print("✓ Simplified script orchestration completed successfully")

    # TEST 6: DATA INTEGRITY (UNCHANGED - WORKING)
    def test_6_data_flow_integrity_orchestration(self):
        """Test data integrity through the complete orchestration flow"""
        print("\n=== TESTING DATA FLOW INTEGRITY ORCHESTRATION ===")

        initial_data = self.create_realistic_historical_data(100)
        initial_composite_ids = set(initial_data['composite_id'])

        save_to_cache(initial_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())

        with patch('tennis_updated.api_call', return_value=[]), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []
            mock_scraper.return_value = mock_scraper_instance

            hist, jeff_data, defaults = load_from_cache_with_scraping()

        final_composite_ids = set(hist['composite_id'])
        missing_ids = initial_composite_ids - final_composite_ids
        assert len(missing_ids) == 0, f"Lost composite_ids during orchestration: {missing_ids}"

        assert hist['composite_id'].nunique() == len(hist), "Should have unique composite_ids"
        assert hist['winner_canonical'].notna().all(), "Should have winner_canonical for all matches"
        assert hist['loser_canonical'].notna().all(), "Should have loser_canonical for all matches"

        print("✓ Data flow integrity orchestration completed successfully")

    # TEST 7: PERFORMANCE (UNCHANGED - WORKING)
    def test_7_performance_orchestration(self):
        """Test performance characteristics of orchestration"""
        print("\n=== TESTING PERFORMANCE ORCHESTRATION ===")

        small_data = self.create_realistic_historical_data(50)
        medium_data = self.create_realistic_historical_data(500)

        with patch('tennis_updated.api_call', return_value=[]), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []
            mock_scraper.return_value = mock_scraper_instance

            # Time operations
            save_to_cache(small_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())
            start_time = time.time()
            hist_small, _, _ = load_from_cache_with_scraping()
            small_duration = time.time() - start_time

            save_to_cache(medium_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())
            start_time = time.time()
            hist_medium, _, _ = load_from_cache_with_scraping()
            medium_duration = time.time() - start_time

        assert small_duration < 10.0, "Small dataset processing should be fast"
        assert medium_duration < 30.0, "Medium dataset processing should be reasonable"

        print(f"✓ Performance orchestration: small={small_duration:.2f}s, medium={medium_duration:.2f}s")

    # TEST 8: ERROR RECOVERY (UNCHANGED - WORKING)
    def test_8_error_recovery_orchestration(self):
        """Test error handling and recovery in orchestration"""
        print("\n=== TESTING ERROR RECOVERY ORCHESTRATION ===")

        valid_data = self.create_realistic_historical_data(50)
        save_to_cache(valid_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())

        with patch('tennis_updated.api_call', side_effect=Exception("API failed")), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:

            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []
            mock_scraper.return_value = mock_scraper_instance

            try:
                hist, _, _ = load_from_cache_with_scraping()
                assert hist is not None, "Should return cached data on API failure"
                assert len(hist) >= 50, "Should maintain cached data count"
            except Exception as e:
                pytest.fail(f"Orchestration should handle API failures gracefully: {e}")

        print("✓ Error recovery orchestration completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--capture=no"])